#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Memory RAM + CVAE (+ ESM2/scGPT + LoRA + GO/evidence + unseen vocab)
Author: You

What’s new vs your last pipeline:
- **ESM2** protein LM drives the **PPI pathway** (frozen table + trainable projection).
- **scGPT** gene-expression LM drives the **TF pathway** (frozen table + trainable projection).
- **Evidence-aware mixer**: uses per-perturbation evidence vector
  [is_TF, has_scGPT, has_ESM2, go_tf_like, go_signal, go_mito, go_cellcycle]
  to bias the TF↔PPI mixture (via a small linear head).
- **LoRA hyper-adapters** on both decoders, conditioned on (hidden, external embedding):
  low-rank delta added to decoder outputs; rank/scale controlled by CLI.
- **Dynamic unseen perturbations (zero-shot)**: during inference we *expand* vocab and
  external tables to include unseen symbols found in val_counts.csv (no random fallback).
- **Cell-type-OOD split**: optional `--heldout_cts "CT1,CT2"` ⇒ these CTs go to **TEST** only;
  VAL remains i.i.d. on train CTs for stable early stopping.
- (Keeps) Heteroscedastic Gaussian emission (μ, log σ) + optional CVAE latent z with KL warmup.

Inputs for external knowledge
-----------------------------
- scGPT embeddings: TSV, first column gene symbol, rest are vector dims.
- ESM2 embeddings: tab-separated CSV/TSV, first column gene symbol (often 'Unnamed: 0'),
  rest are vector dims (whatever your pipeline produced).
- TF list (text, 1 symbol per line): marks perturbations that are transcription factors.
  (We treat your uploaded list as canonical.  :contentReference[oaicite:1]{index=1})
- GO CSV: columns ['target','tag'] with per-target GO tags (used to create mixer biases).

Quick-start (pseudobulk)
------------------------
python hybrid_ram_vcc_cvae_plus.py \
  --h5ad train.h5ad --outdir ./outputs_pb \
  --scgpt_embeddings_tsv /mnt/data/scgpt_embeddings.tsv \
  --esm2_embeddings_csv /path/to/ESM2_gene_embeddings.tsv --esm2_sep '\t' \
  --tf_list /mnt/data/TF_names_v_1.01.txt \
  --go_csv "/mnt/data/GO (1).csv" \
  --target_col target_gene --control_label non-targeting \
  --epochs 50 --batch_size 32 --use_cvae --z_dim 32 --beta_kl 0.5 --kl_warmup_epochs 10

Notes
-----
• All external tables are *frozen* by default; only a small projection to model dims is trained.
• If only scGPT is provided (ESM2 too big), code degrades gracefully to TF-only externalization.
• GO: we use high-level keyword buckets to bias TF↔PPI mixing; gene-level masks are optional.

Requirements:
  python>=3.9, numpy, pandas, scipy, anndata, scanpy, torch, scikit-learn, tqdm, pytorch-lightning
"""

import os, math, json, argparse, re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.serialization

try:
    import scanpy as sc
    import anndata as ad
except Exception as e:
    raise RuntimeError(
        "Install scanpy and anndata: `pip install scanpy anndata`."
    ) from e

from scipy.sparse import issparse
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Lightning wrapper compatibility (same pattern as before)
try:
    import pytorch_lightning as pl
except Exception:
    class _DummyLM(nn.Module):
        @property
        def device(self):
            return next(self.parameters()).device
    pl = type("pl", (), {"LightningModule": _DummyLM})

# ------------------------------- Utilities --------------------------------- #

def seed_everything(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def to_np(x) -> np.ndarray:
    return x.A if issparse(x) else np.asarray(x)

def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

def resolve_device(args) -> torch.device:
    """
    Device priority:
      1) --cpu override
      2) --device explicit choice (cpu/cuda/mps)
      3) auto fallback: cuda -> mps -> cpu
    """
    requested = getattr(args, "device", "auto")
    if getattr(args, "cpu", False):
        if requested != "auto" and requested != "cpu":
            print("[WARN] --cpu overrides --device; using cpu.")
        return torch.device("cpu")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested --device=cuda but CUDA is not available.")
    if requested == "mps":
        if _mps_available():
            return torch.device("mps")
        raise RuntimeError("Requested --device=mps but MPS is not available.")
    raise ValueError(f"Unknown device selection: {requested}")

def group_mean_from_adata(adata: "ad.AnnData", labels: pd.Series, min_cells: int = 20) -> Dict[str, np.ndarray]:
    X = adata.X
    groups: Dict[str, np.ndarray] = {}
    vc = labels.value_counts()
    kept = vc[vc >= min_cells].index.tolist()
    for lab in kept:
        idx = (labels == lab).values
        if idx.sum() < min_cells: continue
        mean_vec = X[idx].mean(axis=0).A1.astype(np.float32) if issparse(X) \
                   else X[idx].mean(axis=0).astype(np.float32).ravel()
        groups[str(lab)] = mean_vec
    return groups

def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_norm = F.normalize(a, dim=1, eps=eps); b_norm = F.normalize(b, dim=1, eps=eps)
    return a_norm @ b_norm.T

def info_nce_loss(z_q: torch.Tensor, z_k: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    B = z_q.size(0)
    if B <= 1: return z_q.new_tensor(0.0)
    sim = cosine_sim_matrix(z_q, z_k) / temperature
    targets = torch.arange(B, device=z_q.device)
    return F.cross_entropy(sim, targets)

def soft_de_targets(y: torch.Tensor, x0: torch.Tensor, tau: float = 0.25) -> torch.Tensor:
    return torch.sigmoid((y - x0).abs() / tau)

def topk_overlap_ratio(delta_true: np.ndarray, delta_pred: np.ndarray, k: int) -> float:
    if k <= 0: return 0.0
    n = min(int(delta_true.shape[0]), int(delta_pred.shape[0]))
    if n <= 0: return 0.0
    k = min(int(k), n)
    idx_true = np.argpartition(-np.abs(delta_true), kth=k-1)[:k]
    idx_pred = np.argpartition(-np.abs(delta_pred), kth=k-1)[:k]
    return len(set(idx_true.tolist()).intersection(set(idx_pred.tolist()))) / float(k)

def pds_from_l1(d_true: np.ndarray, d_pred: np.ndarray, row_chunk: int = 32) -> float:
    """Local PDS approximation. Memory-bounded: chunks the row dimension so the
    peak allocation is O(row_chunk * N * G) instead of O(N * N * G). At VCC
    scale (N ~ 200, G ~ 18000) the chunked version stays under 0.5 GB where
    the original needed ~3 GB; at N ~ 500 the original would OOM.
    Note: this is the local approximation used for early-stop signal during
    training. The gating metric is cell-eval's PDS computed by the official
    Arc tool on the validation deliverable."""
    N, _G = d_true.shape
    if N == 0:
        return 0.0
    ranks_diag = np.empty(N, dtype=np.int64)
    for start in range(0, N, row_chunk):
        end = min(start + row_chunk, N)
        D_chunk = np.abs(d_pred[start:end, None, :] - d_true[None, :, :]).sum(axis=2)
        order = D_chunk.argsort(axis=1)
        ranks = order.argsort(axis=1) + 1  # rank 1 = smallest distance
        for i_local, i_global in enumerate(range(start, end)):
            ranks_diag[i_global] = ranks[i_local, i_global]
    pds = 1.0 - (ranks_diag - 1) / N
    return float(np.mean(pds))

# ------------------- External embedding + helpers -------------------------- #

class FrozenExternalEmbedding(nn.Module):
    """
    A frozen embedding table with optional trainable linear projection to a target dim.
    weights: torch.FloatTensor [vocab_size, ext_dim]
    """
    def __init__(self, weights: torch.FloatTensor, out_dim: int, trainable_proj: bool = True):
        super().__init__()
        assert weights.ndim == 2, "weights must be [V, D_ext]"
        self.emb = nn.Embedding.from_pretrained(weights, freeze=True)  # freeze LM geometry
        self.proj = nn.Linear(self.emb.embedding_dim, out_dim, bias=False)
        self.proj.weight.requires_grad = trainable_proj

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.proj(self.emb(idx))

def _load_tsv_with_firstcol_symbol(path: str, sep: Optional[str] = None, header: Optional[int] = None):
    """
    Generic loader for TSV/CSV where column 0 is gene symbol; remainder numeric.
    If header is None, assumes no header row; else uses the detected header.
    """
    df = pd.read_csv(path, sep=sep, header=header, engine="python")
    # If header exists and gene symbol column is named weirdly (e.g., 'Unnamed: 0'), fix it
    if header is not None:
        gcol = df.columns[0]
        df = df.rename(columns={gcol: "symbol"})
    else:
        # No header: create one
        cols = ["symbol"] + [str(i) for i in range(df.shape[1] - 1)]
        df.columns = cols
    df = df.dropna(subset=[df.columns[0]])
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(np.float32)
    return df

def load_scgpt_table(path: Optional[str], vocab: Dict[str,int]) -> Tuple[Optional[torch.FloatTensor], Dict[str, torch.FloatTensor]]:
    if not path or not os.path.exists(path): return None, {}
    # Most scGPT dumps are TSV without header (first col = gene)
    try:
        df = _load_tsv_with_firstcol_symbol(path, sep="\t", header=None)
    except Exception:
        # allow any separator if needed
        df = _load_tsv_with_firstcol_symbol(path, sep=None, header=None)
    df["symbol"] = df["symbol"].astype(str)
    df = df.drop_duplicates(subset=["symbol"], keep="first")
    dim = df.shape[1] - 1
    # aligned table [|vocab|, dim]
    out = np.zeros((len(vocab), dim), dtype=np.float32)
    sym2vec: Dict[str, torch.FloatTensor] = {}
    M = df.set_index("symbol")
    for g, row in M.iterrows():
        sym2vec[str(g)] = torch.from_numpy(row.values.astype(np.float32))
    for g, idx in vocab.items():
        if g in M.index:
            row = M.loc[g].values.astype(np.float32)
            out[idx] = row
    return torch.from_numpy(out), sym2vec

def load_esm2_table(path: Optional[str], vocab: Dict[str,int], sep: str = "\t") -> Tuple[Optional[torch.FloatTensor], Dict[str, torch.FloatTensor]]:
    if not path or not os.path.exists(path): return None, {}
    # Tab-separated CSV/TSV, first column = symbol (sometimes named 'Unnamed: 0')
    df = pd.read_csv(path, sep=sep, engine="python")
    first = df.columns[0]
    df = df.rename(columns={first: "symbol"})
    df["symbol"] = df["symbol"].astype(str)
    # cast remaining numeric columns
    num_cols = [c for c in df.columns if c != "symbol"]
    df[num_cols] = df[num_cols].astype(np.float32)
    df = df.drop_duplicates(subset=["symbol"], keep="first")
    dim = len(num_cols)
    out = np.zeros((len(vocab), dim), dtype=np.float32)
    sym2vec: Dict[str, torch.FloatTensor] = {}
    M = df.set_index("symbol")
    for g, row in M.iterrows():
        sym2vec[str(g)] = torch.from_numpy(row.values.astype(np.float32))
    for g, idx in vocab.items():
        if g in M.index:
            row = M.loc[g].values.astype(np.float32)
            out[idx] = row
    return torch.from_numpy(out), sym2vec

def load_tf_list(path: Optional[str]) -> set:
    if not path or not os.path.exists(path): return set()
    syms = set()
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s: syms.add(s)
    return syms

_GO_TF_PAT = re.compile(r"TRANSCRIPT|CHROMATIN|DNA_BIND|RNA POLYMERASE|GENE EXPRESSION|HISTONE|NUCLE", re.I)
_GO_SIGNAL_PAT = re.compile(r"SIGNAL(ING| TRANSDUCTION)|KINASE|PHOSPHO|RECEPTOR|MAPK|PI3K|MTOR|JAK|STAT", re.I)
_GO_MITO_PAT = re.compile(r"MITO(CHONDR|)", re.I)
_GO_CYCLE_PAT = re.compile(r"CELL[_ ]?CYCLE|MITOTIC|M PHASE|S PHASE|G1|G2|CHECKPOINT", re.I)

def load_go_bias(go_csv: Optional[str]) -> Dict[str, np.ndarray]:
    """
    Returns dict: gene -> 4-dim float array [go_tf_like, go_signal, go_mito, go_cellcycle]
    based on simple keyword buckets.
    """
    if not go_csv or not os.path.exists(go_csv): return {}
    df = pd.read_csv(go_csv)
    if not {"target", "tag"}.issubset(set(df.columns)):
        # Try to guess
        cols = list(df.columns)
        df = df.rename(columns={cols[0]: "target", cols[1]: "tag"})
    df["target"] = df["target"].astype(str)
    df["tag"] = df["tag"].astype(str)
    agg = {}
    for g, sub in df.groupby("target"):
        tags = " || ".join(sub["tag"].tolist())
        out = np.zeros(4, dtype=np.float32)
        if _GO_TF_PAT.search(tags): out[0] = 1.0
        if _GO_SIGNAL_PAT.search(tags): out[1] = 1.0
        if _GO_MITO_PAT.search(tags): out[2] = 1.0
        if _GO_CYCLE_PAT.search(tags): out[3] = 1.0
        agg[g] = out
    return agg

# ----------------------------- LoRA adapters -------------------------------- #

class LoRATriMul(nn.Module):
    """
    A low‑rank adapter that produces an output delta for a decoder using both
    the current hidden (z) and an external embedding (e):
        delta_out = ((z @ A) ⊙ (e @ C)) @ B^T
    A: [z_dim, r], C: [e_dim, r], B: [r, out_dim]; r = lora_rank (small)
    """
    def __init__(self, z_dim: int, e_dim: int, out_dim: int, r: int = 8, alpha: float = 1.0):
        super().__init__()
        self.r = int(r); self.alpha = float(alpha)
        if r <= 0:
            self.enabled = False
            # dummy params to keep .to(device) happy
            self.A = nn.Linear(z_dim, 1, bias=False)
            self.C = nn.Linear(e_dim, 1, bias=False)
            self.B = nn.Linear(1, out_dim, bias=False)
        else:
            self.enabled = True
            self.A = nn.Linear(z_dim, r, bias=False)
            self.C = nn.Linear(e_dim, r, bias=False)
            self.B = nn.Linear(r, out_dim, bias=False)
            nn.init.zeros_(self.B.weight)  # start from no delta

    def forward(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        if not self.enabled: return z.new_zeros(z.size(0), self.B.out_features)
        a = self.A(z)    # [B, r]
        c = self.C(e)    # [B, r]
        m = a * c        # [B, r]
        return self.alpha * self.B(m)  # [B, out_dim]

# ----------------------------- Model classes -------------------------------- #

class Saturation(nn.Module):
    def __init__(self, n_genes: int, vocab_size: int, sat_reg: float = 1e-2, base_init: float = 2.0, r: int = 128):
        super().__init__()
        self.n_genes = n_genes
        self.base = nn.Parameter(torch.full((n_genes,), float(base_init)))
        self.U = nn.Embedding(vocab_size, r)
        nn.init.normal_(self.U.weight, std=0.02)
        self.V = nn.Parameter(torch.randn(r, n_genes) * 0.01)
        self.sat_reg = float(sat_reg)
    def forward(self, x0: torch.Tensor, proposed: torch.Tensor, pert_idx: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.base + self.U(pert_idx) @ self.V) + 1e-6
        delta = proposed - x0
        return x0 + torch.tanh(delta / scale) * scale
    def reg_loss(self) -> torch.Tensor:
        return self.sat_reg * (self.U.weight.pow(2).mean() + self.V.pow(2).mean())

class HybridMemorySystem(nn.Module):
    def __init__(self, n_genes: int, feat_dim: int, fast_memory_dim: int, slow_memory_dim: int,
                 n_attention_heads: int = 4, memory_bank_size: int = 6, dropout: float = 0.1,
                 slow_decoder_rank: int = 1024):
        super().__init__()
        self.n_genes = int(n_genes)
        self.fast_dim = int(fast_memory_dim)
        self.slow_dim = int(slow_memory_dim)
        self.memory_bank_size = int(memory_bank_size)
        self.fast_upd = nn.GRUCell(input_size=self.fast_dim + feat_dim + 4, hidden_size=self.fast_dim)
        self.q_proj = nn.Linear(self.fast_dim, self.slow_dim, bias=False)
        self.k_proj = nn.Linear(self.slow_dim, self.slow_dim, bias=False)
        self.v_proj = nn.Linear(self.slow_dim, self.slow_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.fast_decoder_lin = nn.Linear(self.fast_dim, self.n_genes, bias=False)
        self.slow_dec_r1 = nn.Linear(self.slow_dim, slow_decoder_rank, bias=False)
        self.slow_dec_r2 = nn.Linear(slow_decoder_rank, self.n_genes, bias=False)
        self.router_in_dim = self.fast_dim + self.slow_dim + 2
        self.router = nn.Sequential(nn.Linear(self.router_in_dim, 64), nn.ReLU(),
                                    nn.Linear(64, 2), nn.Softmax(dim=-1))
        # Random projection for episode summaries
        self.proj_dim = 32
        self.register_buffer("rand_proj", F.normalize(torch.randn(self.n_genes, self.proj_dim), dim=0), persistent=False)
        self.ep_enc = nn.Sequential(
            nn.Linear(self.proj_dim * 3 + self.fast_dim + 3*13, 256), nn.ReLU(),
            nn.Linear(256, self.slow_dim)
        )
    def _stats2(self, x: torch.Tensor):
        m = x.abs().mean(dim=1, keepdim=True); l2 = x.norm(p=2, dim=1, keepdim=True); return m, l2
    def update_fast_memory(self, fast_memory: torch.Tensor, state_delta: torch.Tensor,
                           combined_effect: torch.Tensor, pert_features: torch.Tensor) -> torch.Tensor:
        m1, l1 = self._stats2(state_delta); m2, l2 = self._stats2(combined_effect)
        upd_in = torch.cat([fast_memory, pert_features, m1, l1, m2, l2], dim=1)
        return self.fast_upd(upd_in, fast_memory)
    def retrieve_from_slow_memory(self, current: torch.Tensor, fast_memory: torch.Tensor,
                                  slow_memory_bank: List[torch.Tensor]):
        B = current.size(0)
        if len(slow_memory_bank) == 0:
            return torch.zeros(B, self.slow_dim, device=current.device, dtype=current.dtype), torch.zeros(B, 0, device=current.device, dtype=current.dtype)
        bank = torch.stack(slow_memory_bank, dim=0)
        if bank.ndim == 3: bank = bank.mean(dim=1)
        K = self.k_proj(bank); Q = self.q_proj(fast_memory)
        attn_logits = (Q @ K.T) / math.sqrt(K.size(1))
        attn = self.attn_drop(torch.softmax(attn_logits, dim=1))
        V = self.v_proj(bank); slow_feat = attn @ V
        return slow_feat, attn
    def integrate_memories(self, current, fast_memory, slow_features, pathway_combined):
        f = self.fast_decoder_lin(fast_memory); s = self.slow_dec_r2(F.relu(self.slow_dec_r1(slow_features)))
        return f + s
    def fast_decoder(self, fast_memory): return self.fast_decoder_lin(fast_memory)
    def slow_decoder(self, slow_features): return self.slow_dec_r2(F.relu(self.slow_dec_r1(slow_features)))
    def should_store_episode(self, delta: torch.Tensor, fast_memory: torch.Tensor, t: int):
        return bool(t == 0 or (t % 2 == 1)), None
    def episode_encoder(self, episode_input: torch.Tensor) -> torch.Tensor:
        B = episode_input.size(0); G = self.n_genes
        delta = episode_input[:, 0:G]; inc = episode_input[:, G:2*G]; curr = episode_input[:, 2*G:3*G]; fast = episode_input[:, 3*G:]
        def summarize(X):
            rp = X @ self.rand_proj
            mean = X.mean(dim=1, keepdim=True); std = X.std(dim=1, keepdim=True); l2 = X.norm(p=2, dim=1, keepdim=True)
            topk = torch.topk(X, k=5, dim=1, largest=True).values; botk = torch.topk(-X, k=5, dim=1, largest=True).values
            return torch.cat([rp, mean, std, l2, topk, botk], dim=1)
        feat = torch.cat([summarize(delta), summarize(inc), summarize(curr), fast], dim=1)
        return self.ep_enc(feat)
    def memory_router(self, router_input: torch.Tensor) -> torch.Tensor:
        fast = router_input[:, :self.fast_dim]
        slow = router_input[:, self.fast_dim:self.fast_dim+self.slow_dim]
        curr = router_input[:, self.fast_dim+self.slow_dim:]
        m, l2 = self._stats2(curr)
        return self.router(torch.cat([fast, slow, m, l2], dim=1))

# -------------------- Externalized PPI & TF pathways + LoRA ---------------- #

class PPIPathwayESM(nn.Module):
    def __init__(self, n_genes: int, esm_table: torch.FloatTensor, rank: int = 256,
                 trainable_proj: bool = True, lora_rank: int = 8, lora_alpha: float = 1.0):
        super().__init__()
        self.ext = FrozenExternalEmbedding(esm_table, out_dim=rank, trainable_proj=trainable_proj)
        self.cur_lin = nn.Linear(n_genes, rank, bias=False)
        self.dec     = nn.Linear(rank, n_genes, bias=False)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        # LoRA over decoder output, conditioned on (z, external embedding)
        self.lora = LoRATriMul(z_dim=rank, e_dim=rank, out_dim=n_genes, r=lora_rank, alpha=lora_alpha)

    def forward(self, current: torch.Tensor, pert_idx: torch.Tensor) -> torch.Tensor:
        e = self.ext(pert_idx)           # [B, rank]
        c = self.cur_lin(current)        # [B, rank]
        z = F.relu(e + c)
        y = self.dec(z)
        y = y + self.lora(z, e)          # LoRA delta
        return y

class TFPathwayScGPT(nn.Module):
    def __init__(self, n_genes: int, scgpt_table: torch.FloatTensor, hidden: int = 256,
                 trainable_proj: bool = True, lora_rank: int = 8, lora_alpha: float = 1.0):
        super().__init__()
        self.ext = FrozenExternalEmbedding(scgpt_table, out_dim=hidden, trainable_proj=trainable_proj)
        self.fc1 = nn.Linear(n_genes, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, n_genes, bias=True)
        # LoRA on output, conditioned on (z= h*gate, e=scgpt)
        self.lora = LoRATriMul(z_dim=hidden, e_dim=hidden, out_dim=n_genes, r=lora_rank, alpha=lora_alpha)

    def forward(self, current: torch.Tensor, pert_idx: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(current))             # [B, hidden]
        gate = torch.sigmoid(self.ext(pert_idx))  # [B, hidden]
        z = h * gate
        y = self.fc2(z)
        y = y + self.lora(z, gate)
        return y

# ------------------------------ RAM core (+ CVAE) -------------------------- #

class HybridMemoryRAMLite(nn.Module):
    """
    Now with optional CVAE head + evidence-aware TF/PPI mixer.
      - Encoder q(z|y,x0,pert) via RP+stats (lightweight for 18k genes)
      - z influences initial fast memory and head features
      - Heteroscedastic Gaussian emission (μ=y_mix, log σ from head)
      - Evidence vector E[pert] biases pathway mix towards TF/PPI/other GO buckets
    """
    def __init__(self, n_genes: int, *, vocab_size: Optional[int] = None, max_rounds: int = 8,
                 min_rounds: int = 2, epsilon: float = 1e-3, step_init: float = 0.25,
                 sat_reg: float = 1e-2, feat_dim: int = 32, fast_memory_dim: int = 64,
                 slow_memory_dim: int = 128, memory_bank_size: int = 6, n_attention_heads: int = 4,
                 memory_influence: float = 0.2, sat_rank: int = 128, slow_decoder_rank: int = 1024,
                 use_cvae: bool = False, z_dim: int = 32):
        super().__init__()
        self.n_genes = int(n_genes)
        self.vocab_size = int((n_genes + 2) if (vocab_size is None) else vocab_size)
        self.max_rounds = int(max_rounds); self.min_rounds = int(min_rounds)
        self.epsilon = float(epsilon)
        self.memory_influence = float(memory_influence)
        self.training_memory_scale = 1.0
        self.use_cvae = bool(use_cvae)
        self.z_dim = int(z_dim) if self.use_cvae else 0
        self.z_feat_dim = 16 if self.use_cvae else 0

        self.pert_feat = nn.Embedding(self.vocab_size, feat_dim)
        nn.init.normal_(self.pert_feat.weight, std=0.02)
        self._pert_dropout_p = 0.0  # set from args after construction

        self.memory_system = HybridMemorySystem(
            n_genes=self.n_genes, feat_dim=feat_dim, fast_memory_dim=fast_memory_dim,
            slow_memory_dim=slow_memory_dim, n_attention_heads=n_attention_heads,
            memory_bank_size=memory_bank_size, dropout=0.1, slow_decoder_rank=slow_decoder_rank
        )

        # CVAE encoder (RP + stats for y, x0, delta)
        if self.use_cvae:
            self.enc_proj_dim = 32
            self.register_buffer("enc_proj",
                F.normalize(torch.randn(self.n_genes, self.enc_proj_dim), dim=0),
                persistent=False
            )
            enc_in = 3 * (self.enc_proj_dim + 3) + feat_dim
            self.enc_mu = nn.Sequential(nn.Linear(enc_in, 128), nn.ReLU(), nn.Linear(128, self.z_dim))
            self.enc_logvar = nn.Sequential(nn.Linear(enc_in, 128), nn.ReLU(), nn.Linear(128, self.z_dim))
            self.z_to_fast = nn.Linear(self.z_dim, fast_memory_dim, bias=False)
            self.z_proj = nn.Sequential(nn.Linear(self.z_dim, self.z_feat_dim), nn.ReLU())

        self.pert_to_fast_memory = nn.Sequential(
            nn.Linear(feat_dim, fast_memory_dim * 2), nn.ReLU(),
            nn.Linear(fast_memory_dim * 2, fast_memory_dim), nn.Tanh(),
        )

        head_input_dim = feat_dim + 4 + fast_memory_dim + slow_memory_dim + self.z_feat_dim
        self.feat_norm = nn.LayerNorm(head_input_dim)
        self.halt_head = nn.Sequential(nn.Linear(head_input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
                                       nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.step_head = nn.Sequential(nn.Linear(head_input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
                                       nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.step_bias = nn.Parameter(torch.tensor(step_init))
        self.mix_head = nn.Sequential(nn.Linear(head_input_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.memory_influence_adapter = nn.Sequential(nn.Linear(feat_dim + 4, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.saturation = Saturation(self.n_genes, self.vocab_size, sat_reg=sat_reg, base_init=2.0, r=sat_rank)

        # Generative head
        self.logsigma_head = nn.Sequential(nn.Linear(head_input_dim, 128), nn.ReLU(), nn.Linear(128, self.n_genes))
        self._logsigma_min = -5.0; self._logsigma_max = 1.5
        self.sample_mode = False
        def set_sampling(enabled: bool = True): self.sample_mode = bool(enabled)
        self.set_sampling = set_sampling

        # evidence (dimension is set at runtime)
        self.has_evidence = False
        self.evidence_vec = None
        self.evidence_proj: Optional[nn.Linear] = None

    @torch.no_grad()
    def _stats(self, x: torch.Tensor):
        mean_abs = x.abs().mean(dim=1, keepdim=True); l2 = x.norm(p=2, dim=1, keepdim=True)
        return mean_abs, l2

    def set_evidence_vec(self, evidence_vec: torch.Tensor):
        """
        Safely (re)register the evidence vector buffer.
        Handles first-time and repeat calls gracefully and preserves
        trained projection weights when dimensionality matches.
        """
        old_proj = self.evidence_proj if isinstance(self.evidence_proj, nn.Linear) else None

        # Remove any existing attribute or buffer if present
        if "evidence_vec" in self._buffers:
            del self._buffers["evidence_vec"]
        elif hasattr(self, "evidence_vec"):
            delattr(self, "evidence_vec")
    
        # Register new buffer
        self.register_buffer("evidence_vec", evidence_vec, persistent=False)
        self.has_evidence = True
    
        # Recreate projection layer
        D = evidence_vec.size(1)
        self.evidence_proj = nn.Linear(D, 1, bias=True).to(self.pert_feat.weight.device)
        if old_proj is not None and old_proj.in_features == D and old_proj.out_features == 1:
            with torch.no_grad():
                self.evidence_proj.weight.copy_(old_proj.weight.to(self.evidence_proj.weight.device))
                self.evidence_proj.bias.copy_(old_proj.bias.to(self.evidence_proj.bias.device))
        else:
            nn.init.zeros_(self.evidence_proj.weight)
            nn.init.zeros_(self.evidence_proj.bias)
    
    
    def _enc_features(self, y_obs: torch.Tensor, x0: torch.Tensor, pfeat: torch.Tensor):
        d = y_obs - x0
        rp_y = y_obs @ self.enc_proj; rp_x0 = x0 @ self.enc_proj; rp_d = d @ self.enc_proj
        def s(v): return torch.cat([v.mean(1, keepdim=True), v.std(1, keepdim=True), v.norm(p=2, dim=1, keepdim=True)], dim=1)
        enc = torch.cat([rp_y, s(y_obs), rp_x0, s(x0), rp_d, s(d), pfeat], dim=1)
        return enc

    def _build_feats(self, x0_f32: torch.Tensor, current_f32: torch.Tensor,
                     combined: torch.Tensor, pert_features: torch.Tensor,
                     fast_memory: torch.Tensor, slow_features: torch.Tensor,
                     z_feat: Optional[torch.Tensor] = None):
        mabs_d, l2_d = self._stats(combined)
        mabs_c, l2_c = self._stats(current_f32 - x0_f32)
        scalars = torch.cat([mabs_d, l2_d, mabs_c, l2_c], dim=1)
        parts = [pert_features, scalars, fast_memory, slow_features]
        if z_feat is not None: parts.append(z_feat)
        return torch.cat(parts, dim=1)

    def forward(self, x0: torch.Tensor, tf_pathway: nn.Module, ppi_pathway: nn.Module,
                pert_idx: torch.Tensor, y_obs: Optional[torch.Tensor] = None,
                return_details: bool = False, deterministic: bool = True):
        """
        If y_obs is provided → use q(z|y,x0,pert). Else → prior (z=0 if deterministic else sample N(0,I)).
        deterministic controls z sampling only; predictions remain deterministic unless sample_mode=True.
        """
        in_dtype = x0.dtype; device = x0.device
        x0_f32 = x0.float(); current = x0_f32.clone()
        B, G = current.shape

        pert_features = self.pert_feat(pert_idx)
        # Zero-shot bridge: randomly drop learned embedding so model relies on external priors
        if self.training and self._pert_dropout_p > 0:
            _drop = torch.rand(pert_features.size(0), 1, device=pert_features.device) < self._pert_dropout_p
            pert_features = pert_features.masked_fill(_drop, 0.0)
        # ---- CVAE latent z ----
        z = None; kl = None; z_feat = None
        if self.use_cvae and (y_obs is not None):
            enc = self._enc_features(y_obs.float(), x0_f32, pert_features)
            mu = self.enc_mu(enc); logvar = self.enc_logvar(enc)
            if deterministic:
                z = mu
            else:
                eps = torch.randn_like(mu)
                z = mu + eps * torch.exp(0.5 * logvar)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        elif self.use_cvae:
            z = torch.zeros(B, self.z_dim, device=device) if deterministic else torch.randn(B, self.z_dim, device=device)
        if self.use_cvae:
            z_feat = self.z_proj(z)
            fast_memory = self.pert_to_fast_memory(pert_features) + self.z_to_fast(z)
        else:
            fast_memory = self.pert_to_fast_memory(pert_features)

        slow_memory_bank: List[torch.Tensor] = []
        y_mix = torch.zeros(B, G, device=device, dtype=torch.float32)
        remaining = torch.ones(B, 1, device=device, dtype=torch.float32)

        weights_per_round = []; halts_per_round = []; steps_per_round = []
        attention_maps = []; fast_contrib = []; slow_contrib = []
        last_feats = None

        for t in range(self.max_rounds):
            tf_eff  = tf_pathway(current, pert_idx).float()
            ppi_eff = ppi_pathway(current, pert_idx).float()
            combined_effect = tf_eff + ppi_eff

            fast_memory = self.memory_system.update_fast_memory(fast_memory, current - x0_f32, combined_effect, pert_features)
            slow_features, attn_weights = self.memory_system.retrieve_from_slow_memory(current, fast_memory, slow_memory_bank)
            attention_maps.append(attn_weights.detach())

            base_feats = torch.cat([pert_features, *self._stats(combined_effect), *self._stats(current - x0_f32)], dim=1)
            feats = self._build_feats(x0_f32, current, combined_effect, pert_features, fast_memory, slow_features, z_feat)
            feats = self.feat_norm(feats)
            last_feats = feats

            cos = F.cosine_similarity(tf_eff.detach(), ppi_eff.detach(), dim=1, eps=1e-8).unsqueeze(1)

            ev_bias = torch.zeros(B, 1, device=device, dtype=feats.dtype)
            if self.has_evidence and (self.evidence_vec is not None) and (self.evidence_proj is not None):
                ev_bias = self.evidence_proj(self.evidence_vec[pert_idx])

            esm_bias = torch.zeros_like(ev_bias)
            if hasattr(ppi_pathway, "alpha_logit"):
                esm_bias = torch.sigmoid(ppi_pathway.alpha_logit).view(1,1).to(ev_bias) * 0.5

            gamma = torch.sigmoid(self.mix_head(feats) + ev_bias + 0.5 * cos + esm_bias)
            pathway_combined = gamma * tf_eff + (1.0 - gamma) * ppi_eff

            memory_combined = self.memory_system.integrate_memories(current, fast_memory, slow_features, pathway_combined)
            fast_out = self.memory_system.fast_decoder(fast_memory)
            slow_out = self.memory_system.slow_decoder(slow_features)
            fast_contrib.append(fast_out.detach()); slow_contrib.append(slow_out.detach())

            memory_weight = self.memory_influence_adapter(base_feats)
            effective_influence = self.training_memory_scale * self.memory_influence * memory_weight
            combined_with_memory = pathway_combined + effective_influence * memory_combined

            alpha = torch.sigmoid(self.step_bias) * 0.5 + 0.5 * self.step_head(feats)
            steps_per_round.append(alpha.detach())

            proposed = current + alpha * combined_with_memory
            new_state = self.saturation(x0_f32, proposed, pert_idx)

            should_store, _ = self.memory_system.should_store_episode(new_state - x0_f32, fast_memory, t)
            if should_store:
                episode_input = torch.cat([new_state - x0_f32, new_state - current, current, fast_memory], dim=1)
                episode = self.memory_system.episode_encoder(episode_input.detach())
                slow_memory_bank.append(episode)
                if len(slow_memory_bank) > self.memory_system.memory_bank_size:
                    slow_memory_bank = slow_memory_bank[-self.memory_system.memory_bank_size:]

            p_t = self.halt_head(feats)
            halts_per_round.append(p_t.detach()); w_t = p_t * remaining
            weights_per_round.append(w_t.detach())
            y_mix = y_mix + w_t * new_state; remaining = remaining * (1.0 - p_t)

            router_input = torch.cat([fast_memory, slow_features, current], dim=1)
            _ = self.memory_system.memory_router(router_input).detach()
            current = new_state

            if (t + 1) >= self.min_rounds and (remaining.max().item() <= self.epsilon):
                break

        if remaining.max().item() > 0:
            y_mix = y_mix + remaining * current
            weights_per_round.append(remaining.detach())
            halts_per_round.append(torch.zeros_like(remaining))
            steps_per_round.append(torch.zeros_like(remaining))

        # Heteroscedastic Gaussian emission
        log_sigma = torch.clamp(self.logsigma_head(last_feats), self._logsigma_min, self._logsigma_max)
        if self.sample_mode:
            eps = torch.randn_like(y_mix)
            y_out = y_mix + torch.exp(log_sigma) * eps
        else:
            y_out = y_mix

        yhat = y_out.to(in_dtype)
        if not return_details:
            return yhat
        details = {
            "log_sigma": log_sigma.detach(),
            "kl": kl if kl is not None else yhat.new_tensor(0.0),
            "weights": torch.stack([w.squeeze(1) for w in weights_per_round], dim=1),
            "halt_p": torch.stack([h.squeeze(1) for h in halts_per_round], dim=1),
            "alpha": torch.stack([a.squeeze(1) for a in steps_per_round], dim=1),
        }
        return yhat, details

    def regularization_loss(self) -> torch.Tensor:
        return self.saturation.reg_loss()

# ----------------------------- Baseline stubs (fallback) ------------------- #

class TFPathway(nn.Module):
    def __init__(self, n_genes: int, vocab_size: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(n_genes, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, n_genes, bias=True)
        self.pert_emb = nn.Embedding(vocab_size, hidden)
        nn.init.normal_(self.pert_emb.weight, std=0.02)
    def forward(self, current: torch.Tensor, pert_idx: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(current))
        gate = torch.sigmoid(self.pert_emb(pert_idx))
        z = h * gate
        return self.fc2(z)

class PPIPathway(nn.Module):
    def __init__(self, n_genes: int, vocab_size: int, rank: int = 256):
        super().__init__()
        self.p_emb = nn.Embedding(vocab_size, rank); nn.init.normal_(self.p_emb.weight, std=0.02)
        self.cur_lin = nn.Linear(n_genes, rank, bias=False); self.dec = nn.Linear(rank, n_genes, bias=False)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
    def forward(self, current: torch.Tensor, pert_idx: torch.Tensor) -> torch.Tensor:
        e = self.p_emb(pert_idx); c = self.cur_lin(current); z = F.relu(e + c); return self.dec(z)

# ------------------------------ Datasets ----------------------------------- #

class PseudobulkDataset(Dataset):
    def __init__(self, targets: List[str], y_dict: Dict[str, np.ndarray], vocab: Dict[str, int]):
        self.targets = targets; self.y_dict = y_dict; self.vocab = vocab
        self.G = len(next(iter(y_dict.values())))
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx: int):
        t = self.targets[idx]; y = self.y_dict[t].astype(np.float32); pid = self.vocab[t]
        return torch.from_numpy(y), torch.tensor(pid, dtype=torch.long), t

class SingleCellDataset(Dataset):
    def __init__(self, adata: "ad.AnnData", target_col: str, control_label: str,
                 gene_names: List[str], vocab: Dict[str, int], cell_indices: np.ndarray,
                 x0_matrix: np.ndarray):
        self.adata = adata; self.target_col = target_col; self.control_label = control_label
        self.gene_names = gene_names; self.vocab = vocab; self.idxs = np.asarray(cell_indices)
        self.x0_mat = x0_matrix.astype(np.float32)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i: int):
        ridx = self.idxs[i]; target = str(self.adata.obs[self.target_col].iloc[ridx])
        X = self.adata.X
        y = (X[ridx].toarray().ravel() if issparse(X) else np.asarray(X[ridx]).ravel()).astype(np.float32)
        x0 = self.x0_mat[ridx].astype(np.float32)
        pid = self.vocab.get(target, -1)
        if pid < 0: raise KeyError(f"Perturbation '{target}' not in vocab.")
        return torch.from_numpy(x0), torch.from_numpy(y), torch.tensor(pid, dtype=torch.long), target

# ------------------------------ Warm-start --------------------------------- #

def load_warmstart_weights(model, tf_pathway, ppi_pathway, warmstart_path, device,
                           current_vocab, freeze_pathways=False):
    print(f"[INFO] Loading warm start from: {warmstart_path}")
    # Drop safe_globals (private numpy API broke on numpy >= 2.0). These
    # checkpoints are produced by this script locally and are trusted.
    ckpt = torch.load(warmstart_path, map_location=device, weights_only=False)
    ckpt_vocab = ckpt.get("vocab", {}); ckpt_vocab_size = len(ckpt_vocab); current_vocab_size = len(current_vocab)
    print(f"[INFO] Checkpoint vocab size: {ckpt_vocab_size}, Current vocab size: {current_vocab_size}")

    # Model
    model_state = ckpt["model"]; current_model_state = model.state_dict()
    transferred = skipped = 0
    for name, param in model_state.items():
        if name in current_model_state:
            if param.shape == current_model_state[name].shape:
                current_model_state[name].copy_(param); transferred += 1
            elif "pert_feat.weight" in name:
                # partial copy by shared indices
                common_perts = set(ckpt_vocab.keys()) & set(current_vocab.keys())
                for pert in common_perts:
                    old_idx = ckpt_vocab[pert]; new_idx = current_vocab[pert]
                    current_model_state[name][new_idx] = param[old_idx]
                transferred += 1
            else:
                skipped += 1
        else:
            skipped += 1
    model.load_state_dict(current_model_state)
    print(f"[INFO] Model: transferred {transferred} layers, skipped {skipped}")

    # TF
    tf_state = ckpt["tf"]; current_tf_state = tf_pathway.state_dict(); tf_transferred = 0
    for name, param in tf_state.items():
        if name in current_tf_state and param.shape == current_tf_state[name].shape:
            current_tf_state[name].copy_(param); tf_transferred += 1
    tf_pathway.load_state_dict(current_tf_state)
    print(f"[INFO] TF pathway: transferred {tf_transferred} layers (shape-matched only)")

    # PPI
    ppi_state = ckpt["ppi"]; current_ppi_state = ppi_pathway.state_dict(); ppi_transferred = 0
    for name, param in ppi_state.items():
        if name in current_ppi_state and param.shape == current_ppi_state[name].shape:
            current_ppi_state[name].copy_(param); ppi_transferred += 1
    ppi_pathway.load_state_dict(current_ppi_state)
    print(f"[INFO] PPI pathway: transferred {ppi_transferred} layers (shape-matched only)")

    if freeze_pathways:
        print("[INFO] Freezing TF and PPI pathway parameters")
        for p in tf_pathway.parameters(): p.requires_grad = False
        for p in ppi_pathway.parameters(): p.requires_grad = False
    return model, tf_pathway, ppi_pathway

# ------------------------------ Training ----------------------------------- #

def gaussian_nll(y_true: torch.Tensor, y_mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    return 0.5 * (((y_true - y_mu) ** 2) * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma).mean()

def train_one_epoch(model: HybridMemoryRAMLite, tfm: nn.Module, ppim: nn.Module,
                    loader: DataLoader, x0: torch.Tensor, optimizer: torch.optim.Optimizer,
                    device: torch.device, lambda_pds: float = 0.5, lambda_des: float = 0.5,
                    tau_de: float = 0.25, beta_kl: float = 0.0) -> Dict[str, float]:
    model.train(); tfm.train(); ppim.train()
    total_loss = total_mae = total_pds = total_des = total_kl = 0.0; n = 0
    for y_true, pert_idx, _ in loader:
        y_true = y_true.to(device); pert_idx = pert_idx.to(device)
        B, G = y_true.shape; x0_b = x0.expand(B, G).to(device)

        y_hat, det = model(x0_b, tfm, ppim, pert_idx, y_obs=y_true, return_details=True, deterministic=False)
        log_sigma = det["log_sigma"]; kl = det["kl"]
        nll = gaussian_nll(y_true, y_hat, log_sigma)

        d_hat = y_hat - x0_b; d_true = y_true - x0_b
        loss_pds = info_nce_loss(d_hat, d_true, temperature=0.07)
        p_true_de = soft_de_targets(y_true, x0_b, tau=tau_de); p_pred_de = soft_de_targets(y_hat, x0_b, tau=tau_de)
        loss_des = F.binary_cross_entropy(p_pred_de, p_true_de)

        loss = nll + lambda_pds * loss_pds + lambda_des * loss_des + beta_kl * kl + 1e-4 * model.regularization_loss()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

        total_loss += float(loss.item()) * B
        total_mae  += float(torch.mean(torch.abs(y_hat - y_true)).item()) * B
        total_pds  += float(loss_pds.item()) * B
        total_des  += float(loss_des.item()) * B
        total_kl   += float((kl if isinstance(kl, torch.Tensor) else 0.0)) * B
        n += B
    return {"loss": total_loss/max(1,n), "mae": total_mae/max(1,n), "pds": total_pds/max(1,n),
            "des": total_des/max(1,n), "kl": total_kl/max(1,n)}

@torch.no_grad()
def evaluate(model: HybridMemoryRAMLite, tfm: nn.Module, ppim: nn.Module,
             loader: DataLoader, x0: torch.Tensor, device: torch.device,
             gene_names: List[str], save_preds_path: Optional[str] = None) -> Dict[str, float]:
    model.eval(); tfm.eval(); ppim.eval()
    all_targets: List[str] = []; Y_true: List[np.ndarray] = []; Y_pred: List[np.ndarray] = []
    for y_true, pert_idx, tnames in loader:
        y_true = y_true.to(device); pert_idx = pert_idx.to(device)
        B, G = y_true.shape; x0_b = x0.expand(B, G).to(device)
        y_hat = model(x0_b, tfm, ppim, pert_idx, y_obs=None, return_details=False, deterministic=True)
        Y_true.append(y_true.cpu().numpy()); Y_pred.append(y_hat.cpu().numpy()); all_targets.extend(list(tnames))
    Y_true = np.vstack(Y_true) if len(Y_true) else np.zeros((0, len(gene_names)), dtype=np.float32)
    Y_pred = np.vstack(Y_pred) if len(Y_pred) else np.zeros((0, len(gene_names)), dtype=np.float32)
    if len(all_targets) == 0:
        if save_preds_path is not None:
            pd.DataFrame(columns=gene_names).rename_axis("target").to_csv(save_preds_path)
        return {"mae": 0.0, "pds": 0.0, "des_proxy": 0.0}
    mae = float(np.mean(np.abs(Y_pred - Y_true)))
    d_true = Y_true - x0.cpu().numpy().reshape(1, -1); d_pred = Y_pred - x0.cpu().numpy().reshape(1, -1)
    pds = pds_from_l1(d_true, d_pred)
    G = Y_true.shape[1] if Y_true.size else len(gene_names); k = max(10, min(200, int(0.01 * G)))
    overlaps = [topk_overlap_ratio(d_true[i], d_pred[i], k) for i in range(len(all_targets))]
    des_proxy = float(np.mean(overlaps)) if overlaps else 0.0
    if save_preds_path is not None and len(all_targets) > 0:
        pd.DataFrame(Y_pred, index=all_targets, columns=gene_names).rename_axis("target").to_csv(save_preds_path)
    return {"mae": mae, "pds": pds, "des_proxy": des_proxy}

def train_one_epoch_cells(
    model: HybridMemoryRAMLite, tfm: nn.Module, ppim: nn.Module,
    loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device,
    lambda_pds: float = 0.5, lambda_des: float = 0.5, tau_de: float = 0.25, beta_kl: float = 0.0
) -> Dict[str, float]:
    model.train(); tfm.train(); ppim.train()
    total_loss = total_mae = total_pds = total_des = total_kl = 0.0; n = 0
    for x0, y_true, pert_idx, _ in loader:
        x0 = x0.to(device); y_true = y_true.to(device); pert_idx = pert_idx.to(device)
        y_hat, det = model(x0, tfm, ppim, pert_idx, y_obs=y_true, return_details=True, deterministic=False)
        log_sigma = det["log_sigma"]; kl = det["kl"]; nll = gaussian_nll(y_true, y_hat, log_sigma)
        d_hat, d_true = y_hat - x0, y_true - x0
        loss_pds = info_nce_loss(d_hat, d_true, temperature=0.07)
        p_true_de = soft_de_targets(y_true, x0, tau=tau_de); p_pred_de = soft_de_targets(y_hat, x0, tau=tau_de)
        loss_des = F.binary_cross_entropy(p_pred_de, p_true_de)
        loss = nll + lambda_pds * loss_pds + lambda_des * loss_des + beta_kl * kl + 1e-4 * model.regularization_loss()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        B = y_true.size(0)
        total_loss += float(loss.item()) * B
        total_mae  += float(torch.mean(torch.abs(y_hat - y_true)).item()) * B
        total_pds  += float(loss_pds.item()) * B
        total_des  += float(loss_des.item()) * B
        total_kl   += float((kl if isinstance(kl, torch.Tensor) else 0.0)) * B
        n += B
    return {"loss": total_loss/max(1,n), "mae": total_mae/max(1,n), "pds": total_pds/max(1,n),
            "des": total_des/max(1,n), "kl": total_kl/max(1,n)}

@torch.no_grad()
def evaluate_cells(
    model: HybridMemoryRAMLite, tfm: nn.Module, ppim: nn.Module,
    loader: DataLoader, device: torch.device, gene_names: List[str],
    save_preds_path: Optional[str] = None,
) -> Dict[str, float]:
    model.eval(); tfm.eval(); ppim.eval()
    all_targets = []; Y_true, Y_pred, X0s = [], [], []
    for x0, y_true, pert_idx, tnames in loader:
        x0 = x0.to(device); y_true = y_true.to(device); pert_idx = pert_idx.to(device)
        y_hat = model(x0, tfm, ppim, pert_idx, y_obs=None, return_details=False, deterministic=True)
        Y_true.append(y_true.cpu().numpy()); Y_pred.append(y_hat.cpu().numpy()); X0s.append(x0.cpu().numpy())
        all_targets.extend(list(tnames))
    if not Y_true: return {"mae": 0.0, "pds": 0.0, "des_proxy": 0.0}
    Y_true = np.vstack(Y_true); Y_pred = np.vstack(Y_pred); X0s = np.vstack(X0s)
    mae = float(np.mean(np.abs(Y_pred - Y_true)))
    d_true = Y_true - X0s; d_pred = Y_pred - X0s
    df = pd.DataFrame({"target": all_targets})
    dtrue_pb = pd.DataFrame(d_true).groupby(df["target"]).mean().to_numpy()
    dpred_pb = pd.DataFrame(d_pred).groupby(df["target"]).mean().to_numpy()
    pds = pds_from_l1(dtrue_pb, dpred_pb)
    G = d_true.shape[1]; k = max(10, min(200, int(0.01 * G)))
    overlaps = [topk_overlap_ratio(dtrue_pb[i], dpred_pb[i], k) for i in range(dtrue_pb.shape[0])]
    des_proxy = float(np.mean(overlaps)) if overlaps else 0.0
    if save_preds_path is not None:
        pd.DataFrame(Y_pred, columns=gene_names, index=pd.Index(all_targets, name="target")).to_csv(save_preds_path)
    return {"mae": mae, "pds": pds, "des_proxy": des_proxy}

# ------------------------ Lightning-compatible wrapper --------------------- #

class RAMLightningWrapper(pl.LightningModule):
    """
    Minimal wrapper so your generator can call:
        model(x0_std, g_idx, cov_tensors)
    """
    def __init__(self, core: HybridMemoryRAMLite, tfm: nn.Module, ppim: nn.Module,
                 vocab: Dict[str,int], gene_names: List[str]):
        super().__init__()
        self.core = core; self.tfm = tfm; self.ppim = ppim
        self.vocab = vocab; self.gene_names = gene_names
        self._ctrl_mean = None; self._ctrl_std = None; self._apply_log1p = False

    @property
    def device(self):
        return next(self.core.parameters()).device

    def set_control_stats(self, ctrl_mean: np.ndarray, ctrl_std: np.ndarray, apply_log1p: bool):
        self._ctrl_mean = torch.from_numpy(ctrl_mean.astype(np.float32)).to(self.device)
        self._ctrl_std  = torch.from_numpy(ctrl_std.astype(np.float32)).to(self.device)
        self._apply_log1p = bool(apply_log1p)

    def forward(self, x0_std: torch.Tensor, g_idx: torch.Tensor, cov_tensors=None):
        if self._ctrl_mean is None or self._ctrl_std is None:
            raise RuntimeError("Control stats not set; call set_control_stats() first.")
        x0_raw = x0_std * self._ctrl_std.unsqueeze(0) + self._ctrl_mean.unsqueeze(0)
        y_raw = self.core(x0_raw, self.tfm, self.ppim, g_idx, y_obs=None, return_details=False, deterministic=True)
        y_std = (y_raw - self._ctrl_mean.unsqueeze(0)) / self._ctrl_std.unsqueeze(0)
        return y_std, {}

# ------------------- Unseen generation (enhanced) -------------------------- #

def _expand_embedding(old: nn.Embedding, new_rows: torch.FloatTensor) -> nn.Embedding:
    # Preserve device & dtype of the original embedding
    device = old.weight.device
    dtype  = old.weight.dtype

    W = old.weight.detach().to(device=device, dtype=dtype)
    new_rows = new_rows.to(device=device, dtype=dtype)
    W2 = torch.cat([W, new_rows], dim=0)

    # Create the new embedding directly on the same device/dtype
    new_emb = nn.Embedding(W2.size(0), W2.size(1)).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_emb.weight.copy_(W2)
    new_emb.weight.requires_grad = old.weight.requires_grad
    return new_emb


def extend_vocab_for_unseen(wrapper: RAMLightningWrapper,
                            new_symbols: List[str],
                            scgpt_map: Dict[str, torch.FloatTensor],
                            esm_map:   Dict[str, torch.FloatTensor]):
    core, tfm, ppim, vocab = wrapper.core, wrapper.tfm, wrapper.ppim, wrapper.vocab
    add_syms = [s for s in new_symbols if s not in vocab]
    if not add_syms:
        return

    V_old = len(vocab)
    n_add = len(add_syms)

    # ---------- core: pert_feat ----------
    d_core = core.pert_feat.embedding_dim
    core.pert_feat = _expand_embedding(core.pert_feat, torch.zeros(n_add, d_core))

    # ---------- core: Saturation.U (this is the common culprit) ----------
    if hasattr(core, "saturation") and hasattr(core.saturation, "U"):
        d_U = core.saturation.U.embedding_dim
        core.saturation.U = _expand_embedding(core.saturation.U, torch.zeros(n_add, d_U))

    # ---------- TF pathway ----------
    if hasattr(tfm, "ext") and hasattr(tfm.ext, "emb"):
        # ScGPT-based path
        D = tfm.ext.emb.embedding_dim
        rows = []
        for s in add_syms:
            vec = scgpt_map.get(s, None)
            if isinstance(vec, torch.Tensor):
                rows.append(vec)
            else:
                rows.append(torch.zeros(D))
        tfm.ext.emb = _expand_embedding(tfm.ext.emb, torch.stack(rows, dim=0))
    elif hasattr(tfm, "pert_emb"):
        # Baseline TF path
        D = tfm.pert_emb.embedding_dim
        tfm.pert_emb = _expand_embedding(tfm.pert_emb, torch.zeros(n_add, D))

    # ---------- PPI pathway ----------
    if hasattr(ppim, "ext") and hasattr(ppim.ext, "emb"):
        # ESM-based path
        D = ppim.ext.emb.embedding_dim
        rows = []
        for s in add_syms:
            vec = esm_map.get(s, None)
            if isinstance(vec, torch.Tensor):
                rows.append(vec)
            else:
                rows.append(torch.zeros(D))
        ppim.ext.emb = _expand_embedding(ppim.ext.emb, torch.stack(rows, dim=0))
    elif hasattr(ppim, "p_emb"):
        # Baseline PPI path
        D = ppim.p_emb.embedding_dim
        ppim.p_emb = _expand_embedding(ppim.p_emb, torch.zeros(n_add, D))

    # ---------- update vocab ----------
    for i, s in enumerate(add_syms):
        vocab[s] = V_old + i
    core.vocab_size = len(vocab)  # keep it consistent

    # ---------- ensure modules live on the right device ----------
    dev = wrapper.device
    core.to(dev); tfm.to(dev); ppim.to(dev)

    # ---------- (optional) capacity sanity-check ----------
    V_needed = len(vocab)
    caps = []
    caps.append(("core.pert_feat", core.pert_feat.num_embeddings))
    if hasattr(core, "saturation") and hasattr(core.saturation, "U"):
        caps.append(("core.saturation.U", core.saturation.U.num_embeddings))
    if hasattr(tfm, "ext") and hasattr(tfm.ext, "emb"):
        caps.append(("tfm.ext.emb", tfm.ext.emb.num_embeddings))
    if hasattr(tfm, "pert_emb"):
        caps.append(("tfm.pert_emb", tfm.pert_emb.num_embeddings))
    if hasattr(ppim, "ext") and hasattr(ppim.ext, "emb"):
        caps.append(("ppim.ext.emb", ppim.ext.emb.num_embeddings))
    if hasattr(ppim, "p_emb"):
        caps.append(("ppim.p_emb", ppim.p_emb.num_embeddings))

    for name, cap in caps:
        if cap < V_needed:
            raise RuntimeError(f"[extend_vocab_for_unseen] {name} capacity {cap} < vocab size {V_needed}")



def build_evidence(vocab: Dict[str,int], tf_set: set,
                   scgpt_table: Optional[torch.FloatTensor],
                   esm_table: Optional[torch.FloatTensor],
                   go_bias: Optional[Dict[str, np.ndarray]] = None,
                   scgpt_map: Optional[Dict[str, torch.FloatTensor]] = None,
                   esm_map: Optional[Dict[str, torch.FloatTensor]] = None) -> torch.FloatTensor:
    V = len(vocab)
    evid = np.zeros((V, 7), dtype=np.float32)  # [is_TF, has_scGPT, has_ESM2, go_tf, go_sig, go_mito, go_cycle]
    for g, i in vocab.items():
        evid[i,0] = 1.0 if g in tf_set else 0.0
        has_scgpt = False
        has_esm2 = False
        if scgpt_table is not None and i < scgpt_table.size(0):
            has_scgpt = torch.abs(scgpt_table[i]).sum().item() > 0
        elif scgpt_map is not None and g in scgpt_map:
            has_scgpt = torch.abs(scgpt_map[g]).sum().item() > 0
        if esm_table is not None and i < esm_table.size(0):
            has_esm2 = torch.abs(esm_table[i]).sum().item() > 0
        elif esm_map is not None and g in esm_map:
            has_esm2 = torch.abs(esm_map[g]).sum().item() > 0
        evid[i,1] = 1.0 if has_scgpt else 0.0
        evid[i,2] = 1.0 if has_esm2 else 0.0
        if go_bias and g in go_bias:
            evid[i,3:7] = go_bias[g]
    return torch.from_numpy(evid)

def generate_predictions_enhanced(model: pl.LightningModule, ad_tr: ad.AnnData,
                                  gene_list: List[str], gene_to_idx: Dict[str,int],
                                  ctrl_mask: np.ndarray, val_counts_csv: str,
                                  out_h5ad: str, ctrl_pool_cap: int = 100000,
                                  batch_size: int = 512,
                                  covariates: Optional[List[str]] = None,
                                  cov_defaults: Optional[Dict[str,int]] = None,
                                  dynamic_unseen_vocab: bool = True,
                                  scgpt_map: Optional[Dict[str, torch.FloatTensor]] = None,
                                  esm_map: Optional[Dict[str, torch.FloatTensor]] = None,
                                  tf_set: Optional[set] = None,
                                  go_bias: Optional[Dict[str, np.ndarray]] = None) -> ad.AnnData:
    """
    Synthesizes cells for unseen val_counts CSV using a control pool as baselines.
    Automatically extends vocab and external tables for unseen perturbations if enabled.
    """
    print("[Pred] generating...")
    covariates = covariates or []
    X_ctrl = ad_tr.X[ctrl_mask]
    X_ctrl = X_ctrl.tocsr() if sp.issparse(X_ctrl) else np.asarray(X_ctrl)
    if ctrl_pool_cap and X_ctrl.shape[0] > ctrl_pool_cap:
        sel = np.random.choice(X_ctrl.shape[0], size=ctrl_pool_cap, replace=False)
        X_ctrl = X_ctrl[sel]

    # detect counts → log1p
    vals_sample = X_ctrl[: min(1000, X_ctrl.shape[0])]
    vals_sample = vals_sample.toarray() if sp.issparse(vals_sample) else np.asarray(vals_sample)
    apply_log1p = np.all(np.mod(vals_sample, 1) == 0) and vals_sample.max() > 50

    X_ctrl_arr = X_ctrl.toarray() if sp.issparse(X_ctrl) else np.asarray(X_ctrl)
    X_ctrl_arr = (np.log1p(X_ctrl_arr) if apply_log1p else X_ctrl_arr).astype(np.float32)

    ctrl_mean = X_ctrl_arr.mean(axis=0).astype(np.float32)
    ctrl_std  = (X_ctrl_arr.std(axis=0) + 1e-6).astype(np.float32)

    vc = pd.read_csv(val_counts_csv)
    tg_col = next((c for c in vc.columns if "target" in c.lower() and "gene" in c.lower()), vc.columns[0])
    n_col  = next((c for c in vc.columns if "n" in c.lower() and "cell" in c.lower()), vc.columns[1])
    vc = vc.rename(columns={tg_col: "target_gene", n_col: "n_cells"})
    vc["target_gene"] = vc["target_gene"].astype(str); vc["n_cells"] = vc["n_cells"].astype(int)
    if not (vc["target_gene"].str.lower() == "non-targeting").any():
        vc = pd.concat([vc, pd.DataFrame([{"target_gene":"non-targeting","n_cells":max(10, int(vc["n_cells"].mean()))}])], ignore_index=True)

    # Optionally expand vocab for unseen perts using external maps
    unseen = [tg for tg in vc["target_gene"].tolist() if tg not in model.vocab]
    if dynamic_unseen_vocab and len(unseen) > 0:
        print(f"[Pred] expanding vocab for {len(unseen)} unseen perts: {unseen[:5]}{'...' if len(unseen)>5 else ''}")
        extend_vocab_for_unseen(model, unseen, scgpt_map or {}, esm_map or {})
        # Rebuild evidence for expanded vocab while retaining external embedding presence flags.
        evid = build_evidence(model.vocab, tf_set or set(), None, None, go_bias,
                              scgpt_map=scgpt_map, esm_map=esm_map)
        model.core.set_evidence_vec(evid.to(model.device))

    # Set control stats for the wrapper so it can (de)standardize internally
    if hasattr(model, "set_control_stats"):
        model.set_control_stats(ctrl_mean, ctrl_std, apply_log1p)

    obs_rows, X_rows = [], []
    model.eval()
    with torch.no_grad():
        for _, r in vc.iterrows():
            tg = str(r["target_gene"]); k = int(r["n_cells"])
            all_pred = []
            for start in range(0, k, batch_size):
                end = min(start + batch_size, k)
                idxs = np.random.choice(X_ctrl_arr.shape[0], size=end-start, replace=True)
                x0_std_np = (X_ctrl_arr[idxs] - ctrl_mean) / ctrl_std
                x0 = torch.from_numpy(x0_std_np).float().to(model.device)

                if tg in model.vocab:
                    g_idx = torch.full((end-start,), model.vocab[tg], dtype=torch.long, device=model.device)
                else:
                    # as a last resort (should be rare if expansion is on)
                    g_idx = torch.zeros((end-start,), dtype=torch.long, device=model.device)

                cov_tensors = None
                if covariates:
                    cov_tensors = [torch.full((end-start,), cov_defaults.get(c,0), dtype=torch.long, device=model.device)
                                   for c in covariates]

                yhat_std, _ = model(x0, g_idx, cov_tensors)
                yhat = (yhat_std * torch.from_numpy(ctrl_std).to(yhat_std.device) +
                        torch.from_numpy(ctrl_mean).to(yhat_std.device))
                all_pred.append(yhat.clamp_min(0.0).cpu().numpy().astype(np.float32))
            X_pred_pert = np.vstack(all_pred)
            X_rows.append(X_pred_pert)
            obs_rows.append(pd.DataFrame({"target_gene":[tg]*k}))
            if len(obs_rows) % 10 == 0: print(f"[Pred] processed {len(obs_rows)} perts")

    X_pred = np.vstack(X_rows).astype(np.float32)
    obs_pred = pd.concat(obs_rows, ignore_index=True)
    ad_pred = ad.AnnData(X_pred, obs=obs_pred, var=pd.DataFrame(index=pd.Index(gene_list)))
    ad_pred.X = ad_pred.X.astype(np.float32, copy=False)
    ad_pred.write_h5ad(out_h5ad, compression="gzip")
    print(f"[Pred] saved {ad_pred.n_obs} x {ad_pred.n_vars} -> {out_h5ad}")
    return ad_pred

# ------------------------------ Glue (common) ------------------------------- #

def build_pseudobulks(adata: "ad.AnnData", target_col: str, control_label: str,
                      min_cells_per_pert: int = 20) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    if target_col not in adata.obs.columns:
        raise KeyError(f"obs column '{target_col}' not found.")
    labels = adata.obs[target_col].astype(str)
    if control_label not in set(labels):
        raise KeyError(f"Control label '{control_label}' not found in '{target_col}'.")
    perts = group_mean_from_adata(adata, labels, min_cells=min_cells_per_pert)
    ctrl_dict = group_mean_from_adata(adata, labels == control_label, min_cells=1)
    if str(True) in ctrl_dict:
        x0 = ctrl_dict[str(True)]
    else:
        mask = (labels == control_label).values; X = adata.X
        x0 = X[mask].mean(axis=0).A1.astype(np.float32) if issparse(X) \
             else X[mask].mean(axis=0).astype(np.float32).ravel()
    if control_label in perts: del perts[control_label]
    gene_names = adata.var_names.tolist()
    return x0, perts, gene_names

def build_vocab(y_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
    targets = sorted(list(y_dict.keys())); return {t: i for i, t in enumerate(targets)}

def make_zeroshot_splits(
    targets: List[str], vocab: Dict[str, int],
    scgpt_table: Optional[torch.FloatTensor],
    esm_table: Optional[torch.FloatTensor],
    train_frac: float = 0.7, val_frac: float = 0.15,
    zeroshot_frac: float = 0.15, seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns (train, val_iid, test_seen, test_zeroshot).
    Zero-shot targets have external embeddings but are never in training.
    """
    import random
    rng = random.Random(seed)
    has_ext, no_ext = [], []
    for t in targets:
        idx = vocab.get(t, -1)
        if idx < 0:
            no_ext.append(t)
            continue
        covered = False
        if scgpt_table is not None and idx < scgpt_table.size(0):
            covered = covered or (torch.abs(scgpt_table[idx]).sum().item() > 0)
        if esm_table is not None and idx < esm_table.size(0):
            covered = covered or (torch.abs(esm_table[idx]).sum().item() > 0)
        (has_ext if covered else no_ext).append(t)

    n_zs = max(1, min(int(len(targets) * zeroshot_frac), len(has_ext)))
    rng.shuffle(has_ext)
    zs_targets = has_ext[:n_zs]
    rest = has_ext[n_zs:] + no_ext
    rng.shuffle(rest)

    denom = max(1e-8, (1.0 - zeroshot_frac))
    n_tr = int(len(rest) * train_frac / denom)
    n_va = int(len(rest) * val_frac / denom)
    return rest[:n_tr], rest[n_tr:n_tr+n_va], rest[n_tr+n_va:], zs_targets

def make_splits(targets: List[str], train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42):
    train_targets, temp = train_test_split(targets, train_size=train_frac, random_state=seed, shuffle=True)
    val_size = int(round(len(targets) * val_frac)); val_targets = temp[:val_size]; test_targets = temp[val_size:]
    return train_targets, val_targets, test_targets

# ------------------------------ Main --------------------------------------- #

def main(args):
    seed_everything(args.seed); ensure_dir(args.outdir)
    print(f"[INFO] Reading h5ad: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    adata = adata[adata.obs[args.target_col].notna()].copy()

    # External resources (paths may be None)
    tf_set_all = load_tf_list(args.tf_list)
    go_bias_map = load_go_bias(args.go_csv)

    if args.infer_valcounts:
        # Inference-only branch
        ckpt_path = args.warmstart if (args.warmstart and os.path.exists(args.warmstart)) \
                    else os.path.join(args.outdir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Provide --warmstart or train first.")
        device = resolve_device(args)
        # safe_globals previously listed np.core.multiarray._reconstruct which is
        # a private numpy API that moved/disappeared on numpy >= 2.0. Drop the
        # safe-globals context: these checkpoints are produced by this script
        # locally and we trust them. weights_only=False is required because they
        # embed numpy objects (vocab dict, gene_names list).
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        gene_names = ckpt["gene_names"]; vocab = ckpt["vocab"]; G = len(gene_names)

        # Load external tables aligned to current vocab (for evidence & possible expansion)
        scgpt_table, scgpt_map = load_scgpt_table(args.scgpt_embeddings_tsv, vocab)
        esm_table, esm_map = load_esm2_table(args.esm2_embeddings_csv, vocab, sep=args.esm2_sep)

        # Build model & swap in externalized pathways when available
        model = HybridMemoryRAMLite(
            n_genes=G, vocab_size=len(vocab),
            max_rounds=args.max_rounds, min_rounds=args.min_rounds, epsilon=args.epsilon,
            step_init=args.step_init, sat_reg=args.sat_reg,
            feat_dim=args.feat_dim, fast_memory_dim=args.fast_dim, slow_memory_dim=args.slow_dim,
            memory_bank_size=args.memory_bank, n_attention_heads=4,
            memory_influence=args.memory_influence, sat_rank=args.sat_rank, slow_decoder_rank=args.slow_rank,
            use_cvae=args.use_cvae, z_dim=args.z_dim
        ).to(device)
        model._pert_dropout_p = getattr(args, "pert_dropout", 0.0)

        if scgpt_table is not None:
            tf_pathway = TFPathwayScGPT(n_genes=G, scgpt_table=scgpt_table.float(),
                                        hidden=args.tf_hidden, trainable_proj=not args.freeze_proj,
                                        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha).to(device)
        else:
            tf_pathway = TFPathway(n_genes=G, vocab_size=len(vocab), hidden=args.tf_hidden).to(device)

        if esm_table is not None:
            ppi_pathway = PPIPathwayESM(n_genes=G, esm_table=esm_table.float(),
                                        rank=args.ppi_rank, trainable_proj=not args.freeze_proj,
                                        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha).to(device)
        else:
            ppi_pathway = PPIPathway(n_genes=G, vocab_size=len(vocab), rank=args.ppi_rank).to(device)

        # Register evidence projection before loading checkpoint so learned weights can restore.
        evid = build_evidence(vocab, tf_set_all, scgpt_table, esm_table, go_bias_map,
                              scgpt_map=scgpt_map, esm_map=esm_map)
        model.set_evidence_vec(evid.to(device))

        model.load_state_dict(ckpt["model"], strict=False)
        try:
            tf_state = ckpt["tf"]; tf_pathway.load_state_dict(tf_state, strict=False)
        except Exception: pass
        try:
            ppi_state = ckpt["ppi"]; ppi_pathway.load_state_dict(ppi_state, strict=False)
        except Exception: pass

        wrapper = RAMLightningWrapper(model, tf_pathway, ppi_pathway, vocab=vocab, gene_names=gene_names)

        ctrl_mask = (adata.obs[args.target_col].astype(str) == args.control_label).values
        _ = generate_predictions_enhanced(
            wrapper, adata, gene_names, vocab, ctrl_mask,
            args.val_counts_csv, args.out_h5ad,
            ctrl_pool_cap=args.ctrl_pool_cap, batch_size=args.batch_size_pred,
            covariates=None, cov_defaults=None, dynamic_unseen_vocab=not args.no_dynamic_unseen,
            scgpt_map=scgpt_map, esm_map=esm_map, tf_set=tf_set_all, go_bias=go_bias_map
        )
        return

    # Normalize unless user says otherwise
    if args.skip_norm:
        print("[WARN] Skipping normalization; assuming adata.X already normalized/logged.")
    else:
        print("[INFO] Normalizing counts and log1p ...")
        sc.pp.normalize_total(adata); sc.pp.log1p(adata)

    if not args.single_cell:
        # === PSEUDOBULK MODE ===
        print("[INFO] Pseudobulk mode.")
        x0, y_dict, gene_names = build_pseudobulks(adata, args.target_col, args.control_label, min_cells_per_pert=args.min_cells_per_pert)
        G = len(gene_names); all_targets = sorted(list(y_dict.keys()))
        if len(all_targets) == 0:
            raise RuntimeError("No eligible perturbations after filtering.")
        vocab = build_vocab(y_dict)

        # External tables aligned to vocab
        scgpt_table, _ = load_scgpt_table(args.scgpt_embeddings_tsv, vocab)
        esm_table, _   = load_esm2_table(args.esm2_embeddings_csv, vocab, sep=args.esm2_sep)

        if getattr(args, "zeroshot_eval", False):
            train_t, val_t, test_t, test_zs_t = make_zeroshot_splits(
                all_targets, vocab, scgpt_table, esm_table,
                train_frac=args.train_frac, val_frac=args.val_frac,
                zeroshot_frac=args.zeroshot_frac, seed=args.seed
            )
            print(f"[INFO] Zero-shot split — train:{len(train_t)} val:{len(val_t)} "
                  f"test_seen:{len(test_t)} test_zs:{len(test_zs_t)}")
        else:
            train_t, val_t, test_t = make_splits(
                all_targets, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
            )
            test_zs_t = []
            print(f"[INFO] Split perts — train: {len(train_t)}, val: {len(val_t)}, test: {len(test_t)}")

        # Datasets
        ds_train = PseudobulkDataset(train_t, y_dict, vocab)
        ds_val   = PseudobulkDataset(val_t,   y_dict, vocab)
        ds_test  = PseudobulkDataset(test_t,  y_dict, vocab)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
        dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, drop_last=False)
        dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, drop_last=False)
        if test_zs_t:
            ds_test_zs = PseudobulkDataset(test_zs_t, y_dict, vocab)
            dl_test_zs = DataLoader(ds_test_zs, batch_size=args.batch_size, shuffle=False, drop_last=False)

        device = resolve_device(args)
        print(f"[INFO] Using device: {device}")

        model = HybridMemoryRAMLite(
            n_genes=G, vocab_size=len(vocab),
            max_rounds=args.max_rounds, min_rounds=args.min_rounds,
            epsilon=args.epsilon, step_init=args.step_init, sat_reg=args.sat_reg,
            feat_dim=args.feat_dim, fast_memory_dim=args.fast_dim, slow_memory_dim=args.slow_dim,
            memory_bank_size=args.memory_bank, n_attention_heads=4,
            memory_influence=args.memory_influence, sat_rank=args.sat_rank, slow_decoder_rank=args.slow_rank,
            use_cvae=args.use_cvae, z_dim=args.z_dim
        ).to(device)
        model._pert_dropout_p = getattr(args, "pert_dropout", 0.0)

        # Choose pathway implementations:
        if scgpt_table is not None:
            tf_pathway = TFPathwayScGPT(n_genes=G, scgpt_table=scgpt_table.float(),
                                        hidden=args.tf_hidden, trainable_proj=not args.freeze_proj,
                                        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha).to(device)
        else:
            tf_pathway = TFPathway(n_genes=G, vocab_size=len(vocab), hidden=args.tf_hidden).to(device)

        if esm_table is not None:
            ppi_pathway = PPIPathwayESM(n_genes=G, esm_table=esm_table.float(),
                                        rank=args.ppi_rank, trainable_proj=not args.freeze_proj,
                                        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha).to(device)
        else:
            ppi_pathway = PPIPathway(n_genes=G, vocab_size=len(vocab), rank=args.ppi_rank).to(device)

        # Register evidence before optimizer so evidence_proj gets trained.
        evid = build_evidence(vocab, tf_set_all, scgpt_table, esm_table, go_bias_map)
        model.set_evidence_vec(evid.to(device))

        all_params = list(model.parameters()) + list(tf_pathway.parameters()) + list(ppi_pathway.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        x0_t = torch.from_numpy(x0.astype(np.float32)).to(device).view(1, -1)
        best_val = None; history = []
        for epoch in range(1, args.epochs + 1):
            # beta_kl warmup: 0 at epoch 1, reaches beta_kl at epoch warmup+1.
            beta_curr = min(args.beta_kl * max(0, epoch - 1) / max(1, args.kl_warmup_epochs), args.beta_kl) if args.use_cvae else 0.0
            tr = train_one_epoch(model, tf_pathway, ppi_pathway, dl_train, x0_t, optimizer, device,
                                 lambda_pds=args.lambda_pds, lambda_des=args.lambda_des, tau_de=args.tau_de,
                                 beta_kl=beta_curr)
            va = evaluate(model, tf_pathway, ppi_pathway, dl_val, x0_t, device, gene_names, save_preds_path=None)
            scheduler.step(va["mae"])
            hist_entry = {"epoch": epoch, **{f"train_{k}": v for k,v in tr.items()}, **{f"val_{k}": v for k,v in va.items()}, "beta_kl": beta_curr}
            history.append(hist_entry)
            print(f"[E{epoch:03d}] loss={tr['loss']:.4f} | mae={tr['mae']:.4f} | pds={tr['pds']:.4f} | des={tr['des']:.4f} | kl={tr['kl']:.4f} || "
                  f"VAL mae={va['mae']:.4f} | pds={va['pds']:.4f} | des≈{va['des_proxy']:.4f} | β={beta_curr:.3f}")
            # Best-model selection by --best_metric (default: pds, since the loop
            # gating metric is cell-eval PDS). Fall back to MAE if requested.
            improved = (
                best_val is None
                or (args.best_metric == "pds" and va.get("pds", 0.0) > best_val.get("pds", -1.0))
                or (args.best_metric == "mae" and va["mae"] < best_val["mae"])
            )
            if improved:
                best_val = va
                torch.save({"model": model.state_dict(), "tf": tf_pathway.state_dict(), "ppi": ppi_pathway.state_dict(),
                            "config": vars(args), "gene_names": gene_names, "vocab": vocab, "x0": x0},
                           os.path.join(args.outdir, "best_model.pt"))
        with open(os.path.join(args.outdir, "training_history.json"), "w") as f: json.dump(history, f, indent=2)

        ckpt_path = os.path.join(args.outdir, "best_model.pt")
        if os.path.exists(ckpt_path):
            # Drop safe_globals (private numpy API broke on numpy >= 2.0).
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"]); 
            try: tf_pathway.load_state_dict(ckpt["tf"], strict=False)
            except Exception: pass
            try: ppi_pathway.load_state_dict(ckpt["ppi"], strict=False)
            except Exception: pass
            print(f"[INFO] Loaded best model from {ckpt_path}")
        preds_test_path = os.path.join(args.outdir, "predictions_test.csv")
        test_metrics = evaluate(model, tf_pathway, ppi_pathway, dl_test, x0_t, device, gene_names, save_preds_path=preds_test_path)
        print(f"[TEST] MAE={test_metrics['mae']:.4f} | PDS={test_metrics['pds']:.4f} | DES≈{test_metrics['des_proxy']:.4f}")
        with open(os.path.join(args.outdir, "metrics_test.json"), "w") as f: json.dump(test_metrics, f, indent=2)
        if test_zs_t:
            zs_preds_path = os.path.join(args.outdir, "predictions_zeroshot.csv")
            zs_metrics = evaluate(model, tf_pathway, ppi_pathway, dl_test_zs, x0_t, device, gene_names, save_preds_path=zs_preds_path)
            print(f"[ZERO-SHOT] MAE={zs_metrics['mae']:.4f} | PDS={zs_metrics['pds']:.4f} | DES≈{zs_metrics['des_proxy']:.4f}")
            with open(os.path.join(args.outdir, "metrics_zeroshot.json"), "w") as f:
                json.dump(zs_metrics, f, indent=2)
        with open(os.path.join(args.outdir, "pert_vocab.json"), "w") as f: json.dump(build_vocab(y_dict), f, indent=2)
        pd.Series(x0, index=gene_names, name="x0").to_csv(os.path.join(args.outdir, "control_pseudobulk.csv"))
        print(f"[DONE] Wrote predictions to: {preds_test_path}")
        return

    else:
        # === SINGLE-CELL MODE ===
        print("[INFO] Single-cell mode.")
        gene_names = adata.var_names.tolist(); G = len(gene_names)
        target_col = args.target_col; control_label = args.control_label
        is_control = (adata.obs[target_col].astype(str) == control_label).values

        groupby_cols = [c.strip() for c in args.x0_groupby.split(",") if c.strip()] if args.x0_mode == "per_group" else None
        print(f"[INFO] x0 mode: {'per-group ' + str(groupby_cols) if groupby_cols else 'global'}")
        ctrl_lookup = compute_control_lookup(adata, target_col, control_label, groupby=groupby_cols, min_ctrl=1)
        global_fallback = ctrl_lookup.get((), None)
        x0_mat = per_cell_baseline(adata, groupby_cols, ctrl_lookup, fallback_global=global_fallback)

        treated_mask = ~is_control & adata.obs[target_col].notna().values
        perts = np.unique(adata.obs.loc[treated_mask, target_col].astype(str))
        vocab = {t: i for i, t in enumerate(sorted(perts))}
        print(f"[INFO] {len(vocab)} treated perturbations in vocab.")

        # External tables aligned to vocab
        scgpt_table, _ = load_scgpt_table(args.scgpt_embeddings_tsv, vocab)
        esm_table, _   = load_esm2_table(args.esm2_embeddings_csv, vocab, sep=args.esm2_sep)

        # ---- Split (cell-type-OOD optional)
        heldout_cts = [c.strip() for c in (args.heldout_cts.split(",") if args.heldout_cts else []) if c.strip()]
        if heldout_cts:
            ct_col = args.ct_col if args.ct_col else "cell_type"
            in_ct  = adata.obs[ct_col].astype(str).isin(heldout_cts)
            idx_test = np.where(treated_mask & in_ct)[0]
            idx_pool = np.where(treated_mask & ~in_ct)[0]
            y_pool   = adata.obs[target_col].astype(str).values[idx_pool]
            if len(np.unique(y_pool)) > 1:
                train_idx, val_idx = train_test_split(idx_pool, train_size=args.train_frac, random_state=args.seed, stratify=y_pool)
            else:
                train_idx, val_idx = train_test_split(idx_pool, train_size=args.train_frac, random_state=args.seed)
            print(f"[INFO] Held-out CTs={heldout_cts} -> TEST cells: {len(idx_test)}; TRAIN={len(train_idx)} VAL={len(val_idx)}")
        else:
            idx_all = np.where(treated_mask)[0]
            y_all = adata.obs[target_col].astype(str).values[idx_all]
            if len(np.unique(y_all)) > 1:
                train_idx, temp_idx = train_test_split(idx_all, train_size=args.train_frac, random_state=args.seed, stratify=y_all)
            else:
                train_idx, temp_idx = train_test_split(idx_all, train_size=args.train_frac, random_state=args.seed)
            y_temp = adata.obs[target_col].astype(str).values[temp_idx]
            val_frac_adjusted = args.val_frac / (1 - args.train_frac)
            if len(np.unique(y_temp)) > 1:
                val_idx, test_idx = train_test_split(temp_idx, train_size=val_frac_adjusted, random_state=args.seed, stratify=y_temp)
            else:
                val_idx, test_idx = train_test_split(temp_idx, train_size=val_frac_adjusted, random_state=args.seed)
            idx_test = test_idx
            print(f"[INFO] Cells — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(idx_test)}")

        ds_train = SingleCellDataset(adata, target_col, control_label, gene_names, vocab, train_idx, x0_mat)
        ds_val   = SingleCellDataset(adata, target_col, control_label, gene_names, vocab, val_idx,   x0_mat)
        ds_test  = SingleCellDataset(adata, target_col, control_label, gene_names, vocab, idx_test,  x0_mat)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
        dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, drop_last=False)
        dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, drop_last=False)

        device = resolve_device(args)
        print(f"[INFO] Using device: {device}")

        model = HybridMemoryRAMLite(
            n_genes=G, vocab_size=len(vocab),
            max_rounds=args.max_rounds, min_rounds=args.min_rounds,
            epsilon=args.epsilon, step_init=args.step_init, sat_reg=args.sat_reg,
            feat_dim=args.feat_dim, fast_memory_dim=args.fast_dim, slow_memory_dim=args.slow_dim,
            memory_bank_size=args.memory_bank, n_attention_heads=4,
            memory_influence=args.memory_influence, sat_rank=args.sat_rank, slow_decoder_rank=args.slow_rank,
            use_cvae=args.use_cvae, z_dim=args.z_dim
        ).to(device)
        model._pert_dropout_p = getattr(args, "pert_dropout", 0.0)

        if scgpt_table is not None:
            tf_pathway = TFPathwayScGPT(n_genes=G, scgpt_table=scgpt_table.float(),
                                        hidden=args.tf_hidden, trainable_proj=not args.freeze_proj,
                                        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha).to(device)
        else:
            tf_pathway = TFPathway(n_genes=G, vocab_size=len(vocab), hidden=args.tf_hidden).to(device)

        if esm_table is not None:
            ppi_pathway = PPIPathwayESM(n_genes=G, esm_table=esm_table.float(),
                                        rank=args.ppi_rank, trainable_proj=not args.freeze_proj,
                                        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha).to(device)
        else:
            ppi_pathway = PPIPathway(n_genes=G, vocab_size=len(vocab), rank=args.ppi_rank).to(device)

        # Register evidence before warmstart/optimizer to preserve/load projection weights.
        evid = build_evidence(vocab, tf_set_all, scgpt_table, esm_table, go_bias_map)
        model.set_evidence_vec(evid.to(device))

        if args.warmstart and os.path.exists(args.warmstart):
            model, tf_pathway, ppi_pathway = load_warmstart_weights(
                model, tf_pathway, ppi_pathway, args.warmstart, device, vocab, freeze_pathways=args.freeze_pathways
            )
            all_params = [p for p in list(model.parameters()) + list(tf_pathway.parameters()) + list(ppi_pathway.parameters()) if p.requires_grad]
            optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
        else:
            all_params = list(model.parameters()) + list(tf_pathway.parameters()) + list(ppi_pathway.parameters())
            optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
        # verbose= was deprecated in PyTorch >= 2.2; removed.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        best_val = None; history = []
        for epoch in range(1, args.epochs + 1):
            # beta_kl warmup: 0 at epoch 1, reaches beta_kl at epoch warmup+1.
            beta_curr = min(args.beta_kl * max(0, epoch - 1) / max(1, args.kl_warmup_epochs), args.beta_kl) if args.use_cvae else 0.0
            tr = train_one_epoch_cells(model, tf_pathway, ppi_pathway, dl_train, optimizer, device,
                                       lambda_pds=args.lambda_pds, lambda_des=args.lambda_des, tau_de=args.tau_de,
                                       beta_kl=beta_curr)
            va = evaluate_cells(model, tf_pathway, ppi_pathway, dl_val, device, gene_names, save_preds_path=None)
            scheduler.step(va["mae"])
            hist_entry = {"epoch": epoch, **{f"train_{k}": v for k,v in tr.items()}, **{f"val_{k}": v for k,v in va.items()}, "beta_kl": beta_curr}
            history.append(hist_entry)
            print(f"[E{epoch:03d}] loss={tr['loss']:.4f} | mae={tr['mae']:.4f} | pds={tr['pds']:.4f} | des={tr['des']:.4f} | kl={tr['kl']:.4f} "
                  f"|| VAL mae={va['mae']:.4f} | pds={va['pds']:.4f} | des≈{va['des_proxy']:.4f} | β={beta_curr:.3f}")
            improved = (
                best_val is None
                or (args.best_metric == "pds" and va.get("pds", 0.0) > best_val.get("pds", -1.0))
                or (args.best_metric == "mae" and va["mae"] < best_val["mae"])
            )
            if improved:
                best_val = va
                torch.save({"model": model.state_dict(), "tf": tf_pathway.state_dict(), "ppi": ppi_pathway.state_dict(),
                            "config": vars(args), "gene_names": gene_names, "vocab": vocab},
                           os.path.join(args.outdir, "best_model.pt"))
        with open(os.path.join(args.outdir, "training_history.json"), "w") as f: json.dump(history, f, indent=2)
        ckpt_path = os.path.join(args.outdir, "best_model.pt")
        if os.path.exists(ckpt_path):
            # Drop safe_globals (private numpy API broke on numpy >= 2.0).
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)
            try: tf_pathway.load_state_dict(ckpt["tf"], strict=False)
            except Exception: pass
            try: ppi_pathway.load_state_dict(ckpt["ppi"], strict=False)
            except Exception: pass
            print(f"[INFO] Loaded best model from {ckpt_path}")
        preds_test_path = os.path.join(args.outdir, "predictions_test_cells.csv")
        test_metrics = evaluate_cells(model, tf_pathway, ppi_pathway, dl_test, device, gene_names, save_preds_path=preds_test_path)
        print(f"[TEST] (cells) MAE={test_metrics['mae']:.4f} | PDS={test_metrics['pds']:.4f} | DES≈{test_metrics['des_proxy']:.4f}")
        with open(os.path.join(args.outdir, "metrics_test_cells.json"), "w") as f: json.dump(test_metrics, f, indent=2)
        print(f"[DONE] Wrote per-cell predictions to: {preds_test_path}")
        return

# ----------------------------- Per-cell x0 helpers ------------------------- #

def compute_control_lookup(
    adata: "ad.AnnData", target_col: str, control_label: str,
    groupby: Optional[List[str]] = None, min_ctrl: int = 1,
) -> Dict[Tuple, np.ndarray]:
    labels = adata.obs[target_col].astype(str)
    mask_ctrl = (labels == control_label).values
    if not mask_ctrl.any():
        raise RuntimeError(f"No control cells found for control_label='{control_label}'.")
    X = adata.X
    if not groupby:
        x0 = X[mask_ctrl].mean(axis=0).A1.astype(np.float32) if issparse(X) \
             else X[mask_ctrl].mean(axis=0).astype(np.float32).ravel()
        return {(): x0}
    ctrl_obs = adata.obs.loc[mask_ctrl, groupby]
    keys, inv = np.unique(ctrl_obs.astype(str).agg("||".join, axis=1).values, return_inverse=True)
    lookup = {}
    for i, k in enumerate(keys):
        idx = (inv == i)
        rows = np.where(mask_ctrl)[0][idx]
        if len(rows) < min_ctrl: continue
        mean_vec = X[rows].mean(axis=0).A1.astype(np.float32) if issparse(X) \
                   else X[rows].mean(axis=0).astype(np.float32).ravel()
        lookup[tuple(k.split("||"))] = mean_vec
    return lookup

def per_cell_baseline(
    adata: "ad.AnnData", groupby: Optional[List[str]],
    ctrl_lookup: Dict[Tuple, np.ndarray], fallback_global: Optional[np.ndarray] = None,
) -> np.ndarray:
    N, G = adata.shape
    out = np.zeros((N, G), dtype=np.float32)
    if not groupby:
        key = ()
        base = ctrl_lookup.get(key, fallback_global)
        if base is None:
            raise RuntimeError("Global baseline missing.")
        out[:] = base; return out
    obs_grp = adata.obs[groupby].astype(str).agg("||".join, axis=1).values
    for i, k in enumerate(obs_grp):
        tup = tuple(k.split("||"))
        if tup in ctrl_lookup: out[i] = ctrl_lookup[tup]
        elif () in ctrl_lookup: out[i] = ctrl_lookup[()]
        elif fallback_global is not None: out[i] = fallback_global
        else: raise RuntimeError(f"Missing control baseline for group={tup}.")
    return out

# ------------------------------ CLI ---------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HybridMemoryRAMLite (+ optional CVAE) + ESM2/scGPT + LoRA + GO/evidence.")
    parser.add_argument("--h5ad", type=str, required=True, help="Path to .h5ad file.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for models/predictions.")
    parser.add_argument("--target_col", type=str, default="target_gene", help="obs column with perturbation labels.")
    parser.add_argument("--control_label", type=str, default="non-targeting", help="Label for control cells.")
    parser.add_argument("--min_cells_per_pert", type=int, default=50, help="Minimum cells per perturbation (pseudobulk).")
    parser.add_argument("--skip_norm", action="store_true", help="Skip library-size normalization + log1p.")
    # Splits
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute device (auto: cuda -> mps -> cpu).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_pds", type=float, default=0.5)
    parser.add_argument("--lambda_des", type=float, default=0.5)
    parser.add_argument("--tau_de", type=float, default=0.25)
    # RAM hyperparams
    parser.add_argument("--feat_dim", type=int, default=32)
    parser.add_argument("--fast_dim", type=int, default=64)
    parser.add_argument("--slow_dim", type=int, default=128)
    parser.add_argument("--memory_bank", type=int, default=6)
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--min_rounds", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--step_init", type=float, default=0.25)
    parser.add_argument("--sat_reg", type=float, default=1e-2)
    parser.add_argument("--sat_rank", type=int, default=128)
    parser.add_argument("--slow_rank", type=int, default=1024)
    parser.add_argument("--memory_influence", type=float, default=0.2)
    # Pathway dims
    parser.add_argument("--tf_hidden", type=int, default=256)
    parser.add_argument("--ppi_rank", type=int, default=256)
    # Single-cell specifics
    parser.add_argument("--single_cell", action="store_true", help="Train/evaluate at single-cell level.")
    parser.add_argument("--x0_mode", type=str, default="per_group", choices=["global","per_group"], help="Baseline strategy.")
    parser.add_argument("--x0_groupby", type=str, default="batch,cell_type", help="Grouping columns when x0_mode=per_group.")
    parser.add_argument("--heldout_cts", type=str, default=None, help="Comma-separated held-out cell types for TEST only.")
    parser.add_argument("--ct_col", type=str, default=None, help="Column name for cell type (default: 'cell_type').")
    # Warmstart
    parser.add_argument("--warmstart", type=str, default=None, help="Checkpoint for warm start initialization")
    parser.add_argument("--freeze_pathways", action="store_true", help="Freeze TF/PPI at warmstart")
    # CVAE controls
    parser.add_argument("--use_cvae", action="store_true", help="Enable CVAE latent head")
    parser.add_argument("--z_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--beta_kl", type=float, default=0.5, help="KL weight (max)")
    parser.add_argument("--kl_warmup_epochs", type=int, default=10, help="Linear warmup to beta_kl")
    parser.add_argument("--best_metric", type=str, default="pds", choices=["pds", "mae"],
                        help="Which validation metric drives best-model checkpoint selection. "
                             "Default `pds` aligns with the autoresearch gating metric. "
                             "Use `mae` to reproduce the older MAE-based selection.")
    # Inference: val_counts synthesis
    parser.add_argument("--infer_valcounts", action="store_true", help="Run unseen synthesis from --val_counts_csv to --out_h5ad")
    parser.add_argument("--val_counts_csv", type=str, default="", help="CSV with columns [target_gene, n_cells]")
    parser.add_argument("--out_h5ad", type=str, default="synth_unseen.h5ad", help="Output H5AD path for synthesized cells")
    parser.add_argument("--ctrl_pool_cap", type=int, default=100000, help="Cap for control pool size")
    parser.add_argument("--batch_size_pred", type=int, default=512, help="Batch size for generation")
    # External embeddings & priors
    parser.add_argument("--scgpt_embeddings_tsv", type=str, default=None, help="TSV: first col symbol, rest vector dims")
    parser.add_argument("--esm2_embeddings_csv", type=str, default=None, help="Tab-separated CSV/TSV: first col symbol, rest dims")
    parser.add_argument("--esm2_sep", type=str, default="\t", help="Separator for ESM2 CSV/TSV (default: tab)")
    parser.add_argument("--tf_list", type=str, default=None, help="Text file with TF symbols (one per line)")
    parser.add_argument("--go_csv", type=str, default=None, help="GO CSV with columns [target, tag]")
    parser.add_argument("--freeze_proj", action="store_true", help="Freeze projection from external LMs to pathway dims")
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (0 disables)")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA scaling factor")
    parser.add_argument("--pert_dropout", type=float, default=0.0,
                        help="Prob of zeroing learned pert_feat during training (recommended 0.25)")
    parser.add_argument("--zeroshot_eval", action="store_true",
                        help="Hold out perturbations with external embeddings for zero-shot test")
    parser.add_argument("--zeroshot_frac", type=float, default=0.15,
                        help="Fraction of perturbations for zero-shot test set")
    # Unseen vocab
    parser.add_argument("--no_dynamic_unseen", action="store_true", help="Disable dynamic unseen vocab expansion at inference")

    args = parser.parse_args()
    main(args)
