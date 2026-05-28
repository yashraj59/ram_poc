# VCC POC — Autoresearch on Perturbation Prediction

This is the proof-of-concept for running [autoresearch-bio](https://github.com/yashraj59/autoresearch-bio) on the Arc Institute Virtual Cell Challenge (VCC). The goal is to push cell-eval PDS to **0.80** on the official VCC validation deliverable using a fully-autonomous research loop, then predict the official test set.

## What's in this repo

- `model/model.py` — the perturbation prediction model. A dual-pathway (TF + PPI) adaptive-rounds architecture with a fast + slow memory hierarchy. Bug-fixed copy of the original `vae_esm_og.py` (see "Fixes applied" below).
- `autoresearch.md` — the autoresearch prompt for the agent. Defines the four identity locks (TF concept, PPI concept, adaptive rounds, fast/slow memory), the architectural families, the tiered gates, the stop condition (PDS ≥ 0.80), and the autonomous Debate Council configuration.
- `scripts/` — helper scripts the agent uses to fetch data, run cell-eval, and generate closure plots.
  - `download_vcc_data.sh` — pulls the GCS bucket into `data/vcc/`.
  - `run_cell_eval.sh` — wraps the canonical `cell-eval run` invocation.
  - `generate_closure_plots.py` — template for the eight required closure plots (PDS trajectory, status donut, family bars, MCC floor, lineage backbone, per-seed variance, local-vs-cell-eval calibration, three-acts comparison). Ports the MoFNet PoC plot aesthetic. The agent adapts the `# TODO(agent)` stubs to read the actual run's data.
- `data/` — gitignored. Where the agent stages VCC training data, external embeddings, etc.
- `outputs/` — gitignored. Where the agent writes experiment artifacts (`results.tsv`, journals, checkpoints, etc.).
- `autoresearch/` — directory the agent fills with its working files during the loop.

## What this PoC tests

Two questions:

1. Does autoresearch-bio's discipline layer reliably steer a coding agent toward higher PDS without leaking on the validation deliverable?
2. Does the dual-pathway + adaptive-rounds + memory architecture have headroom above the user's reported baseline?

Stop condition: cell-eval PDS ≥ 0.80 on the official VCC validation deliverable. Once reached, predict the test set, stop, and write `final_report.md`.

## Data

The official Arc Institute VCC data lives at `gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/`. The agent fetches it into `data/vcc/`. The agent also fetches:

- scGPT gene embeddings (HuggingFace or upstream release)
- ESM2 protein embeddings (HuggingFace `facebook/esm2_*`)
- TF list (Lambert et al. 2018 supplementary or equivalent)
- GO annotations (GO Consortium downloads)

Plus any additional cell-type or perturbation data the agent decides will help. The user has explicitly given the agent total data freedom.

## How to launch

```bash
# 1. Install dependencies (in a fresh venv).
pip install -r requirements.txt

# 2. Authenticate to GCS (one-time).
gcloud auth application-default login

# 3. Install cell-eval. The exact install path depends on Arc Institute's release;
# expect a pip install or a clone-and-install from their GitHub.
pip install cell-eval  # or follow the official instructions

# 4. Point your agent at autoresearch.md and let it run.
# The agent reads autoresearch.md, fetches data, builds the four-role split,
# runs EXP000 (the baseline), then iterates families until PDS >= 0.80 or
# the experiment cap is hit.
```

The launch message (chat text the user sends to the agent, separate from `autoresearch.md`) is provided in the chat where this repo was created. It tells the agent to read `autoresearch.md`, fetch the VCC data, and begin Step 0.

## cell-eval invocation

The user's canonical invocation:

```bash
cell-eval run \
  -ap <path_to_predicted>.h5ad \
  -ar <path_to_real>.h5ad \
  --num-threads 64 \
  --profile vcc
```

`scripts/run_cell_eval.sh` wraps this with the correct paths for the loop's standard predictions.

## Fixes applied to the original model

The model file was copied from the user's `vae_esm_og.py` and the following fixes were applied:

1. **`pds_from_l1` memory blow-up**: rewrote to chunk the row dimension so peak allocation is O(row_chunk × N × G) instead of O(N² × G). At VCC scale this avoids OOM during local PDS evaluation.
2. **`safe_globals` numpy private-API breakage**: removed the `safe_globals` context at all four checkpoint-load sites. `np.core.multiarray._reconstruct` is a private API that moved/disappeared on numpy >= 2.0 and was causing checkpoint loads to fail.
3. **`ReduceLROnPlateau(verbose=True)`**: removed the deprecated `verbose=` argument.
4. **Best-model save criterion**: added a `--best_metric {pds,mae}` CLI flag (default `pds`) and updated the single-cell and pseudobulk training loops to use it. Previously the best model was selected by MAE, which does not align with the autoresearch gating metric.
5. **`beta_kl` warmup off-by-one**: fixed so beta is 0 at epoch 1 and reaches `beta_kl` at `kl_warmup_epochs + 1`. Previously beta started at `1/warmup_epochs`.
6. **Dead-code cleanup**: removed three commented-out `set_evidence_vec` definitions, two commented-out helper functions, and one walrus-operator dead variable assignment. Saved ~22 lines.

Original baseline behavior is preserved when the agent passes the same CLI arguments the user originally used.

## Autonomous family amendments

This run softens the default autoresearch-bio amendment rule for one specific case: the autonomous Debate Council is allowed to add new architectural families mid-loop, up to a hard cap of 4 autonomous additions across 200 experiments. Six conditions must hold (literature pass, identity preservation, Skeptic counter-argument, self-critique with concrete weakness, full family documentation, stricter Tier 1 threshold of +0.025 to offset single-vendor council correlation). See `autoresearch.md` §Architectural Families → "Procedure for autonomous family addition" for the full rule.

This softening is scoped to family *additions* only. It does not extend to stop-trigger overrides, threshold relaxation, experiment-cap extension, identity-lock violations, or council-mode changes. Those still require a human turn per autoresearch-bio §14.

## Identity locks

These are the four concepts the autoresearch agent must preserve. Implementation can change, the concept cannot:

1. **TF pathway** — a route that captures transcription-factor-driven effects.
2. **PPI pathway** — a route that captures protein-protein-interaction effects.
3. **Adaptive rounds** — variable per-perturbation compute.
4. **Fast + slow memory hierarchy** — per-step + episodic memory with retrieval.

See `autoresearch.md` for the full Keep / Can Modify / Cannot Modify breakdown.

## License

MIT (same as the parent project). Use, modify, redistribute freely.
