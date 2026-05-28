# Autoresearch: VCC Perturbation Prediction with Adaptive Memory Model

This is the autoresearch prompt for the Virtual Cell Challenge proof-of-concept. The loop is fully autonomous. Do not stop to ask the user for confirmation. The only condition under which you may halt and wait is if the official VCC data files cannot be located or downloaded.

---

## Mission

Train a perturbation-response model on the official Arc Institute Virtual Cell Challenge data and push the cell-eval PDS score on the official validation deliverable to **0.80 or higher**. Once you reach that threshold, stop the search, predict the official test set, and write the final report. If you exhaust the experiment budget without reaching 0.80, close the loop honestly with the strongest-PDS candidate as the model of record and a `SEARCH_CLOSED_NO_NEW_BASELINE_MCC_FLOOR` style closure entry (see `references/statistical_promotion.md`).

---

## Model and System of Record

- **Model code**: `model/model.py` in this repository. This is a cleaned and bug-fixed copy of the user's Virtual Cell Challenge model.
- **Step 0 baseline**: the user's reported baseline path is pseudobulk pretrain → single-cell warmstart, with `--x0_groupby batch,cell_type`. Reproduce this as the first experiment (EXP000) and record the resulting cell-eval PDS as the baseline number.
- **Active model checkpoint**: written to `outputs/<EXP_ID>/best_model.pt` by the training driver. The skill protects this; only an explicit Tier 3 promotion can rebase it.
- **Best-metric selection**: the model's training loop already supports `--best_metric pds`. Keep it as `pds` throughout the loop; this aligns local early-stop with the cell-eval gating metric.

---

## Data

The official VCC data lives at:

```
gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/
```

It contains three components:

1. **Training data** (cells with known perturbations) — used to fit model parameters and as the source for the internal train / internal_val split.
2. **Validation deliverable** (the official Arc validation set, comprising a list of unseen target perturbations with target cell counts, plus the ground-truth h5ad file the cell-eval tool needs as `-ar`) — this is the autoresearch loop's `locked_test`. It is read exactly once per candidate, via cell-eval. Reading it more than once per candidate is `FAIL_TEST_IN_SELECTION` under §3.5 of the skill.
3. **Final test set** — never read until the PDS-0.80 stop trigger fires and the candidate is being promoted.

You have **total freedom** over additional data. If you decide that extra cell-type data, additional perturbation atlases, or supplementary single-cell datasets will help, fetch them yourself from public sources (CELLxGENE, GEO, Allen Brain Atlas, Human Cell Atlas, etc.) and document each one in `external_resources.md` per the rules in `references/biology_addendum.md`.

### Four-role split manifest (required before any training)

Before EXP000 launches, build `outputs/split_manifest.json` per `assets/split_manifest.schema.json` with the following roles:

- **`train`**: cells from the official VCC training file used to fit model parameters.
- **`validation`**: held-out cells from the official VCC training file used for internal early-stop and candidate ranking. Split by `(batch, cell_type)` grouping if practical, otherwise by random sampling stratified by perturbation. Document the split rule in `leakage_preflight.md`.
- **`locked_test`**: the official VCC validation deliverable. The cell-eval invocation reads this. Each candidate reads it once for confirmation.
- **`legacy_test`**: empty for this POC (first run of this loop).

Compute the SHA-256 of the manifest at write time and pin it in `external_resources.md`.

### Leakage pre-flight (§3.5)

Before EXP000 launches, write `outputs/leakage_preflight.md` that enumerates every code path in `model/model.py` that reads each split role. Specifically:

- Confirm that no path in the training loop, status-assignment logic, anchor selection, or warm-start initialization reads from `locked_test`.
- Confirm that the official VCC test set is not loaded anywhere in the training driver.
- Confirm that no frozen on-disk artifacts derived from `locked_test` exist in this repository (it is a fresh setup, so this should be trivial — but record the check explicitly).
- Set `leakage_guard: PASS_NO_TEST_SELECTION` as the default for every `results.tsv` row that runs the standard driver.

---

## Identity: Keep / Can Modify / Cannot Modify

### Keep (the four identity locks)

These are concepts. The user has been explicit that the *concept* must remain even if the *implementation* changes.

1. **TF pathway concept**: a dedicated route that captures transcription-factor-driven perturbation effects. The current implementation uses a gene-expression LM (scGPT) with a frozen embedding table and a trainable projection plus LoRA. You may swap the LM (Geneformer, scFoundation, Evo, learned embedding from scratch), replace it with a knowledge graph (ChIP-seq targets, DoRothEA, ENCODE TF perturbation graphs), or redesign the projection. You may **not** delete the TF pathway, merge it into the PPI pathway, or remove its dedicated decoder.

2. **PPI pathway concept**: a dedicated route that captures protein-protein-interaction effects. Currently uses ESM2 with the same shape as the TF pathway. You may swap to a different protein LM (ProtTrans, ESM3 if available, OpenFold embeddings) or replace with a curated PPI graph (STRING, BioGRID, OmniPath, IntAct). You may **not** delete the PPI pathway.

3. **Adaptive rounds concept**: variable per-perturbation compute. The current implementation is ACT-style (halt head, weighted accumulation across rounds, epsilon stop). You may replace with PonderNet, mixture-of-depths, learned early exit, or any other adaptive-computation mechanism. You may **not** collapse the model into a single forward pass.

4. **Fast + slow memory hierarchy concept**: two memory tiers, one per-step / fast (currently GRU) and one episodic / slow (currently attention bank with episode encoder). You may swap the recurrent cell, the retrieval mechanism, or the episode encoder. You may **not** remove the two-tier structure.

### Cannot Modify (locked files)

- `model/model.py` lines defining the *existence* of `TFPathway*`, `PPIPathway*`, `HybridMemoryRAMLite`'s forward loop with adaptive rounds, and `HybridMemorySystem`'s fast/slow tiers. You may add new classes / replace internal logic, but the four identity concepts must remain detectable by code review.

### Can Modify (everything else)

- The specific LM tables and prior sources (scGPT, ESM2, GO CSV, TF list). The user has handed you full freedom to choose. The Step 0 baseline starts with the legacy choice (scGPT + ESM2 + GO + TF list, fetched as part of the agent's setup), but every subsequent experiment can swap any of these.
- LoRA rank, alpha, on/off per pathway.
- CVAE on/off, `z_dim`, KL warmup, KL weight.
- Saturation layer (replace, remove, redesign).
- Evidence-vector composition and the mixer.
- Heteroscedastic Gaussian emission (count-distribution-correct alternatives like Negative Binomial, ZINB, or Poisson are strongly preferred for raw counts; if you log1p-transform first, Gaussian is acceptable but suboptimal).
- Loss composition and weights (NLL, InfoNCE for PDS surrogate, DES BCE, KL, saturation L2).
- Training schedule: learning rate, batch size, weight decay, scheduler, gradient clipping, mixed precision.
- Training mode: pseudobulk-only, single-cell-only, or two-stage (pseudobulk pretrain → single-cell warmstart). The baseline uses the two-stage path; alternative training modes are separate architectural families.
- Heldout cell types (`--heldout_cts`), heldout perturbations (`--zeroshot_frac`), per-cell baseline strategy (`--x0_mode`, `--x0_groupby`).
- `min_cells_per_pert`, `feat_dim`, `fast_dim`, `slow_dim`, `memory_bank`, `max_rounds`, `min_rounds`, `epsilon`, `tf_hidden`, `ppi_rank`, `sat_rank`, `slow_rank`, `lora_rank`, `lora_alpha`, `pert_dropout`, and all other CLI hyperparameters.

### Identity violations

Any proposed experiment that violates the four identity locks above must be documented in `outputs/identity_violations_considered.md` (under the autoresearch-bio escalation rule) and **not** run. Examples to log there explicitly when you consider them: collapsing TF and PPI into a single pathway, removing the adaptive-rounds loop, replacing fast+slow memory with a single feedforward layer.

---

## Metrics

### Primary (gating)

- **cell-eval PDS** on the official VCC validation deliverable (the `locked_test` role). This is the only metric that gates the stop condition. Compute it by:

  1. Running the model in `--infer_valcounts` mode to predict an h5ad for the validation deliverable.
  2. Invoking `cell-eval` exactly as the user does:
     ```
     cell-eval run -ap <pred>.h5ad -ar <real>.h5ad --num-threads 64 --profile vcc
     ```
  3. Parsing the resulting PDS from the cell-eval output.
  4. Recording this value in the `cell_eval_pds` column of `results.tsv`.

  Each candidate calls cell-eval **once**. Calling it more than once on the same candidate against the same `locked_test` is a `FAIL_TEST_IN_SELECTION` violation.

### Secondary (for early-stop and ranking)

- **Local PDS** computed by `pds_from_l1` in `model/model.py`. This is a chunked O(N·G) approximation of the cell-eval metric. Use it for in-loop early-stop and for inter-candidate ranking before you spend a cell-eval call.
- **MAE** on internal validation (held-out cells from the training file).
- **DES proxy** (top-K overlap of |delta|).

### Protected (must not regress more than the gate value)

- **Local PDS on internal validation**: must not drop by more than 0.02 from the Step 0 baseline. A candidate that improves cell-eval PDS but tanks internal validation PDS is suspect.
- **Train + validation MAE ratio**: catastrophic blow-ups indicate training instability or wrong emission distribution.
- **NaN / Inf in any of `tf_eff`, `ppi_eff`, `combined_effect`, `y_mix`**: any candidate that produces non-finite values during training or inference is automatically `TIER1_DISCARD_NUMERICAL_INSTABILITY`.

### Catastrophic-fail (auto-disqualify)

- Training crash (OOM, CUDA error, NaN loss) — `TIER1_DISCARD_TRAINING_CRASH`.
- cell-eval crash on a candidate's prediction file — `TIER1_DISCARD_CELL_EVAL_CRASH`.
- Identity violation discovered post-hoc (e.g., the candidate's code path removed the PPI pathway through some refactor) — escalate via `identity_violations_considered.md`.

---

## Step 0 Baseline Plan

Before any architectural search, run EXP000 as the Step 0 baseline and record everything in `outputs/step0_baselines/` and `outputs/BASELINE_REGISTRY.md`.

1. Fetch the VCC data from the GCS bucket. Save to `data/vcc/`.
2. Fetch the legacy external resources the user reported using:
   - **scGPT embeddings**: from the scGPT release on HuggingFace or the scGPT GitHub.
   - **ESM2 embeddings**: gene-symbol → ESM2 vector. Generate from the ESM2 model weights if a pre-computed table is not directly available.
   - **TF list**: a canonical transcription-factor list. Lambert et al. 2018 "The Human Transcription Factors" supplementary is a common starting point.
   - **GO annotations**: gene-symbol → GO tag CSV. Generate from `go-basic.obo` and a Human GAF file from the GO Consortium downloads.
3. Document every fetched resource in `outputs/external_resources.md` with provenance (URL, commit/version hash, download date, license).
4. Build the four-role split manifest and write `leakage_preflight.md`.
5. Run the pseudobulk pretrain:
   ```
   python model/model.py --h5ad data/vcc/training.h5ad --outdir outputs/EXP000_PB \
     --target_col target_gene --control_label non-targeting \
     --epochs 50 --batch_size 32 --best_metric pds \
     --scgpt_embeddings_tsv data/external/scgpt.tsv \
     --esm2_embeddings_csv data/external/esm2.tsv --esm2_sep '\t' \
     --tf_list data/external/tf_list.txt --go_csv data/external/go.csv \
     --use_cvae --z_dim 32 --beta_kl 0.5 --kl_warmup_epochs 10
   ```
6. Run the single-cell warmstart:
   ```
   python model/model.py --h5ad data/vcc/training.h5ad --outdir outputs/EXP000_SC \
     --single_cell --x0_mode per_group --x0_groupby batch,cell_type \
     --warmstart outputs/EXP000_PB/best_model.pt \
     --target_col target_gene --control_label non-targeting \
     --epochs 30 --batch_size 64 --best_metric pds \
     --scgpt_embeddings_tsv data/external/scgpt.tsv \
     --esm2_embeddings_csv data/external/esm2.tsv --esm2_sep '\t' \
     --tf_list data/external/tf_list.txt --go_csv data/external/go.csv \
     --use_cvae --z_dim 32 --beta_kl 0.5 --kl_warmup_epochs 10
   ```
7. Run inference on the validation deliverable:
   ```
   python model/model.py --infer_valcounts \
     --warmstart outputs/EXP000_SC/best_model.pt \
     --h5ad data/vcc/training.h5ad \
     --val_counts_csv data/vcc/validation_counts.csv \
     --out_h5ad outputs/EXP000_SC/pred_validation.h5ad \
     --scgpt_embeddings_tsv data/external/scgpt.tsv \
     --esm2_embeddings_csv data/external/esm2.tsv --esm2_sep '\t' \
     --tf_list data/external/tf_list.txt --go_csv data/external/go.csv \
     --outdir outputs/EXP000_SC
   ```
8. Run cell-eval:
   ```
   cell-eval run -ap outputs/EXP000_SC/pred_validation.h5ad \
     -ar data/vcc/validation_real.h5ad \
     --num-threads 64 --profile vcc
   ```
9. Record the resulting PDS in `outputs/results.tsv` as EXP000's `cell_eval_pds` column.

---

## Architectural Families

Six pre-specified families are listed below. The autonomous Debate Council **may amend new families into the search space mid-loop** if the literature pass or the run's own findings surface a credible additional mechanism class. This is a deliberate softening of the default autoresearch-bio amendment rule for this run, specifically scoped to *family additions* during exploration. It does NOT extend to stop-trigger amendments — those still require a supervised human turn or a separate-process council per `references/core_protocol.md §14`.

### Procedure for autonomous family addition

The council may add a new family only when **all six** of the following hold:

1. A literature pass has run within the last 20 experiments and `papers_consulted.md` contains at least one entry that motivates the proposed mechanism class (a concrete published technique, not a model intuition).
2. The proposed family preserves all four identity locks. A family that would delete or merge the TF / PPI / adaptive-rounds / fast+slow-memory concepts is rejected and the proposal goes into `identity_violations_considered.md`.
3. The Skeptic role argues against the proposal explicitly and the council records the steelmanned counter-argument.
4. The proposal passes the self-critique step (the proposer fills `self_identified_weakness` with a concrete weakness, not boilerplate).
5. The new family's documentation matches the existing six families' format: Motivation (specific failure mode it addresses) / Hypothesis / Suggested experiments / Constraints / Stop-pivot rule.
6. The single-vendor council caveat is noted in the amendment block — the council is one model agreeing with itself in five voices, so the new family must clear a stricter Tier 1 threshold of **+0.025 local PDS over Step 0** (versus the +0.02 the pre-specified families need) to compensate for the higher false-positive risk.

### Logging requirements

Every autonomous family amendment must produce:

- A new entry in `family_allocation.md` with the Family number incremented (`Family 6:`, `Family 7:`, …) and the six required documentation fields.
- A journal entry in `research_journal.md` tagged `AUTONOMOUS_FAMILY_AMENDMENT`.
- A row in `papers_consulted.md` for the motivating paper (must already exist per condition 1, but re-cite explicitly).
- A `self_identified_weakness` line in the amendment block.
- A counter in `STATE_OF_PLAY.md`: `autonomous_families_added: <N>`.

### Hard cap on autonomous additions

No more than **4** autonomous family additions across the 200-experiment cap. If the council proposes a fifth, halt with `AUTONOMOUS_FAMILY_LIMIT_REACHED` and write a closure note explaining that the loop tried to expand the search space beyond the discipline budget — a human turn is required to amend beyond this point.

### Hard ban (escalate to user, do not autonomously amend)

The council may NOT autonomously:

- Override or extend the experiment cap (200 is firm; halt and document).
- Lower any Tier 1 / Tier 2 / Tier 3 threshold.
- Relax the multiple-comparison floor.
- Skip the literature pass requirement.
- Promote a Tier 3 candidate without a passed cell-eval against the locked test.
- Add a family that violates an identity lock.
- Override the `same_model_all_roles` configuration to multi-vendor (the user pinned single-vendor explicitly).

These are §14 stop-trigger-level changes and need a human turn.

### The six pre-specified families

### Family 0: Training schedule and loss weights

- **Motivation**: even the baseline architecture may benefit from better lr scheduling, gradient accumulation, longer training, or rebalanced loss weights.
- **Hypothesis**: the baseline lr / lambda_pds / lambda_des / beta_kl / weight_decay are not optimal for this dataset.
- **Suggested experiments**: lr warmup + cosine schedule, gradient accumulation for effective batch size 256, longer pseudobulk training, swept lambda_pds and lambda_des, swept beta_kl, AdamW vs Lion.
- **Constraints**: identity is untouched. Tag as `grid_sweep` branch type when running a sweep; only the best child by validation PDS advances to Tier 2.
- **Stop rule**: retire if three consecutive Tier 1 candidates fail to clear the multiple-comparison floor.

### Family 1: TF prior swap or replace

- **Motivation**: the current scGPT prior may not be the best available. Newer gene-expression LMs (Geneformer, scFoundation, UCE, Evo) or curated knowledge graphs (ChIP-seq targets, DoRothEA, ENCODE) may give a stronger TF signal.
- **Hypothesis**: replacing or augmenting the scGPT table improves cell-eval PDS by improving how the TF pathway represents perturbation effects.
- **Suggested experiments**: (a) swap scGPT for Geneformer embeddings; (b) swap for scFoundation; (c) augment scGPT with DoRothEA TF-target graph as a sparse adjacency mask; (d) replace LM-based prior with a learned-from-scratch TF embedding trained on the VCC training cells.
- **Constraints**: the TF pathway concept must remain. The decoder must still produce a `tf_eff` tensor.
- **Stop rule**: retire if no candidate clears Tier 1 after three attempts.

### Family 2: PPI prior swap or replace

- **Motivation**: same as Family 1 for the PPI side. ESM2 is the legacy choice; ESM3 (if accessible), curated PPI graphs (STRING, BioGRID, OmniPath), or learned protein embeddings may improve the PPI signal.
- **Hypothesis**: replacing or augmenting the ESM2 table improves cell-eval PDS.
- **Suggested experiments**: (a) swap ESM2 for a STRING-based sparse graph; (b) swap for OmniPath; (c) ProtTrans embeddings; (d) augment ESM2 with PPI graph as adjacency.
- **Constraints**: PPI pathway concept must remain.
- **Stop rule**: retire if no candidate clears Tier 1 after three attempts.

### Family 3: Emission distribution

- **Motivation**: gene counts are not Gaussian. The current heteroscedastic Gaussian emission may be a real source of model misspecification.
- **Hypothesis**: switching to a count-distribution-correct emission (Negative Binomial, ZINB, Poisson) on raw counts improves cell-eval PDS.
- **Suggested experiments**: (a) NB with learned dispersion per gene; (b) ZINB with separate dropout head; (c) Poisson + size-factor scaling; (d) keep Gaussian but switch to MSE on raw counts after sqrt-transform.
- **Constraints**: emission head must produce a per-cell per-gene prediction that cell-eval can score.
- **Stop rule**: retire if catastrophic numerical instability or training crashes on more than half the runs.

### Family 4: Adaptive-rounds replacement

- **Motivation**: ACT can be unstable in practice. PonderNet is more rigorous theoretically; mixture-of-depths or learned early exit are alternatives.
- **Hypothesis**: replacing the ACT halt head with a more rigorous adaptive-computation mechanism improves training stability without losing the per-perturbation compute-variability property.
- **Suggested experiments**: (a) PonderNet with KL-to-geometric prior; (b) mixture-of-depths gating; (c) learned-threshold early exit; (d) keep ACT but add a halt-head regularizer (e.g. KL to uniform).
- **Constraints**: variable per-perturbation rounds must remain.
- **Stop rule**: retire if no mechanism improves over baseline ACT after three attempts.

### Family 5: Memory system rebuild

- **Motivation**: the GRU + attention-bank design is one valid choice. Transformer-based memory or key-value memory may be more expressive and easier to scale.
- **Hypothesis**: a transformer-encoder-based slow memory with cross-attention from the fast tier improves PDS, especially when combined with stronger TF/PPI priors.
- **Suggested experiments**: (a) replace the slow memory bank with a 2-layer transformer encoder; (b) key-value memory with explicit episode keys; (c) Linear-attention memory for scalability.
- **Constraints**: fast/slow two-tier structure must remain.
- **Stop rule**: retire if no rebuild beats the baseline GRU + attention bank after two attempts.

---

## Tiered Gates

### Tier 1 (single-seed signal filter)

A single-seed training run with the baseline schedule. Promoted to Tier 2 if:

- Local validation PDS improves by at least 0.02 over Step 0 baseline.
- No protected-metric regression (see §Metrics).
- No NaN / Inf in any pathway output.
- The candidate's code path preserves the four identity locks.

Label as `TIER1_KEEP` if all four conditions hold, `TIER1_DISCARD_<reason>` otherwise. Reserved-string rule applies: no `BEAT`, `SOTA`, `WINS`, `ABOVE_REFERENCE`, etc.

### Tier 2 (multi-seed validation)

5 seeds (66, 1, 7, 17, 23). Promoted to Tier 3 if:

- Mean local validation PDS across 5 seeds improves by at least 0.015 over Step 0.
- Standard deviation across seeds is below 0.025.
- No protected-metric regression on any seed.

If Tier 2 fails, document the seed-spread in the journal and discard.

### Tier 3 (cell-eval PDS confirmation)

The Tier 2-passing candidate's best checkpoint runs inference on the official VCC validation deliverable, then cell-eval is invoked once. The result is recorded in `cell_eval_pds`. Promote the candidate to model of record only if the cell-eval PDS is the new repository-wide maximum.

If at any point a Tier 3 candidate reaches **cell_eval_pds ≥ 0.80**, the stop condition fires immediately. See §Stop conditions.

---

## Multiple-comparison floor

Per `references/statistical_promotion.md`, log the cumulative single-seed candidate count `N` and the floor `z_floor(N) = 2 + sqrt(log(N) / 2)` in `results.tsv` every 10 experiments. If at the experiment cap no candidate clears the floor on local validation PDS over baseline, close with `SEARCH_CLOSED_NO_NEW_BASELINE_MCC_FLOOR`.

---

## When-stuck triggers (literature pass)

A literature pass is required before the next mechanism in a family launches if:

- Three consecutive Tier 1 candidates in that family discard.
- A Tier 2 multi-seed validation fails with a protected-metric regression.
- A metric investigation rules out the family.
- 20 experiments have run since the last literature pass and no candidate has cleared the MCC floor.

When triggered, search at least three of the canonical surfaces for the relevant domain (see `references/core_protocol.md §13`). For this RAM PoC on the VCC task the priorities are: arXiv (q-bio.QM, cs.LG), bioRxiv (perturbation prediction), OpenReview (NeurIPS / ICLR), Connected Papers (from the seed papers below), and Semantic Scholar. Declare the fetch fingerprint at run start and keep it consistent.

Seed papers for the literature trail (already known to the field; the agent should look beyond these):

- The VCC challenge announcement and Arc Institute Virtual Cell Atlas paper(s).
- Geneformer (Theodoris et al. 2023), scGPT (Cui et al. 2024), scFoundation (Hao et al. 2023), UCE (Rosen et al. 2023), Evo (Nguyen et al. 2024).
- ESM2 (Lin et al. 2023), ESM3 (Hayes et al. 2024).
- PonderNet (Banino et al. 2021), ACT (Graves 2016), Mixture-of-Depths (Raposo et al. 2024).
- DoRothEA (Garcia-Alonso et al. 2019), STRING, OmniPath.
- ZINB-VAE (Lopez et al. 2018), scVI, CPA (Lotfollahi et al. 2023), GEARS (Roohani et al. 2024).

Update `papers_consulted.md` per the literature-discipline rule (one entry per family before that family produces its first Tier 1 keep).

---

## Council Mode (Autonomy)

Run in **autonomous Debate Council mode** with **single-vendor** configuration plus **self-critique** enabled (per the user's explicit instruction).

```yaml
council_models: same_model_all_roles
council_diversity: single_vendor    # logged in every closure report
council_self_critique: enabled       # see references/debate_council.md Council Process step 3
```

- **Single-vendor caveat**: the autoresearch-bio skill documents that single-vendor councils have correlated confidence because all five roles share the same underlying LLM. The closure report must include this caveat verbatim. Do not claim consensus as independent evidence.
- **Self-critique step (mandatory)**: before any proposal advances to the steelmanning round, the same agent that wrote the proposal must articulate the single strongest counter-argument to its own proposal and either revise or attach a `self_identified_weakness` field. Boilerplate weaknesses (anything that could be copy-pasted onto any proposal) are rejected with `COUNCIL_PROPOSAL_SELF_CRITIQUE_MISSING`.
- **Escalation**: the council may not make biology-interpretation decisions autonomously. If a Tier 3 candidate is about to promote but the biologist role flags a marker-overlap regression or a pathway-coherence concern, the council must close with `COUNCIL_BIOLOGY_ESCALATION_REQUIRED` and write the closure report.
- **Hard escalation triggers**: any of the conditions in `references/debate_council.md "Hard Escalation Triggers"` halt autonomous mode.

---

## Compute Budget

- **GPU**: single GPU with 96 GB VRAM available (e.g. H100 80GB + workstation card, or H200, or A100 80GB with NVLink — the agent does not need to assume a specific SKU). Train in mixed precision when stable.
- **Experiment cap**: **200 experiments total** across all families.
- **Per-experiment wall-clock guidance**: target under 4 hours per single-seed Tier 1 run with the baseline architecture. If a candidate's wall clock exceeds 8 hours, halt that run and label `TIER1_DISCARD_COMPUTE_TIMEOUT`.
- **cell-eval cost**: each Tier 3 evaluation against the locked_test costs one cell-eval invocation at 64 threads. Budget this into the 200-experiment cap.

## Stop Conditions

The loop halts and writes `final_report.md` when any of these fire:

1. **PDS-0.80 reached on locked_test**. A Tier 3 candidate scores cell-eval PDS ≥ 0.80 on the official VCC validation deliverable. Promote, predict the official VCC test set, write closure.
2. **Experiment cap**. **200 experiments** without reaching PDS-0.80. Close with the strongest-PDS candidate as the model of record and report the gap to 0.80.
3. **MCC floor not cleared**. At the experiment cap, no candidate clears the family-wise multiple-comparison floor over Step 0 baseline. Close with `SEARCH_CLOSED_NO_NEW_BASELINE_MCC_FLOOR`.
4. **Catastrophic failure**: 5+ consecutive `TIER1_DISCARD_TRAINING_CRASH` or `TIER1_DISCARD_CELL_EVAL_CRASH` outcomes. Halt and document the failure mode in `final_report.md`.
5. **Identity violation discovered post-hoc**: the council convenes, the violation goes into `identity_violations_considered.md`, and the loop halts pending amendment.
6. **Locked split spent without re-charter**: if the loop tries to call cell-eval more than once on the same candidate against `locked_test`, halt with `locked_split_spent_without_new_holdout_registered` (per §3.5 of the skill).

---

## Documentation Files

Maintain the following files in the run directory (`outputs/`):

- `results.tsv` with all required lineage columns (`commit`, `experiment_num`, `parent_experiment_ids`, `branch_type`, `subtree_status`, `family`, `tier_reached`, `status`, `primary_metric`, `secondary_metric`, `protected_metric_summary`, `architectural_change`, `description`, `leakage_guard`) plus the VCC-specific column `cell_eval_pds`.
- `research_journal.md` — narrative entry per experiment.
- `architectural_changes_log.md` — per-entry `parameter_delta`, `lines_touched`, `gradient_flow_smoke_passed`, `contribution_ratio_at_init`, `observed_effect_post_tier1`. Template-only entries fail the validator.
- `family_allocation.md` — per-family compute, Tier 1 keeps, Tier 2 passes, Tier 3 wins, status, sweep axes.
- `BASELINE_REGISTRY.md` — Step 0 baseline numbers with provenance.
- `external_resources.md` — every external resource fetched (scGPT, ESM2, GO, TF list, plus anything else the agent decides to pull). Per autoresearch-bio rules: version/commit, URL, license, organism/tissue/protocol, experiment IDs that used it.
- `identity_violations_considered.md` — every proposal that violates an identity lock, recorded and rejected.
- `papers_consulted.md` — literature record per `references/core_protocol.md §13`, with the per-paper template.
- `leakage_preflight.md` — required before EXP000.
- `split_manifest.json` — required before EXP000.
- `METRIC_IDENTITY_DIFF.md` — required if you cross a phase boundary that touches a gate-bearing column.
- `STATE_OF_PLAY.md` — next-action-only file regenerated after every experiment (≤2 KB).
- `insights/INSIGHT_BRIEF_NNN.md` — every 10 experiments.
- `final_report.md` — closure.

---

## Artifact Retention

- Retain forever: results.tsv, journals, registries, predictions of Tier 3 winners, the active model-of-record checkpoint, audit-relevant near-miss checkpoints, all external-resource provenance.
- Delete after Tier 1 discard: large checkpoints (`best_model.pt`) for discards that are not audit-relevant.
- Per-experiment `summary.json` must carry the identity block per `references/artifact_retention.md`.

---

## Closure Plots

Before writing `final_report.md`, generate the closure plots. Use `scripts/generate_closure_plots.py` as a starting template (it ships with the same aesthetic settings the MoFNet PoC used: white background, single-accent-per-plot palette, no em-dashes in captions, 85-character caption wrap). Adapt it to read this run's `results.tsv` and `cell_eval_pds` column.

Required plots (write to `outputs/closure_plots/`):

1. **`01_pds_trajectory.png`** — every experiment as a dot in chronological order, y-axis = cell-eval PDS (where available) and local validation PDS (where cell-eval has not been called). Color by status (discard / keep / Tier 2 / Tier 3 / baseline). Mark the 0.80 stop threshold as a horizontal line and the Step 0 baseline as another. Highlight the promoted candidate (or strongest-PDS candidate if no Tier 3 win).
2. **`02_status_donut.png`** — distribution of outcomes across all experiments. Center text shows the total experiment count. Small slices stack outside via leader lines.
3. **`03_family_bars.png`** — horizontal bars, experiments per family. Caption notes which family produced the model of record (or strongest candidate).
4. **`04_mcc_floor.png`** — the family-wise multiple-comparison floor curve `z_floor(N) = 2 + sqrt(log N / 2)` over the run's experiment count, with the strongest candidate's z-score plotted against it. Use the per-seed std from `BASELINE_REGISTRY.md`.
5. **`05_lineage_backbone.png`** — the promoted lineage in order, with PDS at each backbone node and a one-line annotation of what changed at each step. White stroke around all node labels so they read on every fill (per the MoFNet PoC pattern).
6. **`06_per_seed_variance.png`** — per-seed cell-eval PDS for every Tier 2 and Tier 3 backbone node. Each colored dot is one seed; horizontal tick is the 5-seed mean.
7. **`07_local_vs_celleval_calibration.png`** — scatter of local validation PDS (the model's internal approximation) against cell-eval PDS, one point per experiment that ran both. Identity line for reference. Caption notes how well the local approximation tracks the gating metric.
8. **`08_three_acts.png`** — bar chart of Step 0 baseline / strongest Tier 1 keep / model of record (or strongest Tier 3) / 0.80 stop threshold. Hatched if test-set tuned, solid if confirmation-only.

Aesthetic constraints (match the MoFNet PoC plots):

- White background, single accent color per plot (the existing palette in `scripts/generate_closure_plots.py` is fine: navy `#264653`, teal `#2A9D8F`, coral `#E76F51`, amber `#E9C46A`, lilac `#9B7EBD`, plus muted gray `#9a9a9a` and soft `#d8d8d8`).
- Sans-serif font, generous whitespace, top/right spines off.
- Captions wrap at 85 characters, sit below the plot, no em-dashes.
- All labels readable: dark ink text with a white stroke when text sits on top of a colored fill.

## Closure

When you reach the stop condition (PDS-0.80 or the experiment cap), do this:

1. Predict on the official VCC final test set using the model of record's best checkpoint.
2. Save predictions to `outputs/final_test_predictions.h5ad`.
3. Generate the closure plots per the section above.
4. Write `final_report.md` covering closure trigger, model of record, every architectural family's outcome, the strongest-PDS candidate's cell-eval score, the MCC floor status, retained vs deleted artifacts, the eight closure plots embedded by reference, the single-vendor council caveat, and explicit no-claims wording per `references/biology_addendum.md` ("Improvement over baseline on the validation deliverable; external cohort confirmation pending").
5. Write the `STATE_OF_PLAY.md` to reflect closure.
6. Stop. Do not start a new experiment.

---

## Begin only after Step 0 baselines are complete.
