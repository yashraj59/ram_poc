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

Pre-specified families. The agent may not run an experiment outside these families without first writing an amendment that adds a new family.

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

When triggered, search at least three of the canonical surfaces for the relevant domain (see `references/core_protocol.md §13`). For this VCC POC the priorities are: arXiv (q-bio.QM, cs.LG), bioRxiv (perturbation prediction), OpenReview (NeurIPS / ICLR), Connected Papers (from the seed papers below), and Semantic Scholar. Declare the fetch fingerprint at run start and keep it consistent.

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

Run in **autonomous Debate Council mode** (per the user's instruction to require no human intervention). Configuration:

- **Council models**: prefer multi-vendor if multiple API keys are available. Default config to try first:
  ```yaml
  council_models:
    architect:     anthropic/claude-opus-4.7
    skeptic:       openai/gpt-5.5-pro
    methodologist: google/gemini-pro
    biologist:     anthropic/claude-opus-4.7
    monitor:       openai/gpt-5.5-pro
  ```
  Fall back to `same_model_all_roles` if keys are missing; log `COUNCIL_MULTI_VENDOR_FALLBACK_USED`.
- **Self-critique**: enabled per `references/debate_council.md` Council Process step 3.
- **Escalation**: the council may not make biology-interpretation decisions autonomously. If a Tier 3 candidate is about to promote but the biologist role flags a marker-overlap regression or a pathway-coherence concern, the council must close with `COUNCIL_BIOLOGY_ESCALATION_REQUIRED` and write the closure report.
- **Hard escalation triggers**: any of the conditions in `references/debate_council.md "Hard Escalation Triggers"` halt autonomous mode.

---

## Stop Conditions

The loop halts and writes `final_report.md` when any of these fire:

1. **PDS-0.80 reached on locked_test**. A Tier 3 candidate scores cell-eval PDS ≥ 0.80 on the official VCC validation deliverable. Promote, predict the official VCC test set, write closure.
2. **Experiment cap**. 40 experiments without reaching PDS-0.80. Close with the strongest-PDS candidate as the model of record and report the gap to 0.80.
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

## Closure

When you reach the stop condition (PDS-0.80 or the experiment cap), do this:

1. Predict on the official VCC final test set using the model of record's best checkpoint.
2. Save predictions to `outputs/final_test_predictions.h5ad`.
3. Write `final_report.md` covering closure trigger, model of record, every architectural family's outcome, the strongest-PDS candidate's cell-eval score, the MCC floor status, retained vs deleted artifacts, and explicit no-claims wording per `references/biology_addendum.md` ("Improvement over baseline on the validation deliverable; external cohort confirmation pending").
4. Write the `STATE_OF_PLAY.md` to reflect closure.
5. Stop. Do not start a new experiment.

---

## Begin only after Step 0 baselines are complete.
