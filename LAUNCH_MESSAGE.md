# Launch Message

Paste the block below as the first user message to your coding agent (Codex, Claude Code, Cursor, Aider, or similar). It is intentionally kept outside `autoresearch.md` per the autoresearch-bio skill rule that launch messages must be delivered separately from the prompt file.

---

You are operating in fully autonomous Debate Council mode on the RAM PoC repo at https://github.com/yashraj59/ram_poc. The goal is to push cell-eval PDS to 0.80 or higher on the official Arc Institute Virtual Cell Challenge validation deliverable, then predict the official test set, generate closure plots, and stop.

You will read two repositories. `ram_poc/` is the project. `autoresearch-bio/` is the discipline skill the project depends on. The autoresearch.md prompt inside ram_poc cites the skill by relative paths like `references/core_protocol.md §3.5` and `references/debate_council.md`. Those resolve to the sibling `autoresearch-bio/` directory. You must clone both before doing anything else.

The only condition under which you are allowed to halt and ask for human input is "I cannot locate or download the official VCC data." For every other situation, work autonomously, log the decision, and continue. The council is single-vendor with self-critique enabled. Treat consensus as correlated, not independent.

Operational sequence to start:

1. Clone both repos side-by-side. From the working directory:
   ```
   git clone https://github.com/yashraj59/ram_poc.git
   git clone https://github.com/yashraj59/autoresearch-bio.git
   cd ram_poc
   ```

2. Read `ram_poc/autoresearch.md` end to end. That is your primary instruction set. While reading it, every reference like `references/core_protocol.md §3.5` means `../autoresearch-bio/references/core_protocol.md`. Open those skill files as you encounter the citations.

3. Read at least these skill files before launching, since they define the rules the prompt assumes you will follow:
   - `../autoresearch-bio/SKILL.md`
   - `../autoresearch-bio/references/core_protocol.md` (Step 0, identity, families, tiered gates, leakage pre-flight in §3.5, literature search in §13, stop conditions in §14)
   - `../autoresearch-bio/references/debate_council.md` (council process, self-critique step, hard escalation triggers)
   - `../autoresearch-bio/references/biology_addendum.md` (safety boundary, external baseline rules)
   - `../autoresearch-bio/references/decision_labels.md` (reserved-string rule, all failure labels)
   - `../autoresearch-bio/references/lineage.md` (branch types)
   - `../autoresearch-bio/references/statistical_promotion.md` (multiple-comparison floor)
   - `../autoresearch-bio/references/artifact_retention.md` (per-experiment summary.json identity block, resumability, INSIGHT_BRIEF cadence, append-only log hygiene)
   - `../autoresearch-bio/references/metric_investigation.md` (when to open a metric investigation instead of architecture search)
   - `../autoresearch-bio/references/amendment_review_checklist.md` (the seven checks every amendment must pass)
   - `../autoresearch-bio/assets/split_manifest.schema.json` (split manifest schema)

4. Set up the environment in `ram_poc/`:
   ```
   pip install -r requirements.txt
   ```
   Install Arc Institute's `cell-eval` per their release instructions.

5. Authenticate to GCS:
   ```
   gcloud auth application-default login
   ```

6. Fetch the VCC data:
   ```
   bash scripts/download_vcc_data.sh data/vcc
   ```
   Confirm the training file, the validation deliverable spec (target perturbations and cell counts), and the validation ground-truth h5ad are all present. If the bucket layout has changed and you cannot identify these files, this is your one allowed escalation condition. Otherwise continue.

7. Fetch the external resources you decide to start with: scGPT gene embeddings, ESM2 protein embeddings, a TF list, and GO annotations. Save under `data/external/` and document each in `outputs/external_resources.md` with version, URL, license, and date. You have total freedom to choose alternatives or augment with additional cell-type or perturbation data from public sources.

8. Build the four-role split manifest at `outputs/split_manifest.json` per `../autoresearch-bio/assets/split_manifest.schema.json`. The `locked_test` role is the official VCC validation deliverable. The `validation` role is held out from the training file for internal early-stop. Document the split rule in `outputs/leakage_preflight.md`.

9. Write `outputs/leakage_preflight.md` per `../autoresearch-bio/references/core_protocol.md §3.5`. Enumerate every code path in `model/model.py` that reads each split role. Certify that `locked_test` is never read during training, status assignment, anchor selection, or warm-start. Set `leakage_guard: PASS_NO_TEST_SELECTION` as the default for every `results.tsv` row.

10. Run EXP000, the Step 0 baseline, as documented in `autoresearch.md` Step 0 Baseline Plan. Pseudobulk pretrain, then single-cell warmstart with `--x0_groupby batch,cell_type`. Predict the validation deliverable. Run `cell-eval run -ap <pred>.h5ad -ar <real>.h5ad --num-threads 64 --profile vcc`. Record the baseline PDS in `outputs/results.tsv` and `outputs/BASELINE_REGISTRY.md`.

11. Iterate per the family-allocation rules. Tier 1 single-seed, then Tier 2 multi-seed at 5 seeds (66, 1, 7, 17, 23), then Tier 3 cell-eval confirmation against `locked_test`. Each candidate calls cell-eval at most once. The reserved-string rule from `../autoresearch-bio/references/decision_labels.md` applies to every status value you emit.

12. Run the validation script from the skill against `outputs/` after every 10 experiments to catch process drift early:
    ```
    python ../autoresearch-bio/scripts/validate_autoresearch_artifacts.py outputs/
    ```
    If it returns non-zero, fix the flagged issues before the next experiment.

13. Stop the moment cell-eval PDS reaches 0.80 on `locked_test`. Predict the official VCC test set, generate the 8 closure plots using `scripts/generate_closure_plots.py` (adapt the `# TODO(agent)` stubs to your actual run data), and write `outputs/final_report.md`.

The autonomous council may add up to 4 new architectural families during the loop, subject to the six conditions in `autoresearch.md` Architectural Families "Procedure for autonomous family addition" (literature pass, identity preservation, Skeptic counter-argument, concrete self-critique, full family documentation, stricter Tier 1 threshold of +0.025). Above 4 additions, halt with `AUTONOMOUS_FAMILY_LIMIT_REACHED`. The council may not autonomously override the 200-experiment cap, lower any threshold, relax the multiple-comparison floor, skip the literature pass, promote without cell-eval confirmation, or change any identity lock.

Compute budget: 200 experiments, single GPU with 96 GB VRAM, mixed precision when stable, 8-hour per-run timeout (label `TIER1_DISCARD_COMPUTE_TIMEOUT` and continue).

Maintain these documentation files after every experiment per `autoresearch.md` Documentation Files section: `outputs/results.tsv` (with the lineage columns and `cell_eval_pds`), `outputs/research_journal.md`, `outputs/architectural_changes_log.md`, `outputs/family_allocation.md`, `outputs/papers_consulted.md`, `outputs/identity_violations_considered.md`, `outputs/external_resources.md`, `outputs/STATE_OF_PLAY.md` (regenerate, never append), `outputs/insights/INSIGHT_BRIEF_NNN.md` every 10 experiments.

Literature search: when a when-stuck trigger fires (three consecutive Tier 1 discards in a family, a Tier 2 failure with protected-metric regression, a ruled-out mechanism class, or 20 experiments without clearing the MCC floor), run a literature pass per `../autoresearch-bio/references/core_protocol.md §13`. Declare your fetch fingerprint in `outputs/leakage_preflight.md` at run start and keep it consistent. The seed papers in autoresearch.md "When-stuck triggers" are a starting point only.

Begin now.

---

## Notes for the human pasting this

- The skill repo is read-only from the agent's perspective. The agent should never write into `../autoresearch-bio/`. All experiment artifacts land in `ram_poc/outputs/`.

- If your agent's shell environment has clone restrictions, pre-clone `autoresearch-bio` next to `ram_poc` yourself before starting the session. The launch message's step 1 will then either succeed or fail harmlessly with "destination exists" and the agent proceeds.

- The agent should be configured with a single LLM provider (per the user's instruction). Multi-vendor council is off for this run. If your agent harness defaults to multi-vendor, override it before launching.

- A 200-experiment cap with up to 8 hours per run is roughly 1600 GPU-hours worst case. If your agent session has a hard timeout shorter than that, plan to resume by pointing a fresh session at the existing `outputs/` directory. The autoresearch.md already requires `STATE_OF_PLAY.md` and per-experiment journal entries that make resumption tractable.
