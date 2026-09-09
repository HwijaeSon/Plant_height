# Arabidopsis hypocotyl growth under light regimes

This benchmark uses the original workbook supplied for the study:
**1,818 measured cells**, five genotypes, two lighting regimes and seven
elapsed times (0--72 h, every 12 h), at a fixed **23 degrees C**. Length is
measured in **mm**. The 12L12D treatment starts with light during 0--12 h;
`cR` denotes continuous red light.

| Source label | Identity | 12L12D measurements | cR measurements |
|---|---|---:|---:|
| Col-0 | Columbia-0 wild type | 197 | 198 |
| hy5 | hy5 mutant | 196 | 188 |
| MLB | ML1promoter::phyB-GFP/phyB-9 | 225 | 218 |
| EMS57 | EMS57 line | 94 | 73 |
| phyAB | phyA/phyB double mutant | 231 | 198 |
| Total | | 943 | 875 |

These are measurement counts, **not verified counts of distinct plants**.
Individual longitudinal identifiers are absent. Each genotype/time/condition
cell contains 3--38 replicate measurements. The first measurement's age is
uncertain, so the analysis uses elapsed hours rather than days after germination.

## Data and provenance

- [Original workbook](data/raw/hypocotyl_growth_20230403.xlsx).
- [Tidy measured cells](data/processed/observations.csv), with original sheet
  and cell identifiers. Summary formulas are excluded.
- [Group means, sample SDs and counts](data/processed/group_statistics.csv).
- [Metadata and unresolved experimental details](data/metadata.json).
- [Extraction verification](data/processed/extraction_validation.json): all
  70 recomputed group means match the workbook's cached summary means.
- [Frozen partitions](data/processed/splits/manifest.json).
- [Literature basis and precise model adaptation](reports/model_rationale.md).

The original workbook and processed measurements are supplied as study data.
No dataset DOI or explicit reuse license has yet been assigned. The filename
is retained for provenance and does not independently verify the collection date.

## Evaluation

| Protocol | Train measurements | Validation measurements | Test measurements | Test target |
|---|---:|---:|---:|---|
| Replicate-group holdout | 1,052 | 330 | 436 | Ten population-mean curves, seven times each |
| Unseen-time interpolation | 758 | 240 | 505 | Ten population-mean curves at 36 and 60 h |

Source-row groups within genotype are assigned jointly across times and both
sheets with seed 20260909. This prevents a putative same-row trajectory from
crossing the primary split, without asserting that row numbers identify plants.
Partition retries check only whether every group has train/validation/test
coverage. No height value is used to choose a partition.

The time-holdout protocol trains and validates only at 0, 12, 24, 48 and 72 h.
All observations at 36 and 60 h are reserved for testing; 315 measurements in
the primary test partition at the other five times are unused in this protocol.
The two protocols share a source dataset and are complementary analyses,
not independent biological replications.

Only training replicates form training mean targets; validation and test means
are computed separately. All models receive the same available information.
The primary RMSE is the mean per-genotype/condition curve RMSE. Relative RMSE
is this value divided by the mean of the scored replicate-mean targets, times
100. Individual-measurement RMSE is also retained, so averaging is explicit.
Training-seed SD quantifies initialization variability, not biological uncertainty.
Some means contain only one measurement after partitioning (ranges: train
1–22, validation 1–7, replicate test 1–9). The withheld-time test means use
13–33 measurements each. These two test targets have different averaging
precision as well as different temporal support.

## Run

```bash
.venv/bin/python hypocotyl/code/prepare_data.py
# Run once into a fresh split directory; frozen partitions reject overwrite.
.venv/bin/python hypocotyl/code/make_splits.py
.venv/bin/python hypocotyl/code/verify_model.py
.venv/bin/python hypocotyl/code/run_all.py --gpus 8 9
# Only after the controller has completed and frozen all selections:
.venv/bin/python hypocotyl/code/summarize.py
.venv/bin/python paper/write_hypocotyl_results.py
.venv/bin/python hypocotyl/code/package_release.py
```

The controller screens learning rates and physics coefficients using seed 1,
confirms the top two candidates with seeds 2--3, and ranks completed candidates
by mean validation RMSE. It freezes every model selection for both protocols
before scoring test targets. Process fits are deterministic single runs;
neural models and random forest use three seeds. Use `--resume` to retain
completed trials after an interruption; incomplete attempts are preserved.

The original wheat, maize and Arabidopsis stem-length training sources and
frozen experiment results remain separate from this new benchmark.

## Outputs

The completed comparison contains 122 training/fitting runs and 40 frozen-choice
evaluations. PhytoODE's test RMSE / relative RMSE is **0.344 mm / 5.93%** for
replicate holdout and **0.305 mm / 4.13%** for withheld-time interpolation.
It has the lowest mean among the seven main methods in the time test; the
independently tuned no-physics latent ODE is better in replicate holdout
(0.337 mm / 5.81%). Differences among the leading neural models are small
relative to seed variability. This supports a protocol-dependent conclusion,
not a claim of uniform superiority from physics regularization.

The selected `(lambda_ODE, lambda_K)` pairs are `(50, 0.01)` and `(0.5, 0.01)`,
respectively, with learning rate `0.01` in both protocols. The corresponding
same-LR, same-initialization no-physics controls score 0.335 and 0.345 mm.
Final CPU evaluation was accelerated by calling the unchanged evaluator in
one interpreter after selection froze; [the runtime record](results/light_growth_20260909/evaluation_runtime.json)
documents this. The default controller reproduces the same evaluations
sequentially.

- [Comparison tables and seed statistics](reports/results.md).
- [Full train/validation/test metrics](reports/comparison.csv), including
  individual-measurement RMSE and the separately matched loss-removal control.
- [Selected configurations](reports/selected_configs.csv) and
  [paired loss comparisons](reports/paired_loss_ablation.csv).
- [Raw data overview](reports/data_overview.png),
  [replicate predictions](reports/predictions_replicate.png), and
  [withheld-time predictions](reports/predictions_time_holdout.png).
- [Metric/selection audit](reports/results_audit.json),
  [partition audit](reports/partition_audit.json), and
  [solver refinement](reports/integration_refinement.csv).
- [Data dictionary](DATA_DICTIONARY.md) and a
  [deposit-ready data/code ZIP](release/hypocotyl_data_code_20260909.zip).
  The ZIP is prepared locally; this does not establish a public deposit,
  DOI, or data-reuse license.
