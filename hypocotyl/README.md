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
- [Literature basis and precise model adaptation](reports/light_reference_notes.md).

The original workbook and processed measurements are supplied as study data.
No dataset DOI or explicit reuse license has yet been assigned. The filename
is retained for provenance and does not independently verify the collection date.

## Evaluation

| Protocol | Train measurements | Validation measurements | Test measurements | Test target |
|---|---:|---:|---:|---|
| Replicate-group holdout (primary) | 1,052 | 330 | 436 | Each measured length |
| 36 / 60 h holdout (auxiliary) | 758 | 240 | 505 | Each measured length at 36 / 60 h |

The current benchmark preserves **every measured cell as a separate target**.
It does not impute missing phenotypes or construct replicate-mean targets.
A curve prediction at genotype, light condition, and time is compared directly
with each corresponding measured length. No longitudinal plant identities are
inferred. The genotype/source-row bookkeeping groups and original seed 20260909
remain fixed.

The primary protocol includes all seven observation times in all partitions.
The auxiliary protocol excludes all 36 / 60 h phenotypes from training and
validation; 315 original replicate-test measurements at other times are unused.
Only the primary replicate protocol is reported in the manuscript.

The loss is RMSE pooled over individual training measurements after dividing
lengths by the maximum training measurement (16.018 mm). Reported RMSE is in mm;
relative RMSE divides it by the mean of the actual scored lengths and multiplies
by 100. This denominator does not replace any training target. Unequal replicate
counts produce correspondingly unequal numbers of residuals. Training-seed SD
quantifies optimization variability, not biological uncertainty.

Frozen split JSON files retain the historical mean-target scoring description
for provenance. The **current scoring definition** is in
`code/observation_data.py` and the new experiment's `protocol.json`.

## Run

Run from the repository root:

```bash
.venv/bin/python hypocotyl/code/verify_observation_targets.py
.venv/bin/python hypocotyl/code/run_observation_benchmark.py --gpus 8 9
# After all training and frozen-choice evaluations finish:
.venv/bin/python hypocotyl/code/summarize_observations.py
.venv/bin/python paper/write_hypocotyl_results.py
.venv/bin/python hypocotyl/code/package_release.py
```

The fixed catalog contains 24 paired architecture/optimizer settings for each of
PhytoODE and the no-physics latent ODE, 12 Light-PINN settings, four LSTM settings,
four RF leaf sizes, and two process ODEs. All 70 are trained from scratch at seed
1 separately for both protocols. The leading three configurations of each latent
model and two configurations of each other stochastic baseline receive seeds 2
and 3. Selection uses only mean validation RMSE. A strictly matched control sets
both physics coefficients to zero at the selected PhytoODE settings.

The complete budget is 140 screening fits, 48 confirmation fits, and any missing
matched-control seeds. Every neural fit uses 2,000 epochs and the same checkpoint
rule. Four persistent workers use GPUs 8 and 9, two per GPU. The catalog and source
hashes are frozen in `results/individual_observations_20260909/protocol.json`.
The test partitions had been inspected in preliminary analyses; no test score is
used to rank or extend the current search. All model selections for both protocols
freeze before new test scoring. Use `--resume` after an interruption; incomplete
attempts are retained under diagnostics.

## Outputs

- [Current raw-observation comparison](reports/individual_observations_20260909/results.md).
- [Train/validation/test metrics](reports/individual_observations_20260909/comparison.csv).
- [All-genotype primary predictions](reports/individual_observations_20260909/predictions_replicate.png).
- [Auxiliary time-holdout predictions](reports/individual_observations_20260909/predictions_time_holdout.png).
- [Individual predictions](reports/individual_observations_20260909/individual_predictions.csv.gz).
- [Fitted light/dark rates](reports/individual_observations_20260909/light_growth_parameters.csv).
- [Target verification](reports/individual_observations_20260909/target_verification.json)
  and [result audit](reports/individual_observations_20260909/results_audit.json).
- [Literature basis](reports/light_reference_notes.md): New Phytologist, Nature,
  and Molecular Systems Biology; the switched equation is our stated adaptation.
- [Raw data overview](reports/data_overview.png), [split visualizations](reports/holdout_visualization/README.md),
  and [data dictionary](DATA_DICTIONARY.md).
- [Data/code ZIP prepared for deposit](release/hypocotyl_data_code_20260909.zip).

The prediction figures show every test measurement separately. Only model
predictions are averaged across training seeds. The split-explanation figures
also display individual measurements without replicate-mean markers.

## Historical analysis archive

The mean-target experiments remain available in `results/light_growth_20260909`
and `results/primary_retuning_20260909`, with reports under `reports/results.md`
and `reports/primary_retuning_20260909/`. Their training sources are unchanged.
Their main scores concern replicate means and must not be mixed into the new
individual-observation table. The current entry point fits and scores raw
observations for all methods; it reuses no mean-target checkpoint.
