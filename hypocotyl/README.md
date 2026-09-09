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

Source-row groups within genotype are assigned jointly across times and both
sheets with seed 20260909. This prevents a putative same-row trajectory from
crossing the primary split, without asserting that row numbers identify plants.
Partition retries check only whether every group has train/validation/test
coverage. No height value is used to choose a partition.

All seven observation times occur in all three partitions. The manuscript uses this replicate-group evaluation throughout the hypocotyl section.

Only training replicates form training mean targets; validation and test means
are computed separately. All models receive the same available information.
The primary RMSE is the mean per-genotype/condition curve RMSE. Relative RMSE
is this value divided by the mean of the scored replicate-mean targets, times
100. Individual-measurement RMSE is also retained, so averaging is explicit.
Training-seed SD quantifies initialization variability, not biological uncertainty.
Some means contain only one measurement after partitioning (ranges: train
1–22, validation 1–7, replicate test 1–9).

## Run

The original benchmark is frozen under `results/light_growth_20260909`.
The primary follow-up leaves its training sources, partitions, and fitted
baseline outputs unchanged:

```bash
.venv/bin/python hypocotyl/code/retune_primary.py --gpus 8 9
# After training and frozen-choice evaluation complete:
.venv/bin/python hypocotyl/code/summarize_primary.py
.venv/bin/python paper/write_hypocotyl_results.py
.venv/bin/python hypocotyl/code/package_release.py
```

The follow-up screens 80 fresh PhytoODE configurations: 32 loss/learning-rate
grid points at the original capacity and 48 stratified combinations of
capacity, learning rate, both loss coefficients, weight decay, and training
length. The best eight seed-1 candidates receive seeds 2 and 3. Three-seed
mean validation RMSE selects the final candidate, with both fully confirmed
original PhytoODE configurations eligible as incumbents. Three additional
runs disable both physics coefficients at the selected architecture and
training settings. This is a fixed budget of 99 new fits; it is not extended
based on test performance. The full candidate list and stopping rule are
recorded before training in `results/primary_retuning_20260909/protocol.json`.

The initial test scores had already been inspected before this follow-up.
Training and selection load training/validation targets only, but the reused
test partition is not a fresh blinded confirmation. Other baselines retain
their original independent searches, so this is not an equal-compute comparison.
Use `--resume` to retain completed trials after interruption; partial attempts
are preserved in the diagnostics directory.

The recorded screening stage reused GPU worker interpreters with two workers
per GPU to reduce launch overhead. The unchanged frozen training function was
called for each candidate, and the original controller resumed for selection,
confirmation, and test evaluation. `screening_runtime.json` records this
execution detail; the default controller can reproduce the same experiment
without the optional runtime helper.

## Outputs

- [Current primary comparison and selected settings](reports/primary_retuning_20260909/results.md).
- [Train/validation/test metrics](reports/primary_retuning_20260909/comparison.csv),
  including individual-measurement RMSE, the current matched loss-removal control,
  and the earlier same-learning-rate no-physics control. The initial control
  remains better on test than the newly tuned PhytoODE; a broad superiority
  claim is not supported.
- [All-genotype primary predictions](reports/primary_retuning_20260909/predictions_replicate.png).
- [Source, metric, selection, and pairing audit](reports/primary_retuning_20260909/results_audit.json).
- [Raw data overview](reports/data_overview.png), [partition audit](reports/partition_audit.json),
  and [data dictionary](DATA_DICTIONARY.md).
- [Deposit-ready data/code ZIP](release/hypocotyl_data_code_20260909.zip).
  The ZIP is prepared locally; this does not establish a public deposit,
  DOI, or data-reuse license.

## Historical analysis archive

The original 122 fits and 40 evaluations are retained unchanged in
`results/light_growth_20260909`, with [their original report](reports/results.md).
That archive includes an auxiliary time-holdout analysis that is excluded
from the current manuscript. Its results must not be substituted for the
replicate-group primary evaluation. The initial primary PhytoODE result
was 0.343717 mm / 5.931501%; the original independently learning-rate-tuned
latent ODE result was 0.336836 mm / 5.812771%.
