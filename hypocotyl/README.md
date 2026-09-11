# Author-collected Arabidopsis hypocotyl dataset

This dataset was collected for the accompanying PhytoODE study. The submission
evaluation uses plants grown at **23°C under 12 h light / 12 h dark (12L12D)**.
Hypocotyl length was measured in **mm** at **0, 12, 24, 36, 48, 60, and 72 h**.
The first 0–12 h interval was illuminated. Time is elapsed time from the first
measurement; developmental age at that measurement has not been confirmed.

| Genotype label | Description | Measurements |
|---|---|---:|
| Col-0 | Columbia-0 wild type | 197 |
| hy5 | hy5 mutant; allele not recorded | 196 |
| MLB | ML1promoter::phyB-GFP/phyB-9 | 225 |
| EMS57 | EMS57 line; molecular lesion not recorded | 94 |
| phyAB | phyA/phyB double mutant; alleles not recorded | 231 |
| Total | | 943 |

Each genotype/time group has 6–38 measured replicates. Irradiance, spectral
composition of the 12L12D treatment, growth medium, independent batch identifiers,
and exact age at time zero are not established in the available records.

## Files

- `data/raw/hypocotyl_growth_20230403.xlsx`: original workbook, preserved byte for
  byte. It also contains a continuous-red-light (`cR`) sheet. That sheet is not
  used in the manuscript experiment.
- `data/observations_12L12D.csv`: all 943 retained measurements with source-cell
  identifiers and fixed partition membership.
- `data/split_assignments.csv`: immutable observation-ID-to-partition mapping.
- `data/metadata.json`: confirmed experimental metadata and recorded omissions.
- `data/processed/single_condition_20260909/replicate/`: the exact training,
  validation, and test CSVs, split-specific mean targets, and training scale.

| Field | Meaning |
|---|---|
| `observation_id` | Unique `12L12D:cell` identifier, not a longitudinal plant ID |
| `sheet` | Source sheet; always `12L12D` in the tidy submission table |
| `genotype` | One of the five labels above |
| `elapsed_hours` | Time since the first observation |
| `length` | Measured hypocotyl length in mm |
| `source_cell` | Excel cell coordinate |
| `source_row`, `source_column` | One-based Excel indices |
| `split` | `train`, `val`, or `test` |

Mean-target CSVs contain `genotype`, `elapsed_hours`, `mean_length_mm`, and
`n_observations`. They are calculated separately within each partition.

## What is held out?

The workbook does not verify that the same plant was measured over time. We
therefore model genotype-level mean growth from replicate snapshots. Records
sharing genotype and source row remain in one partition across times, without
asserting that a row identifies a longitudinal plant.

The fixed partitions contain **547 training, 171 validation, and 225 test
measurements**. Every partition contains all five genotypes at all seven times,
giving **35 mean targets per partition**. The model is fitted to training means;
validation and test means are used exclusively in their respective evaluations.
This is a replicate-group holdout within one experiment, not a new photoperiod,
genotype, or experimental-batch holdout.

Blank workbook cells remain absent. No missing length or time group is imputed.
Means include only measured cells from their own partition; whole-dataset means
are not used as training targets. The training scale is the maximum raw training
length, **13.719 mm**. Relative RMSE normalizes mean genotype-trajectory RMSE by
the mean of the scored partition-specific means. Individual-length pooled RMSE
is a separate secondary metric.

## Verify and run

From the repository root:

```bash
python scripts/prepare_hypocotyl.py
# Optionally rebuild target CSVs in a new directory for inspection:
python scripts/prepare_hypocotyl.py --output outputs/hypocotyl-reextracted
python run.py evaluate --dataset hypocotyl --seed 1 \
  --output outputs/hypocotyl-paper-evaluation
python run.py train --dataset hypocotyl --seed 1 --device cuda:0 \
  --output outputs/hypocotyl-paper-training
```

Verification re-extracts only measurement cells in rows 3–44, checks their
partition membership, reproduces the split-specific means and training scale,
and compares them to the submitted targets. Spreadsheet summary formulas and
chart-input cells are excluded. The original partition files are not overwritten.

PhytoODE receives genotype, time, and the prescribed binary illumination schedule.
The logistic reference has a constant growth-rate and carrying-capacity parameter
per genotype; it does **not** fit separate light/dark logistic rates. Illumination
conditions the learned latent vector field. The integration grid is 3 h, and
all 23 interior nodes contribute to the logistic derivative residual. See
`code/light_input_model.py` and `code/single_condition_models.py`.

The five primary baselines use genotype and time without illumination. Additional
`run.py --model` choices for this dataset are `latent_ode`, `logistic_pinn`, `lstm`,
`rf`, and `logistic`. The manuscript's feature comparisons are
`phytoode_no_light` and `phytoode_light_fixed`; neither replaces the
validation-selected primary `phytoode` model.

The partition was reused after preliminary test inspection. PhytoODE received
additional tuning and an input feature not supplied to the retained baselines.
These details are part of the submitted evaluation protocol.
