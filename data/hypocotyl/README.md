# Author-collected Arabidopsis hypocotyl trajectories

The released cohort contains **Col-0, hy5, MLB, and phyAB**. Plants were grown
at 23°C under 12 h light / 12 h dark. Individual lengths were recorded in mm
every 12 h from 0 to 72 h. Genotype and source-workbook row identify the same
plant across times. Developmental age at the first measurement, independent
batch identifiers, irradiance, and some mutant alleles are not established.

| Genotype | Description | Released 12L12D observations |
|---|---|---:|
| Col-0 | Columbia-0 wild type | 197 |
| hy5 | hy5 mutant; allele not recorded | 196 |
| MLB | ML1promoter::phyB-GFP/phyB-9 | 225 |
| phyAB | phyA/phyB double mutant; alleles not recorded | 231 |
| Total | | 849 |

The measurement-only workbook preserves approved cells at their original
coordinates. Blank columns contain no released genotype; summary formulas and
embedded charts are omitted. Its cR sheet retains the four approved genotypes
under continuous red light, but those measurements are not used in this task.

## Forecasting protocol

A plant is eligible if it has at least one observed length at 0–36 h. One Col-0
plant with only a 48-h observation is excluded. The resulting cohort contains
**141 plants and 848 observations**, preserving the original missing entries.

| Partition | Times (h) | Observations | Plants with observations |
|---|---|---:|---:|
| Input and training | 0, 12, 24, 36 | 500 | 141 |
| Validation | 48 | 116 | 116 |
| Test | 60, 72 | 232 | 120 |

The same plants contribute to different temporal partitions. No 48-h value is
added to the input when predicting 60/72 h. There is no target averaging or
imputation. For final PhytoODE and its matched control, the scale is the maximum original
training length and stays fixed across removal masks. The four reference
baselines retain their original per-mask training scale. A masked zero is a computational placeholder.

## Missingness

There are 564 possible early-measurement slots, of which 64 are naturally missing.
At least one observation is retained for each plant at every removal level.

| Additional removal | Retained early observations | Total missingness |
|---|---:|---:|
| None | 500 | 11.35% |
| 25% | 375 | 33.51% |
| 50% | 250 | 55.67% |

Masks use seeds `20260910 + training_seed` and are nested across removal levels.
The final PhytoODE/control pair shares masks for seeds 101–105. The four
reference models use seeds 1–3 (one natural Logistic ODE fit). All fits start
from scratch. Future targets and the eligible cohort stay fixed.

## Models and results

The models are PhytoODE, Latent ODE, Logistic-PINN, LSTM-NN, RF, and Logi-ODE.
All receive genotype, elapsed time, available early lengths and masks. Final
PhytoODE and its matched control also receive the known light schedule in the
latent vector field. PhytoODE has 1,194 parameters, genotype-specific mean rates
and light/dark contrasts, and an individual capacity head conditioned on the
initial observations. It uses `lambda_ode=300`, `lambda_k=0`; the otherwise
identical control changes only `lambda_ode` to zero. Epoch selection uses 48-h
validation observations.

Primary RMSE averages each evaluated plant's masked RMSE. Relative RMSE is
`100 × primary RMSE / pooled observed target mean`. SD describes seed and mask
variation, not independent biological cohorts. The final pair uses five seeds;
reference models use three (one for natural Logistic ODE).

[Comparison data](../../results/comparison_hypocotyl.csv) combine the final pair with
the four reference baselines. All recorded predictions and metrics are in
`results/hypocotyl/`; fitted objects are in `checkpoints/hypocotyl/`.

## Data dictionary

Data files are in `processed/`.

| Field | Meaning |
|---|---|
| observation_id | Unique sheet and source-cell identifier |
| plant_id | Genotype and original Excel row |
| genotype | One of the four approved labels above |
| source_row, source_cell | Original measurement row and cell |
| elapsed_hours | Hours since the first measurement |
| length_mm | Observed individual length |
| plant_index | Position in the current plant table |
| g_idx | Current genotype index: Col-0=0, hy5=1, MLB=2, phyAB=3 |
| split | Training, validation, or test partition |

`observations.csv` contains the complete eligible cohort; `train.csv`, `val.csv`,
and `test.csv` are its temporal partitions. `plants.csv` records the plant
indices. `excluded_observations.csv` contains only the ineligible Col-0
observation. `config.json` records the release-workbook checksum and cohort
counts, and `metadata.json` records experimental metadata.

## Reproduce

From the repository root:

```bash
python src/scripts/prepare_hypocotyl.py
python run.py evaluate --dataset hypocotyl --seed 101 --output outputs/hypocotyl
python src/scripts/verify_models.py --dataset hypocotyl --output outputs/hypocotyl-check
```

See the root README for individual-model training and checkpoint evaluation.
