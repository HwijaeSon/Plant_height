# Author-collected Arabidopsis hypocotyl trajectories

Plants were grown at **23°C under 12 h light / 12 h dark (12L12D)**. Length was
measured in **mm** at **0, 12, 24, 36, 48, 60, and 72 h**. The first 0–12 h
interval was illuminated. Time is elapsed time from the first measurement;
developmental age at that measurement has not been confirmed.

The contributor confirmed that **genotype × original Excel row identifies the
same plant across times**. Different genotypes or rows identify different plants.
The current experiment uses these individual trajectories, with their original
missing entries. No replicate means or interpolated/imputed targets are used.

| Genotype | Description | Raw observations |
|---|---|---:|
| Col-0 | Columbia-0 wild type | 197 |
| hy5 | hy5 mutant; allele not recorded | 196 |
| MLB | ML1promoter::phyB-GFP/phyB-9 | 225 |
| EMS57 | EMS57 line; molecular lesion not recorded | 94 |
| phyAB | phyA/phyB double mutant; alleles not recorded | 231 |
| Total | | 943 |

Irradiance, spectral composition, growth medium, independent batch identifiers,
and exact age at time zero are not established in the available records.
Continuous-red-light (`cR`) measurements remain in the original workbook but are
excluded from all current model inputs, targets, normalization, and evaluation.

## Current evaluation: forecast each plant from its initial measurements

A plant is included if it has at least one observed length at 0–36 h. This removes
one plant (`Col-0:row37`) with only a 48-h measurement, leaving **157 plants and
942 observations**. Future lengths or future observation availability are not
used to decide eligibility.

| Partition | Times (h) | Individual observations | Plants with observations |
|---|---|---:|---:|
| Input and training | 0, 12, 24, 36 | 551 | 157 |
| Validation | 48 | 128 | 128 |
| Test | 60, 72 | 263 | 136 |

The same plant can occur in each temporal partition. This measures future growth
of plants with an observed prefix, not performance on new plants or genotypes.
The 48-h value is never supplied as an input for 60/72-h predictions. The forecast
origin is 36 h; test horizons are 24 and 36 h beyond that origin. Training error
measures reconstruction of the observed input prefix.

A predictor receives genotype, four masked prefix lengths, four presence masks,
and time. PhytoODE and its input-matched `latent_ode_light` control also receive
the known binary illumination schedule. A unique plant ID is not a model input.
Different plants of one genotype can produce different trajectories through their
numeric prefix. Zero tensor entries with mask zero are placeholders and make no
contribution to data loss or scored errors.

The maximum observed training length is **9.602 mm** under natural missingness.
Primary RMSE averages the masked RMSE of each evaluated plant. Relative RMSE is
`100 × mean per-plant RMSE / mean scored individual target`; it is not MAPE.
Pooled observation-level RMSE is recorded separately in each result file.

## Controlled missingness experiment

Of 628 possible prefix slots (157 plants × four times), 77 are naturally missing.
The additional-removal experiment retrains **every model** with shared masks:

| Removed fraction of available prefix measurements | Training observations retained | Missing fraction of all prefix slots |
|---|---:|---:|
| 0% (natural missingness) | 551 | 12.26% |
| 25% | 413 | 34.24% |
| 50% | 275 | 56.21% |

Removal is random without examining lengths, is nested across severity levels,
and retains at least one observation per plant. Mask seeds are `20260910 +
training_seed` for training seeds 1–3. Each removed value is excluded from both
inputs and the training loss. Normalization is fitted again to retained prefix
values only. Validation and test cells and the eligible cohort remain fixed.

The matched light-input latent ODE uses the exact PhytoODE architecture,
initialization, optimizer, and masks with **both physics coefficients zero**.
Hyperparameters are frozen in `configs/hypocotyl_forecast.json`; no search is
performed after inspecting these forecasting results. Checkpoint selection uses
48-h validation only. Standard deviations across added-missingness runs combine
mask and initialization variation, not biological-cohort uncertainty.

## Files and source traceability

- `data/raw/hypocotyl_growth_20230403.xlsx`: unchanged original workbook.
- `data/processed/prefix_forecast_20260911/observations.csv`: all 942 retained
  observations with plant and source-cell identifiers.
- `plants.csv` in the same folder: the 157 plant IDs and genotype indices.
- `train.csv`, `val.csv`, `test.csv`: fixed temporal partitions.
- `excluded_observations.csv`: the single excluded measurement and its source.
- `config.json`: counts, scale, eligibility rule, and original-workbook SHA-256.
- `data/metadata.json`: confirmed experimental details and recorded omissions.
- `results/prefix_forecast_20260911/`: 61 fitted runs, frozen plan, checkpoints,
  validation histories, prediction arrays, selected epochs, and per-run metrics.
- `reports/prefix_forecast_20260911/`: comparison CSV/Markdown, paired physics
  differences, and data audit. Figure files are generated locally from the code.

| CSV field | Meaning |
|---|---|
| `observation_id` | Unique `12L12D:cell` measurement identifier |
| `plant_id` | Longitudinal ID: genotype and original Excel row |
| `genotype` | One of the five labels above |
| `source_row`, `source_cell` | Original Excel row and cell |
| `elapsed_hours` | Time since the first observation |
| `length_mm` | Observed individual length |
| `plant_index` | Position in the bundled plant table |
| `split` | Temporal training, validation, or test partition |

Earlier mean-target and replicate-group experiments are preserved in the prior
Git tag `submission-20260911-code-data`. Their metrics and checkpoints do not
apply to this forecasting protocol.

## Reproduce

Run from the repository root after installing `requirements.txt`:

```bash
python scripts/prepare_hypocotyl.py
# Optional full audit of masks, saved metrics, and every neural checkpoint:
python hypocotyl/code/audit_forecast_artifacts.py
python run.py evaluate --dataset hypocotyl --seed 1 \
  --output outputs/hypocotyl-evaluation
python run.py train --dataset hypocotyl --model phytoode --seed 1 \
  --device cuda:0 --output outputs/hypocotyl-training
python run.py train --dataset hypocotyl --model latent_ode_light --seed 1 \
  --device cuda:0 --drop-fraction 0.5 --output outputs/hypocotyl-matched-missing

# All seven methods, three seeds, and three missingness conditions (61 fits).
python hypocotyl/code/run_forecast_benchmark.py --gpus 0 1 2 \
  --output outputs/retrained-forecast-benchmark

# Export bundled scores and generate both forecast/missingness figures.
python hypocotyl/code/report_forecast.py \
  --output outputs/forecast-report --figures outputs/forecast-figures
```

Other model names are `latent_ode` (without illumination), `logistic_pinn`,
`lstm`, `rf`, and `logistic`. Classical baselines use CPU. Neural models train
for 1,500 epochs; RF uses 300 trees. Logistic ODE uses genotype-specific rate and
capacity and one fitted initial length per plant. PhytoODE's ordinary logistic
regularizer has genotype-specific constant rate/capacity, while illumination
conditions the latent dynamics. It does not estimate separate light/dark rates.
The derivative penalty is applied at 23 interior 3-h grid nodes, including times
with no observed length. See `code/forecast_models.py` and `code/forecast_trial.py`.
