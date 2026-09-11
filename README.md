# PhytoODE

**Learning plant growth dynamics across genotypes and observation regimes**

This repository contains the code, selected models, and author-collected
dataset for the PhytoODE manuscript, together with separately recorded follow-up
experiments. PhytoODE combines genotype-conditioned latent
neural ODE dynamics with logistic derivative regularization in phenotype space.
The three published temperature datasets additionally use carrying-capacity
regularization. The retained hypocotyl PhytoODE omits explicit environmental
inputs and the capacity penalty. Hypocotyl models
also receive each plant's observed 0–36 h lengths and presence masks to forecast
later individual lengths.

The pinned submission snapshot contains the manuscript experiments. The complete
development history and exploratory experiments are archived at
[development commit `9671ca4`](https://github.com/HwijaeSon/Plant_height/tree/9671ca42605ad6ab1543c527daa6466fa3e80e06).
This snapshot contains code, data, configurations, checkpoints, and machine-readable
results, with README documentation for reproduction. The final configuration is defined by [`configs/paper.json`](configs/paper.json).
The current code-and-data submission version is tagged `submission-20260911-logistic-forecast`. Verification
results are recorded in [`submission/validation.json`](submission/validation.json).

## Current manuscript configuration

For hypocotyls, the manuscript predicts individual length from genotype, elapsed
time, and the masked 0–36 h prefix, using an individual r/K head, **lambda_ODE=100**, and **lambda_K=0**. The public
`phytoode` model name in `run.py` loads and trains this 1,183-parameter model.
The five baselines are Latent ODE, Logistic-PINN, LSTM-NN, Random forest, and
Logistic ODE. Their results and all three temperature-dataset experiments are
unchanged. The original manuscript snapshot remains available at
`submission-20260911-prefix-forecast`. Historical variants are excluded from
the current six-model tables and figures.

The initial ten-coefficient validation search selected 100. A later expanded
search selected 10,000 by validation; the manuscript retains 100 after review
of that search and its test results. This retention is not the expanded-search
validation optimum or independent test confirmation. Complete search records are
preserved below and in the linked reports.

[Current train/validation/test tables](hypocotyl/reports/adopted_lambda100_20260911/comparison.md) ·
[Checkpoint and retention record](hypocotyl/reports/adopted_lambda100_20260911/adoption.json)

<details>
<summary>Archived follow-up experiments and coefficient-search history</summary>

## Follow-up: individual logistic parameters from the observed prefix

The follow-up tagged `experiment-20260911-prefix-parameters` conditions the
PhytoODE r/K head on genotype **and** the observed 0–36 h lengths and masks,
sets `lambda_K=0`, and searches only `lambda_ODE`. The preceding manuscript
snapshot is available at `submission-20260911-prefix-forecast`. Current manuscript
commands below use the retained no-light coefficient 100 in `configs/paper.json`. This follow-up uses the separate
`configs/hypocotyl_prefix_parameters_20260911.json` profile and commands here.

For the **light-input variant**, ten coefficients (0.01–1000) were compared with three seeds using 48-h validation.
The selected coefficient is **100**. It was transferred unchanged to both
additional-missingness conditions. Only PhytoODE was retrained; all baseline
scores are retained from the preceding experiment. The new model has 1,231
parameters, including the individual parameter head.

| Additional prefix removal | Previous PhytoODE test RMSE (mm) | New PhytoODE test RMSE (mm) | New relative RMSE |
|---|---:|---:|---:|
| Natural missingness | 1.703 | 1.280 ± 0.052 | 17.15 ± 0.70% |
| 25% | 1.702 | 1.299 ± 0.187 | 17.40 ± 2.51% |
| 50% | 1.575 | 1.422 ± 0.166 | 19.06 ± 2.22% |

The changes improve mean test error relative to the preceding PhytoODE in all
three conditions. The no-light latent ODE has lower mean test error under natural
missingness and 25% removal; Logistic-PINN has the lowest mean at 50% removal.
This is a combined model/regularization/search follow-up on the same dataset,
with coefficient selection using validation only.

```bash
# Access the follow-up from a checkout of the submission tag.
git fetch origin tag experiment-20260911-prefix-parameters
git checkout experiment-20260911-prefix-parameters

# Evaluate the bundled, validation-selected follow-up model.
python hypocotyl/code/prefix_parameter_trial.py evaluate \
  --checkpoint hypocotyl/results/prefix_parameters_20260911/selected/drop_0/seed1/checkpoint.pt \
  --output outputs/individual-parameter-evaluation

# Reproduce its coefficient search and missingness evaluation (36 training fits).
python hypocotyl/code/run_prefix_parameter_search.py --gpus 0 1 2 \
  --output outputs/individual-parameter-search

# Plot the bundled search, individual trajectories, and baseline comparison.
python hypocotyl/code/report_prefix_parameters.py \
  --output outputs/individual-parameter-report --figures outputs/individual-parameter-figures
```

[Full protocol and reproduction details](hypocotyl/reports/prefix_parameters_20260911/README.md) ·
[Train/validation/test tables and coefficient search](hypocotyl/reports/prefix_parameters_20260911/comparison.md)

### Removing the light input

The second variant removes illumination from both the prefix encoder and latent
vector field, while retaining the initial-length r/K head and `lambda_K=0`.
The same ten-coefficient validation search independently selects **lambda_ODE=100**.
This equals the predefined fixed reference from the light-input variant, so the
fixed-coefficient and separately selected comparisons coincide. The no-light
model has **1,183 parameters**. Time remains an input; all plants share the same
12L12D cycle.

Test cells below are **RMSE (mm) / relative RMSE (%)**, averaged across three seeds.
The complete report includes standard deviations and train/validation scores.

| Model | Natural missingness | +25% prefix removal | +50% prefix removal |
|---|---:|---:|---:|
| PhytoODE, prefix r/K, with light | 1.280 / 17.15% | 1.299 / 17.40% | 1.422 / 19.06% |
| PhytoODE, prefix r/K, without light | 1.150 / 15.41% | 1.317 / 17.65% | **1.224 / 16.40%** |
| Previous no-light latent ODE | **1.143 / 15.31%** | **1.235 / 16.55%** | 1.292 / 17.32% |
| Previous Logistic-PINN | 1.336 / 17.90% | 1.277 / 17.11% | 1.248 / 16.72% |

Omitting illumination reduces PhytoODE's mean test error under natural missingness
and 50% additional removal, but slightly increases it at 25%. The no-light
PhytoODE has the lowest mean at 50% removal among the models in that comparison. The
differences from the strongest baselines are small relative to seed/mask variation;
these results do not establish consistent or statistically significant superiority.
All six baselines and the preceding genotype-head PhytoODE are reused unchanged.

```bash
# Evaluate the bundled no-light model on CPU.
python hypocotyl/code/no_light_prefix_trial.py evaluate \
  --checkpoint hypocotyl/results/prefix_parameters_no_light_20260911/evaluated/lambda_100/drop_0/seed1/checkpoint.pt \
  --output outputs/no-light-prefix-evaluation

# Reproduce the 30-fit search, six transfer fits, and frozen test evaluations.
python hypocotyl/code/run_no_light_prefix_search.py --gpus 0 1 2 \
  --output outputs/no-light-prefix-search

# Plot both feature variants, all baselines, and the coefficient searches.
python hypocotyl/code/report_no_light_prefix.py \
  --output outputs/no-light-prefix-report --figures outputs/no-light-prefix-figures

# Verify training objectives, masking, source hashes, and all nine checkpoints.
python hypocotyl/code/audit_no_light_prefix.py
```

[No-light protocol](hypocotyl/reports/prefix_parameters_no_light_20260911/README.md) ·
[All-model train/validation/test comparison](hypocotyl/reports/prefix_parameters_no_light_20260911/comparison.md)

### Expanded coefficient search

The subsequent `experiment-20260911-expanded-lambda` follow-up broadens the
no-light model's `lambda_ODE` grid from **0.01–1,000 to 0.001–10,000**, adding
resolution near 0.1 and 100. The architecture, prefix r/K head, `lambda_K=0`,
initialization and 1,500-epoch schedule remain unchanged. Twelve additional
coefficients are trained with three seeds (36 new fits), and the 30 previous
validation-only fits are reused. All **22 candidates** are ranked by 48-h
validation RMSE; the selected coefficient is shared across missingness settings.

The expanded grid selects **lambda_ODE=10,000** at its upper bound. Natural-missingness
48-h validation RMSE decreases from **0.5807 to 0.5726 mm** (three-seed means),
but 60/72-h test error increases in every missingness condition:

| Additional prefix removal | Previous lambda=100 | Expanded selection lambda=10,000 |
|---|---:|---:|
| Natural missingness | **1.150 / 15.41%** | 1.579 / 21.16% |
| 25% | **1.317 / 17.65%** | 1.524 / 20.43% |
| 50% | **1.224 / 16.40%** | 1.549 / 20.76% |

Cells are RMSE (mm) / relative RMSE (%), averaged across three seeds. The
expanded validation search therefore does not yield improved test forecasts.
Both configurations and their results remain available; the selected coefficient
is recorded according to validation, without retrospective test-based reranking.
The upper-bound selection also leaves the optimum beyond the sampled range unresolved.

```bash
git fetch origin tag experiment-20260911-expanded-lambda
git checkout experiment-20260911-expanded-lambda

# Reproduce the extended search, using the bundled original candidates.
python hypocotyl/code/run_expanded_no_light_search.py --gpus 1 2 \
  --output outputs/expanded-no-light-search

# Export the bundled expanded search, all-model tables and prediction plots.
python hypocotyl/code/report_expanded_no_light.py \
  --output outputs/expanded-no-light-report --figures outputs/expanded-no-light-figures

# Evaluate its selected checkpoint on CPU.
python hypocotyl/code/expanded_no_light_trial.py evaluate \
  --checkpoint hypocotyl/results/expanded_no_light_lambda_20260911/selected/drop_0/seed1/checkpoint.pt \
  --output outputs/expanded-no-light-evaluation
```

[Expanded-search protocol](hypocotyl/reports/expanded_no_light_lambda_20260911/README.md) ·
[All candidates and train/validation/test results](hypocotyl/reports/expanded_no_light_lambda_20260911/comparison.md)

</details>

## Installation

```bash
git clone --depth 1 --branch submission-20260911-logistic-forecast \
  https://github.com/HwijaeSon/Plant_height.git
cd Plant_height
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If GitHub requires authentication for this repository, use an account with
repository access; pushing this snapshot does not change repository visibility.

The verified environment uses Python 3.12 and PyTorch 2.9.0; package versions are
pinned in `requirements.txt`. CPU execution is supported. For a CPU-only PyTorch
installation, install `torch==2.9.0` from the
[official CPU wheel index](https://download.pytorch.org/whl/cpu) before installing
the remaining requirements. GPU execution requires a compatible CUDA-enabled
PyTorch installation. The reported models were trained with CUDA; changes in
hardware or numerical libraries can change retraining results.

All commands below run from the repository root. New outputs go under the ignored
`outputs/` directory. A run refuses to overwrite an existing output directory.

## Quick start: our hypocotyl dataset

The author-collected data and the selected checkpoints are included; no external
download is needed for this example.

```bash
# Verify individual trajectories, missingness, and temporal partitions.
python scripts/prepare_hypocotyl.py

# Evaluate the submitted PhytoODE checkpoint on CPU.
python run.py evaluate --dataset hypocotyl --seed 1 \
  --output outputs/hypocotyl-evaluation

# Check forward propagation, physics derivatives, and backpropagation.
python run.py smoke --dataset hypocotyl --output outputs/hypocotyl-smoke

# Retrain the selected configuration for its complete 1,500-epoch schedule.
python run.py train --dataset hypocotyl --seed 1 --device cuda:0 \
  --output outputs/hypocotyl-training
```

Replace `--device cuda:0` with `--device cpu` to train without a GPU. Smoke runs
perform two optimizer steps, evaluate training/validation only, and are explicitly
labelled **not paper results**. They retain the full learning-rate schedule.

## Datasets and evaluation protocols

| Dataset | Included observations | Evaluation unit | Environment | Reported height unit |
|---|---|---|---|---|
| Wheat | 19 genotypes across four field seasons | Train: 2018–2019; validation: 2022; test: 2021 | Air temperature | m |
| Maize | 402 genotypes, 3,072 plot trajectories, 31,894 measurements | Train: 2018–2019; validation: 2020; test: 2021 | Air temperature | Relative UAV height |
| Arabidopsis stem length | 9 genotypes, two temperature regimes, 180 plants, four measurements per plant | Plants 1–6: train; 7–8: validation; 9–10: test | Temperature regime | cm |
| Author-collected hypocotyl length | 157 plants, 942 retained measurements, five genotypes | Input/train: 0–36 h (551); validation: 48 h (128); test: 60/72 h (263) | 12L12D at 23°C; no environment input to PhytoODE | mm |

All evaluated genotype identities occur during training. The hypocotyl experiment
uses **12L12D only, individual longitudinal plants, and future-time holdout**.
The contributor confirmed that genotype and Excel row identify the same plant
across times. Missing lengths are neither averaged into genotype targets nor
imputed. A plant needs at least one observed 0–36 h value to be included.
Details and the data dictionary are in [`hypocotyl/README.md`](hypocotyl/README.md).

### Download the published datasets

```bash
python scripts/download_data.py --dataset all
python scripts/prepare_data.py --dataset all
```

To prepare one dataset, replace `all` with `wheat`, `maize`, or `arabidopsis` in both
commands. To verify existing downloads without fetching anything:

```bash
python scripts/download_data.py --dataset all --verify-only
```

The downloader obtains the exact upstream inputs used in this study. Individual
data files are checked against SHA-256 values in
[`configs/data_sources.json`](configs/data_sources.json). Supplement archives are
containers: their extracted workbook contents, rather than potentially variable
ZIP metadata, are verified. Existing mismatched files are not overwritten.

- **Wheat:** the aligned phenotype/environment CSV and kinship table come from
  the [Shao et al. companion repository](https://github.com/YingjieShao/PINN_for_plant_height_forecasting/tree/3da92f51f42d3fde5e06f6fc8ce8f233490a389c),
  pinned to commit `3da92f51f42d3fde5e06f6fc8ce8f233490a389c`.
  These are the processed benchmark inputs, not raw ETH sensor data.
  The model uses one-hot genotype encoding; the existing loader also reads the
  kinship table. See [Shao et al., DOI 10.1016/j.compag.2026.111988](https://doi.org/10.1016/j.compag.2026.111988)
  for the reference protocol. Loading performs the published alignment, cropping,
  split, and normalization; no additional preprocessing command is required.
- **Maize:** supplementary workbooks come from
  [Sweet et al., DOI 10.1111/tpj.17092](https://doi.org/10.1111/tpj.17092)
  via [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11629746/supplementaryFiles).
  Station temperature files are downloaded from the
  [author repository](https://github.com/HirschLabUMN/WiDiv_Drone_Height/tree/8fed4415f99f7ee4f8e43fb088319622bedc81a3),
  pinned to commit `8fed4415f99f7ee4f8e43fb088319622bedc81a3`.
  The related original deposit is [DRUM, DOI 10.13020/SKJN-QX31](https://doi.org/10.13020/SKJN-QX31).
  The submission downloader fetches tabular inputs needed by preprocessing, not
  UAV rasters. The reported response has relative UAV-height units; it must not
  be interpreted as centimetres or metres.
- **Arabidopsis stem length:** the supplementary workbook from
  [Ebrahimi Naghani et al., DOI 10.1186/s12870-024-05394-w](https://doi.org/10.1186/s12870-024-05394-w)
  is obtained through [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11285529/supplementaryFiles).
  `prepare_data.py` extracts the primary inflorescence stem-length sheets into a
  720-row table. This dataset is distinct from the hypocotyl dataset collected here.

External raw and processed inputs are excluded from this submission snapshot and
restored by the commands above. Their original terms and attribution requirements
apply; see [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Train PhytoODE or evaluate the submitted models

The same interface supports all four datasets:

```bash
python run.py train --dataset wheat --seed 1 --device cuda:0 \
  --output outputs/wheat-training
python run.py train --dataset maize --seed 1 --device cuda:0 \
  --output outputs/maize-training
python run.py train --dataset arabidopsis --seed 1 --device cuda:0 \
  --output outputs/arabidopsis-training
```

Use seeds `1`, `2`, and `3` in separate output directories for the manuscript's
three-seed evaluation. The default model is `phytoode`; `--model latent_ode` sets
both physics coefficients to zero. For hypocotyls, `latent_ode` shares the
predictive backbone, inputs, initialization, and optimizer settings; its unused
auxiliary head has 64 fewer parameters than PhytoODE.

| Dataset | λ_ODE | λ_K | Epochs | Parameters |
|---|---:|---:|---:|---:|
| Wheat | 3.16227766 | 0.1 | 1,500 | 1,655 |
| Maize | 0.5 | 0.5 | 3,000 | 9,003 |
| Arabidopsis stem length | 0.5 | 0.1 | 1,500 | 4,607 |
| Hypocotyl length | 100 | 0 | 1,500 | 1,183 |

Training uses the paper's architecture, optimizer, learning-rate schedule,
gradient clipping, and dataset-specific validation checkpoint rule. It saves the
selected checkpoint and its selection record before evaluating test targets.
`history.csv`, `checkpoint.pt`, `selection.json`, `predictions.npz`, and
`result.json` are written to the chosen directory. For temperature datasets,
reported training metrics use replicate-averaged curves, while fitting uses
individual plot/plant trajectories.

Evaluate bundled checkpoints without retraining:

```bash
python run.py evaluate --dataset maize --seed 1 --device cpu \
  --output outputs/maize-evaluation
```

Evaluate a newly trained checkpoint:

```bash
python run.py evaluate --dataset wheat --seed 1 \
  --checkpoint outputs/wheat-training/checkpoint.pt \
  --output outputs/wheat-reloaded
```

Prediction arrays are stored on each model's integration grid in training units.
The `scale`/`height_scale_mm` definitions in the data loaders convert them to the
reported physical or relative-height unit. `result.json` metrics are already in
the reported units. These examples run on the supplied study datasets; adapting
to a new experiment requires a data loader and a new validation design.

## Baselines and result analysis

[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) describes baseline commands, the
manuscript-to-code mapping, and the distinction between re-evaluating saved
predictions and retraining models. Wheat LSTM-NN and Logi-PINN table entries are
the reference authors' reported outputs, not newly trained replacements.

```bash
# Export comparison results as CSV and Markdown.
python scripts/report_results.py --output outputs/comparison-tables

# Regenerate the manuscript prediction figures after external data preparation.
python scripts/make_figures.py --output outputs/prediction-figures

# Verify the checksums of the submitted code, data, configurations, and results.
python scripts/verify_submission.py
```

Expected PhytoODE test scores are mean ± sample standard deviation over seeds 1–3:

| Dataset | RMSE | Relative RMSE |
|---|---:|---:|
| Wheat | 0.03032 ± 0.00066 m | 10.24 ± 0.22% |
| Maize | 56.68 ± 4.37 relative UAV-height units | 18.40 ± 1.42% |
| Arabidopsis stem length | 2.800 ± 0.016 cm | 14.06 ± 0.08% |
| Hypocotyl length | 1.150 ± 0.149 mm | 15.41 ± 2.00% |

RMSE is averaged over trajectories. Relative RMSE is `100 × mean trajectory
RMSE / mean scored target`, calculated separately for each partition; it is not
MAPE. The denominator is shared by all models within a dataset and partition.
Hypocotyl targets are observed individual lengths; primary RMSE averages per-plant
masked RMSE, and pooled observation-level RMSE is recorded separately. Natural
missingness results give PhytoODE the lowest mean 48-h validation error and a
60/72-h test error close to the latent ODE. PhytoODE has the lowest
mean test error at 50% additional prefix removal among the evaluated predictors.

## Missing-prefix robustness

Every hypocotyl baseline was retrained after removing 25% or 50% of the available
0–36 h values from both inputs and training targets. Masks are paired across
models, nested by severity, and retain at least one value per plant. The total
missing fraction is 12.3% naturally, 34.2% after 25% removal, and 56.2% after 50%
removal. The 48/60/72 h targets remain unchanged. Scaling uses retained training
observations only. The retained coefficient 100 is shared across missingness
levels; checkpoint epochs use 48-h validation.

```bash
python run.py train --dataset hypocotyl --model phytoode --seed 1 \
  --drop-fraction 0.5 --device cuda:0 --output outputs/hypocotyl-missing
python run.py evaluate --dataset hypocotyl --model latent_ode --seed 1 \
  --drop-fraction 0.5 --output outputs/hypocotyl-matched-evaluation
python hypocotyl/code/report_adopted_hypocotyl.py \
  --output outputs/forecast-report --figures outputs/forecast-figures
```

With 50% additional removal, PhytoODE's test RMSE is **1.224 ± 0.169 mm**,
versus **1.292 ± 0.357 mm** for the latent ODE and **1.248 ± 0.118 mm**
for Logistic-PINN. It has the lowest mean in this condition, but the differences
are modest relative to seed/mask variation and are not consistent across all
three paired runs. Natural missingness and 25% additional removal favor the
latent ODE by mean test error. Full model/condition tables are in
`hypocotyl/reports/adopted_lambda100_20260911/comparison.md`.

## Interpretation and reproducibility

- Wheat retains the reference initial-fill mask: 475 of 1,368 scored test
  positions precede the first actual observation. The other datasets use observed
  targets. Masked zeros are storage placeholders, not imputed training targets.
- The three temperature experiments use the known environmental sequence without
  target-trajectory phenotype inputs. Hypocotyl forecasting uses early observed
  individual lengths, masks, and elapsed time; future phenotypes never enter
  the encoder. All evaluated genotypes are represented during training.
- The earlier coefficient studies reused inspected test partitions. The maize
  `(0.5, 0.5)` setting was retained after a later search and was not that search's
  validation optimum. Temperature-dataset controls were not independently retuned.
- The current hypocotyl experiment changes the evaluation task and target unit.
  Previous genotype-mean scores cannot be compared numerically with these
  individual-plant future-time errors. Earlier code and results remain available
  at Git tag `submission-20260911-code-data`.
- Hypocotyl coefficient 100 was selected in the initial grid and retained after
  reviewing a wider search and its test results. The wider grid's validation
  winner was 10,000. The latent ODE shares the predictive backbone and
  initialization; PhytoODE adds 64 auxiliary parameter-head weights. Baselines
  were not independently retuned.
- Standard deviations reflect three initialization seeds under natural
  missingness and three paired mask/initialization realizations after additional
  removal. They are not biological-cohort confidence intervals. Bundled
  checkpoints preserve the reported runs; retraining on different hardware may
  change numerical results.

## Data and citation

The submitted hypocotyl workbook, tidy observations, and fixed partitions are
available under [`hypocotyl/data`](hypocotyl/data). No DOI has been assigned to
this GitHub snapshot. Cite the accompanying manuscript using
[`CITATION.cff`](CITATION.cff), and cite each original external dataset when using
it. This snapshot does not assign an additional open-source or data-reuse licence;
upstream rights are retained and enquiries about reuse should be directed to the
corresponding author.
