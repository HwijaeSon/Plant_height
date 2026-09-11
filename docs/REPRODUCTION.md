# Reproduction guide

`run.py` is the submission entry point. It retains the model implementations used
in the reported runs and provides a shared CPU/GPU interface for fixed-setting
training and checkpoint evaluation. The archived development commit is recorded
in `submission/source_files.json`; it contains the complete search histories.

## Manuscript-to-code mapping

| Manuscript component | Implementation / record |
|---|---|
| Latent state, environmental encoder, genotype embedding, RK4, decoder | `wheat/code/model.py` |
| Chain-rule phenotype derivative and logistic penalties | `wheat/code/model.py: LatentODEHeightModel.dydt`, `physics_losses` |
| Wheat alignment, initial-fill mask, year partition | `wheat/code/data.py` |
| Maize cohort extraction and station temperatures | `maize/code/prepare_data.py`, `maize/code/data.py` |
| Arabidopsis stem-length preprocessing and plant partition | `arabidopsis/code/prepare_data.py`, `arabidopsis/code/run_experiment.py` |
| Temperature-dataset architectures and training schedules | `experiments/physics_ablation.py: setup`, `configs/paper.json` |
| Hypocotyl ordinary logistic reference and baseline models | `hypocotyl/code/forecast_models.py` |
| Hypocotyl no-light dynamics and individual prefix-conditioned r/K head | `hypocotyl/code/no_light_prefix_model.py`, `prefix_parameter_model.py` |
| Hypocotyl prefix, temporal targets, missing masks, and metrics | `hypocotyl/code/forecast_data.py` |
| Fixed-configuration training / selected checkpoint evaluation | `run.py` |
| Four manuscript tables | `results/comparison_temperature.csv`, `hypocotyl/reports/adopted_lambda100_20260911/comparison.csv` |
| Temperature prediction figures | `visualization/temperature.py` |
| Hypocotyl prediction figure | `visualization/hypocotyl.py: make_figure` |

`scripts/make_figures.py` generates plots under the requested output directory. The first three
figure panels display PhytoODE and the five baseline families; the unregularized
latent ODE is in their tables. The hypocotyl figure also displays its unregularized
latent ODE, as in the final manuscript.

## Baseline training

Run these commands after installing dependencies and preparing the relevant
dataset. Use fresh output directories. CUDA indices refer to visible devices;
for example, `CUDA_VISIBLE_DEVICES=2 ... --device cuda:0` uses physical GPU 2.

### Matched unregularized latent ODE

```bash
python run.py train --dataset wheat --model latent_ode --seed 1 \
  --device cuda:0 --output outputs/wheat-latent-ode
```

Use `maize` or `arabidopsis` for the corresponding matched architecture, and repeat
with seeds 2 and 3. Both `lambda_ode` and `lambda_k` are zero; optimizer weight
decay remains unchanged. The auxiliary parameter head is retained to preserve
the architecture and random initialization sequence. For hypocotyls the no-light latent ODE shares the retained PhytoODE predictive
backbone, inputs and initialization, but its unused auxiliary head has 64 fewer
parameters. `latent_ode_light` is a separate illumination-input variant.

### Wheat

```bash
python wheat/code/run_additional_baselines.py \
  --output outputs/wheat-process-rf
```

This fits Logistic ODE, Temperature ODE, and RF on the original reference scoring
mask. The LSTM-NN and Logi-PINN table values are the first three seeds of the
reference authors' released summaries, retained with their original precision.
Their source is the [pinned companion repository](https://github.com/YingjieShao/PINN_for_plant_height_forecasting/tree/3da92f51f42d3fde5e06f6fc8ce8f233490a389c).
To retrain those reference methods, follow its `MultipleGenotypeModel.py` /
`run_sbatch.py` workflow with split 0 and one-hot genotype encoding. Such retraining
is distinct from re-exporting the published summary rows used in our table.
`wheat/results/reference/test_predictions_seed1_3_provenance.json` records the
upstream files used in the figure; table summaries and trajectory archives retain
their respective upstream reporting conventions.

### Maize

```bash
python maize/code/run_experiment.py --models process --seed 1 --device cpu \
  --output outputs/maize-baselines
python maize/code/run_experiment.py --models rf lstm pinn --seed 1 --device cuda:0 \
  --output outputs/maize-baselines
```

Repeat the second command with seeds 2 and 3. Process models run once. The default
reference schedule is 3,000 epochs. To reproduce the final PhytoODE use `run.py`,
not the historical `--models latent` option in this internal baseline module.

### Arabidopsis stem length

```bash
python arabidopsis/code/run_experiment.py --skip-latent --device cuda:0 \
  --seeds 1 2 3 --reference-epochs 3000 --physics-warmup-epochs 500 \
  --ode-lr 1e-4 --output outputs/arabidopsis-baselines
```

This fits the process ODEs, RF, LSTM-NN, and Logi-PINN. The submission-only
`--skip-latent` switch skips the superseded latent-model default while leaving
baseline fitting unchanged. Use `run.py` for the final PhytoODE and its control.

### Hypocotyl length

```bash
python run.py train --dataset hypocotyl --model logistic --seed 1 \
  --output outputs/hypocotyl-logistic
python run.py train --dataset hypocotyl --model rf --seed 1 \
  --output outputs/hypocotyl-rf
python run.py train --dataset hypocotyl --model logistic_pinn --seed 1 \
  --device cuda:0 --output outputs/hypocotyl-pinn
python run.py train --dataset hypocotyl --model lstm --seed 1 \
  --device cuda:0 --output outputs/hypocotyl-lstm
python run.py train --dataset hypocotyl --model latent_ode --seed 1 \
  --device cuda:0 --output outputs/hypocotyl-latent-ode
```

Repeat stochastic models with seeds 2 and 3; Logistic ODE is fitted once.
The current models receive each plant's masked 0–36 h prefix. Baseline settings
are frozen in `configs/hypocotyl_forecast.json` and were retained without another
search. Only `latent_ode_light` additionally receives illumination; train it with
the same command, changing `--model`. Both its physics coefficients are zero.
The default PhytoODE uses the no-light prefix-conditioned parameter head and
coefficients `(100, 0)`, resolved through `configs/paper.json`.

For the added-missingness study use `--drop-fraction 0.25` or `0.5`. The default
mask seed is `20260910 + training_seed`; the same values are hidden across models.
All methods are refitted for each mask. Logistic ODE is fitted once naturally and
three times at each added-missingness level because the masks differ.

```bash
python hypocotyl/code/run_forecast_benchmark.py --gpus 0 1 2 \
  --output outputs/retrained-forecast-benchmark
python hypocotyl/code/report_forecast.py \
  --results outputs/retrained-forecast-benchmark \
  --output outputs/retrained-forecast-report --figures outputs/retrained-forecast-figures
```

The commands above reproduce the archived original benchmark, including its
genotype-only, light-input PhytoODE. For the current manuscript PhytoODE, use
`run.py train --dataset hypocotyl --model phytoode` for seeds 1–3 and each
removal level; `report_adopted_hypocotyl.py` exports the recorded current tables
and figures. The archived complete benchmark performs 61 fits. A frozen run plan records code hashes,
models, seeds, and masks; resuming skips completed runs and rejects changes to
the frozen plan. Incomplete run directories require inspection before restarting.
Only GPUs explicitly listed with `--gpus` are used; two workers per GPU are the
default. Classical baselines run on CPU.

## Stored runs and regenerated reports

The submission preserves selected checkpoints and predictions, as well as the
three recorded hypocotyl coefficient-search trees. Their original dated paths are retained to preserve
provenance. `configs/paper.json` resolves the checkpoints used by `run.py`.
Histories are included where present in the original selected runs; no missing
training records have been fabricated.

`scripts/report_results.py` exports the recorded main comparison as CSV and
Markdown, with one table per dataset and bold minima based on unrounded means. It does not rerank models
after a new experiment. New evaluations write their own `result.json` and
`predictions.npz`, which can be compared with the submitted values. The complete
development history remains at the source commit identified in the submission
manifest, including the searches described in Methods.

## Numerical reproducibility

The original neural experiments used three seeds, Adam with cosine learning-rate
decay, gradient clipping at 1, and validation-based checkpoint selection. For
wheat and Arabidopsis, recurrent weights were initialized on the training device;
maize used CPU recurrent initialization before transfer. `run.py` preserves this
ordering. CUDA/CPU kernels can produce different initial weights or optimization
trajectories. Reported standard deviations describe initialization variation (and paired
mask variation in the hypocotyl removal study), not biological cohorts. Use the bundled checkpoints to evaluate the submitted fits on another
machine, and retain the reported precision when comparing results.

## Individual-parameter PhytoODE follow-up

`hypocotyl/code/prefix_parameter_model.py` adds masked initial lengths to the
r/K head. `prefix_parameter_trial.py` trains with `lambda_K=0` and reports only
training/validation until final evaluation. `run_prefix_parameter_search.py`
freezes the ten-coefficient, three-seed search and then transfers the selected
coefficient unchanged to both added-missingness settings. It freezes nine
checkpoint hashes before test evaluation. `report_prefix_parameters.py` compares
the selected models with the unchanged earlier baseline outputs.

The complete protocol, commands, and current scores are in
[`hypocotyl/reports/prefix_parameters_20260911`](../hypocotyl/reports/prefix_parameters_20260911/README.md).
Use the explicitly named follow-up scripts for this experiment; `run.py` uses
the retained no-light coefficient 100 in `configs/paper.json`; the prior
manuscript snapshot remains at tag `submission-20260911-prefix-forecast`.

The additional no-light comparison uses `no_light_prefix_model.py`,
`no_light_prefix_trial.py`, and `run_no_light_prefix_search.py`. Both the prefix
encoder and latent vector field omit illumination, while the prefix-conditioned
r/K head and `lambda_K=0` are retained. The same ten-coefficient grid is searched
using validation only. The validation winner and a predefined coefficient of 100
are evaluated across all missingness conditions, deduplicating identical fits.
Use `report_no_light_prefix.py` to export both feature variants alongside all
unchanged baselines, and `audit_no_light_prefix.py` to verify the artifacts.
See [the no-light protocol](../hypocotyl/reports/prefix_parameters_no_light_20260911/README.md).

The subsequent range extension uses `expanded_no_light_trial.py` and
`run_expanded_no_light_search.py`. The wrapper selects the expanded profile while
calling the same no-light trainer. The runner hashes and reuses the original 30
validation-only fits, adds 36 fits, and ranks all 22 coefficients using 48-h
validation. `report_expanded_no_light.py` exports the merged search and all-model
comparison; `audit_expanded_no_light.py` checks the reused sources, common
initialization, loss, validation selection and final checkpoints. See the
[expanded-search protocol](../hypocotyl/reports/expanded_no_light_lambda_20260911/README.md).
