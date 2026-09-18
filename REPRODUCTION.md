# Reproduction guide

## Implementation map

| Component | File |
|---|---|
| Genotype embedding, latent vector field, RK4, decoder, chain-rule phenotype derivative | `src/wheat/model.py` |
| Temperature task setup, original scaling and metrics | `src/temperature.py` |
| Wheat data alignment, year splits and reference scoring masks | `src/wheat/data.py` |
| Maize extraction, temperature and cohort preparation | `src/maize/prepare_data.py`, `data.py` |
| Arabidopsis stem measurements, partitions and reference architectures | `src/arabidopsis/prepare_data.py`, `run_experiment.py`, `models.py` |
| Hypocotyl extraction, plant identity, observation masks and splits | `src/hypocotyl/forecast_data.py` |
| Hypocotyl neural model and fixed normalization | `src/hypocotyl/selected_model.py`, `phase_model.py`, `architecture.py` |
| Light-modulated logistic RHS | `src/hypocotyl/growth_law.py` |
| Hypocotyl reference models | `src/hypocotyl/forecast_models.py` |
| Training and checkpoint evaluation | `run.py`, `src/hypocotyl/selected_trial.py`, `forecast_trial.py` |
| Manuscript tables | `results/comparison_temperature.csv`, `comparison_hypocotyl.csv` |
| Six manuscript figures | `src/visualization/temperature.py`, `hypocotyl.py` |

Model configurations are provided in `src/configs/`. Checkpoint hashes and
experiment metadata identify the fitted models used for the reported results.

## Model and metric conventions

Temperature PhytoODE uses `(lambda_ode, lambda_k)` of `(3.16227766, 0.1)` for
wheat, `(0.5, 0.5)` for maize, and `(0.5, 0.1)` for Arabidopsis stems. Its matched
unregularized model sets both coefficients to zero. The maize capacity-only
ablation sets only `lambda_ode=0`; evaluate it with `--model capacity_only`.

Hypocotyl PhytoODE uses `(300, 0)`, a genotype-specific mean rate and light/dark
contrast, and an individual capacity head conditioned on genotype and the masked
initial observations. Its otherwise identical control uses `(0, 0)`. The decoder
Jacobian times the latent ODE vector field gives the phenotype derivative;
residuals use 18 nonswitching collocation points on the 3-hour integration grid.
The 12-hour measurement grid is not filled with artificial targets. Additional
25% stochastic encoder-input dropout is used during training in both models;
this is separate from the fixed experimental observation-removal masks.

RMSE is the mean of masked per-trajectory/plant RMSEs. Relative RMSE is
`100 * RMSE / mean(observed scored target)`; it is not MAPE. Field benchmark
training uses plot measurements, while their reported scores use the retained
replicate-averaged genotype/year curves. Arabidopsis stem scores likewise retain
the original group-mean scoring protocol. Hypocotyl scores always use individual
observed lengths. Original wheat reference summaries retain their published
rounding; reference trajectory files retain the source evaluation conventions.

## Fitted-model availability

`checkpoints/inventory.csv` lists the available checkpoints, forests, process fits and
prediction-only references. The PhytoODE models, all matched latent controls,
maize LSTM/PINN, wheat/maize forests, and all hypocotyl models have fitted objects.
The process ODEs have parameter JSON files. The Arabidopsis process parameters
were saved at float32 precision, so regenerated predictions may differ slightly
from the original double-precision fit.

Wheat LSTM-NN/Logi-PINN comparisons use the reference authors' summaries and
predictions. Original fitted weights are unavailable for these models and for
the Arabidopsis stem LSTM-NN, Logi-PINN and RF comparisons. Their saved predictions
support reproduction of the reported comparisons; predictions for new inputs
require retraining with the implementations described below.

## Train reference baselines

Prepare each dataset first. Commands below use fresh output directories.

Wheat Logistic ODE, Temperature ODE and RF:

```bash
python src/wheat/run_additional_baselines.py --output outputs/wheat-baselines
```

For the authors' wheat LSTM/Logi-PINN implementation, fetch the exact companion
repository revision and follow its `MultipleGenotypeModel.py` / `run_sbatch.py`
workflow with split 0 and one-hot genotype encoding:

```bash
git clone https://github.com/YingjieShao/PINN_for_plant_height_forecasting.git outputs/wheat-reference-source
git -C outputs/wheat-reference-source checkout 3da92f51f42d3fde5e06f6fc8ce8f233490a389c
```

The reported wheat reference values are taken from the authors' archived
summaries. Retraining can produce different values because of numerical and
environmental differences.

Maize (fit process models first; PINN uses the fitted logistic initialization):

```bash
python src/maize/run_experiment.py --models process --seed 1 --device cpu \
  --output outputs/maize-baselines
python src/maize/run_experiment.py --models rf lstm pinn --seed 1 --device cuda:0 \
  --output outputs/maize-baselines
```

Repeat the second command for seeds 2 and 3. The original reference schedule is
3,000 epochs. Use root `run.py` for PhytoODE and its matched control.

Arabidopsis stem baselines:

```bash
python src/arabidopsis/run_experiment.py --skip-latent --device cuda:0 \
  --seeds 1 2 3 --reference-epochs 3000 --physics-warmup-epochs 500 \
  --ode-lr 1e-4 --output outputs/arabidopsis-baselines
```

Hypocotyl reference models are `logistic_pinn`, `lstm`, `rf`, and `logistic`:

```bash
python run.py train --dataset hypocotyl --model logistic_pinn --seed 1 \
  --device cuda:0 --output outputs/hypocotyl-pinn
python run.py evaluate --dataset hypocotyl --model logistic_pinn --seed 1 \
  --output outputs/hypocotyl-pinn-evaluation
python run.py evaluate --dataset hypocotyl --model rf --seed 1 \
  --output outputs/hypocotyl-rf-evaluation
```

Reference settings are in `src/configs/hypocotyl_forecast.json`. Repeat for seeds
1–3 and `--drop-fraction 0`, `0.25`, `0.5`; natural Logistic ODE needs only seed 1.
The PhytoODE/control profiles are in `src/configs/hypocotyl_phytoode.json` and
`hypocotyl_latent_ode.json`, with seeds 101–105 at each removal fraction. All
masks use `20260910 + seed` and retain at least one initial observation per plant.
The experiment comprises 34 reference fits and 30 PhytoODE/control fits.
PhytoODE and its control share the five initializations and removal masks;
reference baselines use a separate set of three seeds.

## Evaluate pretrained temperature baselines

```bash
python src/scripts/evaluate_baselines.py --dataset wheat --model rf --seed 1 \
  --output outputs/wheat-rf-evaluation
python src/scripts/evaluate_baselines.py --dataset maize --model pinn --seed 1 \
  --output outputs/maize-pinn-evaluation
python src/scripts/evaluate_baselines.py --dataset arabidopsis --model temperature \
  --output outputs/arabidopsis-temperature-evaluation
```

Wheat supports `logistic`, `temperature`, `rf`; maize additionally supports
`lstm`, `pinn`; Arabidopsis supports the two saved process fits. Unsupported
original fitted objects are reported explicitly. Classical fits run on CPU.
Hypocotyl evaluation uses the root `run.py` for all six methods.

## Numerical verification

After downloading and preparing external data:

```bash
python src/scripts/verify_submission.py
python src/scripts/verify_models.py --dataset all --output outputs/verified-models
python src/scripts/report_results.py --output outputs/tables
python src/scripts/make_figures.py --output outputs/figures
```

The checkpoint check compares predictions with the archived arrays using
float32 tolerances. The manifest protects every distributed file except itself.
CPU and CUDA implementations can differ in initialization and optimization.
Evaluation of the supplied checkpoints reproduces the reported fitted models
without retraining.
