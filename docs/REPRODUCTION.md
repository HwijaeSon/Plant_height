# Reproduction guide

## Implementation map

| Component | File |
|---|---|
| Genotype embedding, latent vector field, RK4, decoder, chain-rule phenotype derivative | `wheat/code/model.py` |
| Temperature task setup, original scaling and metrics | `experiments/physics_ablation.py` |
| Wheat data alignment, year splits and reference scoring masks | `wheat/code/data.py` |
| Maize extraction, temperature and cohort preparation | `maize/code/prepare_data.py`, `data.py` |
| Arabidopsis stem measurements, partitions and reference architectures | `arabidopsis/code/prepare_data.py`, `run_experiment.py`, `models.py` |
| Hypocotyl extraction, plant identity, observation masks and splits | `hypocotyl/code/forecast_data.py` |
| Final hypocotyl neural model and fixed normalization | `hypocotyl/code/selected_model.py`, `phase_model.py`, `architecture.py` |
| Light-modulated logistic RHS | `hypocotyl/code/growth_law.py` |
| Hypocotyl reference models | `hypocotyl/code/forecast_models.py` |
| Training and checkpoint evaluation | `run.py`, `hypocotyl/code/selected_trial.py`, `forecast_trial.py` |
| Manuscript tables | `results/comparison_temperature.csv`, `comparison_hypocotyl.csv` |
| Six manuscript figures | `visualization/temperature.py`, `hypocotyl.py` |

Only final configurations and selected fits are distributed. Original fit
metadata/checkpoint hashes are retained; paths in historical provenance records
may identify the original research environment and are not runtime dependencies.

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

`submission/models.csv` lists every retained checkpoint, forest, process fit and
prediction-only reference. The PhytoODE models, all matched latent controls,
maize LSTM/PINN, wheat/maize forests, and all hypocotyl models have fitted objects.
The process ODEs have parameter JSON files. The Arabidopsis process parameters
were saved at float32 precision, so regenerated predictions may differ slightly
from the original double-precision fit.

**Unavailable original weights:** wheat LSTM-NN/Logi-PINN rows use the original
authors' released summaries and prediction archives; their fitted weights are
not bundled. Arabidopsis stem LSTM-NN/Logi-PINN/RF training did not save fitted
objects. Their original prediction records and implementations are supplied.
These records reproduce reported results but do not provide a fitted model for
new inputs. We do not replace them with a new fit while claiming it is the
reported checkpoint.

## Train reference baselines

Prepare each dataset first. Commands below use fresh output directories.

Wheat Logistic ODE, Temperature ODE and RF:

```bash
python wheat/code/run_additional_baselines.py --output outputs/wheat-baselines
```

For the authors' wheat LSTM/Logi-PINN implementation, fetch the exact companion
repository revision and follow its `MultipleGenotypeModel.py` / `run_sbatch.py`
workflow with split 0 and one-hot genotype encoding:

```bash
git clone https://github.com/YingjieShao/PINN_for_plant_height_forecasting.git outputs/wheat-reference-source
git -C outputs/wheat-reference-source checkout 3da92f51f42d3fde5e06f6fc8ce8f233490a389c
```

This source is fetched rather than re-licensed or redistributed here. Upstream
reference retraining is distinct from the archived summary rows used in our table.

Maize (fit process models first; PINN uses the fitted logistic initialization):

```bash
python maize/code/run_experiment.py --models process --seed 1 --device cpu \
  --output outputs/maize-baselines
python maize/code/run_experiment.py --models rf lstm pinn --seed 1 --device cuda:0 \
  --output outputs/maize-baselines
```

Repeat the second command for seeds 2 and 3. The original reference schedule is
3,000 epochs. Use root `run.py` for the selected PhytoODE/control configurations.

Arabidopsis stem baselines:

```bash
python arabidopsis/code/run_experiment.py --skip-latent --device cuda:0 \
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

Reference settings are in `configs/hypocotyl_forecast.json`. Repeat for seeds
1–3 and `--drop-fraction 0`, `0.25`, `0.5`; natural Logistic ODE needs only seed 1.
The final PhytoODE/control profiles are in `configs/hypocotyl_phytoode.json` and
`hypocotyl_latent_ode.json`, with seeds 101–105 at each removal fraction. All
masks use `20260910 + seed` and retain at least one initial observation per plant.
This is 34 reference fits and 30 final-model/control fits; the two cohorts of
seeds must not be described as one fully matched six-model experiment.

## Evaluate retained temperature baselines

```bash
python scripts/evaluate_baselines.py --dataset wheat --model rf --seed 1 \
  --output outputs/wheat-rf-evaluation
python scripts/evaluate_baselines.py --dataset maize --model pinn --seed 1 \
  --output outputs/maize-pinn-evaluation
python scripts/evaluate_baselines.py --dataset arabidopsis --model temperature \
  --output outputs/arabidopsis-temperature-evaluation
```

Wheat supports `logistic`, `temperature`, `rf`; maize additionally supports
`lstm`, `pinn`; Arabidopsis supports the two saved process fits. Unsupported
original fitted objects are reported explicitly. Classical fits run on CPU.
Hypocotyl evaluation uses the root `run.py` for all six methods.

## Numerical verification

After downloading and preparing external data:

```bash
python scripts/verify_submission.py
python scripts/verify_models.py --dataset all --output outputs/verified-models
python scripts/report_results.py --output outputs/tables
python scripts/make_figures.py --output outputs/figures
```

The checkpoint check compares predictions with the archived arrays using
float32 tolerances. The manifest protects every distributed file except itself.
No downloads, retraining or plotting overwrite archived results. CPU and CUDA
can differ in initialization and optimization; evaluate retained weights for
the closest reproduction of reported results.
