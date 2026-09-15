# Reproduction guide

`run.py` is the submission entry point. It retains the model implementations used
in the reported runs and provides a shared CPU/GPU interface for fixed-setting
training and checkpoint evaluation. The current release manifest records the retained code, data, and fitted models.

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
| Hypocotyl latent dynamics and individual prefix-conditioned r/K head | `hypocotyl/code/no_light_prefix_model.py`, `prefix_parameter_model.py` |
| Hypocotyl prefix, temporal targets, missing masks, and metrics | `hypocotyl/code/forecast_data.py` |
| Fixed-configuration training / selected checkpoint evaluation | `run.py` |
| Manuscript comparison tables | `results/comparison_temperature.csv`, `hypocotyl/reports/four_genotypes_20260915/comparison.csv` |
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
the architecture and random initialization sequence. For hypocotyls the latent ODE shares the retained PhytoODE predictive
backbone, inputs and initialization, but its unused auxiliary head has 64 fewer
parameters.

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

The current release uses only Col-0, hy5, MLB, and phyAB. The released workbook
and processed cohort contain no additional genotypes. All six models were
retrained with the fixed manuscript settings, and the genotype dimensions follow
the approved label list in `forecast_data.py`.

```bash
python scripts/prepare_hypocotyl.py
python hypocotyl/code/run_approved_benchmark.py --gpus 0 \
  --output outputs/four-genotype-benchmark
python hypocotyl/code/report_approved_benchmark.py \
  --results outputs/four-genotype-benchmark \
  --output outputs/retrained-report --figures outputs/retrained-figures
```

This runs 52 fits across the six methods, three seeds, and three missingness
conditions. Classical baselines run on CPU. Only GPUs listed in `--gpus` are
used. A frozen plan records the code, data configuration, and run settings.
Completed jobs can be resumed; incomplete directories require inspection before
restarting.

For a single model, use `run.py train --dataset hypocotyl --model MODEL` with a
new `--output` directory. Models are `phytoode`, `latent_ode`, `logistic_pinn`,
`lstm`, `rf`, and `logistic`. Use seeds 1–3 and `--drop-fraction` 0, 0.25, or 0.5.
The default mask seed is `20260910 + training_seed`. All models share each mask,
and at least one observed initial length is retained per plant.

PhytoODE training writes a validation-selected checkpoint and scores training
and validation only. Use `run.py evaluate` with `--checkpoint` to score the
future test observations. The full benchmark runner performs both stages. The
other predictors similarly select or fit using training/validation before final
scoring. `configs/paper.json` records bundled checkpoint paths and checksums.

## Verification and reporting

```bash
python scripts/verify_submission.py
python hypocotyl/code/audit_approved_benchmark.py
python scripts/report_results.py --output outputs/tables
python scripts/make_figures.py --output outputs/figures
```

The hypocotyl audit verifies allowed genotype labels, source-cell extraction,
shared and nested masks, cohort sizes, selected checkpoint hashes, and RMSEs
recomputed from saved predictions. The plotter derives missingness percentages
from the current cohort and draws one plant for each approved genotype. The
three temperature-dataset figure implementations and recorded results are
unchanged.

## Numerical reproducibility

Neural runs use the recorded seeds, Adam, cosine learning-rate decay, gradient
clipping, and validation-based checkpoint selection. CUDA and CPU kernels may
produce different initial weights or optimization trajectories. Use the bundled
checkpoints to evaluate the recorded fits on another machine. Standard
deviations describe initialization and paired-mask variation, not independent
biological cohorts.
