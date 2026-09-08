# Physics-loss ablation and coefficient tuning

`physics_ablation.py` imports the existing wheat, maize and Arabidopsis data
loaders and latent ODE implementations. It compares the manuscript's fixed
PhytoODE settings with two loss ablations:

| Variant | Report label | Logistic derivative residual | Maximum-height consistency |
|---|---|---|---|
| `full` | PhytoODE | Original coefficient | Original coefficient |
| `no_ode_residual` | PhytoODE (K loss only); partial ablation | Zero | Original coefficient |
| `no_biological_loss` | Latent Neural ODE (no physics); main baseline | Zero | Zero |

The main **Latent Neural ODE without physics loss** baseline is the last row:
`lambda_ODE = lambda_K = 0`. Removing only the ODE residual is a partial ablation
that still contains a physics loss. Both-zero runs already exist for all three
datasets and seeds 1--3; the reporting clarification requires no new training.
Optimizer weight decay is
unchanged. The auxiliary logistic head remains instantiated in every variant
to preserve parameter initialization; it receives no gradients when both
biological losses are absent. All variants have identical prediction
architectures, seed-specific initial weights, data splits, input information,
training budgets and validation-based checkpoint selection rules.

The original implementation initializes the wheat/Arabidopsis LSTM on GPU
and the tuned maize LSTM on CPU before moving it to GPU. These differences
between datasets are preserved and kept identical between variants.

The launcher first reproduces seed 1 of each full model, verifies its selected
epoch and all-split errors against the frozen manuscript results, then
distributes seeds 1--3 of each ablation across the selected GPUs. Test targets
are not used during training or checkpoint selection. There is no new
hyperparameter search for either ablation.

```bash
.venv/bin/python experiments/run_physics_ablation.py \
  --gpus 0 2 3 --output experiments/results/physics_ablation_repeat
.venv/bin/python experiments/summarize_physics_ablation.py \
  --output experiments/results/physics_ablation_repeat
```

Use a new output directory for each independent experiment. Completed run
directories are protected against overwrite. A failed/incomplete run is
preserved for diagnosis; move it aside explicitly before restarting it.

The [2026-09-08 experiment](results/physics_ablation_20260908/README.md)
contains the full train/validation/test RMSE and relative-error comparison,
paired seed differences, a LaTeX table, and a figure. The audit also verifies
checkpoint hashes, initial-weight equality, original scoring denominators,
validation-only selection and agreement between saved predictions and metrics.

The [Korean interpretation](results/physics_ablation_20260908/interpretation_ko.md)
distinguishes the derivative residual from maximum-height regularization and
explains the limits of the observed three-seed differences.

## Validation-only ODE-residual coefficient tuning

`tune_lambda_ode.py` varies only the logistic derivative-residual coefficient.
The maximum-height coefficient, architecture, paired initialization, optimizer,
training length, data splits and original checkpoint selection rule stay fixed.
Wheat and Arabidopsis run concurrently on GPU 2; maize runs on GPU 3. Each
training invocation records only train/validation metrics.

The search screens a fixed positive grid with seed 1, refines its best interval
once, and confirms the top three positive candidates with seeds 2 and 3. The
original coefficient and zero-residual references are reused. Final ranking
uses three-seed mean validation RMSE and includes zero. The best positive
coefficient and the overall winner, which may be zero, are both reported.
All dataset selections are frozen before evaluating the selected positive
coefficient on test once per seed.

```bash
.venv/bin/python experiments/tune_lambda_ode.py \
  --short-gpu 2 --long-gpu 3 --maize-workers 2 \
  --output experiments/results/lambda_ode_repeat
.venv/bin/python experiments/lambda_ode_status.py \
  --output experiments/results/lambda_ode_repeat
.venv/bin/python experiments/summarize_lambda_ode.py \
  --output experiments/results/lambda_ode_repeat
```

Use a fresh output directory. The search records source/data hashes and checks
the original seed-1 training prefix before new trials. The final audit checks
paired initialization, unchanged settings, validation-only selection, checkpoint
hashes and metrics recomputed from saved predictions. Each dataset's
`selected_config.json` stores its selected effective coefficient and checkpoint
references. A single full-length selected-coefficient run can be reproduced with
`lambda_ode_trial.py train --dataset DATASET --seed SEED --lambda-ode VALUE
--output NEW_DIRECTORY`; it produces train/validation results only.

The [2026-09-08 coefficient search](results/lambda_ode_tuning_20260908/README.md)
contains all-split RMSE / relative RMSE tables, validation sensitivity and test
comparison figures, a LaTeX table, selected configurations and an audit. Its
[Korean interpretation](results/lambda_ode_tuning_20260908/interpretation_ko.md)
distinguishes validation selection from observed test performance.
The primary [PhytoODE versus no-physics Latent Neural ODE table](results/lambda_ode_tuning_20260908/physics_vs_latent_ode_table.tex)
uses the both-zero baseline. The full ablation table separately retains the
K-loss-only model and original PhytoODE, including the less favorable maize
tuning results.

The preceding ablation's test scores were already inspected before this search.
These final test scores are follow-up evaluations on reused test sets, not a
new independent confirmation. Three seeds describe initialization variation;
they do not establish statistical significance or uncertainty across years.

`resume_lambda_ode.py --output DIRECTORY` supports recovery from an interrupted
confirmation phase before final test evaluation. It retains completed trials
and frozen finalist lists, preserves incomplete attempts under `diagnostics/`,
and restarts incomplete jobs with their original seeds and full schedules.
The interrupted training prefix must match the replacement run. Recovery has
its own manifest and code hash; it does not modify the original search protocol
or frozen training sources. The 2026-09-08 search required one such recovery.

## Joint tuning of both physics coefficients

`tune_joint_lambdas.py` varies `lambda_ODE` and `lambda_K` together while
preserving the same dataset-specific architecture, initialization, optimizer,
epoch budget and checkpoint rule. It includes both one-loss boundaries and the
both-zero no-physics baseline. Previous validation-only coefficient records and
the existing both-zero controls are reused.

The frozen protocol screens a 6-by-5 Cartesian grid per dataset with seed 1,
adds one four-corner geometric refinement, and confirms the top three
both-positive pairs and the best pair on each one-loss boundary with seeds 2
and 3. Final selection uses three-seed mean validation RMSE. Both the best
strictly positive pair (PhytoODE) and the overall winner are reported. All
three dataset selections are frozen before evaluating either selected pair
on test. The same previously inspected test sets are reused; this is a
follow-up experiment, not independent confirmation.

```bash
.venv/bin/python experiments/tune_joint_lambdas.py \
  --short-gpu 2 --long-gpu 3 --output experiments/results/joint_lambda_repeat
.venv/bin/python experiments/joint_lambda_status.py \
  --output experiments/results/joint_lambda_repeat
# After interruption, rerun the controller with the same output and --resume.
.venv/bin/python experiments/summarize_joint_lambdas.py \
  --output experiments/results/joint_lambda_repeat
```

Wheat and Arabidopsis initially share GPU 2 with one trial each; after
Arabidopsis selection freezes, two wheat trials may run concurrently. Two
maize trials share GPU 3. Before each dataset search, the launcher checks
historical full-model and both-zero training prefixes and verifies that
resumed and uninterrupted 20-epoch states and histories are bitwise identical.
The original full cosine schedule is retained even in these short checks.

`joint_lambda_trial.py` saves the model, optimizer, scheduler, validation window
and all RNG states every 100 epochs and on graceful termination. It can resume
the same unfinished trial with `--resume`, without changing the validation
schedule. Diagnostic tensors and optimizer snapshots stay local; completed
checkpoints, predictions, configurations, histories and selection records are
versioned. The original single-coefficient experiment files remain unchanged.

The [joint-search report](results/joint_lambda_tuning_20260908/README.md)
contains all-split RMSE / relative RMSE tables, a LaTeX table, selected
configurations, validation-grid and paired-test figures, and an audit of the
selection and prediction-derived metrics. `all_baselines.md` also includes the
previously trained non-PhytoODE baselines, without rerunning or retuning them.
