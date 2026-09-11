# Prefix-conditioned logistic parameters, without the capacity penalty

This follow-up modifies **PhytoODE only**. Its parameter head now reads the
four-dimensional genotype embedding, four observed initial lengths, and four
presence masks. The MLP expands from 4→8→2 to 12→8→2. Its two outputs are
individual, time-independent growth rate and scaled carrying capacity.
The unchanged bounds are `0 < r_i < 0.5 /h` and `0 < kappa_i < 2`, with
`K_i = training_scale_mm × kappa_i`.

The genotype columns and the remainder of the model retain the preceding
initialization. Added prefix/mask columns start at zero and learn through the
logistic derivative residual. The model has **1,231 parameters**, including the
auxiliary head. There is no categorical plant-ID input.

`lambda_K=0` removes the finite-window maximum/capacity penalty. Carrying capacity
still appears in the logistic derivative residual and remains a learned parameter.
The objective is masked per-plant prefix RMSE plus `lambda_ODE × derivative MSE`.
All 23 interior 3-h nodes remain collocation points. The numerical solver, latent
state and encoder dimensions, binary illumination, and 1,500-epoch optimizer
schedule are unchanged.

## Frozen search and evaluation

The positive coefficient grid is `[0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]`.
Each coefficient is trained with seeds 1–3 under natural missingness. The winner
minimizes the three-seed mean **actual 48-h validation RMSE** of the selected
checkpoints. Checkpoint epochs retain the preceding smoothed validation rule.
The winning coefficient is then held fixed while PhytoODE is retrained under
25% and 50% additional removal, using the preceding paired masks and scales.

All 30 search runs and six transfer runs save training/validation results only.
The runner freezes the coefficient and all nine selected checkpoint hashes before
loading test targets for final evaluation. Only those nine checkpoints receive
test scores. No candidates are added in response to test performance.

Inputs/training are observed individual lengths at 0–36 h; validation is at 48 h;
test is at 60/72 h. There is no target averaging or phenotype imputation. Baseline
results are reused from `prefix_forecast_20260911` without retraining or retuning.
This follow-up combines three changes (parameter-head inputs, capacity penalty,
and coefficient search) and does not isolate their separate causal effects.
The dataset had already been analyzed in the preceding experiments.

## Reproduce

From the repository root:

```bash
# Complete the 30-fit search, six transfer fits, and final nine evaluations.
python hypocotyl/code/run_prefix_parameter_search.py --gpus 0 1 2 \
  --output outputs/prefix-parameter-search

# Summarize the newly fitted runs and compare to the bundled baseline results.
python hypocotyl/code/report_prefix_parameters.py \
  --results outputs/prefix-parameter-search \
  --output outputs/prefix-parameter-report --figures outputs/prefix-parameter-figures

# Reproduce figures/tables from the bundled follow-up instead of retraining.
python hypocotyl/code/report_prefix_parameters.py \
  --output outputs/bundled-prefix-parameter-report --figures outputs/bundled-prefix-parameter-figures

# Audit the bundled search records, masking, objectives and selected checkpoints.
python hypocotyl/code/audit_prefix_parameters.py
```

A single fit can be run with `prefix_parameter_trial.py train --lambda-ode VALUE
--seed 1 --device cuda:0 --output outputs/one-fit`. It evaluates training and
validation only. To evaluate its checkpoint explicitly use the `evaluate` mode
with `--checkpoint outputs/one-fit/checkpoint.pt` and a new output directory.
The complete search runner enforces the prescribed selection sequence.

`comparison.md` and `comparison.csv` contain all model/condition/split scores;
`validation_search.csv` contains the validation-only search results. The result
directory contains `lambda_selection.json`, `test_evaluation_plan.json`, trained
checkpoints, histories and predictions. Each selected run also provides
`individual_parameters.csv` with the fitted r/K values and observation count.
These inferred parameters are model estimates, not directly measured physiology.
