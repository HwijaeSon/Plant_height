# Individual-parameter PhytoODE without illumination inputs

This follow-up removes the binary light input from both the prefix encoder and
latent vector field. The growth law remains autonomous logistic growth,
`dH_i/dt = r_i H_i (1 - H_i/K_i)`. The auxiliary r/K head reads the genotype
embedding, four masked 0–36 h lengths and four presence masks (12→8→2 MLP).
There is no categorical plant ID. Time remains an input, so omitting explicit
illumination does not remove time information associated with the common cycle.

`lambda_K=0` disables the finite-window capacity penalty. Carrying capacity
still learns through the logistic derivative residual. Its scale and bounds are
unchanged: `0 < r_i < 0.5 /h`, `0 < K_i < 2 × training_scale_mm`.
The objective is masked per-plant prefix RMSE plus `lambda_ODE × derivative MSE`.
All 23 interior 3-h collocation nodes and the 1,500-epoch schedule are unchanged.

The no-light model has **1,183 parameters**, compared with 1,231 with light.
Its backbone and original genotype-head weights preserve the initialization of
the preceding no-light latent ODE at each seed. Additional prefix/mask columns
are initialized to zero. The encoder receives length, presence and time only;
the vector field receives latent state, genotype embedding and time. The `env`
argument remains in the common API for compatibility, but its values are unused.
Architecture and checkpoint audits verify invariance to perturbed light values
and masked placeholder lengths.

## Selection and fixed-coefficient comparison

The frozen coefficient grid is `[0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]`,
with seeds 1–3. The three-seed mean natural-missingness 48-h validation RMSE
selects the coefficient. The selected coefficient and the predefined reference
**100** (the preceding light-input winner) are then transferred unchanged to 25%
and 50% removal of available prefix observations. Identical coefficients are
deduplicated. Checkpoint epochs follow the original smoothed-validation rule.

The runner completes all candidate and transfer fits with training/validation
scores only, freezes both coefficient variants and all unique checkpoint hashes,
and then evaluates test targets. It does not add candidates based on test results.
The selected coefficient and counts are recorded in `completion.json` and
`lambda_selection.json` under the result directory. `comparison.md` gives all
train/validation/test scores; `light_input_comparison.csv` isolates the new
feature variants. `validation_search.csv` contains no test scores.

This comparison uses the same 157 plants, 551 available prefix measurements,
128 validation measurements and 263 test measurements as the previous benchmark.
The original masks, seed assignments and training-derived scales are retained.
Removed prefix values are absent from both inputs and training loss. Targets are
individual observed lengths; no genotype averaging or missing-value imputation
is used. The six baselines and original genotype-head PhytoODE are reused exactly.

The fixed-coefficient comparison holds loss weights, head design, schedule and
masks fixed, but changes input dimensions and backbone initialization. It does
not isolate illumination through a matched-weight perturbation. Both feature
variants were selected on this previously analyzed dataset using validation
scores only; the three-seed SD is not uncertainty across independent cohorts.

## Reproduce

Run from the repository root, after installing the pinned requirements:

```bash
# Thirty candidate fits; six or twelve transfer fits; final frozen evaluations.
python hypocotyl/code/run_no_light_prefix_search.py --gpus 0 1 2 \
  --output outputs/no-light-prefix-search

# Report a newly trained search against the bundled preceding experiments.
python hypocotyl/code/report_no_light_prefix.py \
  --results outputs/no-light-prefix-search \
  --output outputs/no-light-prefix-report --figures outputs/no-light-prefix-figures

# Reproduce the bundled results and plots without retraining.
python hypocotyl/code/report_no_light_prefix.py \
  --output outputs/bundled-no-light-report --figures outputs/bundled-no-light-figures

# Audit objectives, masks, selection, checkpoint reloads and light invariance.
python hypocotyl/code/audit_no_light_prefix.py

# Train one chosen coefficient; this command does not evaluate the test set.
python hypocotyl/code/no_light_prefix_trial.py train --lambda-ode 100 \
  --seed 1 --device cuda:0 --output outputs/one-no-light-fit

# Explicitly evaluate that checkpoint, after fixing its configuration.
python hypocotyl/code/no_light_prefix_trial.py evaluate \
  --checkpoint outputs/one-no-light-fit/checkpoint.pt --output outputs/one-no-light-evaluation
```

The `run.py` entry point and `configs/paper.json` continue to reproduce the pinned
preceding manuscript configuration. These follow-ups have their own scripts,
profiles, results and reports, under tag `experiment-20260911-prefix-parameters`.

The trajectory figure uses the same five plants as the preceding report, chosen
by genotype, source-row order and incomplete prefix counts without consulting
future lengths. Curves average three seed predictions; observed points retain
individual measurements. Grey shading denotes the actual dark intervals for
context, even for models that receive no light input. Missingness error bars show
sample SD across the three paired training-seed/mask realizations.
