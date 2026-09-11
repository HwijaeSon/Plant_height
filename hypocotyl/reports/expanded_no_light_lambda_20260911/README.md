# Expanded lambda_ODE search for no-light PhytoODE

This experiment changes **only the coefficient search grid**. The no-light
prefix-conditioned PhytoODE architecture, `lambda_K=0`, seed assignments,
training schedule and evaluation protocol are inherited unchanged from
`prefix_parameters_no_light_20260911`. The growth-rate/carrying-capacity head
receives the genotype embedding, masked initial lengths and presence masks.
There are 1,183 parameters. Illumination is absent from both the encoder and
latent vector field; time remains an input.

The range expands from **0.01–1,000 to 0.001–10,000**, with denser coverage near
0.1 and 100, which had the lowest validation errors in the previous grid.
The additional values are:

`[0.001, 0.003, 0.03, 0.05, 0.2, 0.3, 75, 150, 200, 300, 3000, 10000]`.

The 10 previous candidates (30 training/validation-only fits) are reused by file
hash; the 12 additional candidates are trained with seeds 1–3 (36 new fits).
The trainer is unchanged: `expanded_no_light_trial.py` selects a separate profile
and invokes `no_light_prefix_trial.py`. No previous frozen source or result is
overwritten. Each fit completes 1,500 epochs. The selected checkpoint minimizes
the trailing three-evaluation mean validation RMSE from epoch 200 onward. Coefficients
are ranked by the three-seed mean actual validation RMSE of those checkpoints.

The winner is transferred unchanged to the two additional-missingness conditions
(25% and 50% removal of available prefix values). All nine checkpoint hashes are
frozen before any new test evaluation. If coefficient 100 is selected again,
the previous transfer fits and final evaluations are reused without retraining
or rescoring. The previous test results were already reported before this search;
this is a follow-up on the same dataset, not an independent confirmatory test.
No test scores enter candidate ranking or checkpoint selection.

Inputs/training targets remain individual available 0–36 h lengths, validation
is at 48 h and test is at 60/72 h. The same 157 plants and paired masks/scales are
used. Missing lengths are neither averaged into genotype targets nor imputed.
`lambda_K=0` removes the capacity penalty; K still learns through the logistic
derivative residual. Baselines are reused unchanged.

## Validation selection

The selected coefficient is **10000**, with natural-missingness
48-h validation RMSE **0.5726 ± 0.0042 mm**.
The preceding coefficient 100 gives 0.5807 ± 0.0050 mm. The winner is the upper
bound of the expanded grid, so this search does not locate an interior optimum
or establish a globally optimal coefficient.

## Final comparison

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

## Reproduction

From the repository root after installing `requirements.txt`:

```bash
# Extend the bundled prior validation search; do not repeat the 30 old fits.
python hypocotyl/code/run_expanded_no_light_search.py --gpus 1 2 \
  --output outputs/expanded-no-light-search

# Summarize freshly trained results against the bundled preceding experiments.
python hypocotyl/code/report_expanded_no_light.py \
  --results outputs/expanded-no-light-search \
  --output outputs/expanded-no-light-report --figures outputs/expanded-no-light-figures

# Reproduce the bundled expanded-search tables and figures without training.
python hypocotyl/code/report_expanded_no_light.py \
  --output outputs/bundled-expanded-report --figures outputs/bundled-expanded-figures

# Verify source/reused-file hashes, initialization, loss, selection and outputs.
python hypocotyl/code/audit_expanded_no_light.py

# Evaluate a bundled selected checkpoint on CPU.
python hypocotyl/code/expanded_no_light_trial.py evaluate \
  --checkpoint hypocotyl/results/expanded_no_light_lambda_20260911/selected/drop_0/seed1/checkpoint.pt \
  --output outputs/expanded-no-light-evaluation
```

`comparison.md`/`comparison.csv` provide every model/condition/split, including
the preceding coefficient-100 PhytoODE if the winner differs. Cells use mean
per-plant RMSE (mm) / relative RMSE (%); SD describes three training-seed/mask
realizations. `validation_search.csv` records all 22 candidate means and SDs.
The search figure distinguishes reused and added candidates. Individual-plant
plots use the same prefix-selected examples as the preceding report; they do not
select examples using future lengths. Grey bands show actual dark periods for
context, even though the selected model receives no illumination input.

The prior manuscript snapshot remains pinned in `configs/paper.json`. This
follow-up uses its own profile and entry points and is released as
`experiment-20260911-expanded-lambda`.
