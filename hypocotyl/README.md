# Hypocotyl growth: 12L12D-only benchmarks

The comparisons use **943 measured lengths** from the author's 12 h light /
12 h dark experiment at **23°C**, across five genotypes and seven elapsed times
(0, 12, 24, 36, 48, 60, 72 h). Length is in **mm**. The continuous-red-light (`cR`)
sheet is excluded from this comparison. The complete original workbook remains
preserved, including its 875 cR measurements, for provenance and earlier analyses.

In the reference comparison, models receive **genotype and elapsed time only**. There are no illumination,
duty-cycle, accumulated-exposure, spectrum, or temperature features. The physical
reference is the ordinary logistic law, with one constant rate and one capacity
per genotype. No light-switched ODE or light-conditioned PINN is included.
An additional [PhytoODE-only follow-up](reports/light_input_20260910/results.md)
adds the known binary light state to PhytoODE while retaining this ordinary
logistic physics, the same data/targets, and every earlier baseline fit.

## Current targets and partitions

Following the requested return to the initial evaluation, targets are **replicate
means computed separately inside train, validation, and test**. This is an explicit
change from the previous individual-observation comparison. No full-dataset mean
is used for training. Missing cells are omitted before averaging, and absent time
groups have no target; missing phenotype values are never imputed or interpolated.

| Protocol | Raw train / validation / test measurements | Mean targets train / validation / test |
|---|---|---|
| Replicate holdout (primary) | 547 / 171 / 225 | 35 / 35 / 35 |
| 36 / 60 h holdout (auxiliary) | 397 / 125 / 257 | 25 / 25 / 10 |

Primary partitions contain every observed time. The auxiliary protocol withholds
all measurements at 36 and 60 h from training and validation; the 164 original
replicate-test measurements at other times are unused. This auxiliary evaluation
is retained in the report and figures, outside the manuscript's primary evaluation.

The original genotype/source-row assignments are filtered, not reassigned. Rows
are bookkeeping groups, **not verified longitudinal plant IDs**. Each model fits
five genotype-level curves and does not reconstruct individual plant trajectories.
Earlier test results on the parent partitions were inspected; this is not a new,
independent test cohort. All current configurations are selected using validation
only and frozen before new test scoring. Training height scale is **13.719 mm**,
the maximum retained training measurement, separately checked for both protocols.

Primary RMSE is the mean of the five genotype-curve RMSEs against the scored
replicate means. Relative RMSE is 100 × this RMSE / the mean of those scored
replicate-mean targets. It is not MAPE. The report retains pooled individual-length
RMSE separately. The two metrics must not be combined in one comparison column.

## Models and fixed search

- **PhytoODE:** latent dimension 8; genotype embedding 4; vector-field width 16,
  one hidden layer; decoder width 16; time encoder width 8. There are 1,055
  parameters including the logistic parameter head. The latent flow generates
  length; an ordinary logistic derivative residual and finite-window capacity
  penalty regularize the decoded curve.
- **Latent ODE:** identical architecture, both physics coefficients zero;
  learning rate selected independently. A separate **matched** control uses the
  selected PhytoODE settings and initial weights with both coefficients zero.
- **Logistic-PINN:** coordinate MLP, genotype embedding 4, hidden widths 32/32;
  direct time differentiation and the same ordinary logistic penalties. This is
  distinct from the LSTM-based Logi-PINN in the temperature benchmarks.
- **LSTM-NN:** time-only recurrent input, recurrent widths 16/8, genotype embedding
  4, dense head width 16.
- **Random forest:** genotype one-hot encoding and normalized time, 300 trees.
- **Logistic ODE:** separate rate, capacity, and initial height per genotype,
  15 parameters in total; eight deterministic starting points on training means.

The original search budget is retained. Per protocol, 16 PhytoODE and 16
Logistic-PINN candidates cross learning rates {0.003, 0.01}, lambda_ODE
{0.5, 5, 50, 500}, and lambda_K {0.01, 0.1}. Latent ODE and LSTM screen the two
learning rates; RF screens minimum leaf sizes {1, 2, 4}. Neural models use 1,500
epochs, Adam with weight decay 1e-4 on all parameters, cosine decay, and gradient
clipping at 1. After epoch 200, checkpoints minimize the trailing three validation
evaluations spaced 20 epochs apart. The best two candidates per stochastic model
are confirmed with seeds 2 and 3, and selected by three-seed validation RMSE.
Search budgets differ between families. Both protocols total **120 fresh fits**
and **38 final evaluations**, including the already-trained matched controls.

The 3 h integration grid has 25 nodes. All 23 interior nodes contribute to the
logistic residual. The decoder chain rule converts normalized-time derivatives
back to hours by division by 72. No finite differences of measured lengths are
used. The capacity penalty targets the predicted maximum within the observation
window; it should not be interpreted as independently measured mature length.

## Reproduction and outputs

The filtered frozen data are included under `data/processed/single_condition_20260909`.
Do not regenerate or overwrite these partitions. On a fresh extraction without
that folder, `prepare_single_condition.py` creates it from the parent partitions.

```bash
.venv/bin/python hypocotyl/code/verify_single_condition.py
.venv/bin/python hypocotyl/code/run_single_condition.py --gpus 8 9
.venv/bin/python hypocotyl/code/summarize_single_condition.py
.venv/bin/python paper/write_hypocotyl_results.py
.venv/bin/python hypocotyl/code/package_release.py
```

The runner refuses to overwrite an existing run; `--resume` verifies frozen
source/data hashes before reusing any finished trial. Candidate searches never
load test targets. Final scoring evaluates saved predictions after all selections
are frozen.

- [Current results, both holdouts and all three splits](reports/single_condition_20260909/results.md).
- [Machine-readable comparison](reports/single_condition_20260909/comparison.csv).
- [Replicate-holdout curves](reports/single_condition_20260909/predictions_replicate.png).
- [36/60-h holdout curves](reports/single_condition_20260909/predictions_time_holdout.png).
- [Verification](reports/single_condition_20260909/verification.json) and [result audit](reports/single_condition_20260909/results_audit.json).
- [Logistic parameters](reports/single_condition_20260909/logistic_parameters.csv).
- [Data fields](DATA_DICTIONARY.md) and [current model rationale](reports/model_rationale.md).
- [Data/code bundle](release/hypocotyl_data_code_20260909.zip), prepared for deposit.

In the current prediction figures, black points are test replicate means, faint
gray points are individual test measurements, and curves are mean predictions over
three training seeds (the process ODE is a single fit). No inferred individual
plant trajectories or light-phase forcing are shown.

The executed trainer exports detailed history every 200 epochs, so these
1,500-epoch runs have CSV histories through epoch 1,400. Final checkpoints and
predictions are complete. The audit records this logging gap; three selected
checkpoints lie beyond the exported history. Their stored epochs and predictions
are verified directly. No missing history is fabricated, and a separate diagnostic
replay was not substituted into the comparison.

## PhytoODE with binary illumination input (2026-09-10)

This follow-up adds **only one environmental channel** to PhytoODE: light on = 1
and light off = 0. The first 0–12 h interval is on, followed by alternating 12 h
intervals. cR remains excluded. The known schedule enters the encoder and latent
vector field; phenotype observations never enter the encoder. All plants share
the same schedule, so this channel is a deterministic function of elapsed time.
It tests an explicit phase feature, not transfer to a different photoperiod.

The same ordinary logistic derivative residual and capacity penalty are retained.
There is one constant rate and capacity per genotype, with **no separate light and
dark logistic rates**. The hidden sizes are unchanged; the added channel increases
the parameter count from 1,055 to 1,103. RK4 keeps illumination constant within
each interval, including its final stage, and switches at the next interval.
The original 23 interior residual nodes use the right-hand derivative at switches.
The decoder chain rule still divides by 72 to convert normalized time to hours.

The fixed search screens 32 configurations per holdout: the original 16 PhytoODE
grid combinations and 16 stratified log-space samples of learning rate,
lambda_ODE, lambda_K, and weight decay. All fits use 1,500 epochs. The leading
three seed-1 candidates, plus the previous selected no-input hyperparameter
anchor when necessary, are confirmed with seeds 2 and 3. The lowest mean
validation RMSE selects the tuned model. The anchor is also reported to separate
the feature change at fixed settings from additional tuning. Input dimensions
change, so identical initial weights are not claimed. Both holdouts' selections
freeze before current test scoring; previously inspected test partitions are
explicitly acknowledged. Search budgets are unequal across the retained models.

The new trainer exports the complete validation history through epoch 1,500.
The original experiment and manuscript remain the reference; this comparison
does not silently replace their tables or fitted models.

```bash
.venv/bin/python hypocotyl/code/verify_light_input.py
.venv/bin/python hypocotyl/code/run_light_input.py --gpus 8 9
.venv/bin/python hypocotyl/code/summarize_light_input.py
```

- [Full comparison, both holdouts, train/validation/test](reports/light_input_20260910/results.md).
- [Comparison CSV](reports/light_input_20260910/comparison.csv) and [selected settings](reports/light_input_20260910/selected_configs.json).
- [Test-error plot](reports/light_input_20260910/test_error_comparison.png).
- [Replicate-holdout curves](reports/light_input_20260910/predictions_replicate.png).
- [36/60-h holdout curves](reports/light_input_20260910/predictions_time_holdout.png).
- [PhytoODE variants, replicate](reports/light_input_20260910/phytoode_variants_replicate.png) and [36/60 h](reports/light_input_20260910/phytoode_variants_time_holdout.png).
- [Input/derivative verification](reports/light_input_20260910/verification.json) and [results audit](reports/light_input_20260910/results_audit.json).
- LaTeX tables: [replicate](reports/light_input_20260910/table_replicate.tex), [36/60 h](reports/light_input_20260910/table_time_holdout.tex).
- [Compiled train/validation/test tables (PDF)](reports/light_input_20260910/tables.pdf).

## Earlier experiments

The two-condition mean-target runs are preserved in `results/light_growth_20260909`
and `results/primary_retuning_20260909`. The two-condition individual-target rerun
is preserved in `results/individual_observations_20260909` with its [own report](reports/individual_observations_20260909/results.md).
They use different targets and/or different data from the current run, and their
scores should not be substituted into the current tables. The [light-reference notes](reports/light_reference_notes.md)
document the earlier switched model. The current ordinary logistic reference is
supported by Alimchandani et al. (2026), New Phytologist, doi:10.1111/nph.70576.
