# Maize experiments

The final fixed-protocol comparison is stored in
[`chronological_final_seed1_3/`](chronological_final_seed1_3/).
All 14 model runs are complete. See
[`comparison.md`](chronological_final_seed1_3/comparison.md) and
[`comparison.csv`](chronological_final_seed1_3/comparison.csv).
Each neural model and RF uses seeds 1–3; deterministic process baselines are
fitted once. GPU workers use physical GPUs 1, 2 and 6 in the recorded run.

The subsequent **ours-only hyperparameter search** is stored in
[`tuning_20260904/final/`](tuning_20260904/final/README.md).
It completed 95 runs across 67 configurations in 180 minutes on GPUs 0–2.
The final setting was frozen using mean 2020 validation RMSE over seeds 1–3
before evaluating those three checkpoints on 2021. Test rRMSE decreased from
25.84 ± 4.26% to 18.40 ± 1.42%. Its comparison retains the original baseline
results; only our model received this additional search budget.

Reproduce from the project root:

```bash
bash maize/code/run_all.sh 1 2 6
```

The local `chronological_seed1_3/` and
`chronological_common_selection_seed1_3/` directories are validation-only
checkpoint-policy audits. They are excluded from both the final comparison
and Git because they may contain interrupted runs. Local `smoke*` directories
contain short implementation checks, including an early
temperature-optimizer precision check, and are also ignored.

No entries from audit or smoke folders are aggregated into the final table.
The final protocol removes arbitrary checkpoint epoch floors and uses the
minimum validation RMSE every ten epochs. PINN checkpoints must occur after
the 500-epoch physics warm-up. Model architectures, optimizer settings and
full training budgets follow the existing repository experiments.

Large random-forest model binaries, run logs, and per-trial tuning arrays are
reproducible intermediate artifacts and are Git-ignored. The repository keeps
the complete tuning configurations and rankings, final checkpoints and
predictions, metrics, validation records, and publication figures.
