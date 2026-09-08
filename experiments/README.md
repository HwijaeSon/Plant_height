# Matched physics-loss ablation

`physics_ablation.py` imports the existing wheat, maize and Arabidopsis data
loaders and latent ODE implementations. It compares the manuscript's fixed
PhytoODE settings with two loss ablations:

| Variant | Logistic derivative residual | Maximum-height consistency |
|---|---|---|
| `full` | Original coefficient | Original coefficient |
| `no_ode_residual` | Zero | Original coefficient |
| `no_biological_loss` | Zero | Zero |

The last row is the pure data-trained latent ODE. Optimizer weight decay is
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
