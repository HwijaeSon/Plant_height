# Wheat experiment

## Data

`data/align_height_env_same_length.csv` is the processed wheat height and
environment table used by Shao et al. (2026). It contains four field seasons
(2018, 2019, 2021, and 2022). The experiment uses the 19 genotypes present in
all four seasons. `data/kinship_matrix_astle.csv` is retained for the optional
kinship ablation; the reported model uses one-hot genotype encoding.

The paper-matched split is training on 2018/2019, validation on 2022, and
testing on 2021. The test-year height measurements are used only for scoring.

Source paper: Shao et al., “Physics-Informed Neural Network methods for
predicting plant height development,” *Computers and Electronics in
Agriculture* 251 (2026), 111988, DOI: 10.1016/j.compag.2026.111988.

## Reproduce the reported model

From the repository root:

```bash
python wheat/code/train.py \
  --data wheat/data/align_height_env_same_length.csv \
  --kinship wheat/data/kinship_matrix_astle.csv \
  --val_year 2022 --test_year 2021 \
  --genetic_encoding one_hot \
  --latent_dim 16 --g_embed_dim 4 \
  --ode_hidden 16 --ode_layers 1 \
  --dec_hidden 16 --enc_hidden 8 \
  --lr 0.01 --weight_physic 2 --weight_ymax 0.1 \
  --epochs 1500 --min_epoch 600 --val_smooth 10 \
  --seeds 1 2 3 --device cuda \
  --out_dir wheat/results/reproduced --tag tuned_latent_ode
```

The implementation uses a fixed-step differentiable RK4 solver. The complete
machine-readable configuration is in `results/final_config.json`.

## Results

- `results/paper_results_3seed.csv`: manuscript table rows.
- `results/tuned_latent_ode_seed1_3.csv`: per-seed final metrics.
- `results/training_history_seed1_3.csv`: learning curves.
- `results/test_predictions_seed1_3.npz`: predictions used by the paper plot.
- `results/checkpoints/`: exact checkpoints for seeds 1--3.
- `results/reference/`: original one-hot LSTM-NN and Logi-PINN output tables.
- `results/TUNING_RESULTS.md`: validation-only capacity and hyperparameter
  selection summary.
