# Prefix-conditioned logistic parameters without light inputs

Only PhytoODE was retrained. Both the prefix encoder and latent vector field omit illumination. The r/K head receives genotype embedding, masked individual 0–36 h lengths and their presence masks. lambda_K=0 removes the capacity penalty; K still receives gradients through the logistic residual.

The three-seed mean natural-missingness 48-h validation RMSE selected **lambda_ODE=100**. The separately predefined fixed-reference coefficient is **100**, selected previously for the light-input model. Both coefficients are transferred unchanged across additional missingness; identical configurations are evaluated once.

Completed 30 candidate and 6 transfer fits. Training outputs contain training/validation scores only. All unique checkpoint hashes were frozen before test scoring. This is a follow-up on a dataset analyzed previously; this search ranks configurations by validation only.

The fixed-coefficient comparison holds the parameter-head design, loss weights, schedule and masks fixed. Removing input channels changes the backbone dimensions (1231 to 1183 parameters) and its initialization. It is therefore a feature-variant comparison, not a matched-weight perturbation. The original six baselines and genotype-head PhytoODE were not retrained or retuned.

Cells: mean per-plant RMSE (mm) / relative RMSE (%), mean ± sample SD across three seeds/masks. Relative RMSE is 100 times per-plant RMSE divided by the mean of scored individual lengths. Train: reconstruction of available 0–36 h inputs; validation: 48 h; test: 60/72 h. No observed target is averaged or imputed. Natural-missingness Logistic ODE is deterministic (one fit).

## Additional removal of available prefix observations: 0%

| Model | Train | Validation | Test |
|---|---:|---:|---:|
| Logistic ODE | 0.409 / 11.79% | 0.769 / 13.53% | 1.708 / 22.89% |
| Random forest | **0.028 ± 0.001 / 0.81 ± 0.03%** | 1.018 ± 0.000 / 17.92 ± 0.00% | 2.608 ± 0.001 / 34.95 ± 0.01% |
| LSTM-NN | 0.295 ± 0.020 / 8.51 ± 0.58% | 0.871 ± 0.013 / 15.33 ± 0.24% | 2.003 ± 0.288 / 26.83 ± 3.86% |
| Logistic-PINN | 0.367 ± 0.016 / 10.57 ± 0.47% | 0.616 ± 0.027 / 10.84 ± 0.47% | 1.336 ± 0.083 / 17.90 ± 1.11% |
| Latent ODE (no light) | 0.400 ± 0.013 / 11.53 ± 0.38% | 0.627 ± 0.046 / 11.03 ± 0.81% | **1.143 ± 0.155 / 15.31 ± 2.07%** |
| Latent ODE (+ light) | 0.377 ± 0.018 / 10.87 ± 0.51% | 0.604 ± 0.015 / 10.62 ± 0.27% | 1.439 ± 0.056 / 19.28 ± 0.76% |
| PhytoODE (previous genotype head) | 0.356 ± 0.059 / 10.27 ± 1.70% | **0.564 ± 0.007 / 9.92 ± 0.13%** | 1.703 ± 0.085 / 22.82 ± 1.13% |
| PhytoODE (+ light; prefix head; lambda=100) | 0.383 ± 0.015 / 11.04 ± 0.44% | 0.570 ± 0.012 / 10.03 ± 0.21% | 1.280 ± 0.052 / 17.15 ± 0.70% |
| PhytoODE (no light; prefix head; selected lambda=100) | 0.402 ± 0.032 / 11.58 ± 0.93% | 0.581 ± 0.005 / 10.22 ± 0.09% | 1.150 ± 0.149 / 15.41 ± 2.00% |

## Additional removal of available prefix observations: 25%

| Model | Train | Validation | Test |
|---|---:|---:|---:|
| Logistic ODE | 0.330 ± 0.006 / 9.39 ± 0.11% | 0.755 ± 0.071 / 13.28 ± 1.25% | 1.661 ± 0.027 / 22.26 ± 0.36% |
| Random forest | **0.050 ± 0.007 / 1.42 ± 0.20%** | 1.295 ± 0.022 / 22.80 ± 0.39% | 2.915 ± 0.046 / 39.06 ± 0.62% |
| LSTM-NN | 0.291 ± 0.031 / 8.28 ± 0.94% | 0.931 ± 0.186 / 16.39 ± 3.28% | 1.775 ± 0.309 / 23.79 ± 4.14% |
| Logistic-PINN | 0.321 ± 0.009 / 9.13 ± 0.22% | 0.673 ± 0.034 / 11.84 ± 0.61% | 1.277 ± 0.166 / 17.11 ± 2.23% |
| Latent ODE (no light) | 0.390 ± 0.021 / 11.08 ± 0.51% | 0.619 ± 0.057 / 10.89 ± 1.01% | **1.235 ± 0.409 / 16.55 ± 5.48%** |
| Latent ODE (+ light) | 0.387 ± 0.026 / 11.00 ± 0.71% | 0.621 ± 0.054 / 10.93 ± 0.94% | 1.514 ± 0.286 / 20.28 ± 3.84% |
| PhytoODE (previous genotype head) | 0.371 ± 0.042 / 10.56 ± 1.21% | **0.547 ± 0.013 / 9.63 ± 0.23%** | 1.702 ± 0.070 / 22.80 ± 0.94% |
| PhytoODE (+ light; prefix head; lambda=100) | 0.378 ± 0.028 / 10.75 ± 0.74% | 0.602 ± 0.037 / 10.60 ± 0.64% | 1.299 ± 0.187 / 17.40 ± 2.51% |
| PhytoODE (no light; prefix head; selected lambda=100) | 0.389 ± 0.026 / 11.07 ± 0.68% | 0.617 ± 0.034 / 10.85 ± 0.60% | 1.317 ± 0.179 / 17.65 ± 2.40% |

## Additional removal of available prefix observations: 50%

| Model | Train | Validation | Test |
|---|---:|---:|---:|
| Logistic ODE | 0.183 ± 0.012 / 5.31 ± 0.40% | 0.666 ± 0.093 / 11.72 ± 1.63% | 1.789 ± 0.225 / 23.98 ± 3.02% |
| Random forest | **0.066 ± 0.006 / 1.92 ± 0.15%** | 1.709 ± 0.070 / 30.08 ± 1.23% | 3.415 ± 0.047 / 45.75 ± 0.63% |
| LSTM-NN | 0.237 ± 0.026 / 6.86 ± 0.82% | 1.249 ± 0.338 / 21.99 ± 5.95% | 2.640 ± 0.455 / 35.38 ± 6.09% |
| Logistic-PINN | 0.270 ± 0.059 / 7.84 ± 1.78% | 0.735 ± 0.009 / 12.93 ± 0.16% | 1.248 ± 0.118 / 16.72 ± 1.58% |
| Latent ODE (no light) | 0.340 ± 0.015 / 9.85 ± 0.50% | 0.757 ± 0.247 / 13.33 ± 4.35% | 1.292 ± 0.357 / 17.32 ± 4.79% |
| Latent ODE (+ light) | 0.373 ± 0.008 / 10.82 ± 0.13% | 0.786 ± 0.020 / 13.84 ± 0.35% | 2.010 ± 0.168 / 26.94 ± 2.24% |
| PhytoODE (previous genotype head) | 0.305 ± 0.017 / 8.84 ± 0.42% | **0.561 ± 0.028 / 9.87 ± 0.48%** | 1.575 ± 0.150 / 21.10 ± 2.01% |
| PhytoODE (+ light; prefix head; lambda=100) | 0.324 ± 0.045 / 9.39 ± 1.35% | 0.688 ± 0.095 / 12.12 ± 1.67% | 1.422 ± 0.166 / 19.06 ± 2.22% |
| PhytoODE (no light; prefix head; selected lambda=100) | 0.336 ± 0.030 / 9.75 ± 0.95% | 0.677 ± 0.099 / 11.91 ± 1.75% | **1.224 ± 0.169 / 16.40 ± 2.27%** |

## Validation-only coefficient search

| lambda_ODE | Validation RMSE (mm) |
|---:|---:|
| 0.01 | 0.6213 ± 0.0360 |
| 0.1 | 0.5817 ± 0.0026 |
| 0.5 | 0.6084 ± 0.0211 |
| 1 | 0.6204 ± 0.0582 |
| 5 | 0.6007 ± 0.0123 |
| 10 | 0.6069 ± 0.0203 |
| 50 | 0.5927 ± 0.0138 |
| 100 | **0.5807 ± 0.0050** |
| 500 | 0.6071 ± 0.0062 |
| 1000 | 0.6148 ± 0.0105 |
