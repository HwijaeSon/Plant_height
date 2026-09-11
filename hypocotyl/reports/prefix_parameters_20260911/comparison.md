# PhytoODE with an individual prefix-conditioned parameter head

The r/K head receives genotype embedding, four observed prefix lengths and four presence masks. The finite-window capacity loss is disabled (lambda_K=0). The remaining architecture and training schedule are unchanged.

Selected lambda_ODE: **100**, ranked by three-seed mean 48-h validation RMSE under natural missingness.
This coefficient is transferred unchanged to both additional-missingness conditions. All 30 search fits and six transfer fits save training/validation results only. The coefficient and all nine checkpoint hashes are frozen before test evaluation.

Only PhytoODE was retrained. Previous baseline and genotype-head PhytoODE results are reused unchanged. This follow-up combines a new parameter head, removal of the capacity penalty, and coefficient search; it does not isolate the individual effects of these changes. The same dataset had been analyzed previously.

Cells report mean per-plant RMSE (mm) / relative RMSE (%), mean ± sample SD. Training error measures reconstruction of the observed 0–36 h inputs; validation is at 48 h and test at 60/72 h.

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
| PhytoODE (prefix head; lambda_K=0) | 0.383 ± 0.015 / 11.04 ± 0.44% | 0.570 ± 0.012 / 10.03 ± 0.21% | 1.280 ± 0.052 / 17.15 ± 0.70% |

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
| PhytoODE (prefix head; lambda_K=0) | 0.378 ± 0.028 / 10.75 ± 0.74% | 0.602 ± 0.037 / 10.60 ± 0.64% | 1.299 ± 0.187 / 17.40 ± 2.51% |

## Additional removal of available prefix observations: 50%

| Model | Train | Validation | Test |
|---|---:|---:|---:|
| Logistic ODE | 0.183 ± 0.012 / 5.31 ± 0.40% | 0.666 ± 0.093 / 11.72 ± 1.63% | 1.789 ± 0.225 / 23.98 ± 3.02% |
| Random forest | **0.066 ± 0.006 / 1.92 ± 0.15%** | 1.709 ± 0.070 / 30.08 ± 1.23% | 3.415 ± 0.047 / 45.75 ± 0.63% |
| LSTM-NN | 0.237 ± 0.026 / 6.86 ± 0.82% | 1.249 ± 0.338 / 21.99 ± 5.95% | 2.640 ± 0.455 / 35.38 ± 6.09% |
| Logistic-PINN | 0.270 ± 0.059 / 7.84 ± 1.78% | 0.735 ± 0.009 / 12.93 ± 0.16% | **1.248 ± 0.118 / 16.72 ± 1.58%** |
| Latent ODE (no light) | 0.340 ± 0.015 / 9.85 ± 0.50% | 0.757 ± 0.247 / 13.33 ± 4.35% | 1.292 ± 0.357 / 17.32 ± 4.79% |
| Latent ODE (+ light) | 0.373 ± 0.008 / 10.82 ± 0.13% | 0.786 ± 0.020 / 13.84 ± 0.35% | 2.010 ± 0.168 / 26.94 ± 2.24% |
| PhytoODE (previous genotype head) | 0.305 ± 0.017 / 8.84 ± 0.42% | **0.561 ± 0.028 / 9.87 ± 0.48%** | 1.575 ± 0.150 / 21.10 ± 2.01% |
| PhytoODE (prefix head; lambda_K=0) | 0.324 ± 0.045 / 9.39 ± 1.35% | 0.688 ± 0.095 / 12.12 ± 1.67% | 1.422 ± 0.166 / 19.06 ± 2.22% |

## Validation-only coefficient search

| lambda_ODE | Validation RMSE (mm) |
|---:|---:|
| 0.01 | 0.6043 ± 0.0373 |
| 0.1 | 0.6167 ± 0.0287 |
| 0.5 | 0.5904 ± 0.0063 |
| 1 | 0.5997 ± 0.0191 |
| 5 | 0.6049 ± 0.0310 |
| 10 | 0.6072 ± 0.0405 |
| 50 | 0.5917 ± 0.0251 |
| 100 | **0.5700 ± 0.0119** |
| 500 | 0.6073 ± 0.0132 |
| 1000 | 0.6140 ± 0.0172 |
