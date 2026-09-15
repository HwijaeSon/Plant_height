# Four-genotype hypocotyl forecasting

All models were retrained from scratch on Col-0, hy5, MLB, and phyAB using fixed manuscript settings. Early observations at 0–36 h are inputs/training targets; validation is at 48 h and testing at 60/72 h.

Cells: mean per-plant RMSE (mm) / relative RMSE (%), with sample SD over the paired training seeds and masks. The natural-missingness Logi-ODE was fitted once. Bold marks the smallest unrounded mean.

## Additional removal: 0% (total missingness 11.35%)

| Model | Train | Validation | Test |
|---|---:|---:|---:|
| Logi-ODE | 0.378 / 11.40% | 0.753 / 13.84% | 1.684 / 23.41% |
| RF | **0.032 ± 0.000 / 0.96 ± 0.01%** | 0.944 ± 0.000 / 17.35 ± 0.01% | 2.587 ± 0.002 / 35.96 ± 0.03% |
| LSTM-NN | 0.310 ± 0.017 / 9.36 ± 0.50% | 0.771 ± 0.121 / 14.17 ± 2.23% | 1.525 ± 0.351 / 21.20 ± 4.88% |
| Logistic-PINN | 0.332 ± 0.011 / 10.01 ± 0.34% | 0.571 ± 0.026 / 10.50 ± 0.47% | 1.509 ± 0.092 / 20.98 ± 1.28% |
| Latent ODE | 0.376 ± 0.009 / 11.33 ± 0.26% | **0.518 ± 0.011 / 9.53 ± 0.20%** | **1.071 ± 0.096 / 14.89 ± 1.33%** |
| PhytoODE | 0.380 ± 0.006 / 11.46 ± 0.19% | 0.529 ± 0.007 / 9.72 ± 0.13% | 1.284 ± 0.135 / 17.85 ± 1.87% |

## Additional removal: 25% (total missingness 33.51%)

| Model | Train | Validation | Test |
|---|---:|---:|---:|
| Logi-ODE | 0.297 ± 0.019 / 8.90 ± 0.51% | 0.673 ± 0.024 / 12.37 ± 0.43% | 1.578 ± 0.230 / 21.94 ± 3.20% |
| RF | **0.058 ± 0.006 / 1.75 ± 0.17%** | 1.202 ± 0.031 / 22.09 ± 0.58% | 2.918 ± 0.076 / 40.57 ± 1.05% |
| LSTM-NN | 0.244 ± 0.002 / 7.30 ± 0.12% | 0.971 ± 0.079 / 17.86 ± 1.45% | 1.958 ± 0.409 / 27.22 ± 5.68% |
| Logistic-PINN | 0.284 ± 0.019 / 8.52 ± 0.53% | 0.571 ± 0.035 / 10.50 ± 0.64% | 1.315 ± 0.164 / 18.28 ± 2.29% |
| Latent ODE | 0.348 ± 0.030 / 10.41 ± 0.78% | 0.537 ± 0.031 / 9.87 ± 0.58% | 1.294 ± 0.102 / 17.99 ± 1.41% |
| PhytoODE | 0.348 ± 0.052 / 10.42 ± 1.46% | **0.528 ± 0.044 / 9.70 ± 0.81%** | **1.220 ± 0.092 / 16.96 ± 1.28%** |

## Additional removal: 50% (total missingness 55.67%)

| Model | Train | Validation | Test |
|---|---:|---:|---:|
| Logi-ODE | 0.169 ± 0.005 / 5.03 ± 0.09% | 0.663 ± 0.070 / 12.19 ± 1.28% | 1.855 ± 0.234 / 25.79 ± 3.26% |
| RF | **0.060 ± 0.005 / 1.79 ± 0.15%** | 1.594 ± 0.074 / 29.30 ± 1.36% | 3.357 ± 0.053 / 46.67 ± 0.74% |
| LSTM-NN | 0.252 ± 0.020 / 7.48 ± 0.64% | 1.203 ± 0.116 / 22.11 ± 2.13% | 2.572 ± 0.265 / 35.75 ± 3.68% |
| Logistic-PINN | 0.219 ± 0.018 / 6.49 ± 0.50% | 0.712 ± 0.062 / 13.09 ± 1.13% | 1.295 ± 0.042 / 18.00 ± 0.58% |
| Latent ODE | 0.285 ± 0.031 / 8.45 ± 0.83% | **0.548 ± 0.040 / 10.07 ± 0.73%** | 1.332 ± 0.246 / 18.52 ± 3.42% |
| PhytoODE | 0.292 ± 0.011 / 8.66 ± 0.40% | 0.555 ± 0.031 / 10.20 ± 0.57% | **1.027 ± 0.103 / 14.28 ± 1.43%** |

