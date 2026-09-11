# Expanded coefficient search for no-light PhytoODE

**Selected lambda_ODE=10000; lambda_K=0.** The r/K head continues to receive genotype, masked individual 0–36 h lengths and presence masks. Illumination is absent from the encoder and vector field. Architecture, initialization, optimizer schedule, data and missingness masks are unchanged.

The coefficient grid was expanded from 0.01–1000 to 0.001–10000, with additional resolution near 0.1 and 100. The 30 original training/validation-only fits were reused; 12 additional coefficients were trained with seeds 1–3 (36 new search fits). All 22 candidates were ranked by three-seed mean 48-h validation RMSE.

There were 6 new missingness-transfer fits, 9 new test evaluations, and 0 reused test evaluations. All selected checkpoint hashes were frozen before any new test scoring. This expanded search follows prior analysis of the same dataset and reported coefficient-100 test results; candidate ranking uses validation only.

Cells: mean per-plant RMSE (mm) / relative RMSE (%), mean ± sample SD. Train: available 0–36 h measurements; validation: 48 h; test: 60/72 h. Missing targets are not averaged or imputed. Baselines are reused unchanged.

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
| PhytoODE (prefix r/K, light, lambda=100) | 0.383 ± 0.015 / 11.04 ± 0.44% | 0.570 ± 0.012 / 10.03 ± 0.21% | 1.280 ± 0.052 / 17.15 ± 0.70% |
| PhytoODE (prefix r/K, no light, previous lambda=100) | 0.402 ± 0.032 / 11.58 ± 0.93% | 0.581 ± 0.005 / 10.22 ± 0.09% | 1.150 ± 0.149 / 15.41 ± 2.00% |
| PhytoODE (prefix r/K, no light, expanded selection lambda=10000) | 0.469 ± 0.003 / 13.51 ± 0.09% | 0.573 ± 0.004 / 10.08 ± 0.07% | 1.579 ± 0.166 / 21.16 ± 2.22% |

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
| PhytoODE (prefix r/K, light, lambda=100) | 0.378 ± 0.028 / 10.75 ± 0.74% | 0.602 ± 0.037 / 10.60 ± 0.64% | 1.299 ± 0.187 / 17.40 ± 2.51% |
| PhytoODE (prefix r/K, no light, previous lambda=100) | 0.389 ± 0.026 / 11.07 ± 0.68% | 0.617 ± 0.034 / 10.85 ± 0.60% | 1.317 ± 0.179 / 17.65 ± 2.40% |
| PhytoODE (prefix r/K, no light, expanded selection lambda=10000) | 0.431 ± 0.028 / 12.25 ± 0.76% | 0.573 ± 0.028 / 10.09 ± 0.49% | 1.524 ± 0.086 / 20.43 ± 1.15% |

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
| PhytoODE (prefix r/K, light, lambda=100) | 0.324 ± 0.045 / 9.39 ± 1.35% | 0.688 ± 0.095 / 12.12 ± 1.67% | 1.422 ± 0.166 / 19.06 ± 2.22% |
| PhytoODE (prefix r/K, no light, previous lambda=100) | 0.336 ± 0.030 / 9.75 ± 0.95% | 0.677 ± 0.099 / 11.91 ± 1.75% | **1.224 ± 0.169 / 16.40 ± 2.27%** |
| PhytoODE (prefix r/K, no light, expanded selection lambda=10000) | 0.384 ± 0.017 / 11.12 ± 0.45% | 0.590 ± 0.025 / 10.38 ± 0.44% | 1.549 ± 0.235 / 20.76 ± 3.15% |

## All validation candidates

| lambda_ODE | Origin | Validation RMSE (mm) |
|---:|---|---:|
| 0.001 | Added | 0.6137 ± 0.0266 |
| 0.003 | Added | 0.5958 ± 0.0319 |
| 0.01 | Reused | 0.6213 ± 0.0360 |
| 0.03 | Added | 0.5925 ± 0.0232 |
| 0.05 | Added | 0.6329 ± 0.0568 |
| 0.1 | Reused | 0.5817 ± 0.0026 |
| 0.2 | Added | 0.6201 ± 0.0507 |
| 0.3 | Added | 0.6021 ± 0.0310 |
| 0.5 | Reused | 0.6084 ± 0.0211 |
| 1 | Reused | 0.6204 ± 0.0582 |
| 5 | Reused | 0.6007 ± 0.0123 |
| 10 | Reused | 0.6069 ± 0.0203 |
| 50 | Reused | 0.5927 ± 0.0138 |
| 75 | Added | 0.6257 ± 0.0435 |
| 100 | Reused | 0.5807 ± 0.0050 |
| 150 | Added | 0.5934 ± 0.0090 |
| 200 | Added | 0.5993 ± 0.0152 |
| 300 | Added | 0.6060 ± 0.0144 |
| 500 | Reused | 0.6071 ± 0.0062 |
| 1000 | Reused | 0.6148 ± 0.0105 |
| 3000 | Added | 0.6099 ± 0.0199 |
| 10000 | Added | **0.5726 ± 0.0042** |
