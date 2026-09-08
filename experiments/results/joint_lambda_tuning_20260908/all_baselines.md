# All model comparisons

Mean RMSE / relative RMSE (%). Bold marks the best displayed mean within each dataset and split. Existing baseline results are reused without retraining or retuning; stochastic models have three seeds and deterministic ODE fits one. Units: wheat m, maize relative UAV height, Arabidopsis cm. These follow-up test sets were previously inspected. Full per-seed metrics and SD are provided in the CSV files.

| Dataset | Model | Train | Validation | Test |
|---|---|---:|---:|---:|
| Wheat | LSTM-NN | 0.03433 / 11.47% | 0.07400 / 22.20% | 0.05667 / 19.13% |
| Wheat | Logi-PINN | 0.03433 / 11.47% | 0.06633 / 19.90% | 0.06567 / 22.17% |
| Wheat | Logi-ODE | 0.09730 / 32.50% | 0.05990 / 17.97% | 0.10590 / 35.76% |
| Wheat | Temp-ODE | 0.08982 / 30.01% | 0.05741 / 17.23% | 0.09343 / 31.55% |
| Wheat | RF | **0.00487 / 1.63%** | 0.08809 / 26.43% | 0.09929 / 33.52% |
| Wheat | Original PhytoODE | 0.02244 / 7.50% | 0.03804 / 11.41% | 0.03063 / 10.34% |
| Wheat | PhytoODE (K loss only) | 0.02253 / 7.53% | 0.03775 / 11.33% | 0.03047 / 10.29% |
| Wheat | Latent Neural ODE (no physics) | 0.02108 / 7.04% | 0.03834 / 11.50% | 0.03091 / 10.44% |
| Wheat | PhytoODE (ODE coefficient tuned) | 0.02231 / 7.45% | **0.03768 / 11.30%** | **0.03032 / 10.24%** |
| Wheat | PhytoODE (joint tuning) | 0.02231 / 7.45% | **0.03768 / 11.30%** | **0.03032 / 10.24%** |
| Maize | Logi-ODE | 47.29 / 15.15% | 124.63 / 49.17% | 128.98 / 41.87% |
| Maize | Temp-ODE | 47.35 / 15.17% | 110.12 / 43.45% | 110.29 / 35.80% |
| Maize | RF | **7.03 / 2.25%** | 131.78 / 51.99% | 139.61 / 45.31% |
| Maize | LSTM-NN | 42.62 / 13.66% | **40.92 / 16.15%** | 98.77 / 32.06% |
| Maize | Logi-PINN | 33.18 / 10.63% | 48.14 / 19.00% | 108.71 / 35.28% |
| Maize | Original PhytoODE | 41.90 / 13.43% | 47.25 / 18.64% | 56.68 / 18.40% |
| Maize | PhytoODE (K loss only) | 38.09 / 12.21% | 48.14 / 18.99% | **55.83 / 18.12%** |
| Maize | Latent Neural ODE (no physics) | 43.25 / 13.86% | 60.75 / 23.97% | 67.78 / 22.00% |
| Maize | PhytoODE (ODE coefficient tuned) | 39.20 / 12.56% | 44.94 / 17.73% | 60.53 / 19.65% |
| Maize | PhytoODE (joint tuning) | 39.20 / 12.56% | 44.94 / 17.73% | 60.53 / 19.65% |
| Arabidopsis | Logi-ODE | 4.944 / 23.50% | 5.274 / 26.19% | 5.179 / 26.00% |
| Arabidopsis | Temp-ODE | 3.678 / 17.48% | 4.174 / 20.73% | 4.121 / 20.69% |
| Arabidopsis | RF | **0.153 / 0.73%** | 3.217 / 15.97% | 2.878 / 14.45% |
| Arabidopsis | LSTM-NN | 1.701 / 8.09% | 3.343 / 16.60% | 2.951 / 14.82% |
| Arabidopsis | Logi-PINN | 1.721 / 8.18% | 3.316 / 16.47% | 2.987 / 15.00% |
| Arabidopsis | Original PhytoODE | 1.401 / 6.66% | 3.246 / 16.12% | 2.866 / 14.39% |
| Arabidopsis | PhytoODE (K loss only) | 1.416 / 6.73% | 3.241 / 16.10% | 2.852 / 14.32% |
| Arabidopsis | Latent Neural ODE (no physics) | 1.188 / 5.64% | 3.278 / 16.28% | 2.993 / 15.03% |
| Arabidopsis | PhytoODE (ODE coefficient tuned) | 1.421 / 6.75% | **3.191 / 15.85%** | **2.800 / 14.06%** |
| Arabidopsis | PhytoODE (joint tuning) | 1.421 / 6.75% | **3.191 / 15.85%** | **2.800 / 14.06%** |
