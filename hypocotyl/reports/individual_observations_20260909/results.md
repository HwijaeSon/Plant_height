# Hypocotyl benchmark: individual observed measurements

No replicate means are used as targets. Missing phenotype cells have no residual. RMSE pools individual measurements; relative RMSE divides by their observed mean. Every model was fitted anew. The two latent models received equal screening and confirmation budgets. The frozen partitions had been inspected in earlier analyses; current selection used validation only.

## replicate

| Model | Train RMSE / rRMSE | Validation RMSE / rRMSE | Test RMSE / rRMSE |
|---|---|---|---|
| PhytoODE | 0.7536 ± 0.0065 / 13.16 ± 0.11% | 0.7183 ± 0.0015 / 12.33 ± 0.03% | 0.7788 ± 0.0061 / 13.72 ± 0.11% |
| Latent ODE (no physics) | 0.7551 ± 0.0019 / 13.19 ± 0.03% | **0.7158 ± 0.0010 / 12.29 ± 0.02%** | 0.7815 ± 0.0045 / 13.77 ± 0.08% |
| Light-PINN | 0.7673 ± 0.0047 / 13.40 ± 0.08% | 0.7264 ± 0.0044 / 12.47 ± 0.08% | 0.7929 ± 0.0048 / 13.97 ± 0.08% |
| LSTM-NN | 0.7389 ± 0.0025 / 12.91 ± 0.04% | 0.7190 ± 0.0056 / 12.35 ± 0.10% | **0.7741 ± 0.0029 / 13.64 ± 0.05%** |
| Random forest | **0.7313 ± 0.0002 / 12.77 ± 0.00%** | 0.7283 ± 0.0005 / 12.51 ± 0.01% | 0.7783 ± 0.0006 / 13.71 ± 0.01% |
| Light-logistic ODE | 0.8207 / 14.34% | 0.8020 / 13.77% | 0.8435 / 14.86% |
| Logistic ODE | 0.8070 / 14.10% | 0.7713 / 13.24% | 0.8272 / 14.57% |
| Latent ODE (matched) | 0.7508 ± 0.0050 / 13.12 ± 0.09% | 0.7176 ± 0.0021 / 12.32 ± 0.04% | 0.7769 ± 0.0049 / 13.69 ± 0.09% |

All RMSE values are in mm. SD is across training seeds. Observations in the figures are individual measurements; only model predictions are averaged across seeds.

## time_holdout

| Model | Train RMSE / rRMSE | Validation RMSE / rRMSE | Test RMSE / rRMSE |
|---|---|---|---|
| PhytoODE | 0.6929 ± 0.0038 / 13.40 ± 0.07% | 0.6602 ± 0.0027 / 12.48 ± 0.05% | 0.9255 ± 0.0048 / 12.96 ± 0.07% |
| Latent ODE (no physics) | 0.6925 ± 0.0030 / 13.39 ± 0.06% | 0.6603 ± 0.0024 / 12.49 ± 0.05% | **0.9251 ± 0.0053 / 12.95 ± 0.07%** |
| Light-PINN | 0.6969 ± 0.0013 / 13.48 ± 0.03% | 0.6607 ± 0.0037 / 12.49 ± 0.07% | 0.9418 ± 0.0033 / 13.19 ± 0.05% |
| LSTM-NN | **0.6903 ± 0.0069 / 13.35 ± 0.13%** | **0.6591 ± 0.0062 / 12.46 ± 0.12%** | 0.9303 ± 0.0141 / 13.03 ± 0.20% |
| Random forest | 0.6913 ± 0.0009 / 13.37 ± 0.02% | 0.6598 ± 0.0012 / 12.48 ± 0.02% | 1.2663 ± 0.0005 / 17.73 ± 0.01% |
| Light-logistic ODE | 0.7666 / 14.83% | 0.7575 / 14.32% | 1.0138 / 14.20% |
| Logistic ODE | 0.7726 / 14.94% | 0.7359 / 13.92% | 0.9387 / 13.14% |
| Latent ODE (matched) | 0.6925 ± 0.0030 / 13.39 ± 0.06% | 0.6603 ± 0.0024 / 12.49 ± 0.05% | **0.9251 ± 0.0053 / 12.95 ± 0.07%** |

All RMSE values are in mm. SD is across training seeds. Observations in the figures are individual measurements; only model predictions are averaged across seeds.
