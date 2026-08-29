# Wheat compact Latent Neural ODE tuning

## Protocol

- Multiple-genotype one-hot setting with 19 wheat genotypes.
- Reference-matched split: train 2018/2019, validation 2022, test 2021.
- Fixed preprocessing, temperature input, RK4 solver, 1,500 epochs, minimum checkpoint epoch 600, and 10-evaluation validation smoothing.
- Architecture screen: 10 configurations, seeds 1--2; selected using validation RMSE only.
- Learning-rate/physics screen: two Pareto architectures, seeds 1--2; selected using validation RMSE only.
- Confirmation: the configurations were fixed before running seeds 3--5. Final numbers combine seeds 1--5.

## Final five-seed results

| Model | Architecture L/G/ODE/Dec/Enc | LR | Parameters | Reduction | Train RMSE (m) | Val RMSE (m) | Test RMSE (m) | Seconds/seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original Latent ODE | 16/4/32x2/32/16 | 0.005 | 4,647 | 0% | 0.02298 +/- 0.00047 | 0.04071 +/- 0.00262 | 0.03122 +/- 0.00066 | 295.0 |
| **Tuned bottleneck** | **16/4/16x1/16/8** | **0.01** | **1,655** | **64.4%** | 0.02238 +/- 0.00018 | **0.03873 +/- 0.00124** | **0.03081 +/- 0.00055** | 236.1 |
| Tuned micro | 4/2/8x1/8/4 | 0.01 | 393 | 91.5% | 0.02443 +/- 0.00062 | 0.04025 +/- 0.00337 | 0.03310 +/- 0.00157 | 235.2 |

L/G/ODE/Dec/Enc denotes latent dimension, genotype-embedding dimension, ODE width x depth, decoder width, and LSTM encoder width. All three final models use physics weight 2.0 and ymax weight 0.1.

## Reference-size comparison

| Model | Parameters | Test RMSE (m) |
|---|---:|---:|
| Tuned bottleneck | 1,655 | 0.03081 +/- 0.00055 |
| **Tuned micro** | **393** | **0.03310 +/- 0.00157** |
| Reference LSTM-NN, local five GitHub rows | 451 | 0.05760 +/- 0.01244 |
| Reference Logi-PINN, local five GitHub rows | 463 | 0.06060 +/- 0.01117 |

The micro model is 12.9% smaller than the reference LSTM and 15.1% smaller than Logi-PINN. Its mean test RMSE is 42.5% and 45.4% lower, respectively. The bottleneck model is 64.4% smaller than the original Latent ODE and has a 1.3% lower mean test RMSE. The paired five-seed bottleneck--original test difference is not statistically significant (two-sided paired t-test p=0.389), so the defensible conclusion is performance preservation under substantial compression, not a confirmed accuracy gain.

## Interpretation

1. **Recommended accuracy/size trade-off:** latent 16, genotype embedding 4, ODE width 16 with one hidden layer, decoder 16, encoder 8, LR 0.01, physics weight 2, ymax weight 0.1.
2. **Reference-size alternative:** the 393-parameter micro model remains much more accurate than the local reference outputs, but its test RMSE is 6.0% higher than the original Latent ODE and more variable across seeds.
3. Parameter reduction only shortened mean training time from 295.0 to 236.1 seconds/seed (20.0%), because integration over 170 RK4 time points dominates computation.
4. This tuning was conducted on one year split. A paper-level efficiency claim should confirm the fixed tuned architecture over the other five year splits without retuning on their test years.
