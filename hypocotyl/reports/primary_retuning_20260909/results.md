# Hypocotyl primary evaluation after expanded PhytoODE tuning

The same frozen replicate-group partitions are used throughout. This is an exploratory follow-up after the original test results had been inspected. The 80-candidate search and final choice used validation scores only. Baselines retain the initial tuning budgets; the matched no-physics control uses the final PhytoODE architecture and optimizer settings.

| Model | Train RMSE / relative error | Validation RMSE / relative error | Test RMSE / relative error |
|---|---|---|---|
| PhytoODE | 0.211 ± 0.010 / 3.64 ± 0.18% | **0.300 ± 0.001 / 5.14 ± 0.03%** | 0.337 ± 0.002 / 5.81 ± 0.04% |
| Latent ODE (no physics) | 0.212 ± 0.024 / 3.66 ± 0.41% | 0.311 ± 0.006 / 5.33 ± 0.10% | 0.337 ± 0.003 / 5.81 ± 0.05% |
| Light-PINN | 0.204 ± 0.019 / 3.52 ± 0.33% | 0.318 ± 0.003 / 5.45 ± 0.05% | 0.347 ± 0.015 / 5.99 ± 0.26% |
| LSTM-NN | **0.148 ± 0.010 / 2.55 ± 0.17%** | 0.319 ± 0.010 / 5.47 ± 0.17% | 0.352 ± 0.004 / 6.07 ± 0.07% |
| Random forest | 0.290 ± 0.008 / 5.00 ± 0.13% | 0.391 ± 0.004 / 6.70 ± 0.07% | 0.401 ± 0.011 / 6.91 ± 0.19% |
| Light-logistic ODE | 0.361 / 6.22% | 0.446 / 7.64% | 0.465 / 8.03% |
| Logistic ODE | 0.338 / 5.82% | 0.402 / 6.89% | 0.439 / 7.57% |
| Latent ODE (matched architecture and training) | 0.219 ± 0.011 / 3.78 ± 0.19% | 0.300 ± 0.002 / 5.14 ± 0.03% | 0.340 ± 0.005 / 5.87 ± 0.09% |
| Latent ODE (initial control) | 0.173 ± 0.019 / 2.98 ± 0.33% | 0.314 ± 0.009 / 5.37 ± 0.15% | **0.335 ± 0.002 / 5.78 ± 0.03%** |

RMSE is in mm. Bold identifies unrounded minima among all rows; some differences are below the displayed precision. The final two rows are the current matched control and the original same-learning-rate control. Both physics coefficients are zero in every no-physics comparison. SD denotes variation across three training seeds, not biological uncertainty.

## Validation-selected PhytoODE configuration

```json
{
  "model": "phytoode",
  "lr": 0.0015527014849295356,
  "lambda_ode": 0.12644235454360606,
  "lambda_k": 0.002702328745365828,
  "weight_decay": 0.00036574216923513037,
  "epochs": 3000,
  "architecture": {
    "latent_dim": 12,
    "g_embed_dim": 4,
    "ode_hidden": 24,
    "ode_layers": 1,
    "dec_hidden": 24,
    "enc_hidden": 12
  }
}
```

## Before / after

The initial primary PhytoODE test result was 0.343717 mm / 5.931501%. The original independently learning-rate-selected latent ODE remains 0.336836 mm / 5.812771%. These previously inspected values did not determine the new candidate ranking or stopping rule.

Unrounded test means: PhytoODE 0.33677124 mm / 5.811645%; original latent ODE 0.33683649 mm / 5.812771%; matched latent ODE 0.34044009 mm / 5.874958%.

The gap from the original latent ODE is only 0.00006525 mm (0.019%), much smaller than the seed standard deviations: the methods are effectively tied in this comparison. Relative to the matched control, PhytoODE mean test RMSE is 1.078% lower. Three seeds and a reused test partition do not establish statistical superiority.

The earlier no-physics control at the initial architecture and learning rate 0.01 achieved 0.33470282 mm / 5.775950%, which is lower than the newly tuned PhytoODE result. It remains a historical control, not a baseline selected using test error: its original validation mean was worse than the independently selected learning rate 0.003. The expanded search therefore did not establish superiority over all evaluated latent ODE configurations.
