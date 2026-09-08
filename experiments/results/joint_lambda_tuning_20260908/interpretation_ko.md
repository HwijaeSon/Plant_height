# 두 physics 계수 공동 튜닝 결과

선택 기준은 seed 1–3의 평균 validation RMSE입니다. PhytoODE는 두 계수가 모두 양수인 후보에서 선택했고, 계수가 0인 후보까지 포함한 전체 1위도 별도로 기록했습니다.

**Wheat, Maize, Arabidopsis는 이전 ODE 계수 튜닝의 최적 조합을 유지합니다. 이번 탐색에서 새로 검증한 조합은 세 seed 평균 validation RMSE를 개선하지 못했습니다. 선택된 체크포인트와 test 오차도 기존과 같습니다. 정해진 탐색 범위와 예산 내 결과이며 전역 최적이라는 뜻은 아닙니다.**

- Wheat: PhytoODE (λODE, λK) = (3.16227766, 0.1); 전체 validation 1위 = (3.16227766, 0.1).
- Maize: PhytoODE (λODE, λK) = (500.0, 0.5); 전체 validation 1위 = (500.0, 0.5).
- Arabidopsis: PhytoODE (λODE, λK) = (0.5, 0.1); 전체 validation 1위 = (0.5, 0.1).

| 데이터셋 | 비교 대상 | Test RMSE 감소율 | 개선된 seed 수 |
|---|---|---:|---:|
| Wheat | Latent Neural ODE (no physics) | +1.91% | 3/3 |
| Wheat | Original PhytoODE | +1.00% | 3/3 |
| Wheat | PhytoODE (K loss only) | +0.49% | 2/3 |
| Wheat | PhytoODE (ODE coefficient tuned) | +0.00% | 0/3 |
| Maize | Latent Neural ODE (no physics) | +10.70% | 3/3 |
| Maize | Original PhytoODE | -6.78% | 1/3 |
| Maize | PhytoODE (K loss only) | -8.40% | 0/3 |
| Maize | PhytoODE (ODE coefficient tuned) | +0.00% | 0/3 |
| Arabidopsis | Latent Neural ODE (no physics) | +6.46% | 3/3 |
| Arabidopsis | Original PhytoODE | +2.31% | 3/3 |
| Arabidopsis | PhytoODE (K loss only) | +1.85% | 3/3 |
| Arabidopsis | PhytoODE (ODE coefficient tuned) | +0.00% | 0/3 |

감소율이 음수면 공동 튜닝의 test 오차가 증가한 것입니다. Validation 개선이 test 개선을 보장하지 않으며, seed 3개의 차이만으로 통계적 유의성이나 모든 환경에서의 우위를 주장할 수 없습니다.

Latent Neural ODE (no physics)는 λODE=λK=0입니다. K loss only는 λODE만 0인 별도의 부분 ablation입니다. 모든 비교는 동일한 latent ODE 구조를 유지하므로 ODE 적분 자체의 효과를 검증한 실험은 아닙니다.

기존에 확인한 test set을 다시 사용한 후속 실험입니다. 이번 탐색의 계수 선택에는 test 오차를 사용하지 않았습니다. RMSE / Relative RMSE의 전체 train·validation·test 표와 표준편차는 README.md와 joint_lambda_table.tex에 있습니다.
