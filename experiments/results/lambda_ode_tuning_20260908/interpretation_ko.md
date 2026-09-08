# ODE loss 계수 튜닝 결과 해석

**Physics loss가 없는 Latent Neural ODE(lambda_ODE = lambda_K = 0) 대비, 현재 PhytoODE의 test 평균 오차는 밀, 옥수수, Arabidopsis에서 감소했다.**

## 비교 모델의 정의

- **Latent Neural ODE — physics loss 없음:** lambda_ODE = 0, lambda_K = 0. 두 생물학적 손실을 모두 제거하며, 동일한 latent ODE 구조와 optimizer weight decay는 유지한다. 기존 `no_biological_loss` 실험이 이 baseline이다.
- **PhytoODE — K loss만 유지:** lambda_ODE = 0, lambda_K > 0. ODE 잔차만 제거한 부분 ablation이며, physics loss를 전부 제거한 baseline이 아니다.
- **PhytoODE:** 두 손실을 모두 유지한다. 이번 튜닝은 lambda_K를 고정하고 lambda_ODE만 선택했다.

앞선 요약에서 K loss를 유지한 부분 ablation을 중심으로 비교한 탓에, 원래 요청한 physics loss 전체 제거 baseline과 혼동될 수 있었다. 표·그림·캡션의 명칭과 주 비교 대상을 바로잡았다. 두 계수를 모두 0으로 둔 9개 기존 학습을 재사용하며 수치, checkpoint, 계수 선택은 변경하지 않았다.

이번 실험에서는 latent Neural ODE 구조를 유지하고 logistic ODE 잔차 손실의 계수만 조절했다. 최대높이 손실 계수는 밀·Arabidopsis 0.1, 옥수수 0.5로 고정했다. 데이터 loss만 사용하는 비교 모델도 latent ODE 구조를 사용한다.

손실은 `L_data + lambda_ODE * L_ODE + lambda_K * L_K`이다. 계수가 크더라도 ODE 손실이 학습을 지배한다는 뜻은 아니다. 데이터 손실은 높이의 RMSE인 반면 ODE 잔차는 일별 성장률의 제곱오차이므로 수치와 단위가 다르다. 여기서 높이는 모델 내부의 단위이며 옥수수는 정규화된 높이를 사용한다. 실제 가중 손실 크기는 `loss_contributions.csv`에 기록했다. 손실 크기와 gradient 영향력도 구분해야 한다.

## 데이터셋별 결과

### 밀

Validation 평균 RMSE로 선택된 ODE 계수는 2 → **3.16228**이다. 선택 모델의 validation RMSE는 0.0376804, physics loss가 없는 baseline은 0.0383418이다. lambda_K를 원래 값으로 고정한 ODE 계수 탐색(0 포함)의 최적값은 **3.16228**이다.

선택 모델의 test RMSE는 **0.0303217 ± 0.00066237 m**, relative RMSE는 **10.237 ± 0.224%**이다. Physics loss가 없는 baseline 대비 test 평균 오차 감소율은 +1.91%이며, 같은 시드끼리 비교하면 3/3개에서 더 낮다.

별도로 K loss만 유지한 부분 ablation 대비 감소율은 +0.49%, 튜닝 전 PhytoODE 대비 감소율은 +1.00%이다. 감소율이 음수이면 현재 튜닝 모델의 오차가 더 높다는 뜻이다.

### 옥수수

Validation 평균 RMSE로 선택된 ODE 계수는 0.5 → **500**이다. 선택 모델의 validation RMSE는 44.9427, physics loss가 없는 baseline은 60.7529이다. lambda_K를 원래 값으로 고정한 ODE 계수 탐색(0 포함)의 최적값은 **500**이다.

선택 모델의 test RMSE는 **60.5258 ± 7.49953 relative UAV height**, relative RMSE는 **19.646 ± 2.434%**이다. Physics loss가 없는 baseline 대비 test 평균 오차 감소율은 +10.70%이며, 같은 시드끼리 비교하면 3/3개에서 더 낮다.

별도로 K loss만 유지한 부분 ablation 대비 감소율은 -8.40%, 튜닝 전 PhytoODE 대비 감소율은 -6.78%이다. 감소율이 음수이면 현재 튜닝 모델의 오차가 더 높다는 뜻이다.

### Arabidopsis

Validation 평균 RMSE로 선택된 ODE 계수는 2 → **0.5**이다. 선택 모델의 validation RMSE는 3.19147, physics loss가 없는 baseline은 3.27778이다. lambda_K를 원래 값으로 고정한 ODE 계수 탐색(0 포함)의 최적값은 **0.5**이다.

선택 모델의 test RMSE는 **2.79967 ± 0.0163463 cm**, relative RMSE는 **14.058 ± 0.082%**이다. Physics loss가 없는 baseline 대비 test 평균 오차 감소율은 +6.46%이며, 같은 시드끼리 비교하면 3/3개에서 더 낮다.

별도로 K loss만 유지한 부분 ablation 대비 감소율은 +1.85%, 튜닝 전 PhytoODE 대비 감소율은 +2.31%이다. 감소율이 음수이면 현재 튜닝 모델의 오차가 더 높다는 뜻이다.

## 논문에서 주장할 수 있는 범위

현재 결과는 보고된 설정에서 두 physics loss를 포함한 PhytoODE가 physics loss를 모두 제거한 동일한 latent Neural ODE보다 세 데이터셋의 평균 test 오차가 낮음을 보여준다. 이 비교는 두 손실의 공동 효과를 평가한다. ODE 잔차 항 자체가 모든 데이터셋에서 이롭다거나, lambda_ODE 튜닝이 모든 데이터셋의 test 오차를 줄였다는 의미는 아니다. 특히 옥수수에서는 튜닝 후 모델이 K-loss-only 모델 및 튜닝 전 PhytoODE보다 test 오차가 높다. 계수는 미리 정한 validation 절차로 선택했으며 test 순위로 다시 바꾸지 않았다.

이전 ablation의 test 결과를 확인한 뒤 시작한 후속 튜닝이다. 따라서 이번 test 점수는 기존 test set을 재사용한 평가이며, 새로운 독립 검증으로 제시하면 안 된다. 3개 시드의 표준편차는 초기화 변동성이고 통계적 유의성 또는 새로운 연도에 대한 불확실성을 확정하지 않는다. 독립 연도 또는 반복된 외부 분할로 확인하면 physics loss의 일반화 효과를 더 강하게 주장할 수 있다.

ODE 잔차만 제거한 비교는 최대높이 손실이 있는 조건에서 잔차 항의 효과를 평가한다. 두 biological loss를 모두 제거한 비교는 두 항의 공동 효과이므로, 그 차이를 전부 ODE 잔차 덕분이라고 해석해서는 안 된다. 이번 결과만으로 latent ODE 구조 자체의 필요성을 입증할 수도 없다.

Train / validation / test 전체 비교는 `README.md`, 기존 baseline까지 포함한 표는 `all_baselines.md`에 있다. Physics loss 전체 제거 baseline과의 주 비교 LaTeX 표는 `physics_vs_latent_ode_table.tex`이며, 부분 ablation까지 포함한 표는 `lambda_ode_table.tex`이다. 기존 원고의 결과와 이번 후속 튜닝 결과는 별도 파일로 보존했다.

## 실행 중단과 복구

밀의 계수 3.162277660, 시드 2 학습이 578 epoch에서 종료 신호로 중단되어 동일한 초기값과 전체 1,500 epoch 일정으로 재실행했다. 중단된 부분 실행은 완료 결과에 포함하지 않았고, 재실행의 초기 학습 기록이 중단 전 기록과 일치함을 검증했다. 기존 45개 완료 결과, 후보 목록, 계수 선택 기준과 test 평가 규칙은 유지했다. 종료 신호의 발신 원인은 확인되지 않았다.
