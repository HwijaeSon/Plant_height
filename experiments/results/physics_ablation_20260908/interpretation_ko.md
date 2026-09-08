# Physics loss 제거 실험 결과

**Physics loss가 없는 Latent Neural ODE baseline은
`lambda_ODE = lambda_K = 0`으로 두 항을 모두 제거한 모델이다.**
이 기준으로는 기존 PhytoODE가 세 데이터셋 모두 평균 test 오차가 낮다.
ODE 잔차만 제거하고 K loss를 유지한 모델은 별도의 부분 ablation이며,
physics loss 전체 제거 baseline으로 부르면 안 된다.

세 데이터셋에서 같은 구조, 초기 가중치, 데이터 분할, 학습 횟수와
validation 기반 체크포인트 선택 규칙을 유지했다. 제거 조건마다 시드
1–3을 실행했으며, 기존 PhytoODE의 시드 1도 다시 학습해 원래 선택 epoch와
train/validation/test 오차가 재현되는지 확인했다.

## 시험 결과

아래 값은 세 시드 평균의 **RMSE / Relative Error (%)**이다. ± 표본표준편차와
train/validation/test 전체 값은 [전체 보고서](README.md)에 있다.

| 조건 | 밀 (m / %) | 옥수수 (relative UAV unit / %) | Arabidopsis (cm / %) |
|---|---:|---:|---:|
| 기존 PhytoODE | 0.03063 / 10.34 | 56.68 / 18.40 | 2.866 / 14.39 |
| PhytoODE — K loss만 유지 (부분 ablation) | **0.03047 / 10.29** | **55.83 / 18.12** | **2.852 / 14.32** |
| Latent Neural ODE — physics loss 없음 (두 계수 모두 0) | 0.03091 / 10.44 | 67.78 / 22.00 | 2.993 / 15.03 |

첫 제거 조건은 logistic ODE 미분 잔차의 가중치만 0으로 만들고 최대높이
일치 항은 유지한다. 두 번째는 두 항의 가중치를 모두 0으로 만든 순수
데이터 학습 latent ODE이다. optimizer의 weight decay는 원래 값을 유지했다.

## 해석

- **ODE 미분 잔차의 추가적인 성능 개선은 이번 실험에서 확인되지 않았다.**
  잔차 제거 모델의 평균 test RMSE가 밀 0.51%, 옥수수 1.50%, Arabidopsis
  0.48% 낮았다. 차이가 작고, 밀과 Arabidopsis에서는 기존 모델의 오차가 더
  낮은 시드가 각각 2/3개였다. 세 시드 평균의 순위를 유의한 성능 차이로
  해석해서는 안 된다.
- **두 생물학적 손실을 모두 제거하면 평균 test RMSE가 증가했다.** 기존 모델
  대비 증가율은 밀 0.93%, 옥수수 19.58%, Arabidopsis 4.44%였다. 옥수수에서는
  세 시드 모두 순수 데이터 모델의 오차가 더 컸다.
- 미분 잔차가 없는 두 조건을 비교하면, 최대높이 일치 항을 유지한 모델이
  세 데이터셋 모두 평균 오차가 더 낮다. 특히 옥수수에서는 최대높이 제약의
  기여가 더 뚜렷하게 관찰된다. 두 손실을 함께 제거한 결과의 차이를 전부
  ODE 미분 잔차의 효과라고 설명할 수는 없다.

현재 결과로 본문에서 강조할 근거는 **연속시간 잠재 동역학 구조의 예측력과,
특정 조건에서 관찰되는 최대높이 정규화의 이점**이다. “ODE physics residual이
세 작물의 성능 향상을 이끈다”는 주장은 이 ablation으로 지지되지 않는다.

## 비교의 범위

기존 full 모델에 대해 선택된 하이퍼파라미터를 고정한 제거 실험이며,
각 제거 모델을 별도로 최적화한 결과가 아니다. 세 시드는 서로 다른
독립 실험 연도나 생물학적 반복 실험을 뜻하지 않는다. test 평균이 가장 낮은
제거 모델을 이 결과만 보고 새 최종 모델로 채택하면 test 기반 선택이 되므로,
기존 최종 모델을 교체하지 않고 ablation 결과로 보고한다.

Relative Error는 해당 분할의 평균 평가 타깃으로 기존 곡선 평균 RMSE를
나눈 rRMSE이며 MAPE가 아니다. 밀의 기존 평가 마스크에는 최초 관측 이전
채움값이 포함된다. 모든 조건이 동일한 마스크와 분모를 사용했다.

[LaTeX 표](ablation_table.tex) · [비교 그림](test_relative_errors.png) ·
[개별 시드 결과](per_seed_metrics.csv) · [검증 기록](validation.json)
