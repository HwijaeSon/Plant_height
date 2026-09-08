# 현재 채택한 PhytoODE 설정 (2026-09-08)

옥수수는 사용자 지정으로 **λODE = 0.5, λK = 0.5**를 채택했다. 밀과 Arabidopsis는 공동 튜닝에서 선택된 계수를 유지한다. 아래 표는 기존 시드 1–3의 결과를 재사용하며, 새 학습이나 새 테스트 평가를 수행하지 않았다.

| 데이터셋 | λODE | λK |
|---|---:|---:|
| 밀 (m) | 3.16227766 | 0.1 |
| 옥수수 (상대 UAV 높이) | 0.5 | 0.5 |
| Arabidopsis (cm) | 0.5 | 0.1 |

설정은 [phytoode_config.json](../../phytoode_config.json), 학습 진입점은 [run_phytoode.py](../../run_phytoode.py)다. 모델 구조, 학습 횟수와 검증 기반 체크포인트 선택 규칙은 기존과 같다.

## Train / Validation / Test 비교

각 셀은 **RMSE / Relative error (%)**의 시드 3개 평균이다. 굵은 값은 각 데이터셋·분할에서 아래 두 모델 중 가장 낮은 값이다. 표준편차와 시드별 값은 [comparison.csv](comparison.csv), [per_seed_metrics.csv](per_seed_metrics.csv)에 있다.

| 데이터셋 | 모델 | Train | Validation | Test |
|---|---|---:|---:|---:|
| 밀 (m) | PhytoODE (adopted) | 0.02231 / 7.45% | **0.03768 / 11.30%** | **0.03032 / 10.24%** |
| 밀 (m) | Latent Neural ODE (no physics) | **0.02108 / 7.04%** | 0.03834 / 11.50% | 0.03091 / 10.44% |
| 옥수수 (상대 UAV 높이) | PhytoODE (adopted) | **41.897 / 13.43%** | **47.247 / 18.64%** | **56.684 / 18.40%** |
| 옥수수 (상대 UAV 높이) | Latent Neural ODE (no physics) | 43.252 / 13.86% | 60.753 / 23.97% | 67.781 / 22.00% |
| Arabidopsis (cm) | PhytoODE (adopted) | 1.421 / 6.75% | **3.191 / 15.85%** | **2.800 / 14.06%** |
| Arabidopsis (cm) | Latent Neural ODE (no physics) | **1.188 / 5.64%** | 3.278 / 16.28% | 2.993 / 15.03% |

Relative error는 각 곡선 RMSE의 평균을 해당 분할에서 점수를 계산한 전체 목표값의 평균으로 나눈 백분율이다(MAPE가 아님). 기존 관측 마스크와 단위를 유지했다. No-physics baseline은 λODE와 λK가 모두 0이다.

옥수수의 세 시드·세 분할 점수는 기존 예측값에서 다시 계산해 모두 일치함을 확인했다([검증 기록](verification.json)).

## 옥수수 선택 이력

공동 튜닝의 검증 기준 선택은 **λODE = 500, λK = 0.5**였으며, 그 기록은 [원래 보고서](../joint_lambda_tuning_20260908/README.md)에 보존했다. 해당 설정의 검증/시험 평균은 44.943 / 17.73%, 60.526 / 19.65%다. 현재 채택한 0.5 / 0.5의 검증/시험 평균은 47.247 / 18.64%, 56.684 / 18.40%다. 따라서 현재 채택값을 공동 튜닝의 검증 최적값으로 설명하지 않는다. 이미 확인한 테스트 세트의 후속 결과이며 독립적인 재검증을 뜻하지 않는다.

채택한 옥수수 체크포인트는 원래 `l160` 설정의 시드 1–3이며, 체크포인트 선택 epoch는 190, 730, 910이다. 시드 1은 기존 재현 실험의 동일 결과 체크포인트를 참조한다. 시드 2–3의 기존 로컬 체크포인트는 이 폴더에 복사해 버전 관리하며, 원본 경로와 SHA-256을 현재 설정 파일에 기록했다.

```bash
# 선택한 GPU에서 현재 옥수수 설정을 재학습하는 예시 (새 출력 폴더 사용)
.venv/bin/python experiments/run_phytoode.py \
  --dataset maize --seed 1 --gpu 3 \
  --output experiments/results/reproduced/maize_seed1
# --dry-run을 추가하면 실행할 설정만 확인한다.
```
