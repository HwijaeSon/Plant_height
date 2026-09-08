# 옥수수 데이터 준비: Sweet et al. (2024)

논문 DOI: [10.1111/tpj.17092](https://onlinelibrary.wiley.com/doi/10.1111/tpj.17092).
다운로드·가공 및 실험일: 2026-09-04. 데이터 준비와 6개 모델 비교를 완료했다.

주 비교 세트는 **402 유전자형, 3,072 시험구 시계열, 31,894 실제 관측값**이다.
2018–2019년 학습 / 2020년 검증 / 2021년 시험으로 구성했다.
높이는 **상대 UAV 높이**이며 cm/m로 해석하면 안 된다.

현재 PhytoODE는 **λODE = 0.5, λK = 0.5**를 사용한다. 기존 해당 설정의
시드 1–3 결과를 재사용하며, 2021년 시험 RMSE는 **56.684 ± 4.370**,
Relative error는 **18.40 ± 1.42%**다. [현재 설정](../experiments/phytoode_config.json)과
[Train/Validation/Test 비교표](../experiments/results/adopted_phytoode_20260908/README.md)에
재현 명령과 계수 선택 이력을 정리했다.

초기 튜닝 전 비교에서 2021년 시험 RMSE는 우리 모델 **79.611 ± 13.117**, 가장 좋은 baseline인
LSTM-NN **98.767 ± 10.196**으로, 세 시드 평균 기준 19.40% 감소했다.
2020년 검증 RMSE는 LSTM-NN이 더 낮다. 현재 결과는 한 개 시험 연도의 비교이며
여러 환경에 대한 일관된 우위를 뜻하지 않는다.

- [모델 비교 보고서](results/chronological_final_seed1_3/comparison.md), [비교 CSV](results/chronological_final_seed1_3/comparison.csv).
- [개별 실행 결과](results/chronological_final_seed1_3/all_runs.csv), [비교 그림](results/chronological_final_seed1_3/figures/comparison.png).
- [예측 그림과 설명](results/chronological_final_seed1_3/figures/README.md): 오차 수준별 성장 곡선 6개, 전체 유전자형·관측일별 오차 비교, PNG/PDF 및 원자료 CSV.
- [우리 모델 하이퍼파라미터 튜닝 결과](results/tuning_20260904/final/README.md): 약 3시간, 67개 설정·95회 실행. 시험 rRMSE 25.84±4.26% → 18.40±1.42%.
- [지표·선택 검증](results/chronological_final_seed1_3/validation.json), [체크포인트 재현 검증](results/chronological_final_seed1_3/model_validation.json).

- [데이터 분석 보고서](reports/dataset_report.md): 연도별 수, 제외 사유, 부록 불일치, 실험 설계.
- [분석 그림](reports/figures/dataset_overview.png), [PDF](reports/figures/dataset_overview.pdf).
- [학습용 관측 테이블](data/processed/height_observations.csv).
- [모델 입력 배열](data/processed/model_ready/chronological.npz), [배열·정규화 명세](data/processed/model_ready/chronological.json).
- [파일 설명](data/README.md), [원본 출처·SHA-256](data/download_manifest.json).
- [검증 결과](reports/validation.json).

프로젝트 루트에서 기존 가상환경으로 재현한다:

```bash
.venv/bin/python maize/code/download_data.py
.venv/bin/python maize/code/prepare_data.py
.venv/bin/python maize/code/data.py
.venv/bin/python maize/code/verify_data.py
.venv/bin/python maize/code/plot_dataset.py
```

다른 연도 분할은 `data/processed/year_splits.json`에 정의된 이름으로 지정한다.
예: `.venv/bin/python maize/code/data.py --split test2020_val2021`.
각 분할의 스케일러는 해당 학습 연도로 다시 계산한다.

`code/data.py`의 `make_dataset()`이 반환하는 `y, mask, env, s, ds, g_idx`는
기존 `wheat/code/model.py` 입력 형식과 맞는다. 실제 forward pass를 검증했다.
`code/run_experiment.py`가 이 로더와 기존 모델 구조를 연결한다.

초기 튜닝 전 학습 및 비교표를 재현하려면 비어 있는 GPU 세 개를 지정한다:

```bash
bash maize/code/run_all.sh 1 2 6
```

모델은 Logi-ODE, Temp-ODE, RF, LSTM-NN, Logi-PINN, Latent Neural ODE다.
결정론적 process 모델은 1회, 나머지는 시드 1–3으로 총 14회 실행했다.
Neural ODE 1,500 epoch, LSTM/PINN 3,000 epoch를 모두 실행하고,
매 10 epoch의 검증 RMSE가 가장 낮은 checkpoint를 선택했다.
PINN은 500 epoch warm-up 이후 checkpoint만 선택 대상이다.

코드·입력 파일 SHA-256과 설정은 결과 폴더의 `protocol.json`에 기록했다.
이전 설정 점검과 smoke 실행은 [결과 폴더 안내](results/README.md)에 명시했으며,
최종 비교표에는 포함하지 않는다.
