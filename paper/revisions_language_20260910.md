# Manuscript language revision — 10 September 2026

원고 전체의 어휘, 문장 구조, 시제를 AI·농업 응용 연구논문 문체로 수정했습니다.

- [전체 LaTeX — 복사·붙여넣기용](main_standalone.tex)
- [Overleaf 업로드용 원고](overleaf_20260910.zip)
- [수정된 PDF](main.pdf)

## 적용한 원칙

- 일반적 배경과 모델·수식의 정의는 현재형, 수행한 실험과 관측 결과는 과거형으로 서술했습니다.
- 데이터 공개는 완료된 것으로 표현하지 않고 공개 예정임을 명시했습니다.
- 결과는 평균 오차, 비교 대상, 평가 조건을 먼저 제시하고 해당 실험의 의미를 설명했습니다.
- 실험 진행 과정처럼 읽히는 `again achieved`, `the preceding experiment`,
  `bookkeeping assignments`, `Four limitations define the next experiments`
  등의 표현을 구체적인 연구 방법과 결과 중심으로 수정했습니다.
- 본문에서는 `unregularized latent ODE`, `mean test RMSE`, `rRMSE`,
  `sparse temporal sampling` 등의 용어를 일관되게 사용했습니다.
  표와 그림의 기존 모델 명칭은 유지했습니다.
- 표, 수식, 인용 키, 그림 파일을 보존했습니다. 데이터 분할, 결측 처리,
  하이퍼파라미터, 이전 test 확인, 모델별 입력 및 탐색 예산의 차이도 유지했습니다.

## 표현 예시

| 수정 전 | 수정 후 |
|---|---|
| On our author-collected hypocotyl dataset, PhytoODE again achieved… | PhytoODE achieved the lowest mean validation and test errors on the hypocotyl dataset collected in this study. |
| The light-input PhytoODE follow-up screens 32 configurations… | For PhytoODE with illumination input, 32 configurations were evaluated with seed 1. |
| RF again fitted the training data most closely… | RF attained the lowest training error and the highest test error. |
| Four limitations define the next experiments. | Several aspects of the evaluation limit the scope of these conclusions. |

## 편집 및 내보내기

`main.tex`, `hypocotyl_methods.tex`, `hypocotyl_results.tex`를 직접 편집한 뒤
아래 명령으로 본문을 보존하면서 통합 LaTeX와 Overleaf 파일을 갱신합니다.

```bash
.venv/bin/python paper/export_manuscript.py --date 20260910
```

이 명령은 학습을 재실행하거나 표·그림의 수치를 재생성하지 않습니다.
이전 `revisions_20260910.md`는 실험 결과 갱신 당시의 문안을 기록한 파일입니다.
최신 본문은 위의 통합 LaTeX를 사용합니다.
