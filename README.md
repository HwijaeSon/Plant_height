# PhytoODE

**Learning plant growth dynamics across genotypes and observation regimes**

PhytoODE combines genotype-conditioned latent neural ODE dynamics with logistic
regularization of the predicted phenotype. This code and data release contains
the three published temperature benchmarks and a hypocotyl forecasting benchmark
restricted to **Col-0, hy5, MLB, and phyAB**.

The current version is `submission-20260915-four-genotypes`. All six hypocotyl
predictors were retrained from scratch on this approved cohort with the existing
manuscript settings. The earlier hypocotyl fitted models and reports are not part
of this release. Temperature-dataset results are unchanged.

[Results](hypocotyl/reports/four_genotypes_20260915/comparison.md) ·
[Hypocotyl data and dictionary](hypocotyl/README.md) ·
[Reproduction guide](docs/REPRODUCTION.md)

## Datasets and tasks

| Dataset | Cohort | Training / validation / test |
|---|---|---|
| Wheat height | 19 genotypes | 2018–2019 / 2022 / 2021 |
| Maize relative UAV height | 402 genotypes | 2018–2019 / 2020 / 2021 |
| Arabidopsis stem length | Nine genotypes, two temperature regimes, four observations per plant | Plants 1–6 / 7–8 / 9–10 within each genotype–temperature group |
| Hypocotyl length | Four genotypes, 141 plants, 848 observations | 0–36 h / 48 h / 60 and 72 h |

The first three tasks use genotype, time, and a known temperature sequence.
Hypocotyl models use genotype, elapsed time, observed early lengths, and their
observation masks. They forecast the same plants at subsequent times without
using future lengths as inputs. All evaluated genotype identities occur in
training. Hypocotyl PhytoODE uses no explicit environmental channel.

## Install and verify

```bash
python -m pip install -r requirements.txt
python scripts/verify_submission.py
python scripts/prepare_hypocotyl.py
python hypocotyl/code/audit_approved_benchmark.py
```

The released workbook contains approved measurement cells only and preserves
their source sheet and cell coordinates. It has no embedded summary charts.
The continuous-red-light sheet contains only the same approved genotypes and
is excluded from the manuscript forecasting task.

## Evaluate and train

```bash
# Evaluate a bundled model.
python run.py evaluate --dataset hypocotyl --model phytoode --seed 1 \
  --output outputs/hypocotyl-evaluation

# Train PhytoODE; this stage uses training/validation data only.
python run.py train --dataset hypocotyl --model phytoode --seed 1 \
  --device cuda:0 --output outputs/hypocotyl-training

# Score the validation-selected checkpoint at the future test times.
python run.py evaluate --dataset hypocotyl --model phytoode --seed 1 \
  --checkpoint outputs/hypocotyl-training/checkpoint.pt \
  --output outputs/hypocotyl-new-evaluation

# Retrain all six models, three seeds, and three missingness conditions.
python hypocotyl/code/run_approved_benchmark.py --gpus 0 \
  --output outputs/four-genotype-benchmark
```

The five baseline names are `latent_ode`, `logistic_pinn`, `lstm`, `rf`, and
`logistic`. Add `--drop-fraction 0.25` or `0.5` for additional missingness. All
models share the same masks for each seed and removal level, with at least one
early observation per plant. The benchmark performs 52 fits: stochastic models
use three seeds; Logi-ODE is fitted once naturally and once per added mask.

Model settings and checkpoint paths are in `configs/paper.json`. Hypocotyl
PhytoODE retains logistic coefficient 100 and no capacity penalty, with an
individual parameter network receiving the early lengths and their masks.
Checkpoint epochs are selected using 48-h validation. This fixed-setting rerun
does not establish independent biological validation of the model configuration.

## Reports and plots

```bash
python scripts/report_results.py --output outputs/tables
python hypocotyl/code/report_approved_benchmark.py \
  --output outputs/hypocotyl-report --figures outputs/hypocotyl-figures
python hypocotyl/code/report_approved_benchmark.py \
  --results outputs/four-genotype-benchmark \
  --output outputs/retrained-report --figures outputs/retrained-figures
```

The individual-trajectory plot contains one plant from each of the four
genotypes. Both its predictions and the missingness error curves use only the
newly fitted models. RMSE is averaged over individual plants; relative RMSE
divides this score by the pooled observed target mean. Standard deviations
describe initialization and mask variation, rather than biological-cohort
uncertainty.

## Published data sources

External inputs can be downloaded and prepared using the scripts described in
[the reproduction guide](docs/REPRODUCTION.md). Their source records and checksums
are in `configs/data_sources.json`.

- Wheat: [Roth et al.](https://doi.org/10.34133/plantphenomics.0185), processed
  following [Shao et al.](https://doi.org/10.1016/j.compag.2026.111988) and the
  [reference implementation](https://github.com/YingjieShao/PINN_for_plant_height_forecasting).
- Maize: [Sweet et al.](https://doi.org/10.1111/tpj.17092), with the
  [data deposit](https://doi.org/10.13020/SKJN-QX31).
- Arabidopsis stem length: [Ebrahimi Naghani et al.](https://doi.org/10.1186/s12870-024-05394-w).

See `THIRD_PARTY.md` for attribution and reuse information. A dataset DOI and
explicit reuse licence have not been assigned to the author-collected data.
