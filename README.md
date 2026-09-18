# PhytoODE

**Learning plant growth dynamics across genotypes and observation regimes**

Code, data access, fitted models, evaluation and visualization for the manuscript
version dated **18 September 2026** (`submission-20260918`). This snapshot contains
only the reported temperature benchmarks, the final individual hypocotyl
forecasting experiment, their baselines and reported loss ablations. Search runs,
previous hypocotyl experiments and manuscript source/build files are excluded.

[Reproduction guide](docs/REPRODUCTION.md) · [Hypocotyl dataset](hypocotyl/README.md) ·
[Model inventory](submission/models.csv) · [Data sources and attribution](THIRD_PARTY.md)

## Data and prediction tasks

| Dataset | Cohort | Training / validation / test |
|---|---|---|
| Wheat height | 19 genotypes | Growing seasons 2018–2019 / 2022 / 2021 |
| Maize relative UAV height | 402 genotypes, 3,072 plots | 2018–2019 / 2020 / 2021 |
| Arabidopsis stem length | 9 genotypes × 2 temperature regimes, 180 plants | Plants 1–6 / 7–8 / 9–10 in each group |
| Author-collected hypocotyl length | Col-0, hy5, MLB, phyAB; 141 plants, 848 observed lengths | 0–36 h / 48 h / 60 and 72 h of each plant |

The first three tasks predict height from genotype, time and temperature. The
hypocotyl task forecasts each plant from its available 0–36 h lengths, their
masks and genotype. The final PhytoODE also receives the known 12 h light / 12 h
dark schedule. Future measured lengths are never encoder inputs. Genotype
identities are represented in training; these are not unseen-genotype tests.

External measurements are downloaded from pinned repositories or publication
supplements; they are not redistributed here. The approved author-collected
workbook and its individual-observation CSVs are included. Hypocotyl missing
lengths remain missing: no mean targets, interpolation or length imputation.
See the reproduction guide for the distinct aggregation/scoring conventions of
the three existing benchmarks.

## Install and prepare

Python 3.12 was used. CPU evaluation works; training can use a compatible CUDA
PyTorch installation. GPU indices in commands refer to visible devices.

```bash
git clone https://github.com/HwijaeSon/Plant_height.git
cd Plant_height
git checkout submission-20260918
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_submission.py
python scripts/download_data.py --dataset all
python scripts/prepare_data.py --dataset all
```

Download/preparation can be restricted to `wheat`, `maize`, or `arabidopsis`.
Hypocotyl data need no download: `python scripts/prepare_hypocotyl.py` verifies
that extracting the released workbook reproduces the supplied CSVs. External
file URLs and SHA-256 checksums are in `configs/data_sources.json`.

## Evaluate trained PhytoODE models

```bash
python run.py evaluate --dataset wheat --seed 1 --output outputs/wheat
python run.py evaluate --dataset maize --seed 1 --output outputs/maize
python run.py evaluate --dataset arabidopsis --seed 1 --output outputs/arabidopsis
python run.py evaluate --dataset hypocotyl --seed 101 --output outputs/hypocotyl
python run.py evaluate --dataset hypocotyl --model latent_ode --seed 101 \
  --drop-fraction 0.5 --output outputs/hypocotyl-no-ode-drop50
```

Use seeds **1–3** for the three temperature datasets; use **101–105** for the
final hypocotyl PhytoODE and its otherwise identical unregularized control.
The latter changes only `lambda_ode` from 300 to 0; both have `lambda_k=0` and
1,194 parameters. The four hypocotyl reference baselines use seeds **1–3**
(Logistic ODE uses one fit without added missingness). Thus only the final
PhytoODE/control comparison uses the five matched initializations and masks.

Hypocotyl removal fractions are `0`, `0.25`, and `0.5`, corresponding to total
prefix missingness of 11.35%, 33.51%, and 55.67%. The two final models retain the
original training normalization scale across removals. The earlier reference
baselines retain their original per-mask normalization and settings.

`configs/paper.json` indexes checkpoint paths and hashes. The historical directory
names of temperature result files preserve provenance; only selected fits are
included. Some older baseline weights were never saved: all available weights,
process parameters and prediction archives are included, and
[the inventory](submission/models.csv) states availability explicitly.

## Train a model

```bash
python run.py train --dataset wheat --seed 1 --device cuda:0 \
  --output outputs/wheat-trained
python run.py train --dataset hypocotyl --seed 101 --device cuda:0 \
  --output outputs/hypocotyl-trained
python run.py evaluate --dataset hypocotyl --seed 101 \
  --checkpoint outputs/hypocotyl-trained/checkpoint.pt \
  --output outputs/hypocotyl-trained-evaluation
```

Training uses the retained configuration and validation-based checkpoint
selection. Hypocotyl PhytoODE/control training scores train/validation only;
future test evaluation is a separate command. Use `--model latent_ode` for the
unregularized architecture. Baseline training/evaluation commands and the
complete seed/removal protocol are in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).
Every command requires a new output directory.

## Tables, figures and verification

```bash
python scripts/report_results.py --output outputs/tables
python scripts/make_figures.py --output outputs/figures
python scripts/verify_models.py --dataset all --output outputs/restored-models
```

The report exports train/validation/test **RMSE / relative RMSE (%)** with the
lowest mean in bold, plus all hypocotyl missingness results. Figures reproduce
the three temperature growth comparisons, individual hypocotyl forecasts,
missingness comparison, and fitted phase-rate comparison. Generated PNG/PDF
figures stay in the local output directory.

The verification command restores all indexed PhytoODE/control checkpoints and
all hypocotyl reference fits and compares predictions with the recorded arrays.
See `scripts/evaluate_baselines.py` for additional fitted temperature baselines.
CPU/CUDA arithmetic can differ slightly; retraining is not guaranteed to be
bitwise identical. Standard deviations describe seed/mask variation, not
independent biological replication.

## Citation and reuse

Please cite the accompanying PhytoODE manuscript and the original data/model
sources listed in [THIRD_PARTY.md](THIRD_PARTY.md). Software citation metadata are
in `CITATION.cff`. This snapshot does not assign a licence to third-party data or
code. Dataset metadata identify information not recorded during collection.
