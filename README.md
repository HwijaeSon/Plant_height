# PhytoODE

**Learning plant growth dynamics across genotypes and observation regimes**

PhytoODE is a framework for learning plant growth dynamics from sparse and
incomplete observations. It combines genotype-conditioned latent neural ordinary
differential equations with logistic regularization of the predicted phenotype.
The framework is evaluated on wheat and maize height, Arabidopsis stem length,
and individual hypocotyl growth.

This repository provides the model implementations, dataset preparation,
pretrained models, and evaluation and visualization tools for the accompanying
paper. The Arabidopsis hypocotyl dataset collected in this study is included.

[Reproduction guide](REPRODUCTION.md) · [Hypocotyl dataset](data/hypocotyl/README.md) ·
[Pretrained model inventory](checkpoints/inventory.csv) · [Data sources](THIRD_PARTY.md)

## Installation

The code was evaluated with Python 3.12 and PyTorch 2.9. Models can be evaluated
on CPU; GPU training requires a compatible CUDA installation.

```bash
git clone --branch main https://github.com/HwijaeSon/Plant_height.git
cd Plant_height
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Datasets

| Dataset | Cohort | Training / validation / test |
|---|---|---|
| Wheat height | 19 genotypes | Growing seasons 2018–2019 / 2022 / 2021 |
| Maize relative UAV height | 402 genotypes, 3,072 plots | Growing seasons 2018–2019 / 2020 / 2021 |
| Arabidopsis stem length | 9 genotypes, 2 temperature regimes, 180 plants | Plants 1–6 / 7–8 / 9–10 within each genotype–temperature group |
| Arabidopsis hypocotyl length | 4 genotypes, 141 plants, 848 measurements | 0–36 h / 48 h / 60 and 72 h of each plant |

The wheat, maize and stem-length models predict growth from genotype, time and
temperature. The hypocotyl model forecasts each plant from its observed initial
lengths, observation masks, genotype and known light schedule. Each evaluation
uses genotypes represented in training.

### Published datasets

Download and prepare the three published datasets:

```bash
python src/scripts/download_data.py --dataset all
python src/scripts/prepare_data.py --dataset all
```

To prepare one dataset, replace `all` with `wheat`, `maize`, or `arabidopsis`.
Source URLs and file checksums are provided in
[src/configs/data_sources.json](src/configs/data_sources.json), with publication
and repository references in [THIRD_PARTY.md](THIRD_PARTY.md).

### Hypocotyl dataset

The included workbook contains measurements from Col-0, hy5, MLB and phyAB plants
grown at 23°C under a 12 h light / 12 h dark cycle. Lengths were measured every
12 h over 72 h. Each trajectory represents an individual plant; missing lengths
are handled with observation masks. The model uses available measurements from
0–36 h to forecast subsequent growth, with 48 h reserved for validation and
60/72 h for testing.

The workbook, individual-observation CSVs and experimental metadata are in
[data/hypocotyl](data/hypocotyl). Verify the workbook-to-CSV extraction with:

```bash
python src/scripts/prepare_hypocotyl.py
```

See the [dataset documentation](data/hypocotyl/README.md) for plant identifiers,
measurement units, missingness and experimental metadata.

## Evaluation

Evaluate the supplied PhytoODE checkpoints:

```bash
python run.py evaluate --dataset wheat --seed 1 --output outputs/wheat
python run.py evaluate --dataset maize --seed 1 --output outputs/maize
python run.py evaluate --dataset arabidopsis --seed 1 --output outputs/arabidopsis
python run.py evaluate --dataset hypocotyl --seed 101 --output outputs/hypocotyl
```

The three temperature benchmarks use seeds **1–3**. Hypocotyl PhytoODE and its
matched unregularized control use seeds **101–105**; the four reference baselines
use seeds **1–3**, with one Logistic ODE fit under natural missingness. Model
settings and checkpoint paths are listed in
[src/configs/paper.json](src/configs/paper.json).

Use `--model latent_ode` to evaluate the unregularized control. For hypocotyls,
`--drop-fraction 0.25` or `0.5` evaluates removal of 25% or 50% of the observed
initial measurements. For example:

```bash
python run.py evaluate --dataset hypocotyl --model latent_ode --seed 101 \
  --drop-fraction 0.5 --output outputs/hypocotyl-latent-ode-drop50
```

Pretrained PhytoODE and matched control models are available for all four
datasets. Some baseline results are provided as saved predictions without
original fitted weights. The [model inventory](checkpoints/inventory.csv) and
[reproduction guide](REPRODUCTION.md#fitted-model-availability) document
availability and baseline evaluation commands.

## Training

Train PhytoODE using the paper's configurations and validation-based checkpoint
selection:

```bash
python run.py train --dataset wheat --seed 1 --device cuda:0 \
  --output outputs/wheat-trained
python run.py train --dataset hypocotyl --seed 101 --device cuda:0 \
  --output outputs/hypocotyl-trained
```

For hypocotyl PhytoODE and its matched control, training evaluates the training
and validation partitions. Evaluate the selected checkpoint on all partitions
with a separate command:

```bash
python run.py evaluate --dataset hypocotyl --seed 101 \
  --checkpoint outputs/hypocotyl-trained/checkpoint.pt \
  --output outputs/hypocotyl-trained-evaluation
```

Use `--device cpu` for CPU training and `--model latent_ode` for the unregularized
control. Evaluation and training runs require a new output directory. Detailed
baseline commands, hyperparameters, normalization and missingness protocols are
provided in [REPRODUCTION.md](REPRODUCTION.md).

## Reproducing results

Generate the comparison tables and growth-curve figures from the recorded
predictions:

```bash
python src/scripts/report_results.py --output outputs/tables
python src/scripts/make_figures.py --output outputs/figures
```

Tables report training, validation and test RMSE and relative RMSE. Relative RMSE
is defined as `100 × RMSE / mean observed target` for the evaluated partition.
The reproduction guide specifies trajectory aggregation and observation masks
for each dataset. Figures include growth predictions, the hypocotyl missingness
comparison and fitted light/dark growth-rate parameters.

Verify file integrity and reproduce predictions from the supplied checkpoints:

```bash
python src/scripts/verify_submission.py
python src/scripts/verify_models.py --dataset all --output outputs/verified-models
```

The model verification command compares restored predictions with the recorded
arrays using floating-point tolerances. Numerical results from retraining may
vary across hardware and software environments.

## Repository structure

```text
src/           Model implementations, data processing, evaluation, plotting and configurations
data/          Hypocotyl measurements and downloaded external datasets
checkpoints/   Neural network weights, fitted forests and ODE parameters
results/       Recorded predictions, evaluation metrics and experiment metadata
run.py         Training and checkpoint evaluation entry point
```

Checkpoints and results are grouped by dataset. File names identify the model,
seed and, for hypocotyls, observation-removal level. Generated tables, figures
and new training runs are written to the specified output directory.

## Citation

Please cite **PhytoODE: Learning plant growth dynamics across genotypes and
observation regimes** and the original datasets when using this work.
Software citation metadata are available in [CITATION.cff](CITATION.cff).
Data and reference-model attribution are provided in
[THIRD_PARTY.md](THIRD_PARTY.md).
