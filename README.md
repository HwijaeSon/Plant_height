# Genotype-conditioned latent Neural ODE for plant growth

This repository contains the data, code, and audited results used for the
wheat and *Arabidopsis thaliana* experiments in the accompanying manuscript.
The public-facing files are organized by dataset:

```text
.
├── wheat/
│   ├── data/       # processed field data and kinship matrix
│   ├── code/       # latent Neural ODE training and analysis
│   └── results/    # 3-seed paper outputs, checkpoints, tables, figures
├── arabidopsis/
│   ├── data/       # supplementary workbook and processed stem lengths
│   ├── code/       # preprocessing, baselines, training, plotting
│   └── results/    # 3-seed paper outputs, fitted ODE parameters, figures
├── maize/          # Sweet et al. (2024) data and 3-seed model comparison
├── paper/          # LaTeX manuscript, bibliography, and final figures
└── legacy/         # exploratory and superseded experiments (git-ignored)
```

The two prediction tasks are deliberately different. Wheat uses a
year-held-out split across 19 genotypes. Arabidopsis uses a plant-held-out
split across nine genotypes and two temperature regimes, with only four stem
measurements per plant. The latter is a low-time-resolution external
validation, not an unseen-genotype or unseen-temperature test.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dataset-specific commands and result provenance are documented in
[`wheat/README.md`](wheat/README.md) and
[`arabidopsis/README.md`](arabidopsis/README.md). The manuscript notes are in
[`paper/README.md`](paper/README.md).

The additional [maize dataset](maize/README.md) includes a completed
year-held-out comparison: 402 shared genotypes, 3,072 plot trajectories, and
31,894 observed targets. Its UAV heights use relative units, not metres.
Source-file discrepancies and preprocessing are documented in the
[maize data report](maize/reports/dataset_report.md).
The [maize model comparison](maize/results/chronological_final_seed1_3/comparison.md)
reports a 2021 test RMSE of `79.611 ± 13.117` relative-height units for our
latent Neural ODE versus `98.767 ± 10.196` for LSTM-NN (seeds 1--3).
The subsequent [ours-only maize tuning](maize/results/tuning_20260904/final/README.md)
selected the configuration using 2020 validation only and reduced its 2021
test rRMSE from `25.84 ± 4.26%` to `18.40 ± 1.42%`.

The [three-dataset relative-error comparison](paper/relative_errors/README.md)
expresses the existing RMSE and MAE as percentages of each evaluation split's
mean target. Its main maize `Ours` row now uses that validation-selected tuned
result; the pre-tuning row remains in the full table. It includes all available
baselines and explicitly documents the initial fill points in the original
wheat scoring mask.

## Audited paper results

- Wheat tuned latent Neural ODE (1,655 parameters; seeds 1--3): test RMSE
  `0.030629 ± 0.000620 m`.
- Arabidopsis latent Neural ODE (seeds 1--3): test RMSE
  `2.865980 ± 0.092808 cm`.
- Prediction figures are restricted to the observed time span; no temporal
  extrapolation is displayed.

## Before public release

The source datasets and baseline outputs retain their original provenance.
Confirm redistribution terms for every included data file, add the final
author/affiliation/funding information in the manuscript, and choose a code
license before publishing the repository.
