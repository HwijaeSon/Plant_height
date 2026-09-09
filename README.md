# Genotype-conditioned latent Neural ODE for plant growth

This repository contains the data, code, and audited results used for the
wheat, maize and *Arabidopsis thaliana* experiments in the accompanying manuscript.
The files are organized by dataset:

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
├── hypocotyl/      # author-collected snapshots and a 12L12D-only evaluation
├── paper/          # LaTeX manuscript, bibliography, and final figures
└── legacy/         # exploratory and superseded experiments (git-ignored)
```

The prediction tasks use different generalization units. Wheat uses a
year-held-out split across 19 genotypes. Arabidopsis uses a plant-held-out
split across nine genotypes and two temperature regimes, with only four stem
measurements per plant. The latter is a low-time-resolution external
validation, not an unseen-genotype or unseen-temperature test.
The additional [hypocotyl experiment](hypocotyl/README.md) currently uses 943
12L12D snapshot measurements across five genotypes and seven times at 23°C.
The cR condition is excluded. Six methods receive genotype and elapsed time only,
with an ordinary logistic reference for PhytoODE and Logistic-PINN. Following the
restored initial protocol, primary training and evaluation targets are replicate
means calculated separately within each partition. Missing cells are omitted,
and missing time groups are not imputed. Individual-length RMSE is reported
separately. Source-row groups are not verified longitudinal plant identifiers.
A [PhytoODE-only follow-up](hypocotyl/reports/light_input_20260910/results.md)
adds the known binary illumination state, retains ordinary logistic physics,
and tunes its hyperparameters while keeping all earlier baseline fits and
evaluation targets unchanged. Both holdouts and a fixed-hyperparameter feature
comparison are included; this follow-up has an additional tuning budget.
The manuscript adopts its validation-selected model for the primary replicate
holdout, with 0.3460 mm / 6.85% test error. The author-collected dataset and
evaluation workflow are described as planned for public release with the article.

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

The [matched physics-loss ablation](experiments/README.md) compares the same
PhytoODE architectures and training settings with the logistic ODE residual
removed, and with both biological loss terms removed, on all three datasets.
It preserves paired initialization and validation-only checkpoint selection.

The subsequent [ODE-residual coefficient search](experiments/results/lambda_ode_tuning_20260908/README.md)
varies only the residual weight and selects it using three-seed validation
errors. It compares the selected positive weight with the original weight,
zero residual weight, and removal of both biological losses. All new candidate
training runs exclude test evaluation; final follow-up scores reuse the test
sets already inspected in the preceding ablation.
That historical search selected weights approximately 3.162 (wheat), 500
(maize), and 0.5 (Arabidopsis). The current manuscript subsequently retained
maize's original `(0.5, 0.5)` pair; see the
[adopted configurations](experiments/phytoode_config.json).
**Latent Neural ODE without physics loss sets both the ODE-residual
and maximum-height coefficients to zero.** Tuned PhytoODE has lower mean test
error than this baseline on all three datasets. The K-loss-only model retains
the maximum-height coefficient and is a separate partial ablation; it is not
the no-physics baseline. Against that partial ablation, tuned PhytoODE improves
wheat and Arabidopsis but worsens maize. Maize tuning also worsens test error
relative to original PhytoODE, so the joint-loss comparison does not establish
a uniform benefit from the ODE-residual term or from coefficient tuning.

## Audited paper results

- Wheat tuned latent Neural ODE (1,655 parameters; seeds 1--3): test RMSE
  `0.030322 ± 0.000662 m`.
- Maize adopted latent Neural ODE (seeds 1--3): test RMSE
  `56.684367 ± 4.369632` relative UAV-height units.
- Arabidopsis latent Neural ODE (seeds 1--3): test RMSE
  `2.799668 ± 0.016346 cm`.
- [12L12D-only hypocotyl results](hypocotyl/reports/single_condition_20260909/results.md)
  report both holdouts, all three splits, three-seed comparisons, and a matched
  no-physics control. The restored initial search uses previously inspected
  partitions and unequal candidate budgets across model families.
- Prediction figures are restricted to the observed time span; no temporal
  extrapolation is displayed.

## Before public release

The source datasets and baseline outputs retain their original provenance.
Confirm redistribution terms for every included data file, add the final
author/affiliation/funding information in the manuscript, and choose a code
license before publishing the repository.
