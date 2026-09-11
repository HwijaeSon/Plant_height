# PhytoODE

**Learning plant growth dynamics across genotypes and observation regimes**

This is the submission snapshot of the code, selected models, and author-collected
dataset for the PhytoODE manuscript. PhytoODE combines genotype-conditioned latent
neural ODE dynamics with logistic derivative and carrying-capacity regularization
in phenotype space. Environmental inputs are temperature for the published
datasets and binary illumination for the hypocotyl experiment.

This branch contains the experiments reported in the manuscript. The development
history and exploratory experiments remain on the repository's `main` branch.
The final configuration is defined by [`configs/paper.json`](configs/paper.json).
The fixed submission version is tagged `submission-20260911`. Verification
results are recorded in [`submission/validation.json`](submission/validation.json).

## Installation

```bash
git clone --depth 1 --branch codex/submission-20260911 \
  https://github.com/HwijaeSon/Plant_height.git
cd Plant_height
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If GitHub requires authentication for this repository, use an account with
repository access; pushing this snapshot does not change repository visibility.

The verified environment uses Python 3.12 and PyTorch 2.9.0; package versions are
pinned in `requirements.txt`. CPU execution is supported. For a CPU-only PyTorch
installation, install `torch==2.9.0` from the
[official CPU wheel index](https://download.pytorch.org/whl/cpu) before installing
the remaining requirements. GPU execution requires a compatible CUDA-enabled
PyTorch installation. The reported models were trained with CUDA; changes in
hardware or numerical libraries can change retraining results.

All commands below run from the repository root. New outputs go under the ignored
`outputs/` directory. A run refuses to overwrite an existing output directory.

## Quick start: our hypocotyl dataset

The author-collected data and the selected checkpoints are included; no external
download is needed for this example.

```bash
# Verify extraction from the workbook, partition membership, and mean targets.
python scripts/prepare_hypocotyl.py

# Evaluate the submitted PhytoODE checkpoint on CPU.
python run.py evaluate --dataset hypocotyl --seed 1 \
  --output outputs/hypocotyl-evaluation

# Check forward propagation, physics derivatives, and backpropagation.
python run.py smoke --dataset hypocotyl --output outputs/hypocotyl-smoke

# Retrain the selected configuration for its complete 1,500-epoch schedule.
python run.py train --dataset hypocotyl --seed 1 --device cuda:0 \
  --output outputs/hypocotyl-training
```

Replace `--device cuda:0` with `--device cpu` to train without a GPU. Smoke runs
perform two optimizer steps, evaluate training/validation only, and are explicitly
labelled **not paper results**. They retain the full learning-rate schedule.

## Datasets and evaluation protocols

| Dataset | Included observations | Evaluation unit | Environment | Reported height unit |
|---|---|---|---|---|
| Wheat | 19 genotypes across four field seasons | Train: 2018–2019; validation: 2022; test: 2021 | Air temperature | m |
| Maize | 402 genotypes, 3,072 plot trajectories, 31,894 measurements | Train: 2018–2019; validation: 2020; test: 2021 | Air temperature | Relative UAV height |
| Arabidopsis stem length | 9 genotypes, two temperature regimes, 180 plants, four measurements per plant | Plants 1–6: train; 7–8: validation; 9–10: test | Temperature regime | cm |
| Author-collected hypocotyl length | 943 measurements, five genotypes, seven times | 547/171/225 measurements in train/validation/test; 35 mean targets per partition | Binary light under 12L12D at 23°C | mm |

All evaluated genotype identities occur during training. The hypocotyl manuscript
experiment uses **12L12D only and replicate-group holdout**. Continuous red light
and the exploratory 36/60-hour holdout are excluded from the submission evaluation.
Details and the data dictionary are in [`hypocotyl/README.md`](hypocotyl/README.md).

### Download the published datasets

```bash
python scripts/download_data.py --dataset all
python scripts/prepare_data.py --dataset all
```

To prepare one dataset, replace `all` with `wheat`, `maize`, or `arabidopsis` in both
commands. To verify existing downloads without fetching anything:

```bash
python scripts/download_data.py --dataset all --verify-only
```

The downloader obtains the exact upstream inputs used in this study. Individual
data files are checked against SHA-256 values in
[`configs/data_sources.json`](configs/data_sources.json). Supplement archives are
containers: their extracted workbook contents, rather than potentially variable
ZIP metadata, are verified. Existing mismatched files are not overwritten.

- **Wheat:** the aligned phenotype/environment CSV and kinship table come from
  the [Shao et al. companion repository](https://github.com/YingjieShao/PINN_for_plant_height_forecasting/tree/3da92f51f42d3fde5e06f6fc8ce8f233490a389c),
  pinned to commit `3da92f51f42d3fde5e06f6fc8ce8f233490a389c`.
  These are the processed benchmark inputs, not raw ETH sensor data.
  The model uses one-hot genotype encoding; the existing loader also reads the
  kinship table. See [Shao et al., DOI 10.1016/j.compag.2026.111988](https://doi.org/10.1016/j.compag.2026.111988)
  for the reference protocol. Loading performs the published alignment, cropping,
  split, and normalization; no additional preprocessing command is required.
- **Maize:** supplementary workbooks come from
  [Sweet et al., DOI 10.1111/tpj.17092](https://doi.org/10.1111/tpj.17092)
  via [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11629746/supplementaryFiles).
  Station temperature files are downloaded from the
  [author repository](https://github.com/HirschLabUMN/WiDiv_Drone_Height/tree/8fed4415f99f7ee4f8e43fb088319622bedc81a3),
  pinned to commit `8fed4415f99f7ee4f8e43fb088319622bedc81a3`.
  The related original deposit is [DRUM, DOI 10.13020/SKJN-QX31](https://doi.org/10.13020/SKJN-QX31).
  The submission downloader fetches tabular inputs needed by preprocessing, not
  UAV rasters. The reported response has relative UAV-height units; it must not
  be interpreted as centimetres or metres.
- **Arabidopsis stem length:** the supplementary workbook from
  [Ebrahimi Naghani et al., DOI 10.1186/s12870-024-05394-w](https://doi.org/10.1186/s12870-024-05394-w)
  is obtained through [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11285529/supplementaryFiles).
  `prepare_data.py` extracts the primary inflorescence stem-length sheets into a
  720-row table. This dataset is distinct from the hypocotyl dataset collected here.

External raw and processed inputs are excluded from this submission snapshot and
restored by the commands above. Their original terms and attribution requirements
apply; see [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Train PhytoODE or evaluate the submitted models

The same interface supports all four datasets:

```bash
python run.py train --dataset wheat --seed 1 --device cuda:0 \
  --output outputs/wheat-training
python run.py train --dataset maize --seed 1 --device cuda:0 \
  --output outputs/maize-training
python run.py train --dataset arabidopsis --seed 1 --device cuda:0 \
  --output outputs/arabidopsis-training
```

Use seeds `1`, `2`, and `3` in separate output directories for the manuscript's
three-seed evaluation. The default model is `phytoode`; `--model latent_ode` sets
both physics coefficients to zero. For hypocotyls this baseline uses genotype and
time without illumination, matching its manuscript input specification.

| Dataset | λ_ODE | λ_K | Epochs | Parameters |
|---|---:|---:|---:|---:|
| Wheat | 3.16227766 | 0.1 | 1,500 | 1,655 |
| Maize | 0.5 | 0.5 | 3,000 | 9,003 |
| Arabidopsis stem length | 0.5 | 0.1 | 1,500 | 4,607 |
| Hypocotyl length | 500 | 0.1 | 1,500 | 1,103 |

Training uses the paper's architecture, optimizer, learning-rate schedule,
gradient clipping, and dataset-specific validation checkpoint rule. It saves the
selected checkpoint and its selection record before evaluating test targets.
`history.csv`, `checkpoint.pt`, `selection.json`, `predictions.npz`, and
`result.json` are written to the chosen directory. For temperature datasets,
reported training metrics use replicate-averaged curves, while fitting uses
individual plot/plant trajectories.

Evaluate bundled checkpoints without retraining:

```bash
python run.py evaluate --dataset maize --seed 1 --device cpu \
  --output outputs/maize-evaluation
```

Evaluate a newly trained checkpoint:

```bash
python run.py evaluate --dataset wheat --seed 1 \
  --checkpoint outputs/wheat-training/checkpoint.pt \
  --output outputs/wheat-reloaded
```

Prediction arrays are stored on each model's integration grid in training units.
The `scale`/`height_scale_mm` definitions in the data loaders convert them to the
reported physical or relative-height unit. `result.json` metrics are already in
the reported units. These examples run on the supplied study datasets; adapting
to a new experiment requires a data loader and a new validation design.

## Baselines, tables, and figures

[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) describes baseline commands, the
manuscript-to-code mapping, and the distinction between re-evaluating saved
predictions and retraining models. Wheat LSTM-NN and Logi-PINN table entries are
the reference authors' reported outputs, not newly trained replacements.

```bash
# Export the manuscript's four comparison tables from recorded metrics.
python scripts/report_results.py --output outputs/paper-tables

# Regenerate the manuscript prediction figures after external data preparation.
python scripts/make_figures.py --output outputs/paper-figures

# Verify the checksums of the submitted code, data, configurations, and results.
python scripts/verify_submission.py
```

Expected PhytoODE test scores are mean ± sample standard deviation over seeds 1–3:

| Dataset | RMSE | Relative RMSE |
|---|---:|---:|
| Wheat | 0.03032 ± 0.00066 m | 10.24 ± 0.22% |
| Maize | 56.68 ± 4.37 relative UAV-height units | 18.40 ± 1.42% |
| Arabidopsis stem length | 2.800 ± 0.016 cm | 14.06 ± 0.08% |
| Hypocotyl length | 0.346 ± 0.008 mm | 6.85 ± 0.15% |

RMSE is averaged over trajectories. Relative RMSE is `100 × mean trajectory
RMSE / mean scored target`, calculated separately for each partition; it is not
MAPE. The denominator is shared by all models within a dataset and partition.
Hypocotyl targets are partition-specific genotype/time replicate means, not
individual lengths. The latter are reported as a separate pooled RMSE.

## Interpretation and reproducibility

- Wheat retains the reference initial-fill mask: 475 of 1,368 scored test
  positions precede the first actual observation. Other datasets use observed
  targets. Missing tensor entries with zero masks are placeholders, not imputed
  zero-length training targets.
- The environmental encoder uses the known environmental sequence over the
  prediction interval. These experiments assess scenario-conditioned prediction
  within observed time intervals, not prospective forecasting or unseen-genotype
  prediction.
- The coefficient studies and hypocotyl extension reused previously inspected
  test partitions. The maize `(0.5, 0.5)` setting was retained after the later
  search and was not that search's validation optimum. The temperature-dataset
  unregularized controls were not independently retuned.
- Hypocotyl PhytoODE received the illumination feature and additional tuning;
  its retained baselines did not. This is a comparison of complete predictors,
  not a matched isolation of the physics-loss effect. The fixed-hyperparameter
  illumination comparison is included separately because it is discussed in the
  manuscript.
- Standard deviations describe training-seed variation, not uncertainty across
  independent experiments. Retraining on another platform need not reproduce
  every reported digit; bundled checkpoints and predictions preserve the exact
  submitted runs.

## Data and citation

The submitted hypocotyl workbook, tidy observations, and fixed partitions are
available under [`hypocotyl/data`](hypocotyl/data). No DOI has been assigned to
this GitHub snapshot. Cite the accompanying manuscript using
[`CITATION.cff`](CITATION.cff), and cite each original external dataset when using
it. This snapshot does not assign an additional open-source or data-reuse licence;
upstream rights are retained and enquiries about reuse should be directed to the
corresponding author.
