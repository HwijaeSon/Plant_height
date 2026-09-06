# Data files and provenance

All paths below are relative to `maize/data/`.

| File | Meaning |
|---|---|
| `download_manifest.json` | 53 source files: source URLs, bytes, SHA-256, pinned GitHub commit, supplement-name mapping |
| `raw/supplementary/` | Europe PMC archive, all 15 supplementary workbooks, supplementary figures PDF |
| `raw/drum/` | Deposited extracted heights, canopy cover, hourly weather, hand-height archives, README, date–GDD table |
| `raw/github/` | Author analysis code, station daily temperatures, plot/genotype mapping, flowering metadata |
| `raw/source/` | Article HTML and repository inventories; additional discovery snapshots are retained |
| `processed/height_all_slots.csv.gz` | All 48,880 S2/S3 plot-date slots, including 7,462 missing cleaned values and mapping/QC flags |
| `processed/height_observations.csv` | Primary cohort, 31,894 observed plot-date rows; no interpolated targets |
| `processed/height_genotype_means.csv` | Primary cohort, 17,374 observed genotype-year-date means, with available replicate counts |
| `processed/trajectory_inventory.csv` | All 4,160 source-row × year trajectories, selection flags and exclusion reasons |
| `processed/plot_metadata.csv` | 4,160 physical plot IDs and metadata from S4; duplicate-key and LOESS-presence flags |
| `processed/genotypes.csv` | Stable 402-label vocabulary and zero-based indices |
| `processed/flights.csv` | All 47 published flight dates, corrected calendar years, source notes, both GDD sources |
| `processed/weather_daily.csv` | 384 unique year-days, DAP 0–95 each year; station temperature and thermal features |
| `processed/weather_envirotype_daily.csv` | 360 S5 rows, DAP 0–89; auxiliary weather, kept separately |
| `processed/loess_reference_wide.csv.gz` | S4 original plot rows, 27 height values and 26 growth-rate intervals; derived reference targets |
| `processed/manual_height_reference.csv` | S14 measurements: 1,725 values, 40 plots, 2020–2021, cm and m |
| `processed/year_splits.json` | All 12 ordered validation/test-year choices, two remaining training years |
| `processed/model_ready/chronological.npz` | Training plot tensors and replicate-averaged evaluation tensors; loads with `allow_pickle=False` |
| `processed/model_ready/chronological.json` | Shapes, observed-target counts, train-only scalers and unit definitions |

`height_relative` preserves the published S3 numerical values. `height_raw` is
the corresponding S2 pre-normalization number, not a metric height. `source_row`
is the one-based Excel row in S2/S3. `replicate` identifies field replicates,
not individual maize plants. `plot_id` is joined only when genotype/replicate/year
maps uniquely to an S4 physical plot. Unresolved rows retain an explicit source
trajectory ID. `observed` marks nonmissing S3 values; ground flights are flagged
separately and excluded from the primary cohort.

Array keys use prefixes `train_`, `train_eval_`, `val_`, and `test_`:

| Suffix | Shape | Meaning |
|---|---|---|
| `y` | N × 96 | Relative height / training maximum; zero placeholder when mask is zero |
| `mask` | N × 96 | 1 only at actual observed target dates |
| `env` | N × 96 × 1 | Standardized station `(Tmax + Tmin)/2` in Celsius |
| `s` | N × 96 | Trapezoidal thermal integral / training-year mean final integral |
| `ds` | N × 96 | Derivative of `s` with respect to `tau = DAP/95` |
| `g_idx` | N | Genotype index |
| `year`, `plot` | N | Trajectory metadata |

Predictions and RMSE/MAE are converted to original **relative-height units** by
multiplying by `y_scale`. There is no justified conversion from this target to
centimetres or metres. No genomic kinship matrix is provided or fabricated.

Source links: [article and supplements](https://pmc.ncbi.nlm.nih.gov/articles/PMC11629746/),
[Europe PMC download](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11629746/supplementaryFiles),
[DRUM](https://doi.org/10.13020/SKJN-QX31),
[author repository](https://github.com/HirschLabUMN/WiDiv_Drone_Height/tree/8fed4415f99f7ee4f8e43fb088319622bedc81a3).
DRUM metadata states CC0 1.0; the article states CC BY 4.0. The author repository
has no top-level LICENSE in the pinned tree, so its snapshot is fetched locally
by `download_data.py` for provenance and is not redistributed through this
repository.
