# Hypocotyl dataset fields

The original workbook is preserved byte for byte. Its two sheets are `12L12D`
and `cR`. Rows 3–44, columns B–AJ contain measured lengths; blank cells are
missing observations, not zero-length plants. Workbook mean/SD/test formulas
and chart-input sections are excluded from the tidy observations.

| Field | Meaning |
|---|---|
| `observation_id` | Unique source-sheet and Excel-cell identifier; not a plant ID |
| `sheet` | `12L12D` or continuous red light (`cR`) |
| `genotype` | Source label: Col-0, hy5, MLB, EMS57, or phyAB |
| `elapsed_hours` | 0, 12, 24, 36, 48, 60, or 72 hours from the first observation |
| `length` | Measured hypocotyl length, mm |
| `source_cell` | Original Excel cell coordinate |
| `source_row` | Original row; retained for bookkeeping without inferring plant identity |
| `source_column` | One-based original Excel column number |
| `split` | Frozen train/val/test membership, present only in partition files |

`group_statistics.csv` additionally reports measurement count `n`, arithmetic
`mean`, sample standard deviation `sd` (ddof=1), and `spreadsheet_mean`, the
cached Excel AVERAGE used to verify extraction. These summaries use all
measurements for descriptive and extraction-verification purposes. The current
training pipeline does not use these full-dataset summaries as targets.
`prepare_single_condition.py` filters the frozen partitions to 12L12D and stores
new split-specific `train_means.csv`, `val_means.csv`, and `test_means.csv` files
under `processed/single_condition_20260909/<protocol>/`. Their fields are
`genotype`, `elapsed_hours`, `mean_length_mm`, and `n_observations`.
`single_condition_data.py` uses these means as targets; missing time groups have
no target entry. Only genotype and elapsed time enter model inputs. The earlier
`observation_data.py` retains raw individual targets for the previous two-condition
experiment, and `data.py` reproduces the original two-condition mean-target runs.

Missing genotype alleles, irradiance, the 12L12D spectrum, growth medium,
independent batch identifiers and age at the first measurement are not supplied
by the workbook. The confirmed construct for MLB is
`ML1promoter::phyB-GFP/phyB-9`. Further experimental details can be appended in
a versioned metadata update without modifying the original workbook.

The deposit bundle includes the raw workbook, tidy data, frozen partitions,
provenance hashes, model code and its shared latent-ODE source. It does not
include the full fitted search outputs; the current runs remain under
`hypocotyl/results/single_condition_20260909` in the research repository.
To rerun with the included frozen partitions, install the dependencies and
run `verify_single_condition.py`, `run_single_condition.py` and
`summarize_single_condition.py` from the package root
(scripts are in `hypocotyl/code/`). Manuscript-updating commands in the main
README require the full repository's `paper/` directory, which is excluded
from this data package. Do not rerun `make_splits.py` over the frozen partition
directory. The recorded CUDA PyTorch build may require the corresponding
PyTorch package index; the root `requirements.txt` supplies portable ranges.

Public repository visibility, a permanent dataset identifier and an explicit
data-reuse license have not been established. Preparing this bundle is not
itself a public deposit.
