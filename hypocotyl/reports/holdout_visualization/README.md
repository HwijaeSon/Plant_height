# Frozen holdout partition visualizations

The two figures show the actual frozen partitions without refitting a model or
changing the manuscript's primary evaluation. Reproduce them from the repository
root with `.venv/bin/python hypocotyl/code/visualize_holdout_splits.py`.

The upper grid aggregates all five genotypes and both light conditions. Its rows
refer to the original replicate-protocol assignments, and its colors and symbols
show each observation's role in the displayed protocol. Cell counts are actual
measurements, not numbers of independently identified plants.

The lower panels show Col-0 observations in each light condition. Small marks are
raw measurements; large marks are the corresponding split-specific time-point
means. Horizontal jitter only separates overlapping observations. No individual
plant trajectories are inferred or connected. Gray background bands indicate the
dark intervals of the 12 h light / 12 h dark condition.

- **Replicate holdout:** all seven times occur in train, validation, and test;
  measurement counts are 1,052 / 330 / 436. The same genotype/source-row bookkeeping
  group stays in one partition across times and conditions. Source rows have not
  been verified as longitudinal plant identities.
- **36 / 60 h holdout:** all observations at these two times are test-only. Train
  and validation contain only 0, 12, 24, 48, and 72 h, with counts 758 / 240;
  test contains 505 measurements. The original replicate-test observations at the
  five retained times are unused (315 measurements, shown with gray crosses).

The latter is an auxiliary time-interpolation protocol retained for explanation;
it remains excluded from the current manuscript's primary evaluation. These are
data-partition figures, not prediction or model-performance comparisons.

`data_audit.json` records the source partition hashes and plotted counts.
