# Arabidopsis stem-length experiment

## Data and split

The phenotype is primary inflorescence stem length from the supplementary
workbook of Ebrahimi Naghani et al. (2024). It contains nine genotypes, two
temperature regimes, ten plants per genotype-condition combination, and four
measurements per plant (720 observations total).

- Training plants: identifiers 1--6
- Validation plants: identifiers 7--8
- Test plants: identifiers 9--10

All genotype-temperature groups occur in every split, but individual plants
do not cross splits. Metrics are computed on replicate-averaged curves. This
tests transfer to new biological replicates at very low temporal resolution;
it does not test unseen genotypes or temperatures.

Source paper: Ebrahimi Naghani et al., *BMC Plant Biology* 24, 721 (2024),
DOI: 10.1186/s12870-024-05394-w.

## Reproduce

The workbook is already included. To download it again and rebuild the
processed table:

```bash
bash arabidopsis/code/download_data.sh
python arabidopsis/code/prepare_data.py
```

Run all baselines and the latent Neural ODE:

```bash
python arabidopsis/code/run_experiment.py \
  --device cuda --seeds 1 2 3 \
  --epochs 1500 --reference-epochs 3000 \
  --physics-warmup-epochs 500 --ode-lr 1e-4 \
  --output arabidopsis/results/reproduced

python arabidopsis/code/plot_results.py \
  --results arabidopsis/results/reproduced
```

Genotype-specific logistic parameters are fitted using training plants only
and used to initialize Logi-PINN. Its model learning rate is `1e-3`, ODE
parameter learning rate is `1e-4`, and the ODE parameters are frozen during a
500-epoch data-only warm-up.

## Results

- `results/paper_results_3seed.csv`: manuscript table rows.
- `results/all_runs_seed1_3.csv`: per-model, per-seed metrics.
- `results/comparison_seed1_3.csv`: aggregated metrics in centimetres.
- `results/predictions_seed1_3.csv`: predictions used by the paper plot.
- `results/fitted_process_parameters.json`: training-only logistic fits.
- `results/figures/`: final no-extrapolation prediction figure.
