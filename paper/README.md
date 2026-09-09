# PhytoODE plant-growth manuscript draft

This folder contains an Elsevier-style LaTeX draft aimed primarily at
*Computers and Electronics in Agriculture*.  The same `elsarticle` source can
be retargeted to *Artificial Intelligence in Agriculture* by changing the
`\journal{}` line and adapting the cover letter.

The current manuscript uses the [adopted PhytoODE configuration](../experiments/phytoode_config.json):
wheat `(3.16227766, 0.1)`, maize `(0.5, 0.5)`, and Arabidopsis `(0.5, 0.1)`
in `(lambda_ODE, lambda_K)` order. Its tables also include the matched
both-zero latent ODE control. [Exact replacement snippets](revisions_20260908.md)
are provided for copying into an external manuscript. The maize retention
decision, the historical validation winner `(500, 0.5)`, and prior inspection
of the test sets are stated explicitly in Methods.

The [hypocotyl extension](../hypocotyl/README.md) adds original replicate
snapshot data at fixed 23°C under 12L12D and continuous red light. Its seven
methods are selected using validation only before any new test scoring.
The original three dataset results remain unchanged. The title and common
notation now cover environmental conditioning by either temperature or light.

## Files

- `main.tex`: complete English draft with Introduction, Materials and methods,
  wheat, maize, Arabidopsis stem-length and hypocotyl Results, and Conclusion. The `\model` macro
  expands to `PhytoODE`; prose treats it as a proper model name.
- `references.bib`: the original references plus the light-growth literature. The Introduction cites 17 papers from
  *Computers and Electronics in Agriculture* and 13 from *Artificial
  Intelligence in Agriculture*, plus foundational Neural ODE, latent ODE,
  neural CDE, PINN, and universal differential-equation papers.
- `make_figures.py`: regenerates the manuscript figures directly from the
  audited experiment outputs.
- `figures/`: generated wheat, maize, and Arabidopsis prediction figures. The
  three figures share the same six-model legend, colors, line hierarchy, and
  observed-point marker. Every model line is clipped to the sample's
  first--last observed time interval.
- `hypocotyl_methods.tex` and `hypocotyl_results.tex`: the added experiment,
  included by `main.tex`. Its figure retains the common method colors, replaces
  the temperature-specific process baseline with Light-logistic ODE, and
  includes the both-zero latent ODE using a dashed line.
- [revisions_20260909.md](revisions_20260909.md): copyable title, abstract,
  Methods, Results, tables and conclusion additions for the light experiment.
- `main_standalone.tex`: the complete current manuscript with the two
  hypocotyl `\input` files expanded for pasting into external editors.
- `overleaf_20260909.zip`: uploadable manuscript source, bibliography, and
  the four referenced figure PDFs.
- [relative_errors/README.md](relative_errors/README.md): **historical, before
  the follow-up coefficient study** wheat,
  Arabidopsis, and maize errors divided by each split's mean evaluation target,
  with CSV tables, a PNG/PDF figure, a ready-to-input LaTeX table reporting
  `RMSE / relative error (%)`, and the original scoring-mask limitations.
  The main maize `Ours` row uses the validation-selected tuned result; its
  pre-tuning result is retained as a non-primary row.
  Regenerate with `.venv/bin/python paper/relative_errors.py`.
  That script reproduces the historical table. The current full comparison
  is [current_results.csv](current_results.csv); current PhytoODE/control
  scores are sourced from the [adoption report](../experiments/results/adopted_phytoode_20260908/README.md).

## Compile

On Overleaf or a TeX installation containing Elsevier's `elsarticle` class:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

A temporary Tectonic 0.17.0 toolchain was downloaded for local PDF verification;
the default shell still has no LaTeX engine on its path. The completed PDF is
`main.pdf` (a local build artifact), with source hashes and remaining compiler
warnings recorded in `compile_report_20260909.json`. The reproducible source
files and figures are tracked in Git.

## Audit status before submission

1. The wheat result is protocol-matched: train 2018/2019, validation 2022,
   and test 2021. Capacity and coefficients were ranked using validation,
   but the follow-up coefficient study reused previously inspected test sets.
   Before submission, confirm the fixed 1,655-parameter
   model over the other five year combinations without test-year retuning.
2. The maize result uses 402 genotypes across four years and retains the
   original `(0.5, 0.5)` configuration following the later coefficient study.
   This differs from that study's validation optimum. It demonstrates held-out-year performance across known
   genotypes, not unseen-genotype prediction; keep that distinction explicit.
3. Author, affiliation, and funding text follows the supplied manuscript.
   Complete the remaining data/code availability placeholder before submission.
4. Tables consistently report seeds 1--3 as mean ± sample standard deviation.
   Deterministic process ODEs are explicitly marked as single fits.
5. The Arabidopsis result is a new-plant split within observed genotypes and
   temperatures.  Keep the current wording unless genotype-held-out or
   temperature-held-out experiments are added.
6. The environmental encoder reads the complete temperature scenario.  The
   manuscript correctly calls this offline scenario-conditioned prediction,
   not causal real-time forecasting.
7. Time extrapolation is not evaluated or displayed.  Prediction figures show
   only the interval supported by actual observations.
8. Hypocotyl rows are bookkeeping groups, not verified longitudinal plant IDs.
   Its primary targets are split-specific replicate means. The entirely
   withheld times are 36 and 60 h; both lighting regimes are known in training.
   The independently LR-tuned no-physics baseline and the same-LR paired
   control are different comparisons. The public dataset URL/DOI, reuse
   license and missing experimental metadata remain to be supplied.

## Regenerate figures

```bash
.venv/bin/python paper/make_figures.py
# After the separate hypocotyl training controller completes:
.venv/bin/python hypocotyl/code/summarize.py
.venv/bin/python paper/write_hypocotyl_results.py
```
