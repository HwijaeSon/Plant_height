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

## Files

- `main.tex`: complete English draft with Introduction, Materials and methods,
  wheat, maize, and Arabidopsis Results, and Conclusion. The `\model` macro
  expands to `PhytoODE`; prose treats it as a proper model name.
- `references.bib`: 40 cited records.  The Introduction cites 17 papers from
  *Computers and Electronics in Agriculture* and 13 from *Artificial
  Intelligence in Agriculture*, plus foundational Neural ODE, latent ODE,
  neural CDE, PINN, and universal differential-equation papers.
- `make_figures.py`: regenerates the manuscript figures directly from the
  audited experiment outputs.
- `figures/`: generated wheat, maize, and Arabidopsis prediction figures. The
  three figures share the same six-model legend, colors, line hierarchy, and
  observed-point marker. Every model line is clipped to the sample's
  first--last observed time interval.
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

The current machine does not contain a LaTeX engine or `elsarticle.cls`, so the
source was checked for balanced braces and missing citation keys but no local
PDF was produced.

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

## Regenerate figures

```bash
.venv/bin/python paper/make_figures.py
```
