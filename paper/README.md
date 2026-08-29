# Latent Neural ODE plant-growth manuscript draft

This folder contains an Elsevier-style LaTeX draft aimed primarily at
*Computers and Electronics in Agriculture*.  The same `elsarticle` source can
be retargeted to *Artificial Intelligence in Agriculture* by changing the
`\journal{}` line and adapting the cover letter.

## Files

- `main.tex`: complete English draft with Introduction, Materials and methods,
  wheat and Arabidopsis Results, and Conclusion.
- `references.bib`: 39 cited records.  The Introduction cites 17 papers from
  *Computers and Electronics in Agriculture* and 13 from *Artificial
  Intelligence in Agriculture*, plus foundational Neural ODE, latent ODE,
  neural CDE, PINN, and universal differential-equation papers.
- `make_figures.py`: regenerates the manuscript figures directly from the
  audited experiment outputs.
- `figures/`: generated wheat and Arabidopsis prediction figures.  Every model
  line is clipped to the sample's first--last observed time interval.

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

1. The wheat result is now protocol-matched: train 2018/2019, validation 2022,
   and test 2021.  Capacity and learning/physics hyperparameters were selected
   with validation only.  Before submission, confirm the fixed 1,655-parameter
   model over the other five year combinations without test-year retuning.
2. Replace red author, affiliation, data/code availability, and funding TODOs.
3. Tables consistently report seeds 1--3 as mean ± sample standard deviation.
   Deterministic process ODEs are explicitly marked as single fits.
4. The Arabidopsis result is a new-plant split within observed genotypes and
   temperatures.  Keep the current wording unless genotype-held-out or
   temperature-held-out experiments are added.
5. The environmental encoder reads the complete temperature scenario.  The
   manuscript correctly calls this offline scenario-conditioned prediction,
   not causal real-time forecasting.
6. Time extrapolation is not evaluated or displayed.  Prediction figures show
   only the interval supported by actual observations.

## Regenerate figures

```bash
.venv/bin/python paper/make_figures.py
```
