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

The [hypocotyl extension](../hypocotyl/README.md) adds **943 author-collected
measurements** at fixed 23°C under **12L12D only**; continuous red light is excluded.
It includes five genotypes, seven times, and 6–38 replicate measurements per
genotype/time. The primary manuscript holds out replicate groups and scores
35 split-specific means per partition. PhytoODE receives binary illumination
in its encoder and latent vector field, while its physics remains ordinary
constant-rate logistic growth. The validation-selected model obtains
**0.3460 ± 0.0077 mm / 6.85 ± 0.15%** test RMSE / relative RMSE, below all five
retained baseline families. Its selected `(lambda_ODE, lambda_K)` is `(500, 0.1)`.
The original three dataset results remain unchanged.

The lower test score of the prespecified fixed-settings light-input variant
is disclosed separately; it does not replace the validation-selected main model.
Only PhytoODE receives the additional search and light channel. The preceding
no-light latent ODE therefore is not an input-matched loss ablation of this model.
The original test partition had already been inspected. The manuscript explains
these procedures and describes public dataset release **as planned**, without
inventing a DOI, license, or completed public deposit. The auxiliary 36/60-h
holdout remains in the experiment report, outside the primary manuscript.

## Files

- `main.tex`: complete English draft with Introduction, Materials and methods,
  wheat, maize, Arabidopsis stem-length and hypocotyl Results, and Conclusion. The `\model` macro
  expands to `PhytoODE`; prose treats it as a proper model name.
- `references.bib`: the original references plus the light-growth literature. The Introduction cites 17 papers from
  *Computers and Electronics in Agriculture* and 13 from *Artificial
  Intelligence in Agriculture*, plus foundational Neural ODE, latent ODE,
  neural CDE, PINN, and universal differential-equation papers.
- `make_figures.py`: regenerates the three temperature-experiment figures.
  `write_hypocotyl_results.py` regenerates the current hypocotyl text, table,
  and figure from audited light-input results.
- `figures/`: generated wheat, maize, and Arabidopsis prediction figures. The
  three figures share the same six-model legend, colors, line hierarchy, and
  observed-point marker. Every model line is clipped to the sample's
  first--last observed time interval.
- `hypocotyl_methods.tex` and `hypocotyl_results.tex`: the added experiment,
  included by `main.tex`. Its figure shows all five genotypes with the selected
  PhytoODE in blue, the five baseline families, test means, individual lengths,
  and shaded dark intervals. The baseline is ordinary Logistic ODE.
- [revisions_20260910.md](revisions_20260910.md): current copyable abstract,
  Methods, Results, table/caption, conclusion and data-availability passages.
  Earlier dated revision files reproduce the earlier experiments.
- `main_standalone.tex`: the complete current manuscript with the two
  hypocotyl `\input` files expanded for pasting into external editors.
- `overleaf_20260910.zip`: current uploadable manuscript source, bibliography, and
  the four referenced figure PDFs.
- [relative_errors/README.md](relative_errors/README.md): **historical, before
  the follow-up coefficient study** wheat,
  Arabidopsis, and maize errors divided by each split's mean evaluation target,
  with CSV tables, a PNG/PDF figure, a ready-to-input LaTeX table reporting
  `RMSE / relative error (%)`, and the original scoring-mask limitations.
  The main maize `Ours` row uses the validation-selected tuned result; its
  pre-tuning result is retained as a non-primary row.
  Regenerate with `.venv/bin/python paper/relative_errors.py`.
  That script reproduces the historical table. The current comparison for the
  three temperature-conditioned datasets is [current_results.csv](current_results.csv); current PhytoODE/control
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
warnings recorded in `compile_report_20260910.json`. The reproducible source
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
   Its targets are split-specific replicate means at all seven observation
   times, within the single 12L12D regime. Binary illumination is an explicit
   feature of the known schedule, not evidence of transfer to a new photoperiod.
   The no-light baseline and fixed-settings feature comparison must not be
   presented as a matched loss ablation of the selected light-input model.
   The public dataset URL/DOI, reuse
   license and missing experimental metadata remain to be supplied.

## Regenerate figures

```bash
.venv/bin/python paper/make_figures.py
# The completed light-input follow-up is the current hypocotyl source:
.venv/bin/python paper/write_hypocotyl_results.py
```
