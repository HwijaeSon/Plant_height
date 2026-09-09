# References for the light-dependent logistic constraint

The current reference uses non-MDPI publications. The exact switched equation is
our phenomenological construction, not an equation reproduced from these papers.

| Reference | Specific support | What it does not establish |
|---|---|---|
| Alimchandani, Branchini & Routier-Kierzkowska (2026), *New Phytologist* 249, 299–324. [doi:10.1111/nph.70576](https://doi.org/10.1111/nph.70576) | Box 1 defines logistic organ growth; the hypocotyl analysis and Fig. 3 apply it to light-grown and dark-grown plants. | A fitted two-rate switching law within each daily cycle. |
| Nozue et al. (2007), *Nature* 448, 358–361. [doi:10.1038/nature05946](https://doi.org/10.1038/nature05946) | Experimental evidence that light and the circadian clock jointly regulate hypocotyl elongation through PIF4/PIF5. | Logistic saturation or constant growth rates throughout each light/dark phase. |
| Seaton et al. (2015), *Molecular Systems Biology* 11, 776. [doi:10.15252/msb.20145766](https://doi.org/10.15252/msb.20145766) | A quantitative clock/PIF model links light-dependent regulation to hypocotyl growth. | A direct derivation of the two-parameter rate switch used here. |

The New Phytologist article was first published online on 9 November 2025 and
assigned to the 2026 volume. Box 1 uses the shifted logistic expression

\[
S(t)=\frac{S_{\max}}{1+\exp[-k(t-t_{\mathrm{mid}})]}+d.
\]

We fix the additional offset to zero and extend the constant rate to two
genotype-specific effective rates:

\[
\frac{dH_{g,c}}{dt}
=\{r_{g,L}L_c(t)+r_{g,D}[1-L_c(t)]\}
 H_{g,c}\left(1-\frac{H_{g,c}}{K_g}\right).
\]

This is a modeling choice supported by the two complementary kinds of evidence.
It is not a reduction obtained by eliminating molecular states from the published
clock models. Each rate is constant within its light phase; neither a circadian
oscillator nor transient light-switch responses are represented. No constraint
forces the dark rate to exceed the light rate, particularly for photoreceptor
mutants. Shared rates across the two experimental light conditions remain an
approximation because the 12L12D spectrum is unknown and cR is continuous red.

The process baseline fits these rates, capacity, and initial heights directly to
individual training measurements. PhytoODE learns its own positive auxiliary
rates and capacity through regularization. These are distinct parameter fits.

Verification: publisher full text and the archived full-text MathML for Box 1
were inspected. Primary links are supplied above; no journal metric or impact
factor is used to justify the equation.
