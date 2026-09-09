# Current ordinary-logistic model rationale

The current comparison uses only 12L12D observations. It supplies genotype and
elapsed time to every predictor, with no environmental features. The physical
reference is dH/dt = r H (1 - H/K), with constant genotype-specific r and K.

Alimchandani, Branchini & Routier-Kierzkowska (2026), *New Phytologist* 249,
299–324, [doi:10.1111/nph.70576](https://doi.org/10.1111/nph.70576), supports
logistic descriptions of hypocotyl growth. We use the zero-offset form. This
reference does not establish that the plants reached mature length at 72 h or
that a constant rate resolves circadian growth regulation.

The process ODE fits r, K, and initial height to each genotype's training means.
PhytoODE predicts length through a neural latent flow and estimates its own r and
K through auxiliary penalties. These are distinct parameter fits. The capacity
penalty is finite-window maximum consistency, not an observation of maturity.

Primary targets are within-partition replicate means, following the user's return
to the initial evaluation. Missing cells are omitted before averaging and absent
time groups are not imputed. See [the current report](single_condition_20260909/results.md)
and [benchmark documentation](../README.md).

The earlier two-condition switched-model references are preserved in
[light_reference_notes.md](light_reference_notes.md). The initial mean-target
rationale is preserved in [the archive](archive/model_rationale_mean_targets.md).
