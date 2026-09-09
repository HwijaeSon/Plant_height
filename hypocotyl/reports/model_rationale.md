# Light-dependent hypocotyl growth reference

Two directly relevant primary research sources were located:

- Pay et al. (2022), *Modelling of plant circadian clock for characterizing
  hypocotyl growth under different light quality conditions*, **in silico Plants**,
  4(1), diac001. [DOI](https://doi.org/10.1093/insilicoplants/diac001).
  This study connects light quality, photoreceptor/clock regulation and hypocotyl
  elongation. It motivates treating illumination as an environmental driver,
  rather than trying to infer a temperature response from a constant temperature.
- You et al. (2026), *Mathematical Modeling of Circadian Clock-Regulated Hypocotyl
  Elongation Accelerated by Artificial Photoperiods in Arabidopsis*,
  [consulted preprint version 1, Eq. (13)](https://www.preprints.org/manuscript/202606.1001).
  Its growth equation multiplies a PIF-dependent rate by `H (1 - H/K)`.
  The preprint page also links a subsequently published
  [Mathematics article](https://doi.org/10.3390/math14142556).
  The equation inspected here is explicitly the preprint equation; the
  publisher's full text could not be retrieved for version comparison.

## Model used in this benchmark

There are seven observation times, two known lighting regimes and no molecular
time series. We therefore fit a reduced, light-switched logistic reference:

\[
\frac{dH_g(t)}{dt}
=\{r_{g,L}L(t)+r_{g,D}[1-L(t)]\}
H_g(t)\left(1-\frac{H_g(t)}{K_g}\right).
\]

Here `L=1` denotes light and `L=0` darkness. Time is measured in hours, rates
have units h^-1, and reported length is in mm. For 12L12D, the light intervals
are [0,12), [24,36), and [48,60); cR is continuously illuminated.

This equation is **our coarse adaptation**, not a reproduction of the complete
circadian/PIF model. It replaces an unobserved molecular growth-rate driver
with two fitted phase rates. Neither circadian oscillator parameters nor PIF
concentrations are inferred. We do not impose `r_dark > r_light`: genotype,
developmental phase and light quality can affect these effective fitted rates.

Let `A(t)` be accumulated light exposure. For a fitted initial height `H0`,
the reference has an exact solution:

\[
R_g(t)=r_{g,L}A(t)+r_{g,D}[t-A(t)],\qquad
H_g(t)=\frac{K_g}{1+(K_g/H_{g,0}-1)\exp[-R_g(t)]}.
\]

The light-logistic baseline shares the two rates and capacity within genotype
and estimates an initial height separately for each genotype/condition pair:
25 fitted parameters. The ordinary logistic baseline fits rate, capacity and
initial-height fraction independently to each of the ten genotype/condition
curves: 30 fitted parameters. Both are fitted using training means only and
eight deterministic starting points; evaluated-trajectory heights are never
prediction inputs.

## PhytoODE adaptation

- The latent ODE receives genotype, normalized elapsed time, binary light and
  the known light duty cycle (0.5 for 12L12D, 1 for cR). Temperature is fixed at
  23 degrees C and is not fitted as an explanatory variable.
- The existing latent-ODE family is retained: latent dimension 8, genotype
  embedding 4, one vector-field hidden layer of width 16, decoder width 16,
  and initial-state LSTM width 8. These sizes were fixed before comparison.
- The logistic head maps genotype embeddings to two positive hourly rates
  and capacity. In training-scaled height units, rates lie in (0,0.5) h^-1 and
  capacity in (0,2). Scaling uses only the largest training measurement.
- The decoded directional derivative is evaluated by automatic
  differentiation as `grad_z decoder · latent_field / 72`. Its discrepancy
  with the light-switched logistic right-hand side forms the ODE penalty.
- A 3-h RK4 grid contains 25 nodes. Binary light is held constant within each
  interval, including that interval's right-end solver stage. The next interval
  uses the new light state. No smoothing ramp is inserted around switches.
- Derivative residuals use the 18 interior nodes away from 12-h switching
  boundaries. These collocation points do not require phenotype measurements.
  The maximum-height consistency penalty uses the whole predicted grid. It is
  a finite-window regularizer, not a measurement of final mature length.
- Disabling physics means setting **both** coefficients to zero. A separately
  learning-rate-tuned no-physics model and a strictly matched control at the
  selected PhytoODE learning rate are retained.

The comparison additionally includes an LSTM, random forest, and a coordinate
MLP Light-PINN. The latter uses genotype, elapsed time and light duty cycle as
prediction inputs, with instantaneous light in its residual; its local time
derivative is obtained by automatic differentiation. It is not the old
LSTM-based Logi-PINN implementation from the wheat benchmark.

## What the experiment can establish

The target is a genotype/condition population-mean growth curve reconstructed
from replicate snapshots. Source rows have no verified longitudinal identity;
they are not concatenated into plant trajectories. A conservative partition
keeps each genotype/source-row group together across times and conditions.

The primary comparison holds out measurement groups within observed conditions.
The time-holdout comparison removes **all** 36-h and 60-h observations from
training and validation, testing interpolation between other measured times.
This provides an explicit sparse-time test, beyond simply having a 12-h
observation interval. All models start from scratch for each protocol.

The two lighting regimes are present in training. This is not prediction of
an unseen photoperiod, and spectral effects cannot be separated from regime
effects: cR is red light, while the spectrum of the 12L12D treatment is not
documented. Twelve-hour measurements cannot resolve intracycle circadian peaks.
Age at the first measurement, independent growth batches and irradiance are
not established by the supplied workbook. These limits must remain explicit
in the manuscript and release metadata.
