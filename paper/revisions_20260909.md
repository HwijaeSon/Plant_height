# Hypocotyl additions and changed manuscript passages (2026-09-09)

The main tables contain actual held-out results; no all-dataset SOTA claim is inferred.

## Preamble addition for figure placement

```latex
\usepackage{placeins}
```

## Replacement title

```latex
\title{PhytoODE: Learning plant growth dynamics across genotypes, temperature, and light regimes}
```

## Replacement abstract

```latex
\begin{abstract}
Predicting plant growth across genotypes and environments requires flexible dynamics under sparse observation. We present \model{}, a genotype- and environment-conditioned latent neural ordinary differential equation model. An encoder reads a known environmental scenario, a learned latent flow evolves continuously, and a decoder predicts phenotype without target-curve measurements as inputs. Logistic derivative and capacity penalties weakly regularize the decoded trajectory. Three temperature-conditioned benchmarks cover wheat with 19 genotypes, UAV maize with 402 genotypes, and Arabidopsis stem length with four observations per plant. Mean test RMSEs were $0.03032$~m, $56.68$ relative UAV-height units, and $2.800$~cm, respectively, corresponding to relative RMSEs of $10.24\%$, $18.40\%$, and $14.06\%$. These were lower than the compared benchmark families and matched no-physics controls, although the follow-up coefficient study reused previously inspected test sets. A fourth, author-collected dataset comprises 1,818 hypocotyl measurements across five genotypes and two light regimes at 23~$^{\circ}$C. We extend the growth constraint to separate light/dark rates and fit population-mean curves from replicate snapshots. With every 36-h and 60-h measurement withheld from training and validation, \model{} achieved $0.305$~mm RMSE ($4.13\%$ relative RMSE). All choices for this new comparison were frozen before test scoring. The experiments address field-season transfer, genotype-panel scale, sparse temporal sampling, and environmental-driver substitution. Evaluated genotypes and treatment regimes are represented during training; the new experiment does not establish unseen-photoperiod prediction or individual-plant trajectory recovery.
\end{abstract}
```

## New Methods subsections

```latex
\subsection{Author-collected hypocotyl measurements under light regimes}
\label{subsec:hypocotyldata}

We additionally analysed an author-provided experiment measuring \textit{Arabidopsis thaliana} hypocotyl length at a constant 23~$^{\circ}$C under 12-h light/12-h dark (12L12D) and continuous red light (cR). Five genotypes were included: Col-0, \textit{hy5}, MLB (\textit{ML1} promoter::phyB-GFP/\textit{phyB-9}), EMS57, and the \textit{phyA phyB} double mutant. Measurements were recorded in millimetres at seven elapsed times, 0--72~h in 12-h increments; the first 0--12-h interval was illuminated in 12L12D. The workbook contains 1,818 measured values, comprising 943 in 12L12D and 875 in cR, with 3--38 replicates in each genotype--condition--time group. Summary formulas were excluded, and all 70 recomputed group means matched the workbook summaries. The developmental age at the first observation was not confirmed; accordingly, time denotes elapsed hours rather than days after germination. Irradiance, the 12L12D spectrum, independent growth batches, and some mutant alleles remain undocumented.

The workbook does not establish longitudinal plant identifiers. We therefore model ten genotype--condition population-mean curves assembled from replicate snapshots, without interpreting each spreadsheet row as a repeatedly measured plant. For a conservative replicate-group holdout, genotype/source-row bookkeeping groups were assigned jointly across all times and both conditions to training, validation, and test partitions, using seed 20260909. A group was not allowed to cross partitions; partition generation checked coverage using identifiers only. This produced 1,052 training, 330 validation, and 436 test measurements. Split-specific means were computed independently, and all models were trained on the same 70 training means. These means contain 1--22 training, 1--7 validation, and 1--9 test measurements per group; the smallest groups therefore offer limited replication. This protocol evaluates held-out measurements within known genotype--condition combinations, without establishing a split between independent experimental batches.

A second protocol tests interpolation at entirely unobserved times. Every 36-h and 60-h measurement was removed from training and validation. Training and validation used the remaining five times, with 758 and 240 measurements, respectively; all 505 measurements at the two withheld times formed the test set. The other 315 measurements from the primary test partition were unused in this protocol. Models were trained from scratch on its 50 training means and scored on 20 withheld-time means. Both analyses use the same biological dataset and are not independent replications. The primary metric remains the mean RMSE across the ten population curves, with relative RMSE defined by Eq.~\eqref{eq:rrmse}. Pooled individual-measurement RMSE is reported separately to expose the difference between predicting a population mean and an individual plant.

\subsubsection{Light-dependent growth reference and latent dynamics}
\label{subsec:lightmodel}

Light-quality-dependent hypocotyl models link elongation to photoreceptor and circadian regulation \citep{pay2022hypocotyl}. Equation~(13) of the consulted preprint version of \citet{you2026hypocotylpreprint} combines a PIF-dependent effective rate with logistic saturation. Because the present data contain no molecular measurements and only seven observation times, we use a reduced light-switched logistic reference,
\begin{equation}
 \frac{dH_g}{dt}=\rho_g(t)H_g\left(1-\frac{H_g}{K_g}\right),
 \qquad \rho_g(t)=r_{g,L}L(t)+r_{g,D}[1-L(t)],
 \label{eq:light_logistic}
\end{equation}
where $L(t)$ is one in light and zero in darkness. The positive rates $r_{g,L}$ and $r_{g,D}$ are expressed in h$^{-1}$, without imposing an ordering between them. This is our phenomenological reduction, not a reproduction of the full circadian/PIF system. For accumulated illumination $A(t)=\int_0^t L(s)\,ds$, the process baseline uses the exact solution
\begin{equation}
 H_{g,c}(t)=\frac{K_g}{1+(K_g/H_{g,c}(0)-1)
 \exp\{-r_{g,L}A(t)-r_{g,D}[t-A(t)]\}}.
 \label{eq:light_solution}
\end{equation}
The two rates and capacity are shared across conditions within genotype, while initial height is fitted separately for each genotype--condition pair, giving 25 parameters. The ordinary logistic baseline instead fits three parameters independently to each of the ten curves (30 parameters). Both use training means only, with eight deterministic starting points; evaluated-curve initial heights are never supplied as inputs.

For \model{}, the environmental channels are binary illumination and the known light duty cycle (0.5 for 12L12D; 1 for cR). The initial-state encoder reads this complete scenario. Temperature is constant and is not fitted as a driver. The architecture was fixed at latent dimension 8, genotype embedding 4, vector-field width 16 with one hidden layer, decoder width 16, and encoder width 8, for 1,160 parameters including the auxiliary head. Phenotypes are divided by the maximum individual training measurement, 16.018~mm for each protocol. A genotype-dependent head produces $r_{g,L},r_{g,D}\in(0,0.5)$~h$^{-1}$ and $K_g\in(0,2)$ in scaled-height units.

The latent flow generates the prediction; Eq.~\eqref{eq:light_logistic} supplies only the observable-space derivative residual. The chain rule in Eq.~\eqref{eq:decoder_chain_day} is unchanged, with $S=72$~h: $d\hat y/dt=(\nabla_{\mathbf z}G_\psi)^{\top}\mathbf F_\theta/72$. Automatic differentiation retains the graph through this derivative and through the Runge--Kutta solver. No finite differences of measured lengths are used. A 3-h grid gives 25 integration nodes. Illumination remains constant within each integration interval, including its right-end solver stage, and changes at the next interval; interpolation does not create an artificial ramp at a light switch. The derivative penalty is evaluated at the 18 interior non-switch nodes per curve, excluding the 12-h boundaries. The capacity penalty uses the maximum over the complete predicted grid. It is a finite-window consistency constraint, not a measurement of mature hypocotyl length.

\subsubsection{Baselines and validation-only selection}

The comparison includes \model{}, Latent ODE with both physics coefficients zero, a coordinate-MLP Light-PINN, LSTM-NN, random forest, light-logistic ODE, and ordinary logistic ODE. Light-PINN uses genotype, elapsed time, and duty cycle to predict length, and instantaneous illumination in the same derivative residual. It has a four-dimensional genotype embedding and two hidden layers of width 32. LSTM-NN uses the same embedding dimension, two recurrent layers of widths 16 and 8, and a dense hidden layer of width 16. The Light-PINN time derivative is obtained directly by automatic differentiation of its coordinate network; this is a distinct baseline from the earlier LSTM-based Logi-PINN. Random forest receives genotype, time, illumination, duty cycle, and accumulated light exposure, all determined by the known scenario. All learned models share the same training targets and available genotype/light information.

All neural runs used 1,500 epochs, Adam with weight decay $10^{-4}$, gradient clipping at 1, and cosine learning-rate decay. Checkpoints were ranked after epoch 200 using the mean of the latest three validation evaluations, spaced 20 epochs apart. For both \model{} and Light-PINN, seed-1 screening crossed learning rates $\{0.003,0.01\}$, $\lambda_{\mathrm{ODE}}\in\{0.5,5,50,500\}$, and $\lambda_K\in\{0.01,0.1\}$. The no-physics latent ODE and LSTM screened the same two learning rates. Random forest used 300 trees and minimum leaf sizes $\{1,2,4\}$. The best two candidates per stochastic model were confirmed with seeds 2 and 3; selection then used three-seed mean validation RMSE. This gives 82 screening and 40 confirmation runs across the two protocols. Search sizes differ by model; this is a recorded finite-grid comparison, not an equal-compute claim. Every model choice for both protocols was frozen before any test scoring.

The principal no-physics baseline is independently learning-rate tuned. An additional paired control retains the selected \model{} learning rate, architecture, initial weights, optimization schedule, and checkpoint rule while setting $\lambda_{\mathrm{ODE}}=\lambda_K=0$. Its auxiliary parameter head remains instantiated but receives no gradient. This separates the comparison with a tuned baseline from the stricter experiment that changes only the loss terms. Seed standard deviations describe training variability rather than biological uncertainty. Loss coefficients are not directly comparable across datasets because output scaling, physical time units, and residual magnitudes differ.

```

## New Results subsection and tables

```latex
\subsection{Hypocotyl growth: sparse replicate snapshots and light forcing}
\label{subsec:hypocotylresults}

The additional experiment extends the environmental input from temperature to an explicit lighting schedule, while changing the observation unit from verified plant/plot trajectories to genotype--condition means assembled from replicate snapshots. Table~\ref{tab:hypocotyl_replicate} evaluates held-out replicate measurements; Table~\ref{tab:hypocotyl_time_holdout} evaluates the two times absent from both training and validation.

For replicate-group holdout, validation selected learning rate 0.01 and $(\lambda_{\mathrm{ODE}},\lambda_K)=(50,0.01)$. \model{} achieved test RMSE / relative RMSE of $0.344\pm0.008$~mm / $5.93\pm0.14\%$. The smallest mean test error was attained by the independently tuned no-physics latent ODE ($0.337\pm0.003$~mm / $5.81\pm0.05\%$); the physics-regularized model therefore did not lead this comparison. Its mean RMSE was 2.04\% higher than that of the independently learning-rate-tuned no-physics latent ODE.

\begin{table}[t]
\centering
\small
\caption{Hypocotyl replicate-group holdout errors: RMSE (mm) / relative RMSE (\%). Bold marks the smallest mean in each split. Stochastic models use three seeds (mean $\pm$ sample SD); process ODEs are single deterministic fits. Scores compare population-mean curves, with individual-measurement errors reported separately.}
\label{tab:hypocotyl_replicate}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccc}
\toprule
Model & Train & Validation & Test \\
\midrule
PhytoODE & $0.207\pm0.034 / 3.57\pm0.58$ & $0.313\pm0.007 / 5.35\pm0.12$ & $0.344\pm0.008 / 5.93\pm0.14$ \\
Latent ODE (no physics) & $0.212\pm0.024 / 3.66\pm0.41$ & $\mathbf{0.311\pm0.006 / 5.33\pm0.10}$ & $\mathbf{0.337\pm0.003 / 5.81\pm0.05}$ \\
Light-PINN & $0.204\pm0.019 / 3.52\pm0.33$ & $0.318\pm0.003 / 5.45\pm0.05$ & $0.347\pm0.015 / 5.99\pm0.26$ \\
LSTM-NN & $\mathbf{0.148\pm0.010 / 2.55\pm0.17}$ & $0.319\pm0.010 / 5.47\pm0.17$ & $0.352\pm0.004 / 6.07\pm0.07$ \\
Random forest & $0.290\pm0.008 / 5.00\pm0.13$ & $0.391\pm0.004 / 6.70\pm0.07$ & $0.401\pm0.011 / 6.91\pm0.19$ \\
Light-logistic ODE & $0.361 / 6.22$ & $0.446 / 7.64$ & $0.465 / 8.03$ \\
Logistic ODE & $0.338 / 5.82$ & $0.402 / 6.89$ & $0.439 / 7.57$ \\
\bottomrule
\end{tabular}}
\end{table}


For unseen-time interpolation, validation selected learning rate 0.01 and $(\lambda_{\mathrm{ODE}},\lambda_K)=(0.5,0.01)$. \model{} achieved test RMSE / relative RMSE of $0.305\pm0.039$~mm / $4.13\pm0.53\%$. This was the smallest mean test error among the seven methods under this protocol. Its mean RMSE was 1.61\% lower than that of the independently learning-rate-tuned no-physics latent ODE.

\begin{table}[t]
\centering
\small
\caption{Hypocotyl unseen-time interpolation errors: RMSE (mm) / relative RMSE (\%). Bold marks the smallest mean in each split. Stochastic models use three seeds (mean $\pm$ sample SD); process ODEs are single deterministic fits. Scores compare population-mean curves, with individual-measurement errors reported separately.}
\label{tab:hypocotyl_time_holdout}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccc}
\toprule
Model & Train & Validation & Test \\
\midrule
PhytoODE & $0.141\pm0.015 / 2.73\pm0.30$ & $0.315\pm0.011 / 6.04\pm0.20$ & $\mathbf{0.305\pm0.039 / 4.13\pm0.53}$ \\
Latent ODE (no physics) & $0.149\pm0.025 / 2.89\pm0.49$ & $0.308\pm0.003 / 5.91\pm0.06$ & $0.310\pm0.018 / 4.20\pm0.24$ \\
Light-PINN & $0.140\pm0.036 / 2.72\pm0.69$ & $\mathbf{0.303\pm0.003 / 5.81\pm0.06}$ & $0.308\pm0.011 / 4.17\pm0.15$ \\
LSTM-NN & $\mathbf{0.044\pm0.027 / 0.86\pm0.52}$ & $0.314\pm0.009 / 6.02\pm0.18$ & $0.343\pm0.041 / 4.65\pm0.56$ \\
Random forest & $0.388\pm0.017 / 7.53\pm0.33$ & $0.457\pm0.011 / 8.76\pm0.21$ & $0.809\pm0.034 / 10.96\pm0.46$ \\
Light-logistic ODE & $0.324 / 6.28$ & $0.450 / 8.63$ & $0.493 / 6.67$ \\
Logistic ODE & $0.334 / 6.47$ & $0.429 / 8.23$ & $0.327 / 4.43$ \\
\bottomrule
\end{tabular}}
\end{table}


\begin{figure}[!htbp]
\centering
\includegraphics[width=0.96\linewidth]{figures/hypocotyl_light_interpolation.pdf}
\caption{Hypocotyl predictions for the wild type (Col-0) and the defined MLB construct under 12L12D and continuous red light (cR). These panels illustrate the unseen-time protocol: hollow gray points are training replicate means at 0, 12, 24, 48 and 72~h; black points are test means at 36 and 60~h, which were absent from both training and validation. Colored lines are three-seed mean predictions, except for the single-fit process ODEs. Gray shading marks darkness in 12L12D. Curves describe population means, not tracked individual plants. All five genotypes are shown in the accompanying full-panel figures.}
\label{fig:hypocotyllight}
\end{figure}

In the stricter paired comparison at the selected learning rate and initial weights, the no-physics control achieved $0.335\pm0.002$~mm / $5.78\pm0.03\%$ for replicate holdout and $0.345\pm0.038$~mm / $4.68\pm0.51\%$ for unseen-time interpolation. Relative to these controls, \model{} had 2.69\% higher mean RMSE in replicate holdout and 11.60\% lower mean RMSE in time interpolation. These changes concern the combined derivative and capacity penalties. The ablation does not isolate the derivative penalty alone. The benefit therefore depends on the observation protocol, and the small differences between independently tuned neural models require confirmation in additional biological batches.

For comparison with the population-mean scores, pooled individual-measurement RMSE for \model{} was $0.784\pm0.005$~mm in the replicate holdout and $0.925\pm0.012$~mm at withheld times. Averaging replicates reduces the biological variation present in individual lengths, so the primary scores must not be interpreted as individual-plant prediction errors.

The withheld-time experiment provides an explicit test of interpolation from five observation times, beyond merely reporting performance on a 12-h measurement grid. The original workbook, cell-resolved extraction, verified summaries and frozen partitions also provide a reproducible author-collected dataset package. These are distinct contributions from the field-season and temperature experiments. Nevertheless, both lighting regimes and all five genotypes occur during training, and a single temperature does not identify temperature dependence. The unspecified 12L12D spectrum prevents separation of photoperiod and spectral effects. Twelve-hour sampling cannot resolve within-cycle circadian peaks, and unknown longitudinal identities and batch structure preclude claims about individual dynamics or independent-cohort generalization. The light-switched reference should consequently be interpreted as an effective growth constraint, rather than a recovered molecular clock.

As a numerical check, halving the integration step from 3 to 1.5~h while preserving the original scenario encoder changed the selected \model{} predictions by at most 0.000004~mm across the six protocol--seed fits. This check evaluates solver discretization sensitivity, not biological uncertainty.

\FloatBarrier

```

## Additional conclusion paragraph

```latex
The hypocotyl experiment extends the formulation to explicit illumination at fixed temperature and to population curves constructed from replicate snapshots. Its test relative RMSEs were $5.93\%$ for held-out replicate measurements and $4.13\%$ for entirely withheld observation times. This supports assessment under a new forcing variable and observation structure, while the accompanying data package makes extraction and partitioning inspectable. The biological conclusions remain limited to the observed genotypes and lighting regimes; unknown plant identities and batch structure and incomplete light metadata require further experimental documentation.


```

The generalized environmental notation and architecture wording are already incorporated in `main.tex` and the self-contained `main_standalone.tex`. Add the two new BibTeX entries from `references.bib`. Public deposition details remain to be supplied.
