# Individual-observation experiment: replacement passages

Measured cells are retained individually; neither means nor imputed phenotypes are targets. The former mean-target comparison paragraphs are replaced by the new raw-observation results. Partition provenance remains stated once in the hypocotyl Methods.

## Abstract

```latex
\begin{abstract}
Predicting plant growth across genotypes and environments requires flexible dynamics under sparse observation. We present \model{}, a genotype- and environment-conditioned latent neural ordinary differential equation model. An encoder reads a known environmental scenario, a learned latent flow evolves continuously, and a decoder predicts phenotype without target-curve measurements as inputs. Logistic derivative and capacity penalties weakly regularize the decoded trajectory. Three temperature-conditioned benchmarks cover wheat with 19 genotypes, UAV maize with 402 genotypes, and Arabidopsis stem length with four observations per plant. Mean test RMSEs were $0.03032$~m, $56.68$ relative UAV-height units, and $2.800$~cm, respectively, corresponding to relative RMSEs of $10.24\%$, $18.40\%$, and $14.06\%$. These were lower than the compared benchmark families and matched no-physics controls. A fourth, author-collected dataset comprises 1,818 hypocotyl measurements across five genotypes and two light regimes at 23~$^{\circ}$C. A light-switched logistic constraint introduces separate effective light and dark growth rates. In this experiment, all models learn directly from individual measured lengths, with no phenotype imputation or replicate averaging. On 436 held-out measurements, \model{} achieved $0.779$~mm RMSE ($13.72\%$ relative RMSE); the independently tuned no-physics latent ODE obtained $0.782$~mm ($13.77\%$). The experiments address field-season transfer, genotype-panel scale, sparse temporal sampling, and environmental-driver substitution. Evaluated genotypes and treatment regimes are represented during training.
\end{abstract}
```

## Methods

```latex
\subsection{Author-collected hypocotyl measurements under light regimes}
\label{subsec:hypocotyldata}

We additionally analysed an author-provided experiment measuring \textit{Arabidopsis thaliana} hypocotyl length at a constant 23~$^{\circ}$C under 12-h light/12-h dark (12L12D) and continuous red light (cR). Five genotypes were included: Col-0, \textit{hy5}, MLB (\textit{ML1} promoter::phyB-GFP/\textit{phyB-9}), EMS57, and the \textit{phyA phyB} double mutant. Measurements were recorded in millimetres at seven elapsed times, 0--72~h in 12-h increments; the first 0--12-h interval was illuminated in 12L12D. The workbook contains 1,818 measured values, comprising 943 in 12L12D and 875 in cR, with 3--38 replicates in each genotype--condition--time group. Summary formulas were excluded, and all 70 recomputed group means matched the workbook summaries. The developmental age at the first observation was not confirmed; accordingly, time denotes elapsed hours rather than days after germination. Irradiance, the 12L12D spectrum, independent growth batches, and some mutant alleles remain undocumented.

The workbook does not establish longitudinal plant identifiers. We fit genotype--condition growth curves directly to individual measured lengths, without interpreting each spreadsheet row as a repeatedly measured plant. For a conservative replicate-group holdout, genotype/source-row bookkeeping groups were assigned jointly across all times and both conditions to training, validation, and test partitions, using seed 20260909. A group was not allowed to cross partitions; partition generation checked coverage using identifiers only. This produced 1,052 training, 330 validation, and 436 test measurements. Every measured cell remains a separate target for every model. Blank cells are omitted; no phenotype value is imputed, interpolated, or replaced by a replicate mean. The workbook summary means are used only to verify extraction. This protocol evaluates held-out measurements within known genotype--condition combinations, without establishing a split between independent experimental batches.

All seven observation times occur in each partition, with unequal numbers of available replicates. Let $\Omega_s$ contain the measured cells in partition $s$, and let cell $j$ have genotype $g_j$, condition $c_j$, elapsed time $t_j$, and measured length $H_j$. With the training-only scale $a$, the hypocotyl data loss is
\begin{equation}
 \mathcal L_{\mathrm{data}}^{\mathrm{hyp}}
 =\left[\frac{1}{|\Omega_{\mathrm{train}}|}
 \sum_{j\in\Omega_{\mathrm{train}}}
 \left(\hat y_{g_j,c_j}(t_j)-\frac{H_j}{a}\right)^2\right]^{1/2}.
 \label{eq:hypocotyl_data_loss}
\end{equation}
The model prediction at a given genotype, condition, and time is compared separately with every available replicate. Missing entries contribute no residual or data-loss gradient. Individual identity is not a model input, so this does not imply recovery of distinct individual trajectories. The primary scores pool the individual measured residuals,
\begin{align}
 \mathrm{RMSE}_s&=\left[\frac{1}{|\Omega_s|}\sum_{j\in\Omega_s}
 (a\hat y_{g_j,c_j}(t_j)-H_j)^2\right]^{1/2},\\
 \mathrm{rRMSE}_s&=100\,\frac{\mathrm{RMSE}_s}
 {|\Omega_s|^{-1}\sum_{j\in\Omega_s}H_j}.
 \label{eq:hypocotyl_metrics}
\end{align}
These scores retain variation among individual lengths. Each measured cell has equal weight; more densely observed genotype--condition--time groups consequently contribute more residuals. Relative RMSE is mean-normalized RMSE, not mean absolute percentage error.

\subsubsection{Light-dependent growth reference and latent dynamics}
\label{subsec:lightmodel}

Logistic functions have been applied to hypocotyl elongation under light and dark growth conditions \citep{alimchandani2026atlas}. Light and the circadian clock jointly regulate elongation through PIF4/PIF5 \citep{nozue2007rhythmic}, and quantitative clock-output models incorporate this light-dependent regulation \citep{seaton2015linked}. These observations motivate a compact phenomenological reference. We use a zero-offset logistic law and introduce two effective growth-rate parameters selected by the known illumination schedule:
\begin{equation}
 \frac{dH_g}{dt}=\rho_g(t)H_g\left(1-\frac{H_g}{K_g}\right),
 \qquad \rho_g(t)=r_{g,L}L(t)+r_{g,D}[1-L(t)],
 \label{eq:light_logistic}
\end{equation}
where $L(t)$ is one in light and zero in darkness. The positive rates $r_{g,L}$ and $r_{g,D}$ are expressed in h$^{-1}$, without imposing an ordering between them. Equation~\eqref{eq:light_logistic} is our extension combining logistic saturation with a binary rate switch; the cited studies do not propose this exact equation. The rates represent effective behavior within each phase and do not resolve circadian gating or rapid switching transients. For accumulated illumination $A(t)=\int_0^t L(s)\,ds$, the process baseline uses the exact solution
\begin{equation}
 H_{g,c}(t)=\frac{K_g}{1+(K_g/H_{g,c}(0)-1)
 \exp\{-r_{g,L}A(t)-r_{g,D}[t-A(t)]\}}.
 \label{eq:light_solution}
\end{equation}
The two rates and capacity are shared across conditions within genotype, while initial height is fitted separately for each genotype--condition pair, giving 25 parameters. The ordinary logistic baseline instead fits three parameters independently to each of the ten curves (30 parameters). Both minimize residuals against every individual training measurement, with eight deterministic starting points; evaluated-curve initial heights are never supplied as inputs.

For \model{}, the environmental channels are binary illumination and the known light duty cycle (0.5 for 12L12D; 1 for cR). The initial-state encoder reads this complete scenario. Temperature is constant and is not fitted as a driver. Phenotypes are divided by the maximum individual training measurement, 16.018~mm. A genotype-dependent head produces $r_{g,L},r_{g,D}\in(0,0.5)$~h$^{-1}$ and $K_g\in(0,2)$ in scaled-height units. Model capacity and optimization settings are selected as described below.

The latent flow generates the prediction; Eq.~\eqref{eq:light_logistic} supplies only the observable-space derivative residual. The chain rule in Eq.~\eqref{eq:decoder_chain_day} is unchanged, with $S=72$~h: $d\hat y/dt=(\nabla_{\mathbf z}G_\psi)^{\top}\mathbf F_\theta/72$. Automatic differentiation retains the graph through this derivative and through the Runge--Kutta solver. No finite differences of measured lengths are used. A 3-h grid gives 25 integration nodes. Illumination remains constant within each integration interval, including its right-end solver stage, and changes at the next interval; interpolation does not create an artificial ramp at a light switch. The derivative penalty is evaluated at the 18 interior non-switch nodes per curve, excluding the 12-h boundaries. The capacity penalty uses the maximum over the complete predicted grid. It is a finite-window consistency constraint, not a measurement of mature hypocotyl length.

\subsubsection{Baselines and validation-only selection}

The comparison includes \model{}, Latent ODE with both physics coefficients zero, a coordinate-MLP Light-PINN, LSTM-NN, random forest, light-logistic ODE, and ordinary logistic ODE. Light-PINN uses genotype, elapsed time, and duty cycle to predict length, and instantaneous illumination in the same derivative residual. It has a four-dimensional genotype embedding and two hidden layers of width 32. LSTM-NN uses the same embedding dimension, two recurrent layers of widths 16 and 8, and a dense hidden layer of width 16. The Light-PINN time derivative is obtained directly by automatic differentiation of its coordinate network; this is a distinct baseline from the earlier LSTM-based Logi-PINN. Random forest receives genotype, time, illumination, duty cycle, and accumulated light exposure, all determined by the known scenario. All learned models share the same training targets and available genotype/light information.

All models are fitted anew to individual observations. The neural models use 2,000 epochs, Adam, gradient clipping at 1, and cosine learning-rate decay. Checkpoints are selected after epoch 200 using the mean of the latest three validation RMSE evaluations, spaced 20 epochs apart. Screening and all checkpoint decisions use individual-observation validation RMSE.

For both \model{} and the independently tuned no-physics latent ODE, the search contains 24 paired architecture/optimizer configurations. Three capacity presets have $(d_z,d_g,w_F,n_F,w_G,w_E)$ equal to $(8,4,16,1,16,8)$, $(12,4,24,1,24,12)$, or $(16,4,32,1,32,16)$; these are crossed with learning rates $\{0.001,0.003,0.006,0.01\}$ and weight decays $\{10^{-4},10^{-3}\}$. One pair is replaced by the preceding validation-selected optimizer setting (learning rate 0.0015527, weight decay 0.00036574, and the 12-dimensional architecture). The positive coefficients for \model{} are assigned by fixed stratified log-space sampling over $\lambda_{\mathrm{ODE}}\in[10^{-2},10^2]$ and $\lambda_K\in[10^{-5},10^{-1}]$, with the same replaced pair retaining $(0.126442,0.00270233)$. The no-physics candidates have both coefficients zero. Each model independently selects its best three-seed mean validation RMSE from its leading three seed-1 candidates.

Light-PINN screens 12 configurations. The learning rates are $0.001$, $0.003$, and $0.01$; the coefficient pairs are $(0.05,10^{-4})$, $(0.5,0.003)$, $(5,0.01)$, and $(50,0.1)$. LSTM-NN screens four learning rates $\{0.001,0.003,0.006,0.01\}$. Both use weight decay $10^{-4}$. Random forest uses 300 trees and minimum leaf sizes $\{1,2,4,8\}$, fitting one training row per measured cell. The leading two candidates for each of these three stochastic baselines are confirmed with seeds 2 and 3. Together with the two deterministic process baselines, the primary search comprises 70 screening fits and 24 confirmation fits; a matched control requires at most three additional fits.

The replicate partitions were retained from preliminary analyses in which test results had already been inspected. In the present experiment, all models are retrained, and configurations are selected using training and validation data and frozen before test scoring. The predefined search is not extended according to test results.

The matched no-physics control uses the selected \model{} architecture, learning rate, weight decay, initial weights, schedule, and checkpoint rule, with $\lambda_{\mathrm{ODE}}=\lambda_K=0$. Its auxiliary head is instantiated identically but receives no gradient. Existing fits from the present raw-observation search are reused when their configuration and seed match exactly. This control changes only the two physics penalties; the independently tuned latent ODE provides a separate comparison with equal screening and confirmation budgets. All stochastic results report the mean and sample standard deviation across three training seeds. These seed deviations quantify optimization variability, not biological sampling uncertainty.

```

## Results, table and caption

```latex
\subsection{Hypocotyl growth: individual observations under light regimes}
\label{subsec:hypocotylresults}

The author-collected experiment tests environmental-driver substitution and learning from incomplete replicate observations. Temperature was fixed at 23~$^{\circ}$C, while the known illumination schedule supplied the environmental input. All models were trained directly on the 1,052 measured training lengths. Table~\ref{tab:hypocotyl_replicate} reports errors against the 330 validation and 436 test measurements individually, without constructing replicate-mean targets. The same seven observed times and ten genotype--condition combinations occur in each partition.

Validation selected a 1,160-parameter \model{} with latent dimension 8, genotype embedding 4, vector-field width 16 and depth 1, decoder width 16, and encoder width 8. The learning rate was 0.01, weight decay 0.0001, and $(\lambda_{\mathrm{ODE}},\lambda_K)=(0.0161777,0.000813047)$. All neural candidates used 2,000 epochs and validation-selected checkpoints.

\model{} achieved test RMSE / relative RMSE of $0.7788\pm0.0061$~mm / $13.72\pm0.11\%$, compared with $0.7815\pm0.0045$~mm / $13.77\pm0.08\%$ for the independently tuned no-physics latent ODE. The smallest mean test RMSE was attained by LSTM-NN ($0.7741$~mm / $13.64\%$).

\begin{table}[t]
\centering
\small
\caption{Hypocotyl replicate-group holdout: individual-observation RMSE (mm) / relative RMSE (\%). Each measured cell is scored separately. Stochastic models report mean $\pm$ sample SD across three seeds; process ODEs are single fits. Bold marks the smallest unrounded mean in each split. The matched control uses the selected PhytoODE architecture and training settings with both physics coefficients zero.}
\label{tab:hypocotyl_replicate}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccc}
\toprule
Model & Train & Validation & Test \\
\midrule
PhytoODE & $0.754\pm0.007 / 13.16\pm0.11$ & $0.718\pm0.001 / 12.33\pm0.03$ & $0.779\pm0.006 / 13.72\pm0.11$ \\
Latent ODE (no physics) & $0.755\pm0.002 / 13.19\pm0.03$ & $\mathbf{0.716\pm0.001 / 12.29\pm0.02}$ & $0.782\pm0.005 / 13.77\pm0.08$ \\
Light-PINN & $0.767\pm0.005 / 13.40\pm0.08$ & $0.726\pm0.004 / 12.47\pm0.08$ & $0.793\pm0.005 / 13.97\pm0.08$ \\
LSTM-NN & $0.739\pm0.002 / 12.91\pm0.04$ & $0.719\pm0.006 / 12.35\pm0.10$ & $\mathbf{0.774\pm0.003 / 13.64\pm0.05}$ \\
Random forest & $\mathbf{0.731\pm0.000 / 12.77\pm0.00}$ & $0.728\pm0.000 / 12.51\pm0.01$ & $0.778\pm0.001 / 13.71\pm0.01$ \\
Light-logistic ODE & $0.821 / 14.34$ & $0.802 / 13.77$ & $0.844 / 14.86$ \\
Logistic ODE & $0.807 / 14.10$ & $0.771 / 13.24$ & $0.827 / 14.57$ \\
\midrule
Latent ODE (matched) & $0.751\pm0.005 / 13.12\pm0.09$ & $0.718\pm0.002 / 12.32\pm0.04$ & $0.777\pm0.005 / 13.69\pm0.09$ \\
\bottomrule
\end{tabular}}
\end{table}


The control with the same architecture, optimizer settings, initial weights, schedule, and checkpoint rule, but both physics coefficients zero, obtained $0.7769\pm0.0049$~mm / $13.69\pm0.09\%$. \model{} had 0.24\% higher mean test RMSE than this control. This comparison measures the combined contribution of the derivative and capacity penalties. The independently tuned latent ODE has the same number of screening and confirmation fits as \model{}; its configuration is selected separately by validation.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.96\linewidth]{figures/hypocotyl_light_replicates.pdf}
\caption{Hypocotyl predictions for Col-0 and MLB under 12L12D and continuous red light (cR). Every black point is one held-out measured length; observations are not averaged. Small horizontal offsets separate overlapping points and do not change their recorded times. Colored curves show the mean prediction across three training seeds, except for the single-fit process ODEs. Gray shading indicates darkness in 12L12D. All seven times are represented in training and validation using separate replicate groups. Each genotype--condition curve is fitted to individual measurements, without inferring longitudinal plant identities. The matched loss-removal control is reported in Table~\ref{tab:hypocotyl_replicate}; all five genotypes are shown in the accompanying full-panel figure.}
\label{fig:hypocotyllight}
\end{figure}

This experiment demonstrates a shared growth-prediction framework across species, organs, scales, and environmental drivers. Its seven measurement times remain sparse relative to the continuous latent dynamics, and unequal replicate counts and missing cells are handled by evaluating residuals only where measurements exist. Observation-level scoring retains the variation among measured plants. Since individual covariates are unavailable, plants with the same genotype, lighting regime, and observation time receive the same predicted length; the experiment does not establish individualized trajectory recovery.

The original workbook, cell-level provenance, extraction checks, and fixed partitions form a reproducible author-collected dataset package. Both light regimes and all five genotypes are represented during training. A single temperature cannot identify temperature dependence, and the unrecorded 12L12D spectrum prevents separation of spectral and photoperiod effects. The phase-specific logistic rates provide a compact reference, while 12-h sampling and unknown plant identities and batch structure limit inference about circadian timing and independent-cohort generalization.

Halving the integration step from 3 to 1.5~h while preserving the encoder inputs changed \model{} predictions by at most 0.000010~mm across the three seeds.

\FloatBarrier

```

## Hypocotyl conclusion paragraph

```latex
The hypocotyl experiment extends the formulation to explicit illumination at fixed temperature and direct fitting to incomplete individual observations. On 436 held-out measured lengths, its test relative RMSE was $13.72\%$, compared with $13.77\%$ for the independently tuned no-physics latent ODE and $13.69\%$ for the matched control. All seven times are represented in each partition, and missing phenotype cells contribute no data residual. The accompanying data package makes extraction and partitioning inspectable. Further biological evaluation requires independently documented cohorts and more complete light metadata.
```
