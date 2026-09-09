# Primary replicate evaluation and changed manuscript passages (2026-09-09)

The manuscript uses replicate-group holdout for hypocotyls. The expanded PhytoODE search is an exploratory follow-up on an already inspected test partition; numerical rankings are reported as observed.

## Replacement abstract

```latex
\begin{abstract}
Predicting plant growth across genotypes and environments requires flexible dynamics under sparse observation. We present \model{}, a genotype- and environment-conditioned latent neural ordinary differential equation model. An encoder reads a known environmental scenario, a learned latent flow evolves continuously, and a decoder predicts phenotype without target-curve measurements as inputs. Logistic derivative and capacity penalties weakly regularize the decoded trajectory. Three temperature-conditioned benchmarks cover wheat with 19 genotypes, UAV maize with 402 genotypes, and Arabidopsis stem length with four observations per plant. Mean test RMSEs were $0.03032$~m, $56.68$ relative UAV-height units, and $2.800$~cm, respectively, corresponding to relative RMSEs of $10.24\%$, $18.40\%$, and $14.06\%$. These were lower than the compared benchmark families and matched no-physics controls. A fourth, author-collected dataset comprises 1,818 hypocotyl measurements across five genotypes and two light regimes at 23~$^{\circ}$C. We extend the growth constraint to separate light/dark rates and fit population-mean curves from replicate snapshots at seven times. On held-out replicate measurements, \model{} achieved $0.337$~mm RMSE ($5.81\%$ relative RMSE), without establishing superiority over the no-physics latent ODE controls. Follow-up tuning reused previously inspected test partitions; updated results therefore require independent confirmation. The experiments address field-season transfer, genotype-panel scale, sparse temporal sampling, and environmental-driver substitution. Evaluated genotypes and treatment regimes are represented during training, and the new dataset does not establish unseen-photoperiod prediction or individual-plant trajectory recovery.
\end{abstract}
```

## Replacement Methods subsections

```latex
\subsection{Author-collected hypocotyl measurements under light regimes}
\label{subsec:hypocotyldata}

We additionally analysed an author-provided experiment measuring \textit{Arabidopsis thaliana} hypocotyl length at a constant 23~$^{\circ}$C under 12-h light/12-h dark (12L12D) and continuous red light (cR). Five genotypes were included: Col-0, \textit{hy5}, MLB (\textit{ML1} promoter::phyB-GFP/\textit{phyB-9}), EMS57, and the \textit{phyA phyB} double mutant. Measurements were recorded in millimetres at seven elapsed times, 0--72~h in 12-h increments; the first 0--12-h interval was illuminated in 12L12D. The workbook contains 1,818 measured values, comprising 943 in 12L12D and 875 in cR, with 3--38 replicates in each genotype--condition--time group. Summary formulas were excluded, and all 70 recomputed group means matched the workbook summaries. The developmental age at the first observation was not confirmed; accordingly, time denotes elapsed hours rather than days after germination. Irradiance, the 12L12D spectrum, independent growth batches, and some mutant alleles remain undocumented.

The workbook does not establish longitudinal plant identifiers. We therefore model ten genotype--condition population-mean curves assembled from replicate snapshots, without interpreting each spreadsheet row as a repeatedly measured plant. For a conservative replicate-group holdout, genotype/source-row bookkeeping groups were assigned jointly across all times and both conditions to training, validation, and test partitions, using seed 20260909. A group was not allowed to cross partitions; partition generation checked coverage using identifiers only. This produced 1,052 training, 330 validation, and 436 test measurements. Split-specific means were computed independently, and all models were trained on the same 70 training means. These means contain 1--22 training, 1--7 validation, and 1--9 test measurements per group; the smallest groups therefore offer limited replication. This protocol evaluates held-out measurements within known genotype--condition combinations, without establishing a split between independent experimental batches.

All seven observation times occur in each partition. This evaluation concerns held-out replicate measurements under known genotype--condition combinations, consistent with separating biological samples rather than observation times. The primary metric is the mean RMSE across the ten population curves, with relative RMSE defined by Eq.~\eqref{eq:rrmse}. Pooled individual-measurement RMSE is reported separately to expose the difference between predicting a population mean and an individual plant.

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

For \model{}, the environmental channels are binary illumination and the known light duty cycle (0.5 for 12L12D; 1 for cR). The initial-state encoder reads this complete scenario. Temperature is constant and is not fitted as a driver. Phenotypes are divided by the maximum individual training measurement, 16.018~mm. A genotype-dependent head produces $r_{g,L},r_{g,D}\in(0,0.5)$~h$^{-1}$ and $K_g\in(0,2)$ in scaled-height units. Model capacity and optimization settings are selected as described below.

The latent flow generates the prediction; Eq.~\eqref{eq:light_logistic} supplies only the observable-space derivative residual. The chain rule in Eq.~\eqref{eq:decoder_chain_day} is unchanged, with $S=72$~h: $d\hat y/dt=(\nabla_{\mathbf z}G_\psi)^{\top}\mathbf F_\theta/72$. Automatic differentiation retains the graph through this derivative and through the Runge--Kutta solver. No finite differences of measured lengths are used. A 3-h grid gives 25 integration nodes. Illumination remains constant within each integration interval, including its right-end solver stage, and changes at the next interval; interpolation does not create an artificial ramp at a light switch. The derivative penalty is evaluated at the 18 interior non-switch nodes per curve, excluding the 12-h boundaries. The capacity penalty uses the maximum over the complete predicted grid. It is a finite-window consistency constraint, not a measurement of mature hypocotyl length.

\subsubsection{Baselines and validation-only selection}

The comparison includes \model{}, Latent ODE with both physics coefficients zero, a coordinate-MLP Light-PINN, LSTM-NN, random forest, light-logistic ODE, and ordinary logistic ODE. Light-PINN uses genotype, elapsed time, and duty cycle to predict length, and instantaneous illumination in the same derivative residual. It has a four-dimensional genotype embedding and two hidden layers of width 32. LSTM-NN uses the same embedding dimension, two recurrent layers of widths 16 and 8, and a dense hidden layer of width 16. The Light-PINN time derivative is obtained directly by automatic differentiation of its coordinate network; this is a distinct baseline from the earlier LSTM-based Logi-PINN. Random forest receives genotype, time, illumination, duty cycle, and accumulated light exposure, all determined by the known scenario. All learned models share the same training targets and available genotype/light information.

The initial comparison used 1,500 epochs, Adam with weight decay $10^{-4}$, gradient clipping at 1, and cosine learning-rate decay. Checkpoints were ranked after epoch 200 using the mean of the latest three validation evaluations, spaced 20 epochs apart. For both \model{} and Light-PINN, seed-1 screening crossed learning rates $\{0.003,0.01\}$, $\lambda_{\mathrm{ODE}}\in\{0.5,5,50,500\}$, and $\lambda_K\in\{0.01,0.1\}$. The no-physics latent ODE and LSTM screened the same two learning rates. Random forest used 300 trees and minimum leaf sizes $\{1,2,4\}$. The best two candidates per stochastic model were confirmed with seeds 2 and 3; selection then used three-seed mean validation RMSE. The original no-physics latent ODE has latent dimension 8, genotype embedding 4, vector-field and decoder width 16, a single vector-field hidden layer, and encoder width 8 (1,160 parameters including the unused auxiliary head).

A subsequent, fixed-budget \model{} search evaluated 80 additional configurations on the same training and validation partitions. Thirty-two configurations crossed learning rates $\{0.001,0.003,0.006,0.01\}$, $\lambda_{\mathrm{ODE}}\in\{0.05,0.5,5,50\}$, and $\lambda_K\in\{10^{-4},0.003\}$ at the original architecture, weight decay $10^{-4}$, and 3,000 epochs. Forty-eight configurations used stratified sampling in log space over learning rates $[10^{-3.3},10^{-1.7}]$, $\lambda_{\mathrm{ODE}}\in[10^{-2.3},10^{2.7}]$, $\lambda_K\in[10^{-5},10^{-1.5}]$, and weight decay $[10^{-6},10^{-2}]$, with 2,000, 3,000, or 4,500 epochs. Six capacity configurations used $(d_z,d_g,w_F,n_F,w_G,w_E)$ equal to $(4,2,8,1,8,4)$, $(4,4,16,1,16,8)$, $(8,4,16,1,16,8)$, $(12,4,24,1,24,12)$, $(16,4,32,1,32,16)$, or $(8,4,24,2,24,8)$, where the entries denote latent and genotype dimensions, vector-field width and depth, decoder width, and encoder width. Eight leading seed-1 candidates were confirmed with seeds 2 and 3, and the smallest three-seed mean validation RMSE selected the final configuration, with both previously confirmed \model{} configurations remaining eligible. The original baseline fits were retained, so tuning budgets are unequal.

This follow-up was initiated after the original test results had been inspected. The additional search used training and validation targets only, and its selected configuration and paired control were frozen before new test scoring. The search budget and stopping rule did not depend on the resulting test error. Nevertheless, evaluation on the reused test partition is exploratory and does not constitute an independent blinded confirmation.

The principal no-physics baseline retains its independent learning-rate selection from the initial comparison. An additional paired control is trained with the selected \model{} learning rate, architecture, weight decay, initial weights, optimization schedule, and checkpoint rule while setting $\lambda_{\mathrm{ODE}}=\lambda_K=0$. Its auxiliary parameter head remains instantiated but receives no gradient. The additional budget is 80 screening fits, 16 confirmation fits, and three paired-control fits. This separates the comparison with an independently tuned baseline from the stricter experiment that changes only the loss terms. Seed standard deviations describe training variability rather than biological uncertainty. Loss coefficients are not directly comparable across datasets because output scaling, physical time units, and residual magnitudes differ.

The initial 1,160-parameter no-physics control at learning rate 0.01 is also retained as a historical comparison. Its validation mean was higher than that of the independently selected learning rate 0.003. Reporting this earlier control does not change the validation-based selection rule or replace it with test-based selection.

```

## Replacement Results subsection and table

```latex
\subsection{Hypocotyl growth: replicate snapshots under light regimes}
\label{subsec:hypocotylresults}

The author-collected experiment extends the environmental input from temperature to an explicit lighting schedule and the observation structure to population curves assembled from replicate snapshots. Table~\ref{tab:hypocotyl_replicate} evaluates 436 held-out measurements as 70 genotype--condition--time means. Training, validation, and test each cover the same seven times and ten genotype--condition combinations. Thus the sparse 12-h sampling is a property of the observations, while generalization is evaluated across the separated replicate groups.

Validation selected learning rate 0.0015527, weight decay 0.00036574, and $(\lambda_{\mathrm{ODE}},\lambda_K)=(0.12644,0.0027023)$. The selected architecture has latent dimension 12, genotype embedding 4, vector-field width 24 and depth 1, decoder width 24, and encoder width 12, for 2,224 parameters including the auxiliary head. Training ran for 3,000 epochs, with the reported checkpoint selected by validation. \model{} achieved test RMSE / relative RMSE of $0.337\pm0.002$~mm / $5.81\pm0.04\%$. The smallest mean test error was attained by the initial-architecture no-physics control ($0.335\pm0.002$~mm / $5.78\pm0.03\%$). \model{} had mean test RMSE 0.02\% lower than that of the independently learning-rate-tuned latent ODE ($0.337\pm0.003$~mm / $5.81\pm0.05\%$). The absolute gap was only 0.000065~mm, far below the seed standard deviations, so these two methods were effectively tied under this comparison.

\begin{table}[t]
\centering
\small
\caption{Hypocotyl replicate-group holdout: RMSE (mm) / relative RMSE (\%). Bold marks the smallest unrounded mean among all rows; subprecision differences are discussed in the text. Stochastic fits use three seeds (mean $\pm$ sample SD); process ODEs are single fits. The main no-physics baseline retains independent learning-rate selection. The matched control uses final PhytoODE settings; the initial control retains the earlier architecture and learning rate 0.01. Both physics coefficients are zero in every no-physics row. Scores compare population-mean curves.}
\label{tab:hypocotyl_replicate}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccc}
\toprule
Model & Train & Validation & Test \\
\midrule
PhytoODE & $0.211\pm0.010 / 3.64\pm0.18$ & $\mathbf{0.300\pm0.001 / 5.14\pm0.03}$ & $0.337\pm0.002 / 5.81\pm0.04$ \\
Latent ODE (no physics) & $0.212\pm0.024 / 3.66\pm0.41$ & $0.311\pm0.006 / 5.33\pm0.10$ & $0.337\pm0.003 / 5.81\pm0.05$ \\
Light-PINN & $0.204\pm0.019 / 3.52\pm0.33$ & $0.318\pm0.003 / 5.45\pm0.05$ & $0.347\pm0.015 / 5.99\pm0.26$ \\
LSTM-NN & $\mathbf{0.148\pm0.010 / 2.55\pm0.17}$ & $0.319\pm0.010 / 5.47\pm0.17$ & $0.352\pm0.004 / 6.07\pm0.07$ \\
Random forest & $0.290\pm0.008 / 5.00\pm0.13$ & $0.391\pm0.004 / 6.70\pm0.07$ & $0.401\pm0.011 / 6.91\pm0.19$ \\
Light-logistic ODE & $0.361 / 6.22$ & $0.446 / 7.64$ & $0.465 / 8.03$ \\
Logistic ODE & $0.338 / 5.82$ & $0.402 / 6.89$ & $0.439 / 7.57$ \\
\midrule
Latent ODE (matched) & $0.219\pm0.011 / 3.78\pm0.19$ & $0.300\pm0.002 / 5.14\pm0.03$ & $0.340\pm0.005 / 5.87\pm0.09$ \\
Latent ODE (initial control) & $0.173\pm0.019 / 2.98\pm0.33$ & $0.314\pm0.009 / 5.37\pm0.15$ & $\mathbf{0.335\pm0.002 / 5.78\pm0.03}$ \\
\bottomrule
\end{tabular}}
\end{table}


The initial \model{} comparison gave $0.344$~mm RMSE ($5.93\%$ relative RMSE). The present configuration was selected in the subsequent validation-only search described in Methods. Because that search followed inspection of the initial test results and retained the original baseline fits, the updated comparison has unequal tuning budgets and a reused test partition; it is an exploratory follow-up.

For completeness, the earlier no-physics control at the initial 1,160-parameter architecture and learning rate 0.01 achieved $0.335\pm0.002$~mm / $5.78\pm0.03\%$. This is lower than the newly tuned \model{} result. It was not selected as the principal no-physics baseline because its validation mean was worse than that at learning rate 0.003; it is retained as a historical control rather than selected using test error. The expanded search therefore does not establish superiority over all evaluated latent ODE configurations.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.96\linewidth]{figures/hypocotyl_light_replicates.pdf}
\caption{Hypocotyl predictions in the replicate-group holdout for the wild type (Col-0) and the defined MLB construct under 12L12D and continuous red light (cR). Black points are means of the held-out replicate measurements at each of the seven observation times. Colored lines are three-seed mean predictions, except for the single-fit process ODEs. Gray shading marks darkness in 12L12D. All seven times are represented in training and validation using their respective replicate groups. Curves describe population means, not tracked individual plants. The no-physics latent ODE uses its original independent learning-rate selection; the matched loss-removal control is reported in Table~\ref{tab:hypocotyl_replicate} and the text. All five genotypes are shown in the accompanying full-panel figure.}
\label{fig:hypocotyllight}
\end{figure}

The additional no-physics control, matched to the selected architecture, learning rate, weight decay, initial weights, schedule, and checkpoint rule, achieved $0.340\pm0.005$~mm / $5.87\pm0.09\%$. Relative to this control, \model{} had 1.08\% lower mean test RMSE. Both $\lambda_{\mathrm{ODE}}$ and $\lambda_K$ are zero in the control. This comparison concerns the combined derivative and capacity penalties and does not isolate the derivative penalty alone. The small difference across three seeds does not establish statistical superiority. Seed variation describes optimization variability; independent biological batches are needed to assess reproducibility.

Pooled individual-measurement RMSE for \model{} was $0.7794\pm0.0003$~mm. Averaging replicates reduces the biological variation present in individual lengths, so the population-mean scores must not be interpreted as individual-plant prediction errors.

This experiment demonstrates how the same latent-dynamics formulation can use a lighting schedule at fixed temperature and learn from sparsely sampled replicate measurements. The original workbook, cell-resolved extraction, verified summaries, and frozen partitions provide an inspectable author-collected dataset package. Both lighting regimes and all five genotypes occur during training, and a single temperature does not identify temperature dependence. The unspecified 12L12D spectrum prevents separation of photoperiod and spectral effects. Twelve-hour sampling cannot resolve within-cycle circadian peaks, and unknown longitudinal identities and batch structure preclude claims about individual dynamics or independent-cohort generalization. The light-switched reference is an effective growth constraint rather than a recovered molecular clock.

As a numerical check, halving the integration step from 3 to 1.5~h while preserving the scenario encoder changed the selected \model{} predictions by at most 0.000012~mm across the three seeds. This evaluates solver discretization sensitivity rather than biological uncertainty.

\FloatBarrier

```

## Replacement conclusion paragraph

```latex
The hypocotyl experiment extends the formulation to explicit illumination at fixed temperature and to population curves constructed from replicate snapshots. Its test relative RMSE was $5.81\%$ for held-out replicate measurements, with all seven observation times represented in each partition. This effectively tied the validation-selected no-physics baseline and did not improve on the initial-architecture no-physics control; the current matched comparison showed only a small mean difference. The unequal follow-up tuning budgets and reused test partition further limit interpretation of the regularizers. The accompanying data package makes extraction and partitioning inspectable. The biological conclusions remain limited to the observed genotypes and lighting regimes; unknown plant identities and batch structure and incomplete light metadata require further experimental documentation.
```

The Introduction and task definition in main.tex and main_standalone.tex also use the replicate-group protocol. Public deposition details remain to be supplied.
