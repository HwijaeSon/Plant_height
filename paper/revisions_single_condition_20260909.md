# 12L12D-only experiment: replacement passages

cR is excluded. Inputs are genotype and elapsed time only. Training and primary metrics use split-specific replicate means, following the restored initial protocol. The primary manuscript evaluation remains replicate-group holdout; the auxiliary 36/60-h evaluation is retained in the experiment report.

## Abstract

```latex
\begin{abstract}
Predicting plant growth across genotypes and environments requires flexible dynamics under sparse observation. We present \model{}, a genotype-conditioned latent neural ordinary differential equation model with optional environmental inputs. An encoder initializes the latent state, a learned continuous flow evolves it, and a decoder predicts phenotype without target-curve measurements as inputs. Logistic derivative and capacity penalties weakly regularize the decoded trajectory. Three temperature-conditioned benchmarks cover wheat with 19 genotypes, UAV maize with 402 genotypes, and Arabidopsis stem length with four observations per plant. Mean test RMSEs were $0.03032$~m, $56.68$ relative UAV-height units, and $2.800$~cm, respectively, corresponding to relative RMSEs of $10.24\%$, $18.40\%$, and $14.06\%$. These were lower than the compared benchmark families and matched no-physics controls. A fourth, author-collected experiment uses 943 hypocotyl measurements across five genotypes under 12-h light/12-h dark at 23~$^{\circ}$C. All models in this experiment receive genotype and elapsed time only, and use split-specific replicate means for training and primary evaluation. With an ordinary logistic reference, \model{} achieved $0.365$~mm RMSE ($7.22\%$ relative RMSE) on 35 held-out mean targets formed from 225 measurements; the independently selected no-physics latent ODE obtained $0.368$~mm ($7.28\%$). The experiments address field-season transfer, genotype-panel scale, sparse temporal sampling, and genotype-level mean growth from replicate snapshots. Evaluated genotypes and treatment regimes are represented during training.
\end{abstract}
```

## Methods

```latex
\subsection{Author-collected hypocotyl dataset and evaluation}
\label{subsec:hypocotyldata}

We analysed an author-collected experiment measuring \textit{Arabidopsis thaliana} hypocotyl length at a constant 23~$^{\circ}$C under 12-h light/12-h dark (12L12D). The five genotypes are Col-0, \textit{hy5}, MLB (\textit{ML1} promoter::phyB-GFP/\textit{phyB-9}), EMS57, and the \textit{phyA phyB} double mutant. Lengths were recorded in millimetres at seven elapsed times, 0--72~h in 12-h increments; the first 0--12-h interval was illuminated. We retain 943 measurements from the 12L12D sheet. The continuous-red-light sheet is excluded from every partition in this analysis. Time denotes elapsed hours because developmental age at the first observation was not confirmed. Irradiance, the light spectrum, independent growth batches, and some mutant alleles remain undocumented.

The workbook does not establish longitudinal plant identifiers. We therefore fit five genotype-level growth curves to replicate snapshots. The existing genotype/source-row bookkeeping assignments were filtered to 12L12D without reassignment; each group remains in one partition across observation times. This gives 547 training, 171 validation, and 225 test measurements, with all genotypes and times represented in every partition. The protocol holds out replicate groups within one experiment, rather than independently documented experimental batches.

Training and primary evaluation use replicate means calculated separately within each partition. Let $\bar H_{s,g}(t)$ average the available lengths for genotype $g$ and time $t$ in partition $s$ only. Each partition contains 35 targets, one per genotype--time combination. Blank cells are omitted before averaging; an absent genotype--time group supplies no target. No missing time is filled by interpolation or imputation. Individual measurements are retained for secondary RMSE reporting but are not the targets of the primary comparison.

With the training-only scale $a=13.719$~mm, the data loss is
\begin{equation}
 \mathcal L_{\mathrm{data}}^{\mathrm{hyp}}
 =\frac{1}{5}\sum_{g=1}^{5}\left[
 \frac{1}{|\mathcal T_{\mathrm{train},g}|}
 \sum_{t\in\mathcal T_{\mathrm{train},g}}
 \left(\hat y_g(t)-\frac{\bar H_{\mathrm{train},g}(t)}{a}\right)^2
 \right]^{1/2},
 \label{eq:hypocotyl_data_loss}
\end{equation}
where $\mathcal T_{s,g}$ contains times with measurements in partition $s$. Primary RMSE is the mean of the five genotype-curve RMSEs against $\bar H_{s,g}(t)$, in millimetres. Relative RMSE divides this score by the mean of all scored replicate-mean targets in that partition and multiplies by 100. It is mean-normalized RMSE, not mean absolute percentage error. A replicate mean has equal weight within its genotype curve regardless of its replicate count. These scores assess prediction of mean growth rather than individual lengths.

\subsubsection{Ordinary logistic reference without environmental inputs}
\label{subsec:lightmodel}

Logistic functions have been used to describe hypocotyl elongation \citep{alimchandani2026atlas}. We use an ordinary, zero-offset logistic reference,
\begin{equation}
 \frac{dH_g}{dt}=r_g H_g\left(1-\frac{H_g}{K_g}\right),
 \qquad
 H_g(t)=\frac{K_g}{1+(K_g/H_g(0)-1)\exp(-r_g t)}.
 \label{eq:hypocotyl_logistic}
\end{equation}
Each genotype has a positive, time-independent rate $r_g$ and a capacity $K_g$. The process baseline fits $(r_g,K_g,H_g(0))$ separately for five genotypes, giving 15 parameters, using residuals against training replicate means and eight deterministic starting points. Initial height is estimated from training data; no held-out target is supplied as an input. The rate is an effective coefficient over the observation window, not a separate estimate of light- and dark-phase growth.

Every predictor receives genotype and elapsed time only. Illumination, duty cycle, accumulated light exposure, spectrum, and temperature are not input features. The 12L12D schedule and 23~$^{\circ}$C temperature describe the experiment but do not drive the models. This comparison therefore concerns growth under one fixed regime and does not evaluate transfer to other lighting or temperature conditions.

For \model{}, a four-dimensional genotype embedding and a time-sequence encoder of width 8 initialize an eight-dimensional latent state. The vector field has one hidden layer of width 16 and takes latent state, genotype embedding, and normalized time $\tau=t/72$ as inputs. The decoder has width 16. The encoder receives the time grid and genotype without phenotype measurements or environmental channels. An auxiliary genotype-dependent head outputs $r_g\in(0,0.5)$~h$^{-1}$ and $K_g\in(0,2)$ in scaled-height units. Including this head, the model has 1,055 parameters.

The latent flow determines predicted length; Eq.~\eqref{eq:hypocotyl_logistic} supplies only a soft residual on the decoded derivative,
\begin{equation}
 \frac{d\hat y_g}{dt}
 =\frac{1}{72}\nabla_{\mathbf z}G_\psi(\mathbf z_g)^{\top}
 \mathbf F_\theta(\mathbf z_g,\mathbf e_g,\tau).
 \label{eq:hypocotyl_derivative}
\end{equation}
Automatic differentiation computes the decoder gradient and retains the graph through the residual and Runge--Kutta solver. Thus the regularizer can update the latent dynamics, decoder, encoder, genotype embedding, and auxiliary parameters. No finite differences of measurements or replicate means are used. A 3-h grid provides 25 integration nodes; all 23 interior nodes contribute to the logistic derivative residual independently of the observation mask. The capacity penalty is the mean absolute difference between $K_g$ and the maximum predicted length on this grid. It is a finite-window consistency constraint, not an observation of mature hypocotyl length.

\subsubsection{Baselines and validation-based selection}

The comparison includes \model{}, a no-physics latent ODE, Logistic-PINN, LSTM-NN, random forest, and ordinary logistic ODE. Logistic-PINN is a coordinate MLP with a four-dimensional genotype embedding, normalized time, two hidden layers of width 32, and the same logistic derivative and capacity penalties. Its derivative is obtained directly by automatic differentiation of the coordinate network. It differs architecturally from the LSTM-based Logi-PINN in the temperature benchmarks. LSTM-NN uses time as its recurrent input, recurrent widths 16 and 8, and a four-dimensional genotype embedding concatenated before a dense head of width 16. Random forest uses genotype one-hot encoding and normalized time. No light-switched process ODE is included.

All models are trained anew using the initial-version search budget. PhytoODE and Logistic-PINN each screen 16 configurations, crossing learning rates $\{0.003,0.01\}$, derivative coefficients $\{0.5,5,50,500\}$, and capacity coefficients $\{0.01,0.1\}$. The no-physics latent ODE and LSTM-NN each screen the two learning rates. Random forest uses 300 trees and minimum leaf sizes $\{1,2,4\}$. Neural candidates run for 1,500 epochs using Adam, weight decay $10^{-4}$ on all trainable parameters, gradient clipping at 1, and cosine learning-rate decay. Checkpoints are selected after epoch 200 by the mean of the latest three validation evaluations, spaced 20 epochs apart. Each stochastic model's two leading seed-1 candidates are confirmed with seeds 2 and 3; the smallest three-seed mean validation RMSE selects its configuration. The process ODE is a deterministic fit. The primary comparison therefore comprises 40 screening and 20 confirmation fits. Search budgets differ across model families.

Both physics coefficients are zero in the no-physics latent ODE. An additional matched control uses the selected PhytoODE learning rate, architecture, initial weights, schedule, and checkpoint rule with both coefficients zero. Its auxiliary head is instantiated identically but receives no gradient. The required zero-loss fits are already among the confirmed candidates. Standard deviations describe variability across three training seeds, not biological sampling uncertainty.

The partitions were retained from preliminary analyses in which test results had already been inspected. The present subset, candidate catalog, and stopping rule are fixed before training. Current configurations are selected using training and validation data and frozen before new test scoring; no candidate is added on the basis of the resulting test error.

```

## Results, table and caption

```latex
\subsection{Hypocotyl growth: sparse replicate means under one fixed regime}
\label{subsec:hypocotylresults}

The author-collected experiment evaluates prediction of genotype-level hypocotyl growth from sparse replicate snapshots under one fixed regime. Only the 943 measurements under 12L12D at 23~$^{\circ}$C are retained. The models receive genotype and elapsed time, with no light or temperature inputs. The 547 training, 171 validation, and 225 test measurements form 35 separate replicate-mean targets in each partition. Table~\ref{tab:hypocotyl_replicate} reports the mean of five genotype-curve RMSEs against these split-specific means. These scores assess mean growth and are distinct from errors against individual lengths.

Validation selected a learning rate of 0.01 and $(\lambda_{\mathrm{ODE}},\lambda_K)=(50,0.01)$ for the 1,055-parameter \model{}. All neural candidates used 1,500 epochs and validation-selected checkpoints.

\model{} achieved test RMSE / relative RMSE of $0.3646\pm0.0079$~mm / $7.22\pm0.16\%$, compared with $0.3679\pm0.0070$~mm / $7.28\pm0.14\%$ for the independently selected no-physics latent ODE. The smallest mean test RMSE was attained by Logistic-PINN ($0.3607$~mm / $7.14\%$).

\begin{table}[t]
\centering
\small
\caption{12L12D-only replicate-group holdout: mean genotype-curve RMSE (mm) / relative RMSE (\%), evaluated against split-specific replicate means. Stochastic methods report mean $\pm$ sample SD over three seeds; the logistic ODE is fitted once. Bold denotes the smallest unrounded mean in each split. All models use genotype and elapsed time only. The matched control shares the selected PhytoODE settings with both physics coefficients zero.}
\label{tab:hypocotyl_replicate}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccc}
\toprule
Model & Train & Validation & Test \\
\midrule
PhytoODE & $0.191\pm0.002 / 3.76\pm0.04$ & $0.293\pm0.001 / 5.78\pm0.02$ & $0.365\pm0.008 / 7.22\pm0.16$ \\
Latent ODE (no physics) & $0.190\pm0.007 / 3.75\pm0.15$ & $0.297\pm0.003 / 5.86\pm0.06$ & $0.368\pm0.007 / 7.28\pm0.14$ \\
Logistic-PINN & $0.219\pm0.013 / 4.32\pm0.26$ & $0.304\pm0.002 / 6.00\pm0.04$ & $\mathbf{0.361\pm0.014 / 7.14\pm0.27}$ \\
LSTM-NN & $\mathbf{0.137\pm0.029 / 2.71\pm0.58}$ & $\mathbf{0.288\pm0.014 / 5.68\pm0.28}$ & $0.361\pm0.007 / 7.15\pm0.14$ \\
Random forest & $0.364\pm0.013 / 7.17\pm0.26$ & $0.472\pm0.027 / 9.33\pm0.53$ & $0.452\pm0.014 / 8.95\pm0.28$ \\
Logistic ODE & $0.370 / 7.29$ & $0.408 / 8.07$ & $0.485 / 9.61$ \\
\midrule
Latent ODE (matched) & $0.190\pm0.007 / 3.75\pm0.15$ & $0.297\pm0.003 / 5.86\pm0.06$ & $0.368\pm0.007 / 7.28\pm0.14$ \\
\bottomrule
\end{tabular}}
\end{table}


The matched control with both physics coefficients zero obtained $0.3679\pm0.0070$~mm / $7.28\pm0.14\%$. \model{} had 0.89\% lower mean test RMSE. This comparison changes only the derivative and capacity penalties while retaining the architecture, learning rate, initial weights, schedule, and checkpoint rule. It separates their combined effect from independent learning-rate selection.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.96\linewidth]{figures/hypocotyl_single_condition.pdf}
\caption{Hypocotyl growth predictions for Col-0 and MLB in the 12L12D-only replicate-group holdout. Black points are test replicate means, which are the primary evaluation targets; faint gray points show the contributing individual measurements. Small horizontal offsets separate individual points without changing their recorded times. Colored curves average predictions over three training seeds, except the single-fit logistic ODE. All models receive genotype and elapsed time only. All seven observed times are represented in training and validation through separate replicate groups. The matched no-physics control is reported in Table~\ref{tab:hypocotyl_replicate}; the accompanying full-panel figure includes all five genotypes.}
\label{fig:hypocotyllight}
\end{figure}

This experiment extends the framework to an author-collected organ-growth dataset whose measurements are sparse in time and assembled from unequal numbers of replicate plants. The model shares parameters across five genotype curves and uses the ordinary logistic reference without an environmental input stream. Missing cells are omitted when constructing each split-specific mean, and missing time groups contribute no data residual. The experiment demonstrates prediction of mean trajectories from replicate snapshots; it does not establish recovery of individual plant trajectories, unseen-genotype generalization, or transfer to another environment.

As a secondary check against the individual test lengths, pooled RMSE was $0.7377$~mm for \model{} and $0.7394$~mm for the independently selected no-physics latent ODE. The difference from the primary metric reflects both replicate variation and the different aggregation rules. The original workbook, cell-level provenance, filtered partitions, and analysis code form an inspectable dataset package. Independently documented batches and confirmed plant identities would be needed for stronger biological generalization claims.

Halving the integration step from 3 to 1.5~h with fixed encoder inputs changed \model{} predictions by at most 0.000122~mm across the three seeds.

\FloatBarrier

```

## Hypocotyl conclusion paragraph

```latex
The hypocotyl experiment extends the formulation to mean growth from replicate snapshots under one fixed regime, using genotype and time without environmental inputs. Against 35 held-out replicate-mean targets formed from 225 measurements, its test relative RMSE was $7.22\%$, compared with $7.28\%$ for the independently selected no-physics latent ODE and $7.28\%$ for the matched control. All seven times are represented in each partition. Blank cells are omitted before within-partition averaging, and absent time groups supply no target. The accompanying data package makes extraction and partitioning inspectable. This comparison concerns genotype-level mean trajectories; stronger claims about individual plants and independent cohorts require additional measurements.
```
