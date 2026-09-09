# Light-input PhytoODE: manuscript replacements

주 결과: validation으로 선택한 PhytoODE, 반복 관측 holdout. 36/60시간 평가는 본문에 포함하지 않습니다. 기존 설정에 빛만 추가한 비교 결과는 Results의 별도 문단에 명시했습니다. 데이터는 공개 예정으로 서술합니다.

## Abstract

```latex
\begin{abstract}
Predicting plant growth across genotypes and environments requires flexible dynamics under sparse observation. We present \model{}, a genotype-conditioned latent neural ordinary differential equation model with environmental inputs. An encoder initializes the latent state, a learned continuous flow evolves it, and a decoder predicts phenotype without target-curve measurements as inputs. Logistic derivative and capacity penalties regularize the decoded trajectory. Three temperature-conditioned benchmarks cover wheat with 19 genotypes, UAV maize with 402 genotypes, and Arabidopsis stem length with four observations per plant. Mean test RMSEs were $0.03032$~m, $56.68$ relative UAV-height units, and $2.800$~cm, corresponding to relative RMSEs of $10.24\%$, $18.40\%$, and $14.06\%$. These were lower than the compared benchmark families and matched no-physics controls. We additionally collected 943 hypocotyl-length measurements across five Arabidopsis genotypes at seven times under 12-h light/12-h dark at 23~$^{\circ}$C. Using binary illumination input and an ordinary logistic reference, validation-selected PhytoODE achieved $0.346$~mm RMSE ($6.85\%$ relative RMSE) on held-out replicate means, outperforming all five baseline families. The author-collected dataset and reproducible evaluation workflow will be publicly released with the article. Together, the experiments address field-season transfer, genotype-panel scale, sparse temporal sampling, and mean growth from unequal replicate samples. Performance claims refer to the recorded input and tuning procedures; evaluated genotypes and treatment regimes are represented during training.
\end{abstract}
```

## Methods

```latex
\subsection{Author-collected hypocotyl dataset: sparse replicates under a light--dark cycle}
\label{subsec:hypocotyldata}

To complement the three published datasets, we collected hypocotyl-length measurements from \textit{Arabidopsis thaliana} grown at a constant 23~$^{\circ}$C under 12-h light/12-h dark (12L12D). The five genotypes are Col-0, \textit{hy5}, MLB (\textit{ML1} promoter::phyB-GFP/\textit{phyB-9}), EMS57, and the \textit{phyA phyB} double mutant. Lengths were measured in millimetres over three days at seven elapsed times, 0, 12, 24, 36, 48, 60, and 72~h; the first 0--12-h interval was illuminated. This study uses 943 measurements: 197 for Col-0, 196 for \textit{hy5}, 225 for MLB, 94 for EMS57, and 231 for \textit{phyA phyB}. The number of available replicates per genotype--time combination ranges from 6 to 38. Thus, the dataset combines coarse temporal sampling with unequal replicate counts and incomplete spreadsheet records. The continuous-red-light sheet is excluded from every partition in this analysis. Time denotes elapsed hours because developmental age at the first observation was not confirmed. Irradiance, the light spectrum, independent growth batches, and some mutant alleles remain undocumented in the available record.

We will publicly release the dataset with the article, including the original workbook, individual measurements with source-cell identifiers, confirmed experimental metadata, and the fixed evaluation partitions. Extraction scripts and checks will document the transformation from the workbook to the model targets. This author-collected resource provides a complementary benchmark for genotype-level growth from sparsely sampled replicate plants, with a known illumination schedule rather than a varying temperature input.

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

\subsubsection{Light-conditioned latent dynamics and an ordinary logistic reference}
\label{subsec:lightmodel}

Logistic organ-growth descriptions of hypocotyl elongation motivate a parsimonious reference \citep{alimchandani2026atlas}. We use its ordinary, zero-offset form,
\begin{equation}
 \frac{dH_g}{dt}=r_g H_g\left(1-\frac{H_g}{K_g}\right),
 \qquad
 H_g(t)=\frac{K_g}{1+(K_g/H_g(0)-1)\exp(-r_g t)}.
 \label{eq:hypocotyl_logistic}
\end{equation}
Each genotype has a positive, time-independent rate $r_g$ and a capacity $K_g$. The process baseline fits $(r_g,K_g,H_g(0))$ separately for five genotypes, giving 15 parameters, using residuals against training replicate means and eight deterministic starting points. Initial height is estimated from training data; no held-out target is supplied as an input. The rate is an effective coefficient over the observation window, not a separate estimate of light- and dark-phase growth.

Light and circadian regulation contribute to diurnal hypocotyl growth \citep{nozue2007rhythmic}. To make the experimental phase explicit, \model{} receives the known binary illumination state,
\begin{equation}
 L(t)=
 \begin{cases}
  1,&0\leq (t\bmod 24)<12,\\
  0,&12\leq (t\bmod 24)<24,
 \end{cases}
 \qquad t\ \text{in hours}.
 \label{eq:hypocotyl_light}
\end{equation}
Temperature is constant and is not supplied as an input. The light schedule is prescribed experimental information, including its state at the 72-h endpoint; it contains no phenotype observations. All genotypes share this one schedule, so $L(t)$ is a deterministic function of elapsed time. The experiment evaluates the utility of an explicit light-phase feature within a known regime, not prediction under a new photoperiod or identification of a causal light response.

For \model{}, a four-dimensional genotype embedding and an encoder of width 8 initialize an eight-dimensional latent state. The encoder reads normalized time $\tau=t/72$ and the binary light schedule. The vector field has one hidden layer of width 16 and takes latent state, genotype embedding, normalized time, and instantaneous light state as inputs. A decoder of width 16 returns the scaled length $\hat y_g$, with physical prediction $\hat H_g=a\hat y_g$. An auxiliary genotype-dependent head outputs $r_g\in(0,0.5)$~h$^{-1}$ and a scaled capacity $\kappa_g\in(0,2)$, corresponding to $K_g=a\kappa_g$ in millimetres. Including this head, the light-input model has 1,103 parameters; the corresponding model without the light channel has 1,055.

The latent flow determines predicted length; Eq.~\eqref{eq:hypocotyl_logistic} supplies only a soft residual on the decoded derivative,
\begin{equation}
 \frac{d\hat y_g}{dt}
 =\frac{1}{72}\nabla_{\mathbf z}G_\psi(\mathbf z_g)^{\top}
 \mathbf F_\theta(\mathbf z_g,L(72\tau),\mathbf e_g,\tau).
 \label{eq:hypocotyl_derivative}
\end{equation}
Automatic differentiation computes the decoder gradient and retains the graph through the residual and Runge--Kutta solver. Thus the regularizer can update the latent dynamics, decoder, encoder, genotype embedding, and auxiliary parameters. No finite differences of measurements or replicate means are used. A 3-h integration grid $\mathcal G$ includes every light switch. Each Runge--Kutta interval uses the same illumination state through its final stage; the new light state applies in the next interval. Predicted length is continuous at a switch, while its derivative can change. At switching nodes we use the right-hand derivative in Eq.~\eqref{eq:hypocotyl_derivative}; differentiation of the binary switch itself is unnecessary.

All 23 interior grid nodes, denoted $\mathcal C$, contribute to the physics residual independently of the observation mask:
\begin{align}
 \mathcal L_{\mathrm{ODE}}^{\mathrm{hyp}}
 &=\frac{1}{5|\mathcal C|}\sum_{g=1}^{5}\sum_{t\in\mathcal C}
 \left[\frac{d\hat y_g}{dt}
 -r_g\hat y_g\left(1-\frac{\hat y_g}{\kappa_g}\right)\right]^2,
 \label{eq:hypocotyl_residual}\\
 \mathcal L_K^{\mathrm{hyp}}
 &=\frac{1}{5}\sum_{g=1}^{5}
 \left|\kappa_g-\max_{t\in\mathcal G}\hat y_g(t)\right|.
 \label{eq:hypocotyl_capacity}
\end{align}
The training objective is $\mathcal L_{\mathrm{data}}^{\mathrm{hyp}}+\lambda_{\mathrm{ODE}}\mathcal L_{\mathrm{ODE}}^{\mathrm{hyp}}+\lambda_K\mathcal L_K^{\mathrm{hyp}}$. The latent dynamics can respond to illumination, whereas the logistic reference retains one time-independent $r_g$ and $\kappa_g$ per genotype. There are no separately fitted light- and dark-phase logistic rates. The capacity term imposes consistency with the predicted maximum within the observation window; it does not use an observation of mature hypocotyl length.

\subsubsection{Baselines and validation-based selection}

The primary comparison includes the validation-selected light-input \model{}, a no-physics latent ODE, Logistic-PINN, LSTM-NN, random forest, and ordinary logistic ODE. The five baseline families retain their fitted genotype-and-time models, without the binary light channel. The latent ODE has the same hidden dimensions as \model{}, with both physics coefficients zero and its learning rate selected independently; its input dimensions differ because it omits illumination. Logistic-PINN is a coordinate MLP with a four-dimensional genotype embedding, normalized time, two hidden layers of width 32, and the same ordinary logistic derivative and capacity penalties. Its derivative is obtained directly by automatic differentiation of the coordinate network. It differs architecturally from the LSTM-based Logi-PINN in the temperature benchmarks. LSTM-NN uses time as its recurrent input, recurrent widths 16 and 8, and a four-dimensional genotype embedding concatenated before a dense head of width 16. Random forest uses genotype one-hot encoding and normalized time.

The preceding genotype-and-time experiment screened 16 configurations each for PhytoODE and Logistic-PINN, crossing learning rates $\{0.003,0.01\}$, derivative coefficients $\{0.5,5,50,500\}$, and capacity coefficients $\{0.01,0.1\}$. The no-physics latent ODE and LSTM-NN screened the two learning rates; random forest used 300 trees and minimum leaf sizes $\{1,2,4\}$. The two leading seed-1 candidates per stochastic model were confirmed with seeds 2 and 3, and three-seed mean validation RMSE selected the retained configurations. These baseline fits are reused unchanged with the same partitions and evaluation targets.

The light-input PhytoODE follow-up screens 32 configurations with seed 1. Sixteen reuse the preceding PhytoODE grid; sixteen use stratified samples in log space for learning rate, $\lambda_{\mathrm{ODE}}$, $\lambda_K$, and weight decay. Their ranges are $[10^{-3},10^{-1.7}]$, $[10^{-2},10^3]$, $[10^{-4},10^{-0.5}]$, and $[10^{-6},10^{-3}]$, respectively; two sampled configurations instead use zero weight decay. All neural fits run for 1,500 epochs with Adam, gradient clipping at 1, and cosine learning-rate decay. Checkpoints are selected from epoch 200 onward by the mean of the latest three validation evaluations, spaced 20 epochs apart. The three leading seed-1 light-input candidates are confirmed with seeds 2 and 3. A prespecified comparison additionally adds the light channel while retaining the preceding selected PhytoODE settings, $(\lambda_{\mathrm{ODE}},\lambda_K)=(50,0.01)$ and learning rate 0.01; this configuration is also confirmed with all three seeds. The smallest three-seed mean validation RMSE among confirmed candidates selects the main PhytoODE configuration. The primary follow-up comprises 32 screening and eight confirmation fits.

This fixed-settings comparison distinguishes adding the light channel from further hyperparameter selection, although the input change also changes weight-matrix dimensions and initialization. Neither it nor the retained no-light latent ODE is a matched loss-only ablation of the light-input model. The benchmark compares complete fitted predictors under their recorded input and tuning procedures, with additional search applied only to PhytoODE. Standard deviations describe variability across three training seeds, not biological sampling uncertainty or a significance test.

The partitions were retained from preliminary analyses in which test results had already been inspected. The follow-up candidate catalog and stopping rule were fixed before training. Current configurations were selected using training and validation data and frozen before new test scoring; no candidate was added on the basis of the resulting test error. The planned public release will include these protocols, configurations, validation histories, and saved predictions so that the comparison can be reproduced.

```

## Results, table and figure caption

```latex
\subsection{Author-collected hypocotyl growth under a light--dark cycle}
\label{subsec:hypocotylresults}

On our author-collected hypocotyl dataset, \model{} again achieved the lowest mean test error among the compared baseline families. The experiment adds a distinct setting to the temperature-conditioned benchmarks: growth at constant 23~$^{\circ}$C under a known light--dark cycle, observed through unequal numbers of replicate plants at only seven times. The 943 measurements span five genotypes and are evaluated as 35 split-specific replicate means per partition. This tests prediction of genotype-level mean growth from sparse replicate snapshots.

Validation selected learning rate 0.01, $\lambda_{\mathrm{ODE}}=500$, $\lambda_K=0.1$, and weight decay $10^{-4}$ for the 1,103-parameter light-input \model{}. The model retains the ordinary logistic reference described in Section~\ref{subsec:lightmodel}; illumination drives the latent vector field, while the reference rate and capacity remain time-independent within each genotype.

Table~\ref{tab:hypocotyl_replicate} reports train, validation, and test RMSE / relative RMSE. \model{} achieved $0.3460\pm0.0077$~mm / $6.85\pm0.15\%$, compared with $0.3607\pm0.0136$~mm / $7.14\pm0.27\%$ for Logistic-PINN, the strongest baseline by mean test RMSE, and $0.3679\pm0.0070$~mm / $7.28\pm0.14\%$ for the no-physics latent ODE. These correspond to RMSE reductions of 4.07\% and 5.95\%, respectively. PhytoODE also attained the lowest mean validation RMSE among the six selected models, extending its benchmark-leading performance to measurements collected by the authors.

\begin{table}[t]
\centering
\small
\caption{Author-collected 12L12D hypocotyl dataset: replicate-group holdout. Each cell reports RMSE (mm) / relative RMSE (\%), as mean $\pm$ sample SD over three seeds; logistic ODE is fitted once. Metrics use the 35 split-specific replicate means in each partition. Bold denotes the smallest unrounded mean among these six selected models. PhytoODE uses the known binary light schedule and the configuration selected by validation; the five genotype-and-time baselines retain their preceding fits. Input and search procedures are described in Section~\ref{subsec:lightmodel}.}
\label{tab:hypocotyl_replicate}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccc}
\toprule
Model & Train & Validation & Test \\
\midrule
Logistic ODE & $0.370\,/\,7.29$ & $0.408\,/\,8.07$ & $0.485\,/\,9.61$ \\
Random forest & $0.364\pm0.013\,/\,7.17\pm0.26$ & $0.472\pm0.027\,/\,9.33\pm0.53$ & $0.452\pm0.014\,/\,8.95\pm0.28$ \\
LSTM-NN & $\mathbf{0.137\pm0.029\,/\,2.71\pm0.58}$ & $0.288\pm0.014\,/\,5.68\pm0.28$ & $0.361\pm0.007\,/\,7.15\pm0.14$ \\
Logistic-PINN & $0.219\pm0.013\,/\,4.32\pm0.26$ & $0.304\pm0.002\,/\,6.00\pm0.04$ & $0.361\pm0.014\,/\,7.14\pm0.27$ \\
Latent ODE (no physics) & $0.190\pm0.007\,/\,3.75\pm0.15$ & $0.297\pm0.003\,/\,5.86\pm0.06$ & $0.368\pm0.007\,/\,7.28\pm0.14$ \\
\model{} & $0.190\pm0.008\,/\,3.75\pm0.15$ & $\mathbf{0.278\pm0.001\,/\,5.48\pm0.02}$ & $\mathbf{0.346\pm0.008\,/\,6.85\pm0.15}$ \\
\bottomrule
\end{tabular}}
\end{table}


\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{figures/hypocotyl_light_input.pdf}
\caption{Predicted hypocotyl growth for all five genotypes in the author-collected 12L12D experiment at 23~$^{\circ}$C. Blue curves show the validation-selected PhytoODE with binary illumination input and ordinary logistic regularization; the other methods receive genotype and elapsed time. Curves average predictions across three seeds, except the single-fit logistic ODE. Black points are test replicate means, and faint gray points show contributing individual measurements with small horizontal offsets for visibility. Shaded intervals denote darkness. Each split contains distinct replicate groups at all seven observed times. Model curves between these times do not represent additional measurements.}
\label{fig:hypocotyllight}
\end{figure}

Figure~\ref{fig:hypocotyllight} shows the five genotype-specific length curves. The shared model represents their different elongation magnitudes with a common light-phase input. Thus, the same latent ODE formulation accommodates illumination as well as temperature, while retaining a single logistic reference for each genotype. Sparse observations and unequal replicate counts enter through the split-specific targets and observation mask; blank spreadsheet cells are never replaced by artificial lengths.

A prespecified feature comparison retained the preceding PhytoODE hyperparameters and added only the light channel. Its test score was $0.3293\pm0.0057$~mm / $6.52\pm0.11\%$, compared with $0.3646\pm0.0079$~mm / $7.22\pm0.16\%$ without light input. The fixed-settings light-input variant had lower test error than the subsequently tuned variant, but its mean validation RMSE was higher ($0.2886$ versus $0.2777$~mm). The main table therefore retains the configuration chosen by validation. These additional comparisons support the usefulness of the explicit phase feature under the recorded procedures; the retained no-light baselines and unequal tuning budgets do not isolate a physics-loss effect.

As a secondary evaluation against individual test lengths, pooled RMSE was $0.7328$~mm for \model{} and $0.7394$~mm for the no-physics latent ODE. These values include within-group biological variation and use a different aggregation rule from the primary mean-trajectory metric. The primary result concerns held-out replicate means in one genotype panel and one prescribed photoperiod; independent batches and additional photoperiods would test broader transfer.

Beyond predictive accuracy, this experiment contributes an original dataset for reproducible comparison. The planned public release links each retained measurement to its workbook cell and includes fixed partitions, target-construction code, model configurations, validation histories, and predictions. It will allow subsequent methods to use the same sparse sampling pattern and replicate-level evaluation protocol.

Halving the integration step from 3 to 1.5~h, with the encoder inputs held fixed, changed the selected PhytoODE predictions by at most $2.5\times10^{-6}$~mm across the three seeds.

\FloatBarrier

```

## Hypocotyl conclusion paragraph

```latex
The author-collected hypocotyl dataset extends the formulation to light-conditioned mean growth from sparse replicate snapshots at constant temperature. Validation-selected PhytoODE achieved test RMSE $0.346$~mm and relative RMSE $6.85\%$, lower than all five compared baseline families, including Logistic-PINN ($7.14\%$) and the no-physics latent ODE ($7.28\%$). The latent vector field uses the known illumination state while retaining an ordinary logistic reference, demonstrating an environmental input beyond temperature within the same model family. The 943 measurements, seven observation times, unequal replicate counts, and missing spreadsheet entries provide a complementary setting to the three published datasets. We will release the original data and reproducible evaluation workflow with the article. The demonstrated task is prediction of held-out genotype-level replicate means under an observed photoperiod.
```

## Data and code availability

```latex
The three published datasets are identified in Sections~\ref{subsec:wheatdata}--\ref{subsec:arabdata}. The author-collected hypocotyl dataset and its reproducible evaluation workflow will be publicly released with the article. The release will include the original workbook, cell-resolved lengths, confirmed metadata and documented omissions, frozen train/validation/test partitions, extraction and target-construction scripts, and analysis code. Model configurations, validation histories, checkpoints, and prediction arrays will accompany the data. A permanent dataset identifier, repository link, and data-reuse license will be provided with the final deposit.
```
