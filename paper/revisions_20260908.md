# Copy-paste replacements for the adopted PhytoODE results

Source: paper/main.tex; 2026-09-08. Existing figure filenames are retained; replace the wheat and Arabidopsis image files with the regenerated versions.

## 1. Abstract: replace the two sentences starting with Under the respective held-out protocols

```latex
Under the respective held-out protocols, \model{} achieved lower mean test RMSE than five benchmark families and an architecture-matched latent ODE without physics losses: $0.03032\pm0.00066$~m for wheat, $56.68\pm4.37$ relative UAV-height units for maize, and $2.800\pm0.016$~cm for Arabidopsis. These correspond to relative RMSEs of $10.24\pm0.22\%$, $18.40\pm1.42\%$, and $14.06\pm0.08\%$, respectively. The coefficient study is a follow-up analysis on previously inspected test sets; the reported maize setting retains the original coefficients rather than the later validation-optimal pair.
```

## 2. Observable-space logistic regularization: replace the sentence giving the coefficients

```latex
We denote the coefficient of the ODE derivative residual by $\lambda_{\mathrm{ODE}}\equiv\lambda_{\mathrm{phys}}$. The reported $(\lambda_{\mathrm{ODE}},\lambda_K)$ pairs are $(3.16227766,0.1)$ for wheat, $(0.5,0.5)$ for maize, and $(0.5,0.1)$ for Arabidopsis. Their selection and the maize retention decision are described in Section~\ref{subsec:physicsablationmethods}.
```

## 3. Wheat capacity and hyperparameter selection: replace the paragraph

```latex
Initial capacity tuning used validation RMSE from 2022 to rank configurations. Ten capacity configurations varied latent dimension (4--16), genotype-embedding dimension (2--4), ODE width (8--32) and depth (one or two hidden layers), decoder width (8--32), and encoder width (4--16). Seeds 1 and 2 were used for screening. The Pareto candidates were then evaluated with learning rates $2\times10^{-3}$, $5\times10^{-3}$, and $10^{-2}$ and logistic-residual weights 0, 0.5, 2, and 5. This initial search selected the 1,655-parameter architecture with learning rate $10^{-2}$, $\lambda_{\mathrm{ODE}}=2$, and $\lambda_K=0.1$. The subsequent coefficient study retained the architecture and training schedule and selected $(\lambda_{\mathrm{ODE}},\lambda_K)=(3.16227766,0.1)$ using three-seed mean validation RMSE (Section~\ref{subsec:physicsablationmethods}); this is the configuration reported in Table~\ref{tab:wheat}. A 393-parameter micro model was retained as a reference-size efficiency ablation.
```

## 4. Maize UAV dataset: replace the hyperparameter-selection paragraph

```latex
The initial maize hyperparameter search used 2020 validation RMSE and completed 95 runs covering 67 configurations, including three-seed confirmation and comparisons of 1,500-, 3,000-, and 4,500-epoch budgets. Among the 14 configurations completed for all three seeds, it selected a 3,000-epoch model with latent dimension 8, genotype embedding 8, ODE width 16, decoder width 32, encoder width 32, learning rate $10^{-2}$, weight decay $10^{-4}$, and $(\lambda_{\mathrm{ODE}},\lambda_K)=(0.5,0.5)$. That initial selection and its checkpoint hashes were frozen before evaluating the 2021 test set. The present manuscript retains these coefficients and the corresponding three checkpoints following the later coefficient study; this retention decision is distinguished from that study's validation optimum in Section~\ref{subsec:physicsablationmethods}.
```

## 5. Methods: insert this subsection immediately before Baselines and evaluation

```latex
\subsection{Physics-coefficient study and matched loss ablation}
\label{subsec:physicsablationmethods}

We compared \model{} with an identical latent neural ODE trained with $\lambda_{\mathrm{ODE}}=\lambda_K=0$, denoted Latent ODE (no physics). The latent dynamics, decoder, initial-state encoder, genotype embedding, paired initial weights, training budget, optimizer, and validation-based checkpoint rule were retained. The auxiliary logistic head remained instantiated to preserve paired initialization but received no gradient when both losses were disabled. Optimizer weight decay remained $10^{-4}$. Setting only $\lambda_{\mathrm{ODE}}=0$ while retaining $\lambda_K>0$ defines a separate K-only partial ablation, not the no-physics control.

After an ODE-coefficient search, we jointly varied both coefficients while fixing all other hyperparameters. For each dataset, a $6\times5$ grid and four geometric refinements were screened with seed 1; the three leading both-positive pairs and the best pair on each one-loss boundary were confirmed with seeds 2 and 3. Previously completed trials were reused, and selection used mean validation RMSE over three seeds. The joint study completed 112 new full-length runs and retained the preceding validation-selected pairs: $(3.16227766,0.1)$ for wheat, $(500,0.5)$ for maize, and $(0.5,0.1)$ for Arabidopsis.

For maize, we subsequently chose to retain the original $(0.5,0.5)$ setting after reviewing these results. Its validation RMSE was 47.247, compared with 44.943 for $(500,0.5)$; the corresponding test RMSEs were 56.684 and 60.526. Thus, the reported maize setting is a retained configuration, not the optimum of the later validation search. Test results had already been inspected before the follow-up coefficient studies, so these comparisons do not constitute independent confirmation. The no-physics control was not independently retuned.
```

## 6. Baselines and evaluation: replace the opening clause

```latex
In addition to the no-physics latent ODE control, each dataset was evaluated against the same five baseline families:
```

## 7. Baselines and evaluation: replace the figure-description clause

```latex
They show PhytoODE, the five benchmark families, and observed targets; the no-physics control is reported in the tables,
```

## 8. Wheat Results: replace the opening paragraph

```latex
Using the tuned 1,655-parameter architecture, \model{} achieved the lowest mean validation and test errors in the protocol-matched wheat comparison (Table~\ref{tab:wheat}). Its test RMSE / rRMSE was $0.03032\pm0.00066$~m / $10.24\pm0.22\%$, representing reductions of 46.5\% relative to LSTM-NN, 53.8\% relative to Logi-PINN, and 1.91\% relative to the matched no-physics latent ODE. The no-physics model fitted the training trajectories more closely, but \model{} had lower validation and test errors. This modest difference is consistent with a regularization benefit under the recorded split and training settings.
```

## 9. wheat table: replace the PhytoODE row with these two rows

```latex
Latent ODE (no physics) & $0.02108\pm0.00006\,/\,7.04\pm0.02$ & $0.03834\pm0.00087\,/\,11.50\pm0.26$ & $0.03091\pm0.00023\,/\,10.44\pm0.08$ \\
\model{} (ours) & $0.02231\pm0.00007\,/\,7.45\pm0.02$ & $\mathbf{0.03768\pm0.00077\,/\,11.30\pm0.23}$ & $\mathbf{0.03032\pm0.00066\,/\,10.24\pm0.22}$ \\
```

## 10. maize table: replace the PhytoODE row with these two rows

```latex
Latent ODE (no physics) & $43.25\pm5.76\,/\,13.86\pm1.85$ & $60.75\pm10.03\,/\,23.97\pm3.96$ & $67.78\pm13.62\,/\,22.00\pm4.42$ \\
\model{} (ours) & $41.90\pm14.37\,/\,13.43\pm4.61$ & $47.25\pm2.60\,/\,18.64\pm1.03$ & $\mathbf{56.68\pm4.37\,/\,18.40\pm1.42}$ \\
```

## 11. arabidopsis table: replace the PhytoODE row with these two rows

```latex
Latent ODE (no physics) & $1.188\pm0.028\,/\,5.64\pm0.13$ & $3.278\pm0.047\,/\,16.28\pm0.23$ & $2.993\pm0.066\,/\,15.03\pm0.33$ \\
\model{} (ours) & $1.421\pm0.030\,/\,6.75\pm0.14$ & $\mathbf{3.191\pm0.033\,/\,15.85\pm0.16}$ & $\mathbf{2.800\pm0.016\,/\,14.06\pm0.08}$ \\
```

## 12. Maize Results: replace the opening clause

```latex
With the retained $(\lambda_{\mathrm{ODE}},\lambda_K)=(0.5,0.5)$ configuration,
```

## 13. Maize Results: replace the comparison-scope clause

```latex
among the five benchmark families and the no-physics latent ODE control,
```

## 14. Maize Results: replace the clause before the LSTM-NN result

```latex
and was 42.6\% lower than the best test result among the five benchmark families,
```

## 15. Maize table caption: replace the final sentence

```latex
Only PhytoODE received the additional hyperparameter search; its retained coefficient setting and the matched no-physics control are described in Section~\ref{subsec:physicsablationmethods}.
```

## 16. Maize figure caption: replace the configuration sentence

```latex
The \model{} curves use the retained $(\lambda_{\mathrm{ODE}},\lambda_K)=(0.5,0.5)$ configuration.
```

## 17. Arabidopsis Results: replace the opening paragraph

```latex
\model{} achieved the lowest mean validation and test errors on the Arabidopsis primary-stem dataset (Table~\ref{tab:arabidopsis}). Its test RMSE / rRMSE was $2.800\pm0.016$~cm / $14.06\pm0.08\%$, compared with $2.878\pm0.010$~cm / $14.45\pm0.05\%$ for RF and $2.993\pm0.066$~cm / $15.03\pm0.33\%$ for the no-physics latent ODE. These correspond to mean test RMSE reductions of 2.7\% and 6.46\%, respectively. This ordering was obtained with only four observed dates per plant, supporting the applicability of the regularized continuous-time model under sparse temporal sampling; three initialization seeds alone do not establish statistical significance.
```

## 18. Arabidopsis table: remove bold from the RF validation cell

```latex
$3.217\pm0.008\,/\,15.97\pm0.04$
```

## 19. Arabidopsis Results: replace the paragraph immediately after the table

```latex
RF attained the smallest training error, while \model{} attained the smallest mean validation and test errors. The no-physics latent ODE also fitted the training curves more closely than \model{} but generalized less accurately to the held-out plants. These results are consistent with a benefit from the combined growth regularizers under sparse observation, while the small advantage over RF should be assessed on additional biological cohorts.
```

## 20. Arabidopsis temperature-regime paragraph: replace the two PhytoODE results

```latex
$2.126\pm0.056$~cm under nAT and $3.474\pm0.047$~cm under hAT
```

## 21. Results: insert this subsection immediately before Conclusion

```latex
\subsection{Contribution of the physics regularizers}
\label{subsec:physicsablationresults}

Relative to the matched latent ODE with both physics coefficients set to zero, the adopted PhytoODE configurations reduced mean test RMSE by 1.91\% for wheat, 16.37\% for maize, and 6.46\% for Arabidopsis (Tables~\ref{tab:wheat}--\ref{tab:arabidopsis}). This comparison concerns the combined derivative-residual and carrying-capacity penalties. It does not isolate the contribution of the derivative residual alone. In particular, the maize K-only model achieved a slightly lower mean test RMSE / rRMSE of $55.83\pm6.15$ / $18.12\pm2.00\%$ than the adopted full model, although its validation RMSE was higher (48.143 versus 47.247). Therefore, these results support a benefit of the adopted physics-regularized configuration over the both-zero control under the recorded training settings, but not a universal test-error benefit from adding the derivative residual to K regularization. The no-physics control was not separately retuned, and the reused test sets limit independent confirmation.
```

## 22. Conclusion: replace the test-performance sentence

```latex
It achieved lower mean test RMSE than the five benchmark families and the matched no-physics latent ODE in wheat, maize, and Arabidopsis, with rRMSEs of $10.24\pm0.22\%$, $18.40\pm1.42\%$, and $14.06\pm0.08\%$, respectively.
```

## 23. Conclusion limitations: replace the first limitation

```latex
First, capacity and hyperparameters were tuned on a single validation year for wheat and maize, and the subsequent coefficient study reused previously inspected test sets. The retained maize setting also differs from the later validation optimum. Fixed configurations and independently tuned controls should therefore be evaluated over additional validation--test year rotations and independent cohorts.
```

## 24. Maize figure discussion: replace the opening clause

```latex
Figure~\ref{fig:maizecurves} compares PhytoODE and the five benchmark families
```
