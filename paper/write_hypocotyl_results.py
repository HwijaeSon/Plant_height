"""Export the primary replicate evaluation to the manuscript; archive other protocols."""
from pathlib import Path
import json
import re
import zipfile

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'
OLD_REPORT=ROOT/'hypocotyl/reports'
NEW_REPORT=OLD_REPORT/'primary_retuning_20260909'
ORDER=['phytoode','latent_ode','light_pinn','lstm','rf','light_logistic','logistic']
NAMES=dict(phytoode=r'\model{}',latent_ode='the independently learning-rate-tuned no-physics latent ODE',
    latent_ode_matched='the matched no-physics latent ODE',
    latent_ode_initial_control='the initial-architecture no-physics control',
    light_pinn='Light-PINN',lstm='LSTM-NN',rf='random forest',light_logistic='light-logistic ODE',logistic='logistic ODE')


def main():
    report=NEW_REPORT
    assert json.loads((report/'results_audit.json').read_text())['status']=='passed'
    frame=pd.read_csv(report/'comparison.csv')
    frame=frame[frame.protocol.eq('replicate')]
    selection=json.loads((ROOT/'hypocotyl/results/primary_retuning_20260909/chosen_phytoode.json').read_text())
    setting=selection['selected']['config']
    selected_runs=json.loads((report/'selected_configs.json').read_text())['runs']
    n_params=next(r['n_params'] for r in selected_runs if r['model']=='phytoode')
    def row(m,s='test'):
        return frame[frame.model.eq(m)&frame.split.eq(s)].iloc[0]
    def fmt(r):
        return f"${r.rmse_mean:.3f}\\pm{r.rmse_sd:.3f}$~mm / ${r.relative_error_mean:.2f}\\pm{r.relative_error_sd:.2f}\\%$"
    def difference(other):
        gain=100*(1-row('phytoode').rmse_mean/row(other).rmse_mean)
        return f'{abs(gain):.2f}\\% '+('lower' if gain>=0 else 'higher')
    full=row('phytoode'); pure=row('latent_ode'); paired=row('latent_ode_matched')
    initial=row('latent_ode_initial_control')
    winner=frame[frame.split.eq('test')].sort_values('rmse_mean').iloc[0].model
    arch=setting['architecture']
    sections=[r'\subsection{Hypocotyl growth: replicate snapshots under light regimes}',
        r'\label{subsec:hypocotylresults}','',
        r'The author-collected experiment extends the environmental input from temperature to an explicit lighting schedule and the observation structure to population curves assembled from replicate snapshots. Table~\ref{tab:hypocotyl_replicate} evaluates 436 held-out measurements as 70 genotype--condition--time means. Training, validation, and test each cover the same seven times and ten genotype--condition combinations. Thus the sparse 12-h sampling is a property of the observations, while generalization is evaluated across the separated replicate groups.','']
    sentence=(f'Validation selected learning rate {setting["lr"]:.5g}, weight decay {setting["weight_decay"]:.5g}, and '
        f'$(\\lambda_{{\\mathrm{{ODE}}}},\\lambda_K)=({setting["lambda_ode"]:.5g},{setting["lambda_k"]:.5g})$. '
        f'The selected architecture has latent dimension {arch["latent_dim"]}, genotype embedding {arch["g_embed_dim"]}, '
        f'vector-field width {arch["ode_hidden"]} and depth {arch["ode_layers"]}, decoder width {arch["dec_hidden"]}, '
        f'and encoder width {arch["enc_hidden"]}, for {n_params:,} parameters including the auxiliary head. '
        f'Training ran for {setting["epochs"]:,} epochs, with the reported checkpoint selected by validation. '
        r'\model{} achieved test RMSE / relative RMSE of '+fmt(full)+'. ')
    if winner=='phytoode':
        sentence+='This was the smallest mean test error across the reported methods and the matched control. '
    else:
        sentence+=f'The smallest mean test error was attained by {NAMES[winner]} ({fmt(row(winner))}). '
    sentence+=r'\model{}'+f' had mean test RMSE {difference("latent_ode")} than that of the independently learning-rate-tuned latent ODE ({fmt(pure)}).'
    sentence+=f' The absolute gap was only {abs(pure.rmse_mean-full.rmse_mean):.6f}~mm, far below the seed standard deviations, so these two methods were effectively tied under this comparison.'
    sections += [sentence,'',(report/'table_replicate.tex').read_text(),'']
    sections += [r'The initial \model{} comparison gave $0.344$~mm RMSE ($5.93\%$ relative RMSE). '
        r'The present configuration was selected in the subsequent validation-only search described in Methods. '
        r'Because that search followed inspection of the initial test results and retained the original baseline fits, '
        r'the updated comparison has unequal tuning budgets and a reused test partition; it is an exploratory follow-up.','']
    sections += ['For completeness, the earlier no-physics control at the initial 1,160-parameter architecture and learning rate 0.01 achieved '+fmt(initial)+'. '+
        r'This is lower than the newly tuned \model{} result. It was not selected as the principal no-physics baseline because its validation mean was worse than that at learning rate 0.003; it is retained as a historical control rather than selected using test error. The expanded search therefore does not establish superiority over all evaluated latent ODE configurations.','']
    sections += [r'\begin{figure}[!htbp]',r'\centering',
        r'\includegraphics[width=0.96\linewidth]{figures/hypocotyl_light_replicates.pdf}',
        r'\caption{Hypocotyl predictions in the replicate-group holdout for the wild type (Col-0) and the defined MLB construct under 12L12D and continuous red light (cR). Black points are means of the held-out replicate measurements at each of the seven observation times. Colored lines are three-seed mean predictions, except for the single-fit process ODEs. Gray shading marks darkness in 12L12D. All seven times are represented in training and validation using their respective replicate groups. Curves describe population means, not tracked individual plants. The no-physics latent ODE uses its original independent learning-rate selection; the matched loss-removal control is reported in Table~\ref{tab:hypocotyl_replicate} and the text. All five genotypes are shown in the accompanying full-panel figure.}',
        r'\label{fig:hypocotyllight}',r'\end{figure}','',
        'The additional no-physics control, matched to the selected architecture, learning rate, weight decay, initial weights, schedule, and checkpoint rule, achieved '+fmt(paired)+'. '+
        r'Relative to this control, \model{} had '+difference('latent_ode_matched')+r' mean test RMSE. Both $\lambda_{\mathrm{ODE}}$ and $\lambda_K$ are zero in the control. This comparison concerns the combined derivative and capacity penalties and does not isolate the derivative penalty alone. The small difference across three seeds does not establish statistical superiority. Seed variation describes optimization variability; independent biological batches are needed to assess reproducibility.','',
        f'Pooled individual-measurement RMSE for \\model{{}} was ${full.individual_rmse_mean:.4f}\\pm{full.individual_rmse_sd:.4f}$~mm. Averaging replicates reduces the biological variation present in individual lengths, so the population-mean scores must not be interpreted as individual-plant prediction errors.','',
        r'This experiment demonstrates how the same latent-dynamics formulation can use a lighting schedule at fixed temperature and learn from sparsely sampled replicate measurements. The original workbook, cell-resolved extraction, verified summaries, and frozen partitions provide an inspectable author-collected dataset package. Both lighting regimes and all five genotypes occur during training, and a single temperature does not identify temperature dependence. The unspecified 12L12D spectrum prevents separation of photoperiod and spectral effects. Twelve-hour sampling cannot resolve within-cycle circadian peaks, and unknown longitudinal identities and batch structure preclude claims about individual dynamics or independent-cohort generalization. The light-switched reference is an effective growth constraint rather than a recovered molecular clock.','']
    refinement=pd.read_csv(report/'integration_refinement.csv')
    if 'protocol' in refinement: refinement=refinement[refinement.protocol.eq('replicate')]
    maximum=refinement.max_absolute_difference_mm.max()
    sections += [f'As a numerical check, halving the integration step from 3 to 1.5~h while preserving the scenario encoder changed the selected \\model{{}} predictions by at most {maximum:.6f}~mm across the three seeds. This evaluates solver discretization sensitivity rather than biological uncertainty.','',r'\FloatBarrier','']
    (PAPER/'hypocotyl_results.tex').write_text('\n'.join(sections))
    path=PAPER/'main.tex'; text=path.read_text()
    text=text.replace('with explicit withholding of complete observation times and a reproducible dataset package',
        'with held-out replicate measurements and a reproducible dataset package')
    text=text.replace('The hypocotyl protocols evaluate held-out replicate measurements and entirely withheld observation times.',
        'The hypocotyl experiment evaluates held-out replicate measurements at the same observed times.')
    abstract=r'''Predicting plant growth across genotypes and environments requires flexible dynamics under sparse observation. We present \model{}, a genotype- and environment-conditioned latent neural ordinary differential equation model. An encoder reads a known environmental scenario, a learned latent flow evolves continuously, and a decoder predicts phenotype without target-curve measurements as inputs. Logistic derivative and capacity penalties weakly regularize the decoded trajectory. Three temperature-conditioned benchmarks cover wheat with 19 genotypes, UAV maize with 402 genotypes, and Arabidopsis stem length with four observations per plant. Mean test RMSEs were $0.03032$~m, $56.68$ relative UAV-height units, and $2.800$~cm, respectively, corresponding to relative RMSEs of $10.24\%$, $18.40\%$, and $14.06\%$. These were lower than the compared benchmark families and matched no-physics controls. A fourth, author-collected dataset comprises 1,818 hypocotyl measurements across five genotypes and two light regimes at 23~$^{\circ}$C. We extend the growth constraint to separate light/dark rates and fit population-mean curves from replicate snapshots at seven times. On held-out replicate measurements, \model{} achieved HYP_RESULT, without establishing superiority over the no-physics latent ODE controls. Follow-up tuning reused previously inspected test partitions; updated results therefore require independent confirmation. The experiments address field-season transfer, genotype-panel scale, sparse temporal sampling, and environmental-driver substitution. Evaluated genotypes and treatment regimes are represented during training, and the new dataset does not establish unseen-photoperiod prediction or individual-plant trajectory recovery.'''
    abstract=abstract.replace('HYP_RESULT',f'${full.rmse_mean:.3f}$~mm RMSE (${full.relative_error_mean:.2f}\\%$ relative RMSE)')
    abstract=abstract.replace('PURE_RESULT',f'${pure.rmse_mean:.3f}$~mm (${pure.relative_error_mean:.2f}\\%$)')
    abstract=abstract.replace('MATCHED_RESULT',f'${paired.rmse_mean:.3f}$~mm (${paired.relative_error_mean:.2f}\\%$)')
    text=re.sub(r'(?s)(\\begin\{abstract\}\n).*?(\n\\end\{abstract\})',lambda m:m[1]+abstract+m[2],text)
    addition=(r'The hypocotyl experiment extends the formulation to explicit illumination at fixed temperature and to population curves constructed from replicate snapshots. '
        f'Its test relative RMSE was ${full.relative_error_mean:.2f}\\%$ for held-out replicate measurements, with all seven observation times represented in each partition. '
        r'This effectively tied the validation-selected no-physics baseline and did not improve on the initial-architecture no-physics control; the current matched comparison showed only a small mean difference. The unequal follow-up tuning budgets and reused test partition further limit interpretation of the regularizers. '
        r'The accompanying data package makes extraction and partitioning inspectable. The biological conclusions remain limited to the observed genotypes and lighting regimes; unknown plant identities and batch structure and incomplete light metadata require further experimental documentation.')
    text=re.sub(r'The hypocotyl experiment extends the formulation[^\n]*',lambda m:addition,text)
    path.write_text(text)
    def flatten(s):
        return re.sub(r'\\input\{([^}]+)\}',lambda m:(PAPER/(m[1]+'.tex')).read_text(),s)
    flat=flatten(text)
    forbidden=['time_holdout','hypocotyl_light_interpolation','unseen-time','withheld-time','withheld observation times','36-h and 60-h']
    assert not any(term in flat for term in forbidden)
    (PAPER/'main_standalone.tex').write_text(flat)
    snippets=['# Primary replicate evaluation and changed manuscript passages (2026-09-09)','',
        'The manuscript uses replicate-group holdout for hypocotyls. The expanded PhytoODE search is an exploratory follow-up on an already inspected test partition; numerical rankings are reported as observed.','',
        '## Replacement abstract','', '```latex',r'\begin{abstract}',abstract,r'\end{abstract}','```','',
        '## Replacement Methods subsections','', '```latex',(PAPER/'hypocotyl_methods.tex').read_text(),'```','',
        '## Replacement Results subsection and table','', '```latex',(PAPER/'hypocotyl_results.tex').read_text(),'```','',
        '## Replacement conclusion paragraph','', '```latex',addition,'```','',
        'The Introduction and task definition in main.tex and main_standalone.tex also use the replicate-group protocol. Public deposition details remain to be supplied.','']
    (PAPER/'revisions_20260909.md').write_text('\n'.join(snippets).rstrip()+'\n')
    files={'main.tex':flat.encode(),'references.bib':(PAPER/'references.bib').read_bytes()}
    for figure in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',flat):
        files[figure]=(PAPER/figure).read_bytes()
    with zipfile.ZipFile(PAPER/'overleaf_20260909.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            item=zipfile.ZipInfo(name,date_time=(2026,9,9,0,0,0)); item.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(item,data)
    print('Wrote primary-protocol manuscript, replacement passages, and Overleaf package.')


if __name__=='__main__':main()
