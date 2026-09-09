"""Insert audited hypocotyl results and export self-contained manuscript snippets."""
from pathlib import Path
import json
import re
import zipfile

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'
REPORT=ROOT/'hypocotyl/reports'
ORDER=['phytoode','latent_ode','light_pinn','lstm','rf','light_logistic','logistic']
NAMES=dict(phytoode=r'\model{}',latent_ode='the independently tuned no-physics latent ODE',
    light_pinn='Light-PINN',lstm='LSTM-NN',rf='random forest',light_logistic='light-logistic ODE',logistic='logistic ODE')


def main():
    assert json.loads((REPORT/'results_audit.json').read_text())['status']=='passed'
    frame=pd.read_csv(REPORT/'comparison.csv')
    selection=json.loads((ROOT/'hypocotyl/results/light_growth_20260909/selections.json').read_text())
    def row(p,m,s='test'):
        return frame[frame.protocol.eq(p)&frame.model.eq(m)&frame.split.eq(s)].iloc[0]
    def fmt(r):
        return f"${r.rmse_mean:.3f}\\pm{r.rmse_sd:.3f}$~mm / ${r.relative_error_mean:.2f}\\pm{r.relative_error_sd:.2f}\\%$"
    def improvement(p,other):
        return 100*(1-row(p,'phytoode').rmse_mean/row(p,other).rmse_mean)
    sections=[r'\subsection{Hypocotyl growth: sparse replicate snapshots and light forcing}',
        r'\label{subsec:hypocotylresults}', '',
        r'The additional experiment extends the environmental input from temperature to an explicit lighting schedule, while changing the observation unit from verified plant/plot trajectories to genotype--condition means assembled from replicate snapshots. Table~\ref{tab:hypocotyl_replicate} evaluates held-out replicate measurements; Table~\ref{tab:hypocotyl_time_holdout} evaluates the two times absent from both training and validation.', '']
    for p in ['replicate','time_holdout']:
        r=row(p,'phytoode')
        ranked=frame[frame.protocol.eq(p)&frame.split.eq('test')&frame.model.isin(ORDER)].sort_values('rmse_mean')
        winner=ranked.iloc[0].model
        setting=selection['selected'][p]['phytoode']['config']
        protocol_name='replicate-group holdout' if p=='replicate' else 'unseen-time interpolation'
        sentence=f'For {protocol_name}, validation selected learning rate {setting["lr"]:g} and $(\\lambda_{{\\mathrm{{ODE}}}},\\lambda_K)=({setting["lambda_ode"]:g},{setting["lambda_k"]:g})$. '
        sentence+=r'\model{} achieved test RMSE / relative RMSE of '+fmt(r)+'. '
        if winner=='phytoode':
            sentence+='This was the smallest mean test error among the seven methods under this protocol. '
        else:
            sentence+=f'The smallest mean test error was attained by {NAMES[winner]} ({fmt(row(p,winner))}); the physics-regularized model therefore did not lead this comparison. '
        gain=improvement(p,'latent_ode')
        sentence+=f'Its mean RMSE was {abs(gain):.2f}\\% '+('lower' if gain>=0 else 'higher')+' than that of the independently learning-rate-tuned no-physics latent ODE.'
        sections += [sentence,'',(REPORT/f'table_{p}.tex').read_text(),'']
    sections += [r'\begin{figure}[!htbp]',r'\centering',
        r'\includegraphics[width=0.96\linewidth]{figures/hypocotyl_light_interpolation.pdf}',
        r'\caption{Hypocotyl predictions for the wild type (Col-0) and the defined MLB construct under 12L12D and continuous red light (cR). These panels illustrate the unseen-time protocol: hollow gray points are training replicate means at 0, 12, 24, 48 and 72~h; black points are test means at 36 and 60~h, which were absent from both training and validation. Colored lines are three-seed mean predictions, except for the single-fit process ODEs. Gray shading marks darkness in 12L12D. Curves describe population means, not tracked individual plants. All five genotypes are shown in the accompanying full-panel figures.}',
        r'\label{fig:hypocotyllight}',r'\end{figure}','']
    gains=[improvement(p,'latent_ode_matched') for p in ['replicate','time_holdout']]
    matched=[f"{abs(gain):.2f}\\% "+('lower' if gain>=0 else 'higher') for gain in gains]
    sections += ['In the stricter paired comparison at the selected learning rate and initial weights, the no-physics control achieved '+
        fmt(row('replicate','latent_ode_matched'))+' for replicate holdout and '+fmt(row('time_holdout','latent_ode_matched'))+
        r' for unseen-time interpolation. Relative to these controls, \model{} had '+matched[0]+' mean RMSE in replicate holdout and '+matched[1]+
        ' mean RMSE in time interpolation. These changes concern the combined derivative and capacity penalties. The ablation does not isolate the derivative penalty alone. The benefit therefore depends on the observation protocol, and the small differences between independently tuned neural models require confirmation in additional biological batches.','']
    r1,r2=row('replicate','phytoode'),row('time_holdout','phytoode')
    sections += [f'For comparison with the population-mean scores, pooled individual-measurement RMSE for \\model{{}} was ${r1.individual_rmse_mean:.3f}\\pm{r1.individual_rmse_sd:.3f}$~mm in the replicate holdout and ${r2.individual_rmse_mean:.3f}\\pm{r2.individual_rmse_sd:.3f}$~mm at withheld times. Averaging replicates reduces the biological variation present in individual lengths, so the primary scores must not be interpreted as individual-plant prediction errors.','',
        r'The withheld-time experiment provides an explicit test of interpolation from five observation times, beyond merely reporting performance on a 12-h measurement grid. The original workbook, cell-resolved extraction, verified summaries and frozen partitions also provide a reproducible author-collected dataset package. These are distinct contributions from the field-season and temperature experiments. Nevertheless, both lighting regimes and all five genotypes occur during training, and a single temperature does not identify temperature dependence. The unspecified 12L12D spectrum prevents separation of photoperiod and spectral effects. Twelve-hour sampling cannot resolve within-cycle circadian peaks, and unknown longitudinal identities and batch structure preclude claims about individual dynamics or independent-cohort generalization. The light-switched reference should consequently be interpreted as an effective growth constraint, rather than a recovered molecular clock.', '']
    refinement=pd.read_csv(REPORT/'integration_refinement.csv').max_absolute_difference_mm.max()
    sections += [f'As a numerical check, halving the integration step from 3 to 1.5~h while preserving the original scenario encoder changed the selected \\model{{}} predictions by at most {refinement:.6f}~mm across the six protocol--seed fits. This check evaluates solver discretization sensitivity, not biological uncertainty.','']
    sections += [r'\FloatBarrier','']
    (PAPER/'hypocotyl_results.tex').write_text('\n'.join(sections))
    main_path=PAPER/'main.tex'; text=main_path.read_text()
    marker=r'\subsection{Contribution of the physics regularizers}'
    if r'\input{hypocotyl_results}' not in text:
        text=text.replace(marker,r'\input{hypocotyl_results}'+'\n\n'+marker)
    abstract=r'''Predicting plant growth across genotypes and environments requires flexible dynamics under sparse observation. We present \model{}, a genotype- and environment-conditioned latent neural ordinary differential equation model. An encoder reads a known environmental scenario, a learned latent flow evolves continuously, and a decoder predicts phenotype without target-curve measurements as inputs. Logistic derivative and capacity penalties weakly regularize the decoded trajectory. Three temperature-conditioned benchmarks cover wheat with 19 genotypes, UAV maize with 402 genotypes, and Arabidopsis stem length with four observations per plant. Mean test RMSEs were $0.03032$~m, $56.68$ relative UAV-height units, and $2.800$~cm, respectively, corresponding to relative RMSEs of $10.24\%$, $18.40\%$, and $14.06\%$. These were lower than the compared benchmark families and matched no-physics controls, although the follow-up coefficient study reused previously inspected test sets. A fourth, author-collected dataset comprises 1,818 hypocotyl measurements across five genotypes and two light regimes at 23~$^{\circ}$C. We extend the growth constraint to separate light/dark rates and fit population-mean curves from replicate snapshots. With every 36-h and 60-h measurement withheld from training and validation, \model{} achieved HYP_RESULT. All choices for this new comparison were frozen before test scoring. The experiments address field-season transfer, genotype-panel scale, sparse temporal sampling, and environmental-driver substitution. Evaluated genotypes and treatment regimes are represented during training; the new experiment does not establish unseen-photoperiod prediction or individual-plant trajectory recovery.'''
    abstract=abstract.replace('HYP_RESULT',f"${r2.rmse_mean:.3f}$~mm RMSE (${r2.relative_error_mean:.2f}\\%$ relative RMSE)")
    text=re.sub(r'(?s)(\\begin\{abstract\}\n).*?(\n\\end\{abstract\})',lambda m:m[1]+abstract+m[2],text)
    conclusion=r'Four limitations define the next experiments.'
    addition=(r'The hypocotyl experiment extends the formulation to explicit illumination at fixed temperature and to population curves constructed from replicate snapshots. '
        f'Its test relative RMSEs were ${r1.relative_error_mean:.2f}\\%$ for held-out replicate measurements and ${r2.relative_error_mean:.2f}\\%$ for entirely withheld observation times. '
        r'This supports assessment under a new forcing variable and observation structure, while the accompanying data package makes extraction and partitioning inspectable. The biological conclusions remain limited to the observed genotypes and lighting regimes; unknown plant identities and batch structure and incomplete light metadata require further experimental documentation.'+'\n\n')
    if 'The hypocotyl experiment extends the formulation' not in text:
        text=text.replace(conclusion,addition+conclusion)
    main_path.write_text(text)
    # A single file is convenient for pasting into external manuscript editors.
    flatten=lambda s: re.sub(r'\\input\{([^}]+)\}',lambda m:(PAPER/(m[1]+'.tex')).read_text(),s)
    (PAPER/'main_standalone.tex').write_text(flatten(text))
    snippets=['# Hypocotyl additions and changed manuscript passages (2026-09-09)','',
        'The main tables contain actual held-out results; no all-dataset SOTA claim is inferred.','',
        '## Preamble addition for figure placement','', '```latex',r'\usepackage{placeins}','```','',
        '## Replacement title','', '```latex',re.search(r'\\title\{.*\}',text)[0],'```','',
        '## Replacement abstract','', '```latex',r'\begin{abstract}',abstract,r'\end{abstract}','```','',
        '## New Methods subsections','', '```latex',(PAPER/'hypocotyl_methods.tex').read_text(),'```','',
        '## New Results subsection and tables','', '```latex',(PAPER/'hypocotyl_results.tex').read_text(),'```','',
        '## Additional conclusion paragraph','', '```latex',addition,'```','',
        'The generalized environmental notation and architecture wording are already incorporated in `main.tex` and the self-contained `main_standalone.tex`. Add the two new BibTeX entries from `references.bib`. Public deposition details remain to be supplied.','']
    (PAPER/'revisions_20260909.md').write_text('\n'.join(snippets))
    archive_files={'main.tex':(PAPER/'main_standalone.tex').read_bytes(),
        'references.bib':(PAPER/'references.bib').read_bytes()}
    for figure in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',flatten(text)):
        archive_files[figure]=(PAPER/figure).read_bytes()
    with zipfile.ZipFile(PAPER/'overleaf_20260909.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(archive_files.items()):
            item=zipfile.ZipInfo(name,date_time=(2026,9,9,0,0,0)); item.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(item,data)
    print('Wrote audited results, manuscript update, standalone source and replacement snippets.')


if __name__=='__main__':main()
