"""Update manuscript and copyable passages from audited 12L12D-only results."""
from pathlib import Path
import json
import re
import zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'; REPORT=ROOT/'hypocotyl/reports/single_condition_20260909'
RESULTS=ROOT/'hypocotyl/results/single_condition_20260909'
NAMES=dict(phytoode=r'\model{}',latent_ode='the independently selected no-physics latent ODE',
    latent_ode_matched='the matched no-physics control',logistic_pinn='Logistic-PINN',lstm='LSTM-NN',
    rf='random forest',logistic='the logistic ODE')


def main():
    audit=json.loads((REPORT/'results_audit.json').read_text())
    assert audit['status']=='passed' and audit['cR_observations_used']==audit['environmental_feature_count']==0
    assert audit['target_type']=='split_specific_replicate_means_12L12D_only'
    frame=pd.read_csv(REPORT/'comparison.csv'); frame=frame[frame.protocol.eq('replicate')]
    choice=json.loads((RESULTS/'selections.json').read_text())['selected']['replicate']['phytoode']['config']
    def row(model): return frame[frame.model.eq(model)&frame.split.eq('test')].iloc[0]
    def fmt(r): return f'${r.rmse_mean:.4f}\\pm{r.rmse_sd:.4f}$~mm / ${r.relative_error_mean:.2f}\\pm{r.relative_error_sd:.2f}\\%$'
    full,pure,matched=[row(m) for m in ['phytoode','latent_ode','latent_ode_matched']]
    winner=frame[frame.split.eq('test')].sort_values('rmse_mean').iloc[0].model
    lines=[r'\subsection{Hypocotyl growth: sparse replicate means under one fixed regime}',
        r'\label{subsec:hypocotylresults}','',
        r'The author-collected experiment evaluates prediction of genotype-level hypocotyl growth from sparse replicate snapshots under one fixed regime. Only the 943 measurements under 12L12D at 23~$^{\circ}$C are retained. The models receive genotype and elapsed time, with no light or temperature inputs. The 547 training, 171 validation, and 225 test measurements form 35 separate replicate-mean targets in each partition. Table~\ref{tab:hypocotyl_replicate} reports the mean of five genotype-curve RMSEs against these split-specific means. These scores assess mean growth and are distinct from errors against individual lengths.','',
        f'Validation selected a learning rate of {choice["lr"]:.6g} and '
        f'$(\\lambda_{{\\mathrm{{ODE}}}},\\lambda_K)=({choice["lambda_ode"]:.6g},{choice["lambda_k"]:.6g})$ '
        r'for the 1,055-parameter \model{}. All neural candidates used 1,500 epochs and validation-selected checkpoints.','',
        r'\model{} achieved test RMSE / relative RMSE of '+fmt(full)+', compared with '+fmt(pure)+' for the independently selected no-physics latent ODE. '+
        ('It attained the smallest mean test RMSE among the reported models and matched control.' if winner=='phytoode' else
         'The smallest mean test RMSE was attained by '+NAMES[winner]+f' (${row(winner).rmse_mean:.4f}$~mm / ${row(winner).relative_error_mean:.2f}\\%$).'),'',
        (REPORT/'table_replicate.tex').read_text(),'']
    gain=100*(1-full.rmse_mean/matched.rmse_mean)
    lines += ['The matched control with both physics coefficients zero obtained '+fmt(matched)+'. '+
        r'\model{} had '+f'{abs(gain):.2f}\\% '+('lower' if gain>=0 else 'higher')+
        r' mean test RMSE. This comparison changes only the derivative and capacity penalties while retaining the architecture, learning rate, initial weights, schedule, and checkpoint rule. It separates their combined effect from independent learning-rate selection.','',
        r'\begin{figure}[!htbp]',r'\centering',r'\includegraphics[width=0.96\linewidth]{figures/hypocotyl_single_condition.pdf}',
        r'\caption{Hypocotyl growth predictions for Col-0 and MLB in the 12L12D-only replicate-group holdout. Black points are test replicate means, which are the primary evaluation targets; faint gray points show the contributing individual measurements. Small horizontal offsets separate individual points without changing their recorded times. Colored curves average predictions over three training seeds, except the single-fit logistic ODE. All models receive genotype and elapsed time only. All seven observed times are represented in training and validation through separate replicate groups. The matched no-physics control is reported in Table~\ref{tab:hypocotyl_replicate}; the accompanying full-panel figure includes all five genotypes.}',
        r'\label{fig:hypocotyllight}',r'\end{figure}','',
        r'This experiment extends the framework to an author-collected organ-growth dataset whose measurements are sparse in time and assembled from unequal numbers of replicate plants. The model shares parameters across five genotype curves and uses the ordinary logistic reference without an environmental input stream. Missing cells are omitted when constructing each split-specific mean, and missing time groups contribute no data residual. The experiment demonstrates prediction of mean trajectories from replicate snapshots; it does not establish recovery of individual plant trajectories, unseen-genotype generalization, or transfer to another environment.','',
        f'As a secondary check against the individual test lengths, pooled RMSE was ${full.individual_rmse_mean:.4f}$~mm for \\model{{}} and '
        f'${pure.individual_rmse_mean:.4f}$~mm for the independently selected no-physics latent ODE. '
        r'The difference from the primary metric reflects both replicate variation and the different aggregation rules. The original workbook, cell-level provenance, filtered partitions, and analysis code form an inspectable dataset package. Independently documented batches and confirmed plant identities would be needed for stronger biological generalization claims.','']
    refinement=pd.read_csv(REPORT/'integration_refinement.csv')
    maximum=refinement[refinement.protocol.eq('replicate')].max_absolute_difference_mm.max()
    lines += [f'Halving the integration step from 3 to 1.5~h with fixed encoder inputs changed \\model{{}} predictions by at most {maximum:.6f}~mm across the three seeds.','',r'\FloatBarrier','']
    results='\n'.join(lines); (PAPER/'hypocotyl_results.tex').write_text(results)
    abstract=r'''Predicting plant growth across genotypes and environments requires flexible dynamics under sparse observation. We present \model{}, a genotype-conditioned latent neural ordinary differential equation model with optional environmental inputs. An encoder initializes the latent state, a learned continuous flow evolves it, and a decoder predicts phenotype without target-curve measurements as inputs. Logistic derivative and capacity penalties weakly regularize the decoded trajectory. Three temperature-conditioned benchmarks cover wheat with 19 genotypes, UAV maize with 402 genotypes, and Arabidopsis stem length with four observations per plant. Mean test RMSEs were $0.03032$~m, $56.68$ relative UAV-height units, and $2.800$~cm, respectively, corresponding to relative RMSEs of $10.24\%$, $18.40\%$, and $14.06\%$. These were lower than the compared benchmark families and matched no-physics controls. A fourth, author-collected experiment uses 943 hypocotyl measurements across five genotypes under 12-h light/12-h dark at 23~$^{\circ}$C. All models in this experiment receive genotype and elapsed time only, and use split-specific replicate means for training and primary evaluation. With an ordinary logistic reference, \model{} achieved HYP_RESULT on 35 held-out mean targets formed from 225 measurements; the independently selected no-physics latent ODE obtained PURE_RESULT. The experiments address field-season transfer, genotype-panel scale, sparse temporal sampling, and genotype-level mean growth from replicate snapshots. Evaluated genotypes and treatment regimes are represented during training.'''
    abstract=abstract.replace('HYP_RESULT',f'${full.rmse_mean:.3f}$~mm RMSE (${full.relative_error_mean:.2f}\\%$ relative RMSE)').replace('PURE_RESULT',f'${pure.rmse_mean:.3f}$~mm (${pure.relative_error_mean:.2f}\\%$)')
    source=(PAPER/'main.tex').read_text()
    source=re.sub(r'(?s)(\\begin\{abstract\}\n).*?(\n\\end\{abstract\})',lambda m:m[1]+abstract+m[2],source)
    source=source.replace('across genotypes, temperature, and light regimes','across genotypes and observation regimes')
    source=source.replace('an author-collected hypocotyl experiment under two light regimes at fixed temperature, with held-out replicate measurements and a reproducible dataset package',
        'an author-collected hypocotyl experiment under one fixed regime, predicting held-out replicate means from genotype and time with a reproducible dataset package')
    source=source.replace('The environmental input is temperature in the first three datasets and illumination plus duty cycle in the hypocotyl experiment.',
        'The environmental input is temperature in the first three datasets; the hypocotyl experiment omits environmental channels and uses genotype and time only.')
    source=source.replace('The hypocotyl experiment evaluates held-out replicate measurements at the same observed times.',
        'The hypocotyl experiment evaluates means of held-out replicate groups at the same observed times.')
    source=source.replace('Discontinuous illumination uses a 3-h grid and piecewise-constant forcing, as detailed in Section~\\ref{subsec:lightmodel}.',
        'The hypocotyl experiment uses a 3-h grid without environmental forcing, as detailed in Section~\\ref{subsec:lightmodel}.')
    source=source.replace('the additional light-conditioned hypocotyl experiment','the additional hypocotyl experiment under a fixed regime')
    conclusion=(r'The hypocotyl experiment extends the formulation to mean growth from replicate snapshots under one fixed regime, using genotype and time without environmental inputs. '
        f'Against 35 held-out replicate-mean targets formed from 225 measurements, its test relative RMSE was ${full.relative_error_mean:.2f}\\%$, '
        f'compared with ${pure.relative_error_mean:.2f}\\%$ for the independently selected no-physics latent ODE '
        f'and ${matched.relative_error_mean:.2f}\\%$ for the matched control. '
        r'All seven times are represented in each partition. Blank cells are omitted before within-partition averaging, and absent time groups supply no target. The accompanying data package makes extraction and partitioning inspectable. This comparison concerns genotype-level mean trajectories; stronger claims about individual plants and independent cohorts require additional measurements.')
    source=re.sub(r'The hypocotyl experiment extends the formulation[^\n]*',lambda m:conclusion,source)
    (PAPER/'main.tex').write_text(source)
    flat=re.sub(r'\\input\{([^}]+)\}',lambda m:(PAPER/(m[1]+'.tex')).read_text(),source)
    assert 'illumination plus duty cycle' not in flat and 'light-switched logistic constraint' not in flat
    assert 'test results had already been inspected' in flat
    assert 'time_holdout' not in flat and 'hypocotyl_light_interpolation' not in flat
    (PAPER/'main_standalone.tex').write_text(flat)
    snippets=['# 12L12D-only experiment: replacement passages','',
        'cR is excluded. Inputs are genotype and elapsed time only. Training and primary metrics use split-specific replicate means, following the restored initial protocol. The primary manuscript evaluation remains replicate-group holdout; the auxiliary 36/60-h evaluation is retained in the experiment report.','',
        '## Abstract','', '```latex',r'\begin{abstract}',abstract,r'\end{abstract}','```','',
        '## Methods','', '```latex',(PAPER/'hypocotyl_methods.tex').read_text(),'```','',
        '## Results, table and caption','', '```latex',results,'```','',
        '## Hypocotyl conclusion paragraph','', '```latex',conclusion,'```','']
    for name in ['revisions_single_condition_20260909.md','revisions_20260909.md']:
        (PAPER/name).write_text('\n'.join(snippets))
    figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',flat)
    archive=PAPER/'overleaf_20260909.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('main.tex',flat); z.write(PAPER/'references.bib','references.bib')
        for name in sorted(set(figures)): z.write(PAPER/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and z.read('main.tex').decode()==flat
    print('Updated single-condition manuscript, standalone source, copyable passages, and Overleaf bundle.')


if __name__=='__main__': main()
