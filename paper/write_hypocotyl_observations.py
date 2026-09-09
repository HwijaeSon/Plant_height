"""Update the manuscript from audited, individual-observation results only."""
from pathlib import Path
import json
import re
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/'paper'
REPORT = ROOT/'hypocotyl/reports/individual_observations_20260909'
RESULTS = ROOT/'hypocotyl/results/individual_observations_20260909'
NAMES = dict(phytoode=r'\model{}',latent_ode='the independently tuned no-physics latent ODE',
    latent_ode_matched='the matched no-physics control',light_pinn='Light-PINN',lstm='LSTM-NN',
    rf='random forest',light_logistic='the light-logistic ODE',logistic='the logistic ODE')


def main():
    audit = json.loads((REPORT/'results_audit.json').read_text())
    assert audit['status']=='passed' and audit['target_type']=='individual_observed_cells'
    assert audit['replicate_mean_targets']==audit['imputed_phenotype_targets']==0
    frame = pd.read_csv(REPORT/'comparison.csv')
    frame = frame[frame.protocol.eq('replicate')]
    selection = json.loads((RESULTS/'selections.json').read_text())
    setting = selection['selected']['replicate']['phytoode']['config']
    selected = json.loads((REPORT/'selected_configs.json').read_text())
    n_params = next(r['n_params'] for r in selected['runs'] if r['protocol']=='replicate' and r['model']=='phytoode')
    def row(m,s='test'): return frame[frame.model.eq(m)&frame.split.eq(s)].iloc[0]
    def fmt(r):
        return f'${r.rmse_mean:.4f}\\pm{r.rmse_sd:.4f}$~mm / ${r.relative_error_mean:.2f}\\pm{r.relative_error_sd:.2f}\\%$'
    full,pure,matched=[row(m) for m in ['phytoode','latent_ode','latent_ode_matched']]
    winner=frame[frame.split.eq('test')].sort_values('rmse_mean').iloc[0].model
    arch=setting['architecture']
    sections=[r'\subsection{Hypocotyl growth: individual observations under light regimes}',
        r'\label{subsec:hypocotylresults}','',
        r'The author-collected experiment tests environmental-driver substitution and learning from incomplete replicate observations. Temperature was fixed at 23~$^{\circ}$C, while the known illumination schedule supplied the environmental input. All models were trained directly on the 1,052 measured training lengths. Table~\ref{tab:hypocotyl_replicate} reports errors against the 330 validation and 436 test measurements individually, without constructing replicate-mean targets. The same seven observed times and ten genotype--condition combinations occur in each partition.','',
        f'Validation selected a {n_params:,}-parameter \\model{{}} with latent dimension {arch["latent_dim"]}, '
        f'genotype embedding {arch["g_embed_dim"]}, vector-field width {arch["ode_hidden"]} and depth {arch["ode_layers"]}, '
        f'decoder width {arch["dec_hidden"]}, and encoder width {arch["enc_hidden"]}. '
        f'The learning rate was {setting["lr"]:.6g}, weight decay {setting["weight_decay"]:.6g}, '
        f'and $(\\lambda_{{\\mathrm{{ODE}}}},\\lambda_K)=({setting["lambda_ode"]:.6g},{setting["lambda_k"]:.6g})$. '
        f'All neural candidates used {setting["epochs"]:,} epochs and validation-selected checkpoints.','']
    text=r'\model{} achieved test RMSE / relative RMSE of '+fmt(full)+', compared with '+fmt(pure)+' for the independently tuned no-physics latent ODE.'
    if winner=='phytoode':
        text += r' It attained the smallest mean test RMSE among the reported models and the matched control.'
    else:
        text += ' The smallest mean test RMSE was attained by '+NAMES[winner]+f' (${row(winner).rmse_mean:.4f}$~mm / ${row(winner).relative_error_mean:.2f}\\%$).'
    sections += [text,'',(REPORT/'table_replicate.tex').read_text(),'']
    gain=100*(1-full.rmse_mean/matched.rmse_mean)
    sections += ['The control with the same architecture, optimizer settings, initial weights, schedule, and checkpoint rule, but both physics coefficients zero, obtained '+fmt(matched)+'. '+
        r'\model{} had '+f'{abs(gain):.2f}\\% '+('lower' if gain>=0 else 'higher')+
        r' mean test RMSE than this control. This comparison measures the combined contribution of the derivative and capacity penalties. The independently tuned latent ODE has the same number of screening and confirmation fits as \model{}; its configuration is selected separately by validation.','',
        r'\begin{figure}[!htbp]',r'\centering',
        r'\includegraphics[width=0.96\linewidth]{figures/hypocotyl_light_replicates.pdf}',
        r'\caption{Hypocotyl predictions for Col-0 and MLB under 12L12D and continuous red light (cR). Every black point is one held-out measured length; observations are not averaged. Small horizontal offsets separate overlapping points and do not change their recorded times. Colored curves show the mean prediction across three training seeds, except for the single-fit process ODEs. Gray shading indicates darkness in 12L12D. All seven times are represented in training and validation using separate replicate groups. Each genotype--condition curve is fitted to individual measurements, without inferring longitudinal plant identities. The matched loss-removal control is reported in Table~\ref{tab:hypocotyl_replicate}; all five genotypes are shown in the accompanying full-panel figure.}',
        r'\label{fig:hypocotyllight}',r'\end{figure}','',
        r'This experiment demonstrates a shared growth-prediction framework across species, organs, scales, and environmental drivers. Its seven measurement times remain sparse relative to the continuous latent dynamics, and unequal replicate counts and missing cells are handled by evaluating residuals only where measurements exist. Observation-level scoring retains the variation among measured plants. Since individual covariates are unavailable, plants with the same genotype, lighting regime, and observation time receive the same predicted length; the experiment does not establish individualized trajectory recovery.','',
        r'The original workbook, cell-level provenance, extraction checks, and fixed partitions form a reproducible author-collected dataset package. Both light regimes and all five genotypes are represented during training. A single temperature cannot identify temperature dependence, and the unrecorded 12L12D spectrum prevents separation of spectral and photoperiod effects. The phase-specific logistic rates provide a compact reference, while 12-h sampling and unknown plant identities and batch structure limit inference about circadian timing and independent-cohort generalization.','']
    refinement=pd.read_csv(REPORT/'integration_refinement.csv')
    maximum=refinement[refinement.protocol.eq('replicate')].max_absolute_difference_mm.max()
    sections += [f'Halving the integration step from 3 to 1.5~h while preserving the encoder inputs changed \\model{{}} predictions by at most {maximum:.6f}~mm across the three seeds.','',r'\FloatBarrier','']
    result_text='\n'.join(sections)
    (PAPER/'hypocotyl_results.tex').write_text(result_text)
    source=(PAPER/'main.tex').read_text()
    abstract=r'''Predicting plant growth across genotypes and environments requires flexible dynamics under sparse observation. We present \model{}, a genotype- and environment-conditioned latent neural ordinary differential equation model. An encoder reads a known environmental scenario, a learned latent flow evolves continuously, and a decoder predicts phenotype without target-curve measurements as inputs. Logistic derivative and capacity penalties weakly regularize the decoded trajectory. Three temperature-conditioned benchmarks cover wheat with 19 genotypes, UAV maize with 402 genotypes, and Arabidopsis stem length with four observations per plant. Mean test RMSEs were $0.03032$~m, $56.68$ relative UAV-height units, and $2.800$~cm, respectively, corresponding to relative RMSEs of $10.24\%$, $18.40\%$, and $14.06\%$. These were lower than the compared benchmark families and matched no-physics controls. A fourth, author-collected dataset comprises 1,818 hypocotyl measurements across five genotypes and two light regimes at 23~$^{\circ}$C. A light-switched logistic constraint introduces separate effective light and dark growth rates. In this experiment, all models learn directly from individual measured lengths, with no phenotype imputation or replicate averaging. On 436 held-out measurements, \model{} achieved HYP_RESULT; the independently tuned no-physics latent ODE obtained PURE_RESULT. The experiments address field-season transfer, genotype-panel scale, sparse temporal sampling, and environmental-driver substitution. Evaluated genotypes and treatment regimes are represented during training.'''
    abstract=abstract.replace('HYP_RESULT',f'${full.rmse_mean:.3f}$~mm RMSE (${full.relative_error_mean:.2f}\\%$ relative RMSE)')
    abstract=abstract.replace('PURE_RESULT',f'${pure.rmse_mean:.3f}$~mm (${pure.relative_error_mean:.2f}\\%$)')
    source=re.sub(r'(?s)(\\begin\{abstract\}\n).*?(\n\\end\{abstract\})',lambda m:m[1]+abstract+m[2],source)
    conclusion=(r'The hypocotyl experiment extends the formulation to explicit illumination at fixed temperature and direct fitting to incomplete individual observations. '
        f'On 436 held-out measured lengths, its test relative RMSE was ${full.relative_error_mean:.2f}\\%$, '
        f'compared with ${pure.relative_error_mean:.2f}\\%$ for the independently tuned no-physics latent ODE '
        f'and ${matched.relative_error_mean:.2f}\\%$ for the matched control. '
        r'All seven times are represented in each partition, and missing phenotype cells contribute no data residual. The accompanying data package makes extraction and partitioning inspectable. Further biological evaluation requires independently documented cohorts and more complete light metadata.')
    source=re.sub(r'The hypocotyl experiment extends the formulation[^\n]*',lambda m:conclusion,source)
    (PAPER/'main.tex').write_text(source)
    flat=re.sub(r'\\input\{([^}]+)\}',lambda m:(PAPER/(m[1]+'.tex')).read_text(),source)
    assert not any(term in flat for term in ['you2026hypocotylpreprint','effectively tied','hypocotyl_light_interpolation','time_holdout'])
    assert 'test results had already been inspected' in flat
    (PAPER/'main_standalone.tex').write_text(flat)
    snippets=['# Individual-observation experiment: replacement passages','',
        'Measured cells are retained individually; neither means nor imputed phenotypes are targets. The former mean-target comparison paragraphs are replaced by the new raw-observation results. Partition provenance remains stated once in the hypocotyl Methods.','',
        '## Abstract','', '```latex',r'\begin{abstract}',abstract,r'\end{abstract}','```','',
        '## Methods','', '```latex',(PAPER/'hypocotyl_methods.tex').read_text(),'```','',
        '## Results, table and caption','', '```latex',result_text,'```','',
        '## Hypocotyl conclusion paragraph','', '```latex',conclusion,'```','']
    (PAPER/'revisions_individual_observations_20260909.md').write_text('\n'.join(snippets))
    # Keep the established copyable revision entry point current.
    (PAPER/'revisions_20260909.md').write_text('\n'.join(snippets))
    figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',flat)
    archive=PAPER/'overleaf_20260909.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('main.tex',flat)
        z.write(PAPER/'references.bib','references.bib')
        for name in sorted(set(figures)): z.write(PAPER/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and z.read('main.tex').decode()==flat
    print('Updated manuscript, standalone source, copyable passages, and verified Overleaf bundle.')


if __name__=='__main__': main()
