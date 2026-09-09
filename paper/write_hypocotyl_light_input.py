"""Adopt validation-selected light-input PhytoODE in the primary manuscript."""
from pathlib import Path
import hashlib
import json
import re
import sys
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'; REPORT=ROOT/'hypocotyl/reports/light_input_20260910'
RESULTS=ROOT/'hypocotyl/results/light_input_20260910'
PREVIOUS=ROOT/'hypocotyl/results/single_condition_20260909'
ORDER=['phytoode_light','latent_ode','logistic_pinn','lstm','rf','logistic']
TABLE_ORDER=['logistic','rf','lstm','logistic_pinn','latent_ode','phytoode_light']
NAMES=dict(phytoode_light=r'\model{}',latent_ode='Latent ODE (no physics)',logistic_pinn='Logistic-PINN',
    lstm='LSTM-NN',rf='Random forest',logistic='Logistic ODE')
COLORS=dict(phytoode_light='#0072B2',latent_ode='#332288',logistic_pinn='#D55E00',lstm='#009E73',rf='#999999',logistic='#E69F00')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())


def main_table(frame):
    lines=[r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Author-collected 12L12D hypocotyl dataset: replicate-group holdout. Each cell reports RMSE (mm) / relative RMSE (\%), as mean $\pm$ sample SD over three seeds; logistic ODE is fitted once. Metrics use the 35 split-specific replicate means in each partition. Bold denotes the smallest unrounded mean among these six selected models. PhytoODE uses the known binary light schedule and the configuration selected by validation; the five genotype-and-time baselines retain their preceding fits. Input and search procedures are described in Section~\ref{subsec:lightmodel}.}',
        r'\label{tab:hypocotyl_replicate}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{lccc}',
        r'\toprule',r'Model & Train & Validation & Test \\',r'\midrule']
    for model in TABLE_ORDER:
        cells=[]
        for split in ['train','val','test']:
            part=frame[frame.split.eq(split)&frame.model.isin(ORDER)]
            r=part[part.model.eq(model)].iloc[0]
            value=f'{r.rmse_mean:.3f}\\,/\\,{r.relative_error_mean:.2f}'
            if r.n_seeds>1: value=f'{r.rmse_mean:.3f}\\pm{r.rmse_sd:.3f}\\,/\\,{r.relative_error_mean:.2f}\\pm{r.relative_error_sd:.2f}'
            if r.rmse_mean==part.rmse_mean.min(): value=r'\mathbf{'+value+'}'
            cells.append('$'+value+'$')
        lines.append(NAMES[model]+' & '+' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}}',r'\end{table}',''])


def make_figure():
    sys.path.insert(0,str(ROOT/'hypocotyl/code'))
    from single_condition_data import load,GENOTYPES,HOURS
    config,_,batches=load('replicate',splits=('test',)); scale=config['height_scale_mm']
    raw,means=batches['test']['observations'],batches['test']['means']
    predictions={}; sources=[]
    for model in ORDER:
        root=RESULTS if model=='phytoode_light' else PREVIOUS
        seeds=[1] if model=='logistic' else [1,2,3]
        values=[]
        for seed in seeds:
            path=root/'replicate/final'/model/f'seed{seed}'/'predictions.npz'
            with np.load(path) as a: values.append(a['prediction'].astype(float))
            sources.append(path)
        predictions[model]=np.stack(values).mean(axis=0)*scale
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,3,figsize=(9.5,6.1)); axes=axes.ravel()
    for i,(ax,g) in enumerate(zip(axes,GENOTYPES)):
        for start in [12,36,60]: ax.axvspan(start,start+12,color='0.90',alpha=.6,lw=0,zorder=0)
        for model in reversed(ORDER):
            ax.plot(HOURS,predictions[model][i],color=COLORS[model],ls='--' if model=='latent_ode' else '-',
                lw=2.1 if model=='phytoode_light' else 1.3,zorder=4 if model=='phytoode_light' else 2)
        points=raw[raw.genotype.eq(g)]
        jitter=points.observation_id.map(lambda s:(int(hashlib.sha256(s.encode()).hexdigest()[:8],16)/(2**32-1)-.5)*1.2)
        ax.scatter(points.elapsed_hours+jitter,points.length,color='0.5',alpha=.25,s=10,linewidths=0,zorder=1)
        group=means[means.genotype.eq(g)]
        ax.scatter(group.elapsed_hours,group.mean_length_mm,color='black',s=20,edgecolors='white',linewidths=.4,zorder=6)
        ax.set(title=g,xlabel='Elapsed time (h)',ylabel='Hypocotyl length (mm)',xlim=(-2,74))
        ax.set_xticks([0,12,24,36,48,60,72]); ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.12)
    handles=[Line2D([0],[0],color=COLORS[m],lw=2,ls='--' if m=='latent_ode' else '-',
        label='PhytoODE (ours)' if m=='phytoode_light' else NAMES[m]) for m in ORDER]
    handles += [Line2D([0],[0],marker='o',color='black',ls='',markersize=5,label='Observed test mean'),
        Line2D([0],[0],marker='o',color='0.65',ls='',markersize=4,label='Individual test length'),
        Patch(facecolor='0.90',alpha=.6,label='Dark interval (12 h)')]
    axes[-1].axis('off');axes[-1].legend(handles=handles,loc='center',frameon=False,fontsize=9.5)
    fig.tight_layout()
    dest=PAPER/'figures/hypocotyl_light_input'
    for ext in ['pdf','png']:fig.savefig(dest.with_suffix('.'+ext),dpi=240,bbox_inches='tight')
    plt.close(fig)
    return sources


def replace_checked(source,old,new):
    if old in source: return source.replace(old,new)
    assert new in source,old
    return source


def main():
    audit=read(REPORT/'results_audit.json'); selection=read(RESULTS/'selections.json')
    assert audit['status']=='passed' and audit['retained_baseline_metrics_exact'] and audit['all_metrics_recomputed']
    assert audit['selection_sha256']==sha(RESULTS/'selections.json')
    for path,digest in read(RESULTS/'protocol.json')['source_sha256'].items(): assert sha(ROOT/path)==digest
    frame=pd.read_csv(REPORT/'comparison.csv',float_precision='round_trip');frame=frame[frame.protocol.eq('replicate')]
    choice=selection['selected']['replicate']['phytoode_light']['config']
    assert choice['lambda_ode']==500 and choice['lambda_k']==.1 and choice['lr']==.01
    def row(m):return frame[frame.model.eq(m)&frame.split.eq('test')].iloc[0]
    def fmt(r):return f'${r.rmse_mean:.4f}\\pm{r.rmse_sd:.4f}$~mm / ${r.relative_error_mean:.2f}\\pm{r.relative_error_sd:.2f}\\%$'
    full,pure,pinn,previous,anchor=[row(m) for m in ['phytoode_light','latent_ode','logistic_pinn','phytoode','phytoode_light_anchor']]
    assert all(full.rmse_mean<row(m).rmse_mean for m in ORDER[1:])
    gain_pinn=100*(1-full.rmse_mean/pinn.rmse_mean);gain_pure=100*(1-full.rmse_mean/pure.rmse_mean)
    plot_sources=make_figure()
    table=main_table(frame)
    lines=[r'\subsection{Author-collected hypocotyl growth under a light--dark cycle}',
        r'\label{subsec:hypocotylresults}','',
        r'On our author-collected hypocotyl dataset, \model{} again achieved the lowest mean test error among the compared baseline families. The experiment adds a distinct setting to the temperature-conditioned benchmarks: growth at constant 23~$^{\circ}$C under a known light--dark cycle, observed through unequal numbers of replicate plants at only seven times. The 943 measurements span five genotypes and are evaluated as 35 split-specific replicate means per partition. This tests prediction of genotype-level mean growth from sparse replicate snapshots.','',
        r'Validation selected learning rate 0.01, $\lambda_{\mathrm{ODE}}=500$, $\lambda_K=0.1$, and weight decay $10^{-4}$ for the 1,103-parameter light-input \model{}. '
        r'The model retains the ordinary logistic reference described in Section~\ref{subsec:lightmodel}; illumination drives the latent vector field, while the reference rate and capacity remain time-independent within each genotype.','',
        r'Table~\ref{tab:hypocotyl_replicate} reports train, validation, and test RMSE / relative RMSE. '
        r'\model{} achieved '+fmt(full)+', compared with '+fmt(pinn)+' for Logistic-PINN, the strongest baseline by mean test RMSE, and '+fmt(pure)+' for the no-physics latent ODE. '
        f'These correspond to RMSE reductions of {gain_pinn:.2f}\\% and {gain_pure:.2f}\\%, respectively. '
        r'PhytoODE also attained the lowest mean validation RMSE among the six selected models, extending its benchmark-leading performance to measurements collected by the authors.','',table,'',
        r'\begin{figure}[!htbp]',r'\centering',r'\includegraphics[width=\linewidth]{figures/hypocotyl_light_input.pdf}',
        r'\caption{Predicted hypocotyl growth for all five genotypes in the author-collected 12L12D experiment at 23~$^{\circ}$C. Blue curves show the validation-selected PhytoODE with binary illumination input and ordinary logistic regularization; the other methods receive genotype and elapsed time. Curves average predictions across three seeds, except the single-fit logistic ODE. Black points are test replicate means, and faint gray points show contributing individual measurements with small horizontal offsets for visibility. Shaded intervals denote darkness. Each split contains distinct replicate groups at all seven observed times. Model curves between these times do not represent additional measurements.}',
        r'\label{fig:hypocotyllight}',r'\end{figure}','',
        r'Figure~\ref{fig:hypocotyllight} shows the five genotype-specific length curves. The shared model represents their different elongation magnitudes with a common light-phase input. Thus, the same latent ODE formulation accommodates illumination as well as temperature, while retaining a single logistic reference for each genotype. Sparse observations and unequal replicate counts enter through the split-specific targets and observation mask; blank spreadsheet cells are never replaced by artificial lengths.','',
        r'A prespecified feature comparison retained the preceding PhytoODE hyperparameters and added only the light channel. Its test score was '+fmt(anchor)+', compared with '+fmt(previous)+' without light input. '
        r'The fixed-settings light-input variant had lower test error than the subsequently tuned variant, but its mean validation RMSE was higher '
        f'(${frame[frame.model.eq("phytoode_light_anchor")&frame.split.eq("val")].iloc[0].rmse_mean:.4f}$ versus ${frame[frame.model.eq("phytoode_light")&frame.split.eq("val")].iloc[0].rmse_mean:.4f}$~mm). '
        r'The main table therefore retains the configuration chosen by validation. These additional comparisons support the usefulness of the explicit phase feature under the recorded procedures; the retained no-light baselines and unequal tuning budgets do not isolate a physics-loss effect.','',
        f'As a secondary evaluation against individual test lengths, pooled RMSE was ${full.individual_rmse_mean:.4f}$~mm for \\model{{}} and ${pure.individual_rmse_mean:.4f}$~mm for the no-physics latent ODE. '
        r'These values include within-group biological variation and use a different aggregation rule from the primary mean-trajectory metric. The primary result concerns held-out replicate means in one genotype panel and one prescribed photoperiod; independent batches and additional photoperiods would test broader transfer.','',
        r'Beyond predictive accuracy, this experiment contributes an original dataset for reproducible comparison. The planned public release links each retained measurement to its workbook cell and includes fixed partitions, target-construction code, model configurations, validation histories, and predictions. It will allow subsequent methods to use the same sparse sampling pattern and replicate-level evaluation protocol.','']
    ref=pd.read_csv(REPORT/'integration_refinement.csv');maximum=ref[ref.protocol.eq('replicate')&ref.model.eq('phytoode_light')].max_absolute_difference_mm.max()
    bound=f'{maximum:.1e}'.split('e')
    lines += [f'Halving the integration step from 3 to 1.5~h, with the encoder inputs held fixed, changed the selected PhytoODE predictions by at most ${bound[0]}\\times10^{{{int(bound[1])}}}$~mm across the three seeds.','',r'\FloatBarrier','']
    results='\n'.join(lines);(PAPER/'hypocotyl_results.tex').write_text(results)
    abstract=r'''Predicting plant growth across genotypes and environments requires flexible dynamics under sparse observation. We present \model{}, a genotype-conditioned latent neural ordinary differential equation model with environmental inputs. An encoder initializes the latent state, a learned continuous flow evolves it, and a decoder predicts phenotype without target-curve measurements as inputs. Logistic derivative and capacity penalties regularize the decoded trajectory. Three temperature-conditioned benchmarks cover wheat with 19 genotypes, UAV maize with 402 genotypes, and Arabidopsis stem length with four observations per plant. Mean test RMSEs were $0.03032$~m, $56.68$ relative UAV-height units, and $2.800$~cm, corresponding to relative RMSEs of $10.24\%$, $18.40\%$, and $14.06\%$. These were lower than the compared benchmark families and matched no-physics controls. We additionally collected 943 hypocotyl-length measurements across five Arabidopsis genotypes at seven times under 12-h light/12-h dark at 23~$^{\circ}$C. Using binary illumination input and an ordinary logistic reference, validation-selected PhytoODE achieved HYP_RESULT on held-out replicate means, outperforming all five baseline families. The author-collected dataset and reproducible evaluation workflow will be publicly released with the article. Together, the experiments address field-season transfer, genotype-panel scale, sparse temporal sampling, and mean growth from unequal replicate samples. Performance claims refer to the recorded input and tuning procedures; evaluated genotypes and treatment regimes are represented during training.'''
    abstract=abstract.replace('HYP_RESULT',f'${full.rmse_mean:.3f}$~mm RMSE (${full.relative_error_mean:.2f}\\%$ relative RMSE)')
    source=(PAPER/'main.tex').read_text()
    source=re.sub(r'(?s)(\\begin\{abstract\}\n).*?(\n\\end\{abstract\})',lambda m:m[1]+abstract+m[2],source)
    replacements=[
        ('an author-collected hypocotyl experiment under one fixed regime, predicting held-out replicate means from genotype and time with a reproducible dataset package',
         'an author-collected hypocotyl dataset, planned for public release, that evaluates light-conditioned prediction of mean growth from sparse, unequally replicated measurements under a fixed photoperiod'),
        ('The environmental input is temperature in the first three datasets; the hypocotyl experiment omits environmental channels and uses genotype and time only.',
         'The environmental input is temperature in the first three datasets and binary illumination state in the author-collected hypocotyl experiment.'),
        ('The hypocotyl experiment uses a 3-h grid without environmental forcing, as detailed in Section~\\ref{subsec:lightmodel}.',
         'The hypocotyl experiment uses a 3-h grid with piecewise-constant illumination, changing state at the 12-h light boundaries, as detailed in Section~\\ref{subsec:lightmodel}.'),
        (r'We compared \model{} with an identical latent neural ODE trained with $\lambda_{\mathrm{ODE}}=\lambda_K=0$, denoted Latent ODE (no physics).',
         r'For the three temperature-conditioned datasets, we compared \model{} with an identical latent neural ODE trained with $\lambda_{\mathrm{ODE}}=\lambda_K=0$, denoted Latent ODE (no physics).')]
    for old,new in replacements:
        if old.startswith('We compared') and new in source: continue
        source=replace_checked(source,old,new)
    conclusion=(r'The author-collected hypocotyl dataset extends the formulation to light-conditioned mean growth from sparse replicate snapshots at constant temperature. '
        f'Validation-selected PhytoODE achieved test RMSE ${full.rmse_mean:.3f}$~mm and relative RMSE ${full.relative_error_mean:.2f}\\%$, '
        f'lower than all five compared baseline families, including Logistic-PINN (${pinn.relative_error_mean:.2f}\\%$) and the no-physics latent ODE (${pure.relative_error_mean:.2f}\\%$). '
        r'The latent vector field uses the known illumination state while retaining an ordinary logistic reference, demonstrating an environmental input beyond temperature within the same model family. The 943 measurements, seven observation times, unequal replicate counts, and missing spreadsheet entries provide a complementary setting to the three published datasets. We will release the original data and reproducible evaluation workflow with the article. The demonstrated task is prediction of held-out genotype-level replicate means under an observed photoperiod.')
    source=re.sub(r'(?:The hypocotyl experiment extends the formulation|The author-collected hypocotyl dataset extends the formulation)[^\n]*',lambda m:conclusion,source)
    availability=(r'The three published datasets are identified in Sections~\ref{subsec:wheatdata}--\ref{subsec:arabdata}. '
        r'The author-collected hypocotyl dataset and its reproducible evaluation workflow will be publicly released with the article. '
        r'The release will include the original workbook, cell-resolved lengths, confirmed metadata and documented omissions, frozen train/validation/test partitions, extraction and target-construction scripts, and analysis code. '
        r'Model configurations, validation histories, checkpoints, and prediction arrays will accompany the data. A permanent dataset identifier, repository link, and data-reuse license will be provided with the final deposit.')
    source=re.sub(r'(\\section\*\{Data and code availability\}\n)[^\n]*',lambda m:m[1]+availability,source)
    (PAPER/'main.tex').write_text(source)
    flat=re.sub(r'\\input\{([^}]+)\}',lambda m:(PAPER/(m[1]+'.tex')).read_text(),source)
    for stale in ['without environmental inputs.','All models use genotype and elapsed time only.',
                  'hypocotyl experiment omits environmental channels','hypocotyl_single_condition.pdf',
                  'time_holdout','36/60','36 / 60','hypocotyl_light_interpolation']:
        assert stale not in flat,stale
    assert 'test results had already been inspected' in flat
    (PAPER/'main_standalone.tex').write_text(flat)
    snippets=['# Light-input PhytoODE: manuscript replacements','',
        '주 결과: validation으로 선택한 PhytoODE, 반복 관측 holdout. 36/60시간 평가는 본문에 포함하지 않습니다. 기존 설정에 빛만 추가한 비교 결과는 Results의 별도 문단에 명시했습니다. 데이터는 공개 예정으로 서술합니다.','']
    for title,value in [('Abstract',r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}'),
        ('Methods',(PAPER/'hypocotyl_methods.tex').read_text()),('Results, table and figure caption',results),
        ('Hypocotyl conclusion paragraph',conclusion),('Data and code availability',availability)]:
        snippets += ['## '+title,'','```latex',value,'```','']
    (PAPER/'revisions_20260910.md').write_text('\n'.join(snippets))
    figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',flat)
    archive=PAPER/'overleaf_20260910.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('main.tex',flat);z.write(PAPER/'references.bib','references.bib')
        for name in sorted(set(figures)):z.write(PAPER/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and z.read('main.tex').decode()==flat
        for name in figures:assert z.read(name)==(PAPER/name).read_bytes()
    sources=[REPORT/'comparison.csv',REPORT/'results_audit.json',RESULTS/'selections.json',PAPER/'hypocotyl_methods.tex',Path(__file__)]+plot_sources
    provenance=dict(status='verified_against_audited_results',protocol='replicate',selected_model='phytoode_light',
        selected_config=choice,main_table_models=ORDER,best_test_model_in_primary_table='phytoode_light',
        fixed_settings_comparison_disclosed=True,light_input_present=True,ordinary_logistic_physics=True,
        data_publication_status='planned',time_holdout_in_main_manuscript=False,
        sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        reported_metrics={m:{s:frame[frame.model.eq(m)&frame.split.eq(s)].iloc[0][['rmse_mean','rmse_sd','relative_error_mean','relative_error_sd']].to_dict() for s in ['train','val','test']} for m in ORDER},
        figure_sha256=sha(PAPER/'figures/hypocotyl_light_input.pdf'),overleaf_sha256=sha(archive))
    (PAPER/'hypocotyl_manuscript_provenance_20260910.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Updated main manuscript, six-model table, five-genotype figure, copyable passages and Overleaf bundle.')


if __name__=='__main__': main()
