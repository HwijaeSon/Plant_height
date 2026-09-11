"""Export the manuscript's retained logistic-regularized model and five baselines."""
import argparse
import json
import numpy as np
from pathlib import Path
import pandas as pd
import report_forecast as baseline
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data = baseline.data
style = baseline.style
seeds = baseline.seeds
model_line = baseline.model_line

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'hypocotyl/reports/adopted_lambda100_20260911'
SOURCE=ROOT/'hypocotyl/reports/prefix_parameters_no_light_20260911/comparison.csv'
ORDER=['logistic','rf','lstm','logistic_pinn','latent_ode','phytoode']
LABELS=dict(logistic='Logistic ODE',rf='Random forest',lstm='LSTM-NN',logistic_pinn='Logistic-PINN',
            latent_ode='Latent ODE',phytoode='PhytoODE')


def directory(drop,model,seed):
    if model=='phytoode':
        return ROOT/f'hypocotyl/results/prefix_parameters_no_light_20260911/evaluated/lambda_100/drop_{round(100*drop)}/seed{seed}'
    return ROOT/f'hypocotyl/results/prefix_forecast_20260911/drop_{round(100*drop)}/{model}/seed{seed}'


def export(output=REPORT,figures=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(SOURCE)
    frame=frame[frame.model.isin(ORDER)].copy();frame['label']=frame.model.map(LABELS)
    assert len(frame)==54
    frame.to_csv(output/'comparison.csv',index=False)
    md=['# Retained manuscript configuration: prefix-conditioned PhytoODE, lambda_ODE=100, lambda_K=0','',
        'The individual r/K head receives genotype, masked prefix lengths and masks. '
        'The five baseline results are unchanged. The original ten-coefficient validation search selected 100; '
        'it was retained after inspection of a later expanded search and its test results. '
        'That expanded search selected 10000 on validation. The manuscript configuration is therefore '
        'not the expanded validation optimum, and its retention does not constitute independent test confirmation.','',
        'Cells: mean per-plant RMSE (mm) / relative RMSE (%), mean ± sample SD. '
        'Train: 0–36 h; validation: 48 h; test: 60/72 h. Bold marks the smallest unrounded mean in each column.','']
    for drop in [0.,.25,.5]:
        part=frame[frame.additional_prefix_drop.eq(drop)]
        md += [f'## Additional removal of available prefix measurements: {drop:.0%}','','| Model | Train | Validation | Test |','|---|---:|---:|---:|']
        for model in ORDER:
            cells=[]
            for split in ['train','val','test']:
                sub=part[part.split.eq(split)];r=sub[sub.model.eq(model)].iloc[0]
                cell=f'{r.rmse_mean:.3f} / {r.relative_error_mean:.2f}%'
                if r.n_seeds>1:cell=f'{r.rmse_mean:.3f} ± {r.rmse_sd:.3f} / {r.relative_error_mean:.2f} ± {r.relative_error_sd:.2f}%'
                if r.rmse_mean==sub.rmse_mean.min():cell='**'+cell+'**'
                cells.append(cell)
            md.append('| '+LABELS[model]+' | '+' | '.join(cells)+' |')
        md.append('')
    (output/'comparison.md').write_text('\n'.join(md).rstrip()+'\n')
    if figures is not None:
        make_figures(Path(figures), frame)
    return frame


def make_figures(output, frame):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);style()
    cfg,plants,x,batches,raw=data.load(include_test=True)
    examples=[]
    for genotype in data.GENOTYPES:
        candidates=plants[plants.genotype.eq(genotype)].copy()
        counts=x['prefix_mask'].numpy().sum(1)[candidates.plant_index]
        candidates['prefix_n']=counts
        incomplete=candidates[candidates.prefix_n.lt(4)]
        if len(incomplete):candidates=incomplete[incomplete.prefix_n.eq(incomplete.prefix_n.max())]
        examples.append(int(candidates.sort_values('source_row').iloc[0].plant_index))
    source=dict(selection_rule='For each genotype choose the lowest Excel row among plants with the largest incomplete prefix count (fallback: all plants). No validation/test availability or length used.',
        plants=plants[plants.plant_index.isin(examples)].to_dict('records'))
    source['models'] = {m: LABELS[m] for m in ORDER}
    source['inputs'] = ['genotype', 'elapsed time', 'observed prefix', 'prefix mask']
    (output/'hypocotyl_figure_examples.json').write_text(json.dumps(source,indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(13.2,7.4));axes=axes.ravel()
    predictions={}
    for model in ORDER:
        arrays=[]
        for seed in seeds(0,model):
            run=json.loads((directory(0,model,seed)/'result.json').read_text())
            with np.load(directory(0,model,seed)/'predictions.npz') as values:
                arrays.append(values['prediction']*run['config']['height_scale_mm'])
        predictions[model]=np.mean(arrays,axis=0)
    observations=pd.concat(list(raw.values()))
    for ax,idx in zip(axes,examples):
        plant=plants.iloc[idx];record=observations[observations.plant_index.eq(idx)]
        ax.axvline(36,color='#555555',ls=':',lw=1)
        for model in ORDER:ax.plot(data.HOURS,predictions[model][idx],**model_line(model))
        for split,marker,face in [('train','o','black'),('val','s','white'),('test','^','black')]:
            part=record[record.split.eq(split)]
            ax.scatter(part.elapsed_hours,part.length_mm,s=34,marker=marker,facecolors=face,edgecolors='black',linewidths=1,zorder=6)
        ax.set(title=f'{plant.genotype} · plant row {plant.source_row}',xlabel='Elapsed time (h)',ylabel='Hypocotyl length (mm)',xlim=(-2,74))
        ax.set_xticks([0,12,24,36,48,60,72]);ax.grid(axis='y',alpha=.15)
    axes[-1].axis('off')
    handles=[Line2D([],[],label=LABELS[m],**{k:v for k,v in model_line(m).items() if k!='zorder'}) for m in ORDER[::-1]]
    handles += [Line2D([],[],ls='',marker=m,color='black',markerfacecolor=f,label=l) for m,f,l in [
        ('o','black','Observed input / train (0–36 h)'),('s','white','Validation (48 h)'),('^','black','Test (60, 72 h)')]]
    axes[-1].legend(handles=handles,loc='center',frameon=False,fontsize=10)
    fig.subplots_adjust(left=.065,right=.99,bottom=.09,top=.96,hspace=.35,wspace=.26)
    for ext in ['png','pdf']:fig.savefig(output/f'hypocotyl_prefix_forecast.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.8))
    for ax,metric,ylabel in zip(axes,['rmse','relative_error'],['Test RMSE (mm)','Test relative RMSE (%)']):
        for model in ORDER:
            part=frame[frame.model.eq(model)&frame.split.eq('test')].sort_values('additional_prefix_drop')
            ax.errorbar([12.26,34.24,56.21],part[metric+'_mean'],yerr=part[metric+'_sd'],marker='o',ms=4,capsize=3,label=LABELS[model],**model_line(model))
        ax.set_xticks([12.26,34.24,56.21],['12.3%\nNatural','34.2%\n+25% removal','56.2%\n+50% removal'])
        ax.set(xlabel='Missing fraction among all four prefix slots',ylabel=ylabel);ax.margins(x=.18);ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1],labels[::-1],loc='upper center',bbox_to_anchor=(.5,1.),ncol=3,frameon=False,fontsize=9.5)
    fig.subplots_adjust(left=.07,right=.99,bottom=.2,top=.79,wspace=.23)
    for ext in ['png','pdf']:fig.savefig(output/f'hypocotyl_missingness.{ext}')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=REPORT);p.add_argument('--figures',type=Path)
    args=p.parse_args();frame=export(args.output,args.figures)
    print(f'Exported {len(frame)} retained manuscript rows')


if __name__=='__main__':main()
