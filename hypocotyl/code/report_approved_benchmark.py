"""Recompute the four-genotype comparison and its two manuscript figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import forecast_data as data

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'hypocotyl/results/four_genotypes_20260915'
REPORT=ROOT/'hypocotyl/reports/four_genotypes_20260915'
ORDER=['logistic','rf','lstm','logistic_pinn','latent_ode','phytoode']
LABELS=dict(logistic='Logi-ODE',rf='RF',lstm='LSTM-NN',logistic_pinn='Logistic-PINN',latent_ode='Latent ODE',phytoode='PhytoODE')
COLORS=dict(logistic='#E69F00',rf='#999999',lstm='#009E73',logistic_pinn='#D55E00',latent_ode='#CC79A7',phytoode='#0072B2')

def seeds(drop,model):return [1] if model=='logistic' and drop==0 else [1,2,3]
def directory(drop,model,seed,results=RESULTS):
    path=Path(results)/f'drop_{round(100*drop)}'/model/f'seed{seed}'
    return path/'evaluation' if model=='phytoode' else path
def model_line(model):return dict(color=COLORS[model],lw=2.4 if model=='phytoode' else 1.6,zorder=4 if model=='phytoode' else 2)

def summarize(output=REPORT,results=RESULTS):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];per_run=[]
    for drop in [0.,.25,.5]:
        for model in ORDER:
            runs=[json.loads((directory(drop,model,seed,results)/'result.json').read_text()) for seed in seeds(drop,model)]
            assert all(r['status']=='complete' and r['config']['protocol']==data.PROTOCOL for r in runs)
            for split in ['train','val','test']:
                row=dict(protocol=data.PROTOCOL,dataset='hypocotyl',additional_prefix_drop=drop,model=model,label=LABELS[model],split=split,n_seeds=len(runs),
                         prefix_missing_fraction=runs[0]['config']['data']['prefix_missing_fraction'])
                for metric in ['rmse','relative_error','pooled_rmse','mean_target']:
                    vals=[r['metrics'][split][metric] for r in runs]
                    row[metric+'_mean']=float(np.mean(vals));row[metric+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.
                for field in ['n_plants','n_observations']:row[field]=runs[0]['metrics'][split][field]
                rows.append(row)
            for run in runs:
                for split in ['train','val','test']:
                    per_run.append(dict(additional_prefix_drop=drop,model=model,seed=run['config']['seed'],split=split,
                        **{k:v for k,v in run['metrics'][split].items() if k!='per_plant_rmse'}))
    frame=pd.DataFrame(rows);frame.to_csv(output/'comparison.csv',index=False)
    pd.DataFrame(per_run).to_csv(output/'per_run_metrics.csv',index=False)
    md=['# Four-genotype hypocotyl forecasting','',
        'All models were retrained from scratch on Col-0, hy5, MLB, and phyAB using fixed manuscript settings. '
        'Early observations at 0–36 h are inputs/training targets; validation is at 48 h and testing at 60/72 h.','',
        'Cells: mean per-plant RMSE (mm) / relative RMSE (%), with sample SD over the paired training seeds and masks. '
        'The natural-missingness Logi-ODE was fitted once. Bold marks the smallest unrounded mean.','']
    for drop in [0.,.25,.5]:
        part=frame[frame.additional_prefix_drop.eq(drop)]
        md += [f'## Additional removal: {drop:.0%} (total missingness {100*part.prefix_missing_fraction.iloc[0]:.2f}%)','',
               '| Model | Train | Validation | Test |','|---|---:|---:|---:|']
        for model in ORDER:
            cells=[]
            for split in ['train','val','test']:
                group=part[part.split.eq(split)];row=group[group.model.eq(model)].iloc[0]
                value=f'{row.rmse_mean:.3f} / {row.relative_error_mean:.2f}%'
                if row.n_seeds>1:value=f'{row.rmse_mean:.3f} ± {row.rmse_sd:.3f} / {row.relative_error_mean:.2f} ± {row.relative_error_sd:.2f}%'
                cells.append('**'+value+'**' if row.rmse_mean==group.rmse_mean.min() else value)
            md.append('| '+LABELS[model]+' | '+' | '.join(cells)+' |')
        md.append('')
    (output/'comparison.md').write_text('\n'.join(md)+'\n')
    return frame

def make_figures(output,frame,results=RESULTS):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220})
    cfg,plants,x,_,raw=data.load(include_test=True)
    examples=[]
    for genotype in data.GENOTYPES:
        candidates=plants[plants.genotype.eq(genotype)].copy()
        candidates['early_n']=x['prefix_mask'].numpy().sum(1)[candidates.plant_index]
        incomplete=candidates[candidates.early_n.lt(4)]
        if len(incomplete):candidates=incomplete[incomplete.early_n.eq(incomplete.early_n.max())]
        examples.append(int(candidates.sort_values('source_row').iloc[0].plant_index))
    (output/'hypocotyl_figure_examples.json').write_text(json.dumps(dict(
        selection_rule='Lowest source row among plants with the greatest incomplete early-observation coverage; no future availability or lengths used.',
        plants=plants[plants.plant_index.isin(examples)].to_dict('records')),indent=2)+'\n')
    predictions={}
    for model in ORDER:
        arrays=[]
        for seed in seeds(0,model):
            folder=directory(0,model,seed,results);run=json.loads((folder/'result.json').read_text())
            with np.load(folder/'predictions.npz',allow_pickle=False) as values:
                arrays.append(values['prediction']*run['config']['height_scale_mm'])
        predictions[model]=np.mean(arrays,axis=0)
    fig,axes=plt.subplots(2,2,figsize=(11.2,8.2));observations=pd.concat(list(raw.values()))
    titles={'Col-0':'Col-0','hy5':r'$\mathit{hy5}$','MLB':'MLB','phyAB':r'$\mathit{phyA\ phyB}$'}
    for ax,idx in zip(axes.ravel(),examples):
        plant=plants.iloc[idx];record=observations[observations.plant_index.eq(idx)]
        ax.axvline(36,color='#555555',ls=':',lw=1)
        for model in ORDER:ax.plot(data.HOURS,predictions[model][idx],**model_line(model))
        for split,marker,face in [('train','o','black'),('val','s','white'),('test','^','black')]:
            part=record[record.split.eq(split)]
            ax.scatter(part.elapsed_hours,part.length_mm,s=34,marker=marker,facecolors=face,edgecolors='black',linewidths=1,zorder=6)
        ax.set(title=titles[plant.genotype]+f' · plant row {plant.source_row}',xlabel='Elapsed time (h)',ylabel='Hypocotyl length (mm)',xlim=(-2,74))
        ax.set_xticks([0,12,24,36,48,60,72]);ax.grid(axis='y',alpha=.15)
    handles=[Line2D([],[],label=LABELS[m],**{k:v for k,v in model_line(m).items() if k!='zorder'}) for m in ORDER[::-1]]
    handles += [Line2D([],[],ls='',marker=m,color='black',markerfacecolor=f,label=l) for m,f,l in [
        ('o','black','Input / train (0–36 h)'),('s','white','Validation (48 h)'),('^','black','Test (60, 72 h)')]]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=10,bbox_to_anchor=(.5,.005))
    fig.subplots_adjust(left=.08,right=.98,bottom=.21,top=.96,hspace=.35,wspace=.25)
    for ext in ['png','pdf']:fig.savefig(output/f'hypocotyl_prefix_forecast.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.8))
    points=frame[['additional_prefix_drop','prefix_missing_fraction']].drop_duplicates().sort_values('additional_prefix_drop')
    missing=100*points.prefix_missing_fraction.to_numpy()
    ticks=[f'{m:.1f}%\n'+('Natural' if d==0 else f'+{d:.0%} removal') for d,m in zip(points.additional_prefix_drop,missing)]
    for ax,metric,ylabel in zip(axes,['rmse','relative_error'],['Test RMSE (mm)','Test relative RMSE (%)']):
        for model in ORDER:
            part=frame[frame.model.eq(model)&frame.split.eq('test')].sort_values('additional_prefix_drop')
            ax.errorbar(missing,part[metric+'_mean'],yerr=part[metric+'_sd'],marker='o',ms=4,capsize=3,label=LABELS[model],**model_line(model))
        ax.set_xticks(missing,ticks);ax.set(xlabel='Missing fraction of initial measurements',ylabel=ylabel)
        ax.margins(x=.18);ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1],labels[::-1],loc='upper center',bbox_to_anchor=(.5,1),ncol=3,frameon=False,fontsize=9.5)
    fig.subplots_adjust(left=.07,right=.99,bottom=.2,top=.79,wspace=.23)
    for ext in ['png','pdf']:fig.savefig(output/f'hypocotyl_missingness.{ext}')
    plt.close(fig)

def export(output=REPORT,figures=None,results=RESULTS):
    frame=summarize(output,results)
    if figures is not None:make_figures(figures,frame,results)
    return frame

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPORT);parser.add_argument('--results',type=Path,default=RESULTS)
    parser.add_argument('--figures',type=Path)
    args=parser.parse_args();print(f'Exported {len(export(args.output,args.figures,args.results))} comparison rows')
