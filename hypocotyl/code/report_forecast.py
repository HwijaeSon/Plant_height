"""Summarize all frozen forecast runs and draw individual trajectories and missingness results."""
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
RESULTS=ROOT/'hypocotyl/results/prefix_forecast_20260911'
REPORT=ROOT/'hypocotyl/reports/prefix_forecast_20260911'
ORDER=['logistic','rf','lstm','logistic_pinn','latent_ode','latent_ode_light','phytoode']
LABELS=dict(logistic='Logistic ODE',rf='Random forest',lstm='LSTM-NN',logistic_pinn='Logistic-PINN',
    latent_ode='Latent ODE (no light)',latent_ode_light='Latent ODE (+ light)',phytoode='PhytoODE')
COLORS=dict(logistic='#E69F00',rf='#999999',lstm='#009E73',logistic_pinn='#D55E00',
    latent_ode='#CC79A7',latent_ode_light='#6A3D9A',phytoode='#0072B2')

def directory(drop,model,seed):return RESULTS/f'drop_{round(drop*100)}'/model/f'seed{seed}'
def seeds(drop,model):return [1] if drop==0 and model=='logistic' else [1,2,3]
def runs(drop,model):return [json.loads((directory(drop,model,seed)/'result.json').read_text()) for seed in seeds(drop,model)]

def summarize(output=REPORT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[];paired=[];per_run=[]
    for drop in [0.,.25,.5]:
        for model in ORDER:
            selected=runs(drop,model)
            assert all(r['status']=='complete' for r in selected)
            for split in ['train','val','test']:
                row=dict(protocol=data.PROTOCOL,dataset='hypocotyl',additional_prefix_drop=drop,model=model,
                    label=LABELS[model],split=split,n_seeds=len(selected))
                for metric in ['rmse','relative_error','pooled_rmse','mean_target']:
                    vals=[r['metrics'][split][metric] for r in selected]
                    row[metric+'_mean']=np.mean(vals);row[metric+'_sd']=np.std(vals,ddof=1) if len(vals)>1 else 0.
                for field in ['n_plants','n_observations']:row[field]=selected[0]['metrics'][split][field]
                records.append(row)
            for run in selected:
                for split in ['train','val','test']:
                    per_run.append(dict(additional_prefix_drop=drop,model=model,seed=run['config']['seed'],split=split,
                        **{k:v for k,v in run['metrics'][split].items() if k!='per_plant_rmse'}))
        for seed in [1,2,3]:
            phy=json.loads((directory(drop,'phytoode',seed)/'result.json').read_text())
            lat=json.loads((directory(drop,'latent_ode_light',seed)/'result.json').read_text())
            assert phy['config']['initial_state_sha256']==lat['config']['initial_state_sha256']
            assert phy['config']['data']['removed_training_observation_ids']==lat['config']['data']['removed_training_observation_ids']
            for split in ['val','test']:
                paired.append(dict(additional_prefix_drop=drop,seed=seed,split=split,
                    phytoode=phy['metrics'][split]['rmse'],latent_ode_light=lat['metrics'][split]['rmse'],
                    difference=phy['metrics'][split]['rmse']-lat['metrics'][split]['rmse']))
    frame=pd.DataFrame(records);frame.to_csv(output/'comparison.csv',index=False)
    pd.DataFrame(per_run).to_csv(output/'per_run_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_physics_comparison.csv',index=False)
    markdown=['# Individual-plant prefix forecasting','',
        'Inputs and training targets: observed 0–36 h lengths. Validation: 48 h. Test: 60 and 72 h. No target averaging or imputation.',
        'Cells: mean per-plant RMSE (mm) / relative RMSE (%). ± denotes sample SD over three paired seed/mask realizations; natural-missingness Logistic ODE is deterministic and fitted once.',
        'Training error measures reconstruction of the available input prefix. Bold indicates the smallest unrounded mean in each column.','']
    for drop in [0.,.25,.5]:
        part=frame[frame.additional_prefix_drop.eq(drop)]
        markdown += [f'## Additional removal of observed prefix values: {drop:.0%}','',
            '| Model | Train | Validation | Test |','|---|---:|---:|---:|']
        for model in ORDER:
            cells=[]
            for split in ['train','val','test']:
                sub=part[part.split.eq(split)];r=sub[sub.model.eq(model)].iloc[0]
                cell=f'{r.rmse_mean:.3f} / {r.relative_error_mean:.2f}%'
                if r.n_seeds>1:cell=f'{r.rmse_mean:.3f} ± {r.rmse_sd:.3f} / {r.relative_error_mean:.2f} ± {r.relative_error_sd:.2f}%'
                if r.rmse_mean==sub.rmse_mean.min():cell='**'+cell+'**'
                cells.append(cell)
            markdown.append('| '+LABELS[model]+' | '+' | '.join(cells)+' |')
        markdown.append('')
    (output/'comparison.md').write_text('\n'.join(markdown).rstrip()+'\n')
    return frame

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220,'axes.labelsize':10})

def model_line(model):return dict(color=COLORS[model],lw=2.4 if model=='phytoode' else 1.6,
    ls='--' if model=='latent_ode_light' else '-',zorder=4 if model=='phytoode' else 2)

def make_figures(output):
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
        for left in [12,36,60]:ax.axvspan(left,left+12,color='#dddddd',alpha=.32,lw=0,zorder=0)
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
    frame=pd.read_csv(REPORT/'comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.8))
    for ax,metric,ylabel in zip(axes,['rmse','relative_error'],['Test RMSE (mm)','Test relative RMSE (%)']):
        for model in ORDER:
            part=frame[frame.model.eq(model)&frame.split.eq('test')].sort_values('additional_prefix_drop')
            ax.errorbar([12.26,34.24,56.21],part[metric+'_mean'],yerr=part[metric+'_sd'],marker='o',ms=4,capsize=3,label=LABELS[model],**model_line(model))
        ax.set_xticks([12.26,34.24,56.21],['12.3%\nNatural','34.2%\n+25% removal','56.2%\n+50% removal'])
        ax.set(xlabel='Missing fraction among all four prefix slots',ylabel=ylabel);ax.margins(x=.18);ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1],labels[::-1],loc='upper center',bbox_to_anchor=(.5,1.),ncol=4,frameon=False,fontsize=9.5)
    fig.subplots_adjust(left=.07,right=.99,bottom=.2,top=.79,wspace=.23)
    for ext in ['png','pdf']:fig.savefig(output/f'hypocotyl_missingness.{ext}')
    plt.close(fig)

def main():
    global RESULTS,REPORT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=REPORT)
    p.add_argument('--results',type=Path,default=RESULTS)
    p.add_argument('--figures',type=Path);args=p.parse_args();RESULTS=args.results;REPORT=args.output;summarize(args.output)
    if args.figures:make_figures(args.figures)

if __name__=='__main__':main()
