"""Report the enlarged validation grid and compare its selection with prior fits."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import report_forecast as baseline
from run_expanded_no_light_search import ROOT, RESULTS, PREVIOUS

REPORT=ROOT/'hypocotyl/reports/expanded_no_light_lambda_20260911'
OLD=ROOT/'hypocotyl/results/prefix_forecast_20260911'
LIGHT=ROOT/'hypocotyl/results/prefix_parameters_20260911'


def summarize(results,output):
    output.mkdir(parents=True,exist_ok=True)
    selection=json.loads((results/'lambda_selection.json').read_text())
    completion=json.loads((results/'completion.json').read_text())
    assert not completion['failures']
    chosen=selection['selected_lambda_ode']
    previous=pd.read_csv(ROOT/'hypocotyl/reports/prefix_parameters_no_light_20260911/comparison.csv')
    labels=dict(baseline.LABELS,phytoode_original='PhytoODE (previous genotype head)',
        phytoode_light='PhytoODE (prefix r/K, light, lambda=100)',
        phytoode_previous='PhytoODE (prefix r/K, no light, previous lambda=100)',
        phytoode=f'PhytoODE (prefix r/K, no light, expanded selection lambda={chosen:g})')
    previous['model']=previous.model.replace({'phytoode':'phytoode_previous'})
    if chosen==100:
        previous=previous[~previous.model.eq('phytoode_previous')]
    previous['label']=previous.model.map(labels)
    added=[];per_seed=[]
    for drop in [0.,.25,.5]:
        runs=[]
        for seed in [1,2,3]:
            folder=results/f'selected/drop_{round(100*drop)}/seed{seed}'
            run=json.loads((folder/'result.json').read_text());cfg=run['config']
            assert cfg['lambda_ode']==chosen and cfg['lambda_k']==0
            prior=json.loads((PREVIOUS/f'evaluated/lambda_100/drop_{round(100*drop)}/seed{seed}/config.json').read_text())
            assert cfg['height_scale_mm']==prior['height_scale_mm']
            assert cfg['data']['removed_training_observation_ids']==prior['data']['removed_training_observation_ids']
            runs.append(run)
            for split in ['train','val','test']:
                per_seed.append(dict(additional_prefix_drop=drop,seed=seed,split=split,
                    **{k:v for k,v in run['metrics'][split].items() if k!='per_plant_rmse'}))
        for split in ['train','val','test']:
            row=dict(protocol='prefix_forecast_20260911',dataset='hypocotyl',model='phytoode',label=labels['phytoode'],
                additional_prefix_drop=drop,split=split,n_seeds=3)
            for metric in ['rmse','relative_error','pooled_rmse','mean_target']:
                values=[r['metrics'][split][metric] for r in runs]
                row[metric+'_mean']=np.mean(values);row[metric+'_sd']=np.std(values,ddof=1)
            for metric in ['n_plants','n_observations']:
                row[metric]=runs[0]['metrics'][split][metric]
            added.append(row)
    frame=pd.concat([previous,pd.DataFrame(added)],ignore_index=True)
    frame.to_csv(output/'comparison.csv',index=False)
    pd.DataFrame(per_seed).to_csv(output/'selected_per_seed_metrics.csv',index=False)
    search=pd.read_csv(results/'validation_search.csv').sort_values('lambda_ode')
    search.to_csv(output/'validation_search.csv',index=False)
    order=['logistic','rf','lstm','logistic_pinn','latent_ode','latent_ode_light','phytoode_original','phytoode_light']
    if chosen!=100:order.append('phytoode_previous')
    order.append('phytoode')
    md=['# Expanded coefficient search for no-light PhytoODE','',
        f'**Selected lambda_ODE={chosen:g}; lambda_K=0.** The r/K head continues to receive genotype, '
        'masked individual 0–36 h lengths and presence masks. Illumination is absent from the encoder and vector field. '
        'Architecture, initialization, optimizer schedule, data and missingness masks are unchanged.','',
        'The coefficient grid was expanded from 0.01–1000 to 0.001–10000, with additional resolution near 0.1 and 100. '
        'The 30 original training/validation-only fits were reused; 12 additional coefficients were trained with seeds 1–3 '
        '(36 new search fits). All 22 candidates were ranked by three-seed mean 48-h validation RMSE.','',
        f"There were {completion['transfer_fits']} new missingness-transfer fits, {completion['new_test_evaluations']} new test evaluations, "
        f"and {completion['reused_test_evaluations']} reused test evaluations. All selected checkpoint hashes were frozen before any new test scoring. "
        'This expanded search follows prior analysis of the same dataset and reported coefficient-100 test results; '
        'candidate ranking uses validation only.','',
        'Cells: mean per-plant RMSE (mm) / relative RMSE (%), mean ± sample SD. '
        'Train: available 0–36 h measurements; validation: 48 h; test: 60/72 h. '
        'Missing targets are not averaged or imputed. Baselines are reused unchanged.','']
    for drop in [0.,.25,.5]:
        part=frame[frame.additional_prefix_drop.eq(drop)]
        md += [f'## Additional removal of available prefix observations: {drop:.0%}','','| Model | Train | Validation | Test |','|---|---:|---:|---:|']
        for model in order:
            cells=[]
            for split in ['train','val','test']:
                sub=part[part.split.eq(split)];r=sub[sub.model.eq(model)].iloc[0]
                cell=f'{r.rmse_mean:.3f} / {r.relative_error_mean:.2f}%'
                if r.n_seeds>1:
                    cell=f'{r.rmse_mean:.3f} ± {r.rmse_sd:.3f} / {r.relative_error_mean:.2f} ± {r.relative_error_sd:.2f}%'
                if r.rmse_mean==sub.rmse_mean.min():cell='**'+cell+'**'
                cells.append(cell)
            md.append('| '+labels[model]+' | '+' | '.join(cells)+' |')
        md.append('')
    md += ['## All validation candidates','','| lambda_ODE | Origin | Validation RMSE (mm) |','|---:|---|---:|']
    for r in search.itertuples():
        cell=f'{r.val_rmse_mean:.4f} ± {r.val_rmse_sd:.4f}'
        if r.lambda_ode==chosen:cell='**'+cell+'**'
        md.append(f'| {r.lambda_ode:g} | {"Reused" if r.origin=="previous" else "Added"} | {cell} |')
    (output/'comparison.md').write_text('\n'.join(md).rstrip()+'\n')
    return frame,search,chosen,order


def plots(results,report,output,search,chosen,order):
    output.mkdir(parents=True,exist_ok=True)
    baseline.ORDER=order
    baseline.LABELS=dict(baseline.LABELS,phytoode=f'PhytoODE (expanded, lambda={chosen:g})',
        phytoode_original='PhytoODE (genotype r/K)',phytoode_light='PhytoODE (prefix r/K, + light)',
        phytoode_previous='PhytoODE (no light, lambda=100)')
    baseline.COLORS=dict(baseline.COLORS,phytoode_original='#586B7A',phytoode_light='#56B4E9',phytoode_previous='#1A4F72')
    baseline.REPORT=report
    def directory(drop,model,seed):
        if model=='phytoode':return results/f'selected/drop_{round(100*drop)}/seed{seed}'
        if model=='phytoode_previous':return PREVIOUS/f'evaluated/lambda_100/drop_{round(100*drop)}/seed{seed}'
        if model=='phytoode_light':return LIGHT/f'selected/drop_{round(100*drop)}/seed{seed}'
        return OLD/f'drop_{round(100*drop)}'/('phytoode' if model=='phytoode_original' else model)/f'seed{seed}'
    def line(model):
        return dict(color=baseline.COLORS[model],lw=2.5 if model=='phytoode' else 1.5,
            ls=':' if model=='phytoode_previous' else '--' if model in ['phytoode_original','phytoode_light','latent_ode_light'] else '-',
            zorder=4 if model=='phytoode' else 2)
    baseline.directory=directory;baseline.model_line=line
    baseline.make_figures(output);baseline.style()
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.plot(search.lambda_ode,search.val_rmse_mean,color='#0072B2',alpha=.5,lw=1)
    for origin,label,color,face in [('previous','Previous 10 candidates','#777777','white'),('current','Added 12 candidates','#0072B2','#0072B2')]:
        part=search[search.origin.eq(origin)]
        ax.errorbar(part.lambda_ode,part.val_rmse_mean,yerr=part.val_rmse_sd,fmt='o',
            capsize=3,color=color,mfc=face,label=label+' (mean ± SD)')
    winner=search[search.lambda_ode.eq(chosen)].iloc[0]
    ax.scatter([chosen],[winner.val_rmse_mean],marker='*',s=170,color='#D55E00',zorder=5,label=f'Selected: {chosen:g}')
    ax.set(xscale='log',xlabel=r'$\lambda_{\mathrm{ODE}}$ ($\lambda_K=0$, no light input)',
        ylabel='48-h validation RMSE (mm)',title='Expanded search: 22 coefficients, three seeds each')
    ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(output/f'expanded_lambda_validation_search.{ext}')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=RESULTS)
    p.add_argument('--output',type=Path,default=REPORT)
    p.add_argument('--figures',type=Path)
    args=p.parse_args();frame,search,chosen,order=summarize(args.results,args.output)
    if args.figures:plots(args.results,args.output,args.figures,search,chosen,order)
    print(frame[frame.model.eq('phytoode')][['additional_prefix_drop','split','rmse_mean','relative_error_mean']].to_string(index=False))


if __name__=='__main__':
    main()
