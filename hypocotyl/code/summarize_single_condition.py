"""Audit and report the 12L12D-only benchmark, with mean and individual metrics separated."""
from pathlib import Path
import json
import hashlib
import shutil
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from single_condition_data import load,score,curve_rmse,inputs,GENOTYPES,HOURS,DATA
from single_condition_models import build,physics_losses
from run_single_condition import ROOT,OUT,MODELS,PROTOCOLS,sha,atomic

REPORT=ROOT/'hypocotyl/reports/single_condition_20260909'
ORDER=MODELS+['latent_ode_matched']
LABELS=dict(phytoode='PhytoODE',latent_ode='Latent ODE (no physics)',
    logistic_pinn='Logistic-PINN',lstm='LSTM-NN',rf='Random forest',logistic='Logistic ODE',
    latent_ode_matched='Latent ODE (matched)')
COLORS=dict(phytoode='#0072B2',latent_ode='#332288',logistic_pinn='#D55E00',
            lstm='#009E73',rf='#999999',logistic='#E69F00')


def winners(part):
    return {s:set(part[part.split.eq(s)&part.rmse_mean.eq(part[part.split.eq(s)].rmse_mean.min())].model)
            for s in ['train','val','test']}


def table(frame,protocol):
    part=frame[frame.protocol.eq(protocol)]; best=winners(part)
    caption='12L12D-only replicate-group holdout' if protocol=='replicate' else '12L12D-only auxiliary 36/60-h holdout'
    lines=[r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{'+caption+r': mean genotype-curve RMSE (mm) / relative RMSE (\%), evaluated against split-specific replicate means. Stochastic methods report mean $\pm$ sample SD over three seeds; the logistic ODE is fitted once. Bold denotes the smallest unrounded mean in each split. All models use genotype and elapsed time only. The matched control shares the selected PhytoODE settings with both physics coefficients zero.}',
        r'\label{tab:hypocotyl_'+protocol+r'}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{lccc}',
        r'\toprule',r'Model & Train & Validation & Test \\',r'\midrule']
    for m in ORDER:
        cells=[]
        for s in ['train','val','test']:
            r=part[part.model.eq(m)&part.split.eq(s)].iloc[0]
            value=f'{r.rmse_mean:.3f} / {r.relative_error_mean:.2f}'
            if r.n_seeds>1:
                value=f'{r.rmse_mean:.3f}\\pm{r.rmse_sd:.3f} / {r.relative_error_mean:.2f}\\pm{r.relative_error_sd:.2f}'
            if m in best[s]: value=r'\mathbf{'+value+'}'
            cells.append('$'+value+'$')
        if m=='latent_ode_matched': lines.append(r'\midrule')
        lines.append(LABELS[m]+' & '+' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}}',r'\end{table}',''])


def plot(predictions,batches,scale,destination,protocol,compact=False):
    genotypes=['Col-0','MLB'] if compact else GENOTYPES
    fig,axes=plt.subplots(1,2,figsize=(8.5,4.1)) if compact else plt.subplots(2,3,figsize=(13,7.6))
    axes=np.asarray(axes).ravel(); raw=batches['test']['observations']; means=batches['test']['means']
    for ax,g in zip(axes,genotypes):
        i=GENOTYPES.index(g)
        if protocol=='time_holdout':
            for t in [36,60]: ax.axvline(t,color='0.75',ls=':',lw=.9,zorder=0)
        for m in MODELS:
            ax.plot(HOURS,predictions[m][i]*scale,color=COLORS[m],ls='--' if m=='latent_ode' else '-',
                    lw=2.2 if m=='phytoode' else 1.5,zorder=4 if m=='phytoode' else 2)
        points=raw[raw.genotype.eq(g)]
        jitter=points.observation_id.map(lambda s:(int(hashlib.sha256(s.encode()).hexdigest()[:8],16)/(2**32-1)-.5)*1.2)
        ax.scatter(points.elapsed_hours+jitter,points.length,color='0.4',alpha=.20,s=14,linewidths=0,zorder=1)
        group=means[means.genotype.eq(g)]
        ax.scatter(group.elapsed_hours,group.mean_length_mm,color='black',s=27,zorder=6,linewidths=.5,edgecolors='white')
        ax.set_title(g,fontsize=12); ax.set_xlabel('Elapsed time (h)'); ax.set_ylabel('Hypocotyl length (mm)')
        ax.set_xlim(-2,74); ax.set_ylim(bottom=0); ax.set_xticks([0,12,24,36,48,60,72])
        ax.grid(alpha=.12); ax.spines[['top','right']].set_visible(False)
    handles=[Line2D([0],[0],color=COLORS[m],ls='--' if m=='latent_ode' else '-',lw=2,label=LABELS[m]) for m in MODELS]
    handles += [Line2D([0],[0],marker='o',color='black',ls='',markersize=5,label='Observed test mean'),
                Line2D([0],[0],marker='o',color='0.65',alpha=.5,ls='',markersize=4,label='Individual test measurement')]
    if compact:
        fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,fontsize=8.4)
        fig.tight_layout(rect=(0,0,1,.81))
    else:
        axes[-1].axis('off'); axes[-1].legend(handles=handles,loc='center',frameon=False,fontsize=11)
        title='Replicate holdout' if protocol=='replicate' else '36 / 60 h holdout'
        fig.suptitle(title+' — 12L12D only; genotype and time inputs',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.95))
    for ext in ['png','pdf']: fig.savefig(destination.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    torch.set_num_threads(2); REPORT.mkdir(parents=True,exist_ok=True)
    read=lambda path:json.loads(path.read_text())
    completed,selection,protocol=[read(OUT/name) for name in ['completed.json','selections.json','protocol.json']]
    assert completed['status']=='complete' and completed['selection_sha256']==sha(OUT/'selections.json')
    assert completed['full_training_trials']==120 and completed['final_evaluations']==38
    assert all(sha(ROOT/p)==h for p,h in protocol['source_sha256'].items())
    rows=[]; settings=[]; records=[]; losses=[]; refinements=[]; parameter_rows=[]; curve_rows=[]; target_rows=[]; raw_rows=[]
    for p in PROTOCOLS:
        config,x,batches=load(p,splits=('train','val','test')); scale=config['height_scale_mm']; predictions={}
        for split,batch in batches.items():
            raw=batch['observations']; assert raw.sheet.eq('12L12D').all()
            calculated=raw.groupby(['genotype','elapsed_hours']).length.mean().sort_index()
            stored=batch['means'].set_index(['genotype','elapsed_hours']).mean_length_mm.sort_index()
            np.testing.assert_allclose(calculated,stored,atol=1e-14,rtol=1e-14)
        for m,choice in selection['selected'][p].items():
            if 'ranking' in choice:
                assert choice['selected_candidate']==min(choice['ranking'],key=lambda r:(r['val_mean'],r['candidate']))['candidate']
                for candidate in choice['ranking']:
                    seeds=[1] if m=='logistic' else [1,2,3]
                    values=[read(OUT/p/'trials'/candidate['candidate']/f'seed{s}'/'result.json')['metrics']['val']['rmse'] for s in seeds]
                    np.testing.assert_array_equal(values,candidate['per_seed_validation'])
        for record in [r for r in selection['runs'] if r['protocol']==p]:
            m,s=record['model'],record['seed']; source=ROOT/record['path']; final=OUT/p/'final'/m/f'seed{s}'
            trained,evaluated=read(source/'result.json'),read(final/'result.json')
            assert sha(source/'result.json')==record['result_sha256']==evaluated['source_result_sha256']
            assert evaluated['selection_sha256']==sha(OUT/'selections.json')
            assert trained['test_evaluations']==0 and evaluated['test_evaluations']==1
            original,scored=np.load(source/'predictions.npz'),np.load(final/'predictions.npz')
            assert not any(k.startswith('test_') for k in original.files)
            pred=original['prediction']; np.testing.assert_array_equal(pred,scored['prediction'])
            assert pred.shape==(5,25) and np.isfinite(pred).all()
            predictions.setdefault(m,[]).append(pred)
            for split,batch in batches.items():
                metrics=score(pred,batch,scale)
                for key,value in metrics.items(): np.testing.assert_allclose(value,evaluated['metrics'][split][key],rtol=0,atol=0)
                for key in ['case','time','target']: np.testing.assert_array_equal(batch[key].numpy(),scored[f'{split}_{key}'])
                rows.append(dict(protocol=p,model=m,seed=s,split=split,**{k:v for k,v in metrics.items() if k!='per_curve_rmse'}))
                curve_rows.extend(dict(protocol=p,model=m,seed=s,split=split,genotype=g,rmse=v) for g,v in zip(GENOTYPES,metrics['per_curve_rmse']))
                means=batch['means']
                physical=np.asarray(pred,dtype=float)*scale
                target_rows.extend(dict(protocol=p,model=m,seed=s,split=split,genotype=r.genotype,
                    elapsed_hours=r.elapsed_hours,observed_mean_mm=r.mean_length_mm,n_observations=r.n_observations,
                    predicted_mm=float(physical[GENOTYPES.index(r.genotype),int(r.elapsed_hours/3)])) for r in means.itertuples())
                raw_rows.extend(dict(protocol=p,model=m,seed=s,split=split,observation_id=r.observation_id,
                    observed_mm=r.length,predicted_mm=float(physical[GENOTYPES.index(r.genotype),int(r.elapsed_hours/3)]))
                    for r in batch['observations'].itertuples())
            if trained['checkpoint_sha256']:
                assert sha(source/'checkpoint.pt')==trained['checkpoint_sha256']
                history=pd.read_csv(source/'history.csv')
                # The frozen trainer exported history every 200 epochs, through
                # 1400 of 1500. Never infer or invent the unexported tail.
                checkpoint=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
                assert checkpoint['epoch']==trained['best_epoch']
                assert 200<=trained['best_epoch']<=trained['config']['epochs']
                assert trained['best_epoch']%20==0
                if trained['best_epoch']<=history.epoch.max():
                    best=history[history.epoch.ge(200)].sort_values(['selection_rmse','epoch']).iloc[0]
                    assert int(best.epoch)==trained['best_epoch']
                else:
                    net=build(trained['config']); net.load_state_dict(checkpoint['state_dict']); net.eval()
                    with torch.no_grad(): actual=net(**x)['pred'].numpy()
                    np.testing.assert_allclose(actual,pred,rtol=1e-5,atol=1e-5)
            settings.append(dict(protocol=p,model=m,seed=s,n_params=trained['n_params'],best_epoch=trained['best_epoch'],config=trained['config'],source=record['path']))
            records.append(dict(protocol=p,model=m,seed=s,prediction_sha256=sha(source/'predictions.npz'),final_result_sha256=sha(final/'result.json')))
            if m=='phytoode':
                net=build(trained['config']); ck=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
                net.load_state_dict(ck['state_dict']); net.eval(); out=net(**x); ode,k=physics_losses(net,out,x)
                losses.append(dict(protocol=p,seed=s,data_loss=float(curve_rmse(out['pred'],batches['train']).detach()),
                    ode_loss=float(ode.detach()),k_loss=float(k.detach()),weighted_ode=float(ode.detach())*trained['config']['lambda_ode'],weighted_k=float(k.detach())*trained['config']['lambda_k']))
                with torch.no_grad():
                    coarse=out['pred'].detach().numpy()
                    refined=net(**inputs(np.arange(0.,72.1,1.5)),encoder_time=x['time'])['pred'].numpy()[:,::2]
                np.testing.assert_allclose(coarse,pred,atol=1e-5,rtol=1e-5)
                delta=(refined-coarse)*scale
                refinements.append(dict(protocol=p,seed=s,max_absolute_difference_mm=float(np.abs(delta).max()),rms_difference_mm=float(np.sqrt(np.mean(delta**2)))))
                parameter_rows.extend(dict(protocol=p,model=m,seed=s,genotype=g,r_per_hour=float(out['r'][i].detach()),K_mm=float(out['K'][i].detach())*scale)
                                      for i,g in enumerate(GENOTYPES))
            if m=='logistic':
                parameter_rows.extend(dict(protocol=p,model=m,seed=s,genotype=r['genotype'],r_per_hour=r['r_per_hour'],
                    K_mm=r['K_normalized']*scale,H0_mm=r['initial_fraction']*r['K_normalized']*scale)
                    for r in read(source/'parameters.json')['parameters'])
        for seed in [1,2,3]:
            full=next(r for r in settings if r['protocol']==p and r['model']=='phytoode' and r['seed']==seed)
            pure=next(r for r in settings if r['protocol']==p and r['model']=='latent_ode_matched' and r['seed']==seed)
            assert pure['config']==(full['config']|dict(model='latent_ode',lambda_ode=0.,lambda_k=0.))
            assert read(ROOT/full['source']/'result.json')['initial_state_sha256']==read(ROOT/pure['source']/'result.json')['initial_state_sha256']
        means={m:np.mean(preds,axis=0) for m,preds in predictions.items()}
        plot(means,batches,scale,REPORT/f'predictions_{p}',p)
        if p=='replicate':
            compact=REPORT/'predictions_replicate_compact'; plot(means,batches,scale,compact,p,compact=True)
            if (ROOT/'paper/figures').is_dir():
                for ext in ['png','pdf']: shutil.copyfile(compact.with_suffix('.'+ext),ROOT/f'paper/figures/hypocotyl_single_condition.{ext}')
    frame=pd.DataFrame(rows); frame.to_csv(REPORT/'per_seed_metrics.csv',index=False)
    aggregates=[]
    for (p,m,s),part in frame.groupby(['protocol','model','split'],sort=False):
        entry=dict(protocol=p,model=m,split=s,n_seeds=len(part),n_observations=int(part.n_observations.iloc[0]),n_targets=int(part.n_scored_group_means.iloc[0]))
        for metric in ['rmse','relative_error','individual_rmse']:
            entry[metric+'_mean']=float(part[metric].mean()); entry[metric+'_sd']=float(part[metric].std(ddof=1)) if len(part)>1 else 0.
        aggregates.append(entry)
    comparison=pd.DataFrame(aggregates); comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(REPORT/'per_curve_metrics.csv',index=False)
    pd.DataFrame(target_rows).to_csv(REPORT/'mean_target_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(raw_rows).to_csv(REPORT/'individual_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(losses).to_csv(REPORT/'selected_loss_components.csv',index=False)
    pd.DataFrame(refinements).to_csv(REPORT/'integration_refinement.csv',index=False)
    pd.DataFrame(parameter_rows).to_csv(REPORT/'logistic_parameters.csv',index=False)
    atomic(REPORT/'selected_configs.json',dict(selected=selection['selected'],runs=settings))
    text=['# 12L12D only: genotype/time models with an ordinary logistic reference','',
        'cR is excluded. All predictors receive genotype and elapsed time only; no illumination, duty-cycle, accumulated-exposure, spectral, or temperature features are supplied. PhytoODE and Logistic-PINN use a constant-rate logistic derivative residual. The process baseline is the ordinary logistic ODE. All fits are new.','',
        'As requested when restoring the initial protocol, training and primary scoring use split-specific replicate means. Missing time groups have no target; no missing phenotype is imputed. Primary RMSE is the mean of five genotype-curve RMSEs; relative RMSE divides it by the mean of the scored group-mean targets. Individual-measurement RMSE is retained separately and is not interchangeable with the primary score.','',
        'The original source-row partitions were filtered to 12L12D, without reassignment. Earlier results on these partitions had been inspected. All current selections used validation only and froze before current test scoring. The original search budget gives PhytoODE and Logistic-PINN 16 candidates each, latent ODE and LSTM 2 each, RF 3, and logistic ODE 1 per protocol. The two leading stochastic candidates are confirmed with seeds 2 and 3. A matched no-physics control separates loss effects from independent learning-rate selection.','']
    for p in PROTOCOLS:
        config=json.loads((DATA/p/'config.json').read_text()); part=comparison[comparison.protocol.eq(p)]; best=winners(part)
        (REPORT/f'table_{p}.tex').write_text(table(comparison,p))
        text += [f'## {p}','',f"Raw measurements train / validation / test: {config['counts']['train']} / {config['counts']['val']} / {config['counts']['test']}. Mean targets: {config['mean_target_counts']['train']} / {config['mean_target_counts']['val']} / {config['mean_target_counts']['test']}.",'',
            '| Model | Train RMSE / rRMSE | Validation RMSE / rRMSE | Test RMSE / rRMSE |','|---|---|---|---|']
        for m in ORDER:
            cells=[]
            for s in ['train','val','test']:
                r=part[part.model.eq(m)&part.split.eq(s)].iloc[0]
                value=f'{r.rmse_mean:.4f} ± {r.rmse_sd:.4f} / {r.relative_error_mean:.2f} ± {r.relative_error_sd:.2f}%'
                if r.n_seeds==1: value=f'{r.rmse_mean:.4f} / {r.relative_error_mean:.2f}%'
                if m in best[s]: value='**'+value+'**'
                cells.append(value)
            text.append('| '+LABELS[m]+' | '+' | '.join(cells)+' |')
        text += ['', 'RMSE is in mm. SD describes training-seed variability. Black points in the figure are test replicate means; faint gray points are individual test measurements. Curves average predictions across seeds, except the single-fit logistic ODE.', '',
            f'![Prediction curves](predictions_{p}.png)','', '| Model | Secondary individual-measurement test RMSE (mm) |','|---|---:|']
        for m in ORDER:
            r=part[part.model.eq(m)&part.split.eq('test')].iloc[0]
            text.append(f'| {LABELS[m]} | {r.individual_rmse_mean:.4f} |')
        text += ['']
    (REPORT/'results.md').write_text('\n'.join(text))
    atomic(REPORT/'results_audit.json',dict(status='passed',condition='12L12D',
        target_type='split_specific_replicate_means_12L12D_only',environmental_feature_count=0,
        cR_observations_used=0,all_metrics_recomputed=True,split_specific_means_verified=True,
        source_hashes_verified=True,validation_ranking_verified=True,paired_initializations_identical=True,
        prior_test_inspected=True,full_training_fits=120,final_evaluations=38,
        exported_history_last_epoch=1400,training_epochs=1500,
        checkpoint_audit='Saved epochs and prediction arrays verified. Minimum exported validation history checked when it includes the selected epoch; the final 100-epoch history interval was not exported by the executed trainer and is not reconstructed.',
        selection_sha256=sha(OUT/'selections.json'),records=records))
    print(comparison[comparison.split.eq('test')][['protocol','model','rmse_mean','relative_error_mean','individual_rmse_mean']].to_string(index=False))


if __name__=='__main__': main()
