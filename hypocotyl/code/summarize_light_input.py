"""Audit and visualize the PhytoODE-only light-input follow-up."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from single_condition_data import load,score,curve_rmse,GENOTYPES,HOURS,DATA
from light_input_model import build,inputs,physics_losses
from run_light_input import ROOT,OUT,PREVIOUS,PROTOCOLS,sha,atomic

REPORT=ROOT/'hypocotyl/reports/light_input_20260910'
OLD_REPORT=ROOT/'hypocotyl/reports/single_condition_20260909'
OLD_MODELS=['phytoode','latent_ode','logistic_pinn','lstm','rf','logistic','latent_ode_matched']
NEW_MODELS=['phytoode_light','phytoode_light_anchor']
ORDER=NEW_MODELS+OLD_MODELS
PLOT_ORDER=['phytoode_light','phytoode_light_anchor','phytoode','latent_ode','logistic_pinn','lstm','rf','logistic']
LABELS=dict(phytoode_light='PhytoODE + light (tuned)',phytoode_light_anchor='PhytoODE + light (fixed settings)',
    phytoode='PhytoODE (no light input)',latent_ode='Latent ODE (no physics)',
    logistic_pinn='Logistic-PINN',lstm='LSTM-NN',rf='Random forest',logistic='Logistic ODE',
    latent_ode_matched='Latent ODE (matched to no-light PhytoODE)')
COLORS=dict(phytoode_light='#CC3377',phytoode_light_anchor='#CC3377',phytoode='#0072B2',
    latent_ode='#332288',logistic_pinn='#D55E00',lstm='#009E73',rf='#999999',logistic='#E69F00')
TABLE_LABELS=LABELS|dict(phytoode='PhytoODE',phytoode_light_anchor='PhytoODE + light (fixed)',
    latent_ode='Latent ODE',latent_ode_matched='Latent ODE (matched)')


def read(path): return json.loads(path.read_text())


def best_models(part,split):
    sub=part[part.split.eq(split)]
    return set(sub[sub.rmse_mean.eq(sub.rmse_mean.min())].model)


def table(frame,protocol):
    part=frame[frame.protocol.eq(protocol)]
    title='Replicate holdout' if protocol=='replicate' else '36/60-h holdout'
    lines=[r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{'+title+r', 12L12D only: RMSE (mm, mean $\pm$ sample SD) / relative RMSE (\%, mean) across three seeds, except the single-fit logistic ODE. Targets are split-specific replicate means. Bold marks the lowest unrounded mean in each split. Only the two added PhytoODE variants use binary illumination as an input; both retain ordinary logistic physics. Fixed settings reuse the selected no-light PhytoODE hyperparameters. The tuned variant has an additional search budget. The matched latent ODE uses the no-light PhytoODE settings with both physics coefficients zero.}',
        r'\label{tab:hypocotyl_light_'+protocol+r'}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{lccc}',
        r'\toprule',r'Model & Train & Validation & Test \\',r'\midrule']
    for model in ORDER:
        if model in ['phytoode','latent_ode_matched']: lines.append(r'\midrule')
        cells=[]
        for split in ['train','val','test']:
            row=part[part.model.eq(model)&part.split.eq(split)].iloc[0]
            value=f'{row.rmse_mean:.3f} / {row.relative_error_mean:.2f}'
            if row.n_seeds>1:
                value=f'{row.rmse_mean:.3f}\\pm{row.rmse_sd:.3f} / {row.relative_error_mean:.2f}'
            if model in best_models(part,split): value=r'\mathbf{'+value+'}'
            cells.append('$'+value+'$')
        lines.append(TABLE_LABELS[model]+' & '+' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}}',r'\end{table}',''])


def decorate(ax,genotype):
    for start in [12,36,60]: ax.axvspan(start,start+12,color='0.90',alpha=.6,lw=0,zorder=0)
    ax.set_title(genotype,fontsize=12)
    ax.set_xlabel('Elapsed time (h)'); ax.set_ylabel('Hypocotyl length (mm)')
    ax.set_xlim(-2,74); ax.set_xticks([0,12,24,36,48,60,72])
    ax.grid(alpha=.12); ax.spines[['top','right']].set_visible(False)


def plot_curves(predictions,batches,scale,destination,protocol,focused=False):
    fig,axes=plt.subplots(2,3,figsize=(13,7.6)); axes=axes.ravel()
    raw=batches['test']['observations']; means=batches['test']['means']
    models=['phytoode_light','phytoode_light_anchor','phytoode'] if focused else PLOT_ORDER
    for i,(ax,g) in enumerate(zip(axes,GENOTYPES)):
        decorate(ax,g)
        for model in reversed(models):
            values=predictions[model][:,i,:]*scale
            linestyle='--' if model in ['latent_ode','phytoode_light_anchor'] else '-'
            ax.plot(HOURS,values.mean(axis=0),color=COLORS[model],ls=linestyle,
                lw=2.5 if model=='phytoode_light' else 1.8,zorder=4 if model=='phytoode_light' else 2)
            if focused:
                spread=values.std(axis=0,ddof=1)
                ax.fill_between(HOURS,values.mean(axis=0)-spread,values.mean(axis=0)+spread,
                    color=COLORS[model],alpha=.10,linewidth=0,zorder=1)
        points=raw[raw.genotype.eq(g)]
        jitter=points.observation_id.map(lambda s:(int(hashlib.sha256(s.encode()).hexdigest()[:8],16)/(2**32-1)-.5)*1.2)
        ax.scatter(points.elapsed_hours+jitter,points.length,color='0.4',alpha=.20,s=14,linewidths=0,zorder=1)
        group=means[means.genotype.eq(g)]
        ax.scatter(group.elapsed_hours,group.mean_length_mm,color='black',s=29,zorder=6,linewidths=.5,edgecolors='white')
        if protocol=='time_holdout':
            support=batches['train']['means']; support=support[support.genotype.eq(g)]
            ax.scatter(support.elapsed_hours,support.mean_length_mm,facecolors='white',edgecolors='black',
                s=25,linewidths=.8,zorder=5)
        # Fix the lower limit after adding data, preserving its autoscaled top.
        ax.set_ylim(bottom=0)
    handles=[Line2D([0],[0],color=COLORS[m],ls='--' if m in ['latent_ode','phytoode_light_anchor'] else '-',lw=2,label=LABELS[m]) for m in models]
    handles += [Line2D([0],[0],marker='o',color='black',ls='',markersize=5,label='Observed test mean'),
        Line2D([0],[0],marker='o',color='0.65',alpha=.5,ls='',markersize=4,label='Individual test measurement')]
    if protocol=='time_holdout':
        handles.append(Line2D([0],[0],marker='o',markerfacecolor='white',color='black',ls='',markersize=5,label='Observed training mean'))
    handles.append(Patch(facecolor='0.90',alpha=.6,label='Dark interval (12 h)'))
    if focused: handles.append(Patch(facecolor=COLORS['phytoode_light'],alpha=.1,label='Curve bands: ±1 seed SD'))
    axes[-1].axis('off'); axes[-1].legend(handles=handles,loc='center',frameon=False,fontsize=10)
    title='Replicate holdout' if protocol=='replicate' else '36 / 60 h holdout'
    fig.suptitle(title+' — 12L12D only; ordinary logistic physics',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95))
    for ext in ['png','pdf']: fig.savefig(destination.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def plot_errors(frame):
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.9),sharey=True)
    for ax,p in zip(axes,PROTOCOLS):
        part=frame[frame.protocol.eq(p)&frame.split.eq('test')].set_index('model').loc[PLOT_ORDER]
        positions=np.arange(len(PLOT_ORDER))
        ax.barh(positions,part.relative_error_mean,color=[COLORS[m] for m in PLOT_ORDER],alpha=.85,height=.65)
        ax.errorbar(part.relative_error_mean,positions,xerr=part.relative_error_sd,fmt='none',ecolor='0.20',elinewidth=1,capsize=3)
        right=float((part.relative_error_mean+part.relative_error_sd).max()); ax.set_xlim(0,right*1.29)
        for pos,row in zip(positions,part.itertuples()):
            ax.text(row.relative_error_mean+row.relative_error_sd+right*.025,pos,f'{row.relative_error_mean:.2f}%',va='center',fontsize=9)
        ax.set_yticks(positions,[LABELS[m] for m in PLOT_ORDER],fontsize=10)
        ax.set_xlabel('Test relative RMSE (%)'); ax.set_title('Replicate holdout' if p=='replicate' else '36 / 60 h holdout')
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle('12L12D only: existing baselines and PhytoODE with light input',fontsize=13)
    fig.text(.5,.005,'Bars: mean of per-seed scores; whiskers: ±1 seed SD. Logistic ODE: one fit.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.95))
    for ext in ['png','pdf']: fig.savefig(REPORT/f'test_error_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    torch.set_num_threads(2); REPORT.mkdir(parents=True,exist_ok=True)
    completed,selection,protocol=[read(OUT/name) for name in ['completed.json','selections.json','protocol.json']]
    assert completed['status']=='complete' and completed['selection_sha256']==sha(OUT/'selections.json')
    assert completed['final_evaluations']==12 and completed['baseline_refits']==0
    assert all(sha(ROOT/p)==h for p,h in protocol['source_sha256'].items())
    # Audit every new trial, including the final 100 epochs and exact smoothing.
    histories=[]; search_rows=[]
    for path in sorted(OUT.glob('*/trials/*/seed*/result.json')):
        r=read(path); folder=path.parent; history=pd.read_csv(folder/'history.csv',float_precision='round_trip')
        assert history.epoch.tolist()==list(range(20,1501,20)) and r['history_last_epoch']==1500
        assert sha(folder/'history.csv')==r['history_sha256'] and sha(folder/'checkpoint.pt')==r['checkpoint_sha256']
        vals=history.val_rmse_normalized.tolist()
        smooth=[sum(vals[max(0,i-2):i+1])/len(vals[max(0,i-2):i+1]) for i in range(len(vals))]
        np.testing.assert_allclose(smooth,history.selection_rmse_normalized,rtol=0,atol=0)
        winner=min((s,int(e)) for s,e in zip(smooth,history.epoch) if e>=200)
        assert winner==(r['best_smoothed_validation_normalized'],r['best_epoch'])
        assert r['test_evaluations']==0 and set(r['metrics'])=={'train','val'} and r['n_params']==1103
        assert r['config']==protocol['candidates'][folder.parent.name]
        with np.load(folder/'predictions.npz') as a: assert not any(k.startswith('test_') for k in a.files)
        histories.append(dict(protocol=r['protocol'],candidate=folder.parent.name,seed=r['seed'],
            best_epoch=r['best_epoch'],last_epoch=1500,history_sha256=r['history_sha256']))
        search_rows.append(dict(protocol=r['protocol'],candidate=folder.parent.name,seed=r['seed'],
            best_epoch=r['best_epoch'],train_rmse=r['metrics']['train']['rmse'],val_rmse=r['metrics']['val']['rmse'],
            **r['config']))
    assert len(histories)==completed['full_training_trials']
    finalists=read(OUT/'finalists.json'); previous=read(PREVIOUS/'selections.json')
    for p in PROTOCOLS:
        def trial(key,seed): return read(OUT/p/'trials'/key/f'seed{seed}'/'result.json')
        expected=sorted(protocol['candidates'],key=lambda k:(trial(k,1)['metrics']['val']['rmse'],k))[:3]
        if protocol['anchors'][p] not in expected: expected.append(protocol['anchors'][p])
        assert finalists[p]==expected
        choice=selection['selected'][p]['phytoode_light']
        for candidate in choice['ranking']:
            values=[trial(candidate['candidate'],s)['metrics']['val']['rmse'] for s in [1,2,3]]
            assert values==candidate['per_seed_validation'] and sum(values)/3==candidate['val_mean']
        assert choice['selected_candidate']==min(choice['ranking'],key=lambda r:(r['val_mean'],r['candidate']))['candidate']
        anchor=selection['selected'][p]['phytoode_light_anchor']
        assert anchor['config']==(previous['selected'][p]['phytoode']['config']|dict(model='phytoode_light'))
    rows=[]; settings=[]; records=[]; losses=[]; refinements=[]; parameters=[]; curves=[]; targets=[]
    for p in PROTOCOLS:
        data_config,_,batches=load(p,splits=('train','val','test')); scale=data_config['height_scale_mm']; x=inputs()
        predictions={}
        for batch in batches.values():
            computed=batch['observations'].groupby(['genotype','elapsed_hours']).length.mean().sort_index()
            saved=batch['means'].set_index(['genotype','elapsed_hours']).mean_length_mm.sort_index()
            np.testing.assert_allclose(computed,saved,atol=1e-14,rtol=1e-14)
        for root in [PREVIOUS,OUT]:
            selected=previous if root==PREVIOUS else selection
            for record in [r for r in selected['runs'] if r['protocol']==p]:
                model,seed=record['model'],record['seed']; source=ROOT/record['path']; final=root/p/'final'/model/f'seed{seed}'
                trained,evaluated=read(source/'result.json'),read(final/'result.json')
                assert sha(source/'result.json')==record['result_sha256']==evaluated['source_result_sha256']
                assert evaluated['selection_sha256']==sha(root/'selections.json')
                assert trained['test_evaluations']==0 and evaluated['test_evaluations']==1
                with np.load(final/'predictions.npz') as a:
                    pred=a['prediction'].copy()
                    for split,batch in batches.items():
                        for key in ['case','time','target']: np.testing.assert_array_equal(a[f'{split}_{key}'],batch[key].numpy())
                with np.load(source/'predictions.npz') as a:
                    assert not any(k.startswith('test_') for k in a.files)
                    np.testing.assert_array_equal(pred,a['prediction'])
                assert pred.shape==(5,25) and np.isfinite(pred).all()
                predictions.setdefault(model,[]).append(pred)
                for split,batch in batches.items():
                    metrics=score(pred,batch,scale)
                    assert metrics==evaluated['metrics'][split]
                    rows.append(dict(protocol=p,model=model,seed=seed,split=split,**{k:v for k,v in metrics.items() if k!='per_curve_rmse'}))
                    curves.extend(dict(protocol=p,model=model,seed=seed,split=split,genotype=g,rmse=v) for g,v in zip(GENOTYPES,metrics['per_curve_rmse']))
                    physical=np.asarray(pred,dtype=float)*scale
                    targets.extend(dict(protocol=p,model=model,seed=seed,split=split,genotype=r.genotype,elapsed_hours=r.elapsed_hours,
                        observed_mean_mm=r.mean_length_mm,n_observations=r.n_observations,predicted_mm=float(physical[GENOTYPES.index(r.genotype),int(r.elapsed_hours/3)]))
                        for r in batch['means'].itertuples())
                records.append(dict(protocol=p,model=model,seed=seed,final_result_sha256=sha(final/'result.json')))
                if root==PREVIOUS: continue
                checkpoint=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
                assert checkpoint['epoch']==trained['best_epoch'] and checkpoint['config']==trained['config']
                net=build(trained['config']); net.load_state_dict(checkpoint['state_dict']); net.eval()
                out=net(**x); ode,k=physics_losses(net,out,x)
                with torch.no_grad():
                    coarse=out['pred'].detach().numpy()
                    xf=inputs(np.arange(0.,72.1,1.5))
                    fine=net(**xf,encoder_time=x['time'],encoder_env=x['env'])['pred'].numpy()[:,::2]
                np.testing.assert_allclose(coarse,pred,rtol=1e-5,atol=1e-5)
                delta=(fine-coarse)*scale
                refinements.append(dict(protocol=p,model=model,seed=seed,max_absolute_difference_mm=float(np.abs(delta).max()),
                    rms_difference_mm=float(np.sqrt(np.mean(delta**2)))))
                losses.append(dict(protocol=p,model=model,seed=seed,data_loss=float(curve_rmse(out['pred'],batches['train']).detach()),
                    ode_loss=float(ode.detach()),k_loss=float(k.detach()),weighted_ode=float(ode.detach())*trained['config']['lambda_ode'],
                    weighted_k=float(k.detach())*trained['config']['lambda_k']))
                parameters.extend(dict(protocol=p,model=model,seed=seed,genotype=g,r_per_hour=float(out['r'][i].detach()),
                    K_mm=float(out['K'][i].detach())*scale) for i,g in enumerate(GENOTYPES))
                settings.append(dict(protocol=p,model=model,seed=seed,n_params=trained['n_params'],best_epoch=trained['best_epoch'],config=trained['config'],source=record['path']))
        stacked={m:np.stack(values).astype(float) for m,values in predictions.items()}
        plot_curves(stacked,batches,scale,REPORT/f'predictions_{p}',p)
        plot_curves(stacked,batches,scale,REPORT/f'phytoode_variants_{p}',p,focused=True)
    frame=pd.DataFrame(rows); frame.to_csv(REPORT/'per_seed_metrics.csv',index=False)
    aggregates=[]
    for (p,m,s),part in frame.groupby(['protocol','model','split'],sort=False):
        entry=dict(protocol=p,model=m,split=s,n_seeds=len(part),n_observations=int(part.n_observations.iloc[0]),n_targets=int(part.n_scored_group_means.iloc[0]))
        for metric in ['rmse','relative_error','individual_rmse']:
            entry[metric+'_mean']=float(part[metric].mean());entry[metric+'_sd']=float(part[metric].std(ddof=1)) if len(part)>1 else 0.
        aggregates.append(entry)
    comparison=pd.DataFrame(aggregates)
    prior_comparison=pd.read_csv(OLD_REPORT/'comparison.csv',float_precision='round_trip')
    index=['protocol','model','split']
    pd.testing.assert_frame_equal(comparison[comparison.model.isin(OLD_MODELS)].set_index(index).sort_index(),
        prior_comparison.set_index(index).sort_index(),check_exact=True)
    comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(curves).to_csv(REPORT/'per_curve_metrics.csv',index=False)
    pd.DataFrame(targets).to_csv(REPORT/'mean_target_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(losses).to_csv(REPORT/'selected_loss_components.csv',index=False)
    pd.DataFrame(refinements).to_csv(REPORT/'integration_refinement.csv',index=False)
    pd.DataFrame(parameters).to_csv(REPORT/'logistic_parameters.csv',index=False)
    atomic(REPORT/'selected_configs.json',dict(selected=selection['selected'],runs=settings))
    search=pd.DataFrame(search_rows)
    search['selected_tuned']=[key==selection['selected'][p]['phytoode_light']['selected_candidate'] for p,key in zip(search.protocol,search.candidate)]
    search['fixed_settings_anchor']=[key==protocol['anchors'][p] for p,key in zip(search.protocol,search.candidate)]
    search.to_csv(REPORT/'validation_search.csv',index=False)
    plot_errors(comparison)
    text=['# PhytoODE with binary light input: 12L12D-only follow-up','',
        'Only PhytoODE is newly trained. All earlier baseline predictions and scores are retained exactly. cR is excluded; temperature is fixed at 23°C and is not supplied as a feature. The partitions and split-specific mean targets are unchanged. No missing phenotype is filled in.','',
        'The new model receives genotype, normalized elapsed time, and one known binary light state: on = 1 for [0,12), [24,36), [48,60) h; off = 0 for [12,24), [36,48), [60,72) h. The endpoint at 72 h is the next on state. The encoder reads the known schedule, never a height observation. The latent vector field receives the instantaneous light state.','',
        'Physics remains the ordinary logistic residual dh/dt - r_g h (1-h/K_g), with one constant r and K per genotype, plus the original finite-window capacity penalty. There are no separate light/dark logistic rates. The decoder chain rule computes dh/dt = (partial h/partial z) f(z,g,t,L)/72 by automatic differentiation. RK4 holds illumination constant within each interval and changes it at the 12 h boundaries. The original 23 interior physics nodes are retained, using the right-hand derivative at switching times. No derivative of a measured length or of the binary switch is required. Hidden widths are unchanged; adding the channel increases the parameter count from 1,055 to 1,103.','',
        'Each protocol screens 32 fixed seed-1 configurations: the 16 previous PhytoODE grid combinations plus 16 preregistered stratified log-space samples of learning rate, lambda_ODE, lambda_K, and weight decay. All fits run 1,500 epochs. The best three seed-1 candidates, plus the prior-selected no-light hyperparameter anchor if needed, are confirmed with seeds 2 and 3. The tuned variant minimizes mean validation RMSE among confirmed candidates. Both protocols freeze selections before any current test scoring. Checkpoints minimize the trailing three validation evaluations after epoch 200; the complete history is exported through epoch 1,500.','',
        f"Completed {completed['full_training_trials']} new fits and {completed['final_evaluations']} final evaluations. The fixed-settings variant uses all previously selected no-light PhytoODE hyperparameters and the same three seed labels. Adding an input changes the weight-matrix dimensions and initialization, so this is not an identical-initial-weight ablation. The tuned variant additionally receives a larger search budget than the retained baselines. Earlier results on these same partitions have been inspected; this follow-up is not an independent new test cohort. No search extension was made after the current test scores were computed.",'',
        'All plants share one predetermined 12L12D schedule, so illumination is a deterministic function of elapsed time. Adding it supplies an explicit phase feature, not independent evidence of a causal light response or transfer to a different photoperiod.','',
        'Primary RMSE (mm) is the mean of five genotype-curve RMSEs against split-specific replicate means. Relative RMSE is 100 × that RMSE / the mean of the scored replicate-mean targets; it is not MAPE. Table entries average per-seed metrics; they are not scores of an ensemble. SD measures training-seed variability and is not a confidence interval or a test of significance. Individual-measurement RMSE is saved separately in the CSV.','']
    effects=[]
    for p in PROTOCOLS:
        config=read(DATA/p/'config.json'); part=comparison[comparison.protocol.eq(p)]
        (REPORT/f'table_{p}.tex').write_text(table(comparison,p))
        text += [f'## {p}','',f"Raw measurements train / validation / test: {config['counts']['train']} / {config['counts']['val']} / {config['counts']['test']}; mean targets: {config['mean_target_counts']['train']} / {config['mean_target_counts']['val']} / {config['mean_target_counts']['test']}.",'',
            '| Model | Train RMSE / rRMSE | Validation RMSE / rRMSE | Test RMSE / rRMSE |','|---|---|---|---|']
        for m in ORDER:
            cells=[]
            for split in ['train','val','test']:
                row=part[part.model.eq(m)&part.split.eq(split)].iloc[0]
                value=f'{row.rmse_mean:.4f} ± {row.rmse_sd:.4f} / {row.relative_error_mean:.2f} ± {row.relative_error_sd:.2f}%'
                if row.n_seeds==1: value=f'{row.rmse_mean:.4f} / {row.relative_error_mean:.2f}%'
                if m in best_models(part,split): value='**'+value+'**'
                cells.append(value)
            text.append('| '+LABELS[m]+' | '+' | '.join(cells)+' |')
        selected=selection['selected'][p]['phytoode_light']['config']
        text += ['', 'Selected light-input settings: '+', '.join(f'{k}={selected[k]:.7g}' for k in ['lr','lambda_ode','lambda_k','weight_decay'])+'.','']
        test=part[part.split.eq('test')].set_index('model')
        for m in NEW_MODELS:
            effects.append(dict(protocol=p,model=m,no_input_test_rmse=float(test.loc['phytoode','rmse_mean']),
                light_input_test_rmse=float(test.loc[m,'rmse_mean']),
                test_rmse_change_mm=float(test.loc[m,'rmse_mean']-test.loc['phytoode','rmse_mean']),
                test_rmse_reduction_percent=float(100*(1-test.loc[m,'rmse_mean']/test.loc['phytoode','rmse_mean']))))
        if test.loc['phytoode_light_anchor','rmse_mean']<test.loc['phytoode_light','rmse_mean']:
            text += ['The fixed-settings light-input variant has a lower test RMSE than the validation-selected tuned variant. This does not change the frozen validation selection: the fixed-settings row is a prespecified feature comparison, not a model chosen afterward by test performance.','']
        text += [f"Lowest observed test mean: {', '.join(LABELS[m] for m in ORDER if m in best_models(part,'test'))}. This ranking alone does not establish statistical superiority.",'',
            f'![All model curves](predictions_{p}.png)','',
            'Curves average predictions across three seeds; the process logistic ODE has one fit. Black circles are observed test means. Faint gray points are individual test measurements, with deterministic horizontal jitter for visibility. For the 36/60-h holdout, open circles show training means. Shaded vertical intervals denote darkness, not missing data. The previously matched no-physics control is included in the table but omitted from the multi-model plot to reduce overlap.','',
            f'![PhytoODE feature and tuning comparison](phytoode_variants_{p}.png)','',
            'The focused plot compares no light input, light input at fixed previous settings (dashed), and tuned light input (solid). Curve bands show ±1 SD across seeds; they are not measurement uncertainty.','']
    pd.DataFrame(effects).to_csv(REPORT/'feature_effect.csv',index=False)
    (REPORT/'results.md').write_text('\n'.join(text))
    atomic(REPORT/'results_audit.json',dict(status='passed',source_hashes_verified=True,retained_baseline_metrics_exact=True,
        cR_observations_used=0,split_specific_means_verified=True,all_metrics_recomputed=True,
        full_new_validation_histories_verified=True,validation_selection_reconstructed=True,
        new_training_fits=completed['full_training_trials'],new_final_evaluations=12,baseline_refits=0,
        prior_test_inspected=True,additional_tuning_budget=True,selection_sha256=sha(OUT/'selections.json'),
        max_integration_refinement_difference_mm=max(r['max_absolute_difference_mm'] for r in refinements),
        histories=histories,records=records))
    print(comparison[comparison.split.eq('test')][['protocol','model','rmse_mean','rmse_sd','relative_error_mean']].to_string(index=False))


if __name__=='__main__': main()
