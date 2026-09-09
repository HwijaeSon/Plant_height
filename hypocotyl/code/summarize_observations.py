"""Audit raw-observation benchmarks and plot measured cells without averaging them."""
from __future__ import annotations
import json
from pathlib import Path
import hashlib
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from data import CASES, HOURS, GENOTYPES, CONDITIONS, inputs
from observation_data import load_observations, score_observations, observation_rmse
from observation_trial import build
from models import physics_losses
from run_observation_benchmark import ROOT, OUT, MODELS, PROTOCOLS, sha, atomic

REPORT = ROOT/'hypocotyl/reports/individual_observations_20260909'
ORDER = MODELS + ['latent_ode_matched']
LABELS = dict(phytoode='PhytoODE', latent_ode='Latent ODE (no physics)', light_pinn='Light-PINN',
    lstm='LSTM-NN', rf='Random forest', light_logistic='Light-logistic ODE', logistic='Logistic ODE',
    latent_ode_matched='Latent ODE (matched)')
COLORS = dict(phytoode='#0072B2',latent_ode='#332288',light_pinn='#D55E00',lstm='#009E73',
              rf='#999999',light_logistic='#CC79A7',logistic='#E69F00')


def table(frame, protocol):
    part = frame[frame.protocol.eq(protocol)]
    winners = {s:set(part[part.split.eq(s)&part.rmse_mean.eq(part[part.split.eq(s)].rmse_mean.min())].model)
               for s in ['train','val','test']}
    caption = ('Hypocotyl replicate-group holdout' if protocol == 'replicate' else 'Auxiliary 36/60-h holdout')
    lines = [r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{'+caption+r': individual-observation RMSE (mm) / relative RMSE (\%). Each measured cell is scored separately. Stochastic models report mean $\pm$ sample SD across three seeds; process ODEs are single fits. Bold marks the smallest unrounded mean in each split. The matched control uses the selected PhytoODE architecture and training settings with both physics coefficients zero.}',
        r'\label{tab:hypocotyl_'+protocol+r'}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{lccc}',
        r'\toprule',r'Model & Train & Validation & Test \\',r'\midrule']
    for m in ORDER:
        cells = []
        for s in ['train','val','test']:
            row = part[part.model.eq(m)&part.split.eq(s)].iloc[0]
            value = f'{row.rmse_mean:.3f} / {row.relative_error_mean:.2f}'
            if row.n_seeds > 1:
                value = f'{row.rmse_mean:.3f}\\pm{row.rmse_sd:.3f} / {row.relative_error_mean:.2f}\\pm{row.relative_error_sd:.2f}'
            if m in winners[s]: value = r'\mathbf{'+value+'}'
            cells.append('$'+value+'$')
        if m == 'latent_ode_matched': lines.append(r'\midrule')
        lines.append(LABELS[m]+' & '+' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}}',r'\end{table}',''])


def plot_predictions(predictions, batches, scale, destination, protocol, compact=False):
    genotypes = ['Col-0','MLB'] if compact else GENOTYPES
    fig,axes = plt.subplots(2,len(genotypes),figsize=(8,6.2) if compact else (17,7),squeeze=False,sharex=True)
    frame = batches['test']['observations']
    for row,condition in enumerate(CONDITIONS):
        for col,genotype in enumerate(genotypes):
            ax = axes[row,col]; i = CASES.index((genotype,condition))
            if condition == '12L12D':
                for start in [12,36,60]: ax.axvspan(start,start+12,color='0.9',alpha=.65,zorder=0)
            for m in MODELS:
                ax.plot(HOURS,predictions[m][i]*scale,color=COLORS[m],ls='--' if m=='latent_ode' else '-',
                        lw=2 if m=='phytoode' else 1.4,zorder=4 if m=='phytoode' else 2)
            part = frame[frame.genotype.eq(genotype)&frame.sheet.eq(condition)]
            jitter = part.observation_id.map(lambda s:(int(hashlib.sha256(s.encode()).hexdigest()[:8],16)/(2**32-1)-.5)*1.6)
            ax.scatter(part.elapsed_hours+jitter,part.length,s=16,color='black',alpha=.38,zorder=5,linewidths=0)
            ax.set_title(f'{genotype}, {condition}',fontsize=11)
            if row == 1: ax.set_xlabel('Elapsed time (h)')
            if col == 0: ax.set_ylabel('Hypocotyl length (mm)')
            ax.set_xticks([0,12,24,36,48,60,72] if compact else [0,24,48,72])
            ax.set_xlim(-2,74); ax.set_ylim(bottom=0); ax.grid(alpha=.12)
            ax.spines[['top','right']].set_visible(False)
    handles = [Line2D([0],[0],color=COLORS[m],ls='--' if m=='latent_ode' else '-',lw=2,label=LABELS[m]) for m in MODELS]
    handles.append(Line2D([0],[0],marker='o',color='black',ls='',markersize=4,label='Observed test measurement'))
    fig.legend(handles=handles,loc='upper center',ncol=3 if compact else 4,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.85 if compact else .87))
    for extension in ['png','pdf']:
        fig.savefig(destination.with_suffix('.'+extension),dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    torch.set_num_threads(2); REPORT.mkdir(exist_ok=True,parents=True)
    read = lambda p:json.loads(p.read_text())
    complete,selection,protocol = [read(OUT/name) for name in ['completed.json','selections.json','protocol.json']]
    assert complete['status']=='complete' and complete['selection_sha256']==sha(OUT/'selections.json')
    assert all(sha(ROOT/p)==h for p,h in protocol['source_sha256'].items())
    rows=[]; audited=[]; settings=[]; refinements=[]; losses=[]; rates=[]; residual_rows=[]
    for p in PROTOCOLS:
        config,x,batches = load_observations(p,splits=('train','val','test'))
        scale=config['height_scale_mm']; predictions={}
        for m,choice in selection['selected'][p].items():
            if 'ranking' in choice:
                assert choice['selected_candidate']==min(choice['ranking'],key=lambda r:(r['val_mean'],r['candidate']))['candidate']
                for candidate in choice['ranking']:
                    seeds=[1] if m in ['logistic','light_logistic'] else [1,2,3]
                    values=[read(OUT/p/'trials'/candidate['candidate']/f'seed{s}'/'result.json')['metrics']['val']['rmse'] for s in seeds]
                    np.testing.assert_array_equal(values,candidate['per_seed_validation'])
                    assert sum(values)/len(values)==candidate['val_mean']
        for record in [r for r in selection['runs'] if r['protocol']==p]:
            m,s=record['model'],record['seed']; source=ROOT/record['path']
            final=OUT/p/'final'/m/f'seed{s}'
            trained,evaluated=read(source/'result.json'),read(final/'result.json')
            assert sha(source/'result.json')==record['result_sha256']==evaluated['source_result_sha256']
            assert evaluated['selection_sha256']==sha(OUT/'selections.json')
            assert trained['test_evaluations']==0 and evaluated['test_evaluations']==1
            saved,scored=np.load(source/'predictions.npz'),np.load(final/'predictions.npz')
            assert not any(k.startswith('test_') for k in saved.files)
            prediction=saved['prediction']; np.testing.assert_array_equal(prediction,scored['prediction'])
            predictions.setdefault(m,[]).append(prediction)
            for split,batch in batches.items():
                metrics=score_observations(prediction,batch,scale)
                for key,value in metrics.items(): np.testing.assert_allclose(value,evaluated['metrics'][split][key],rtol=0,atol=0)
                for key in ['case','time','length']:
                    np.testing.assert_array_equal(batch[key].numpy(),scored[f'{split}_{key}'])
                rows.append(dict(protocol=p,model=m,seed=s,split=split,**{k:v for k,v in metrics.items() if k!='per_curve_rmse'}))
                frame=batch['observations']
                measured_prediction=np.asarray(prediction,dtype=float)[batch['case'].numpy(),batch['time'].numpy()]*scale
                residual_rows.extend(dict(protocol=p,model=m,seed=s,split=split,observation_id=identity,
                    observed_mm=float(measured),predicted_mm=float(predicted))
                    for identity,measured,predicted in zip(frame.observation_id,frame.length,measured_prediction))
            if trained['checkpoint_sha256']:
                assert sha(source/'checkpoint.pt')==trained['checkpoint_sha256']
                history=pd.read_csv(source/'history.csv')
                best=history[history.epoch.ge(200)].sort_values(['selection_rmse','epoch']).iloc[0]
                assert int(best.epoch)==trained['best_epoch']
            settings.append(dict(protocol=p,model=m,seed=s,best_epoch=trained['best_epoch'],
                n_params=trained['n_params'],config=trained['config'],source=record['path']))
            audited.append(dict(protocol=p,model=m,seed=s,source_result_sha256=sha(source/'result.json'),
                prediction_sha256=sha(source/'predictions.npz'),final_result_sha256=sha(final/'result.json')))
            if m=='phytoode':
                net=build(trained['config']); checkpoint=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
                net.load_state_dict(checkpoint['state_dict']); net.eval(); output=net(**x)
                ode,k=physics_losses(net,output,x)
                losses.append(dict(protocol=p,seed=s,data_loss=float(observation_rmse(output['pred'],batches['train']).detach()),
                    ode_loss=float(ode.detach()),k_loss=float(k.detach()),
                    weighted_ode=float(ode.detach())*trained['config']['lambda_ode'],
                    weighted_k=float(k.detach())*trained['config']['lambda_k']))
                fine=inputs(np.arange(0.,72.1,1.5))
                with torch.no_grad():
                    coarse=output['pred'].detach().numpy()
                    refined=net(**fine,encoder_env=x['env'],encoder_time=x['time'])['pred'].numpy()[:,::2]
                np.testing.assert_allclose(coarse,prediction,atol=1e-5,rtol=1e-5)
                delta=(refined-coarse)*scale
                refinements.append(dict(protocol=p,seed=s,max_absolute_difference_mm=float(np.abs(delta).max()),
                                        rms_difference_mm=float(np.sqrt(np.mean(delta**2)))))
                for i,g in enumerate(GENOTYPES):
                    rates.append(dict(protocol=p,model=m,seed=s,genotype=g,
                        r_light_per_hour=float(output['r_light'][2*i].detach()),
                        r_dark_per_hour=float(output['r_dark'][2*i].detach()),K_mm=float(output['K'][2*i].detach())*scale))
            if m=='light_logistic':
                for i,r in enumerate(read(source/'parameters.json')['parameters']):
                    rates.append(dict(protocol=p,model=m,seed=s,genotype=GENOTYPES[i],
                        r_light_per_hour=r['parameters'][0],r_dark_per_hour=r['parameters'][1],K_mm=r['parameters'][2]*scale))
        for s in [1,2,3]:
            full=next(r for r in settings if r['protocol']==p and r['model']=='phytoode' and r['seed']==s)
            pure=next(r for r in settings if r['protocol']==p and r['model']=='latent_ode_matched' and r['seed']==s)
            assert pure['config']==(full['config']|dict(model='latent_ode',lambda_ode=0.,lambda_k=0.))
            assert read(ROOT/full['source']/'result.json')['initial_state_sha256']==read(ROOT/pure['source']/'result.json')['initial_state_sha256']
        mean_predictions={m:np.mean(values,axis=0) for m,values in predictions.items()}
        plot_predictions(mean_predictions,batches,scale,REPORT/f'predictions_{p}',p)
        if p=='replicate':
            compact=REPORT/'predictions_replicate_compact'
            plot_predictions(mean_predictions,batches,scale,compact,p,compact=True)
            if (ROOT/'paper/figures').is_dir():
                for extension in ['pdf','png']:
                    shutil.copyfile(compact.with_suffix('.'+extension),ROOT/f'paper/figures/hypocotyl_light_replicates.{extension}')
    frame=pd.DataFrame(rows); frame.to_csv(REPORT/'per_seed_metrics.csv',index=False)
    aggregates=[]
    for (p,m,s),group in frame.groupby(['protocol','model','split'],sort=False):
        record=dict(protocol=p,model=m,split=s,n_seeds=len(group),n_observations=int(group.n_observations.iloc[0]))
        for metric in ['rmse','relative_error']:
            record[metric+'_mean']=float(group[metric].mean())
            record[metric+'_sd']=float(group[metric].std(ddof=1)) if len(group)>1 else 0.
        aggregates.append(record)
    comparison=pd.DataFrame(aggregates); comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(refinements).to_csv(REPORT/'integration_refinement.csv',index=False)
    pd.DataFrame(losses).to_csv(REPORT/'selected_loss_components.csv',index=False)
    pd.DataFrame(rates).to_csv(REPORT/'light_growth_parameters.csv',index=False)
    pd.DataFrame(residual_rows).to_csv(REPORT/'individual_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    atomic(REPORT/'selected_configs.json',dict(selected=selection['selected'],runs=settings))
    text=['# Hypocotyl benchmark: individual observed measurements','',
        'No replicate means are used as targets. Missing phenotype cells have no residual. RMSE pools individual measurements; relative RMSE divides by their observed mean. Every model was fitted anew. The two latent models received equal screening and confirmation budgets. The frozen partitions had been inspected in earlier analyses; current selection used validation only.','']
    for p in PROTOCOLS:
        (REPORT/f'table_{p}.tex').write_text(table(comparison,p))
        text += [f'## {p}','', '| Model | Train RMSE / rRMSE | Validation RMSE / rRMSE | Test RMSE / rRMSE |','|---|---|---|---|']
        part=comparison[comparison.protocol.eq(p)]
        winners={s:set(part[part.split.eq(s)&part.rmse_mean.eq(part[part.split.eq(s)].rmse_mean.min())].model)
                 for s in ['train','val','test']}
        for m in ORDER:
            cells=[]
            for s in ['train','val','test']:
                r=part[part.model.eq(m)&part.split.eq(s)].iloc[0]
                value=f'{r.rmse_mean:.4f} ± {r.rmse_sd:.4f} / {r.relative_error_mean:.2f} ± {r.relative_error_sd:.2f}%'
                if r.n_seeds==1: value=f'{r.rmse_mean:.4f} / {r.relative_error_mean:.2f}%'
                if m in winners[s]: value='**'+value+'**'
                cells.append(value)
            text.append('| '+LABELS[m]+' | '+' | '.join(cells)+' |')
        text += ['', 'All RMSE values are in mm. SD is across training seeds. Observations in the figures are individual measurements; only model predictions are averaged across seeds.','']
    (REPORT/'results.md').write_text('\n'.join(text))
    atomic(REPORT/'results_audit.json',dict(status='passed',target_type='individual_observed_cells',
        imputed_phenotype_targets=0,replicate_mean_targets=0,all_metrics_recomputed=True,
        source_hashes_verified=True,validation_ranking_verified=True,paired_initializations_identical=True,
        prior_test_inspected=True,full_training_fits=complete['full_training_trials'],
        final_evaluations=len(audited),selection_sha256=sha(OUT/'selections.json'),records=audited))
    print(comparison[comparison.split.eq('test')][['protocol','model','rmse_mean','relative_error_mean']].to_string(index=False))


if __name__=='__main__': main()
