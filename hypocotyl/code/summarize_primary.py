"""Audit and report primary-split follow-up results; never rank by test for selection."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

from data import CASES, HOURS, GENOTYPES, CONDITIONS, load, inputs, score, curve_rmse
from models import physics_losses
from retune_model import build
from retune_primary import ROOT, OLD, OUT
from summarize import ORDER, LABELS, COLORS, STYLES
from trial import sha, atomic

REPORT = ROOT/'hypocotyl/reports/primary_retuning_20260909'
REPORT_ORDER = ORDER+['latent_ode_matched','latent_ode_initial_control']
LABELS = LABELS | {'latent_ode_initial_control': 'Latent ODE (initial control)'}


def primary_table(frame):
    models=REPORT_ORDER
    winners={s:frame[frame.split.eq(s)].sort_values('rmse_mean').iloc[0].model for s in ['train','val','test']}
    lines=[r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Hypocotyl replicate-group holdout: RMSE (mm) / relative RMSE (\%). Bold marks the smallest unrounded mean among all rows; subprecision differences are discussed in the text. Stochastic fits use three seeds (mean $\pm$ sample SD); process ODEs are single fits. The main no-physics baseline retains independent learning-rate selection. The matched control uses final PhytoODE settings; the initial control retains the earlier architecture and learning rate 0.01. Both physics coefficients are zero in every no-physics row. Scores compare population-mean curves.}',
        r'\label{tab:hypocotyl_replicate}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{lccc}',
        r'\toprule',r'Model & Train & Validation & Test \\',r'\midrule']
    for m in models:
        cells=[]
        for s in ['train','val','test']:
            r=frame[frame.model.eq(m)&frame.split.eq(s)].iloc[0]
            value=f'{r.rmse_mean:.3f} / {r.relative_error_mean:.2f}'
            if r.n_seeds>1:
                value=f'{r.rmse_mean:.3f}\\pm{r.rmse_sd:.3f} / {r.relative_error_mean:.2f}\\pm{r.relative_error_sd:.2f}'
            if m==winners[s]:value=r'\mathbf{'+value+'}'
            cells.append('$'+value+'$')
        if m=='latent_ode_matched':lines.append(r'\midrule')
        label='Latent ODE (matched)' if m=='latent_ode_matched' else LABELS[m]
        lines.append(label+' & '+' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}}',r'\end{table}',''])


def plot(predictions, batches, scale, destination, compact):
    genotypes = ['Col-0', 'MLB'] if compact else GENOTYPES
    fig, axes = plt.subplots(2, len(genotypes), figsize=(8, 6.1) if compact else (17, 7),
                            sharex=True, squeeze=False)
    for row, condition in enumerate(CONDITIONS):
        for col, genotype in enumerate(genotypes):
            ax = axes[row,col]; i = CASES.index((genotype, condition))
            if condition == '12L12D':
                for start in [12, 36, 60]: ax.axvspan(start, start+12, color='0.90', alpha=.65)
            for model in ORDER:
                ax.plot(HOURS, predictions[model][i]*scale, color=COLORS[model], ls=STYLES[model],
                    lw=2 if model=='phytoode' else 1.5, zorder=3 if model=='phytoode' else 2)
            batch = batches['test']; mask = batch['mask'][i].numpy().astype(bool)
            ax.scatter(HOURS[mask], batch['target'][i].numpy()[mask]*scale, color='black', s=25, zorder=6)
            ax.set_title(f'{genotype}, {condition}', fontsize=11)
            if row == 1: ax.set_xlabel('Elapsed time (h)')
            if col == 0: ax.set_ylabel('Hypocotyl length (mm)')
            ax.set_xticks([0,12,24,36,48,60,72] if compact else [0,24,48,72])
            ax.set_xlim(-2,74); ax.set_ylim(bottom=0); ax.grid(alpha=.15)
            ax.spines[['top','right']].set_visible(False)
    handles = [Line2D([0],[0], color=COLORS[m], ls=STYLES[m], lw=2, label=LABELS[m]) for m in ORDER]
    handles += [Line2D([0],[0], marker='o', color='black', ls='', label='Observed test mean')]
    fig.legend(handles=handles, loc='upper center', ncol=3 if compact else 4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0,0,1,.85 if compact else .87))
    for extension in ['pdf','png']:
        fig.savefig(destination.with_suffix('.'+extension), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    read = lambda p: json.loads(p.read_text())
    completed, selection, protocol = [read(OUT/name) for name in ['completed.json','selections.json','protocol.json']]
    assert completed['status']=='complete'
    assert completed['selection_sha256']==sha(OUT/'selections.json')
    assert all(sha(ROOT/p)==h for p,h in protocol['source_sha256'].items())
    old_protocol = read(OLD/'protocol.json')
    assert all(sha(ROOT/p)==h for p,h in old_protocol['source_sha256'].items())
    choice = read(OUT/'chosen_phytoode.json')
    assert selection['chosen_sha256']==sha(OUT/'chosen_phytoode.json')
    assert choice['selected']==min(choice['ranking'], key=lambda r:(r['val_mean'],r['candidate']))
    for candidate in choice['ranking']:
        vals = [read(ROOT/p/'result.json')['metrics']['val']['rmse'] for p in candidate['paths']]
        np.testing.assert_allclose(vals, candidate['per_seed_validation'], rtol=0, atol=0)
        assert sum(vals)/3==candidate['val_mean']
    originals = read(OLD/'selections.json')
    sources = {m: OLD for m in ORDER}
    sources['phytoode']=sources['latent_ode_matched']=OUT
    sources['latent_ode_initial_control']=OLD
    records = selection['runs'] + [r for r in originals['runs'] if r['protocol']=='replicate' and r['model'] not in ['phytoode','latent_ode_matched']]
    records += [r | dict(model='latent_ode_initial_control',evaluation_model='latent_ode_matched')
        for r in originals['runs'] if r['protocol']=='replicate' and r['model']=='latent_ode_matched']
    config, x, batches = load('replicate', splits=('train','val','test'))
    scale = config['height_scale_mm']; rows=[]; settings=[]; predictions={}; refinements=[]; audit_records=[]; losses=[]
    for record in records:
        model,seed = record['model'],record['seed']
        source=ROOT/record['path']; evaluated_path=sources[model]/'replicate/final'/record.get('evaluation_model',model)/f'seed{seed}'
        trained, evaluated = read(source/'result.json'), read(evaluated_path/'result.json')
        assert sha(source/'result.json')==record['result_sha256']==evaluated['source_result_sha256']
        assert trained['test_evaluations']==0 and evaluated['test_evaluations']==1
        assert evaluated['selection_sha256']==sha(sources[model]/'selections.json')
        saved=np.load(source/'predictions.npz'); scored=np.load(evaluated_path/'predictions.npz')
        assert 'test_target' not in saved.files
        prediction=saved['prediction']; np.testing.assert_array_equal(prediction, scored['prediction'])
        predictions.setdefault(model,[]).append(prediction)
        for split,batch in batches.items():
            metrics=score(prediction,batch,scale)
            for metric,value in metrics.items():
                np.testing.assert_allclose(value,evaluated['metrics'][split][metric],rtol=0,atol=0)
            rows.append(dict(protocol='replicate', model=model,seed=seed,split=split,
                **{key:value for key,value in metrics.items() if key!='per_curve_rmse'}))
        if trained.get('checkpoint_sha256'):
            assert sha(source/'checkpoint.pt')==trained['checkpoint_sha256']
            history=pd.read_csv(source/'history.csv')
            best=history[history.epoch.ge(200)].sort_values(['selection_rmse','epoch']).iloc[0]
            assert int(best.epoch)==trained['best_epoch']
        settings.append(dict(model=model, seed=seed, best_epoch=trained['best_epoch'],
            n_params=trained['n_params'], source=str(source.relative_to(ROOT)),config=trained['config']))
        audit_records.append(dict(model=model,seed=seed,source_result_sha256=sha(source/'result.json'),
            prediction_sha256=sha(source/'predictions.npz'),final_result_sha256=sha(evaluated_path/'result.json')))
        if model=='phytoode':
            net=build(choice['selected']['config'])
            checkpoint=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
            net.load_state_dict(checkpoint['state_dict']); net.eval()
            output=net(**x)
            ode,capacity=physics_losses(net,output,x)
            data_loss=curve_rmse(output['pred'],batches['train']['target'],batches['train']['mask'])
            settings_config=choice['selected']['config']
            losses.append(dict(seed=seed,data_loss=float(data_loss.detach()),
                ode_loss=float(ode.detach()),k_loss=float(capacity.detach()),
                weighted_ode_loss=float(ode.detach())*settings_config['lambda_ode'],
                weighted_k_loss=float(capacity.detach())*settings_config['lambda_k']))
            fine=inputs(np.arange(0.,72.1,1.5))
            with torch.no_grad():
                coarse=net(**x)['pred'].numpy()
                refined=net(**fine,encoder_env=x['env'],encoder_time=x['time'])['pred'].numpy()[:,::2]
            np.testing.assert_allclose(coarse,prediction,atol=1e-5,rtol=1e-5)
            difference=(coarse-refined)*scale
            refinements.append(dict(seed=seed,max_absolute_difference_mm=float(np.abs(difference).max()),
                rms_difference_mm=float(np.sqrt(np.mean(difference**2)))))
    for seed in [1,2,3]:
        full=read(ROOT/choice['selected']['paths'][seed-1]/'result.json')
        pure=read(OUT/'trials/latent_ode_matched'/f'seed{seed}'/'result.json')
        assert full['initial_state_sha256']==pure['initial_state_sha256']
        assert pure['config']==(choice['selected']['config'] | dict(model='latent_ode',lambda_ode=0.,lambda_k=0.))
        for key in ['lr','epochs']:
            assert full['config'][key]==pure['config'][key]
        assert pure['config']['lambda_ode']==pure['config']['lambda_k']==0
    REPORT.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows); frame.to_csv(REPORT/'per_seed_metrics.csv',index=False)
    aggregates=[]
    for (p,m,s),group in frame.groupby(['protocol','model','split'],sort=False):
        aggregate=dict(protocol=p,model=m,split=s,n_seeds=len(group))
        for metric in ['rmse','relative_error','individual_rmse']:
            aggregate[metric+'_mean']=float(group[metric].mean())
            aggregate[metric+'_sd']=float(group[metric].std(ddof=1)) if len(group)>1 else 0.
        aggregates.append(aggregate)
    comparison=pd.DataFrame(aggregates); comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(refinements).to_csv(REPORT/'integration_refinement.csv',index=False)
    pd.DataFrame(losses).to_csv(REPORT/'selected_loss_components.csv',index=False)
    atomic(REPORT/'selected_configs.json',dict(selected=choice['selected'],runs=settings))
    (REPORT/'table_replicate.tex').write_text(primary_table(comparison))
    predictions={m:np.mean(values,axis=0) for m,values in predictions.items()}
    plot(predictions,batches,scale,REPORT/'predictions_replicate',False)
    plot(predictions,batches,scale,ROOT/'paper/figures/hypocotyl_light_replicates',True)
    lines=['# Hypocotyl primary evaluation after expanded PhytoODE tuning','',
        'The same frozen replicate-group partitions are used throughout. This is an exploratory follow-up after the original test results had been inspected. The 80-candidate search and final choice used validation scores only. Baselines retain the initial tuning budgets; the matched no-physics control uses the final PhytoODE architecture and optimizer settings.','',
        '| Model | Train RMSE / relative error | Validation RMSE / relative error | Test RMSE / relative error |',
        '|---|---|---|---|']
    winners={s:comparison[comparison.split.eq(s)].sort_values('rmse_mean').iloc[0].model for s in ['train','val','test']}
    for m in REPORT_ORDER:
        cells=[]
        for s in ['train','val','test']:
            r=comparison[comparison.model.eq(m)&comparison.split.eq(s)].iloc[0]
            text=f'{r.rmse_mean:.3f} ± {r.rmse_sd:.3f} / {r.relative_error_mean:.2f} ± {r.relative_error_sd:.2f}%'
            if r.n_seeds==1: text=f'{r.rmse_mean:.3f} / {r.relative_error_mean:.2f}%'
            if m==winners[s]: text='**'+text+'**'
            cells.append(text)
        label='Latent ODE (matched architecture and training)' if m=='latent_ode_matched' else LABELS[m]
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', 'RMSE is in mm. Bold identifies unrounded minima among all rows; some differences are below the displayed precision. The final two rows are the current matched control and the original same-learning-rate control. Both physics coefficients are zero in every no-physics comparison. SD denotes variation across three training seeds, not biological uncertainty.', '',
        '## Validation-selected PhytoODE configuration','', '```json',json.dumps(choice['selected']['config'],indent=2),'```','',
        '## Before / after','',
        'The initial primary PhytoODE test result was 0.343717 mm / 5.931501%. The original independently learning-rate-selected latent ODE remains 0.336836 mm / 5.812771%. These previously inspected values did not determine the new candidate ranking or stopping rule.','']
    test=comparison[comparison.split.eq('test')].set_index('model')
    full,pure,matched=[test.loc[m] for m in ['phytoode','latent_ode','latent_ode_matched']]
    lines += [f'Unrounded test means: PhytoODE {full.rmse_mean:.8f} mm / {full.relative_error_mean:.6f}%; original latent ODE {pure.rmse_mean:.8f} mm / {pure.relative_error_mean:.6f}%; matched latent ODE {matched.rmse_mean:.8f} mm / {matched.relative_error_mean:.6f}%.', '',
        f'The gap from the original latent ODE is only {abs(pure.rmse_mean-full.rmse_mean):.8f} mm ({abs(100*(1-full.rmse_mean/pure.rmse_mean)):.3f}%), much smaller than the seed standard deviations: the methods are effectively tied in this comparison. Relative to the matched control, PhytoODE mean test RMSE is {100*(1-full.rmse_mean/matched.rmse_mean):.3f}% lower. Three seeds and a reused test partition do not establish statistical superiority.', '']
    initial=test.loc['latent_ode_initial_control']
    lines += [f'The earlier no-physics control at the initial architecture and learning rate 0.01 achieved {initial.rmse_mean:.8f} mm / {initial.relative_error_mean:.6f}%, which is lower than the newly tuned PhytoODE result. It remains a historical control, not a baseline selected using test error: its original validation mean was worse than the independently selected learning rate 0.003. The expanded search therefore did not establish superiority over all evaluated latent ODE configurations.', '']
    (REPORT/'results.md').write_text('\n'.join(lines).rstrip()+'\n')
    atomic(REPORT/'results_audit.json',dict(status='passed',selection_sha256=sha(OUT/'selections.json'),
        prior_test_inspected=True,new_full_training_fits=99,new_final_evaluations=6,
        audited_final_predictions=len(records),source_hashes_verified=True,all_metrics_recomputed=True,
        validation_ranking_verified=True,paired_initializations_identical=True,records=audit_records))
    print(comparison[comparison.split.eq('test')][['model','rmse_mean','relative_error_mean']].to_string(index=False))


if __name__=='__main__': main()
