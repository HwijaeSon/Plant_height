"""Audit frozen evaluations and export tables/figures without model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

from data import CASES, HOURS, GENOTYPES, CONDITIONS, load, score, inputs
from models import build
from trial import atomic, sha

ROOT = Path(__file__).resolve().parents[2]
ORDER = ['phytoode', 'latent_ode', 'light_pinn', 'lstm', 'rf', 'light_logistic', 'logistic']
LABELS = dict(phytoode='PhytoODE', latent_ode='Latent ODE (no physics)',
    light_pinn='Light-PINN', lstm='LSTM-NN', rf='Random forest',
    light_logistic='Light-logistic ODE', logistic='Logistic ODE',
    latent_ode_matched='Latent ODE (matched LR)')
COLORS = dict(phytoode='#0072B2', latent_ode='#332288', light_pinn='#D55E00',
    lstm='#009E73', rf='#999999', light_logistic='#CC79A7', logistic='#E69F00')
STYLES = {m: '--' if m == 'latent_ode' else '-' for m in ORDER}


def read(path):
    return json.loads(path.read_text())


def table(frame, protocol):
    sub = frame[frame.protocol.eq(protocol) & frame.model.isin(ORDER)]
    winners = {s: sub[sub.split.eq(s)].sort_values('rmse_mean').iloc[0].model for s in ['train', 'val', 'test']}
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
        r'\caption{Hypocotyl ' + ('replicate-group holdout' if protocol == 'replicate' else 'unseen-time interpolation') +
        r' errors: RMSE (mm) / relative RMSE (\%). Bold marks the smallest mean in each split. Stochastic models use three seeds (mean $\pm$ sample SD); process ODEs are single deterministic fits. Scores compare population-mean curves, with individual-measurement errors reported separately.}',
        r'\label{tab:hypocotyl_' + protocol + '}', r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{lccc}', r'\toprule', r'Model & Train & Validation & Test \\', r'\midrule']
    for m in ORDER:
        cells = []
        for s in ['train', 'val', 'test']:
            row = sub[sub.model.eq(m) & sub.split.eq(s)].iloc[0]
            value = f'{row.rmse_mean:.3f} / {row.relative_error_mean:.2f}'
            if row.n_seeds > 1:
                value = f'{row.rmse_mean:.3f}\\pm{row.rmse_sd:.3f} / {row.relative_error_mean:.2f}\\pm{row.relative_error_sd:.2f}'
            if m == winners[s]: value = r'\mathbf{' + value + '}'
            cells.append('$' + value + '$')
        lines.append(LABELS[m] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']
    return '\n'.join(lines)


def plot_predictions(out, selection, protocol, destination, compact=False):
    config, _, batches = load(protocol, splits=('train', 'test'))
    scale = config['height_scale_mm']
    predictions = {}
    for m in ORDER:
        paths = sorted((out / protocol / 'final' / m).glob('seed*/predictions.npz'))
        predictions[m] = np.mean([np.load(p)['prediction'] * scale for p in paths], axis=0)
    genotypes = ['Col-0', 'MLB'] if compact else GENOTYPES
    fig, axes = plt.subplots(2, len(genotypes), figsize=(8, 6.1) if compact else (17, 7),
                             sharex=True, squeeze=False)
    for row, c in enumerate(CONDITIONS):
        for col, g in enumerate(genotypes):
            ax = axes[row, col]; i = CASES.index((g, c))
            if c == '12L12D':
                for t in [12, 36, 60]: ax.axvspan(t, t+12, color='0.90', alpha=.65, zorder=0)
            for m in ORDER:
                ax.plot(HOURS, predictions[m][i], color=COLORS[m], ls=STYLES[m],
                        lw=2 if m == 'phytoode' else 1.5, zorder=3 if m == 'phytoode' else 2)
            if protocol == 'time_holdout':
                b = batches['train']; mask = b['mask'][i].numpy().astype(bool)
                ax.scatter(HOURS[mask], b['target'][i].numpy()[mask]*scale, marker='o',
                    facecolors='white', edgecolors='0.4', s=33, zorder=5)
            b = batches['test']; mask = b['mask'][i].numpy().astype(bool)
            ax.scatter(HOURS[mask], b['target'][i].numpy()[mask]*scale, color='black', s=25, zorder=6)
            ax.set_title(f'{g}, {c}', fontsize=11)
            if row == 1: ax.set_xlabel('Elapsed time (h)')
            if col == 0: ax.set_ylabel('Hypocotyl length (mm)')
            ax.set_xticks([0, 12, 24, 36, 48, 60, 72] if compact else [0, 24, 48, 72])
            ax.set_xlim(-2, 74); ax.set_ylim(bottom=0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(alpha=.15)
    handles = [Line2D([0], [0], color=COLORS[m], ls=STYLES[m], lw=2, label=LABELS[m]) for m in ORDER]
    handles.append(Line2D([0], [0], marker='o', color='black', ls='', label='Observed test mean'))
    if protocol == 'time_holdout':
        handles.append(Line2D([0], [0], marker='o', markerfacecolor='white', color='0.4', ls='', label='Training mean'))
    fig.legend(handles=handles, loc='upper center', ncol=3 if compact else 4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .85 if compact else .87))
    for extension in ['pdf', 'png']:
        fig.savefig(destination.with_suffix('.'+extension), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'hypocotyl/results/light_growth_20260909')
    args = parser.parse_args(); out = args.output.resolve()
    completed = read(out/'completed.json'); selection = read(out/'selections.json'); protocol = read(out/'protocol.json')
    assert completed['status'] == 'complete'
    assert completed['selection_sha256'] == sha(out/'selections.json')
    assert all(sha(ROOT/p) == h for p, h in protocol['source_sha256'].items())
    extracted = read(ROOT/'hypocotyl/data/processed/extraction_validation.json')
    assert sha(ROOT/'hypocotyl/data/raw/hypocotyl_growth_20230403.xlsx') == extracted['source_sha256']
    manifest = read(ROOT/'hypocotyl/data/processed/splits/manifest.json')
    assert sha(ROOT/'hypocotyl/data/processed/observations.csv') == manifest['source_sha256']
    assert all(sha(ROOT/'hypocotyl'/p) == h for p,h in manifest['files_sha256'].items())
    report = ROOT/'hypocotyl/reports'; report.mkdir(exist_ok=True)
    rows, settings, paired, curves, refinements = [], [], [], [], []
    audited = set()
    for record in selection['runs']:
        p, m, seed = record['protocol'], record['model'], record['seed']
        source = ROOT/record['path']; final = out/p/'final'/m/f'seed{seed}'
        trained, evaluated = read(source/'result.json'), read(final/'result.json')
        assert sha(source/'result.json') == record['result_sha256'] == evaluated['source_result_sha256']
        assert trained['test_evaluations'] == 0 and evaluated['test_evaluations'] == 1
        assert evaluated['selection_sha256'] == sha(out/'selections.json')
        config, _, batches = load(p, splits=('train', 'val', 'test'))
        with np.load(source/'predictions.npz') as saved, np.load(final/'predictions.npz') as result:
            assert 'test_target' not in saved
            np.testing.assert_array_equal(saved['prediction'], result['prediction'])
            pred = result['prediction'].copy()
            for s, batch in batches.items():
                measured = score(pred, batch, config['height_scale_mm'])
                for metric in ['rmse', 'relative_error', 'mean_target', 'individual_rmse']:
                    np.testing.assert_allclose(measured[metric], evaluated['metrics'][s][metric], rtol=1e-12)
                rows.append(dict(protocol=p, model=m, seed=seed, split=s, **{k:v for k,v in measured.items() if k!='per_curve_rmse'}))
                for (g,c), error in zip(CASES, measured['per_curve_rmse']):
                    curves.append(dict(protocol=p, model=m, seed=seed, split=s, genotype=g, condition=c, rmse=error))
        if trained['checkpoint_sha256']:
            assert sha(source/'checkpoint.pt') == trained['checkpoint_sha256']
            history = pd.read_csv(source/'history.csv')
            assert len(history) == 75 and history.epoch.iloc[-1] == 1500
            eligible = history[history.epoch.ge(200)]
            assert int(eligible.loc[eligible.selection_rmse.idxmin(), 'epoch']) == trained['best_epoch']
        settings.append(dict(protocol=p, model=m, seed=seed, candidate=source.parent.name,
            best_epoch=trained['best_epoch'], n_params=trained['n_params'], **{k:v for k,v in trained['config'].items() if k!='model'}))
        audited.add(str(source))
    frame = pd.DataFrame(rows); frame.to_csv(report/'per_seed_metrics.csv', index=False)
    pd.DataFrame(settings).to_csv(report/'selected_configs.csv', index=False)
    pd.DataFrame(curves).to_csv(report/'per_curve_metrics.csv', index=False)
    metrics = ['rmse','relative_error','individual_rmse']
    comparison = frame.groupby(['protocol','model','split'],sort=False)[metrics].agg(['mean','std']).reset_index()
    comparison.columns = ['_'.join(c).strip('_').replace('_std','_sd') for c in comparison.columns]
    comparison['n_seeds'] = frame.groupby(['protocol','model','split'],sort=False).size().to_numpy()
    comparison = comparison.fillna(0)
    comparison.to_csv(report/'comparison.csv', index=False)
    # Recheck seed-one screening and the three-seed selection without test scores.
    finalists = read(out/'finalists.json')
    for p in protocol['protocols']:
        for m in ORDER:
            trials = [d for d in (out/p/'trials').iterdir() if read(d/'seed1/result.json')['config']['model']==m]
            expected = sorted(trials, key=lambda d:(read(d/'seed1/result.json')['metrics']['val']['rmse'],d.name))[:2]
            assert [d.name for d in expected] == finalists[p][m]
            ranked = selection['selected'][p][m]['ranking']
            assert ranked == sorted(ranked,key=lambda d:(d['val_mean'],d['candidate']))
            for candidate in ranked:
                values = [read(out/p/'trials'/candidate['candidate']/f'seed{s}'/'result.json')['metrics']['val']['rmse'] for s in ([1] if m in ['logistic','light_logistic'] else [1,2,3])]
                np.testing.assert_allclose(values,candidate['per_seed_validation'],rtol=1e-12)
                np.testing.assert_allclose(np.mean(values),candidate['val_mean'],rtol=1e-12)
        for seed in [1,2,3]:
            a = next(r for r in selection['runs'] if (r['protocol'],r['model'],r['seed'])==(p,'phytoode',seed))
            b = next(r for r in selection['runs'] if (r['protocol'],r['model'],r['seed'])==(p,'latent_ode_matched',seed))
            ra, rb = read(ROOT/a['path']/'result.json'), read(ROOT/b['path']/'result.json')
            assert ra['initial_state_sha256'] == rb['initial_state_sha256']
            assert ra['config']['lr'] == rb['config']['lr']
            assert rb['config']['lambda_ode'] == rb['config']['lambda_k'] == 0
            for s in ['train','val','test']:
                f = frame[frame.protocol.eq(p)&frame.seed.eq(seed)&frame.split.eq(s)].set_index('model')
                paired.append(dict(protocol=p,seed=seed,split=s,phytoode_rmse=f.loc['phytoode','rmse'],
                    matched_no_physics_rmse=f.loc['latent_ode_matched','rmse'],
                    improvement_percent=100*(1-f.loc['phytoode','rmse']/f.loc['latent_ode_matched','rmse'])))
            # Numerical grid refinement preserves the trained scenario encoder.
            model = build('phytoode'); model.load_state_dict(torch.load(ROOT/a['path']/'checkpoint.pt',map_location='cpu',weights_only=True)['state_dict']); model.eval()
            canonical = inputs(); finer = inputs(np.arange(0.,73.,1.5))
            with torch.no_grad():
                base = model(**canonical)['pred']
                refined = model(**finer,encoder_env=canonical['env'],encoder_time=canonical['time'])['pred'][:,::2]
            scale = load(p)[0]['height_scale_mm']
            diff = (base-refined).numpy()*scale
            refinements.append(dict(protocol=p,seed=seed,coarse_step_hours=3,fine_step_hours=1.5,
                max_absolute_difference_mm=float(np.max(np.abs(diff))),rms_difference_mm=float(np.sqrt(np.mean(diff**2)))))
        (report/f'table_{p}.tex').write_text(table(comparison,p))
        plot_predictions(out,selection,p,report/f'predictions_{p}')
    (ROOT/'paper/figures').mkdir(parents=True,exist_ok=True)
    # Historical time-holdout analysis is archived outside the manuscript.
    plot_predictions(out,selection,'time_holdout',report/'predictions_time_holdout_compact',compact=True)
    pd.DataFrame(paired).to_csv(report/'paired_loss_ablation.csv',index=False)
    pd.DataFrame(refinements).to_csv(report/'integration_refinement.csv',index=False)
    atomic(report/'results_audit.json',dict(status='passed',unique_selected_training_runs=len(audited),
        final_evaluations=len(selection['runs']),full_training_trials=completed['full_training_trials'],
        source_data_hashes_verified=len(protocol['source_sha256']),selection_sha256=sha(out/'selections.json'),
        exact_saved_predictions_reused=True,test_scoring_after_all_selections_frozen=True,
        all_paired_initializations_equal=True,selection_and_metrics_recomputed=True,
        integration_refinement=refinements))
    text = ['# Hypocotyl benchmark results','',
        'Cells report RMSE (mm) / relative RMSE (%). Seed variation is not biological uncertainty.',
        'Bold marks the smallest mean among the seven benchmark methods; the matched-LR row is a separate loss ablation.','']
    for p in protocol['protocols']:
        text += ['## '+p,'','| Model | Train | Validation | Test |','|---|---:|---:|---:|']
        winners = {s: comparison[comparison.protocol.eq(p)&comparison.split.eq(s)&comparison.model.isin(ORDER)].sort_values('rmse_mean').iloc[0].model for s in ['train','val','test']}
        for m in ORDER+['latent_ode_matched']:
            cells=[]
            for s in ['train','val','test']:
                row=comparison[comparison.protocol.eq(p)&comparison.model.eq(m)&comparison.split.eq(s)].iloc[0]
                cell=f'{row.rmse_mean:.3f} / {row.relative_error_mean:.2f}%'
                if row.n_seeds > 1: cell=f'{row.rmse_mean:.3f} ± {row.rmse_sd:.3f} / {row.relative_error_mean:.2f} ± {row.relative_error_sd:.2f}%'
                if m == winners[s]: cell='**'+cell+'**'
                cells.append(cell)
            text += ['| '+LABELS[m]+' | '+' | '.join(cells)+' |']
        text += ['']
    (report/'results.md').write_text('\n'.join(text).rstrip()+'\n')
    print(comparison[comparison.split.eq('test')].to_string(index=False))


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
