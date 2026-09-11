"""Report the prefix-parameter PhytoODE search against the frozen baseline runs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import report_forecast as baseline

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / 'hypocotyl/results/prefix_parameters_20260911'
DEFAULT_REPORT = ROOT / 'hypocotyl/reports/prefix_parameters_20260911'
OLD_RESULTS = ROOT / 'hypocotyl/results/prefix_forecast_20260911'
ORDER = ['logistic', 'rf', 'lstm', 'logistic_pinn', 'latent_ode', 'latent_ode_light', 'phytoode_original', 'phytoode']
LABELS = dict(baseline.LABELS, phytoode_original='PhytoODE (previous genotype head)',
              phytoode='PhytoODE (prefix head; lambda_K=0)')


def summarize(results, output):
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((results/'lambda_selection.json').read_text())
    completion = json.loads((results/'completion.json').read_text())
    assert completion['selected_test_evaluations'] == 9 and completion['failures'] == []
    previous = pd.read_csv(ROOT/'hypocotyl/reports/prefix_forecast_20260911/comparison.csv')
    previous['model'] = previous.model.replace({'phytoode': 'phytoode_original'})
    previous['label'] = previous.model.map(LABELS)
    added, seed_rows = [], []
    for drop in [0., .25, .5]:
        fitted = []
        for seed in [1, 2, 3]:
            folder = results/f'selected/drop_{round(drop*100)}/seed{seed}'
            run = json.loads((folder/'result.json').read_text())
            assert run['status'] == 'complete' and run['config']['lambda_k'] == 0
            assert run['config']['lambda_ode'] == selection['selected_lambda_ode']
            old = json.loads((OLD_RESULTS/f'drop_{round(drop*100)}/phytoode/seed{seed}/config.json').read_text())
            assert run['config']['data']['removed_training_observation_ids'] == old['data']['removed_training_observation_ids']
            assert run['config']['height_scale_mm'] == old['height_scale_mm']
            fitted.append(run)
            for split in ['train', 'val', 'test']:
                seed_rows.append(dict(additional_prefix_drop=drop, seed=seed, split=split,
                    **{k:v for k,v in run['metrics'][split].items() if k!='per_plant_rmse'}))
        for split in ['train', 'val', 'test']:
            row = dict(protocol='prefix_forecast_20260911', dataset='hypocotyl', additional_prefix_drop=drop,
                       model='phytoode', label=LABELS['phytoode'], split=split, n_seeds=3)
            for metric in ['rmse', 'relative_error', 'pooled_rmse', 'mean_target']:
                values = [run['metrics'][split][metric] for run in fitted]
                row[metric+'_mean'] = np.mean(values)
                row[metric+'_sd'] = np.std(values, ddof=1)
            for metric in ['n_plants', 'n_observations']:
                row[metric] = fitted[0]['metrics'][split][metric]
            added.append(row)
    frame = pd.concat([previous, pd.DataFrame(added)], ignore_index=True)
    frame.to_csv(output/'comparison.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(output/'selected_per_seed_metrics.csv', index=False)
    search = pd.read_csv(results/'validation_search.csv').sort_values('lambda_ode')
    search.to_csv(output/'validation_search.csv', index=False)
    markdown = ['# PhytoODE with an individual prefix-conditioned parameter head', '',
        'The r/K head receives genotype embedding, four observed prefix lengths and four presence masks. '
        'The finite-window capacity loss is disabled (lambda_K=0). The remaining architecture and training schedule are unchanged.', '',
        f"Selected lambda_ODE: **{selection['selected_lambda_ode']:g}**, ranked by three-seed mean 48-h validation RMSE under natural missingness.",
        'This coefficient is transferred unchanged to both additional-missingness conditions. '
        'All 30 search fits and six transfer fits save training/validation results only. '
        'The coefficient and all nine checkpoint hashes are frozen before test evaluation.', '',
        'Only PhytoODE was retrained. Previous baseline and genotype-head PhytoODE results are reused unchanged. '
        'This follow-up combines a new parameter head, removal of the capacity penalty, and coefficient search; '
        'it does not isolate the individual effects of these changes. The same dataset had been analyzed previously.', '',
        'Cells report mean per-plant RMSE (mm) / relative RMSE (%), mean ± sample SD. '
        'Training error measures reconstruction of the observed 0–36 h inputs; validation is at 48 h and test at 60/72 h.', '']
    for drop in [0., .25, .5]:
        part = frame[frame.additional_prefix_drop.eq(drop)]
        markdown += [f'## Additional removal of available prefix observations: {drop:.0%}', '',
            '| Model | Train | Validation | Test |', '|---|---:|---:|---:|']
        for model in ORDER:
            cells = []
            for split in ['train', 'val', 'test']:
                rows = part[part.split.eq(split)]
                row = rows[rows.model.eq(model)].iloc[0]
                if row.n_seeds > 1:
                    text = f'{row.rmse_mean:.3f} ± {row.rmse_sd:.3f} / {row.relative_error_mean:.2f} ± {row.relative_error_sd:.2f}%'
                else:
                    text = f'{row.rmse_mean:.3f} / {row.relative_error_mean:.2f}%'
                if row.rmse_mean == rows.rmse_mean.min():
                    text = '**'+text+'**'
                cells.append(text)
            markdown.append('| '+LABELS[model]+' | '+' | '.join(cells)+' |')
        markdown.append('')
    markdown += ['## Validation-only coefficient search', '', '| lambda_ODE | Validation RMSE (mm) |', '|---:|---:|']
    for row in search.itertuples():
        text = f'{row.val_rmse_mean:.4f} ± {row.val_rmse_sd:.4f}'
        if row.lambda_ode == selection['selected_lambda_ode']:
            text = '**'+text+'**'
        markdown.append(f'| {row.lambda_ode:g} | {text} |')
    (output/'comparison.md').write_text('\n'.join(markdown).rstrip()+'\n')
    return frame, search, selection


def plots(results, report, output, search, selection):
    output.mkdir(parents=True, exist_ok=True)
    baseline.ORDER = ORDER
    baseline.LABELS = dict(LABELS, phytoode='PhytoODE (prefix r/K)', phytoode_original='PhytoODE (previous)')
    baseline.COLORS = dict(baseline.COLORS, phytoode_original='#56B4E9')
    baseline.REPORT = report
    def directory(drop, model, seed):
        if model == 'phytoode':
            return results/f'selected/drop_{round(drop*100)}/seed{seed}'
        return OLD_RESULTS/f'drop_{round(drop*100)}'/('phytoode' if model=='phytoode_original' else model)/f'seed{seed}'
    def model_line(model):
        return dict(color=baseline.COLORS[model], lw=2.4 if model=='phytoode' else 1.6,
                    ls='--' if model in ['latent_ode_light', 'phytoode_original'] else '-',
                    zorder=4 if model=='phytoode' else 2)
    baseline.directory = directory
    baseline.model_line = model_line
    baseline.make_figures(output)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.errorbar(search.lambda_ode, search.val_rmse_mean, yerr=search.val_rmse_sd,
                marker='o', capsize=3, color='#0072B2', label='Three-seed mean ± SD')
    winner = search[search.lambda_ode.eq(selection['selected_lambda_ode'])].iloc[0]
    ax.scatter([winner.lambda_ode], [winner.val_rmse_mean], s=150, marker='*', color='#D55E00',
                zorder=5, label=f'Selected: {winner.lambda_ode:g}')
    ax.set(xscale='log', xlabel=r'$\lambda_{\mathrm{ODE}}$ ($\lambda_K=0$)',
           ylabel='48-h validation RMSE (mm)', title='Natural-missingness validation search')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output/f'lambda_ode_validation_search.{ext}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    p.add_argument('--output', type=Path, default=DEFAULT_REPORT)
    p.add_argument('--figures', type=Path)
    args = p.parse_args()
    frame, search, selection = summarize(args.results, args.output)
    if args.figures:
        plots(args.results, args.output, args.figures, search, selection)
    print(frame[frame.model.eq('phytoode')][['additional_prefix_drop','split','rmse_mean','relative_error_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
