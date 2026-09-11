"""Compare light/no-light prefix-parameter PhytoODE with the frozen baselines."""
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
RESULTS = ROOT/'hypocotyl/results/prefix_parameters_no_light_20260911'
REPORT = ROOT/'hypocotyl/reports/prefix_parameters_no_light_20260911'
LIGHT = ROOT/'hypocotyl/results/prefix_parameters_20260911'
OLD = ROOT/'hypocotyl/results/prefix_forecast_20260911'


def tag(value):
    return f'{value:g}'.replace('.', 'p')


def summarize(results, output):
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((results/'test_evaluation_plan.json').read_text())
    completion = json.loads((results/'completion.json').read_text())
    assert not completion['failures']
    chosen = plan['coefficient_variants']['selected']
    fixed = plan['coefficient_variants']['fixed100']
    labels = dict(baseline.LABELS,
        phytoode_original='PhytoODE (previous genotype head)',
        phytoode_light='PhytoODE (+ light; prefix head; lambda=100)',
        phytoode=f'PhytoODE (no light; prefix head; selected lambda={chosen:g})',
        phytoode_fixed=f'PhytoODE (no light; prefix head; fixed lambda={fixed:g})')
    previous = pd.read_csv(ROOT/'hypocotyl/reports/prefix_parameters_20260911/comparison.csv')
    previous['model'] = previous.model.replace({'phytoode': 'phytoode_light'})
    previous['label'] = previous.model.map(labels)
    variants = {'phytoode': chosen}
    if chosen != fixed:
        variants['phytoode_fixed'] = fixed
    added, per_seed = [], []
    for model, value in variants.items():
        for drop in [0., .25, .5]:
            runs = []
            for seed in [1, 2, 3]:
                folder = results/f'evaluated/lambda_{tag(value)}/drop_{round(100*drop)}/seed{seed}'
                run = json.loads((folder/'result.json').read_text())
                cfg = run['config']
                assert cfg['lambda_ode'] == value and cfg['lambda_k'] == 0 and cfg['n_params'] == 1183
                lit = json.loads((LIGHT/f'selected/drop_{round(100*drop)}/seed{seed}/config.json').read_text())
                assert cfg['data']['removed_training_observation_ids'] == lit['data']['removed_training_observation_ids']
                assert cfg['height_scale_mm'] == lit['height_scale_mm']
                runs.append(run)
                for split in ['train', 'val', 'test']:
                    per_seed.append(dict(model=model, lambda_ode=value, additional_prefix_drop=drop,
                        seed=seed, split=split, **{k:v for k,v in run['metrics'][split].items() if k!='per_plant_rmse'}))
            for split in ['train', 'val', 'test']:
                row = dict(protocol='prefix_forecast_20260911', dataset='hypocotyl',
                    model=model, label=labels[model], additional_prefix_drop=drop, split=split, n_seeds=3)
                for metric in ['rmse', 'relative_error', 'pooled_rmse', 'mean_target']:
                    values = [run['metrics'][split][metric] for run in runs]
                    row[metric+'_mean'] = np.mean(values)
                    row[metric+'_sd'] = np.std(values, ddof=1)
                for metric in ['n_plants', 'n_observations']:
                    row[metric] = runs[0]['metrics'][split][metric]
                added.append(row)
    frame = pd.concat([previous, pd.DataFrame(added)], ignore_index=True)
    frame.to_csv(output/'comparison.csv', index=False)
    pd.DataFrame(per_seed).to_csv(output/'per_seed_metrics.csv', index=False)
    search = pd.read_csv(results/'validation_search.csv').sort_values('lambda_ode')
    search.to_csv(output/'validation_search.csv', index=False)
    order = ['logistic', 'rf', 'lstm', 'logistic_pinn', 'latent_ode', 'latent_ode_light',
             'phytoode_original', 'phytoode_light', *variants]
    focus = ['phytoode_light', *variants]
    frame[frame.model.isin(focus)].to_csv(output/'light_input_comparison.csv', index=False)
    md = ['# Prefix-conditioned logistic parameters without light inputs', '',
        'Only PhytoODE was retrained. Both the prefix encoder and latent vector field omit illumination. '
        'The r/K head receives genotype embedding, masked individual 0–36 h lengths and their presence masks. '
        'lambda_K=0 removes the capacity penalty; K still receives gradients through the logistic residual.', '',
        f'The three-seed mean natural-missingness 48-h validation RMSE selected **lambda_ODE={chosen:g}**. '
        f'The separately predefined fixed-reference coefficient is **{fixed:g}**, selected previously for the light-input model. '
        'Both coefficients are transferred unchanged across additional missingness; identical configurations are evaluated once.', '',
        f"Completed {completion['search_fits']} candidate and {completion['transfer_fits']} transfer fits. "
        'Training outputs contain training/validation scores only. All unique checkpoint hashes were frozen before test scoring. '
        'This is a follow-up on a dataset analyzed previously; this search ranks configurations by validation only.', '',
        'The fixed-coefficient comparison holds the parameter-head design, loss weights, schedule and masks fixed. '
        'Removing input channels changes the backbone dimensions (1231 to 1183 parameters) and its initialization. '
        'It is therefore a feature-variant comparison, not a matched-weight perturbation. '
        'The original six baselines and genotype-head PhytoODE were not retrained or retuned.', '',
        'Cells: mean per-plant RMSE (mm) / relative RMSE (%), mean ± sample SD across three seeds/masks. '
        'Relative RMSE is 100 times per-plant RMSE divided by the mean of scored individual lengths. '
        'Train: reconstruction of available 0–36 h inputs; validation: 48 h; test: 60/72 h. '
        'No observed target is averaged or imputed. Natural-missingness Logistic ODE is deterministic (one fit).', '']
    for drop in [0., .25, .5]:
        part = frame[frame.additional_prefix_drop.eq(drop)]
        md += [f'## Additional removal of available prefix observations: {drop:.0%}', '',
               '| Model | Train | Validation | Test |', '|---|---:|---:|---:|']
        for model in order:
            cells = []
            for split in ['train', 'val', 'test']:
                sub = part[part.split.eq(split)]
                row = sub[sub.model.eq(model)].iloc[0]
                cell = f'{row.rmse_mean:.3f} / {row.relative_error_mean:.2f}%'
                if row.n_seeds > 1:
                    cell = f'{row.rmse_mean:.3f} ± {row.rmse_sd:.3f} / {row.relative_error_mean:.2f} ± {row.relative_error_sd:.2f}%'
                if row.rmse_mean == sub.rmse_mean.min():
                    cell = '**'+cell+'**'
                cells.append(cell)
            md.append('| '+labels[model]+' | '+' | '.join(cells)+' |')
        md.append('')
    md += ['## Validation-only coefficient search', '', '| lambda_ODE | Validation RMSE (mm) |', '|---:|---:|']
    for row in search.itertuples():
        cell = f'{row.val_rmse_mean:.4f} ± {row.val_rmse_sd:.4f}'
        if row.lambda_ode == chosen:
            cell = '**'+cell+'**'
        md.append(f'| {row.lambda_ode:g} | {cell} |')
    (output/'comparison.md').write_text('\n'.join(md).rstrip()+'\n')
    return frame, search, chosen, fixed, order


def plots(results, report, output, frame, search, chosen, fixed, order):
    output.mkdir(parents=True, exist_ok=True)
    baseline.ORDER = [m for m in order if m != 'phytoode_fixed']
    baseline.LABELS = dict(baseline.LABELS,
        phytoode='PhytoODE (prefix r/K, no light)', phytoode_light='PhytoODE (prefix r/K, + light)',
        phytoode_original='PhytoODE (previous)')
    baseline.COLORS = dict(baseline.COLORS, phytoode_light='#56B4E9', phytoode_original='#586B7A')
    baseline.REPORT = report
    def directory(drop, model, seed):
        if model == 'phytoode':
            return results/f'evaluated/lambda_{tag(chosen)}/drop_{round(100*drop)}/seed{seed}'
        if model == 'phytoode_light':
            return LIGHT/f'selected/drop_{round(100*drop)}/seed{seed}'
        return OLD/f'drop_{round(100*drop)}'/('phytoode' if model=='phytoode_original' else model)/f'seed{seed}'
    def line(model):
        return dict(color=baseline.COLORS[model], lw=2.4 if model=='phytoode' else 1.5,
            ls='--' if model in ['phytoode_light', 'phytoode_original', 'latent_ode_light'] else '-',
            zorder=4 if model=='phytoode' else 2)
    baseline.directory = directory
    baseline.model_line = line
    baseline.make_figures(output)
    baseline.style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    light_search = pd.read_csv(LIGHT/'validation_search.csv').sort_values('lambda_ode')
    for source, label, color in [(light_search, 'With light', '#56B4E9'), (search, 'Without light', '#0072B2')]:
        ax.errorbar(source.lambda_ode, source.val_rmse_mean, yerr=source.val_rmse_sd,
                    marker='o', capsize=3, label=label+' (mean ± SD)', color=color)
    winner = search[search.lambda_ode.eq(chosen)].iloc[0]
    ax.scatter([chosen], [winner.val_rmse_mean], s=140, marker='*', color='#D55E00',
               label=f'No-light selection: {chosen:g}', zorder=5)
    ax.set(xscale='log', xlabel=r'$\lambda_{\mathrm{ODE}}$ ($\lambda_K=0$)',
           ylabel='48-h validation RMSE (mm)', title='Same coefficient grid; three seeds per candidate')
    ax.grid(alpha=.2);ax.legend(frameon=False);fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output/f'lambda_ode_validation_search.{ext}')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    focus = [('phytoode_light', 'With light, lambda=100', '#56B4E9', '--'),
             ('phytoode', f'Without light, selected lambda={chosen:g}', '#0072B2', '-')]
    if chosen != fixed:
        focus.append(('phytoode_fixed', 'Without light, fixed lambda=100', '#D55E00', ':'))
    for ax, metric, ylabel in zip(axes, ['rmse', 'relative_error'], ['Test RMSE (mm)', 'Test relative RMSE (%)']):
        for model, label, color, ls in focus:
            part = frame[frame.model.eq(model)&frame.split.eq('test')].sort_values('additional_prefix_drop')
            ax.errorbar([12.26,34.24,56.21], part[metric+'_mean'], yerr=part[metric+'_sd'],
                        marker='o', capsize=4, color=color, ls=ls, label=label)
        ax.set_xticks([12.26,34.24,56.21], ['12.3%\nNatural', '34.2%\n+25% removal', '56.2%\n+50% removal'])
        ax.set(xlabel='Missing fraction among four prefix slots', ylabel=ylabel)
        ax.margins(x=.15);ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', frameon=False, ncol=1)
    fig.subplots_adjust(left=.08, right=.99, bottom=.22, top=.78, wspace=.26)
    for ext in ['png', 'pdf']:
        fig.savefig(output/f'light_input_comparison.{ext}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--output', type=Path, default=REPORT)
    parser.add_argument('--figures', type=Path)
    args = parser.parse_args()
    frame, search, chosen, fixed, order = summarize(args.results, args.output)
    if args.figures:
        plots(args.results, args.output, args.figures, frame, search, chosen, fixed, order)
    print(frame[frame.model.str.startswith('phytoode')][
        ['model', 'additional_prefix_drop', 'split', 'rmse_mean', 'relative_error_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
