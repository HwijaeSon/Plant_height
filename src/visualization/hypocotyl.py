"""Update manuscript hypocotyl figures from saved predictions, without retraining."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from layout import artifact
HERE = ROOT / 'outputs'
RELEASE = ROOT
PREPARED = ROOT / 'data/hypocotyl/processed'
ORDER = ['logistic', 'rf', 'lstm', 'logistic_pinn', 'no_ode_loss', 'phytoode2']
LABELS = dict(logistic='Logi-ODE (reference)', rf='RF (reference)', lstm='LSTM-NN (reference)',
              logistic_pinn='Logistic-PINN (reference)', no_ode_loss='Latent ODE (unregularized)', phytoode2='PhytoODE')
COLORS = dict(logistic='#E69F00', rf='#999999', lstm='#009E73', logistic_pinn='#D55E00',
              no_ode_loss='#A35D96', phytoode2='#0072B2')
GENOTYPES = ['Col-0', 'hy5', 'MLB', 'phyAB']
TITLES = {'Col-0': 'Col-0', 'MLB': 'MLB', 'hy5': r'$hy5$', 'phyAB': r'$phyAB$'}


def style(model):
    reference = model not in ['phytoode2', 'no_ode_loss']
    return dict(color=COLORS[model], lw=1.35 if reference else 2.1,
                ls='--' if reference else '-', alpha=.8 if reference else 1,
                zorder=2 if reference else 4)


def comparison():
    return pd.read_csv(ROOT / 'results/comparison_hypocotyl.csv')


def save(fig, stem):
    for extension in ['pdf', 'png']:
        fig.savefig(HERE / f'{stem}.{extension}', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def run():
    HERE.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    frame = comparison()
    plants = pd.read_csv(PREPARED / 'plants.csv')
    observations = pd.read_csv(PREPARED / 'observations.csv')
    early_counts = observations[observations.elapsed_hours.le(36)].groupby('plant_index').size()
    selected = []
    for genotype in GENOTYPES:
        candidates = plants[plants.genotype.eq(genotype)].copy()
        candidates['early_n'] = candidates.plant_index.map(early_counts)
        incomplete = candidates[candidates.early_n.lt(4)]
        if len(incomplete):
            candidates = incomplete[incomplete.early_n.eq(incomplete.early_n.max())]
        selected.append(int(candidates.sort_values('source_row').iloc[0].plant_index))
    (HERE / 'figure_examples.json').write_text(json.dumps(dict(
        selection_rule='Unchanged: lowest source row among the greatest incomplete early-observation coverage; no future data used.',
        plants=plants[plants.plant_index.isin(selected)].to_dict('records')), indent=2) + '\n')
    predictions = {}
    target_check = None
    for model in ORDER:
        values = []
        seeds = [1] if model == 'logistic' else [1, 2, 3] if model in ORDER[:4] else [101, 102, 103, 104, 105]
        for seed in seeds:
            key = {'phytoode2':'phytoode', 'no_ode_loss':'latent_ode'}.get(model, model)
            result = json.loads(artifact('hypocotyl', key, seed, 'result.json').read_text())
            arrays = np.load(artifact('hypocotyl', key, seed, 'predictions.npz'))
            np.testing.assert_array_equal(arrays['plant_ids'], plants.plant_id.to_numpy())
            current = tuple(arrays[k] for k in ['prefix_y', 'prefix_mask', 'test_target', 'test_mask'])
            if target_check is None:
                target_check = current
            for a, b in zip(current, target_check):
                np.testing.assert_array_equal(a, b)
            values.append(arrays['prediction'] * result['config']['height_scale_mm'])
        predictions[model] = np.mean(values, axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    for ax, idx in zip(axes.flat, selected):
        plant = plants.iloc[idx]
        record = observations[observations.plant_index.eq(idx)]
        for start in [0, 24, 48]:
            ax.axvspan(start, start+12, color='#e7ba53', alpha=.11, zorder=0)
        ax.axvline(36, color='#555555', ls=':', lw=1)
        for model in ORDER:
            ax.plot(np.arange(0, 73, 3), predictions[model][idx], **style(model))
        for split, marker, face in [('train', 'o', 'black'), ('val', 's', 'white'), ('test', '^', 'black')]:
            part = record[record.split.eq(split)]
            ax.scatter(part.elapsed_hours, part.length_mm, s=34, marker=marker,
                       facecolors=face, edgecolors='black', linewidths=1, zorder=6)
        ax.set(title=TITLES[plant.genotype]+f' · plant row {plant.source_row}',
               xlabel='Elapsed time (h)', ylabel='Hypocotyl length (mm)', xlim=(-2, 74))
        ax.set_xticks([0, 12, 24, 36, 48, 60, 72]); ax.grid(axis='y', alpha=.15)
    handles = [Line2D([], [], label=LABELS[m], **style(m)) for m in ORDER[::-1]]
    handles += [Line2D([], [], ls='', marker=m, color='black', markerfacecolor=f, label=l) for m, f, l in [
        ('o', 'black', 'Input / train (0–36 h)'), ('s', 'white', 'Validation (48 h)'), ('^', 'black', 'Test (60, 72 h)')]]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(.5, .0))
    fig.subplots_adjust(left=.08, right=.98, bottom=.20, top=.96, hspace=.35, wspace=.25)
    save(fig, 'hypocotyl_prefix_forecast')
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    missing = np.array([1-500/564, 1-375/564, 1-250/564])*100
    ticks = [f'{v:.1f}%\n'+label for v, label in zip(missing, ['Natural', '+25% removal', '+50% removal'])]
    for ax, metric, ylabel in zip(axes, ['rmse', 'relative_error'], ['Test RMSE (mm)', 'Test relative RMSE (%)']):
        for model in ORDER:
            part = frame[frame.model.eq(model)&frame.split.eq('test')].sort_values('additional_prefix_drop')
            ax.errorbar(missing, part[metric+'_mean'], yerr=part[metric+'_sd'], marker='o', ms=4,
                        capsize=3, label=LABELS[model], **style(model))
        ax.set_xticks(missing, ticks); ax.set(xlabel='Missing fraction of initial measurements', ylabel=ylabel)
        ax.margins(x=.12); ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(.5, 1), ncol=3, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=.07, right=.99, bottom=.2, top=.79, wspace=.23)
    save(fig, 'hypocotyl_missingness')

    parameters = pd.read_csv(ROOT / 'results/light_dark_parameters.csv')
    parameters.to_csv(HERE / 'light_dark_parameters.csv', index=False)
    order = ['Col-0', 'MLB', 'hy5', 'phyAB']; x = np.arange(4)
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.8))
    for metric, label, color, marker in [('r_light', 'Light', '#c78b16', 'o'), ('r_dark', 'Dark', '#45628c', 's')]:
        values = parameters.groupby('genotype')[metric].agg(['mean', 'std']).reindex(order)
        axes[0].errorbar(x, values['mean'], yerr=values['std'], marker=marker, color=color, lw=1.6, capsize=3, label=label)
    axes[0].legend(frameon=False, fontsize=9); axes[0].set_ylim(0, .105)
    axes[0].set(title='A  Phase-specific rates', ylabel=r'Rate coefficient (h$^{-1}$)')
    seed_colors = ['#3979b7', '#dd8724', '#8660a8', '#399d80', '#cc6472']
    for ax, metric, title, ylabel, ref in [
        (axes[1], 'a', 'B  Genotype contrast', r'Contrast coefficient $a_g$', 0.),
        (axes[2], 'phase_ratio', 'C  Relative phase modulation', 'Light / dark rate ratio', 1.)]:
        values = parameters.groupby('genotype')[metric].agg(['mean', 'std']).reindex(order)
        ax.errorbar(x, values['mean'], yerr=values['std'], marker='D', color='#263b49', lw=1.6, capsize=3)
        for j, seed in enumerate([101, 102, 103, 104, 105]):
            vals = parameters[parameters.seed.eq(seed)].set_index('genotype').reindex(order)[metric]
            ax.scatter(x+(j-2)*.035, vals, color=seed_colors[j], s=24, zorder=4, label=f'Seed {seed}')
        ax.axhline(ref, color='#8e979d', ls='--', lw=.9)
        ax.set(title=title, ylabel=ylabel)
    axes[1].set_ylim(-.025, .46); axes[2].set_ylim(.95, 2.45)
    for ax in axes:
        ax.set_xticks(x, [TITLES[g] for g in order]); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.065, right=.99, top=.90, bottom=.22, wspace=.35)
    save(fig, 'hypocotyl_light_dark_parameters')
    print('Updated two hypocotyl figures and added the PhytoODE parameter figure.')


if __name__ == '__main__':
    run()
