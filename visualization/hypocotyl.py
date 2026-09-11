"""Plot selected PhytoODE and baseline hypocotyl predictions."""
from pathlib import Path
import hashlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'hypocotyl/results/light_input_20260910'
PREVIOUS=ROOT/'hypocotyl/results/single_condition_20260909'
ORDER=['phytoode_light','latent_ode','logistic_pinn','lstm','rf','logistic']
NAMES=dict(phytoode_light='PhytoODE',latent_ode='Latent ODE (no physics)',logistic_pinn='Logistic-PINN',
    lstm='LSTM-NN',rf='Random forest',logistic='Logistic ODE')
COLORS=dict(phytoode_light='#0072B2',latent_ode='#332288',logistic_pinn='#D55E00',lstm='#009E73',rf='#999999',logistic='#E69F00')


def make_figure(output_dir=None):
    sys.path.insert(0,str(ROOT/'hypocotyl/code'))
    from single_condition_data import load,GENOTYPES,HOURS
    config,_,batches=load('replicate',splits=('test',)); scale=config['height_scale_mm']
    raw,means=batches['test']['observations'],batches['test']['means']
    predictions={}; sources=[]
    for model in ORDER:
        root=RESULTS if model=='phytoode_light' else PREVIOUS
        seeds=[1] if model=='logistic' else [1,2,3]
        values=[]
        for seed in seeds:
            path=root/'replicate/final'/model/f'seed{seed}'/'predictions.npz'
            with np.load(path) as a: values.append(a['prediction'].astype(float))
            sources.append(path)
        predictions[model]=np.stack(values).mean(axis=0)*scale
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,3,figsize=(9.5,6.1)); axes=axes.ravel()
    for i,(ax,g) in enumerate(zip(axes,GENOTYPES)):
        for start in [12,36,60]: ax.axvspan(start,start+12,color='0.90',alpha=.6,lw=0,zorder=0)
        for model in reversed(ORDER):
            ax.plot(HOURS,predictions[model][i],color=COLORS[model],ls='--' if model=='latent_ode' else '-',
                lw=2.1 if model=='phytoode_light' else 1.3,zorder=4 if model=='phytoode_light' else 2)
        points=raw[raw.genotype.eq(g)]
        jitter=points.observation_id.map(lambda s:(int(hashlib.sha256(s.encode()).hexdigest()[:8],16)/(2**32-1)-.5)*1.2)
        ax.scatter(points.elapsed_hours+jitter,points.length,color='0.5',alpha=.25,s=10,linewidths=0,zorder=1)
        group=means[means.genotype.eq(g)]
        ax.scatter(group.elapsed_hours,group.mean_length_mm,color='black',s=20,edgecolors='white',linewidths=.4,zorder=6)
        ax.set(title=g,xlabel='Elapsed time (h)',ylabel='Hypocotyl length (mm)',xlim=(-2,74))
        ax.set_xticks([0,12,24,36,48,60,72]); ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.12)
    handles=[Line2D([0],[0],color=COLORS[m],lw=2,ls='--' if m=='latent_ode' else '-',
        label='PhytoODE (ours)' if m=='phytoode_light' else NAMES[m]) for m in ORDER]
    handles += [Line2D([0],[0],marker='o',color='black',ls='',markersize=5,label='Observed test mean'),
        Line2D([0],[0],marker='o',color='0.65',ls='',markersize=4,label='Individual test length'),
        Patch(facecolor='0.90',alpha=.6,label='Dark interval (12 h)')]
    axes[-1].axis('off');axes[-1].legend(handles=handles,loc='center',frameon=False,fontsize=9.5)
    fig.tight_layout()
    dest=(Path(output_dir) if output_dir is not None else ROOT/'outputs/figures')/'hypocotyl_light_input'
    dest.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(dest.with_suffix('.'+ext),dpi=240,bbox_inches='tight')
    plt.close(fig)
    return sources
