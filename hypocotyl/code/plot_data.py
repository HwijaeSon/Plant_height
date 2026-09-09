"""Show the measured snapshot distributions without joining putative plant rows."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import ROOT, GENOTYPES, CONDITIONS


def main():
    frame=pd.read_csv(ROOT/'data/processed/observations.csv')
    rng=np.random.default_rng(20260909)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,5,figsize=(14,6),sharex=True,sharey=True)
    for r,condition in enumerate(CONDITIONS):
        for c,genotype in enumerate(GENOTYPES):
            ax=axes[r,c]
            part=frame[(frame.sheet==condition)&(frame.genotype==genotype)]
            if condition=='12L12D':
                for start in [12,36,60]:ax.axvspan(start,start+12,color='#d6d6d6',alpha=.5,zorder=0)
            ax.scatter(part.elapsed_hours+rng.uniform(-1.4,1.4,len(part)),part.length,
                       s=8,color='#777777',alpha=.4,linewidths=0)
            stats=part.groupby('elapsed_hours').length.agg(['mean','std','count'])
            ax.errorbar(stats.index,stats['mean'],yerr=stats['std'],fmt='o-',lw=1.5,ms=3.5,
                        color='#0072B2',capsize=2)
            ax.set_title(f'{genotype}, {condition}')
            ax.set_xticks([0,24,48,72]);ax.set_xlim(-3,75);ax.grid(alpha=.15)
            if c==0:ax.set_ylabel('Hypocotyl length (mm)')
            if r==1:ax.set_xlabel('Elapsed time (h)')
    fig.suptitle('Original measurements at 23°C: 1,818 observations, 5 genotypes, 2 light regimes',fontsize=12)
    fig.text(.5,.015,'Gray points: measured snapshots (horizontal jitter). Blue: group mean ± sample SD. Shading: dark intervals in 12L12D.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96))
    for ext in ['png','pdf']:fig.savefig(ROOT/f'reports/data_overview.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
