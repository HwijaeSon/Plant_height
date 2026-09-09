"""Genotype and elapsed time only; missing times have no target entry."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/single_condition_20260909'
GENOTYPES=['Col-0','hy5','MLB','EMS57','phyAB']
HOURS=np.arange(0.,73.,3.)


def inputs(hours=HOURS,device='cpu'):
    hours=np.asarray(hours,dtype=float)
    assert hours[0]==0 and hours[-1]==72 and np.allclose(np.diff(hours),hours[1]-hours[0])
    return dict(g_idx=torch.arange(5,device=device),
                time=torch.tensor(hours/72.,dtype=torch.float32,device=device))


def load(protocol,splits=('train','val'),device='cpu'):
    folder=DATA/protocol
    config=json.loads((folder/'config.json').read_text()); batches={}
    for split in splits:
        raw=pd.read_csv(folder/f'{split}.csv'); means=pd.read_csv(folder/f'{split}_means.csv')
        assert raw.sheet.eq('12L12D').all() and means.mean_length_mm.notna().all()
        case=np.array([GENOTYPES.index(g) for g in means.genotype])
        index=(means.elapsed_hours.to_numpy()/3).astype(int)
        np.testing.assert_array_equal(HOURS[index],means.elapsed_hours.to_numpy())
        batches[split]=dict(case=torch.tensor(case,device=device),time=torch.tensor(index,device=device),
            target=torch.tensor(means.mean_length_mm.to_numpy()/config['height_scale_mm'],dtype=torch.float32,device=device),
            observations=raw,means=means)
    return config,inputs(device=device),batches


def curve_rmse(prediction,batch):
    residual=prediction[batch['case'],batch['time']]-batch['target']
    sums=torch.zeros(5,device=residual.device,dtype=residual.dtype).scatter_add(0,batch['case'],residual.square())
    counts=torch.bincount(batch['case'],minlength=5)
    assert bool((counts>0).all())
    return (sums/counts).clamp_min(1e-12).sqrt().mean()


def score(prediction,batch,scale):
    pred=np.asarray(prediction,dtype=float)*scale
    means=batch['means']; case=batch['case'].cpu().numpy(); index=batch['time'].cpu().numpy()
    residual=pred[case,index]-means.mean_length_mm.to_numpy()
    per_curve=[float(np.sqrt(np.mean(residual[case==i]**2))) for i in range(5)]
    rmse=float(np.mean(per_curve)); denominator=float(means.mean_length_mm.mean())
    raw=batch['observations']; raw_case=np.array([GENOTYPES.index(g) for g in raw.genotype])
    raw_time=(raw.elapsed_hours.to_numpy()/3).astype(int)
    return dict(rmse=rmse,relative_error=100*rmse/denominator,mean_target=denominator,
        n_curves=5,n_scored_group_means=len(means),n_observations=len(raw),
        individual_rmse=float(np.sqrt(np.mean((pred[raw_case,raw_time]-raw.length.to_numpy())**2))),
        per_curve_rmse=per_curve)


def save_targets(batches):
    return {f'{split}_{key}':batch[key].detach().cpu().numpy()
            for split,batch in batches.items() for key in ['case','time','target']}
