"""Measured-cell targets only: no replicate averaging or phenotype imputation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import CASES, HOURS, inputs

ROOT = Path(__file__).resolve().parents[1]


def load_observations(protocol, splits=("train", "val"), device="cpu"):
    folder = ROOT / "data/processed/splits" / protocol
    config = json.loads((folder / "config.json").read_text())
    batches = {}
    for split in splits:
        frame = pd.read_csv(folder / f"{split}.csv")
        assert frame.observation_id.is_unique and frame.length.notna().all()
        case = np.array([CASES.index((r.genotype, r.sheet)) for r in frame.itertuples()])
        time = (frame.elapsed_hours.to_numpy() / 3).astype(int)
        np.testing.assert_array_equal(HOURS[time], frame.elapsed_hours.to_numpy())
        batches[split] = dict(case=torch.as_tensor(case, device=device),
            time=torch.as_tensor(time, device=device),
            length=torch.tensor(frame.length.to_numpy()/config['height_scale_mm'], dtype=torch.float32, device=device),
            observations=frame)
    return config, inputs(device=device), batches


def observation_rmse(prediction, batch):
    """Each measured cell contributes one residual; absent cells contribute none."""
    residual = prediction[batch['case'], batch['time']] - batch['length']
    return residual.square().mean().clamp_min(1e-12).sqrt()


def score_observations(prediction, batch, scale):
    grid = np.asarray(prediction, dtype=float) * scale
    frame = batch['observations']
    case = batch['case'].detach().cpu().numpy()
    time = batch['time'].detach().cpu().numpy()
    residual = grid[case, time] - frame.length.to_numpy()
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(frame.length.mean())
    return dict(rmse=rmse, relative_error=100*rmse/denominator,
        mean_observed_length=denominator, n_observations=len(frame),
        n_case_time_groups=int(len(set(zip(case.tolist(), time.tolist())))),
        per_curve_rmse=[float(np.sqrt(np.mean(residual[case == i]**2))) for i in range(len(CASES))])


def save_targets(batches):
    return {f'{split}_{key}': batch[key].detach().cpu().numpy()
            for split, batch in batches.items() for key in ['case', 'time', 'length']}
