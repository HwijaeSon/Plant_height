"""Population curves assembled from snapshot replicates, never row trajectories."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
GENOTYPES = ["Col-0", "hy5", "MLB", "EMS57", "phyAB"]
CONDITIONS = ["12L12D", "cR"]
CASES = [(g, c) for g in GENOTYPES for c in CONDITIONS]
HOURS = np.arange(0., 73., 3.)


def light(hours, condition):
    hours = np.asarray(hours)
    return np.ones_like(hours, dtype=float) if condition == "cR" else ((hours % 24) < 12).astype(float)


def light_exposure(hours, condition):
    hours = np.asarray(hours)
    return hours if condition == "cR" else np.floor(hours / 24) * 12 + np.minimum(hours % 24, 12)


def inputs(hours=HOURS, device="cpu"):
    hours = np.asarray(hours)
    # Duty cycle is known from the specified complete lighting scenario.
    environment = np.stack([np.stack([light(hours, c), np.full_like(hours, .5 if c == "12L12D" else 1.)], axis=-1)
                            for g, c in CASES])
    return dict(g_idx=torch.tensor([GENOTYPES.index(g) for g, c in CASES], device=device),
        env=torch.tensor(environment, dtype=torch.float32, device=device),
        time=torch.tensor(hours / 72., dtype=torch.float32, device=device))


def load(protocol, splits=("train", "val"), device="cpu"):
    folder = ROOT / "data/processed/splits" / protocol
    config = json.loads((folder / "config.json").read_text())
    batches = {}
    for split in splits:
        frame = pd.read_csv(folder / f"{split}.csv")
        target, mask, count = [np.zeros((len(CASES), len(HOURS))) for _ in range(3)]
        for (condition, genotype, hour), group in frame.groupby(["sheet", "genotype", "elapsed_hours"]):
            i, j = CASES.index((genotype, condition)), int(hour / 3)
            target[i, j] = float(group.length.mean()) / config["height_scale_mm"]
            mask[i, j] = 1.
            count[i, j] = len(group)
        batches[split] = dict(target=torch.tensor(target, dtype=torch.float32, device=device),
            mask=torch.tensor(mask, dtype=torch.float32, device=device), counts=count, observations=frame)
    return config, inputs(device=device), batches


def curve_rmse(prediction, target, mask):
    return (((prediction - target).square() * mask).sum(-1) / mask.sum(-1)).clamp_min(1e-12).sqrt().mean()


def score(prediction, batch, scale):
    pred = np.asarray(prediction, dtype=float) * scale
    target = batch["target"].detach().cpu().numpy().astype(float) * scale
    mask = batch["mask"].detach().cpu().numpy().astype(bool)
    per_curve = np.sqrt(((pred-target)**2 * mask).sum(1) / mask.sum(1))
    denominator = float(target[mask].mean())
    rows = batch["observations"]
    obs_pred = np.array([pred[CASES.index((r.genotype, r.sheet)), int(r.elapsed_hours/3)] for r in rows.itertuples()])
    return dict(rmse=float(per_curve.mean()), relative_error=float(per_curve.mean()/denominator*100),
        mean_target=denominator, n_curves=len(CASES), n_scored_group_means=int(mask.sum()),
        n_observations=len(rows), individual_rmse=float(np.sqrt(np.mean((obs_pred-rows.length.to_numpy())**2))),
        per_curve_rmse=per_curve.tolist())
