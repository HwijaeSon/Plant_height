"""Matched loss ablations using the manuscript's frozen dataset-specific settings.

Each invocation runs one dataset/variant/seed. Original model/data modules are
imported without modification. Test predictions are generated only after the
validation-selected checkpoint and its hash have been written to disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("full", "no_ode_residual", "no_biological_loss")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def state_digest(model):
    hasher = hashlib.sha256()
    for key, value in model.state_dict().items():
        hasher.update(key.encode())
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def setup(dataset, device):
    sys.path.insert(0, str(ROOT / "src" / dataset))
    if dataset == "wheat":
        import data
        import model as core
        cfg = json.loads((ROOT / "src/configs/wheat.json").read_text())
        ds = data.make_dataset(
            ROOT / cfg["data"], ROOT / cfg["kinship"], split=cfg["split"],
            start_day=cfg["start_day"], n_t=285-cfg["start_day"],
            env_cols=tuple(cfg["env_cols"]),
            fill_in_na_at_start=not cfg["no_fill_na_start"],
            add_gdd=cfg["add_gdd"], gdd_base=cfg["gdd_base"],
            val_year=cfg["val_year"], test_year=cfg["test_year"], genotypes=cfg["genotypes"],
        )
        model_args = {k: cfg[k] for k in (
            "latent_dim", "g_embed_dim", "ode_hidden", "ode_layers", "dec_hidden",
            "enc_hidden", "init_mode", "time_mode", "genetic_encoding", "n_substeps", "variational")}
        model_args.update(kinship=ds.kinship, days_per_tau=float(ds.train.n_t-1))
        config = dict(model=model_args, epochs=cfg["epochs"], lr=cfg["lr"],
            weight_decay=cfg["l2"], physic=cfg["weight_physic"], ymax=cfg["weight_ymax"],
            eval_every=cfg["eval_every"], min_epoch=cfg["min_epoch"],
            val_smooth=cfg["val_smooth"], permute_training=True,
            scale=1., unit="m", expected_n_params=1655)
        loss_function = core.masked_rmse_loss
    elif dataset == "maize":
        import data
        import run_experiment as baseline
        core = baseline.LATENT
        cfg = json.loads((ROOT / "src/configs/maize.json").read_text())["config"]
        ds = data.make_dataset(split="chronological")
        model_args = {k: cfg[k] for k in (
            "latent_dim", "g_embed_dim", "ode_hidden", "ode_layers", "dec_hidden", "enc_hidden", "time_mode")}
        model_args.update(init_mode="env_encoder", genetic_encoding="one_hot", n_substeps=1,
                          days_per_tau=95.)
        config = dict(model=model_args, epochs=cfg["epochs"], lr=cfg["lr"],
            weight_decay=cfg["weight_decay"], physic=cfg["physic"], ymax=cfg["ymax"],
            eval_every=10, min_epoch=0, val_smooth=1, permute_training=False,
            scale=ds.y_scale, unit="relative UAV height", expected_n_params=9003)
        loss_function = baseline.REFERENCE.curve_rmse
    else:
        import run_experiment as original
        core = original.load_current_latent_module()
        ds = original.load_dataset(original.DEFAULT_DATA)
        model_args = dict(latent_dim=16, g_embed_dim=4, ode_hidden=32, ode_layers=2,
            dec_hidden=32, enc_hidden=16, init_mode="env_encoder", time_mode="calendar",
            genetic_encoding="one_hot", variational=False, n_substeps=1, days_per_tau=22.)
        config = dict(model=model_args, epochs=1500, lr=.005, weight_decay=1e-4,
            physic=2., ymax=.1, eval_every=10, min_epoch=300, val_smooth=10,
            permute_training=False, scale=100., unit="cm", expected_n_params=4607)
        loss_function = core.masked_rmse_loss

    raw, tensors = {}, {}
    for split in ("train", "train_eval", "val", "test"):
        batch = getattr(ds, split)
        if dataset == "arabidopsis":
            values = dict(y=batch.target.numpy(), mask=batch.mask.numpy(),
                env=batch.environment.numpy()[..., None],
                s=np.zeros_like(batch.environment.numpy()), ds=np.zeros_like(batch.environment.numpy()),
                g_idx=batch.genotype_index.numpy())
            meta = dict(genotype=np.asarray(batch.genotype), condition=np.asarray(batch.condition))
        else:
            values = {key: getattr(batch, key) for key in ("y", "mask", "env", "s", "ds", "g_idx")}
            meta = dict(genotype=np.asarray(ds.genotypes)[batch.g_idx], year=batch.year)
        raw[split] = values | meta
        # Keep test targets on the CPU until training/checkpoint selection ends.
        if split != "test":
            tensors[split] = {key: torch.as_tensor(value,
                dtype=torch.long if key == "g_idx" else torch.float32, device=device)
                for key, value in values.items()}
    model_args.update(n_genotypes=len(ds.genotypes), env_dim=1, use_physics=True)
    config["model"] = {key: value for key, value in model_args.items() if key != "kinship"}
    config["n_train"] = len(raw["train"]["y"])
    config["n_times"] = raw["train"]["y"].shape[1]
    return core, loss_function, model_args, config, raw, tensors


def forward(model, batch):
    return model(batch["g_idx"], batch["env"], batch["s"], batch["ds"], 0)


def score(pred, raw, scale):
    mask = raw["mask"].astype(bool)
    errors = (pred.astype(np.float64) - raw["y"].astype(np.float64)) * scale
    per_curve = np.sqrt((errors**2 * mask).sum(axis=1) / mask.sum(axis=1))
    mae = (np.abs(errors) * mask).sum(axis=1) / mask.sum(axis=1)
    mean_target = float(raw["y"][mask].astype(np.float64).mean() * scale)
    return dict(rmse=float(per_curve.mean()), mae=float(mae.mean()),
        relative_error=float(per_curve.mean()/mean_target*100), mean_target=mean_target,
        n_curves=len(pred), n_scored_points=int(mask.sum())), per_curve
