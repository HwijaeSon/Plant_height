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
    sys.path.insert(0, str(ROOT / dataset / "code"))
    if dataset == "wheat":
        import data
        import model as core
        cfg = json.loads((ROOT / "wheat/results/final_config.json").read_text())
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
        import tune_trial
        core = tune_trial.LATENT
        cfg = json.loads((ROOT / "maize/results/tuning_20260904/selected.json").read_text())["config"]
        ds = data.make_dataset(split="chronological")
        model_args = {k: cfg[k] for k in (
            "latent_dim", "g_embed_dim", "ode_hidden", "ode_layers", "dec_hidden", "enc_hidden", "time_mode")}
        model_args.update(init_mode="env_encoder", genetic_encoding="one_hot", n_substeps=1,
                          days_per_tau=95.)
        config = dict(model=model_args, epochs=cfg["epochs"], lr=cfg["lr"],
            weight_decay=cfg["weight_decay"], physic=cfg["physic"], ymax=cfg["ymax"],
            eval_every=10, min_epoch=0, val_smooth=1, permute_training=False,
            scale=ds.y_scale, unit="relative UAV height", expected_n_params=9003)
        loss_function = tune_trial.REFERENCE.curve_rmse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("wheat", "maize", "arabidopsis"))
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.output}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    core, loss_function, model_args, config, raw, batches = setup(args.dataset, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Keep the auxiliary head even in the pure-data variant. Removing it would
    # consume fewer random draws before LSTM initialization, confounding seeds.
    model = core.LatentODEHeightModel(**model_args)
    # The published maize tuner initializes recurrent weights on CPU before
    # moving the model; wheat/Arabidopsis initialize those weights on the GPU.
    if args.dataset != "maize":
        model = model.to(device)
    for name, value in model.named_parameters():
        if "lstm" in name and "weight" in name and value.ndim >= 2:
            torch.nn.init.orthogonal_(value)
    if args.dataset == "maize":
        model = model.to(device)
    assert core.count_parameters(model) == config["expected_n_params"]
    config.update(dataset=args.dataset, variant=args.variant, seed=args.seed,
        initial_state_sha256=state_digest(model),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_name=torch.cuda.get_device_name(), torch_version=torch.__version__,
        source_git_commit=os.popen("git rev-parse HEAD").read().strip())
    weights = dict(physic=config["physic"], ymax=config["ymax"], r=0., mono=0.)
    if args.variant != "full":
        weights["physic"] = 0.
    if args.variant == "no_biological_loss":
        weights["ymax"] = 0.
    config["effective_weights"] = weights
    atomic_json(args.output / "config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    train, val = batches["train"], batches["val"]
    recent, history = [], []
    best, best_epoch, best_state = float("inf"), -1, None
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config["epochs"]+1):
        model.train()
        if config["permute_training"]:
            index = torch.randperm(config["n_train"], device=device)
            batch = {key: value[index] for key, value in train.items()}
        else:
            batch = train
        output = forward(model, batch)
        data_loss = loss_function(output["pred"], batch["y"], batch["mask"])
        loss = data_loss
        components = {}
        if weights["physic"] > 0 or weights["ymax"] > 0:
            components = core.physics_losses(model, output, batch["env"], weights)
            if args.dataset == "maize":
                loss = loss + sum(components.values())
            else:
                for value in components.values():
                    loss = loss + value
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if epoch == 1 and args.variant == "no_biological_loss":
            assert all(p.grad is None for p in model.ode_param_head.parameters())
            assert float((loss-data_loss).detach()) == 0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if epoch % config["eval_every"] == 0 or epoch == config["epochs"]:
            model.eval()
            with torch.no_grad():
                value = float(loss_function(forward(model, val)["pred"], val["y"], val["mask"]))
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite validation loss")
            recent.append(value)
            recent = recent[-config["val_smooth"]:]
            smooth = float(np.mean(recent))
            if epoch >= config["min_epoch"] and len(recent) == config["val_smooth"] and smooth < best:
                best, best_epoch = smooth, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(dict(epoch=epoch, data_loss=float(data_loss.detach()),
                objective=float(loss.detach()), val_rmse=value*config["scale"],
                selection_rmse=smooth*config["scale"], best_epoch=best_epoch,
                best_selection_rmse=best*config["scale"] if np.isfinite(best) else None,
                **{key: float(value.detach()) for key, value in components.items()}))
            elapsed = time.monotonic()-started
            atomic_json(args.output / "progress.json", dict(epoch=epoch, epochs=config["epochs"],
                best_epoch=best_epoch, seconds=elapsed, test_evaluations=0))
            if epoch % 100 == 0:
                pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
                print(f"{args.dataset}/{args.variant}/{args.seed} {epoch}/{config['epochs']} "
                      f"val={value*config['scale']:.6f} elapsed={elapsed:.1f}s", flush=True)
    if best_state is None:
        raise RuntimeError("No eligible validation checkpoint")
    model.load_state_dict(best_state)
    torch.save(dict(state_dict=best_state, config=config, epoch=best_epoch), args.output / "checkpoint.pt")
    frozen = dict(best_epoch=best_epoch, selection_rmse=best*config["scale"],
        checkpoint_sha256=digest(args.output / "checkpoint.pt"), test_evaluations=0)
    atomic_json(args.output / "selection_frozen.json", frozen)
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    predictions, curve_rows, all_metrics = {}, [], {}
    model.eval()
    with torch.no_grad():
        for split in ("train_eval", "val", "test"):
            # The forward call has no target/mask argument, including at test.
            batch = {key: torch.as_tensor(raw[split][key],
                dtype=torch.long if key == "g_idx" else torch.float32, device=device)
                for key in ("g_idx", "env", "s", "ds")}
            pred = forward(model, batch)["pred"].cpu().numpy()
            values, per_curve = score(pred, raw[split], config["scale"])
            label = "train" if split == "train_eval" else split
            all_metrics[label] = values
            predictions[split] = pred
            predictions[split+"_target"] = raw[split]["y"]
            predictions[split+"_mask"] = raw[split]["mask"]
            predictions[split+"_genotype"] = raw[split]["genotype"]
            for i, error in enumerate(per_curve):
                curve_rows.append(dict(split=label, curve=i, genotype=str(raw[split]["genotype"][i]),
                    rmse=float(error), n_observed=int(raw[split]["mask"][i].sum())))
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    pd.DataFrame(curve_rows).to_csv(args.output / "per_curve_metrics.csv", index=False)
    result = dict(dataset=args.dataset, variant=args.variant, seed=args.seed,
        status="complete", n_params=core.count_parameters(model), unit=config["unit"],
        seconds=time.monotonic()-started, peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
        initial_state_sha256=config["initial_state_sha256"],
        checkpoint_sha256=frozen["checkpoint_sha256"], best_epoch=best_epoch,
        epochs=config["epochs"], test_evaluations=1, metrics=all_metrics)
    atomic_json(args.output / "result.json", result)
    print("COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
