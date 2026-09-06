"""One validation-only maize latent ODE tuning trial; never evaluates test targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd
import torch

from data import make_dataset, ROOT
from run_experiment import LATENT, REFERENCE, seed_all, tensor_batch, metrics

BASE = dict(latent_dim=16, g_embed_dim=4, ode_hidden=16, ode_layers=1, dec_hidden=16,
            enc_hidden=8, time_mode="calendar", lr=.01, weight_decay=1e-4,
            physic=2., ymax=.1, mono=0., epochs=1500)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def build(config, n_genotypes):
    model = LATENT.LatentODEHeightModel(n_genotypes=n_genotypes, env_dim=1,
        **{k: config[k] for k in ["latent_dim", "g_embed_dim", "ode_hidden", "ode_layers", "dec_hidden", "enc_hidden", "time_mode"]},
        init_mode="env_encoder", genetic_encoding="one_hot", n_substeps=1, days_per_tau=95., use_physics=True)
    for name, value in model.named_parameters():
        if "lstm" in name and "weight" in name and value.ndim >= 2:
            torch.nn.init.orthogonal_(value)
    return model


def predict(model, batch):
    return model(batch["g_idx"], batch["env"], batch["s"], batch["ds"], 0)["pred"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        raise FileExistsError(f"Trial already complete: {folder}")
    torch.set_num_threads(2)
    ds = make_dataset(split="chronological")
    seed_all(args.seed)
    device = torch.device("cuda")
    model = build(config, len(ds.genotypes)).to(device)
    train, val = tensor_batch(ds.train, device), tensor_batch(ds.val, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    weights = {k: config[k] for k in ["physic", "ymax", "mono"]} | {"r": 0.}
    best, best_epoch, best_state = float("inf"), 0, None
    history = []
    started = time.monotonic()
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.cuda.reset_peak_memory_stats(device)
    atomic_json(folder / "config.json", config)
    for epoch in range(1, config["epochs"]+1):
        model.train()
        output = model(train["g_idx"], train["env"], train["s"], train["ds"], 0)
        data_loss = REFERENCE.curve_rmse(output["pred"], train["y"], train["mask"])
        loss = data_loss
        if any(weights[k] > 0 for k in ["physic", "ymax", "mono"]):
            loss = loss + sum(LATENT.physics_losses(model, output, train["env"], weights).values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite objective at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0 or epoch == config["epochs"] or stopped:
            model.eval()
            with torch.no_grad():
                prediction = predict(model, val)
                value = float(REFERENCE.curve_rmse(prediction, val["y"], val["mask"]))
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite validation RMSE")
            if value < best:
                best, best_epoch = value, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                checkpoint = dict(state_dict=best_state, config=config, seed=args.seed, epoch=epoch,
                    val_rmse=value*ds.y_scale, y_scale=ds.y_scale, genotypes=ds.genotypes)
                torch.save(checkpoint, folder / "checkpoint.pt.part")
                (folder / "checkpoint.pt.part").replace(folder / "checkpoint.pt")
            elapsed = time.monotonic()-started
            history.append(dict(epoch=epoch, train_objective=float(loss.detach()),
                train_data_rmse=float(data_loss.detach())*ds.y_scale, val_rmse=value*ds.y_scale,
                best_epoch=best_epoch, best_val_rmse=best*ds.y_scale,
                grad_norm=float(grad_norm), lr=scheduler.get_last_lr()[0], seconds=elapsed))
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            atomic_json(folder / "progress.json", dict(epoch=epoch, epochs=config["epochs"],
                seed=args.seed, best_val_rmse=best*ds.y_scale, best_epoch=best_epoch,
                elapsed_seconds=elapsed, test_evaluations=0))
            if epoch % 100 == 0 or stopped:
                print(f"epoch={epoch}/{config['epochs']} val={value*ds.y_scale:.4f} "
                      f"best={best*ds.y_scale:.4f}@{best_epoch} seconds={elapsed:.1f}", flush=True)
        if stopped:
            atomic_json(folder / "interrupted.json", dict(epoch=epoch, test_evaluations=0))
            return
    model.load_state_dict(best_state)
    model.eval()
    predictions, recorded = {}, {}
    with torch.no_grad():
        for name in ["train_eval", "val"]:
            batch = getattr(ds, name)
            predictions[name] = predict(model, tensor_batch(batch, device)).cpu().numpy()
            values, _, _ = metrics(predictions[name], batch, ds.y_scale)
            label = "train" if name == "train_eval" else name
            recorded.update({f"{label}_{k}": v for k, v in values.items()})
    np.testing.assert_allclose(recorded["val_rmse"], best*ds.y_scale, atol=2e-5, rtol=1e-6)
    np.savez_compressed(folder / "validation_predictions.npz", **predictions)
    result = dict(seed=args.seed, config=config, n_params=REFERENCE.parameter_count(model), best_epoch=best_epoch,
        epochs=config["epochs"], seconds=time.monotonic()-started, status="complete", test_evaluations=0,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), gpu_name=torch.cuda.get_device_name(device),
        peak_memory_mib=torch.cuda.max_memory_allocated(device)/2**20,
        checkpoint_sha256=hashlib.sha256((folder/"checkpoint.pt").read_bytes()).hexdigest(), **recorded)
    atomic_json(folder / "result.json", result)
    print(f"COMPLETE val={recorded['val_rmse']:.5f} seed={args.seed} epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
