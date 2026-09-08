"""One lambda_ODE trial, preserving the published training protocol.

Training invocations never evaluate test targets. A separate evaluate command
requires a frozen search-selection record with the checkpoint hash.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from physics_ablation import ROOT, setup, forward, score, state_digest, digest, atomic_json

OLD = ROOT / "experiments/results/physics_ablation_20260908"


def initialize(dataset, core, model_args, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = core.LatentODEHeightModel(**model_args)
    if dataset != "maize":
        model = model.to(device)
    for name, value in model.named_parameters():
        if "lstm" in name and "weight" in name and value.ndim >= 2:
            torch.nn.init.orthogonal_(value)
    if dataset == "maize":
        model = model.to(device)
    expected = json.loads((OLD / dataset / f"no_ode_residual_seed{seed}/config.json").read_text())
    actual = state_digest(model)
    if actual != expected["initial_state_sha256"]:
        raise ValueError("Initialization does not match the paired ablation seed")
    return model, actual


def train(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    device = torch.device("cuda")
    core, loss_function, model_args, config, raw, batches = setup(args.dataset, device)
    del raw["test"]
    model, initial_hash = initialize(args.dataset, core, model_args, args.seed, device)
    config.update(dataset=args.dataset, seed=args.seed, lambda_ode=args.lambda_ode,
        initial_state_sha256=initial_hash, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch_version=torch.__version__, gpu_name=torch.cuda.get_device_name(),
        source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    weights = dict(physic=args.lambda_ode, ymax=config["ymax"], r=0., mono=0.)
    config["effective_weights"] = weights
    atomic_json(args.output / "config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    history, recent = [], []
    best, best_epoch, best_state = float("inf"), -1, None
    started = time.monotonic()
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.cuda.reset_peak_memory_stats()
    epochs = 10 if args.verify_prefix else config["epochs"]
    for epoch in range(1, epochs+1):
        model.train()
        batch = batches["train"]
        if config["permute_training"]:
            index = torch.randperm(config["n_train"], device=device)
            batch = {key: value[index] for key, value in batch.items()}
        output = forward(model, batch)
        data_loss = loss_function(output["pred"], batch["y"], batch["mask"])
        components = core.physics_losses(model, output, batch["env"], weights)
        loss = data_loss
        if args.dataset == "maize":
            loss = loss + sum(components.values())
        else:
            for value in components.values():
                loss = loss + value
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite objective at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if epoch % config["eval_every"] == 0 or stopped:
            model.eval()
            with torch.no_grad():
                val = batches["val"]
                value = float(loss_function(forward(model, val)["pred"], val["y"], val["mask"]))
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite validation loss")
            recent.append(value)
            recent = recent[-config["val_smooth"]:]
            smooth = float(np.mean(recent))
            if epoch >= config["min_epoch"] and len(recent) == config["val_smooth"] and smooth < best:
                best, best_epoch = smooth, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save(dict(state_dict=best_state, config=config, epoch=epoch), args.output / "checkpoint.pt.part")
                (args.output / "checkpoint.pt.part").replace(args.output / "checkpoint.pt")
            history.append(dict(epoch=epoch, data_loss=float(data_loss.detach()), objective=float(loss.detach()),
                val_rmse=value*config["scale"], selection_rmse=smooth*config["scale"],
                best_epoch=best_epoch, grad_norm=float(norm),
                L_m=float(components["L_m"].detach()), L_y=float(components["L_y"].detach())))
            pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
            atomic_json(args.output / "progress.json", dict(epoch=epoch, epochs=config["epochs"],
                seconds=time.monotonic()-started, best_epoch=best_epoch,
                best_selection_rmse=best*config["scale"] if np.isfinite(best) else None,
                test_evaluations=0))
            if epoch % 100 == 0:
                print(f"{args.dataset} lambda={args.lambda_ode:g} seed={args.seed} "
                      f"{epoch}/{epochs} val={value*config['scale']:.6f} "
                      f"seconds={time.monotonic()-started:.1f}", flush=True)
        if stopped:
            atomic_json(args.output / "interrupted.json", dict(epoch=epoch, test_evaluations=0))
            raise InterruptedError("Trial interrupted; completed status was not written")
    if args.verify_prefix:
        assert args.seed == 1 and args.lambda_ode == config["physic"]
        old = pd.read_csv(OLD / args.dataset / "full_seed1/history.csv").iloc[0]
        for key in ("data_loss", "objective", "val_rmse"):
            np.testing.assert_allclose(history[0][key], old[key], rtol=1e-7, atol=1e-9)
        atomic_json(args.output / "verification.json", dict(status="passed", epochs=10,
            initial_state_sha256=initial_hash, training_and_validation_prefix="matches reference", test_evaluations=0))
        return
    if best_state is None:
        raise RuntimeError("No eligible validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metrics, predictions = {}, {}
    with torch.no_grad():
        for split in ("train_eval", "val"):
            pred = forward(model, batches[split])["pred"].cpu().numpy()
            values, _ = score(pred, raw[split], config["scale"])
            label = "train" if split == "train_eval" else split
            metrics[label] = values
            predictions[split] = pred
            predictions[split+"_target"] = raw[split]["y"]
            predictions[split+"_mask"] = raw[split]["mask"]
    np.savez_compressed(args.output / "validation_predictions.npz", **predictions)
    result = dict(dataset=args.dataset, seed=args.seed, lambda_ode=args.lambda_ode, status="complete",
        epochs=config["epochs"], best_epoch=best_epoch, selection_rmse=best*config["scale"],
        initial_state_sha256=initial_hash, checkpoint_sha256=digest(args.output / "checkpoint.pt"),
        checkpoint=str((args.output / "checkpoint.pt").relative_to(ROOT)),
        n_params=core.count_parameters(model), unit=config["unit"], metrics=metrics,
        train_rmse=metrics["train"]["rmse"], val_rmse=metrics["val"]["rmse"],
        seconds=time.monotonic()-started, peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
        test_evaluations=0, origin="new validation-only trial")
    atomic_json(args.output / "result.json", result)
    print("COMPLETE " + json.dumps(result), flush=True)


def evaluate(args):
    selected = json.loads(args.selection.read_text())
    if selected["dataset"] != args.dataset:
        raise ValueError("Selection dataset mismatch")
    record = selected["selected_positive"]
    if record["lambda_ode"] <= 0:
        raise ValueError("Expected positive lambda selection")
    checkpoint = ROOT / record["checkpoints"][str(args.seed)]["path"]
    expected = record["checkpoints"][str(args.seed)]["sha256"]
    if digest(checkpoint) != expected:
        raise ValueError("Frozen checkpoint hash mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device = torch.device("cuda")
    core, _, model_args, config, raw, _ = setup(args.dataset, device)
    model = core.LatentODEHeightModel(**model_args).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved.get("state_dict", saved))
    model.eval()
    predictions, metrics = {}, {}
    with torch.no_grad():
        for split in ("train_eval", "val", "test"):
            batch = {key: torch.as_tensor(raw[split][key], dtype=torch.long if key == "g_idx" else torch.float32,
                device=device) for key in ("g_idx", "env", "s", "ds")}
            pred = forward(model, batch)["pred"].cpu().numpy()
            label = "train" if split == "train_eval" else split
            metrics[label], _ = score(pred, raw[split], config["scale"])
            predictions[split] = pred
            predictions[split+"_target"] = raw[split]["y"]
            predictions[split+"_mask"] = raw[split]["mask"]
            predictions[split+"_genotype"] = raw[split]["genotype"]
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    atomic_json(args.output / "result.json", dict(dataset=args.dataset, seed=args.seed,
        lambda_ode=record["lambda_ode"], checkpoint_sha256=expected, selection_sha256=digest(args.selection),
        n_params=core.count_parameters(model), metrics=metrics, unit=config["unit"], test_evaluations=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--dataset", required=True, choices=("wheat", "maize", "arabidopsis"))
    parser.add_argument("--seed", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--lambda-ode", type=float, default=0.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-prefix", action="store_true")
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.lambda_ode) or args.lambda_ode < 0:
        parser.error("lambda must be finite and nonnegative")
    torch.set_num_threads(2)
    if args.mode == "train":
        train(args)
    else:
        if args.selection is None:
            parser.error("Evaluation requires a frozen selection record")
        evaluate(args)


if __name__ == "__main__":
    main()
