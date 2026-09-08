"""One paired (lambda_ODE, lambda_K) trial with resumable validation-only training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import pandas as pd
import torch

from physics_ablation import ROOT, setup, forward, score, state_digest, digest, atomic_json
from lambda_ode_trial import initialize


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args):
    result_path = args.output / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Completed trial must not be overwritten: {args.output}")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    core, loss_function, model_args, config, raw, batches = setup(args.dataset, device)
    del raw["test"]
    model, initial_hash = initialize(args.dataset, core, model_args, args.seed, device)
    weights = dict(physic=args.lambda_ode, ymax=args.lambda_k, r=0., mono=0.)
    config.update(dataset=args.dataset, seed=args.seed, lambda_ode=args.lambda_ode,
                  lambda_k=args.lambda_k, effective_weights=weights,
                  initial_state_sha256=initial_hash, torch_version=torch.__version__,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  gpu_name=torch.cuda.get_device_name())
    if args.resume:
        old_config = json.loads((args.output / "config.json").read_text())
        if old_config != config:
            raise ValueError("Resume configuration/environment differs from the original trial")
    else:
        atomic_json(args.output / "config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    history, recent = [], []
    best, best_epoch, best_state, epoch = float("inf"), -1, None, 0
    elapsed, resumes = 0., []
    if args.resume:
        saved = torch.load(args.output / "resume_state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, recent = saved["history"], saved["recent"]
        best, best_epoch, best_state = saved["best"], saved["best_epoch"], saved["best_state"]
        epoch, elapsed = saved["epoch"], saved["seconds"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        resumes = saved["resumes"] + [dict(resumed_after_epoch=epoch, resumed_unix=time.time())]
        # A hard kill may leave newer progress than the latest complete state.
        pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
        if best_state is not None:
            save_torch(args.output / "checkpoint.pt", dict(state_dict=best_state, config=config, epoch=best_epoch))
    started = time.monotonic()
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.cuda.reset_peak_memory_stats()

    def save_resume():
        save_torch(args.output / "resume_state.pt", dict(
            model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            history=history, recent=recent, best=best, best_epoch=best_epoch, best_state=best_state,
            epoch=epoch, seconds=elapsed + time.monotonic() - started, resumes=resumes,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()))

    last_epoch = min(args.stop_after or config["epochs"], config["epochs"])
    if epoch >= last_epoch and epoch < config["epochs"]:
        raise ValueError("Requested stopping epoch must follow the saved epoch")
    for epoch in range(epoch + 1, last_epoch + 1):
        model.train()
        batch = batches["train"]
        if config["permute_training"]:
            index = torch.randperm(config["n_train"], device=device)
            batch = {key: value[index] for key, value in batch.items()}
        output = forward(model, batch)
        data_loss = loss_function(output["pred"], batch["y"], batch["mask"])
        loss, components = data_loss, {}
        if args.lambda_ode > 0 or args.lambda_k > 0:
            components = core.physics_losses(model, output, batch["env"], weights)
            if args.dataset == "maize":
                loss = loss + sum(components.values())
            else:
                for value in components.values():
                    loss = loss + value
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite objective at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.lambda_ode == args.lambda_k == 0:
            assert all(p.grad is None for p in model.ode_param_head.parameters())
            assert float((loss - data_loss).detach()) == 0.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        # Never insert an off-schedule validation when interrupted.
        if epoch % config["eval_every"] == 0:
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
                save_torch(args.output / "checkpoint.pt", dict(state_dict=best_state, config=config, epoch=epoch))
            history.append(dict(epoch=epoch, data_loss=float(data_loss.detach()), objective=float(loss.detach()),
                val_rmse=value * config["scale"], selection_rmse=smooth * config["scale"],
                best_epoch=best_epoch, grad_norm=float(norm),
                L_m=float(components["L_m"].detach()) if components else 0.,
                L_y=float(components["L_y"].detach()) if components else 0.))
            pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
            atomic_json(args.output / "progress.json", dict(epoch=epoch, epochs=config["epochs"],
                seconds=elapsed + time.monotonic() - started, best_epoch=best_epoch,
                best_selection_rmse=best * config["scale"] if np.isfinite(best) else None,
                test_evaluations=0))
            if epoch % 100 == 0:
                print(f"{args.dataset} ODE={args.lambda_ode:g} K={args.lambda_k:g} seed={args.seed} "
                      f"{epoch}/{config['epochs']} val={value * config['scale']:.7f}", flush=True)
        if epoch % 100 == 0 or stopped or epoch == last_epoch:
            save_resume()
        if stopped:
            atomic_json(args.output / "interrupted.json", dict(epoch=epoch, test_evaluations=0))
            raise InterruptedError("Trial interrupted; optimizer and RNG state saved")
    if epoch < config["epochs"]:
        atomic_json(args.output / "paused.json", dict(status="diagnostic_prefix_only", epoch=epoch,
            final_state_sha256=state_digest(model), initial_state_sha256=initial_hash, test_evaluations=0))
        return
    if best_state is None:
        raise RuntimeError("No eligible validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metrics, predictions = {}, {}
    with torch.no_grad():
        for split in ("train_eval", "val"):
            pred = forward(model, batches[split])["pred"].cpu().numpy()
            label = "train" if split == "train_eval" else split
            metrics[label], _ = score(pred, raw[split], config["scale"])
            predictions[split] = pred
            predictions[split + "_target"] = raw[split]["y"]
            predictions[split + "_mask"] = raw[split]["mask"]
    np.savez_compressed(args.output / "validation_predictions.npz", **predictions)
    result = dict(dataset=args.dataset, seed=args.seed, lambda_ode=args.lambda_ode, lambda_k=args.lambda_k,
        status="complete", epochs=config["epochs"], best_epoch=best_epoch, selection_rmse=best * config["scale"],
        initial_state_sha256=initial_hash, checkpoint_sha256=digest(args.output / "checkpoint.pt"),
        checkpoint=str((args.output / "checkpoint.pt").relative_to(ROOT)), n_params=core.count_parameters(model),
        unit=config["unit"], metrics=metrics, train_rmse=metrics["train"]["rmse"], val_rmse=metrics["val"]["rmse"],
        seconds=elapsed + time.monotonic() - started, peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
        resumes=resumes, test_evaluations=0, origin="new joint validation-only trial")
    atomic_json(result_path, result)
    print("COMPLETE " + json.dumps(result), flush=True)


def evaluate(args):
    selection = json.loads(args.selection.read_text())
    frozen_path = args.selection.parent.parent / "all_selections_frozen.json"
    frozen = json.loads(frozen_path.read_text())
    assert frozen["selection_sha256"][args.dataset] == digest(args.selection)
    assert selection["dataset"] == args.dataset and frozen["new_candidate_test_evaluations"] == 0
    record = selection[args.role]
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
            predictions[split + "_target"] = raw[split]["y"]
            predictions[split + "_mask"] = raw[split]["mask"]
            predictions[split + "_genotype"] = raw[split]["genotype"]
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    atomic_json(args.output / "result.json", dict(dataset=args.dataset, seed=args.seed, role=args.role,
        lambda_ode=record["lambda_ode"], lambda_k=record["lambda_k"], checkpoint_sha256=expected,
        selection_sha256=digest(args.selection), all_selections_sha256=digest(frozen_path),
        n_params=core.count_parameters(model), metrics=metrics, unit=config["unit"], test_evaluations=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--dataset", required=True, choices=("wheat", "maize", "arabidopsis"))
    parser.add_argument("--seed", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--lambda-ode", type=float, default=0.)
    parser.add_argument("--lambda-k", type=float, default=0.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Diagnostic prefix only; preserves the full cosine schedule")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--role", choices=("selected_phyto", "selected_overall"), default="selected_phyto")
    args = parser.parse_args()
    if not all(np.isfinite(v) and v >= 0 for v in (args.lambda_ode, args.lambda_k)):
        parser.error("Both coefficients must be finite and nonnegative")
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("stop-after must be positive")
    if args.mode == "evaluate" and args.selection is None:
        parser.error("Evaluation requires all three dataset selections to be frozen")
    torch.set_num_threads(2)
    (train if args.mode == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
