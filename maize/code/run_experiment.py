"""Shared maize experiment: fixed protocol, training-only fitting, auditable outputs.

GPU models reuse the existing repository architectures without changing them.
Run CPU process baselines once, then distribute seeds across available GPUs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from sklearn.ensemble import RandomForestRegressor
import torch

from data import Dataset, SequenceSet, make_dataset, ROOT


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REFERENCE = load_module("maize_reference_architecture", ROOT.parent / "arabidopsis/code/models.py")
LATENT = load_module("maize_latent_architecture", ROOT.parent / "wheat/code/model.py")
MODEL_NAMES = {"logistic": "Logi-ODE", "temperature": "Temp-ODE", "rf": "RF", "lstm": "LSTM-NN",
               "pinn": "Logi-PINN", "latent": "Latent Neural ODE (ours)"}
SPLITS = ("train_eval", "val", "test")
H0 = 1e-4  # same normalized-target initial condition as the existing process baselines


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(prediction, batch, scale):
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(batch.y, dtype=np.float64)
    mask = batch.mask.astype(bool)
    if pred.shape != target.shape or not np.isfinite(pred).all():
        raise ValueError("Invalid prediction")
    error = (pred - target) * scale
    denom = mask.sum(axis=1)
    rmse = np.sqrt((error**2 * mask).sum(axis=1) / denom)
    mae = (np.abs(error) * mask).sum(axis=1) / denom
    global_rmse = np.sqrt(np.mean(error[mask]**2))
    return {"rmse": float(rmse.mean()), "mae": float(mae.mean()),
            "pooled_rmse": float(global_rmse)}, rmse, mae


def save_result(out, key, seed, ds, predictions, info, state=None):
    folder = out / "runs" / f"{key}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(model=MODEL_NAMES[key], model_key=key, seed=seed, target_unit="relative UAV height",
                  y_scale=ds.y_scale, **info)
    rows = []
    for name in SPLITS:
        batch = getattr(ds, name)
        values, rmses, maes = metrics(predictions[name], batch, ds.y_scale)
        label = "train" if name == "train_eval" else name
        result.update({f"{label}_{metric}": value for metric, value in values.items()})
        for i in range(len(batch)):
            rows.append(dict(model=MODEL_NAMES[key], seed=seed, split=label, year=int(batch.year[i]),
                             genotype=ds.genotypes[batch.g_idx[i]], rmse=rmses[i], mae=maes[i],
                             n_observed=int(batch.mask[i].sum())))
    np.savez_compressed(folder / "predictions.npz", **predictions)
    pd.DataFrame(rows).to_csv(folder / "per_curve_metrics.csv", index=False)
    if state is not None:
        torch.save(state, folder / "checkpoint.pt")
    temporary = folder / "result.json.part"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(folder / "result.json")
    print(f"DONE {key} seed={seed} train={result['train_rmse']:.3f} "
          f"val={result['val_rmse']:.3f} test={result['test_rmse']:.3f}", flush=True)


def logistic(exposure, r, K):
    exponent = np.clip(-r * exposure, -80, 80)
    return K / (1 + (K / H0 - 1) * np.exp(exponent))


def temperature_response(celsius, lower_k, upper_k):
    # SciPy estimates tiny parameter perturbations: float32 would quantize
    # the threshold derivatives to zero and silently prevent their fitting.
    kelvin = np.asarray(celsius, dtype=np.float64) + 273.15
    low = np.exp(np.clip(2000 / kelvin - 2000 / lower_k, -60, 60))
    high = np.exp(np.clip(60000 / upper_k - 60000 / kelvin, -60, 60))
    return 1 / (1 + low + high)


def process_baselines(ds: Dataset, out: Path):
    """Fit only training observations; integrate the daily temperature response."""
    started = time.monotonic()
    n = len(ds.genotypes)
    b = ds.train
    rr, cc = np.where(b.mask > 0)
    geno = b.g_idx[rr]
    target = b.y[rr, cc].astype(np.float64)
    weights = 1 / np.sqrt(b.mask.sum(axis=1)[rr])
    r = np.zeros(n)
    K = np.zeros(n)
    fits = []
    for g in range(n):
        choose = geno == g
        best = None
        for start_rate in [.08, .15, .3]:
            fit = least_squares(lambda p: (logistic(ds.days[cc[choose]], p[0], p[1]) - target[choose]) * weights[choose],
                                [start_rate, max(.1, float(target[choose].max()))],
                                bounds=([1e-4, .01], [.999, 1.999]), max_nfev=2000,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if best is None or fit.cost < best.cost:
                best = fit
        r[g], K[g] = best.x
        fits.append(dict(genotype=ds.genotypes[g], success=bool(best.success), cost=float(best.cost), nfev=int(best.nfev)))
    if not all(f["success"] for f in fits):
        raise RuntimeError("A logistic fit failed to converge")
    out.mkdir(parents=True, exist_ok=True)
    parameters = dict(genotypes=ds.genotypes, r=r.tolist(), K=K.tolist(), h0=H0,
                      target_unit="relative / training y_scale", fits=fits)
    (out / "logistic_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    preds = {name: logistic(ds.days[None, :], r[getattr(ds, name).g_idx, None], K[getattr(ds, name).g_idx, None])
             for name in SPLITS}
    save_result(out, "logistic", 0, ds, preds, dict(n_params=2*n, seconds=time.monotonic()-started,
                best_epoch=0, fit_objective="curve-balanced observed MSE", successful_fits=n))

    started = time.monotonic()
    # Only two training-year weather histories, so build response once per year.
    years = sorted(set(b.year))
    weather = np.stack([b.env[np.flatnonzero(b.year == y)[0], :, 0] * ds.env_std[0] + ds.env_mean[0] for y in years])
    obs_year = np.searchsorted(years, b.year[rr])

    def exposure(temp, low, high):
        response = temperature_response(temp, low, high)
        return np.concatenate([np.zeros((len(temp), 1)), np.cumsum((response[:, :-1]+response[:, 1:])/2, axis=1)], axis=1)

    def residual(p):
        exp = exposure(weather, p[-2], p[-1])
        return (logistic(exp[obs_year, cc], p[geno], p[n+geno]) - target) * weights

    sparsity = lil_matrix((len(target), 2*n+2), dtype=int)
    ix = np.arange(len(target))
    sparsity[ix, geno] = 1
    sparsity[ix, n+geno] = 1
    sparsity[:, -2:] = 1
    init_response = float(temperature_response(weather, 292., 303.).mean())
    initial = np.r_[np.clip(r/init_response, .001, 1.9), K, 292., 303.]
    fit = least_squares(residual, initial, bounds=(np.r_[np.full(n, 1e-4), np.full(n, .01), 275., 301.],
                                                 np.r_[np.full(n, 2.), np.full(n, 1.999), 300., 330.]),
                        jac_sparsity=sparsity.tocsr(), max_nfev=5000, ftol=1e-8, xtol=1e-8, gtol=1e-8,
                        x_scale="jac")
    if not fit.success:
        raise RuntimeError(f"Temperature fit did not converge: {fit.message}")
    p = fit.x
    params = dict(genotypes=ds.genotypes, r=p[:n].tolist(), K=p[n:2*n].tolist(), lower_k=float(p[-2]),
                  upper_k=float(p[-1]), h0=H0, success=bool(fit.success), nfev=int(fit.nfev),
                  message=str(fit.message), cost=float(fit.cost))
    (out / "temperature_parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    preds = {}
    for name in SPLITS:
        batch = getattr(ds, name)
        temp = batch.env[:, :, 0] * ds.env_std[0] + ds.env_mean[0]
        exp = exposure(temp, p[-2], p[-1])
        preds[name] = logistic(exp, p[batch.g_idx, None], p[n+batch.g_idx, None])
    save_result(out, "temperature", 0, ds, preds, dict(n_params=2*n+2, seconds=time.monotonic()-started,
                best_epoch=0, fit_objective="curve-balanced observed MSE", nfev=int(fit.nfev)))


def random_forest(ds, out, seed):
    started = time.monotonic()
    n = len(ds.genotypes)
    def features(batch, rows, cols):
        x = np.zeros((len(rows), n+2), dtype=np.float32)
        x[np.arange(len(rows)), batch.g_idx[rows]] = 1
        x[:, n] = batch.env[rows, cols, 0]
        x[:, n+1] = ds.days[cols] / ds.days[-1]
        return x
    rr, cc = np.where(ds.train.mask > 0)
    model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=2, criterion="squared_error")
    model.fit(features(ds.train, rr, cc), ds.train.y[rr, cc], sample_weight=1/ds.train.mask.sum(axis=1)[rr])
    preds = {}
    for name in SPLITS:
        batch = getattr(ds, name)
        rr, cc = np.indices(batch.y.shape)
        rr, cc = rr.ravel(), cc.ravel()
        values = [model.predict(features(batch, rr[i:i+4096], cc[i:i+4096])) for i in range(0, len(rr), 4096)]
        preds[name] = np.concatenate(values).reshape(batch.y.shape)
    import joblib
    folder = out / "runs" / f"rf_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "forest.joblib", compress=3)
    save_result(out, "rf", seed, ds, preds, dict(n_params=None, seconds=time.monotonic()-started,
                best_epoch=0, n_trees=100, n_tree_nodes=sum(t.tree_.node_count for t in model.estimators_)))


def tensor_batch(seq, device):
    return {name: torch.as_tensor(getattr(seq, name), device=device,
                                 dtype=torch.long if name == "g_idx" else torch.float32)
            for name in ["y", "mask", "env", "s", "ds", "g_idx"]}


def build_model(key, ds, params=None):
    if key == "latent":
        # Frozen wheat final_config.json architecture and hyperparameters.
        model = LATENT.LatentODEHeightModel(n_genotypes=len(ds.genotypes), env_dim=1, latent_dim=16,
                    g_embed_dim=4, ode_hidden=16, ode_layers=1, dec_hidden=16, enc_hidden=8,
                    init_mode="env_encoder", time_mode="calendar", genetic_encoding="one_hot",
                    n_substeps=1, days_per_tau=float(ds.days[-1]), use_physics=True)
        for name, value in model.named_parameters():
            if "lstm" in name and "weight" in name and value.ndim >= 2:
                torch.nn.init.orthogonal_(value)
    else:
        model = REFERENCE.ReferenceLSTM(len(ds.genotypes), physics=key == "pinn", direct_ode_parameters=key == "pinn")
        for name, value in model.named_parameters():
            if "weight" in name and value.ndim >= 2:
                torch.nn.init.orthogonal_(value)
            elif "bias" in name:
                torch.nn.init.zeros_(value)
        if key == "pinn":
            REFERENCE.initialize_logistic_head(model, torch.tensor(params["r"]), torch.tensor(params["K"]))
    return model


def forward(key, model, batch, time_grid, physics=False):
    if key == "latent":
        return model(batch["g_idx"], batch["env"], batch["s"], batch["ds"], 0)
    with torch.backends.cudnn.flags(enabled=not physics):
        return model(batch["g_idx"], batch["env"][:, :, 0], time_grid)


def train_neural(ds, out, key, seed, args):
    seed_all(seed)
    device = torch.device(args.device)
    params = json.loads((out / "logistic_parameters.json").read_text()) if key == "pinn" else None
    if params is not None and params["genotypes"] != ds.genotypes:
        raise ValueError("Logistic initialization genotype order differs")
    model = build_model(key, ds, params).to(device)
    train = tensor_batch(ds.train, device)
    val = tensor_batch(ds.val, device)
    tau = torch.linspace(0., 1., len(ds.days), device=device)
    epochs = args.latent_epochs if key == "latent" else args.reference_epochs
    min_epoch = args.latent_min_epoch if key == "latent" else args.reference_min_epoch
    if min_epoch > epochs:
        raise ValueError("Minimum epoch is greater than epoch budget")
    slow_params = [model.r_raw, model.K_raw] if key == "pinn" else []
    if key == "latent":
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif key == "pinn":
        slow_ids = {id(p) for p in slow_params}
        optimizer = torch.optim.Adam([{"params": [p for p in model.parameters() if id(p) not in slow_ids], "lr": 1e-3},
                                      {"params": slow_params, "lr": 1e-4}])
        scheduler = None
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None
    best = float("inf")
    best_state = None
    best_epoch = 0
    recent = []
    history = []
    folder = out / "runs" / f"{key}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    n_batch = args.batch_size or len(ds.train)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    for epoch in range(1, epochs+1):
        model.train()
        indices = torch.randperm(len(ds.train), device=device) if n_batch < len(ds.train) else torch.arange(len(ds.train), device=device)
        total = 0.
        for i in range(0, len(ds.train), n_batch):
            choose = indices[i:i+n_batch]
            b = {name: value[choose] for name, value in train.items()}
            physics = key == "pinn" and epoch > args.physics_warmup
            time_grid = tau.unsqueeze(0).expand(len(choose), -1).clone()
            if physics:
                time_grid.requires_grad_(True)
            output = forward(key, model, b, time_grid, physics)
            pred = output["pred"] if key == "latent" else output["prediction"]
            loss = REFERENCE.curve_rmse(pred, b["y"], b["mask"])
            if key == "latent":
                losses = LATENT.physics_losses(model, output, b["env"], {"physic": 2., "r": 0., "ymax": .1, "mono": 0.})
                loss = loss + sum(losses.values())
            else:
                loss = loss + REFERENCE.reference_l2_loss(model, weight=1.)
                if physics:
                    # Preserve the existing reference PINN's time-autograd formulation.
                    loss = loss + sum(REFERENCE.logi_pinn_losses(output, time_grid, b["y"], float(ds.days[-1]), 2.).values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {key} loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if key == "pinn" and not physics:
                for p in slow_params:
                    p.grad = None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(choose)
        if scheduler is not None:
            scheduler.step()
        if epoch % args.eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                prediction = forward(key, model, val, tau)
                prediction = prediction["pred"] if key == "latent" else prediction["prediction"]
                val_rmse = float(REFERENCE.curve_rmse(prediction, val["y"], val["mask"]))
            recent.append(val_rmse)
            recent = recent[-args.val_smooth:]
            smooth = float(np.mean(recent))
            eligible = epoch >= min_epoch and (key != "pinn" or epoch > args.physics_warmup)
            if eligible and len(recent) == args.val_smooth and smooth < best:
                best = smooth
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"state_dict": best_state, "epoch": best_epoch, "selection_val_smooth": best,
                            "model_key": key, "seed": seed}, folder / "best_checkpoint.pt")
            history.append(dict(epoch=epoch, selection_eligible=eligible, train_objective=total/len(ds.train), val_rmse=val_rmse*ds.y_scale,
                                val_rmse_smooth=smooth*ds.y_scale, seconds=time.monotonic()-started))
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            print(f"{key} seed={seed} epoch={epoch}/{epochs} loss={total/len(ds.train):.5f} "
                  f"val={val_rmse*ds.y_scale:.3f} best_epoch={best_epoch} elapsed={time.monotonic()-started:.1f}s", flush=True)
    if best_state is None:
        raise RuntimeError("No checkpoint satisfies the fixed selection protocol")
    model.load_state_dict(best_state)
    model.eval()
    predictions = {}
    with torch.no_grad():
        for name in SPLITS:
            b = tensor_batch(getattr(ds, name), device)
            output = forward(key, model, b, tau)
            predictions[name] = (output["pred"] if key == "latent" else output["prediction"]).cpu().numpy()
    info = dict(n_params=REFERENCE.parameter_count(model), best_epoch=best_epoch,
                selection_val_smooth=best*ds.y_scale, seconds=time.monotonic()-started, epochs=epochs,
                device=str(device), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                peak_gpu_memory_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0,
                batch_size=n_batch, selection=f"validation only; averaging window {args.val_smooth}; PINN after warm-up")
    save_result(out, key, seed, ds, predictions, info, state={"state_dict": best_state, "model_key": key, "seed": seed,
                            "genotypes": ds.genotypes, "y_scale": ds.y_scale, "best_epoch": best_epoch})
    del model, optimizer, train, val
    if device.type == "cuda":
        torch.cuda.empty_cache()


def protocol(args, ds):
    paths = [Path(__file__), ROOT / "code/data.py", ROOT.parent / "wheat/code/model.py", ROOT.parent / "arabidopsis/code/models.py",
             ROOT / "data/processed/height_observations.csv", ROOT / "data/processed/weather_daily.csv"]
    return dict(split=args.split, train_years=ds.train_years, val_year=ds.val_year, test_year=ds.test_year,
                n_genotypes=len(ds.genotypes), y_scale=ds.y_scale, input_features=["genotype one-hot", "calendar time", "station temperature"],
                latent_epochs=args.latent_epochs, reference_epochs=args.reference_epochs,
                latent_min_epoch=args.latent_min_epoch, reference_min_epoch=args.reference_min_epoch,
                physics_warmup=args.physics_warmup, eval_every=args.eval_every, val_smooth=args.val_smooth,
                batch_size=args.batch_size or len(ds.train), environment_statistics="training unique year-days",
                latent_profile="wheat/results/final_config.json: small architecture, lr .01, physics 2, ymax .1",
                reference_profile="Arabidopsis reference architecture; lr .001, ODE lr .0001, direct r/K",
                metric="mean of genotype-year observed-point RMSEs in original relative units",
                test_usage="scored after validation-only checkpoint selection; no test-time fitting",
                temperature_adaptation="daily temperature response trapezoid-integrated; fixed H0=1e-4",
                reference_pinn_derivative="existing autograd(prediction.sum(), time), retained for compatibility",
                pinn_selection="only epochs after physics warm-up are eligible",
                source_sha256={str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/chronological_final_seed1_3")
    parser.add_argument("--models", nargs="+", choices=["process", "rf", "lstm", "pinn", "latent"], default=["process", "rf", "lstm", "pinn", "latent"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="chronological")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent-epochs", type=int, default=1500)
    parser.add_argument("--reference-epochs", type=int, default=3000)
    parser.add_argument("--latent-min-epoch", type=int, default=10)
    parser.add_argument("--reference-min-epoch", type=int, default=10)
    parser.add_argument("--physics-warmup", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--val-smooth", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    ds = make_dataset(split=args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    config = protocol(args, ds)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json.loads(json.dumps(config)):
            raise ValueError("Existing output has a different protocol or source hash; use a new output folder")
    else:
        protocol_path.write_text(json.dumps(config, indent=2) + "\n")
    (args.output / f"command_seed{args.seed}_{args.models[0]}.json").write_text(json.dumps(dict(argv=sys.argv, pid=os.getpid(),
                 cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "")), indent=2)+"\n")
    for key in args.models:
        if key == "process":
            process_baselines(ds, args.output)
        elif (args.output / "runs" / f"{key}_seed{args.seed}" / "result.json").exists():
            print(f"Already complete: {key} seed {args.seed}", flush=True)
        elif key == "rf":
            random_forest(ds, args.output, args.seed)
        else:
            train_neural(ds, args.output, key, args.seed, args)


if __name__ == "__main__":
    main()
