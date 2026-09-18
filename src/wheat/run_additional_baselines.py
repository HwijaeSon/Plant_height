"""Fill the wheat comparison with training-only logistic, temperature ODE and RF fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import sklearn
from sklearn.ensemble import RandomForestRegressor

from data import make_dataset

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
SPLITS = {"train": "train_eval", "val": "val", "test": "test"}
H0 = 1e-4  # metres; identical initial-height convention to the Arabidopsis baseline


def logistic(exposure, r, K):
    return K / (1 + (K/H0-1) * np.exp(np.clip(-r*exposure, -80, 80)))


def response(celsius, lower_k, upper_k):
    kelvin = np.asarray(celsius, dtype=np.float64) + 273.15
    return 1 / (1 + np.exp(np.clip(2000/kelvin - 2000/lower_k, -60, 60))
                + np.exp(np.clip(60000/upper_k - 60000/kelvin, -60, 60)))


def exposure(celsius, lower_k, upper_k):
    values = response(celsius, lower_k, upper_k)
    return np.concatenate([np.zeros((len(values), 1)),
        np.cumsum((values[:, :-1]+values[:, 1:])/2, axis=1)], axis=1)


def temperature(batch, ds):
    return batch.env[:, :, 0] * ds.env_std[0] + ds.env_mean[0]


def process_predict(batch, ds, params):
    if "lower_k" in params:
        elapsed = exposure(temperature(batch, ds), params["lower_k"], params["upper_k"])
    else:
        elapsed = np.arange(batch.n_t)[None, :]
    return logistic(elapsed, np.asarray(params["r"])[batch.g_idx, None],
                    np.asarray(params["K"])[batch.g_idx, None])


def features(batch, rows, cols, n_genotypes):
    x = np.zeros((len(rows), n_genotypes+2), dtype=np.float32)
    x[np.arange(len(rows)), batch.g_idx[rows]] = 1
    x[:, n_genotypes] = batch.env[rows, cols, 0]
    x[:, n_genotypes+1] = cols / (batch.n_t-1)
    return x


def forest_predict(model, batch, n_genotypes):
    rows, cols = np.indices(batch.y.shape)
    return model.predict(features(batch, rows.ravel(), cols.ravel(), n_genotypes)).reshape(batch.y.shape)


def score(pred, batch):
    if pred.shape != batch.y.shape or not np.isfinite(pred).all():
        raise ValueError("Invalid predictions")
    mask = batch.mask > 0
    error = np.asarray(pred, dtype=float) - batch.y
    rmse = np.sqrt((error**2*mask).sum(axis=1)/mask.sum(axis=1))
    mae = (np.abs(error)*mask).sum(axis=1)/mask.sum(axis=1)
    return dict(rmse=float(rmse.mean()), mae=float(mae.mean()),
                pooled_rmse=float(np.sqrt((error[mask]**2).mean()))), rmse, mae


def save_run(out, ds, model, key, seed, predictor, seconds, n_params, fit_info):
    folder = out / "runs" / f"{key}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(model=model, model_key=key, seed=seed, seconds=seconds, n_params=n_params,
                  target_unit="m", **fit_info)
    predictions, curves = {}, []
    for split, attr in SPLITS.items():
        batch = getattr(ds, attr)
        predictions[attr] = predictor(batch)
        metrics, rmse, mae = score(predictions[attr], batch)
        result.update({f"{split}_{key}": value for key, value in metrics.items()})
        for i in range(len(batch)):
            curves.append(dict(model=model, seed=seed, split=split, genotype=int(ds.genotypes[batch.g_idx[i]]),
                year=int(batch.year[i]), rmse=rmse[i], mae=mae[i], n_scored=int(batch.mask[i].sum())))
    np.savez_compressed(folder / "predictions.npz", **predictions)
    pd.DataFrame(curves).to_csv(folder / "per_curve_metrics.csv", index=False)
    (folder / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"DONE {model} seed={seed}: train={result['train_rmse']:.6f} "
          f"val={result['val_rmse']:.6f} test={result['test_rmse']:.6f} m", flush=True)
    return result, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/wheat-baselines")
    args = parser.parse_args()
    out = args.output
    if (out / "all_runs.csv").exists():
        raise FileExistsError(f"Completed results already exist; choose a new --output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "src/configs/wheat.json"
    config = json.loads(config_path.read_text())
    ds = make_dataset(data_path=REPO/config["data"], kinship_path=REPO/config["kinship"],
        split=config["split"], start_day=config["start_day"], env_cols=config["env_cols"],
        fill_in_na_at_start=not config["no_fill_na_start"], val_year=config["val_year"], test_year=config["test_year"],
        add_gdd=config["add_gdd"], gdd_base=config["gdd_base"], genotypes=config["genotypes"])
    source_paths = [Path(__file__).resolve(), ROOT/"src/wheat/data.py", config_path, REPO/config["data"], REPO/config["kinship"]]
    hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    protocol = dict(train_years=list(ds.train_years), val_year=ds.val_year, test_year=ds.test_year,
        n_genotypes=len(ds.genotypes), n_training_plots=len(ds.train), h0_m=H0,
        time="days since start of cropped model window (DAP 115), 0..169",
        height_unit="m; original unscaled wheat target", fill_in_na_at_start=not config["no_fill_na_start"],
        training_objective="per-plot balanced masked MSE; point weights = 1 / scored points in plot",
        evaluation="mean of per-genotype-year masked RMSE/MAE; replicate-averaged curves",
        selection="all fitting/initialization uses training data only; no validation or test tuning",
        logistic=dict(rate_bounds=[1e-4, .999], K_bounds_m=[.01, 1.999], initial_rates=[.08, .15, .3], max_nfev=2000),
        temperature=dict(rate_bounds=[1e-4, 2.], K_bounds_m=[.01, 1.999], lower_k_bounds=[275., 300.],
            upper_k_bounds=[301., 330.], initial_thresholds_k=[292., 303.], integration="daily trapezoid", max_nfev=5000),
        rf=dict(n_estimators=100, seeds=[1, 2, 3], n_jobs=2, features=["genotype one-hot", "temperature", "time"]),
        implementation_reference="same CPU baseline family and fitting rules as src/maize/run_experiment.py, applied to wheat in metres",
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__), source_sha256=hashes)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    b, n = ds.train, len(ds.genotypes)
    rows, cols = np.where(b.mask > 0)
    geno, target = b.g_idx[rows], b.y[rows, cols]
    weights = 1 / np.sqrt(b.mask.sum(axis=1)[rows])
    all_runs, all_curves = [], []

    started = time.monotonic()
    r, K, fits = np.zeros(n), np.zeros(n), []
    for g in range(n):
        use = geno == g
        candidates = [least_squares(lambda p: (logistic(cols[use], p[0], p[1])-target[use])*weights[use],
            [initial_rate, max(.1, float(target[use].max()))], bounds=([1e-4, .01], [.999, 1.999]),
            max_nfev=2000, ftol=1e-10, xtol=1e-10, gtol=1e-10) for initial_rate in [.08, .15, .3]]
        fit = min(candidates, key=lambda f: f.cost)
        if not fit.success:
            raise RuntimeError(f"Logistic fit failed: {ds.genotypes[g]}")
        r[g], K[g] = fit.x
        fits.append(dict(genotype=int(ds.genotypes[g]), success=bool(fit.success), cost=float(fit.cost), nfev=int(fit.nfev)))
    logistic_params = dict(r=r.tolist(), K=K.tolist(), h0_m=H0, genotypes=list(map(int, ds.genotypes)), fits=fits)
    (out / "logistic_parameters.json").write_text(json.dumps(logistic_params, indent=2)+"\n")
    result, curves = save_run(out, ds, "Logi-ODE", "logistic", 0, lambda batch: process_predict(batch, ds, logistic_params),
        time.monotonic()-started, 2*n, dict(converged=True))
    all_runs.append(result); all_curves.extend(curves)

    started = time.monotonic()
    years = sorted(set(b.year))
    weather = np.stack([temperature(b, ds)[np.flatnonzero(b.year == y)[0]] for y in years])
    obs_year = np.searchsorted(years, b.year[rows])
    def residual(p):
        elapsed = exposure(weather, p[-2], p[-1])
        return (logistic(elapsed[obs_year, cols], p[geno], p[n+geno])-target)*weights
    sparse = lil_matrix((len(target), 2*n+2), dtype=int)
    ix = np.arange(len(target))
    sparse[ix, geno], sparse[ix, n+geno], sparse[:, -2:] = 1, 1, 1
    init_response = float(response(weather, 292., 303.).mean())
    initial = np.r_[np.clip(r/init_response, .001, 1.9), K, 292., 303.]
    fit = least_squares(residual, initial,
        bounds=(np.r_[np.full(n, 1e-4), np.full(n, .01), 275., 301.],
                np.r_[np.full(n, 2.), np.full(n, 1.999), 300., 330.]),
        jac_sparsity=sparse.tocsr(), max_nfev=5000, ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale="jac")
    if not fit.success:
        raise RuntimeError(f"Temperature fit failed: {fit.message}")
    temp_params = dict(r=fit.x[:n].tolist(), K=fit.x[n:2*n].tolist(), lower_k=float(fit.x[-2]), upper_k=float(fit.x[-1]),
        h0_m=H0, genotypes=list(map(int, ds.genotypes)), success=bool(fit.success), cost=float(fit.cost),
        nfev=int(fit.nfev), optimality=float(fit.optimality), message=str(fit.message))
    (out / "temperature_parameters.json").write_text(json.dumps(temp_params, indent=2)+"\n")
    result, curves = save_run(out, ds, "Temp-ODE", "temperature", 0, lambda batch: process_predict(batch, ds, temp_params),
        time.monotonic()-started, 2*n+2, dict(converged=True, nfev=int(fit.nfev)))
    all_runs.append(result); all_curves.extend(curves)

    for seed in [1, 2, 3]:
        started = time.monotonic()
        forest = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=2, criterion="squared_error")
        forest.fit(features(b, rows, cols, n), target, sample_weight=weights**2)
        folder = out / "runs" / f"rf_seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(forest, folder / "forest.joblib", compress=3)
        result, curves = save_run(out, ds, "RF", "rf", seed, lambda batch: forest_predict(forest, batch, n),
            time.monotonic()-started, None, dict(n_trees=100, n_tree_nodes=sum(t.tree_.node_count for t in forest.estimators_)))
        all_runs.append(result); all_curves.extend(curves)

    # Verify restored models, exported metrics, input hashes and target independence.
    for result in all_runs:
        key, seed = result["model_key"], result["seed"]
        if key == "rf":
            model = joblib.load(out / "runs" / f"rf_seed{seed}" / "forest.joblib")
            predict = lambda batch: forest_predict(model, batch, n)
        else:
            params = json.loads((out / f"{key}_parameters.json").read_text())
            predict = lambda batch: process_predict(batch, ds, params)
        with np.load(out / "runs" / f"{key}_seed{seed}" / "predictions.npz", allow_pickle=False) as saved:
            for split, attr in SPLITS.items():
                batch = getattr(ds, attr)
                np.testing.assert_allclose(predict(batch), saved[attr], rtol=1e-12, atol=1e-12)
                values, _, _ = score(saved[attr], batch)
                for metric, value in values.items():
                    np.testing.assert_allclose(value, result[f"{split}_{metric}"], atol=1e-12)
            probe = ds.test.subset(np.arange(3))
            expected = predict(probe)
            probe.y[:] = np.nan
            probe.mask[:] = 0
            np.testing.assert_allclose(predict(probe), expected, rtol=1e-12, atol=1e-12)
    constant = np.full((1, 170), 20.)
    np.testing.assert_allclose(exposure(constant, 292., 303.),
        np.arange(170)[None, :] * response(20., 292., 303.), rtol=1e-12, atol=1e-12)
    for path, digest in hashes.items():
        if hashlib.sha256((REPO/path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed during fitting: {path}")
    pd.DataFrame(all_runs).to_csv(out / "all_runs.csv", index=False)
    pd.DataFrame(all_curves).to_csv(out / "per_curve_metrics.csv", index=False)
    (out / "validation.json").write_text(json.dumps(dict(status="passed", complete_runs=5,
        source_hashes="unchanged", process_fits="all converged", predictions="restored models match all saved arrays",
        metrics="recomputed from saved arrays for all splits", holdout_targets="mutations do not affect prediction",
        temperature_integration="constant-temperature analytical check passed"), indent=2)+"\n")
    print(f"Verified and saved all five runs: {out}", flush=True)


if __name__ == "__main__":
    main()
