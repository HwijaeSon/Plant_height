"""Masked, daily-grid inputs for the existing latent ODE and shared baselines.

Targets are relative heights divided by a training-only positive scale.
Multiply predictions/errors by y_scale to recover the released relative unit.
No held-out heights enter model features or normalization statistics.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/maize/processed"


@dataclass
class SequenceSet:
    y: np.ndarray
    mask: np.ndarray
    env: np.ndarray
    s: np.ndarray
    ds: np.ndarray
    g_idx: np.ndarray
    year: np.ndarray
    plot: np.ndarray

    def __len__(self):
        return len(self.y)

    @property
    def n_t(self):
        return self.y.shape[1]

    def subset(self, index):
        return SequenceSet(**{f.name: getattr(self, f.name)[index] for f in fields(self)})


@dataclass
class Dataset:
    train: SequenceSet
    val: SequenceSet
    test: SequenceSet
    train_eval: SequenceSet
    genotypes: list[str]
    env_mean: np.ndarray
    env_std: np.ndarray
    gdd_scale: float
    y_scale: float
    train_years: tuple[int, ...]
    val_year: int
    test_year: int
    days: np.ndarray
    env_columns: tuple[str, ...]
    # Genotype identities are available, but a genomic kinship matrix is not.
    kinship: None = None


def average_replicates(seq: SequenceSet) -> SequenceSet:
    values = {f.name: [] for f in fields(seq)}
    for year, g in sorted(set(zip(seq.year.tolist(), seq.g_idx.tolist()))):
        indexes = np.flatnonzero((seq.year == year) & (seq.g_idx == g))
        count = seq.mask[indexes].sum(axis=0)
        target = (seq.y[indexes] * seq.mask[indexes]).sum(axis=0)
        values["y"].append(np.divide(target, count, out=np.zeros_like(target), where=count > 0))
        values["mask"].append((count > 0).astype(np.float32))
        for name in ["env", "s", "ds", "g_idx", "year"]:
            values[name].append(getattr(seq, name)[indexes[0]])
        values["plot"].append(f"{year}:{g}:mean")
    return SequenceSet(**{name: np.asarray(value) for name, value in values.items()})


def make_dataset(processed_dir: Path = PROCESSED, split: str = "chronological",
                 env_columns: tuple[str, ...] = ("temperature_c",)) -> Dataset:
    processed_dir = Path(processed_dir)
    configs = json.loads((processed_dir / "year_splits.json").read_text())
    config = next((c for c in configs if c["name"] == split), None)
    if config is None:
        raise ValueError(f"Unknown split {split}; choose {[c['name'] for c in configs]}")
    train_years = tuple(config["train_years"])
    val_year, test_year = config["val_year"], config["test_year"]
    obs = pd.read_csv(processed_dir / "height_observations.csv", dtype={"genotype": str})
    weather = pd.read_csv(processed_dir / "weather_daily.csv")
    genotypes = pd.read_csv(processed_dir / "genotypes.csv", dtype={"genotype": str}).genotype.tolist()
    if len(set(genotypes)) != len(genotypes):
        raise ValueError("Non-unique genotype vocabulary")
    days = np.sort(weather.dap.unique()).astype(int)
    if not np.array_equal(days, np.arange(len(days))) or len(days) < 2:
        raise ValueError("Expected consecutive DAPs starting at zero")
    if set(obs.year) != set(train_years) | {val_year, test_year}:
        raise ValueError("Split does not cover the dataset years")
    if not env_columns or not set(env_columns).issubset(weather.columns):
        raise ValueError("Unknown environmental feature")
    training_weather = weather[weather.year.isin(train_years)]
    env_mean = training_weather[list(env_columns)].mean().to_numpy()
    env_std = training_weather[list(env_columns)].std(ddof=0).to_numpy()
    env_std = np.where(env_std > 1e-8, env_std, 1.)
    gdd_scale = float(training_weather.groupby("year").thermal_integral_c.max().mean())
    y_scale = float(obs.loc[obs.year.isin(train_years), "height_relative"].max())
    if not (np.isfinite(y_scale) and y_scale > 0 and np.isfinite(gdd_scale) and gdd_scale > 0):
        raise ValueError("Invalid training normalization scale")
    env_by_year, s_by_year, ds_by_year = {}, {}, {}
    for year, w in weather.groupby("year"):
        w = w.sort_values("dap")
        if not np.array_equal(w.dap.to_numpy(), days):
            raise ValueError(f"Incomplete environment grid for {year}")
        env_by_year[year] = ((w[list(env_columns)].to_numpy() - env_mean) / env_std).astype(np.float32)
        s_by_year[year] = (w.thermal_integral_c.to_numpy() / gdd_scale).astype(np.float32)
        # ds/dtau, with tau = dap / (T-1), exactly as expected by wheat/model.py.
        ds_by_year[year] = (w.daily_gdd_c.to_numpy() * (len(days) - 1) / gdd_scale).astype(np.float32)
    values = {f.name: [] for f in fields(SequenceSet)}
    for (year, plot), group in obs.groupby(["year", "plot_id"], sort=True):
        if group.dap.duplicated().any() or group.genotype.nunique() != 1:
            raise ValueError(f"Invalid plot trajectory {plot}")
        y = np.zeros(len(days), dtype=np.float32)
        mask = np.zeros(len(days), dtype=np.float32)
        indices = group.dap.to_numpy(dtype=int)
        y[indices] = group.height_relative.to_numpy() / y_scale
        mask[indices] = 1
        for name, value in dict(y=y, mask=mask, env=env_by_year[year], s=s_by_year[year], ds=ds_by_year[year],
                                g_idx=genotypes.index(group.genotype.iloc[0]), year=year, plot=plot).items():
            values[name].append(value)
    seq = SequenceSet(**{name: np.asarray(value) for name, value in values.items()})
    train = seq.subset(np.isin(seq.year, train_years))
    val = average_replicates(seq.subset(seq.year == val_year))
    test = average_replicates(seq.subset(seq.year == test_year))
    for batch in [train, val, test]:
        if len(batch) == 0 or not (batch.mask.sum(axis=1) > 0).all():
            raise ValueError("Empty training or evaluation trajectory")
        if set(batch.g_idx) != set(range(len(genotypes))):
            raise ValueError("Split has unseen or absent genotype IDs")
        if not all(np.isfinite(getattr(batch, name)).all() for name in ["y", "env", "s", "ds"]):
            raise ValueError("Nonfinite model input")
    return Dataset(train, val, test, average_replicates(train), genotypes, env_mean, env_std,
                   gdd_scale, y_scale, train_years, val_year, test_year, days, tuple(env_columns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="chronological")
    args = parser.parse_args()
    data = make_dataset(split=args.split)
    out = PROCESSED / "model_ready"
    out.mkdir(exist_ok=True)
    arrays = {f"{name}_{f.name}": getattr(getattr(data, name), f.name)
              for name in ["train", "train_eval", "val", "test"] for f in fields(SequenceSet)}
    np.savez_compressed(out / f"{args.split}.npz", **arrays, days=data.days,
                        genotypes=np.asarray(data.genotypes), y_scale=data.y_scale)
    metadata = {
        "split": args.split, "train_years": data.train_years, "val_year": data.val_year, "test_year": data.test_year,
        "n_genotypes": len(data.genotypes), "day_grid": [int(data.days[0]), int(data.days[-1])],
        "environment_features": data.env_columns, "env_mean": data.env_mean.tolist(), "env_std": data.env_std.tolist(),
        "y_scale": data.y_scale, "gdd_scale_c": data.gdd_scale, "target_unit": "relative height / y_scale",
        "original_unit": "relative UAV height, not cm or m", "days_per_tau": len(data.days) - 1,
        "thermal_time": "trapezoidal integral of source daily GDD_C; not published cumulative GDD",
        "normalization_fit_on": "training years only; environment statistics use unique year-days",
        "batches": {name: {"shape_y": list(getattr(data, name).y.shape),
                            "shape_env": list(getattr(data, name).env.shape),
                            "observed_targets": int(getattr(data, name).mask.sum())}
                    for name in ["train", "train_eval", "val", "test"]},
    }
    (out / f"{args.split}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
