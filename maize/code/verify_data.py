"""Scientific data checks, including held-out-data perturbation and model API smoke check."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from data import make_dataset, PROCESSED, ROOT
from prepare_data import read_table, genotype_label


def main() -> None:
    obs = pd.read_csv(PROCESSED / "height_observations.csv", dtype={"genotype": str})
    source = read_table(3)
    # Check every target against its exact, unchanged supplementary cell.
    for row in obs.itertuples():
        date = pd.Timestamp(row.date).strftime("%m%d%Y")
        original = source.iloc[row.source_row - 3]
        if genotype_label(original.Genotype) != row.genotype:
            raise AssertionError("Changed genotype label")
        np.testing.assert_allclose(float(original[date]), row.height_relative, rtol=1e-9)
    data = make_dataset()
    assert len(data.genotypes) == 402
    assert len(obs) == 31894 and obs.plot_id.nunique() == 3072
    assert data.train.y.shape == (1533, 96)
    assert data.val.y.shape == data.test.y.shape == (402, 96)
    assert not set(data.train.plot) & set(obs.loc[obs.year.isin([2020, 2021]), "plot_id"])
    assert int(data.train.mask.sum()) == len(obs[obs.year.isin(data.train_years)])
    assert not set(obs.genotype) & {"B73", "PH207", "ND259", "LH132 - FILL B73"}
    for batch in [data.train, data.train_eval, data.val, data.test]:
        assert (batch.y[batch.mask == 0] == 0).all()
        assert (batch.mask[:, 0] == 0).all()  # no invented planting-day targets
        integral = np.cumsum((batch.ds[:, :-1] + batch.ds[:, 1:]) / (2 * 95), axis=1)
        np.testing.assert_allclose(integral, batch.s[:, 1:], atol=5e-7)
    means = pd.read_csv(PROCESSED / "height_genotype_means.csv", dtype={"genotype": str})
    for batch in [data.train_eval, data.val, data.test]:
        for i, (year, g) in enumerate(zip(batch.year, batch.g_idx)):
            expected = means[(means.year == year) & (means.genotype_index == g)]
            assert int(batch.mask[i].sum()) == len(expected)
            np.testing.assert_allclose(batch.y[i, expected.dap] * data.y_scale,
                                       expected.height_relative, rtol=2e-7)
    with tempfile.TemporaryDirectory(prefix="maize-leakage-check-") as folder:
        temp = Path(folder)
        for name in ["height_observations.csv", "weather_daily.csv", "year_splits.json", "genotypes.csv"]:
            shutil.copy2(PROCESSED / name, temp / name)
        changed = obs.copy()
        changed.loc[~changed.year.isin(data.train_years), "height_relative"] *= 7
        changed.to_csv(temp / "height_observations.csv", index=False)
        weather = pd.read_csv(temp / "weather_daily.csv")
        weather.loc[~weather.year.isin(data.train_years), "temperature_c"] += 20
        weather.to_csv(temp / "weather_daily.csv", index=False)
        altered = make_dataset(temp)
        np.testing.assert_array_equal(data.train.y, altered.train.y)
        np.testing.assert_array_equal(data.train.env, altered.train.env)
        np.testing.assert_array_equal(data.env_mean, altered.env_mean)
        np.testing.assert_array_equal(data.env_std, altered.env_std)
        assert data.y_scale == altered.y_scale and data.gdd_scale == altered.gdd_scale
        assert not np.array_equal(data.val.y, altered.val.y)
        assert not np.array_equal(data.val.env, altered.val.env)
    configs = json.loads((PROCESSED / "year_splits.json").read_text())
    assert len(configs) == 12
    assert len({(c["val_year"], c["test_year"]) for c in configs}) == 12
    for c in configs:
        assert len(set(c["train_years"]) | {c["val_year"], c["test_year"]}) == 4
    arrays = np.load(PROCESSED / "model_ready/chronological.npz", allow_pickle=False)
    np.testing.assert_array_equal(arrays["train_y"], data.train.y)
    # A forward pass only; this does not train or compare models.
    import torch
    spec = importlib.util.spec_from_file_location("wheat_height_model", ROOT.parent / "wheat/code/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.LatentODEHeightModel(n_genotypes=len(data.genotypes), env_dim=1,
                                        days_per_tau=95, use_physics=False, time_mode="thermal_rate")
    with torch.no_grad():
        pred = model(torch.as_tensor(data.train.g_idx[:2], dtype=torch.long),
                     torch.as_tensor(data.train.env[:2]), torch.as_tensor(data.train.s[:2]),
                     torch.as_tensor(data.train.ds[:2]))["pred"]
    assert pred.shape == (2, 96) and torch.isfinite(pred).all()
    checks = ["all 31,894 targets match supplementary cells", "plot/year separation and missing-value masks",
              "replicate means match exported evaluation table", "RK4 thermal rate/integral consistency",
              "held-out height/weather perturbation leaves training inputs and scalers unchanged",
              "12 distinct disjoint year splits", "NPZ loads without pickle and matches loader",
              "existing wheat latent ODE accepts maize tensors (forward only)"]
    (ROOT / "reports/validation.json").write_text(json.dumps({"status": "passed", "checks": checks}, indent=2) + "\n")
    print("PASS: " + "; ".join(checks))


if __name__ == "__main__":
    main()
