"""Create consistently styled manuscript figures from audited predictions."""

from pathlib import Path
import importlib.util
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from layout import artifact
OUT = ROOT / "outputs/figures"
PROFILE = json.loads((ROOT / "src/configs/paper.json").read_text())

# Fixed across all three manuscript figures. The sequence is arranged so a
# four-column Matplotlib legend has the same two-row order as the paper legend.
MODEL_ORDER = [
    "Latent Neural ODE",
    "Random forest",
    "Logi-PINN",
    "LSTM-NN",
    "Temperature ODE",
    "Logistic ODE",
]
COLORS = {
    "Latent Neural ODE": "#0072B2",
    "Random forest": "#999999",
    "Logi-PINN": "#D55E00",
    "LSTM-NN": "#009E73",
    "Temperature ODE": "#CC79A7",
    "Logistic ODE": "#E69F00",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
)


def legend_handles():
    handles = [
        Line2D(
            [0], [0], color=COLORS[name], lw=2.2,
            label="PhytoODE (ours)" if name == "Latent Neural ODE" else name,
        )
        for name in MODEL_ORDER
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle="none",
            markersize=6,
            label="Observed",
        )
    )
    return handles


def add_shared_legend(fig):
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=2.8,
        columnspacing=2.0,
    )


def draw_curves(ax, x, support, predictions):
    for name in MODEL_ORDER:
        ax.plot(
            x[support],
            predictions[name][support],
            color=COLORS[name],
            lw=2.2 if name == "Latent Neural ODE" else 1.7,
            zorder=3 if name == "Latent Neural ODE" else 2,
        )


def save_figure(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def adopted_test_predictions(dataset):
    """Read the audited predictions attached to the adopted checkpoint seeds."""
    config = PROFILE["datasets"][dataset]
    checkpoints = config["models"]["phytoode"]["checkpoints"]
    arrays = []
    metadata = None
    for seed in sorted(checkpoints, key=int):
        result = json.loads((artifact(dataset, "phytoode", seed, "result.json")).read_text())
        assert (result["lambda_ode"], result["lambda_k"]) == (config["lambda_ode"], config["lambda_k"])
        assert result["checkpoint_sha256"] == checkpoints[str(seed)]["sha256"]
        with np.load(artifact(dataset, "phytoode", seed, "predictions.npz"), allow_pickle=False) as saved:
            current = {key: saved[key] for key in ("test_target", "test_mask", "test_genotype")}
            if metadata is not None:
                for key in current:
                    np.testing.assert_array_equal(current[key], metadata[key])
            metadata = current
            arrays.append(saved["test"])
    return np.stack(arrays), metadata


def wheat_examples():
    ours = np.load(ROOT / "results/wheat/test_targets.npz")
    reference = np.load(
        ROOT / "results/wheat/reference_test_predictions_seed1_3.npz"
    )
    genotypes = ours["genotype"].astype(int)
    np.testing.assert_array_equal(reference["genotype"], genotypes)
    adopted, metadata = adopted_test_predictions("wheat")
    np.testing.assert_array_equal(metadata["test_genotype"], genotypes)
    np.testing.assert_array_equal(metadata["test_mask"], ours["mask"])
    np.testing.assert_array_equal(metadata["test_target"], ours["y"])


    def load_baseline(key, seed):
        return np.load(
            artifact("wheat", key, seed, "predictions.npz")
        )["test"]

    predictions = {
        "Latent Neural ODE": adopted.mean(axis=0),
        "Random forest": np.mean(
            [load_baseline("rf", seed) for seed in (1, 2, 3)], axis=0
        ),
        "Logi-PINN": reference["pinn"].mean(axis=0),
        "LSTM-NN": reference["lstm"].mean(axis=0),
        "Temperature ODE": load_baseline("temperature", 0),
        "Logistic ODE": load_baseline("logistic", 0),
    }
    shape = ours["y"].shape
    if any(value.shape != shape for value in predictions.values()):
        raise ValueError("Wheat prediction shapes do not match")

    raw = pd.read_csv(ROOT / "data/wheat/align_height_env_same_length.csv")
    selected = [33, 335, 133]  # NumPy seed 2026; fixed before plotting.
    days = np.arange(115, 115 + shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.45), sharey=True)
    for ax, genotype in zip(axes, selected):
        i = int(np.flatnonzero(genotypes == genotype)[0])
        observed = raw[
            (raw["year_site.harvest_year"] == 2021)
            & (raw["genotype.id"] == genotype)
            & (raw["day_after_start_measure"] >= 115)
        ].dropna(subset=["value"])
        observed = observed.groupby("day_after_start_measure", as_index=False)[
            "value"
        ].mean()
        first = int(observed.day_after_start_measure.min())
        last = int(observed.day_after_start_measure.max())
        support = (days >= first) & (days <= last)
        draw_curves(
            ax,
            days,
            support,
            {name: value[i] for name, value in predictions.items()},
        )
        ax.scatter(
            observed.day_after_start_measure,
            observed.value,
            s=20,
            color="black",
            zorder=5,
        )
        ax.set_title(f"Genotype {genotype}")
        ax.set_xlabel("Aligned day index")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Plant height (m)")
    add_shared_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.80), w_pad=1.2)
    save_figure(fig, "wheat_test_curves_no_extrapolation")


def arabidopsis_examples():
    adopted, metadata = adopted_test_predictions("arabidopsis")
    data = pd.read_csv(ROOT / "results/arabidopsis/predictions_seed1_3.csv")
    data = data[(data["split"] == "test") & (data["genotype"] == "Col-0")]
    stochastic = {"Latent Neural ODE (ours)", "RF", "Logi-PINN", "LSTM-NN"}
    data = data[(~data.model.isin(stochastic)) | data.seed.isin([1, 2, 3])]
    source_names = {
        "Latent Neural ODE": "Latent Neural ODE (ours)",
        "Random forest": "RF",
        "Logi-PINN": "Logi-PINN",
        "LSTM-NN": "LSTM-NN",
        "Temperature ODE": "Temp-ODE",
        "Logistic ODE": "Logi-ODE",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.65), sharey=True)
    for ax, condition in zip(axes, ["nAT", "hAT"]):
        panel = data[data.condition == condition]
        observed = (
            panel[panel.observed == 1]
            .groupby("day_after_sowing", as_index=False)
            .observed_length_m.mean()
        )
        first = int(observed.day_after_sowing.min())
        last = int(observed.day_after_sowing.max())
        for name in MODEL_ORDER:
            curve = (
                panel[
                    (panel.model == source_names[name])
                    & panel.day_after_sowing.between(first, last)
                ]
                .groupby("day_after_sowing", as_index=False)
                .predicted_length_m.mean()
            )
            if name == "Latent Neural ODE":
                curve = pd.DataFrame({"day_after_sowing": np.arange(first, last+1)})
                observed_mask = np.isin(np.arange(27, 50), observed.day_after_sowing)
                matches = ((metadata["test_genotype"] == "Col-0")
                           & np.all(metadata["test_mask"].astype(bool) == observed_mask, axis=1))
                assert matches.sum() == 1
                index = int(np.flatnonzero(matches)[0])
                np.testing.assert_allclose(metadata["test_target"][index, observed_mask],
                                           observed.observed_length_m, atol=1e-8)
                curve["predicted_length_m"] = adopted[:, index].mean(axis=0)[
                    curve.day_after_sowing.to_numpy(dtype=int) - 27]
            ax.plot(
                curve.day_after_sowing,
                curve.predicted_length_m * 100,
                color=COLORS[name],
                lw=2.2 if name == "Latent Neural ODE" else 1.7,
                zorder=3 if name == "Latent Neural ODE" else 2,
            )
        ax.scatter(
            observed.day_after_sowing,
            observed.observed_length_m * 100,
            color="black",
            s=28,
            zorder=5,
        )
        ax.set_xlim(first - 0.5, last + 0.5)
        ax.set_title(f"Col-0, {condition}")
        ax.set_xlabel("Days after sowing")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Primary inflorescence stem length (cm)")
    add_shared_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.80), w_pad=1.2)
    save_figure(fig, "arabidopsis_col0_no_extrapolation")


def maize_examples():
    assert (PROFILE["datasets"]["maize"]["lambda_ode"],
            PROFILE["datasets"]["maize"]["lambda_k"]) == (0.5, 0.5)
    spec = importlib.util.spec_from_file_location(
        "paper_maize_data", ROOT / "src/maize/data.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    dataset = module.make_dataset(split="chronological")
    examples = pd.read_csv(ROOT / "results/maize/figure_genotypes.csv", dtype={"genotype": str})

    def load(key, seed):
        return np.load(artifact("maize", key, seed, "predictions.npz"))["test"]

    predictions = {
        "Latent Neural ODE": np.mean(
            [load("phytoode", seed) for seed in (1, 2, 3)], axis=0
        ),
        "Random forest": np.mean(
            [load("rf", seed) for seed in (1, 2, 3)], axis=0
        ),
        "Logi-PINN": np.mean(
            [load("pinn", seed) for seed in (1, 2, 3)], axis=0
        ),
        "LSTM-NN": np.mean(
            [load("lstm", seed) for seed in (1, 2, 3)], axis=0
        ),
        "Temperature ODE": load("temperature", 0),
        "Logistic ODE": load("logistic", 0),
    }

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.65), sharex=True, sharey=True)
    ordering = [0, 2, 4, 1, 3, 5]
    for ax, case in zip(axes.flat, examples.iloc[ordering].itertuples()):
        i = int(case.test_row)
        observed = np.flatnonzero(dataset.test.mask[i])
        support = np.arange(observed[0], observed[-1] + 1)
        scaled = {
            name: value[i] * dataset.y_scale for name, value in predictions.items()
        }
        draw_curves(ax, dataset.days, support, scaled)
        ax.scatter(
            dataset.days[observed],
            dataset.test.y[i, observed] * dataset.y_scale,
            color="black",
            s=22,
            zorder=5,
        )
        ax.set_title(f"Genotype {case.genotype}")
        ax.grid(alpha=0.18)
    for ax in axes[-1]:
        ax.set_xlabel("Days after planting")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative UAV height")
    add_shared_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.87), h_pad=1.2, w_pad=1.0)
    save_figure(fig, "maize_test_predictions")


if __name__ == "__main__":
    wheat_examples()
    arabidopsis_examples()
    maize_examples()
