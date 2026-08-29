"""Create manuscript figures from the audited experiment outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)


def wheat_examples():
    results = np.load(ROOT / "wheat/results/test_predictions_seed1_3.npz")
    pred = results["pred"]
    genotypes = results["genotype"]
    raw = pd.read_csv(
        ROOT / "wheat/data/align_height_env_same_length.csv"
    )

    # Reproducible random sample, selected once with NumPy seed 2026.
    selected = [33, 335, 133]
    days = np.arange(115, 115 + pred.shape[2])
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), sharey=True)
    for ax, genotype in zip(axes, selected):
        i = int(np.flatnonzero(genotypes == genotype)[0])
        observed = raw[
            (raw["year_site.harvest_year"] == 2021)
            & (raw["genotype.id"] == genotype)
            & (raw["day_after_start_measure"] >= 115)
        ].dropna(subset=["value"])
        observed = observed.groupby("day_after_start_measure", as_index=False)["value"].mean()
        first, last = int(observed.day_after_start_measure.min()), int(observed.day_after_start_measure.max())
        within = (days >= first) & (days <= last)
        mean = pred[:, i].mean(axis=0)
        sd = pred[:, i].std(axis=0, ddof=1)
        ax.fill_between(days[within], mean[within] - sd[within], mean[within] + sd[within],
                        color="#0072B2", alpha=0.18)
        ax.plot(days[within], mean[within], color="#0072B2", lw=2,
                label="Tuned Latent Neural ODE")
        ax.scatter(observed.day_after_start_measure, observed.value, s=13,
                   color="black", zorder=3,
                   label="Observed")
        ax.set_title(f"Genotype {genotype}")
        ax.set_xlabel("Aligned day index")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Plant height (m)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "wheat_test_curves_no_extrapolation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "wheat_test_curves_no_extrapolation.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def arabidopsis_examples():
    data = pd.read_csv(
        ROOT / "arabidopsis/results/predictions_seed1_3.csv"
    )
    data = data[(data["split"] == "test") & (data["genotype"] == "Col-0")]
    stochastic = {"Latent Neural ODE (ours)", "RF", "Logi-PINN", "LSTM-NN"}
    data = data[(~data.model.isin(stochastic)) | data.seed.isin([1, 2, 3])]
    order = ["Latent Neural ODE (ours)", "RF", "Logi-PINN", "LSTM-NN",
             "Temp-ODE", "Logi-ODE"]
    label = {
        "Latent Neural ODE (ours)": "Latent Neural ODE",
        "RF": "Random forest",
        "Logi-PINN": "Logi-PINN",
        "LSTM-NN": "LSTM-NN",
        "Temp-ODE": "Temperature ODE",
        "Logi-ODE": "Logistic ODE",
    }
    colors = ["#0072B2", "#999999", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.55), sharey=True)
    for ax, condition in zip(axes, ["nAT", "hAT"]):
        panel = data[data.condition == condition]
        observed = (
            panel[panel.observed == 1]
            .groupby("day_after_sowing", as_index=False).observed_length_m.mean()
        )
        first, last = int(observed.day_after_sowing.min()), int(observed.day_after_sowing.max())
        for model, color in zip(order, colors):
            curve = (
                panel[(panel.model == model)
                      & panel.day_after_sowing.between(first, last)]
                .groupby("day_after_sowing", as_index=False).predicted_length_m.mean()
            )
            ax.plot(curve.day_after_sowing, curve.predicted_length_m * 100,
                    lw=1.8, color=color, label=label[model])
        ax.scatter(observed.day_after_sowing, observed.observed_length_m * 100,
                   color="black", s=28, zorder=5, label="Observed")
        ax.set_xlim(first - 0.5, last + 0.5)
        ax.set_title(f"Col-0, {condition}")
        ax.set_xlabel("Days after sowing")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Primary inflorescence stem length (cm)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=0.24, wspace=0.08)
    fig.savefig(OUT / "arabidopsis_col0_no_extrapolation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "arabidopsis_col0_no_extrapolation.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    wheat_examples()
    arabidopsis_examples()
