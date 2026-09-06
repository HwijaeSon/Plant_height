"""Export a static, shareable data-audit figure (no model predictions)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    obs = pd.read_csv(ROOT / "data/processed/height_observations.csv")
    means = pd.read_csv(ROOT / "data/processed/height_genotype_means.csv")
    weather = pd.read_csv(ROOT / "data/processed/weather_daily.csv")
    counts = pd.read_csv(ROOT / "reports/yearly_counts.csv")
    colors = ["#247a91", "#d09025", "#7662a5", "#c65559"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), constrained_layout=True)
    for year, color in zip(counts.year, colors):
        d = means[means.year == year]
        q = d.groupby("dap").height_relative.quantile([.25, .5, .75]).unstack()
        axes[0, 0].plot(q.index, q[.5], marker="o", ms=3, color=color, label=str(year))
        axes[0, 0].fill_between(q.index, q[.25], q[.75], color=color, alpha=.10)
        days = np.sort(d.dap.unique())
        axes[0, 1].scatter(days, [year] * len(days), color=color, s=30)
        w = weather[weather.year == year]
        axes[1, 0].plot(w.dap, w.temperature_c, color=color, lw=1.3, alpha=.9, label=str(year))
    axes[0, 0].set(title="Observed relative height: median and IQR", xlabel="Days after planting", ylabel="Relative UAV height (not cm/m)")
    axes[0, 0].legend(ncol=2, frameon=False)
    axes[0, 1].set(title="Actual plant-observation dates (44 flights)", xlabel="Days after planting", yticks=counts.year)
    axes[0, 1].set_ylim(2017.5, 2021.5)
    axes[1, 0].set(title="Daily station temperature: (Tmax + Tmin) / 2", xlabel="Days after planting", ylabel="Temperature (°C)")
    bars = axes[1, 1].bar(counts.year.astype(str), counts.primary_observations, color=colors, width=.6)
    for bar, row in zip(bars, counts.itertuples()):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+200,
                        f"{row.primary_observations:,}\n{row.primary_plots} plots", ha="center", va="bottom", fontsize=10)
    axes[1, 1].set(title="Primary cohort: 402 shared genotypes", ylabel="Observed plot × date targets", ylim=(0, 14500))
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Maize data audit · Sweet et al. (2024)\n3,072 plot trajectories · 31,894 observations · 4 years", fontsize=15)
    out = ROOT / "reports/figures"
    out.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"dataset_overview.{ext}", dpi=180)
    print(out / "dataset_overview.png")


if __name__ == "__main__":
    main()
