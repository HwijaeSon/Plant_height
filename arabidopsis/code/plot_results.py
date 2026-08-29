from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "results"
COLORS = {
    "Logi-ODE": "#8c8c8c",
    "Temp-ODE": "#a6761d",
    "RF": "#e6ab02",
    "LSTM-NN": "#7570b3",
    "Logi-PINN": "#1b9e77",
    "Latent Neural ODE (ours)": "#d95f02",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--genotype", default="Col-0")
    args = parser.parse_args()

    summary_path = args.results / "comparison.csv"
    prediction_path = args.results / "predictions.csv"
    if not summary_path.exists():
        summary_path = args.results / "comparison_seed1_3.csv"
    if not prediction_path.exists():
        prediction_path = args.results / "predictions_seed1_3.csv"
    summary = pd.read_csv(summary_path)
    predictions = pd.read_csv(prediction_path)

    ordered = summary.sort_values("test_rmse_cm_mean")
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.barh(
        ordered["model"],
        ordered["test_rmse_cm_mean"],
        xerr=ordered["test_rmse_cm_sd"],
        color=[COLORS.get(model, "#777777") for model in ordered["model"]],
        alpha=0.9,
        capsize=3,
    )
    axis.invert_yaxis()
    axis.set_xlabel("Test RMSE (cm; mean ± SD across seeds)")
    axis.set_title("Arabidopsis primary inflorescence stem benchmark")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.results / "test_rmse_comparison.png", dpi=220)
    plt.close(fig)

    sample = predictions[
        (predictions["split"] == "test") & (predictions["genotype"] == args.genotype)
    ].copy()
    if sample.empty:
        available = sorted(predictions["genotype"].unique())
        raise ValueError(f"Unknown genotype {args.genotype!r}; choose from {available}")
    ensemble = (
        sample.groupby(
            ["model", "condition", "day_after_sowing"], as_index=False
        )["predicted_length_m"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, condition in zip(axes, ["nAT", "hAT"]):
        observed = sample[(sample["condition"] == condition) & (sample["observed"] == 1)]
        observed = observed.drop_duplicates(
            ["condition", "genotype", "day_after_sowing"]
        )
        axis.scatter(
            observed["day_after_sowing"],
            observed["observed_length_m"] * 100.0,
            color="black",
            marker="o",
            s=45,
            label="test observation",
            zorder=10,
        )
        for model in summary["model"]:
            curve = ensemble[
                (ensemble["condition"] == condition) & (ensemble["model"] == model)
            ]
            if curve.empty:
                continue
            x = curve["day_after_sowing"].to_numpy()
            mean = curve["mean"].to_numpy() * 100.0
            sd = curve["std"].fillna(0.0).to_numpy() * 100.0
            axis.plot(x, mean, color=COLORS.get(model), linewidth=1.8, label=model)
            if np.any(sd > 0):
                axis.fill_between(x, mean - sd, mean + sd, color=COLORS.get(model), alpha=0.09)
        axis.set_title(f"{args.genotype} — {condition}")
        axis.set_xlabel("Days after sowing")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Primary inflorescence stem length (cm)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(args.results / f"sample_{args.genotype}_test_curves.png", dpi=220)
    plt.close(fig)

    print(args.results / "test_rmse_comparison.png")
    print(args.results / f"sample_{args.genotype}_test_curves.png")


if __name__ == "__main__":
    main()
