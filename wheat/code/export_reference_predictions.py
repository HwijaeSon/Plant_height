"""Extract the published wheat LSTM-NN and Logi-PINN test trajectories.

The upstream repository stores one CSV per model configuration and seed. This
script selects the configuration used by the paper tables, orders its columns
like the audited latent-model output, and writes one compact plotting artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEFAULT_SOURCE = (
    REPO
    / "legacy/upstream/ETH_data_process/code/pinn_result/multiple_g"
)
DEFAULT_OUTPUT = ROOT / "results/reference/test_predictions_seed1_3.npz"

PATTERNS = {
    "lstm": (
        "pinn_weight_None_test_curves_NN_rerun_finalgenotype_one_hot_encoding"
        "smooth_temp_Falseyear_split_train_lr_0.001_hidden_5_fc_hidden_5_"
        "g_embed5_num_layers_1_l2_1.0_ymax_bound_False_smooth_lossFalse_"
        "seed_{seed}.csv"
    ),
    "pinn": (
        "pinn_weight_2_test_curves_PINN_retun_finalgenotype_one_hot_encoding"
        "smooth_temp_Falseyear_split_train_lr_0.001_hidden_5_fc_hidden_5_"
        "g_embed5_num_layers_1_l2_1.0_ymax_bound_True_smooth_lossFalse_"
        "seed_{seed}.csv"
    ),
}


def trajectory_rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    per_curve = np.sqrt(
        np.sum(np.where(mask, (prediction - target) ** 2, 0.0), axis=1)
        / mask.sum(axis=1)
    )
    return float(per_curve.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    audited = np.load(ROOT / "results/test_predictions_seed1_3.npz", allow_pickle=False)
    genotypes = audited["genotype"].astype(int)
    target = audited["y"].astype(float)
    mask = audited["mask"] > 0
    arrays: dict[str, np.ndarray] = {}
    sources: dict[str, dict[str, object]] = {}

    for model, pattern in PATTERNS.items():
        by_seed = []
        for seed in (1, 2, 3):
            path = args.source / pattern.format(seed=seed)
            frame = pd.read_csv(path, index_col=0)
            prediction = frame[[str(g) for g in genotypes]].to_numpy(float).T
            if prediction.shape != target.shape or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid reference predictions: {path}")
            by_seed.append(prediction)
            sources[f"{model}_seed{seed}"] = {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "trajectory_rmse_on_current_mask_m": trajectory_rmse(
                    prediction, target, mask
                ),
            }
        arrays[model] = np.stack(by_seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        lstm=arrays["lstm"],
        pinn=arrays["pinn"],
        genotype=genotypes,
        seeds=np.array([1, 2, 3]),
    )
    provenance = {
        "upstream_repository": "https://github.com/YingjieShao/PINN_for_plant_height_forecasting",
        "upstream_commit": "3da92f51f42d3fde5e06f6fc8ce8f233490a389c",
        "selection": "split 0; one-hot genotype; hidden/fc/embedding 5; l2 1.0; seeds 1-3",
        "note": "Used for trajectory visualization; table RMSE values retain the upstream published summaries.",
        "sources": sources,
    }
    args.output.with_name("test_predictions_seed1_3_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
