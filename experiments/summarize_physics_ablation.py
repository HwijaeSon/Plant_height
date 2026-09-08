"""Audit completed ablations and export paired metrics, manuscript tables, plots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_physics_ablation import DEFAULT_OUTPUT, ROOT, DATASETS, VARIANTS, references, verify_control

NAMES = dict(full="PhytoODE", no_ode_residual="Without ODE residual",
             no_biological_loss="Without biological losses")
DATASET_NAMES = dict(wheat="Wheat", maize="Maize", arabidopsis="Arabidopsis")
UNITS = dict(wheat="m", maize="relative UAV height", arabidopsis="cm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out / "manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"A frozen source changed: {path}")
    controls = [verify_control(out, dataset) for dataset in DATASETS]
    denominators = pd.read_csv(ROOT / "paper/relative_errors/denominators.csv")
    denom = {(row.dataset, row.split): row.mean_target for row in denominators.itertuples()}
    rows, epoch_rows, provenance = [], [], {}
    for ref in references():
        epoch_rows.append({key: ref[key] for key in ("dataset", "variant", "seed", "best_epoch")})
        for split in ("train", "val", "test"):
            error = ref[split+"_rmse"]
            rows.append(dict(dataset=ref["dataset"], variant="full", seed=ref["seed"], split=split,
                rmse=error, relative_error=error/denom[ref["dataset"], split]*100,
                unit=UNITS[ref["dataset"]], source="frozen manuscript result"))
    for dataset in DATASETS:
        control_config = json.loads((out / dataset / "full_seed1/config.json").read_text())
        for seed in (1, 2, 3):
            initial_hashes = []
            for variant in VARIANTS:
                folder = out / dataset / f"{variant}_seed{seed}"
                result = json.loads((folder / "result.json").read_text())
                config = json.loads((folder / "config.json").read_text())
                selection = json.loads((folder / "selection_frozen.json").read_text())
                assert result["status"] == "complete" and result["test_evaluations"] == 1
                assert selection["test_evaluations"] == 0
                assert result["epochs"] == config["epochs"]
                initial_hashes.append(config["initial_state_sha256"])
                for key in ("model", "epochs", "lr", "weight_decay", "eval_every", "min_epoch", "val_smooth", "permute_training"):
                    assert config[key] == control_config[key], (dataset, seed, variant, key)
                assert config["effective_weights"]["physic"] == 0
                assert config["effective_weights"]["ymax"] == (0 if variant == "no_biological_loss" else control_config["ymax"])
                history = pd.read_csv(folder / "history.csv")
                assert int(history.epoch.iloc[-1]) == config["epochs"]
                # Recompute the selection score from the recorded validation series.
                smooth = history.val_rmse.rolling(config["val_smooth"], min_periods=config["val_smooth"]).mean()
                eligible = history.epoch.ge(config["min_epoch"]) & smooth.notna()
                selected_epoch = int(history.loc[smooth[eligible].idxmin(), "epoch"])
                assert selected_epoch == result["best_epoch"]
                if variant == "no_biological_loss":
                    np.testing.assert_allclose(history.objective, history.data_loss, rtol=0, atol=0)
                else:
                    np.testing.assert_allclose(history.L_m, 0, rtol=0, atol=0)
                epoch_rows.append(dict(dataset=dataset, variant=variant, seed=seed, best_epoch=selected_epoch))
                digest = hashlib.sha256((folder / "checkpoint.pt").read_bytes()).hexdigest()
                assert digest == selection["checkpoint_sha256"] == result["checkpoint_sha256"]
                with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
                    for split, key in (("train", "train_eval"), ("val", "val"), ("test", "test")):
                        pred, y = saved[key].astype(float), saved[key+"_target"].astype(float)
                        mask = saved[key+"_mask"].astype(bool)
                        assert pred.shape == y.shape == mask.shape and np.isfinite(pred).all()
                        rmse = np.sqrt((((pred-y)*config["scale"])**2*mask).sum(axis=1)/mask.sum(axis=1)).mean()
                        target_mean = y[mask].mean()*config["scale"]
                        np.testing.assert_allclose(target_mean, denom[dataset, split], rtol=1e-12)
                        np.testing.assert_allclose(rmse, result["metrics"][split]["rmse"], rtol=1e-12)
                        rows.append(dict(dataset=dataset, variant=variant, seed=seed, split=split,
                            rmse=rmse, relative_error=rmse/target_mean*100,
                            unit=UNITS[dataset], source=str(folder.relative_to(ROOT))))
                provenance[str(folder.relative_to(ROOT))] = dict(checkpoint_sha256=digest,
                    predictions_sha256=hashlib.sha256((folder / "predictions.npz").read_bytes()).hexdigest())
            assert len(set(initial_hashes)) == 1, (dataset, seed, "initialization differs")
            if seed == 1:
                assert initial_hashes[0] == control_config["initial_state_sha256"]
    runs = pd.DataFrame(rows)
    assert len(runs) == 81
    runs.to_csv(out / "per_seed_metrics.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out / "selected_epochs.csv", index=False)
    summary = runs.groupby(["dataset", "variant", "split"], sort=False).agg(
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        relative_error_mean=("relative_error", "mean"), relative_error_sd=("relative_error", "std"),
        n_seeds=("seed", "nunique")).reset_index()
    assert summary.n_seeds.eq(3).all()
    summary.to_csv(out / "comparison.csv", index=False)
    table = summary.set_index(["dataset", "variant", "split"])
    changes = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for split in ("train", "val", "test"):
                group = runs[runs.dataset.eq(dataset) & runs.split.eq(split)].pivot(index="seed", columns="variant", values="rmse")
                delta = group[variant]-group.full
                changes.append(dict(dataset=dataset, variant=variant, split=split,
                    mean_rmse_increase=delta.mean(), sd_rmse_increase=delta.std(ddof=1),
                    error_increase_percent=(group[variant].mean()/group.full.mean()-1)*100,
                    full_model_better_seed_count=int((delta>0).sum()), n_seeds=3))
    changes = pd.DataFrame(changes)
    changes.to_csv(out / "paired_changes.csv", index=False)

    def cell(dataset, variant, split, latex=False):
        row = table.loc[dataset, variant, split]
        digits = dict(wheat=5, maize=2, arabidopsis=3)[dataset]
        pm = r"\pm" if latex else " ± "
        value = f"{row.rmse_mean:.{digits}f}{pm}{row.rmse_sd:.{digits}f} / {row.relative_error_mean:.2f}{pm}{row.relative_error_sd:.2f}"
        best = min(table.loc[dataset, name, split].rmse_mean for name in NAMES)
        if row.rmse_mean == best:
            return r"$\mathbf{"+value+"}$" if latex else "**"+value+"**"
        return "$"+value+"$" if latex else value

    text = ["# PhytoODE physics-loss ablation", "",
        "Three fixed seeds (1--3), paired initialization, unchanged model architecture and dataset-specific training/validation rules. Full PhytoODE rows reuse the frozen manuscript runs. A fresh full-model seed-1 control reproduces the selected epoch and all three split errors for each dataset.", "",
        "## Loss definitions", "",
        "- **PhytoODE:** data loss + logistic ODE residual + maximum-height consistency loss.",
        "- **Without ODE residual:** set only the logistic derivative-residual coefficient to zero; keep the maximum-height consistency loss.",
        "- **Without biological losses:** set both coefficients to zero; train with data loss and the same optimizer weight decay. This is the pure data-trained latent ODE comparison.", "",
        "The auxiliary logistic parameter head is retained to preserve the architecture and random-number sequence. It does not affect predictions, and receives no gradients in the pure-data variant. The genotype embedding, temperature encoder, latent ODE, decoder, optimizer, learning-rate schedule, data splits and masks are identical across variants. There is no ablation-specific hyperparameter search.", "",
        "| Dataset | Epochs | ODE residual weight | Maximum-height weight | Checkpoint selection |",
        "|---|---:|---:|---:|---|",
        "| Wheat | 1,500 | 2.0 | 0.1 | 10-evaluation moving mean, eligible from epoch 600 |",
        "| Maize | 3,000 | 0.5 | 0.5 | Minimum validation RMSE, evaluated every 10 epochs |",
        "| Arabidopsis | 1,500 | 2.0 | 0.1 | 10-evaluation moving mean, eligible from epoch 300 |", "",
        "## Train / validation / test comparison", "",
        "Each cell is **RMSE ± sample SD / relative error (%) ± sample SD**. The lowest mean within each dataset and split is bold. Relative error = curve-averaged RMSE / mean scored target of that split × 100; this is not MAPE.", "",
        "| Dataset (RMSE unit) | Model | Train | Validation | Test |",
        "|---|---|---:|---:|---:|"]
    latex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Loss ablation with fixed architecture, training settings, and seeds 1--3. Entries are RMSE $\pm$ SD / relative RMSE (\%) $\pm$ SD. Wheat uses metres, maize uses relative UAV-height units, and Arabidopsis uses centimetres. The lowest mean for each dataset and split is bold.}",
        r"\label{tab:physics-ablation}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llccc}",
        r"\toprule", r"Dataset & Model & Train & Validation & Test \\", r"\midrule"]
    for dataset in DATASETS:
        for variant, name in NAMES.items():
            text.append("| "+DATASET_NAMES[dataset]+" ("+UNITS[dataset]+") | "+name+" | "+" | ".join(cell(dataset, variant, split) for split in ("train", "val", "test"))+" |")
            latex.append(DATASET_NAMES[dataset]+" & "+name+" & "+" & ".join(cell(dataset, variant, split, True) for split in ("train", "val", "test"))+r" \\")
        latex.append(r"\midrule" if dataset != DATASETS[-1] else r"\bottomrule")
    latex += [r"\end{tabular}}", r"\end{table}"]
    (out / "ablation_table.tex").write_text("\n".join(latex)+"\n")
    text += ["", "## Test-error changes relative to full PhytoODE", "",
        "Positive values mean that removing a loss increases error. Negative values mean that the ablated model performs better.", "",
        "| Dataset | Ablation | RMSE change | Error change (%) | Seeds with lower error for full PhytoODE |",
        "|---|---|---:|---:|---:|"]
    for row in changes[changes.split.eq("test")].itertuples():
        text.append(f"| {DATASET_NAMES[row.dataset]} | {NAMES[row.variant]} | {row.mean_rmse_increase:+.5f} | {row.error_increase_percent:+.2f} | {row.full_model_better_seed_count}/3 |")
    text += ["", "![Test relative errors and paired seeds](test_relative_errors.png)", "",
        "## Interpretation limits", "",
        "The error bars show variation across initialization seeds, not independent years or a biological confidence interval. Three seeds do not establish statistical significance. Settings were originally selected for the full model; this measures removal of losses at those fixed settings, not the best achievable accuracy after independently tuning each ablation. A single held-out year is used for wheat and maize. Arabidopsis holds out plants within known genotypes and temperatures. The wheat scoring mask retains the original pre-observation fill points (475 of 1,368 test points); the other datasets score observed points only.", "",
        "The ODE-residual-only ablation isolates the derivative constraint conditional on the maximum-height term. The pure-data comparison measures the joint effect of both biological penalties. It cannot attribute the entire difference specifically to the ODE residual.", "",
        "## Reproduction", "", "From the repository root, using a new output directory:", "", "```bash",
        ".venv/bin/python experiments/run_physics_ablation.py --gpus 0 2 3 --output experiments/results/physics_ablation_repeat",
        ".venv/bin/python experiments/summarize_physics_ablation.py --output experiments/results/physics_ablation_repeat",
        "```", "",
        "Outputs include per-seed and aggregate metrics, paired changes, selected epochs, full-model reproduction checks, initialization hashes, validation histories, frozen selection records, checkpoints and all-split predictions. The original manuscript tables and figures are preserved."]
    (out / "README.md").write_text("\n".join(text)+"\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    colors = ["#0072BD", "#D95319", "#999999"]
    order = list(NAMES)
    for ax, dataset in zip(axes, DATASETS):
        means = [table.loc[dataset, variant, "test"].relative_error_mean for variant in order]
        sds = [table.loc[dataset, variant, "test"].relative_error_sd for variant in order]
        ax.bar(range(3), means, yerr=sds, color=colors, alpha=.8, capsize=4, width=.58)
        paired = runs[runs.dataset.eq(dataset) & runs.split.eq("test")].pivot(index="seed", columns="variant", values="relative_error")
        for seed in (1, 2, 3):
            ax.plot(np.arange(3)+(seed-2)*.055, paired.loc[seed, order], "o-", color="black", alpha=.35,
                    markersize=3.5, linewidth=.7)
        ax.set_xticks(range(3), ["PhytoODE", "Without ODE\nresidual", "Without\nbiological losses"])
        ax.tick_params(axis="x", labelsize=9)
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_ylabel("Test relative RMSE (%)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(out / "test_relative_errors.png", dpi=220)
    fig.savefig(out / "test_relative_errors.pdf")
    plt.close(fig)
    validation = dict(status="passed", expected_ablation_runs=18, completed_ablation_runs=18,
        full_model_controls=controls, source_hashes="unchanged", paired_initialization="identical",
        architectures_and_training_settings="identical", selection="recomputed from validation histories",
        reported_metrics="recomputed from saved predictions and original scoring denominators",
        test_targets="not provided to the training loop or forward call; evaluated after checkpoint freeze",
        artifacts=provenance, summary_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(summary.to_string(index=False))
    print(f"Saved audited results to {out}")


if __name__ == "__main__":
    main()
