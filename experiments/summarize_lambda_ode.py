"""Audit a completed validation-only lambda search and export comparison artifacts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tune_lambda_ode import ROOT, OLD, DEFAULT_OUT, BASE, sha

DATASETS = list(BASE)
LABELS = dict(wheat="Wheat", maize="Maize", arabidopsis="Arabidopsis")
UNITS = dict(wheat="m", maize="relative UAV height", arabidopsis="cm")
NAMES = dict(full="Original PhytoODE", tuned="Tuned PhytoODE",
             no_ode_residual="Without ODE residual", no_biological_loss="Without biological losses")
ORDER = ["full", "tuned", "no_ode_residual", "no_biological_loss"]
COLORS = ["#999999", "#0072BD", "#D95319", "#009E73"]


def read(path):
    return json.loads(Path(path).read_text())


def aggregate(frame, groups):
    return frame.groupby(groups, sort=False).agg(
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        relative_error_mean=("relative_error", "mean"), relative_error_sd=("relative_error", "std"),
        n_seeds=("seed", "nunique")).reset_index()


def check_predictions(path, config, expected, dataset, denominators, splits):
    rows = []
    with np.load(path, allow_pickle=False) as saved:
        if splits == ("train", "val"):
            assert not any("test" in key for key in saved.files)
        for split in splits:
            key = "train_eval" if split == "train" else split
            pred, y = saved[key].astype(float), saved[key+"_target"].astype(float)
            mask = saved[key+"_mask"].astype(bool)
            assert pred.shape == y.shape == mask.shape and np.isfinite(pred).all()
            assert mask.sum(axis=1).min() > 0
            error = (pred-y)*config["scale"]
            rmse = float(np.sqrt((error**2*mask).sum(axis=1)/mask.sum(axis=1)).mean())
            denominator = float(y[mask].mean()*config["scale"])
            np.testing.assert_allclose(denominator, denominators[dataset, split], rtol=1e-12)
            np.testing.assert_allclose(rmse, expected[split]["rmse"], rtol=1e-12)
            np.testing.assert_allclose(rmse/denominator*100, expected[split]["relative_error"], rtol=1e-12)
            rows.append(dict(dataset=dataset, split=split, rmse=rmse,
                relative_error=rmse/denominator*100, unit=UNITS[dataset]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    protocol, completed = read(out/"protocol.json"), read(out/"completed.json")
    assert completed["status"] == "complete" and completed["selected_final_test_evaluations"] == 9
    joint = read(out/"all_selections_frozen.json")
    assert joint["new_candidate_test_evaluations"] == 0
    recovery_audit = None
    if completed.get("recovered_after_interruption"):
        recovery, recovered = read(out/"recovery.json"), read(out/"recovery_completed.json")
        assert recovered["status"] == "complete"
        assert recovery["protocol_sha256"] == sha(out/"protocol.json")
        assert recovery["recovery_code_sha256"] == sha(ROOT/"experiments/resume_lambda_ode.py")
        assert sha(out/"recovery.json") == joint["recovery_manifest_sha256"] == recovered["recovery_manifest_sha256"]
        for job in recovery["jobs"]:
            assert job["status"] == "complete"
            if "interrupted_prefix" in job:
                prefix = pd.DataFrame(job["interrupted_prefix"]).set_index("epoch")
                history = pd.read_csv(ROOT/job["output"]/"history.csv").set_index("epoch")
                np.testing.assert_allclose(history.loc[prefix.index, prefix.columns], prefix, rtol=1e-7, atol=1e-9)
                assert job["prefix_verification"] == "passed"
        recovery_audit = dict(status="passed", incomplete_attempts=sum("interrupted_epoch" in j for j in recovery["jobs"]),
            prefix="recomputed and matched", protocol="unchanged", training_schedule="full restart from identical seed")
    for path, expected in protocol["source_sha256"].items():
        assert sha(ROOT/path) == expected, path
    denominator_frame = pd.read_csv(ROOT/"paper/relative_errors/denominators.csv")
    denominators = {(r.dataset, r.split): r.mean_target for r in denominator_frame.itertuples()}
    runs = pd.read_csv(OLD/"per_seed_metrics.csv").to_dict("records")
    validation_rows, coefficient_rows, provenance, configs, selections = [], [], {}, {}, {}
    new_trials, loss_rows = 0, []
    for dataset in DATASETS:
        folder = out/dataset
        assert read(out/"checks"/dataset/"verification.json")["status"] == "passed"
        control = read(OLD/dataset/"full_seed1/config.json")
        configs[dataset] = control
        selected = read(folder/"selected.json")
        selections[dataset] = selected
        assert selected == next(r for r in joint["datasets"] if r["dataset"] == dataset)
        assert selected["new_candidate_test_evaluations"] == 0
        records = read(folder/"validation_records.json")
        assert len({(r["lambda_ode"], r["seed"]) for r in records}) == len(records)
        positive_screen = sorted((r for r in records if r["seed"] == 1 and r["lambda_ode"] > 0),
                                 key=lambda r: (r["val_rmse"], r["lambda_ode"]))
        finalists = read(folder/"finalists.json")["lambdas"]
        assert finalists == [r["lambda_ode"] for r in positive_screen[:3]]
        refinement = read(folder/"refinement.json")
        grid = sorted(protocol["grid"])
        center = min((r for r in records if r["seed"] == 1 and r["lambda_ode"] in grid),
                     key=lambda r: (r["val_rmse"], r["lambda_ode"]))["lambda_ode"]
        assert center == refinement["center"]
        i = grid.index(center)
        left = center/10 if i == 0 else math.sqrt(grid[i-1]*center)
        right = center*10 if i == len(grid)-1 else math.sqrt(center*grid[i+1])
        expected_refinement = sorted(set(float(format(v, ".10g")) for v in (left, right))-set(grid))
        assert refinement["lambdas"] == expected_refinement
        groups = {}
        for record in records:
            value, seed = record["lambda_ode"], record["seed"]
            assert "test_rmse" not in record and "test" not in record.get("metrics", {})
            groups.setdefault(value, {})[seed] = record
            validation_rows.append(dict(dataset=dataset, lambda_ode=value, seed=seed,
                train_rmse=record["train_rmse"], val_rmse=record["val_rmse"], best_epoch=record["best_epoch"],
                origin=record["origin"], selected_positive=value == selected["selected_positive"]["lambda_ode"]))
        ranking = []
        for value, group in groups.items():
            if set(group) == {1, 2, 3}:
                values = [group[s]["val_rmse"] for s in (1, 2, 3)]
                ranking.append(dict(lambda_ode=value, val_mean=float(np.mean(values)), val_sd=float(np.std(values, ddof=1))))
        assert set(groups) == {0., *grid, *expected_refinement}
        for value in finalists:
            assert set(groups[value]) == {1, 2, 3}
        ranking.sort(key=lambda r: (r["val_mean"], r["val_sd"], r["lambda_ode"]))
        assert len(ranking) == len(selected["ranking"])
        best_positive = next(r for r in ranking if r["lambda_ode"] > 0)
        zero = next(r for r in ranking if r["lambda_ode"] == 0)
        assert selected["positive_beats_zero_on_validation"] == (best_positive["val_mean"] < zero["val_mean"])
        assert best_positive["lambda_ode"] == selected["selected_positive"]["lambda_ode"]
        assert ranking[0]["lambda_ode"] == selected["selected_overall"]["lambda_ode"]
        for actual, frozen in zip(ranking, selected["ranking"]):
            assert actual["lambda_ode"] == frozen["lambda_ode"]
            np.testing.assert_allclose([actual["val_mean"], actual["val_sd"]],
                                      [frozen["val_mean"], frozen["val_sd"]], rtol=1e-5, atol=2e-6)
        winner = best_positive["lambda_ode"]
        coefficient_rows.append(dict(dataset=dataset, original_lambda_ode=BASE[dataset],
            selected_positive_lambda_ode=winner, selected_overall_lambda_ode=ranking[0]["lambda_ode"],
            lambda_ymax=control["ymax"], selected_val_mean=best_positive["val_mean"],
            selected_val_sd=best_positive["val_sd"],
            positive_beats_zero_on_validation=selected["positive_beats_zero_on_validation"]))
        for result_path in sorted((folder/"trials").glob("*/*/result.json")):
            trial = result_path.parent
            result, config = read(result_path), read(trial/"config.json")
            assert result["status"] == "complete" and result["test_evaluations"] == 0
            assert set(result["metrics"]) == {"train", "val"}
            assert config["effective_weights"] == dict(physic=result["lambda_ode"], ymax=control["ymax"], r=0., mono=0.)
            assert config["lambda_ode"] == result["lambda_ode"]
            assert result["n_params"] == control["expected_n_params"]
            for key in ("model", "epochs", "lr", "weight_decay", "eval_every", "min_epoch", "val_smooth", "permute_training"):
                assert config[key] == control[key], (trial, key)
            paired_config = read(OLD/dataset/f"no_ode_residual_seed{result['seed']}/config.json")
            assert config["initial_state_sha256"] == result["initial_state_sha256"] == paired_config["initial_state_sha256"]
            history = pd.read_csv(trial/"history.csv")
            assert history.epoch.iloc[-1] == result["epochs"] == config["epochs"]
            smooth = history.val_rmse.rolling(config["val_smooth"], min_periods=config["val_smooth"]).mean()
            eligible = smooth.notna() & history.epoch.ge(config["min_epoch"])
            epoch = int(history.loc[smooth[eligible].idxmin(), "epoch"])
            assert epoch == result["best_epoch"]
            at_epoch = history[history.epoch.eq(epoch)].iloc[0]
            loss_rows.append(dict(dataset=dataset, lambda_ode=result["lambda_ode"], seed=result["seed"],
                best_epoch=epoch, data_loss=at_epoch.data_loss, weighted_ode_loss=at_epoch.L_m,
                raw_ode_loss=at_epoch.L_m/result["lambda_ode"], weighted_maximum_height_loss=at_epoch.L_y,
                weighted_ode_to_data_ratio=at_epoch.L_m/at_epoch.data_loss,
                selected_positive=result["lambda_ode"] == winner))
            assert sha(trial/"checkpoint.pt") == result["checkpoint_sha256"]
            check_predictions(trial/"validation_predictions.npz", config, result["metrics"], dataset,
                              denominators, ("train", "val"))
            provenance[str(trial.relative_to(ROOT))] = dict(checkpoint_sha256=result["checkpoint_sha256"],
                predictions_sha256=sha(trial/"validation_predictions.npz"), test_evaluations=0)
            new_trials += 1
        effective = {k: v for k, v in control.items() if k not in (
            "variant", "seed", "initial_state_sha256", "cuda_visible_devices", "source_git_commit", "torch_version", "gpu_name")}
        effective.update(physic=winner, lambda_ode=winner,
            effective_weights=dict(physic=winner, ymax=control["ymax"], r=0., mono=0.),
            selection_source=str((folder/"selected.json").relative_to(ROOT)),
            selection_sha256=sha(folder/"selected.json"), seeds=[1, 2, 3],
            checkpoints=selected["selected_positive"]["checkpoints"])
        (folder/"selected_config.json").write_text(json.dumps(effective, indent=2)+"\n")
        for seed in (1, 2, 3):
            final = folder/"final"/f"seed{seed}"
            result = read(final/"result.json")
            assert (final/"result.json").stat().st_mtime >= joint["frozen_unix"]
            assert result["lambda_ode"] == winner and result["seed"] == seed and result["test_evaluations"] == 1
            assert result["selection_sha256"] == sha(folder/"selected.json")
            checkpoint = selected["selected_positive"]["checkpoints"][str(seed)]
            assert sha(ROOT/checkpoint["path"]) == checkpoint["sha256"] == result["checkpoint_sha256"]
            np.testing.assert_allclose(result["metrics"]["val"]["rmse"], groups[winner][seed]["val_rmse"], rtol=1e-6, atol=2e-6)
            rows = check_predictions(final/"predictions.npz", control, result["metrics"], dataset,
                                     denominators, ("train", "val", "test"))
            if winner == BASE[dataset]:
                # Identical frozen checkpoints must not acquire artificial wins
                # from the historical scorer's float32/float64 differences.
                for metric in rows:
                    original = next(r for r in runs if r["dataset"] == dataset and r["variant"] == "full"
                                    and r["seed"] == seed and r["split"] == metric["split"])
                    np.testing.assert_allclose(original["rmse"], metric["rmse"], rtol=1e-6, atol=2e-6)
                    original.update(rmse=metric["rmse"], relative_error=metric["relative_error"],
                        source="original checkpoint reevaluated with the same scorer: "+str(final.relative_to(ROOT)))
            runs += [r | dict(variant="tuned", seed=seed, source=str(final.relative_to(ROOT))) for r in rows]
            provenance[str(final.relative_to(ROOT))] = dict(checkpoint_sha256=checkpoint["sha256"],
                predictions_sha256=sha(final/"predictions.npz"), test_evaluations=1)
    runs = pd.DataFrame(runs)
    assert len(runs) == 108
    runs.to_csv(out/"per_seed_metrics.csv", index=False)
    summary = aggregate(runs, ["dataset", "variant", "split"])
    summary.to_csv(out/"comparison.csv", index=False)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_csv(out/"selected_coefficients.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(out/"loss_contributions.csv", index=False)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(out/"validation_trials.csv", index=False)
    validation.groupby(["dataset", "lambda_ode"], sort=True).agg(val_mean=("val_rmse", "mean"),
        val_sd=("val_rmse", "std"), n_seeds=("seed", "nunique")).reset_index().to_csv(out/"validation_summary.csv", index=False)
    changes = []
    for dataset in DATASETS:
        for split in ("train", "val", "test"):
            paired = runs[runs.dataset.eq(dataset) & runs.split.eq(split)].pivot(index="seed", columns="variant", values="rmse")
            for ref in ("full", "no_ode_residual", "no_biological_loss"):
                delta = paired.tuned-paired[ref]
                changes.append(dict(dataset=dataset, split=split, reference=ref,
                    tuned_minus_reference=delta.mean(), paired_delta_sd=delta.std(ddof=1),
                    error_reduction_percent=(1-paired.tuned.mean()/paired[ref].mean())*100,
                    tuned_better_seed_count=int((delta<0).sum()), n_seeds=3))
    changes = pd.DataFrame(changes)
    changes.to_csv(out/"paired_changes.csv", index=False)
    write_tables(out, summary, coefficients, changes, new_trials, completed)
    write_korean_interpretation(out, summary, coefficients, changes)
    plot(out, runs, summary, validation, selections)
    all_baselines(out, runs)
    audit = dict(status="passed", new_completed_training_trials=new_trials, new_final_test_evaluations=9,
        fixed_settings_and_paired_initialization="verified against prior controls for every new trial",
        checkpoint_selection="recomputed from each validation history", candidate_selection="recomputed from validation-only records",
        prediction_metrics="recomputed from saved predictions, original masks and denominators",
        all_selections_frozen_before_final_tests=True, test_evaluation_limitation=protocol["evaluation_limitation"],
        source_hashes="unchanged", interruption_recovery=recovery_audit,
        artifacts=provenance, summary_code_sha256=sha(Path(__file__)))
    (out/"validation.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(coefficients.to_string(index=False))
    print(summary[summary.split.eq("test")].to_string(index=False))
    print(f"Audit passed: {new_trials} new training trials; 9 selected final evaluations.")


def write_tables(out, summary, coefficients, changes, new_trials, completed):
    table = summary.set_index(["dataset", "variant", "split"])
    test_changes = changes[changes.split.eq("test") & changes.reference.eq("no_ode_residual")]
    lower = [LABELS[r.dataset] for r in test_changes.itertuples() if r.error_reduction_percent > 0]
    higher = [LABELS[r.dataset] for r in test_changes.itertuples() if r.error_reduction_percent < 0]
    outcome = "Against the zero-residual model, the validation-selected positive coefficient lowers mean test error for "+", ".join(lower)+"."
    if higher:
        outcome += " It raises mean test error for "+", ".join(higher)+"; tuning does not make the ODE-residual model uniformly best on test."

    def cell(dataset, variant, split, latex=False):
        row = table.loc[dataset, variant, split]
        digits = dict(wheat=5, maize=2, arabidopsis=3)[dataset]
        pm = r"\pm" if latex else " ± "
        value = f"{row.rmse_mean:.{digits}f}{pm}{row.rmse_sd:.{digits}f} / {row.relative_error_mean:.2f}{pm}{row.relative_error_sd:.2f}"
        best = min(table.loc[dataset, name, split].rmse_mean for name in ORDER)
        if row.rmse_mean == best:
            return r"$\mathbf{"+value+"}$" if latex else "**"+value+"**"
        return "$"+value+"$" if latex else value

    text = ["# Validation-selected ODE-residual coefficient tuning", "",
        "**"+outcome+"**", "",
        "Only the logistic derivative-residual coefficient was varied. Architecture, paired initialization, maximum-height coefficient, optimizer, schedule, training length, data splits, masks and validation checkpoint rules match the previous ablation.", "",
        f"Completed {new_trials} new full-length training trials in {completed['seconds']/3600:.2f} hours, followed by nine final evaluations. Wheat and Arabidopsis ran concurrently on GPU 2; maize used GPU 3.", "",
        "## Search and selection", "",
        "The fixed positive grid was 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100 and 500. Original positive-coefficient and zero-residual results were reused. Seed 1 screened the grid, followed by one refinement using geometric midpoints around its best positive validation coefficient (one decade outward at a grid boundary). The three best positive seed-1 candidates were confirmed with seeds 2 and 3. All coefficients with three seeds, including zero and the original coefficient, entered the final validation ranking. The positive coefficient with the lowest mean validation RMSE was selected; the overall winner including zero is reported separately.", "",
        "No new candidate was evaluated on test during search. All three dataset selections and checkpoint hashes were frozen before any final test evaluation. **The previous ablation's test results had already been inspected before this follow-up search: these are reused-test follow-up scores, not an independent confirmation.**", "",
        "| Dataset | Original ODE coefficient | Selected positive coefficient | Overall validation winner (including zero) | Fixed maximum-height coefficient |",
        "|---|---:|---:|---:|---:|"]
    for row in coefficients.itertuples():
        text.append(f"| {LABELS[row.dataset]} | {row.original_lambda_ode:g} | {row.selected_positive_lambda_ode:.6g} | {row.selected_overall_lambda_ode:.6g} | {row.lambda_ymax:g} |")
    if completed.get("recovered_after_interruption"):
        text += ["", "### Interrupted-run recovery", "",
            "The original controller and one wheat confirmation run received termination signals of unidentified origin. The incomplete wheat run (coefficient 3.162277660, seed 2) stopped at epoch 578 and was excluded. The 45 completed trials and two frozen dataset selections were retained. The interrupted run was restarted from the same seed for the full 1,500 epochs, and the still-pending seed-3 run was completed. The recorded interrupted prefix matches the replacement run. No optimizer, architecture, coefficient candidate, ranking rule or test policy changed. `recovery.json` retains the interrupted prefix and records the recovery procedure; `recovery_completed.json` records completion. The original frozen training/controller sources remain unchanged."]
    text += ["", "![Validation coefficient search](validation_lambda.png)", "",
        "The gray points show seed-1 screening values. Blue points and error bars show the mean and sample SD for coefficients evaluated with all three seeds. The dashed line is the three-seed mean at zero; the selected positive coefficient is marked. A screened candidate with only one seed is not eligible for final selection.", "",
        "## Train / validation / test", "",
        "Cells are **RMSE ± sample SD / relative RMSE (%) ± sample SD**, across seeds 1--3. Bold denotes the lowest mean among these four variants within each dataset and split. Relative RMSE is curve-averaged RMSE divided by the split's mean scored target, multiplied by 100; it is not MAPE. Maize relative UAV-height units are the target units and should not be confused with relative RMSE.", "",
        "| Dataset (RMSE unit) | Variant | Train | Validation | Test |", "|---|---|---:|---:|---:|"]
    latex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Validation-selected tuning of the logistic ODE-residual coefficient with all other training settings fixed. Entries are RMSE $\pm$ SD / relative RMSE (\%) $\pm$ SD for seeds 1--3. Wheat uses metres, maize relative UAV-height units, and Arabidopsis centimetres. Bold marks the lowest mean within each dataset and split. Test sets were previously inspected in the preceding ablation and are reused for this follow-up evaluation.}",
        r"\label{tab:lambda-ode-tuning}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llccc}",
        r"\toprule", r"Dataset & Variant & Train & Validation & Test \\", r"\midrule"]
    for dataset in DATASETS:
        for variant in ORDER:
            text.append("| "+LABELS[dataset]+" ("+UNITS[dataset]+") | "+NAMES[variant]+" | "+" | ".join(cell(dataset, variant, split) for split in ("train", "val", "test"))+" |")
            latex.append(LABELS[dataset]+" & "+NAMES[variant]+" & "+" & ".join(cell(dataset, variant, split, True) for split in ("train", "val", "test"))+r" \\")
        latex.append(r"\bottomrule" if dataset == DATASETS[-1] else r"\midrule")
    latex += [r"\end{tabular}}", r"\end{table}"]
    (out/"lambda_ode_table.tex").write_text("\n".join(latex)+"\n")
    text += ["", "![Test relative errors](test_relative_errors.png)", "",
        "Bars show three-seed means and sample SD; paired points connect the same initialization seed. All four variants retain the latent Neural ODE architecture. The zero-residual variant retains the maximum-height penalty; the final variant removes both biological losses. If the search retains the original coefficient, original and tuned rows use the same reevaluated checkpoints and scores; scorer-rounding differences are not counted as improvements.", "",
        "## Paired test comparisons", "", "Positive error reduction means the tuned model has lower mean error.", "",
        "| Dataset | Reference | Error reduction (%) | Seeds favoring tuned model |", "|---|---|---:|---:|"]
    for row in changes[changes.split.eq("test")].itertuples():
        text.append(f"| {LABELS[row.dataset]} | {NAMES[row.reference]} | {row.error_reduction_percent:+.2f} | {row.tuned_better_seed_count}/3 |")
    text += ["", "Three seeds measure initialization variation, not uncertainty across independent years or experiments. This comparison alone does not establish statistical significance or superiority of the latent ODE architecture over a model without an ODE. Single held-out years and previously seen test scores limit generalization claims. Wheat preserves the original initial-fill scoring mask (475 of 1,368 test points are pre-observation fill); maize and Arabidopsis score observed points only. Arabidopsis holds out plants within known genotype/temperature combinations.", "",
        "The broader baseline table is in [all_baselines.md](all_baselines.md). Baselines reuse their existing frozen results and were not rerun or retuned here. The original manuscript figures and tables remain available separately.", "",
        "## Reproduction", "", "From the repository root, use a fresh output directory:", "", "```bash",
        ".venv/bin/python experiments/tune_lambda_ode.py --short-gpu 2 --long-gpu 3 --maize-workers 2 --output experiments/results/lambda_ode_repeat",
        ".venv/bin/python experiments/summarize_lambda_ode.py --output experiments/results/lambda_ode_repeat", "```", "",
        "Each dataset's `selected_config.json` contains the selected effective coefficient, unchanged training settings and checkpoint references. `selected.json` records the validation ranking and frozen checkpoint hashes; `validation.json` audits every completed trial and recomputes final metrics from predictions. Original full and zero-coefficient references are reused from the preceding experiments, and all new training trials contain train/validation metrics only.", "",
        "`loss_contributions.csv` records raw and weighted ODE losses for new trials at their selected epoch, from the training objective immediately before that epoch's optimizer step. These loss magnitudes help interpret the numerical coefficient but do not measure gradient influence. In the model's internal height scale (normalized heights for maize), the residual is a squared error in height-per-day units, while the data term is RMSE in height units, so coefficient magnitude alone does not describe constraint strength."]
    (out/"README.md").write_text("\n".join(text)+"\n")


def plot(out, runs, summary, validation, selections):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), layout="constrained")
    for ax, dataset in zip(axes, DATASETS):
        values = validation[validation.dataset.eq(dataset)]
        screen = values[values.seed.eq(1) & values.lambda_ode.gt(0)].sort_values("lambda_ode")
        confirmed = values.groupby("lambda_ode").val_rmse.agg(["mean", "std", "count"])
        positive = confirmed[(confirmed.index > 0) & confirmed["count"].eq(3)]
        ax.plot(screen.lambda_ode, screen.val_rmse, "o", color="#999999", ms=4, label="Seed 1 screen")
        ax.errorbar(positive.index, positive["mean"], yerr=positive["std"], fmt="o", color="#0072BD", capsize=3, label="3 seeds: mean ± SD")
        ax.axhline(confirmed.loc[0., "mean"], color="#D95319", ls="--", lw=1.2, label=r"$\lambda_{\mathrm{ODE}}=0$")
        winner = selections[dataset]["selected_positive"]["lambda_ode"]
        ax.axvline(winner, color="#0072BD", lw=.8, alpha=.5)
        ax.set_title(f"{LABELS[dataset]}: selected λ = {winner:.4g}")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda_{\mathrm{ODE}}$")
        ax.set_ylabel(f"Validation RMSE ({UNITS[dataset]})")
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.savefig(out/"validation_lambda.png", dpi=220)
    fig.savefig(out/"validation_lambda.pdf")
    plt.close(fig)
    table = summary.set_index(["dataset", "variant", "split"])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), layout="constrained")
    for ax, dataset in zip(axes, DATASETS):
        means = [table.loc[dataset, v, "test"].relative_error_mean for v in ORDER]
        sds = [table.loc[dataset, v, "test"].relative_error_sd for v in ORDER]
        ax.bar(range(4), means, yerr=sds, color=COLORS, alpha=.8, capsize=3, width=.62)
        paired = runs[runs.dataset.eq(dataset) & runs.split.eq("test")].pivot(index="seed", columns="variant", values="relative_error")
        for seed in (1, 2, 3):
            ax.plot(np.arange(4)+(seed-2)*.04, paired.loc[seed, ORDER], "o-", color="black", alpha=.35, ms=3, lw=.7)
        ax.set_xticks(range(4), ["Original", "Tuned", "Without ODE\nresidual", "Without\nbiological losses"])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(LABELS[dataset])
        ax.set_ylabel("Test relative RMSE (%)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(out/"test_relative_errors.png", dpi=220)
    fig.savefig(out/"test_relative_errors.pdf")
    plt.close(fig)


def all_baselines(out, runs):
    old = pd.read_csv(ROOT/"paper/relative_errors/per_run_relative_errors.csv")
    old = old[old.is_primary & ~old.model.eq("Ours")].copy()
    old["relative_error"] = old.rrmse_percent
    new = runs.copy()
    new["model"] = new.variant.map(NAMES)
    all_runs = pd.concat([old[["dataset", "model", "seed", "split", "rmse", "relative_error", "unit"]],
                          new[["dataset", "model", "seed", "split", "rmse", "relative_error", "unit"]]], ignore_index=True)
    all_runs.to_csv(out/"all_baselines_per_seed.csv", index=False)
    summary = aggregate(all_runs, ["dataset", "model", "split"])
    summary.to_csv(out/"all_baselines.csv", index=False)
    table = summary.set_index(["dataset", "model", "split"])
    text = ["# Comparison with existing baselines", "",
        "Cells are mean RMSE / relative RMSE (%); bold marks the lowest mean within each dataset and split. RMSE units: wheat m, maize relative UAV height, Arabidopsis cm. Models with three seeds are averaged; deterministic Logistic and Temperature ODE fits have one result. Baseline metrics are reused from the existing manuscript table and were not recomputed or retuned. Full per-seed values and SD are in the companion CSV files. Final test sets were already inspected before this follow-up tuning.", "",
        "| Dataset | Model | Train | Validation | Test |", "|---|---|---:|---:|---:|"]
    model_order = list(NAMES.values()) + ["Logi-PINN", "LSTM-NN", "RF", "Temp-ODE", "Logi-ODE"]
    for dataset in DATASETS:
        for model in model_order:
            cells = []
            for split in ("train", "val", "test"):
                row = table.loc[dataset, model, split]
                digits = dict(wheat=5, maize=2, arabidopsis=3)[dataset]
                value = f"{row.rmse_mean:.{digits}f} / {row.relative_error_mean:.2f}"
                best = summary[summary.dataset.eq(dataset) & summary.split.eq(split)].rmse_mean.min()
                cells.append("**"+value+"**" if row.rmse_mean == best else value)
            text.append(f"| {LABELS[dataset]} | {model} | "+" | ".join(cells)+" |")
    (out/"all_baselines.md").write_text("\n".join(text)+"\n")


def write_korean_interpretation(out, summary, coefficients, changes):
    table = summary.set_index(["dataset", "variant", "split"])
    names = dict(wheat="밀", maize="옥수수", arabidopsis="Arabidopsis")
    test_changes = changes[changes.split.eq("test") & changes.reference.eq("no_ode_residual")]
    lower = [names[r.dataset] for r in test_changes.itertuples() if r.error_reduction_percent > 0]
    higher = [names[r.dataset] for r in test_changes.itertuples() if r.error_reduction_percent < 0]
    outcome = "계수 0 모델 대비 test 평균 오차는 "+", ".join(lower)+"에서 감소했다."
    if higher:
        outcome += " "+", ".join(higher)+"에서는 증가했다. 따라서 모든 데이터셋에서 ODE 잔차 손실을 넣은 모델이 최고라는 결론은 뒷받침되지 않는다."
    text = ["# ODE loss 계수 튜닝 결과 해석", "",
        "**"+outcome+"**", "",
        "이번 실험에서는 latent Neural ODE 구조를 유지하고 logistic ODE 잔차 손실의 계수만 조절했다. 최대높이 손실 계수는 밀·Arabidopsis 0.1, 옥수수 0.5로 고정했다. 데이터 loss만 사용하는 비교 모델도 latent ODE 구조를 사용한다.", "",
        "손실은 `L_data + lambda_ODE * L_ODE + lambda_K * L_K`이다. 계수가 크더라도 ODE 손실이 학습을 지배한다는 뜻은 아니다. 데이터 손실은 높이의 RMSE인 반면 ODE 잔차는 일별 성장률의 제곱오차이므로 수치와 단위가 다르다. 여기서 높이는 모델 내부의 단위이며 옥수수는 정규화된 높이를 사용한다. 실제 가중 손실 크기는 `loss_contributions.csv`에 기록했다. 손실 크기와 gradient 영향력도 구분해야 한다.", "",
        "## 데이터셋별 결과", ""]
    for row in coefficients.itertuples():
        dataset = row.dataset
        val_tuned = table.loc[dataset, "tuned", "val"].rmse_mean
        val_zero = table.loc[dataset, "no_ode_residual", "val"].rmse_mean
        test_tuned = table.loc[dataset, "tuned", "test"]
        delta = changes[changes.dataset.eq(dataset) & changes.split.eq("test") & changes.reference.eq("no_ode_residual")].iloc[0]
        text += [f"### {names[dataset]}", "",
            f"양의 계수 후보 중 validation 평균 RMSE로 선택된 값은 {row.original_lambda_ode:g} → **{row.selected_positive_lambda_ode:.6g}**이다. 선택 모델의 validation RMSE는 {val_tuned:.6g}, 계수 0 모델은 {val_zero:.6g}이다. 0까지 포함한 전체 validation 최적 계수는 **{row.selected_overall_lambda_ode:.6g}**이다.", "",
            f"선택 모델의 test RMSE는 **{test_tuned.rmse_mean:.6g} ± {test_tuned.rmse_sd:.6g} {UNITS[dataset]}**, relative RMSE는 **{test_tuned.relative_error_mean:.3f} ± {test_tuned.relative_error_sd:.3f}%**이다. 계수 0 모델 대비 test 평균 오차 감소율은 {delta.error_reduction_percent:+.2f}%이며, 같은 시드끼리 비교하면 {int(delta.tuned_better_seed_count)}/3개에서 더 낮다. 감소율이 음수이면 선택된 physics 모델의 평균 test 오차가 더 높다는 뜻이다.", ""]
    text += ["## 논문에서 주장할 수 있는 범위", "",
        "양의 계수가 0보다 validation에서 좋았는지와 test에서 좋았는지는 구분해서 보고해야 한다. 모든 데이터셋에서 physics loss가 우수하다는 결론은 해당 비교 결과가 뒷받침할 때만 가능하다. 계수 선택은 미리 정한 validation 절차로 완료했으며 test 순위로 계수를 다시 바꾸지 않았다.", "",
        "이전 ablation의 test 결과를 확인한 뒤 시작한 후속 튜닝이다. 따라서 이번 test 점수는 기존 test set을 재사용한 평가이며, 새로운 독립 검증으로 제시하면 안 된다. 3개 시드의 표준편차는 초기화 변동성이고 통계적 유의성 또는 새로운 연도에 대한 불확실성을 확정하지 않는다. 독립 연도 또는 반복된 외부 분할로 확인하면 physics loss의 일반화 효과를 더 강하게 주장할 수 있다.", "",
        "ODE 잔차만 제거한 비교는 최대높이 손실이 있는 조건에서 잔차 항의 효과를 평가한다. 두 biological loss를 모두 제거한 비교는 두 항의 공동 효과이므로, 그 차이를 전부 ODE 잔차 덕분이라고 해석해서는 안 된다. 이번 결과만으로 latent ODE 구조 자체의 필요성을 입증할 수도 없다.", "",
        "Train / validation / test 전체 비교는 `README.md`, 기존 baseline까지 포함한 표는 `all_baselines.md`, LaTeX 표는 `lambda_ode_table.tex`에 있다. 기존 원고의 결과와 이번 후속 튜닝 결과는 별도 파일로 보존했다."]
    if (out/"recovery_completed.json").exists():
        text += ["", "## 실행 중단과 복구", "",
            "밀의 계수 3.162277660, 시드 2 학습이 578 epoch에서 종료 신호로 중단되어 동일한 초기값과 전체 1,500 epoch 일정으로 재실행했다. 중단된 부분 실행은 완료 결과에 포함하지 않았고, 재실행의 초기 학습 기록이 중단 전 기록과 일치함을 검증했다. 기존 45개 완료 결과, 후보 목록, 계수 선택 기준과 test 평가 규칙은 유지했다. 종료 신호의 발신 원인은 확인되지 않았다."]
    (out/"interpretation_ko.md").write_text("\n".join(text)+"\n")


if __name__ == "__main__":
    main()
