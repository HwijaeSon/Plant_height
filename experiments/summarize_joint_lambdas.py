"""Audit the joint search and export all-split tables and scientific figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tune_joint_lambdas import (ROOT, OLD, PREVIOUS, DEFAULT_OUT, BASE, BASE_K, read,
                               pair, job_key, rank, refinement_plan, finalist_plan, sha, atomic)
from summarize_lambda_ode import aggregate, check_predictions, LABELS, UNITS

NAMES = dict(no_biological_loss="Latent Neural ODE (no physics)", full="Original PhytoODE",
             no_ode_residual="PhytoODE (K loss only)", tuned="PhytoODE (ODE coefficient tuned)",
             joint="PhytoODE (joint tuning)", joint_overall="Joint-search overall winner")
ORDER = list(NAMES)
COLORS = dict(no_biological_loss="#999999", full="#56B4E9", no_ode_residual="#D95319",
              tuned="#009E73", joint="#0072BD", joint_overall="#CC79A7")
DIGITS = dict(wheat=5, maize=2, arabidopsis=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    protocol, completed = read(out / "protocol.json"), read(out / "completed.json")
    frozen = read(out / "all_selections_frozen.json")
    assert completed["status"] == "complete" and frozen["new_candidate_test_evaluations"] == 0
    for path, expected in protocol["source_sha256"].items():
        assert sha(ROOT / path) == expected, path
    denominators = {(r.dataset, r.split): r.mean_target for r in
                    pd.read_csv(ROOT / "paper/relative_errors/denominators.csv").itertuples()}
    runs = pd.read_csv(PREVIOUS / "per_seed_metrics.csv").to_dict("records")
    validation_rows, coefficients, losses, provenance = [], [], [], {}
    selections, new_count, final_count = {}, 0, 0
    for dataset in BASE:
        folder = out / dataset
        control = read(OLD / dataset / "full_seed1/config.json")
        assert read(out / "checks" / dataset / "verification.json")["status"] == "passed"
        selected = read(folder / "selected.json")
        selections[dataset] = selected
        assert selected == next(r for r in frozen["datasets"] if r["dataset"] == dataset)
        assert sha(folder / "selected.json") == frozen["selection_sha256"][dataset]
        records = read(folder / "validation_records.json")
        assert len({job_key(r) for r in records}) == len(records)
        ranking, groups = rank(records)
        assert len(ranking) == len(selected["ranking"])
        for actual, saved in zip(ranking, selected["ranking"]):
            assert pair(actual) == pair(saved)
            np.testing.assert_allclose([actual["val_mean"], actual["val_sd"]],
                                       [saved["val_mean"], saved["val_sd"]], rtol=1e-6, atol=2e-6)
        best = next(r for r in ranking if min(pair(r)) > 0)
        assert pair(best) == pair(selected["selected_phyto"])
        assert pair(ranking[0]) == pair(selected["selected_overall"])
        refinement = read(folder / "refinement.json")
        refined = {tuple(p) for p in refinement["pairs"]}
        assert refinement_plan([r for r in records if pair(r) not in refined]) == refinement
        assert finalist_plan(records) == read(folder / "finalists.json")
        for p in read(folder / "finalists.json")["pairs"]:
            assert set(groups[tuple(p)]) == {1, 2, 3}
        assert all((*p, 1) in {job_key(r) for r in records} for p in protocol["grids"][dataset])
        for r in records:
            assert "test_rmse" not in r and "test" not in r.get("metrics", {})
            validation_rows.append({k: r[k] for k in ("dataset", "lambda_ode", "lambda_k", "seed",
                "train_rmse", "val_rmse", "best_epoch", "origin")} | dict(selected_phyto=pair(r) == pair(best)))
        for path in sorted((folder / "trials").glob("*/*/result.json")):
            trial = path.parent
            result, config = read(path), read(trial / "config.json")
            assert result["status"] == "complete" and result["test_evaluations"] == 0
            assert config["effective_weights"] == dict(physic=result["lambda_ode"], ymax=result["lambda_k"], r=0., mono=0.)
            assert result["n_params"] == control["expected_n_params"]
            for k in ("model", "epochs", "lr", "weight_decay", "eval_every", "min_epoch", "val_smooth", "permute_training"):
                assert config[k] == control[k], (trial, k)
            paired = read(OLD / dataset / f"no_biological_loss_seed{result['seed']}/config.json")
            assert result["initial_state_sha256"] == config["initial_state_sha256"] == paired["initial_state_sha256"]
            history = pd.read_csv(trial / "history.csv")
            assert list(history.epoch) == list(range(config["eval_every"], config["epochs"] + 1, config["eval_every"]))
            smooth = history.val_rmse.rolling(config["val_smooth"], min_periods=config["val_smooth"]).mean()
            eligible = smooth.notna() & history.epoch.ge(config["min_epoch"])
            epoch = int(history.loc[smooth[eligible].idxmin(), "epoch"])
            assert epoch == result["best_epoch"]
            assert sha(trial / "checkpoint.pt") == result["checkpoint_sha256"]
            check_predictions(trial / "validation_predictions.npz", config, result["metrics"], dataset,
                              denominators, ("train", "val"))
            row = history[history.epoch.eq(epoch)].iloc[0]
            losses.append(dict(dataset=dataset, lambda_ode=result["lambda_ode"], lambda_k=result["lambda_k"],
                seed=result["seed"], best_epoch=epoch, data_loss=row.data_loss, weighted_ode_loss=row.L_m,
                weighted_k_loss=row.L_y, weighted_ode_to_data=row.L_m / row.data_loss,
                weighted_k_to_data=row.L_y / row.data_loss))
            provenance[str(trial.relative_to(ROOT))] = dict(checkpoint_sha256=result["checkpoint_sha256"],
                predictions_sha256=sha(trial / "validation_predictions.npz"), test_evaluations=0,
                resumptions=len(result["resumes"]))
            new_count += 1
        prior_ode = read(PREVIOUS / dataset / "selected.json")["selected_positive"]["lambda_ode"]
        best_new = next(r for r in ranking if min(pair(r)) > 0 and all(
            row["origin"] == "new joint validation-only trial" for row in groups[pair(r)].values()))
        coefficients.append(dict(dataset=dataset, original_lambda_ode=BASE[dataset], original_lambda_k=BASE_K[dataset],
            previous_tuned_lambda_ode=prior_ode, previous_tuned_lambda_k=BASE_K[dataset],
            retains_previous_tuned_pair=pair(best) == (prior_ode, BASE_K[dataset]),
            selected_lambda_ode=best["lambda_ode"], selected_lambda_k=best["lambda_k"],
            val_mean=best["val_mean"], val_sd=best["val_sd"],
            best_new_lambda_ode=best_new["lambda_ode"], best_new_lambda_k=best_new["lambda_k"],
            best_new_val_mean=best_new["val_mean"], best_new_val_sd=best_new["val_sd"],
            overall_lambda_ode=ranking[0]["lambda_ode"], overall_lambda_k=ranking[0]["lambda_k"],
            overall_val_mean=ranking[0]["val_mean"]))
        effective = {k: v for k, v in control.items() if k not in ("variant", "seed", "initial_state_sha256",
                     "cuda_visible_devices", "source_git_commit", "torch_version", "gpu_name")}
        effective.update(physic=best["lambda_ode"], ymax=best["lambda_k"], lambda_ode=best["lambda_ode"],
            lambda_k=best["lambda_k"], effective_weights=dict(physic=best["lambda_ode"], ymax=best["lambda_k"], r=0., mono=0.),
            selection_source=str((folder / "selected.json").relative_to(ROOT)), selection_sha256=sha(folder / "selected.json"),
            seeds=[1, 2, 3], checkpoints=selected["selected_phyto"]["checkpoints"])
        atomic(folder / "selected_config.json", effective)
        roles = ["selected_phyto"] + (["selected_overall"] if pair(best) != pair(ranking[0]) else [])
        for role in roles:
            chosen = selected[role]
            for seed in (1, 2, 3):
                final = folder / "final" / role / f"seed{seed}"
                result = read(final / "result.json")
                assert pair(result) == pair(chosen) and result["seed"] == seed and result["test_evaluations"] == 1
                assert result["selection_sha256"] == sha(folder / "selected.json")
                assert result["all_selections_sha256"] == sha(out / "all_selections_frozen.json")
                cp = chosen["checkpoints"][str(seed)]
                assert result["checkpoint_sha256"] == cp["sha256"] == sha(ROOT / cp["path"])
                np.testing.assert_allclose(result["metrics"]["val"]["rmse"], groups[pair(chosen)][seed]["val_rmse"], rtol=1e-6, atol=2e-6)
                values = check_predictions(final / "predictions.npz", control, result["metrics"], dataset,
                                          denominators, ("train", "val", "test"))
                # If a previous coefficient wins, equal checkpoints must have equal
                # displayed scores, without a spurious float32/float64 improvement.
                old_tuned = read(PREVIOUS / dataset / "selected.json")["selected_positive"]["lambda_ode"]
                reference_pairs = dict(full=(BASE[dataset], BASE_K[dataset]), tuned=(old_tuned, BASE_K[dataset]),
                                       no_ode_residual=(0., BASE_K[dataset]), no_biological_loss=(0., 0.))
                for variant, p in reference_pairs.items():
                    if pair(chosen) == p:
                        for value in values:
                            previous = next(r for r in runs if r["dataset"] == dataset and r["variant"] == variant
                                            and r["seed"] == seed and r["split"] == value["split"])
                            np.testing.assert_allclose(previous["rmse"], value["rmse"], rtol=1e-6, atol=2e-6)
                            previous.update(rmse=value["rmse"], relative_error=value["relative_error"])
                runs += [r | dict(variant="joint" if role == "selected_phyto" else "joint_overall", seed=seed,
                                 source=str(final.relative_to(ROOT))) for r in values]
                provenance[str(final.relative_to(ROOT))] = dict(checkpoint_sha256=cp["sha256"],
                    predictions_sha256=sha(final / "predictions.npz"), test_evaluations=1)
                final_count += 1
    assert final_count == completed["selected_final_test_evaluations"]
    runs = pd.DataFrame(runs)
    runs.to_csv(out / "per_seed_metrics.csv", index=False)
    summary = aggregate(runs, ["dataset", "variant", "split"])
    summary.to_csv(out / "comparison.csv", index=False)
    coefficients = pd.DataFrame(coefficients)
    coefficients.to_csv(out / "selected_coefficients.csv", index=False)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(out / "validation_trials.csv", index=False)
    validation.groupby(["dataset", "lambda_ode", "lambda_k"]).val_rmse.agg(["mean", "std", "count"]).reset_index().to_csv(out / "validation_summary.csv", index=False)
    pd.DataFrame(losses).to_csv(out / "loss_contributions.csv", index=False)
    changes = []
    for dataset in BASE:
        for split in ("train", "val", "test"):
            paired = runs[runs.dataset.eq(dataset) & runs.split.eq(split)].pivot(index="seed", columns="variant", values="rmse")
            for ref in ("no_biological_loss", "full", "no_ode_residual", "tuned"):
                delta = paired.joint - paired[ref]
                changes.append(dict(dataset=dataset, split=split, reference=ref, mean_paired_delta=delta.mean(),
                    paired_delta_sd=delta.std(ddof=1), error_reduction_percent=(1 - paired.joint.mean() / paired[ref].mean()) * 100,
                    joint_better_seed_count=int((delta < -1e-12).sum()), n_seeds=3))
    changes = pd.DataFrame(changes)
    changes.to_csv(out / "paired_changes.csv", index=False)
    write_report(out, summary, coefficients, changes, new_count, completed, selections)
    plot(out, runs, summary, validation, selections, protocol)
    all_baselines(out, runs)
    atomic(out / "validation.json", dict(status="passed", new_full_training_trials=new_count,
        selected_final_test_evaluations=final_count, fixed_settings_and_paired_initialization="verified",
        checkpoint_selection="recomputed from validation histories", candidate_selection="recomputed from validation-only records",
        predictions="all new train/validation and selected final metrics recomputed with original masks and denominators",
        source_sha256="unchanged", all_selections_frozen_before_final_tests=True,
        resumption_verification="20-epoch resumed and uninterrupted states and histories were bitwise identical for every dataset",
        limitation=protocol["evaluation_limitation"], artifacts=provenance, summary_code_sha256=sha(Path(__file__))))
    print(coefficients.to_string(index=False))
    print(summary[summary.split.eq("test")].to_string(index=False))
    print(f"Audit passed: {new_count} full new trials, {final_count} selected final evaluations")


def write_report(out, summary, coefficients, changes, new_count, completed, selections):
    table = summary.set_index(["dataset", "variant", "split"])
    retained = [LABELS[r.dataset] for r in coefficients.itertuples() if r.retains_previous_tuned_pair]
    outcome = ("The joint search retains the previous ODE-coefficient-tuned pair for " + ", ".join(retained) +
        ". No newly confirmed three-seed pair improves mean validation RMSE for these datasets. Their selected checkpoints and test results are unchanged. This is a result within the stated search budget, not proof of a global optimum.") if retained else "All datasets select a new coefficient pair."
    text = ["# Joint tuning of the PhytoODE physics coefficients", "",
        "**" + outcome + "**", "",
        "Both lambda_ODE and lambda_K were varied. The main no-physics baseline sets both coefficients to zero and retains the same latent ODE, initial weights and optimizer weight decay. K-only and ODE-only boundaries were included in the search.", "",
        f"Completed {new_count} new full-length trials in {completed['seconds'] / 3600:.2f} hours, reusing the previous validation records. Each dataset screened a 6-by-5 Cartesian grid with seed 1, followed by four geometric corner refinements around the best both-positive pair. The three best both-positive pairs and the best pair on each one-loss boundary were confirmed with seeds 2 and 3. All three-seed candidates, including historical controls, entered the final validation ranking.", "",
        "The selected PhytoODE is the both-positive pair with the lowest three-seed mean validation RMSE. The unrestricted validation winner is also reported, even if a coefficient is zero. Test error did not determine either selection. All dataset selections and checkpoint hashes were frozen before the selected final evaluations.", "",
        "**The same test sets had already been inspected during earlier experiments. These are follow-up results on reused test sets, not independent confirmation.** Architecture, optimizer, epoch budgets and validation checkpoint rules remain fixed. The no-physics baseline was not independently retuned.", "",
        "| Dataset | Original (ODE, K) | Previous ODE tuning (ODE, K) | Selected joint tuning (ODE, K) | Overall validation winner (ODE, K) |", "|---|---:|---:|---:|---:|"]
    for r in coefficients.itertuples():
        text.append(f"| {LABELS[r.dataset]} | ({r.original_lambda_ode:g}, {r.original_lambda_k:g}) | ({r.previous_tuned_lambda_ode:.6g}, {r.previous_tuned_lambda_k:g}) | ({r.selected_lambda_ode:.6g}, {r.selected_lambda_k:.6g}) | ({r.overall_lambda_ode:.6g}, {r.overall_lambda_k:.6g}) |")
    text += ["", "The best newly trained both-positive pairs with all three seeds are compared below. The retained candidates also have three seeds. Smaller seed-1 screening errors do not necessarily yield smaller three-seed means.", "",
        "| Dataset | Best new pair (ODE, K) | New validation RMSE ± SD | Selected validation RMSE ± SD |", "|---|---:|---:|---:|"]
    for r in coefficients.itertuples():
        n = DIGITS[r.dataset]
        text.append(f"| {LABELS[r.dataset]} | ({r.best_new_lambda_ode:.6g}, {r.best_new_lambda_k:.6g}) | {r.best_new_val_mean:.{n}f} ± {r.best_new_val_sd:.{n}f} | {r.val_mean:.{n}f} ± {r.val_sd:.{n}f} |")
    text += ["", "## Train / validation / test", "",
        "Cells are **RMSE ± sample SD / relative RMSE (%) ± sample SD**, for seeds 1--3. Bold marks the lowest mean among the displayed variants within each dataset and split. Relative RMSE is the mean per-curve RMSE divided by the mean scored target for that split, times 100; it is not MAPE. Wheat is measured in metres, Arabidopsis in centimetres, and maize in relative UAV-height target units (not percentages).", "",
        "| Dataset | Model | Train | Validation | Test |", "|---|---|---:|---:|---:|"]
    latex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Joint validation-based tuning of $\lambda_{\mathrm{ODE}}$ and $\lambda_K$. Entries are three-seed mean RMSE $\pm$ SD / relative RMSE (\%) $\pm$ SD; bold marks the lowest displayed mean per dataset and split. Wheat: m; maize: relative UAV-height units; Arabidopsis: cm. The no-physics latent Neural ODE sets both coefficients to zero. Architecture and training budgets are fixed; test sets are reused from earlier experiments.}",
        r"\label{tab:joint-physics-tuning}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llccc}",
        r"\toprule", r"Dataset & Model & Train & Validation & Test \\", r"\midrule"]
    for dataset in BASE:
        variants = [v for v in ORDER if (dataset, v, "test") in table.index]
        for variant in variants:
            cells, tex_cells = [], []
            for split in ("train", "val", "test"):
                r = table.loc[dataset, variant, split]
                digits = DIGITS[dataset]
                value = f"{r.rmse_mean:.{digits}f} ± {r.rmse_sd:.{digits}f} / {r.relative_error_mean:.2f} ± {r.relative_error_sd:.2f}"
                best = min(table.loc[dataset, v, split].rmse_mean for v in variants)
                bold = abs(r.rmse_mean - best) <= 1e-12
                cells.append("**" + value + "**" if bold else value)
                value = value.replace(" ± ", r"\pm")
                tex_cells.append(r"$\mathbf{" + value + "}$" if bold else "$" + value + "$")
            text.append("| " + LABELS[dataset] + " | " + NAMES[variant] + " | " + " | ".join(cells) + " |")
            latex.append(LABELS[dataset] + " & " + NAMES[variant] + " & " + " & ".join(tex_cells) + r" \\")
        latex.append(r"\bottomrule" if dataset == list(BASE)[-1] else r"\midrule")
    latex += [r"\end{tabular}}", r"\end{table}"]
    (out / "joint_lambda_table.tex").write_text("\n".join(latex) + "\n")
    text += ["", "## Paired test comparisons", "", "Positive reduction means lower test RMSE after joint tuning.", "",
             "| Dataset | Reference | Error reduction (%) | Seeds with lower error |", "|---|---|---:|---:|"]
    korean = ["# 두 physics 계수 공동 튜닝 결과", "", "선택 기준은 seed 1–3의 평균 validation RMSE입니다. PhytoODE는 두 계수가 모두 양수인 후보에서 선택했고, 계수가 0인 후보까지 포함한 전체 1위도 별도로 기록했습니다.", ""]
    if retained:
        korean += ["**" + ", ".join(retained) + "는 이전 ODE 계수 튜닝의 최적 조합을 유지합니다. 이번 탐색에서 새로 검증한 조합은 세 seed 평균 validation RMSE를 개선하지 못했습니다. 선택된 체크포인트와 test 오차도 기존과 같습니다. 정해진 탐색 범위와 예산 내 결과이며 전역 최적이라는 뜻은 아닙니다.**", ""]
    for dataset in BASE:
        chosen = selections[dataset]
        korean.append(f"- {LABELS[dataset]}: PhytoODE (λODE, λK) = {pair(chosen['selected_phyto'])}; 전체 validation 1위 = {pair(chosen['selected_overall'])}.")
    korean += ["", "| 데이터셋 | 비교 대상 | Test RMSE 감소율 | 개선된 seed 수 |", "|---|---|---:|---:|"]
    for r in changes[changes.split.eq("test")].itertuples():
        line = f"| {LABELS[r.dataset]} | {NAMES[r.reference]} | {r.error_reduction_percent:+.2f}% | {r.joint_better_seed_count}/3 |"
        text.append(line)
        korean.append(line)
    text += ["", "![Seed-1 joint validation grid](validation_grid.png)", "",
        "Each cell shows seed-1 validation RMSE for a prespecified coarse-grid pair, with separate colour scales by dataset. The title gives the final three-seed selection, which may be a refined or historical pair outside the displayed grid. All additional candidates and their seed counts are in validation_trials.csv and validation_summary.csv. A favourable seed-1 cell alone does not determine the final selection.", "",
        "![Paired test errors](test_relative_errors.png)", "",
        "Bars show mean test relative RMSE with sample SD. Connected points represent the same initialization seed across variants. These three seeds quantify initialization variability, not variation across independent seasons or experiments.", "",
        "Wheat preserves the original scoring mask, including 475 initial-fill points among 1,368 test points. Maize scores the original relative UAV-height targets and uses the original training-only normalization. Arabidopsis holds out plants within known genotype/temperature combinations and scores four observations per curve. These constraints limit claims about generalization to unseen genotypes or independent environments.", "",
        "The coefficients multiply losses with different dimensions: the data term is RMSE, the ODE term is squared height-per-day residual, and the K term is absolute maximum-height consistency, all in each model's internal height scale. Large numerical coefficients therefore do not by themselves imply stronger gradient influence. loss_contributions.csv records weighted terms just before the selected epoch's optimizer step.", "",
        "## Reproduction and artifacts", "", "```bash",
        ".venv/bin/python experiments/tune_joint_lambdas.py --output experiments/results/joint_lambda_repeat",
        ".venv/bin/python experiments/joint_lambda_status.py --output experiments/results/joint_lambda_repeat",
        "# If interrupted, use the same folder and add --resume to the training controller.",
        ".venv/bin/python experiments/summarize_joint_lambdas.py --output experiments/results/joint_lambda_repeat", "```", "",
        "The launcher checks historical full-model and both-zero training prefixes, plus bitwise equality of a resumed and uninterrupted 20-epoch run, before each dataset search. Trial state includes optimizer, scheduler and all RNG states. Completed trials and frozen adaptive candidate lists are reused on resume. Ephemeral resume states and diagnostic tensors stay local; selected checkpoints, predictions, compact histories and manifests are versioned.", "",
        "selected_config.json stores the effective selected coefficients and unchanged training settings. validation.json audits source/data hashes, paired initialization, full training budgets, validation selection and prediction-derived metrics. joint_lambda_table.tex requires booktabs and graphicx; resizebox limits the width to the current line. The baseline comparison is in all_baselines.md. Earlier reports are preserved as separate experiments."]
    korean += ["", "감소율이 음수면 공동 튜닝의 test 오차가 증가한 것입니다. Validation 개선이 test 개선을 보장하지 않으며, seed 3개의 차이만으로 통계적 유의성이나 모든 환경에서의 우위를 주장할 수 없습니다.", "",
        "Latent Neural ODE (no physics)는 λODE=λK=0입니다. K loss only는 λODE만 0인 별도의 부분 ablation입니다. 모든 비교는 동일한 latent ODE 구조를 유지하므로 ODE 적분 자체의 효과를 검증한 실험은 아닙니다.", "",
        "기존에 확인한 test set을 다시 사용한 후속 실험입니다. 이번 탐색의 계수 선택에는 test 오차를 사용하지 않았습니다. RMSE / Relative RMSE의 전체 train·validation·test 표와 표준편차는 README.md와 joint_lambda_table.tex에 있습니다."]
    (out / "README.md").write_text("\n".join(text) + "\n")
    (out / "interpretation_ko.md").write_text("\n".join(korean) + "\n")


def plot(out, runs, summary, validation, selections, protocol):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for ax, dataset in zip(axes, BASE):
        grid = protocol["grids"][dataset]
        odes, ks = sorted({p[0] for p in grid}), sorted({p[1] for p in grid})
        values = validation[validation.dataset.eq(dataset) & validation.seed.eq(1)].set_index(["lambda_ode", "lambda_k"])
        matrix = np.array([[values.loc[(ode, k), "val_rmse"] for ode in odes] for k in ks])
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis_r")
        midpoint = (matrix.min() + matrix.max()) / 2
        for i, k in enumerate(ks):
            for j, ode in enumerate(odes):
                ax.text(j, i, f"{matrix[i, j]:.{DIGITS[dataset]}f}", ha="center", va="center", fontsize=7.5,
                        color="white" if matrix[i, j] > midpoint else "black")
        chosen = pair(selections[dataset]["selected_phyto"])
        ax.set_title(f"{LABELS[dataset]}\nSelected (ODE, K) = ({chosen[0]:.4g}, {chosen[1]:.4g})", fontsize=10)
        ax.set_xticks(range(len(odes)), [f"{v:.4g}" for v in odes], rotation=30)
        ax.set_yticks(range(len(ks)), [f"{v:g}" for v in ks])
        ax.set_xlabel(r"$\lambda_{\mathrm{ODE}}$")
        ax.set_ylabel(r"$\lambda_K$")
        fig.colorbar(im, ax=ax, label=f"Seed-1 validation RMSE ({UNITS[dataset]})", shrink=.85)
    fig.savefig(out / "validation_grid.png", dpi=220)
    fig.savefig(out / "validation_grid.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    table = summary.set_index(["dataset", "variant", "split"])
    labels = dict(no_biological_loss="Latent NODE\nNo physics", full="Original\nPhytoODE",
        no_ode_residual="PhytoODE\nK loss only", tuned="PhytoODE\nODE tuning", joint="PhytoODE\nJoint tuning",
        joint_overall="Overall\nwinner")
    for ax, dataset in zip(axes, BASE):
        variants = [v for v in ORDER if (dataset, v, "test") in table.index]
        means = [table.loc[dataset, v, "test"].relative_error_mean for v in variants]
        sds = [table.loc[dataset, v, "test"].relative_error_sd for v in variants]
        x = np.arange(len(variants))
        ax.bar(x, means, yerr=sds, color=[COLORS[v] for v in variants], capsize=3, width=.65, alpha=.85)
        paired = runs[runs.dataset.eq(dataset) & runs.split.eq("test")].pivot(index="seed", columns="variant", values="relative_error")
        for seed in (1, 2, 3):
            ax.plot(x + (seed - 2) * .04, paired.loc[seed, variants], "o-", color="black", alpha=.35, ms=3, lw=.7)
        ax.set_xticks(x, [labels[v] for v in variants], fontsize=7)
        ax.set_title(LABELS[dataset])
        ax.set_ylabel("Test relative RMSE (%)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(out / "test_relative_errors.png", dpi=220)
    fig.savefig(out / "test_relative_errors.pdf")
    plt.close(fig)


def all_baselines(out, runs):
    old = pd.read_csv(ROOT / "paper/relative_errors/per_run_relative_errors.csv")
    old = old[old.is_primary & ~old.model.eq("Ours")].copy()
    old["relative_error"] = old.rrmse_percent
    new = runs.copy()
    new["model"] = new.variant.map(NAMES)
    columns = ["dataset", "model", "seed", "split", "rmse", "relative_error", "unit"]
    all_runs = pd.concat([old[columns], new[columns]], ignore_index=True)
    all_runs.to_csv(out / "all_baselines_per_seed.csv", index=False)
    summary = aggregate(all_runs, ["dataset", "model", "split"])
    summary.to_csv(out / "all_baselines.csv", index=False)
    table = summary.set_index(["dataset", "model", "split"])
    text = ["# All model comparisons", "", "Mean RMSE / relative RMSE (%). Bold marks the best displayed mean within each dataset and split. Existing baseline results are reused without retraining or retuning; stochastic models have three seeds and deterministic ODE fits one. Units: wheat m, maize relative UAV height, Arabidopsis cm. These follow-up test sets were previously inspected. Full per-seed metrics and SD are provided in the CSV files.", "",
            "| Dataset | Model | Train | Validation | Test |", "|---|---|---:|---:|---:|"]
    for dataset in BASE:
        models = list(summary[summary.dataset.eq(dataset)].model.drop_duplicates())
        for model in models:
            cells = []
            for split in ("train", "val", "test"):
                row = table.loc[dataset, model, split]
                value = f"{row.rmse_mean:.{DIGITS[dataset]}f} / {row.relative_error_mean:.2f}%"
                best = min(table.loc[dataset, m, split].rmse_mean for m in models)
                cells.append("**" + value + "**" if abs(row.rmse_mean - best) <= 1e-12 else value)
            text.append("| " + LABELS[dataset] + " | " + model + " | " + " | ".join(cells) + " |")
    (out / "all_baselines.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
