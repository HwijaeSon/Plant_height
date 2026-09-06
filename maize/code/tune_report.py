"""Evaluate only a frozen tuning winner and report it beside the original maize experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import ROOT, make_dataset
from run_experiment import metrics, tensor_batch
from tune_trial import atomic_json, build, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    out = args.results.resolve()
    selected = json.loads((out/"selected.json").read_text())
    protocol = json.loads((out/"protocol.json").read_text())
    completed = json.loads((out/"completed.json").read_text())
    for path, digest in protocol["source_sha256"].items():
        if hashlib.sha256((ROOT.parent/path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Tuning source changed: {path}")
    if any(row["test_evaluations"] != 0 for row in completed):
        raise ValueError("Candidate search recorded test evaluations")
    candidates = pd.DataFrame([dict(candidate=r["candidate"], seed=r["seed"], phase=r["phase"],
        val_rmse=r["val_rmse"], best_epoch=r["best_epoch"], epochs=r["epochs"], n_params=r["n_params"],
        seconds=r["seconds"], **{f"config_{k}": v for k, v in r["config"].items()}) for r in completed])
    ranking = candidates.groupby("candidate").agg(val_mean=("val_rmse", "mean"), val_sd=("val_rmse", "std"),
        n_seeds=("seed", "nunique"), n_params=("n_params", "first"))
    eligible = ranking[ranking.n_seeds.eq(3)].sort_values(["val_mean", "val_sd", "n_params"])
    if eligible.index[0] != selected["candidate"]:
        raise ValueError("Frozen winner does not match validation-only ranking")
    candidates.to_csv(out/"all_validation_trials.csv", index=False)
    eligible.to_csv(out/"validation_ranking.csv")

    final = out/"final"
    figdir = final/"figures"
    figdir.mkdir(parents=True, exist_ok=True)
    ds = make_dataset(split="chronological")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs, curve_rows, observations = [], [], []
    prediction_sets = []
    for seed in [1, 2, 3]:
        trial = out/"trials"/selected["candidate"]/f"seed{seed}"
        checkpoint_path = trial/"checkpoint.pt"
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != selected["checkpoint_sha256"][str(seed)]:
            raise ValueError("Checkpoint changed after final selection")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["config"] != selected["config"] or state["genotypes"] != ds.genotypes:
            raise ValueError("Checkpoint configuration or genotype order mismatch")
        model = build(state["config"], len(ds.genotypes)).to(device)
        model.load_state_dict(state["state_dict"])
        model.eval()
        reference = json.loads((trial/"result.json").read_text())
        result = dict(model="Latent Neural ODE (tuned)", model_key="latent_tuned", seed=seed,
            candidate=selected["candidate"], best_epoch=state["epoch"], n_params=reference["n_params"],
            epochs=reference["epochs"], seconds=reference["seconds"], y_scale=ds.y_scale)
        predictions = {}
        with torch.no_grad():
            for label, attr in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
                batch = getattr(ds, attr)
                predictions[attr] = predict(model, tensor_batch(batch, device)).cpu().numpy()
                values, rmses, maes = metrics(predictions[attr], batch, ds.y_scale)
                result.update({f"{label}_{k}": v for k, v in values.items()})
                if label in ["train", "val"]:
                    np.testing.assert_allclose(values["rmse"], reference[f"{label}_rmse"], rtol=1e-6, atol=2e-5)
                for i in range(len(batch)):
                    curve_rows.append(dict(model=result["model"], seed=seed, split=label, year=int(batch.year[i]),
                        genotype=ds.genotypes[batch.g_idx[i]], rmse=rmses[i], mae=maes[i]))
                rr, cc = np.where(batch.mask > 0)
                observations.append(pd.DataFrame(dict(model=result["model"], seed=seed, split=label,
                    genotype=np.asarray(ds.genotypes)[batch.g_idx[rr]], year=batch.year[rr], dap=ds.days[cc],
                    observed_relative=batch.y[rr, cc].astype(float)*ds.y_scale,
                    predicted_relative=predictions[attr][rr, cc].astype(float)*ds.y_scale)))
            probe = tensor_batch(ds.test.subset(np.arange(3)), device)
            before = predict(model, probe)
            probe["y"].fill_(float("nan")); probe["mask"].zero_()
            torch.testing.assert_close(predict(model, probe), before)
        folder = final/"runs"/f"latent_seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/"predictions.npz", **predictions)
        atomic_json(folder/"result.json", result)
        runs.append(result)
        prediction_sets.append(predictions)
        del model
    runs = pd.DataFrame(runs)
    runs.to_csv(final/"all_runs.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(final/"per_curve_metrics.csv", index=False)
    pd.concat(observations, ignore_index=True).to_csv(final/"observed_predictions.csv.gz", index=False)
    olddir = ROOT/"results/chronological_final_seed1_3"
    comparison = pd.read_csv(olddir/"comparison.csv")
    row = dict(model="Latent Neural ODE (tuned)", model_key="latent_tuned", n_runs=3,
               n_params=int(runs.n_params.iloc[0]), seconds_mean=float(runs.seconds.mean()))
    for split in ["train", "val", "test"]:
        for metric in ["rmse", "mae", "pooled_rmse"]:
            values = runs[f"{split}_{metric}"]
            row[f"{split}_{metric}_mean"] = values.mean()
            row[f"{split}_{metric}_sd"] = values.std(ddof=1)
    comparison = pd.concat([comparison, pd.DataFrame([row])], ignore_index=True)
    denominators = {}
    for split, attr in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
        b = getattr(ds, attr)
        denominator = b.y[b.mask>0].astype(float).mean()*ds.y_scale
        denominators[split] = denominator
        for metric in ["rmse", "mae"]:
            runs[f"{split}_r{metric}_percent"] = runs[f"{split}_{metric}"]/denominator*100
            for statistic in ["mean", "sd"]:
                comparison[f"{split}_r{metric}_percent_{statistic}"] = comparison[f"{split}_{metric}_{statistic}"]/denominator*100
    comparison.to_csv(final/"comparison.csv", index=False)
    runs.to_csv(final/"all_runs.csv", index=False)
    candidates["val_rrmse_percent"] = candidates.val_rmse/denominators["val"]*100
    eligible = eligible.copy()
    eligible["val_rrmse_percent_mean"] = eligible.val_mean/denominators["val"]*100
    eligible["val_rrmse_percent_sd"] = eligible.val_sd/denominators["val"]*100
    candidates.to_csv(out/"all_validation_trials.csv", index=False)
    eligible.to_csv(out/"validation_ranking.csv")
    atomic_json(final/"relative_error_denominators.json", denominators)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        for ext in ["png", "pdf"]:
            fig.savefig(figdir/f"{name}.{ext}", dpi=180)
        plt.close(fig)
    labels = ["Logi-ODE", "Temp-ODE", "RF", "LSTM-NN", "Logi-PINN", "Ours: original", "Ours: tuned"]
    colors = ["#a1a9b4", "#71859e", "#d5a14c", "#678a5b", "#936fab", "#216b91", "#c15b40"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, split in zip(axes, ["val", "test"]):
        vals = comparison[f"{split}_rrmse_percent_mean"]
        errors = comparison[f"{split}_rrmse_percent_sd"].fillna(0)
        ax.barh(labels, vals, xerr=errors, color=colors, capsize=3)
        for i, (value, error) in enumerate(zip(vals, errors)):
            ax.text(value+error+.7, i, f"{value:.2f}%", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.set(xlabel="rRMSE (%)", title="2020 validation" if split == "val" else "2021 test",
               xlim=(0, max(vals+errors)*1.2))
        ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
    fig.suptitle("Maize latent ODE tuning · original and validation-selected model\n"
                 "rRMSE = RMSE / split-wide mean target × 100; error bars: seed SD", fontsize=12)
    save(fig, "tuning_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    single = candidates[candidates.phase.eq("screen")].reset_index(drop=True)
    axes[0].scatter(np.arange(1, len(single)+1), single.val_rrmse_percent, color="#71859e", alpha=.7, s=24)
    axes[0].plot(np.arange(1, len(single)+1), single.val_rrmse_percent.cummin(), color="#216b91", lw=2)
    axes[0].set(xlabel="Completed screening trial", ylabel="Validation rRMSE (%)", title="Single-seed screening")
    top = eligible.head(12).iloc[::-1]
    axes[1].barh(top.index, top.val_rrmse_percent_mean, xerr=top.val_rrmse_percent_sd.fillna(0), capsize=3,
                 color=["#c15b40" if c == selected["candidate"] else "#71859e" for c in top.index])
    axes[1].set(xlabel="Validation rRMSE (%) · mean ± seed SD", title="Best candidates completed on all three seeds")
    save(fig, "validation_search")

    base = protocol["base_config"]
    controls = candidates[candidates.apply(lambda r: r.phase == "screen" and all(
        r[f"config_{key}"] == value for key, value in base.items() if key not in ["lr", "physic"]), axis=1)]
    control_grid = controls.pivot(index="config_lr", columns="config_physic", values="val_rrmse_percent").sort_index()
    controls.to_csv(final/"learning_rate_physics_controls.csv", index=False)
    finalists = json.loads((out/"replicate_selection.json").read_text())
    length_rows = []
    for finalist in finalists:
        config = finalist["config"]
        matches = candidates[candidates.apply(lambda r: all(
            r[f"config_{key}"] == value for key, value in config.items() if key != "epochs"), axis=1)]
        for name, group in matches.groupby("candidate"):
            if group.seed.nunique() == 3:
                length_rows.append(dict(architecture=finalist["candidate"], candidate=name,
                    epochs=int(group.config_epochs.iloc[0]), val_rrmse_percent_mean=group.val_rrmse_percent.mean(),
                    val_rrmse_percent_sd=group.val_rrmse_percent.std(ddof=1)))
    lengths = pd.DataFrame(length_rows).sort_values(["architecture", "epochs"])
    lengths.to_csv(final/"training_length_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    im = axes[0].imshow(control_grid.to_numpy(), cmap="YlOrRd", aspect="auto")
    axes[0].set_xticks(range(len(control_grid.columns)), [f"{x:g}" for x in control_grid.columns])
    axes[0].set_yticks(range(len(control_grid.index)), [f"{x:g}" for x in control_grid.index])
    for i in range(len(control_grid.index)):
        for j in range(len(control_grid.columns)):
            value = control_grid.iloc[i, j]
            if np.isfinite(value):
                axes[0].text(j, i, f"{value:.2f}", ha="center", va="center",
                             color="white" if value > control_grid.to_numpy().mean() else "black")
    axes[0].set(xlabel="Physics loss weight", ylabel="Learning rate", title="Original architecture · seed 1 only")
    fig.colorbar(im, ax=axes[0], label="Validation rRMSE (%)", shrink=.85)
    for name, group in lengths.groupby("architecture"):
        axes[1].errorbar(group.epochs, group.val_rrmse_percent_mean, yerr=group.val_rrmse_percent_sd,
                         marker="o", capsize=4, label=name)
    axes[1].set(xlabel="Training epoch budget", ylabel="Validation rRMSE (%)",
                title="Top three settings · mean ± seed SD", xticks=[1500, 3000, 4500])
    axes[1].legend(frameon=False); axes[1].grid(alpha=.15)
    save(fig, "tuning_controls")

    selected_examples = pd.read_csv(olddir/"figures/selected_genotypes.csv", dtype={"genotype": str})
    old_predictions = {}
    for key in ["logistic", "temperature", "rf", "lstm", "pinn", "latent"]:
        seeds = [0] if key in ["logistic", "temperature"] else [1, 2, 3]
        old_predictions[key] = np.mean([np.load(olddir/"runs"/f"{key}_seed{s}"/"predictions.npz")["test"] for s in seeds], axis=0)
    tuned = np.mean([p["test"] for p in prediction_sets], axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True, layout="constrained")
    for ax, case in zip(axes.flat, selected_examples.iloc[[0, 2, 4, 1, 3, 5]].itertuples()):
        i = int(case.test_row)
        cols = np.flatnonzero(ds.test.mask[i])
        support = np.arange(cols[0], cols[-1]+1)
        ax.scatter(ds.days[cols], ds.test.y[i, cols]*ds.y_scale, c="black", s=24, label="Observed mean", zorder=10)
        for (key, pred), label, color in zip(old_predictions.items(), labels, colors):
            ax.plot(ds.days[support], pred[i, support]*ds.y_scale, color=color,
                    lw=1.8 if key == "latent" else 1.1, ls="--" if key == "latent" else "-", label=label)
        ax.plot(ds.days[support], tuned[i, support]*ds.y_scale, color=colors[-1], lw=2.5, label=labels[-1])
        ax.set_title(case.genotype)
        ax.grid(alpha=.15)
    for ax in axes[-1]: ax.set_xlabel("Days after planting")
    for ax in axes[:, 0]: ax.set_ylabel("Relative UAV height")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Same six previously displayed genotypes · seed-mean predictions", fontsize=12)
    save(fig, "test_predictions")

    def number(record, metric):
        mean, sd = record[f"{metric}_mean"], record[f"{metric}_sd"]
        return f"{mean:.2f}" if pd.isna(sd) else f"{mean:.2f} ± {sd:.2f}"
    elapsed = (time.time()-protocol["started_unix"])/60
    lines = ["# 옥수수 우리 모델 하이퍼파라미터 튜닝", "",
        f"완료 실행 {len(completed)}회, 평가한 설정 {candidates.candidate.nunique()}개, "
        f"3개 시드를 모두 완료한 설정 {len(eligible)}개. 검색 시작 후 약 {elapsed:.1f}분.", "",
        "학습 2018–2019 / 검증 2020 / 시험 2021. 후보 탐색에는 검증 RMSE만 사용했다. "
        "전체 학습을 완료한 시드 1–3의 검증 RMSE 평균이 가장 낮은 설정을 먼저 확정하고, "
        "확정된 세 체크포인트만 시험 평가했다. 시험 결과를 보고 설정이나 시드를 다시 고르지 않았다.", "",
        "이번 탐색은 우리 모델에만 적용했다. baseline 행은 기존 완료 실험의 고정 결과이며, "
        "모델마다 동일한 하이퍼파라미터 탐색 예산을 준 비교는 아니다.", "",
        f"선택 설정: `{selected['candidate']}`, 검증 rRMSE "
        f"**{selected['val_mean']/denominators['val']*100:.2f} ± {selected['val_sd']/denominators['val']*100:.2f}%**.", "",
        f"선택 모델의 파라미터 수는 **{int(runs.n_params.iloc[0]):,}개**다. "
        "기존 옥수수 모델은 3,187개이며, 이번 탐색에는 네트워크 크기 변경도 포함했다.", "",
        "```json", json.dumps(selected["config"], indent=2), "```", "",
        f"[선택 설정](../configs/{selected['candidate']}.json) · [선택 기록·해시](../selected.json) · "
        + " · ".join(f"[시드 {seed} 체크포인트](../trials/{selected['candidate']}/seed{seed}/checkpoint.pt)"
                     for seed in [1, 2, 3]), "",
        "| 모델 | 검증 rRMSE (%) | 시험 rRMSE (%) | 시험 rMAE (%) |", "|---|---:|---:|---:|"]
    for _, record in comparison.iterrows():
        lines.append(f"| {record.model} | {number(record, 'val_rrmse_percent')} | {number(record, 'test_rrmse_percent')} | "
                     f"{number(record, 'test_rmae_percent')} |")
    lines += ["", "선택 설정의 시드별 결과:", "",
        "| 시드 | 학습 예산 (epoch) | 선택 checkpoint (epoch) | 검증 rRMSE (%) | 시험 rRMSE (%) |",
        "|---|---:|---:|---:|---:|"]
    for record in runs.itertuples():
        lines.append(f"| {record.seed} | {record.epochs} | {record.best_epoch} | "
                     f"{record.val_rrmse_percent:.2f} | {record.test_rrmse_percent:.2f} |")
    original = comparison[comparison.model_key.eq("latent")].iloc[0]
    tuned_row = comparison[comparison.model_key.eq("latent_tuned")].iloc[0]
    val_reduction = (1-tuned_row.val_rmse_mean/original.val_rmse_mean)*100
    test_reduction = (1-tuned_row.test_rmse_mean/original.test_rmse_mean)*100
    lines += ["", f"기존 우리 모델 대비 검증 RMSE 감소율 {val_reduction:.2f}%, 시험 RMSE 감소율 {test_reduction:.2f}% "
        "(음수는 오차 증가). rRMSE는 각 분할의 평균 평가 높이로 나눈 백분율이다. "
        "원단위 RMSE/MAE도 comparison.csv에 함께 보관했다. 원단위는 상대 UAV 높이이며 cm/m가 아니다. "
        "±는 3개 시드의 표준편차다.", "",
        "각 유전자형 곡선의 관측 지점에서 RMSE/MAE를 계산한 뒤 곡선 간 평균을 취했다. "
        "이 평균 오차를 해당 분할의 전체 관측 높이 평균으로 나누고 100을 곱해 rRMSE/rMAE를 산출했다. "
        "분모는 relative_error_denominators.json에 기록했다.", "",
        "2020년 한 해에 대한 반복 탐색이므로 검증 최저값 자체에는 선택 편향이 있다. "
        "2021년 시험 결과는 이 고정 설정의 한 연도 평가이며 여러 연도에 대한 우위를 확정하지 않는다.", "",
        "![튜닝 비교](figures/tuning_comparison.png)", "", "![검증 탐색](figures/validation_search.png)", "",
        "![학습률·물리 손실과 학습 길이](figures/tuning_controls.png)", "",
        "학습률·물리 손실 격자는 기존 구조에서 나머지 설정을 고정한 시드 1 비교다. "
        "학습 길이 비교는 각 설정의 시드 1–3을 처음부터 다시 학습한 결과이며, "
        "epoch 예산에 따라 cosine 학습률 스케줄도 함께 달라진다.", "",
        "![예측 비교](figures/test_predictions.png)", "",
        "예측 예시는 튜닝 전에 표시했던 동일한 유전자형 6개다. 첫 관측일부터 마지막 관측일까지만 표시했다.", "",
        "원래 비교 결과는 `chronological_final_seed1_3`에 보존했다. "
        "최종 체크포인트는 상위 폴더 `selected.json`의 후보/시드 경로와 SHA-256으로 고정되어 있다.", ""]
    (final/"README.md").write_text("\n".join(lines))
    atomic_json(final/"validation.json", dict(status="passed", selected_candidate=selected["candidate"],
        validation_ranking="recomputed from completed trials", candidate_test_evaluations=0,
        final_test_evaluations=3, checkpoint_hashes="match frozen selection", source_hashes="unchanged",
        restored_train_val_metrics="match trial results", holdout_targets="mutations do not affect predictions"))
    print(comparison[["model", "val_rrmse_percent_mean", "test_rrmse_percent_mean"]].to_string(index=False))
    print(f"Wrote {final/'README.md'}")


if __name__ == "__main__":
    main()
