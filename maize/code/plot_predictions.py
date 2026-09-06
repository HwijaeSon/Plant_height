"""Plot saved maize predictions; no fitting or checkpoint selection is performed."""
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

from data import ROOT, make_dataset

MODELS = {"logistic": "Logi-ODE", "temperature": "Temp-ODE", "rf": "RF",
          "lstm": "LSTM-NN", "pinn": "Logi-PINN", "latent": "Latent Neural ODE (ours)"}
COLORS = {"logistic": "#a1a9b4", "temperature": "#71859e", "rf": "#d5a14c",
          "lstm": "#678a5b", "pinn": "#936fab", "latent": "#216b91"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/chronological_final_seed1_3")
    args = parser.parse_args()
    out = args.results
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    protocol = json.loads((out / "protocol.json").read_text())
    for path, digest in protocol["source_sha256"].items():
        if hashlib.sha256((ROOT.parent / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Experiment input or source changed: {path}")
    ds = make_dataset(split=protocol["split"])
    batch = ds.test
    genotypes = np.asarray(ds.genotypes)[batch.g_idx]
    obs = batch.y.astype(float) * ds.y_scale
    mask = batch.mask > 0
    curve_metrics = pd.read_csv(out / "per_curve_metrics.csv", dtype={"genotype": str})
    curve_metrics = curve_metrics[curve_metrics.split.eq("test")]
    predictions, errors = {}, {}
    daily_records = []
    for key, name in MODELS.items():
        by_seed = []
        for seed in ([0] if key in ["logistic", "temperature"] else [1, 2, 3]):
            with np.load(out / "runs" / f"{key}_seed{seed}" / "predictions.npz", allow_pickle=False) as saved:
                pred = saved["test"].astype(float) * ds.y_scale
            if pred.shape != obs.shape or not np.isfinite(pred).all():
                raise ValueError(f"Invalid predictions for {key}, seed {seed}")
            rmse = np.sqrt(np.sum(np.where(mask, (pred-obs)**2, 0), axis=1) / mask.sum(axis=1))
            recorded = curve_metrics[curve_metrics.model.eq(name) & curve_metrics.seed.eq(seed)]
            recorded = recorded.set_index("genotype").loc[genotypes, "rmse"].to_numpy()
            np.testing.assert_allclose(rmse, recorded, rtol=1e-6, atol=1e-5)
            by_seed.append(pred)
            for col in np.flatnonzero(mask.any(axis=0)):
                valid = mask[:, col]
                daily_records.append(dict(model=name, seed=seed, dap=int(ds.days[col]),
                    n_genotypes=int(valid.sum()), mae=float(np.abs(pred[valid, col]-obs[valid, col]).mean())))
        predictions[key] = np.stack(by_seed)
        errors[key] = curve_metrics[curve_metrics.model.eq(name)].groupby("genotype").rmse.mean().loc[genotypes].to_numpy()

    paired = pd.DataFrame({"genotype": genotypes, "test_row": np.arange(len(batch)),
                           "ours_rmse": errors["latent"], "lstm_rmse": errors["lstm"]})
    paired["lstm_minus_ours_rmse"] = paired.lstm_rmse - paired.ours_rmse
    paired["ours_lower_rmse"] = paired.ours_rmse < paired.lstm_rmse
    paired.to_csv(figdir / "paired_genotype_errors.csv", index=False)
    ranked = paired.sort_values(["ours_rmse", "genotype"]).reset_index(drop=True)
    quantiles = np.array([0, .1, .45, .55, .9, 1])
    ranks = np.rint(quantiles * (len(ranked)-1)).astype(int)
    selected = ranked.iloc[ranks].copy()
    selected["rank_low_to_high"] = ranks + 1
    selected["target_quantile"] = quantiles
    selected["error_group"] = ["low", "low", "middle", "middle", "high", "high"]
    selected.to_csv(figdir / "selected_genotypes.csv", index=False)
    daily = pd.DataFrame(daily_records)
    daily.to_csv(figdir / "daily_errors.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})

    def save(fig, name):
        for ext in ["png", "pdf"]:
            fig.savefig(figdir / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    raw = pd.read_csv(ROOT / "data/processed/height_observations.csv", dtype={"genotype": str})
    raw = raw[raw.year.eq(ds.test_year)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True, layout="constrained")
    # Columns correspond to low / middle / high error; rows show two examples each.
    for ax, (_, case) in zip(axes.flat, selected.iloc[[0, 2, 4, 1, 3, 5]].iterrows()):
        row = int(case.test_row)
        cols = np.flatnonzero(mask[row])
        support = np.arange(cols[0], cols[-1]+1)
        reps = raw[raw.genotype.eq(case.genotype)]
        ax.scatter(reps.dap, reps.height_relative, s=20, facecolor="none", edgecolor="#9da5ae",
                   linewidth=.9, alpha=.8, label="Field-plot observations", zorder=4)
        ax.scatter(ds.days[cols], obs[row, cols], s=30, c="#17212b", edgecolor="white", linewidth=.5,
                   label="Observed genotype mean", zorder=5)
        for key in ["lstm", "latent"]:
            values = predictions[key][:, row, support]
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            label = "Ours" if key == "latent" else "LSTM-NN"
            ax.fill_between(ds.days[support], mean-sd, mean+sd, color=COLORS[key], alpha=.14, linewidth=0)
            ax.plot(ds.days[support], mean, color=COLORS[key], lw=2.3, label=label, zorder=3)
        ax.set_title(f"{case.error_group.capitalize()} error · {case.genotype}", fontsize=10.5)
        ax.text(.03, .97, f"Ours RMSE {case.ours_rmse:.1f} | LSTM {case.lstm_rmse:.1f}\n"
                f"Ours error rank {case.rank_low_to_high}/{len(batch)}", transform=ax.transAxes,
                va="top", fontsize=9, color="#394653",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=3))
        ax.grid(alpha=.14)
    # Set shared limits after every panel has contributed its data extents.
    axes.flat[0].set_ylim(bottom=-25)
    for ax in axes[-1]:
        ax.set_xlabel("Days after planting (DAP)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative UAV height")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"{ds.test_year} test year · examples spanning our model's error distribution\n"
                 "Lines: three-seed mean | Bands: ±1 seed SD | RMSE: mean of per-seed errors",
                 fontsize=12, fontweight="normal")
    save(fig, "test_error_strata")

    denominator = float(obs[mask].mean())
    all_curve_errors = pd.DataFrame({"genotype": genotypes})
    for key in MODELS:
        all_curve_errors[f"{key}_rrmse_percent"] = errors[key] / denominator * 100
    all_curve_errors.to_csv(figdir / "all_model_genotype_relative_errors.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True, layout="constrained")
    for ax, (_, case) in zip(axes.flat, selected.iloc[[0, 2, 4, 1, 3, 5]].iterrows()):
        row = int(case.test_row)
        cols = np.flatnonzero(mask[row])
        support = np.arange(cols[0], cols[-1]+1)
        reps = raw[raw.genotype.eq(case.genotype)]
        ax.scatter(reps.dap, reps.height_relative, s=20, facecolor="none", edgecolor="#9da5ae",
                   linewidth=.9, alpha=.65, label="Field-plot observations", zorder=4)
        ax.scatter(ds.days[cols], obs[row, cols], s=30, c="#17212b", edgecolor="white", linewidth=.5,
                   label="Observed genotype mean", zorder=5)
        for key, name in MODELS.items():
            mean = predictions[key][:, row, support].mean(axis=0)
            ax.plot(ds.days[support], mean, color=COLORS[key], lw=2.6 if key == "latent" else 1.4,
                    label="Ours" if key == "latent" else name, zorder=3 if key == "latent" else 2)
        ax.set_title(f"{case.error_group.capitalize()} error · {case.genotype}", fontsize=10.5)
        ax.text(.03, .97, f"Ours rRMSE {case.ours_rmse/denominator*100:.1f}%\n"
                f"Ours error rank {case.rank_low_to_high}/{len(batch)}", transform=ax.transAxes,
                va="top", fontsize=9, color="#394653",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=3))
        ax.grid(alpha=.14)
    axes.flat[0].set_ylim(bottom=-25)
    for ax in axes[-1]:
        ax.set_xlabel("Days after planting (DAP)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative UAV height")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"{ds.test_year} test year · all six models on the same six genotypes\n"
                 "Lines: seed-mean predictions | rRMSE: RMSE / test-set mean height × 100",
                 fontsize=12, fontweight="normal")
    save(fig, "test_error_strata_all_models")

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for key, name in MODELS.items():
        values = daily[daily.model.eq(name)].groupby("dap").mae.mean() / denominator * 100
        ax.plot(values.index, values, "o-", color=COLORS[key], lw=2.6 if key == "latent" else 1.3,
                markersize=4, label="Ours" if key == "latent" else name)
    ax.set(xlabel="Days after planting (DAP)", ylabel="rMAE (%)", ylim=(0, None),
           title=f"{ds.test_year} test year · relative error by observation date")
    ax.grid(alpha=.14)
    ax.legend(frameon=False, ncol=3)
    fig.supxlabel("Each date's MAE / overall test-set mean height × 100; errors averaged over seeds", fontsize=9)
    save(fig, "daily_relative_errors_all_models")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), layout="constrained")
    ax = axes[0]
    wins = paired.ours_lower_rmse
    n_wins = int(wins.sum())
    for condition, color, label in [(wins, COLORS["latent"], f"Ours lower: {n_wins}/{len(batch)}"),
                                     (~wins, "#bc6c51", f"LSTM lower or tied: {int((~wins).sum())}/{len(batch)}")]:
        ax.scatter(paired.loc[condition, "lstm_rmse"], paired.loc[condition, "ours_rmse"],
                   s=23, alpha=.65, c=color, edgecolors="none", label=label)
    limit = np.ceil(paired[["lstm_rmse", "ours_rmse"]].to_numpy().max() / 20) * 20
    ax.plot([0, limit], [0, limit], "--", c="#626c76", lw=1, label="Equal RMSE")
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="LSTM-NN RMSE", ylabel="Ours RMSE",
           title=f"Per-genotype errors · {n_wins/len(batch):.1%} lower with ours")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=.14)
    ax = axes[1]
    for key in ["latent", "lstm"]:
        name = MODELS[key]
        values = daily[daily.model.eq(name)].groupby("dap").mae.agg(["mean", "std"])
        ax.plot(values.index, values["mean"], "o-", color=COLORS[key], lw=2,
                markersize=4, label="Ours" if key == "latent" else name)
        ax.fill_between(values.index, values["mean"]-values["std"], values["mean"]+values["std"],
                        color=COLORS[key], alpha=.14, linewidth=0)
    ax.set(xlabel="Days after planting (DAP)", ylabel="Mean absolute error (relative height)",
           title="Error by observation date", ylim=(0, None))
    ax.grid(alpha=.14)
    ax.legend(frameon=False)
    fig.suptitle(f"{ds.test_year} test year · all {len(batch)} genotypes\n"
                 "Errors averaged across three seeds; bands show ±1 seed SD",
                 fontsize=12, fontweight="normal")
    save(fig, "test_error_diagnostics")

    notes = ["# 추가 예측 그림", "", f"{ds.test_year}년 시험 데이터, {len(batch)}개 유전자형. 기존 최종 실험의 저장 예측만 사용했다.",
        "", "## 전체 baseline 성장 곡선", "", "![6개 모델 성장 곡선](test_error_strata_all_models.png)", "",
        "앞서 선택한 동일한 6개 유전자형에 Logi-ODE, Temp-ODE, RF, LSTM-NN, Logi-PINN, 우리 모델을 모두 표시했다. "
        "결정론적 ODE는 1회 적합 예측, 나머지는 시드 1–3의 평균이다. 이 그림에서는 겹침을 줄이기 위해 띠를 생략했다.",
        f"패널 rRMSE는 시드별 RMSE 평균을 전체 시험 평균 높이 {denominator:.6f}로 나누고 100을 곱했다. "
        "패널마다 다른 높이를 분모로 쓰지 않는다. y축은 원래 상대 UAV 높이 단위다.",
        "", "![전체 모델의 날짜별 상대오차](daily_relative_errors_all_models.png)", "",
        "rMAE는 날짜별 MAE를 동일한 전체 시험 평균 높이로 나눈 백분율이다. 관측일 사이 선은 시각적 안내다.",
        "", "## 시드 변동을 포함한 LSTM/우리 모델 성장 곡선", "", "![성장 곡선](test_error_strata.png)", "",
        "유전자형별 우리 모델의 3개 시드 RMSE 평균을 오름차순으로 정렬하고, 0·10·45·55·90·100% 위치에서 선택했다. "
        "동률은 유전자형 ID로 정렬하며, 순위 인덱스는 `round(q * (N-1))`다. "
        "낮은 오차/중간 오차/높은 오차를 열별로 배치했다. 사후 진단용 선택이며 모델 선택에는 사용하지 않았다.",
        "", "검은 점은 해당 날짜의 유전자형별 반복 시험구 평균이며 평가 대상이다. "
        "회색 빈 점은 개별 시험구 관측값으로, 개체별 식물 높이가 아니다. "
        "선은 세 시드 예측의 평균, 띠는 시드 간 표본 표준편차 ±1이다. "
        "띠는 예측 신뢰구간이 아니며, 관측 변동과도 구별한다. 예측은 첫 관측일부터 마지막 관측일까지만 표시한다.",
        "", "패널의 RMSE는 시드별 RMSE를 먼저 계산한 뒤 평균한 값이다. "
        "시드 평균 예측선 자체의 RMSE와 다르며, 기존 비교표와 같은 집계 방식이다. 상대 UAV 단위이며 cm/m가 아니다.",
        "", "## 전체 시험 집합 오차", "", "![오차 진단](test_error_diagnostics.png)", "",
        f"왼쪽: 점 하나는 유전자형 하나이며, 대각선 아래는 우리 모델의 오차가 더 작은 경우다. "
        f"전체 {len(batch)}개 중 {n_wins}개({n_wins/len(batch):.1%})에서 우리 모델의 시드 평균 RMSE가 낮다.",
        "", "오른쪽: 날짜별로 관측이 있는 유전자형들의 MAE를 각 시드에서 계산하고, 세 시드의 평균과 표준편차를 표시했다. "
        "날짜 사이 연결선은 시각적 안내이며, 관측 없는 날짜에서 평가한 결과가 아니다.",
        "", "LSTM-NN은 최종 비교표에서 시험 RMSE가 가장 낮은 baseline이다. "
        "이 그림은 동일한 2021년 시험 결과의 진단이며 다른 연도에 대한 우위를 뜻하지 않는다.",
        "", "## 파일과 재현", "",
        "- 모든 그림은 같은 이름의 PDF로도 저장했다.",
        "- `selected_genotypes.csv`: 선택 ID, 오차, 전체 순위, 분위 위치.",
        "- `paired_genotype_errors.csv`: 전체 유전자형의 LSTM/우리 모델 시드 평균 RMSE.",
        "- `daily_errors.csv`: 모델·시드·관측일별 MAE와 관측 유전자형 수.",
        "- `all_model_genotype_relative_errors.csv`: 402개 유전자형, 전체 모델의 시드 평균 rRMSE.",
        "- `test_examples.png`: 기존 유전자형 ID 기준 균등 선택 예시, 전체 6개 모델 비교.",
        "", "프로젝트 루트에서:", "", "```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python maize/code/plot_predictions.py", "```", "",
        "그림 생성 전 실험 코드·입력 SHA-256을 확인했고, "
        "6개 모델의 모든 유전자형·시드 RMSE를 저장 예측으로 재계산해 원래 표와 일치함을 확인했다.", ""]
    (figdir / "README.md").write_text("\n".join(notes))
    print(selected[["genotype", "ours_rmse", "lstm_rmse", "rank_low_to_high", "error_group"]].to_string(index=False))
    print(f"Ours lower RMSE: {n_wins}/{len(batch)} ({n_wins/len(batch):.1%})")
    print(f"Saved figures and source tables: {figdir}")


if __name__ == "__main__":
    main()
