"""Express existing three-dataset results relative to each split's mean target."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/relative_errors"
MAIZE = ROOT / "maize/results/chronological_final_seed1_3"
MAIZE_TUNED = ROOT / "maize/results/tuning_20260904"
ORDER = ["Logi-ODE", "Temp-ODE", "RF", "LSTM-NN", "Logi-PINN", "Ours"]
COLORS = ["#a1a9b4", "#71859e", "#d5a14c", "#678a5b", "#936fab", "#216b91"]
DATASETS = {"wheat": "Wheat", "arabidopsis": "Arabidopsis", "maize": "Maize"}
SOURCES = set()


def source(path):
    path = Path(path)
    SOURCES.add(path)
    return path


def csv(path, **kwargs):
    return pd.read_csv(source(path), **kwargs)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, source(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    OUT.mkdir(exist_ok=True)
    source(Path(__file__).resolve())
    denominators, runs = [], []

    def add_denominator(dataset, split, values, n_curves, unit, n_filled=0):
        values = np.asarray(values, dtype=float)
        denominator = float(values.mean())
        if not np.isfinite(values).all() or denominator <= 0:
            raise ValueError(f"Invalid relative-error denominator for {dataset}/{split}")
        denominators.append(dict(dataset=dataset, split=split, mean_target=denominator,
            unit=unit, n_curves=n_curves, n_scored_points=len(values),
            n_scored_points_without_actual_observation=n_filled))

    def add_runs(dataset, frame, rmse_suffix="rmse", mae_suffix="mae", scale=1., is_primary=True):
        for row in frame.to_dict("records"):
            for split in ["train", "val", "test"]:
                runs.append(dict(dataset=dataset, model=row["model"].replace("Latent Neural ODE (ours)", "Ours"),
                    seed=int(row["seed"]), split=split, is_primary=is_primary,
                    rmse=row[f"{split}_{rmse_suffix}"]*scale,
                    mae=row.get(f"{split}_{mae_suffix}", np.nan)*scale))

    # Keep the original wheat scoring mask, including pre-observation fill points.
    wd = load_module("relative_wheat_data", ROOT / "wheat/code/data.py")
    config = json.loads(source(ROOT / "wheat/results/final_config.json").read_text())
    kwargs = dict(data_path=source(ROOT/config["data"]), kinship_path=source(ROOT/config["kinship"]),
        split=config["split"], start_day=config["start_day"], env_cols=config["env_cols"],
        val_year=config["val_year"], test_year=config["test_year"],
        add_gdd=config["add_gdd"], gdd_base=config["gdd_base"], genotypes=config["genotypes"])
    wheat = wd.make_dataset(**kwargs, fill_in_na_at_start=not config["no_fill_na_start"])
    wheat_actual = wd.make_dataset(**kwargs, fill_in_na_at_start=False)
    for split, attr in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
        b, actual = getattr(wheat, attr), getattr(wheat_actual, attr)
        add_denominator("wheat", split, b.y[b.mask > 0], len(b), "m",
                        int(((b.mask > 0) & (actual.mask == 0)).sum()))
    wheat_runs = csv(ROOT / "wheat/results/tuned_latent_ode_seed1_3.csv")
    wheat_runs["model"] = "Ours"
    with np.load(source(ROOT / "wheat/results/test_predictions_seed1_3.npz"), allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["mask"], wheat.test.mask)
        np.testing.assert_allclose(saved["y"], wheat.test.y, atol=1e-12)
        np.testing.assert_array_equal(saved["genotype"], np.asarray(wheat.genotypes)[wheat.test.g_idx])
        for seed, pred in zip(saved["seeds"], saved["pred"]):
            rmse = wd.masked_rmse(pred, saved["y"], saved["mask"])
            idx = wheat_runs.seed.eq(seed)
            np.testing.assert_allclose(rmse, wheat_runs.loc[idx, "test_rmse"].iloc[0], atol=1e-8)
            per_curve_mae = (np.abs(pred-saved["y"])*saved["mask"]).sum(axis=1) / saved["mask"].sum(axis=1)
            wheat_runs.loc[idx, "test_mae"] = float(per_curve_mae.mean())
    add_runs("wheat", wheat_runs)
    for model, filename in [("LSTM-NN", "lstm_nn_onehot_original.csv"),
                             ("Logi-PINN", "logi_pinn_onehot_original.csv")]:
        frame = csv(ROOT / "wheat/results/reference" / filename)
        frame = frame[frame.n_split.eq(0) & frame.random_sees.isin([1, 2, 3])].copy()
        if sorted(frame.random_sees.tolist()) != [1, 2, 3]:
            raise ValueError("Expected exactly three wheat reference seed rows")
        frame = frame.rename(columns={"random_sees": "seed", "train_rMSE": "train_rmse",
            "validation_rMSE": "val_rmse", "test_rMSE": "test_rmse"})
        frame["model"] = model
        add_runs("wheat", frame)

    additional = ROOT / "wheat/results/additional_baselines_seed1_3"
    if (additional / "all_runs.csv").exists():
        validation = json.loads(source(additional / "validation.json").read_text())
        if validation["status"] != "passed":
            raise ValueError("Additional wheat baselines have not passed verification")
        baseline_protocol = json.loads(source(additional / "protocol.json").read_text())
        if (baseline_protocol["train_years"] != list(wheat.train_years)
                or baseline_protocol["val_year"] != wheat.val_year
                or baseline_protocol["test_year"] != wheat.test_year
                or baseline_protocol["fill_in_na_at_start"] != (not config["no_fill_na_start"])):
            raise ValueError("Additional wheat baseline protocol does not match the existing comparison")
        for path, digest in baseline_protocol["source_sha256"].items():
            if hashlib.sha256(source(ROOT/path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Additional wheat baseline input or source changed: {path}")
        add_runs("wheat", csv(additional / "all_runs.csv"))

    # The Arabidopsis export contains the exact replicate means used for scoring.
    arab = csv(ROOT / "arabidopsis/results/predictions_seed1_3.csv", dtype={"genotype": str})
    canonical = arab[arab.model.eq("Logi-ODE") & arab.seed.eq(0) & arab.observed.eq(1)]
    for split, frame in canonical.groupby("split"):
        add_denominator("arabidopsis", split, frame.observed_length_m.to_numpy()*100,
                        len(frame[["genotype", "condition"]].drop_duplicates()), "cm")
    arab_runs = csv(ROOT / "arabidopsis/results/all_runs_seed1_3.csv")
    add_runs("arabidopsis", arab_runs, "rmse_m", "mae_m", scale=100.)

    md = load_module("relative_maize_data", ROOT / "maize/code/data.py")
    protocol = json.loads(source(MAIZE / "protocol.json").read_text())
    maize = md.make_dataset(split=protocol["split"])
    for filename in ["height_observations.csv", "weather_daily.csv", "genotypes.csv", "year_splits.json"]:
        source(ROOT / "maize/data/processed" / filename)
    for split, attr in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
        b = getattr(maize, attr)
        add_denominator("maize", split, b.y[b.mask > 0].astype(float)*maize.y_scale, len(b), "relative UAV height")
    maize_original = csv(MAIZE / "all_runs.csv")
    add_runs("maize", maize_original[~maize_original.model_key.eq("latent")])
    maize_before = maize_original[maize_original.model_key.eq("latent")].copy()
    maize_before["model"] = "Ours (before maize tuning)"
    add_runs("maize", maize_before, is_primary=False)

    tuned_validation = json.loads(source(MAIZE_TUNED / "final/validation.json").read_text())
    tuned_selection = json.loads(source(MAIZE_TUNED / "selected.json").read_text())
    if (tuned_validation["status"] != "passed"
            or tuned_validation["candidate_test_evaluations"] != 0
            or tuned_validation["final_test_evaluations"] != 3):
        raise ValueError("Tuned maize result has not passed its frozen-selection audit")
    maize_tuned = csv(MAIZE_TUNED / "final/all_runs.csv")
    if (sorted(maize_tuned.seed.tolist()) != [1, 2, 3]
            or maize_tuned.candidate.nunique() != 1
            or maize_tuned.candidate.iloc[0] != tuned_selection["candidate"]):
        raise ValueError("Tuned maize runs do not match the frozen selection")
    maize_tuned["model"] = "Ours"
    add_runs("maize", maize_tuned)

    denoms = pd.DataFrame(denominators)
    denoms.to_csv(OUT / "denominators.csv", index=False)
    per_run = pd.DataFrame(runs).merge(denoms, on=["dataset", "split"], validate="many_to_one")
    for metric in ["rmse", "mae"]:
        per_run[f"r{metric}_percent"] = per_run[metric] / per_run.mean_target * 100
    per_run.to_csv(OUT / "per_run_relative_errors.csv", index=False)
    records = []
    for (dataset, model, split), frame in per_run.groupby(["dataset", "model", "split"], sort=False):
        if frame.is_primary.nunique() != 1:
            raise ValueError(f"Mixed primary status for {dataset}/{model}/{split}")
        record = dict(dataset=dataset, model=model, split=split, n_runs=len(frame),
                      is_primary=bool(frame.is_primary.iloc[0]))
        for metric in ["rmse", "mae", "rrmse_percent", "rmae_percent"]:
            record[f"{metric}_mean"] = frame[metric].mean()
            record[f"{metric}_sd"] = frame[metric].std(ddof=1)
        records.append(record)
    # Include prior wheat architecture variants in the full report, not the main model chart.
    variants = csv(ROOT / "wheat/results/paper_results_3seed.csv")
    variants = variants[variants.model.isin(["Micro Latent Neural ODE", "Original Latent Neural ODE"])]
    for _, row in variants.iterrows():
        for split in ["train", "val", "test"]:
            denom = denoms[denoms.dataset.eq("wheat") & denoms.split.eq(split)].mean_target.iloc[0]
            records.append(dict(dataset="wheat", model=row.model, split=split, n_runs=int(row.n_seeds), is_primary=False,
                rmse_mean=row[f"{split}_rmse_mean"], rmse_sd=row[f"{split}_rmse_sd"],
                rrmse_percent_mean=row[f"{split}_rmse_mean"]/denom*100,
                rrmse_percent_sd=row[f"{split}_rmse_sd"]/denom*100))
    summary = pd.DataFrame(records).merge(denoms, on=["dataset", "split"], validate="many_to_one")
    summary.to_csv(OUT / "relative_error_summary.csv", index=False)
    test = summary[summary.split.eq("test") & summary.is_primary]

    def display(row, metric):
        mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
        if pd.isna(mean):
            return "—"
        return f"{mean:.2f}" if pd.isna(sd) else f"{mean:.2f} ± {sd:.2f}"

    def table(metric):
        lines = ["| 모델 | 밀 | 애기장대 | 옥수수 |", "|---|---:|---:|---:|"]
        for model in ORDER:
            values = []
            for dataset in DATASETS:
                row = test[test.model.eq(model) & test.dataset.eq(dataset)]
                values.append("—" if row.empty else display(row.iloc[0], metric))
            lines.append("| " + " | ".join([model]+values) + " |")
        return lines

    def combined_table():
        lines = ["| 모델 | 밀 (m / %) | 애기장대 (cm / %) | 옥수수 (relative UAV unit / %) |",
                 "|---|---:|---:|---:|"]
        for model in ORDER:
            values = []
            for dataset in DATASETS:
                frame = test[test.model.eq(model) & test.dataset.eq(dataset)]
                if frame.empty:
                    values.append("—")
                    continue
                row = frame.iloc[0]
                places = decimals[dataset]
                raw = f"{row.rmse_mean:.{places}f}" if pd.isna(row.rmse_sd) else (
                    f"{row.rmse_mean:.{places}f} ± {row.rmse_sd:.{places}f}")
                value = f"{raw} / {display(row, 'rrmse_percent')}"
                if np.isclose(row.rmse_mean, best[dataset], rtol=0, atol=10**(-(places+2))):
                    value = f"**{value}**"
                values.append(value)
            label = "**Ours**" if model == "Ours" else model
            lines.append("| " + " | ".join([label]+values) + " |")
        return lines

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharex=True, sharey=True, layout="constrained")
    max_x = float((test.rrmse_percent_mean + test.rrmse_percent_sd.fillna(0)).max()) * 1.25
    for ax, (dataset, label) in zip(axes, DATASETS.items()):
        rows = test[test.dataset.eq(dataset)].set_index("model")
        for i, (model, color) in enumerate(zip(ORDER, COLORS)):
            if model not in rows.index:
                ax.text(1, i, "No saved result", va="center", fontsize=9, color="#8c949c")
                continue
            row = rows.loc[model]
            value = row.rrmse_percent_mean
            error = 0 if pd.isna(row.rrmse_percent_sd) else row.rrmse_percent_sd
            ax.barh(i, value, xerr=error, color=color, height=.62, capsize=3)
            ax.text(value+error+.8, i, f"{value:.2f}", va="center", fontsize=9)
        ax.set(title=label, xlabel="Test rRMSE (%)", xlim=(0, max_x), yticks=np.arange(len(ORDER)), yticklabels=ORDER)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Relative errors across three datasets\nrRMSE = existing trajectory-mean RMSE / split-wide mean target × 100", fontsize=13)
    fig.supxlabel("Bars: seed mean ± SD | Original evaluation masks retained; wheat includes initial fill points", fontsize=9)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"test_relative_errors.{ext}", dpi=180)
    plt.close(fig)

    decimals = {"wheat": 5, "arabidopsis": 3, "maize": 2}
    units = {"wheat": "m", "arabidopsis": "cm", "maize": "relative UAV unit"}
    latex_models = {"Ours": r"\textbf{Latent Neural ODE (ours)}"}
    best = test.groupby("dataset").rmse_mean.min().to_dict()

    def latex_value(row, dataset):
        places = decimals[dataset]
        rmse = f"{row.rmse_mean:.{places}f}"
        relative = f"{row.rrmse_percent_mean:.2f}"
        if pd.isna(row.rmse_sd):
            value = rf"{rmse}\,/\,{relative}"
        else:
            value = (rf"{rmse}\pm{row.rmse_sd:.{places}f}\,/\,"
                     rf"{relative}\pm{row.rrmse_percent_sd:.2f}")
        if np.isclose(row.rmse_mean, best[dataset], rtol=0, atol=10**(-(places+2))):
            value = rf"\mathbf{{{value}}}"
        return rf"${value}$"

    latex = [r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Test-set errors across the three plant datasets. Each cell reports RMSE / relative error (\%), where relative error is $100\times$ the trajectory-mean RMSE divided by the split-wide mean observed target. Wheat RMSE is in metres, Arabidopsis RMSE is in centimetres, and maize RMSE is in relative UAV-height units. Bold indicates the lowest mean error within each dataset. Values are mean $\pm$ sample standard deviation over seeds 1--3; deterministic ODE baselines were fitted once. The maize result for our model uses the configuration selected by 2020 validation performance, whereas the maize baseline results retain their original fixed evaluations.}",
        r"\label{tab:cross-dataset-relative-error}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccc}", r"\toprule",
        r"Model & Wheat (m / \%) & Arabidopsis (cm / \%) & Maize (relative UAV unit / \%) \\",
        r"\midrule"]
    for model in ORDER:
        cells = []
        for dataset in DATASETS:
            row = test[test.model.eq(model) & test.dataset.eq(dataset)]
            cells.append("--" if row.empty else latex_value(row.iloc[0], dataset))
        latex.append(" & ".join([latex_models.get(model, model)]+cells) + r" \\")
    latex += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    (OUT / "test_rmse_relative_table.tex").write_text("\n".join(latex))

    lines = ["# 세 데이터셋 상대오차 비교", "",
        "메인 `Ours` 행은 각 데이터셋에서 현재 채택한 결과를 사용한다. 옥수수는 2020년 검증 평균으로 "
        "선택한 3,000-epoch 튜닝 모델로 업데이트했다. 옥수수 baseline은 기존 결과를 유지했으며, "
        "튜닝 전 우리 모델은 아래 전체 분할 표에 `Ours (before maize tuning)`으로 보존했다.", "",
        "## 지표 정의", "",
        "**rRMSE(%) = 기존 RMSE / 해당 평가셋의 평균 평가 타깃 × 100**", "",
        "rMAE도 같은 분모를 사용한다. 기존 RMSE는 각 곡선의 마스크 내 RMSE를 계산한 뒤 곡선을 동일 가중 평균한 값이다. "
        "분모는 반복 평균한 평가 곡선들의 `mask=1` 타깃 전체를 모아 산술평균한 값이다. "
        "각 데이터셋·분할에서 모든 모델과 시드에 같은 분모를 적용하므로 모델 순위는 바뀌지 않는다. "
        "곡선마다 각각 정규화한 평균이나 점별 MAPE와는 다른 지표다.", "",
        "분모는 지표를 보고하기 위한 값이며 모델 학습·체크포인트 선택·예측에는 사용하지 않았다. "
        "학습/검증/시험은 각 분할 자체의 분모를 사용한다. 다른 작물의 물리 단위는 백분율 계산에서 소거된다.",
        "", "## 시험 RMSE / relative error (%)", "",
        "각 셀의 첫 값은 원단위 RMSE, 두 번째 값은 rRMSE(%)다.", ""] + combined_table()
    lines += ["", "![세 데이터셋 상대오차](test_relative_errors.png)", "",
        "평균 ± 표본 표준편차, 시드 1–3. 결정론적 Logi-ODE/Temp-ODE는 1회 적합. "
        "표준편차는 시드 변동이며 데이터셋 일반화에 대한 신뢰구간이 아니다.",
        "밀의 Logi-ODE/Temp-ODE/RF는 동일한 기존 분할·마스크를 사용해 추가로 학습한 결과다. "
        "실행 설정·예측·검증은 `wheat/results/additional_baselines_seed1_3/`에 저장했다. "
        "밀 LSTM-NN/Logi-PINN reference는 원저자 출력 CSV의 첫 3개 시드이며, "
        "원본 RMSE가 0.001 m 단위로 반올림되어 있으므로 상대오차의 원자료 정밀도도 그 제한을 받는다.",
        "", "## 시험 rMAE (%)", ""] + table("rmae_percent")
    lines += ["", "밀 우리 모델의 시험 MAE는 저장된 세 시드 예측에서 계산했다. "
        "추가 학습한 Logi-ODE/Temp-ODE/RF도 MAE를 저장했다. "
        "밀 LSTM-NN/Logi-PINN의 기존 reference 출력에는 MAE가 없어 해당 칸은 `—`로 남겼다.",
        "", "## 평가 분모와 분할", "",
        "| 데이터셋 | 시험 분할 | 곡선 수 | 평가점 수 | 평균 평가 타깃 |", "|---|---|---:|---:|---:|"]
    for dataset, split_name in [("wheat", "2021년"), ("arabidopsis", "개체 9–10, 반복 평균"), ("maize", "2021년")]:
        row = denoms[denoms.dataset.eq(dataset) & denoms.split.eq("test")].iloc[0]
        lines.append(f"| {DATASETS[dataset]} | {split_name} | {row.n_curves} | {row.n_scored_points} | {row.mean_target:.6f} {row.unit} |")
    wheat_den = denoms[denoms.dataset.eq("wheat") & denoms.split.eq("test")].iloc[0]
    lines += ["", f"**밀은 기존 reference 평가를 유지하므로 시험 평가점 {wheat_den.n_scored_points}개 중 "
        f"{wheat_den.n_scored_points_without_actual_observation}개가 최초 관측 전 채움값이다.** "
        "이 점들은 기존 RMSE와 이번 분모에 모두 포함된다. 애기장대와 옥수수는 실제 관측 마스크만 사용한다. "
        "따라서 세 작물의 숫자는 단위를 제거한 기존 벤치마크 결과이며, 동일한 관측 조건의 난이도 비교로 해석하지 않는다. "
        "밀·옥수수는 연도 분할, 애기장대는 같은 유전자형·온도 내 새 개체 분할이다.",
        "", "## 학습·검증·시험 rRMSE (%) — 기존 변형 포함", "",
        "| 데이터셋 | 모델 | 학습 | 검증 | 시험 |", "|---|---|---:|---:|---:|"]
    for (dataset, model), frame in summary.groupby(["dataset", "model"], sort=False):
        values = [display(frame[frame.split.eq(split)].iloc[0], "rrmse_percent") for split in ["train", "val", "test"]]
        lines.append("| " + " | ".join([DATASETS[dataset], model]+values) + " |")
    lines += ["", "## 산출물과 재현", "",
        "- `relative_error_summary.csv`: 전체 모델·분할의 원단위 오차, 상대오차 평균/표준편차, 분모와 평가점 수.",
        "- `per_run_relative_errors.csv`: 저장된 개별 시드 결과의 상대오차. 요약 표만 있는 밀의 Micro/Original 변형은 제외.",
        "- `denominators.csv`: 정규화 분모와 실제 관측 없는 평가점 수.",
        "- `provenance.json`: 사용한 입력 파일과 SHA-256.",
        "- `test_relative_errors.png` / `.pdf`: 시험 rRMSE 비교 그림.",
        "- `test_rmse_relative_table.tex`: 각 셀을 `RMSE / relative error (%)`로 표시하고 "
        "데이터셋별 최저 결과를 굵게 한 LaTeX 표.",
        "  `paper/main.tex`에서는 `\\input{relative_errors/test_rmse_relative_table.tex}`로 삽입할 수 있다. "
        "현재 원고에 이미 포함된 `booktabs`, `graphicx` 패키지를 사용한다.",
        "", "```bash", "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper/relative_errors.py", "```", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    provenance = dict(definition="100 * existing macro trajectory RMSE or MAE / pooled masked target mean within split",
        new_training=False, seeds=[1, 2, 3], process_seeds=[0],
        maize_ours="validation-selected tuned result; pre-tuning result retained as non-primary",
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(SOURCES)})
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print("\n".join(table("rrmse_percent")))
    print("\nTest denominators:\n" + denoms[denoms.split.eq("test")].to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
