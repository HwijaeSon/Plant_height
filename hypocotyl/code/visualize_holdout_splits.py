"""Visualize the frozen observation partitions without fitting any model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/processed/splits"
OUT = ROOT / "reports/holdout_visualization"
TIMES = list(range(0, 73, 12))
COLORS = dict(train="#0072B2", val="#E69F00", test="#D55E00", unused="#AAAAAA")
MARKERS = dict(train="o", val="^", test="s", unused="x")
LABELS = dict(train="학습", val="검증", test="시험", unused="미사용")


def load_partitions():
    paths = [SPLITS / protocol / f"{part}.csv"
             for protocol in ("replicate", "time_holdout")
             for part in ("train", "val", "test")]
    frames = {str(path.relative_to(SPLITS)): pd.read_csv(path) for path in paths}
    base = pd.concat([frames[f"replicate/{part}.csv"] for part in ("train", "val", "test")])
    assert base.observation_id.is_unique
    base = base.rename(columns={"split": "original_split"}).copy()
    data = {}
    for protocol in ("replicate", "time_holdout"):
        selected = pd.concat([frames[f"{protocol}/{part}.csv"] for part in ("train", "val", "test")])
        assert selected.observation_id.is_unique
        mapping = selected.set_index("observation_id").split
        data[protocol] = base.assign(display_split=base.observation_id.map(mapping).fillna("unused"))
    temporal = data["time_holdout"]
    assert temporal.loc[temporal.elapsed_hours.isin([36, 60]), "display_split"].eq("test").all()
    assert not temporal.loc[~temporal.elapsed_hours.isin([36, 60]), "display_split"].eq("test").any()
    assert temporal.display_split.eq("unused").sum() == 315
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return data, hashes


def diagram(ax, frame, protocol):
    ax.set_xlim(-1.9, 6.6)
    ax.set_ylim(3.05, -.9)
    ax.axis("off")
    ax.text(-1.85, -.6, "원본 행 기준 반복 그룹", fontsize=13, va="center")
    for j, time in enumerate(TIMES):
        ax.text(j, -.6, f"{time} h", ha="center", va="center", fontsize=15)
    for i, original in enumerate(("train", "val", "test")):
        group = frame.loc[frame.original_split.eq(original)]
        ax.text(-.64, i, f"기존 {LABELS[original]} 그룹", ha="right", va="center", fontsize=14)
        for j, time in enumerate(TIMES):
            cell = group.loc[group.elapsed_hours.eq(time)]
            roles = cell.display_split.unique()
            assert len(roles) == 1
            role = roles[0]
            ax.add_patch(Rectangle((j-.43, i-.34), .86, .68,
                                   facecolor=COLORS[role], alpha=.14, edgecolor="none"))
            ax.scatter(j-.23, i, s=48, marker=MARKERS[role], color=COLORS[role], linewidths=1.7)
            ax.text(j+.06, i, str(len(cell)), ha="center", va="center", fontsize=16)
    if protocol == "time_holdout":
        for j in (3, 5):
            ax.add_patch(Rectangle((j-.46, -.39), .92, 2.78, fill=False,
                                   edgecolor=COLORS["test"], linewidth=1.3))
    ax.text(-1.85, 2.78, "칸의 숫자: 5개 유전형 × 2개 빛 조건을 합친 실제 관측 수", fontsize=13)


def observed_example(ax, frame, condition):
    case = frame.loc[frame.genotype.eq("Col-0") & frame.sheet.eq(condition)].copy()
    if condition == "12L12D":
        for start in (12, 36, 60):
            ax.axvspan(start, start+12, color="#000000", alpha=.045, zorder=0)
        ax.text(.98, .95, "회색 음영: 소등", transform=ax.transAxes,
                ha="right", va="top", fontsize=12, color="#555555")
    # Stable offsets identify display jitter only, never a physical time shift.
    # Raw measurements are deliberately not connected as longitudinal plants.
    case["jitter"] = case.observation_id.map(
        lambda value: (int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) / (2**32 - 1) - .5) * 3.6)
    for role in ("unused", "train", "val", "test"):
        part = case.loc[case.display_split.eq(role)]
        if part.empty:
            continue
        ax.scatter(part.elapsed_hours+part.jitter, part.length, s=20,
                   marker=MARKERS[role], color=COLORS[role], alpha=.36 if role != "unused" else .25,
                   linewidths=.6, zorder=2)
        if role != "unused":
            means = part.groupby("elapsed_hours").length.mean()
            ax.scatter(means.index, means.values, s=90, marker=MARKERS[role], color=COLORS[role],
                       edgecolor="#FFFFFF", linewidths=.9, zorder=4)
    ax.set_title("Col-0 · " + ("12 h light / 12 h dark" if condition == "12L12D" else "Continuous red light"),
                 fontsize=15, pad=10, loc="left")
    ax.set_xlim(-4, 76)
    ax.set_ylim(0, 9.3)
    ax.set_xticks(TIMES)
    ax.set_yticks([0, 3, 6, 9])
    ax.set_xlabel("경과 시간 (h)", fontsize=14)
    ax.set_ylabel("배축 길이 (mm)", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", color="#E5E5E5", lw=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#888888")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 14})
    data, hashes = load_partitions()
    audit = dict(input_sha256=hashes, example_genotype="Col-0", units="mm", protocols={})
    titles = dict(replicate="반복 관측 holdout", time_holdout="36·60시간 holdout")
    subtitles = dict(replicate="같은 시간의 반복 관측을 학습·검증·시험으로 분할",
                     time_holdout="36·60시간의 관측 전체를 시험에만 사용")
    for protocol, frame in data.items():
        counts = frame.display_split.value_counts().to_dict()
        fig = plt.figure(figsize=(12, 9), facecolor="white")
        fig.text(.075, .96, titles[protocol], fontsize=22, va="top")
        fig.text(.075, .91, subtitles[protocol], fontsize=15, va="top")
        roles = ("train", "val", "test") if protocol == "replicate" else ("train", "val", "test", "unused")
        handles = [Line2D([], [], marker=MARKERS[r], color=COLORS[r], linestyle="none", markersize=8,
                          label=f"{LABELS[r]} {counts.get(r, 0):,}개") for r in roles]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.068, .88),
                   ncol=len(roles), frameon=False, fontsize=14, columnspacing=1.7, handletextpad=.5)
        gs = fig.add_gridspec(2, 2, left=.08, right=.97, bottom=.13, top=.81,
                             height_ratios=[1, 1.1], hspace=.37, wspace=.26)
        diagram(fig.add_subplot(gs[0, :]), frame, protocol)
        for col, condition in enumerate(("12L12D", "cR")):
            observed_example(fig.add_subplot(gs[1, col]), frame, condition)
        fig.text(.075, .055, "작은 점: 개별 관측  ·  큰 기호: 해당 분할의 시점별 평균  ·  좌우 흔들림: 겹침 방지", fontsize=12)
        fig.text(.075, .025, "원본 행은 개체 ID로 확인되지 않았으며, 개체별 궤적을 연결한 그림이 아닙니다.", fontsize=12)
        for suffix in ("png", "pdf"):
            fig.savefig(OUT / f"{protocol}_split_visualization.{suffix}", dpi=160)
        plt.close(fig)
        audit["protocols"][protocol] = {
            "counts": counts,
            "counts_by_time": {r: frame.loc[frame.display_split.eq(r)].groupby("elapsed_hours").size()
                               .reindex(TIMES, fill_value=0).tolist() for r in roles},
            "times_hours": TIMES,
            "no_models_fitted_or_changed": True,
        }
    (OUT / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit["protocols"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
