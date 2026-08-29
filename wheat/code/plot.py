"""
plot.py -- train.py 가 저장한 결과로 그림을 만든다.

논문 Plot_analysis_result.py 가 만드는 그림들을 이 코드의 출력 형식
(*_test_predictions.npz, *_history.csv, *_per_genotype.csv, *_summary.csv)에
맞춰 다시 구현했다.

만들어지는 그림
  curves    예측 곡선 ± 시드 표준편차 vs 실측점, 유전자형별       (논문 Fig. 3 / 6 / 8)
  genotype  유전자형별 test RMSE, kinship 유사도 순 정렬          (논문 Fig. 4)
  history   학습 곡선 (train/val/test), best epoch 표시           (발산 진단용)
  height    유전자형 키 vs RMSE 산점도                            (이 코드에서 찾은 결과)
  compare   여러 실행의 test RMSE 비교                            (논문 Fig. 7)

사용
  python plot.py curves   results/trate_mono10_split0_test_predictions.npz
  python plot.py genotype results/trate_mono10_split0_test_predictions.npz
  python plot.py history  results/trate_mono10_split0_history.csv
  python plot.py height   results/trate_mono10_split0_test_predictions.npz
  python plot.py compare  results/A_summary.csv results/B_summary.csv --labels A B
  python plot.py all      results/trate_mono10_split0          # 접두사만 주면 전부
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_kinship, make_dataset

OUT_DIR = Path("figures")


# --------------------------------------------------------------------------- #
# 공통
# --------------------------------------------------------------------------- #
def setup_style():
    """한글이 깨지지 않도록 폰트를 잡는다.

    Windows 는 'Malgun Gothic' 이 기본으로 있어 그대로 잡힌다. 리눅스에서는
    설치된 CJK 폰트 파일을 직접 등록한다. 아무것도 없으면 경고만 내고 진행한다
    (그림은 나오지만 한글이 네모로 보인다).
    """
    from matplotlib import font_manager

    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic",
                 "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    else:
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
            "/Library/Fonts/AppleGothic.ttf",
            "C:/Windows/Fonts/malgun.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                font_manager.fontManager.addfont(path)
                plt.rcParams["font.family"] = \
                    font_manager.FontProperties(fname=path).get_name()
                break
        else:
            print("  [알림] 한글 폰트를 찾지 못했습니다. 한글이 네모로 보일 수 있습니다.")

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"


def save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  저장: {path}")
    return path


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["pred"], d["y"], d["mask"], d["genotype"]


def masked_rmse(pred, y, mask):
    err = ((pred - y) ** 2) * mask
    return np.sqrt(err.sum(-1) / np.clip(mask.sum(-1), 1e-8, None))


def order_by_kinship(genotypes, kinship_path=None, reference=106):
    """기준 유전자형과의 kinship 유사도 내림차순으로 정렬한다.

    논문 order_genotype_based_on_their_similarity 와 같은 방식. 값이 클수록 가깝다.
    반환: (정렬된 유전자형 목록, 기준과의 유사도)
    """
    try:
        all_g, kin = load_kinship(kinship_path)
    except Exception as e:
        print(f"  [알림] kinship 정렬 생략 ({e}). 유전자형 번호순으로 그린다.")
        return sorted(genotypes), None
    if reference not in all_g:
        reference = all_g[0]
    col = kin[:, all_g.index(reference)]
    sim = {g: col[i] for i, g in enumerate(all_g)}
    present = [g for g in genotypes if g in sim]
    order = sorted(present, key=lambda g: -sim[g])
    return order, {g: sim[g] for g in order}


# --------------------------------------------------------------------------- #
# 1. 예측 곡선  (논문 Fig. 3 / 6 / 8)
# --------------------------------------------------------------------------- #
def plot_curves(npz_path, kinship_path=None, ncols=5, tag=None):
    pred, y, mask, geno = load_npz(npz_path)     # pred [S, N, T]
    tag = tag or Path(npz_path).stem.replace("_test_predictions", "")
    order, _ = order_by_kinship(list(geno), kinship_path)
    idx_of = {g: i for i, g in enumerate(geno)}

    n = len(order)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.7 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    t = np.arange(pred.shape[-1])
    rmse_all = masked_rmse(pred.mean(0), y, mask)

    for ax, g in zip(axes, order):
        i = idx_of[g]
        m = mask[i] > 0
        mu, sd = pred[:, i].mean(0), pred[:, i].std(0, ddof=1)
        ax.fill_between(t, mu - sd, mu + sd, color="#C2410C", alpha=0.22, lw=0)
        ax.plot(t, mu, color="#C2410C", lw=1.6, label="예측 (시드 평균)")
        ax.scatter(t[m], y[i][m], s=7, color="#1F2937", alpha=0.65,
                   zorder=3, label="실측")
        ax.set_title(f"유전자형 {g}   RMSE {rmse_all[i]:.4f}", fontsize=10)
        ax.set_ylim(-0.05, 1.35)

    for ax in axes[n:]:
        ax.axis("off")
    # 각 열에서 가장 아래쪽에 실제로 그려진 칸에만 x 라벨을 단다
    for c in range(ncols):
        rows_in_col = [r for r in range(nrows) if r * ncols + c < n]
        if rows_in_col:
            axes[max(rows_in_col) * ncols + c].set_xlabel("일 (파종 후 115일부터)")
    for r in range(nrows):
        if r * ncols < n:
            axes[r * ncols].set_ylabel("초장 (m)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"테스트 연도 예측 곡선 — {tag}   (음영 = 시드 간 ±1 SD)",
                 fontsize=13, y=1.002)
    fig.tight_layout()
    return save(fig, f"{tag}_curves")


# --------------------------------------------------------------------------- #
# 2. 유전자형별 RMSE  (논문 Fig. 4)
# --------------------------------------------------------------------------- #
def plot_genotype(npz_path, kinship_path=None, tag=None):
    pred, y, mask, geno = load_npz(npz_path)
    tag = tag or Path(npz_path).stem.replace("_test_predictions", "")
    per_seed = np.stack([masked_rmse(pred[s], y, mask) for s in range(pred.shape[0])])
    order, sim = order_by_kinship(list(geno), kinship_path)
    idx_of = {g: i for i, g in enumerate(geno)}
    pos = [idx_of[g] for g in order]

    mu, sd = per_seed.mean(0)[pos], per_seed.std(0, ddof=1)[pos]
    ens = masked_rmse(pred.mean(0), y, mask)[pos]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(order))
    ax.bar(x, mu, yerr=sd, capsize=3, color="#93B8D8", edgecolor="#1F4E79",
           lw=0.8, label="시드 평균 ± SD")
    ax.plot(x, ens, "o", color="#C2410C", ms=5, label="시드 앙상블")
    ax.axhline(mu.mean(), color="#6B7280", ls="--", lw=1,
               label=f"전체 평균 {mu.mean():.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45)
    ax.set_xlabel("유전자형" + ("  (106번과의 kinship 유사도 순)" if sim else ""))
    ax.set_ylabel("test RMSE")
    ax.set_title(f"유전자형별 test RMSE — {tag}", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save(fig, f"{tag}_per_genotype")


# --------------------------------------------------------------------------- #
# 3. 학습 곡선
# --------------------------------------------------------------------------- #
def plot_history(hist_path, summary_path=None, tag=None):
    h = pd.read_csv(hist_path)
    tag = tag or Path(hist_path).stem.replace("_history", "")
    best = {}
    if summary_path and os.path.exists(summary_path):
        s = pd.read_csv(summary_path)
        best = dict(zip(s.seed, s.best_epoch))

    seeds = sorted(h.seed.unique()) if "seed" in h else [None]
    ncols = min(len(seeds), 3)
    nrows = int(np.ceil(len(seeds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()

    for ax, sd in zip(axes, seeds):
        d = h[h.seed == sd] if sd is not None else h
        for col, c, lab in [("train_rmse", "#1F4E79", "train"),
                            ("val_rmse", "#0F6E56", "val"),
                            ("test_rmse", "#C2410C", "test")]:
            if col in d and d[col].notna().any():
                ax.plot(d.epoch, d[col], color=c, lw=1.4, label=lab)
        if "val_rmse_smooth" in d:
            ax.plot(d.epoch, d.val_rmse_smooth, color="#0F6E56", lw=1.0,
                    ls=":", label="val (이동평균)")
        if sd in best:
            ax.axvline(best[sd], color="#6B7280", ls="--", lw=1.2)
            ax.annotate(f"best {best[sd]}", (best[sd], ax.get_ylim()[1]),
                        fontsize=8, ha="left", va="top", color="#6B7280")
        ax.set_title(f"seed {sd}", fontsize=10)
        ax.set_ylim(0, min(0.15, max(0.06, h.filter(like="rmse").max().max() * 1.05)))
    for ax in axes[len(seeds):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    # sharex 때문에 위쪽 칸의 눈금이 숨겨지므로, 각 열의 가장 아래 칸에 되살린다
    for c in range(ncols):
        rows_in_col = [r for r in range(nrows) if r * ncols + c < len(seeds)]
        if rows_in_col:
            ax = axes[max(rows_in_col) * ncols + c]
            ax.set_xlabel("epoch")
            ax.tick_params(labelbottom=True)
    for r in range(nrows):
        if r * ncols < len(seeds):
            axes[r * ncols].set_ylabel("RMSE")
    fig.suptitle(f"학습 곡선 — {tag}   (점선 = 선택된 체크포인트)", fontsize=13, y=1.002)
    fig.tight_layout()
    return save(fig, f"{tag}_history")


# --------------------------------------------------------------------------- #
# 4. 유전자형 키 vs RMSE
# --------------------------------------------------------------------------- #
def plot_height(npz_path, data_path=None, kinship_path=None, split=0, tag=None):
    """오차가 유전자형의 키와 상관되는지 본다 (이 코드에서 발견한 패턴)."""
    from scipy import stats

    pred, y, mask, geno = load_npz(npz_path)
    tag = tag or Path(npz_path).stem.replace("_test_predictions", "")
    rmse = masked_rmse(pred.mean(0), y, mask)

    ds = make_dataset(data_path, kinship_path, split=split)
    tr = ds.train_eval
    hs = {}
    for i in range(len(tr)):
        g = ds.genotypes[tr.g_idx[i]]
        m = tr.mask[i] > 0
        hs.setdefault(g, []).append(tr.y[i][m].max())
    height = np.array([np.mean(hs[g]) for g in geno])

    r, p = stats.pearsonr(height, rmse)
    fig, ax = plt.subplots(figsize=(6.6, 5))
    ax.scatter(height, rmse, s=55, color="#93B8D8", edgecolor="#1F4E79", zorder=3)
    for hgt, e, g in zip(height, rmse, geno):
        ax.annotate(str(g), (hgt, e), fontsize=8, xytext=(4, 3),
                    textcoords="offset points", color="#1F4E79")
    b = np.polyfit(height, rmse, 1)
    xs = np.linspace(height.min(), height.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color="#C2410C", lw=1.5, ls="--")
    ax.set_xlabel("학습 연도 평균 최대 초장 (m)")
    ax.set_ylabel("test RMSE (시드 앙상블)")
    ax.set_title(f"유전자형 키 vs 예측 오차 — {tag}\nPearson r = {r:+.3f}  (p = {p:.4f})",
                 fontsize=12)
    fig.tight_layout()
    return save(fig, f"{tag}_height_vs_error")


# --------------------------------------------------------------------------- #
# 5. 실행 간 비교  (논문 Fig. 7)
# --------------------------------------------------------------------------- #
def plot_compare(summary_paths, labels=None, paper=True):
    labels = labels or [Path(p).stem.replace("_summary", "") for p in summary_paths]
    rows = []
    for path, lab in zip(summary_paths, labels):
        d = pd.read_csv(path)
        for _, r in d.iterrows():
            for split, col in [("train", "train_rmse"), ("val", "val_rmse"),
                               ("test", "test_rmse")]:
                rows.append({"설정": lab, "구분": split, "rmse": r[col]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 4, 5))
    colors = {"train": "#1F4E79", "val": "#0F6E56", "test": "#C2410C"}
    width = 0.26
    for k, (split, c) in enumerate(colors.items()):
        sub = df[df["구분"] == split]
        stat = sub.groupby("설정").rmse.agg(["mean", "std"]).reindex(labels)
        x = np.arange(len(labels)) + (k - 1) * width
        ax.bar(x, stat["mean"], width, yerr=stat["std"], capsize=3,
               color=c, alpha=0.85, label=split)
        for xi, lab in zip(x, labels):
            pts = sub[sub["설정"] == lab].rmse.values
            ax.scatter(np.full_like(pts, xi) + np.random.uniform(-.05, .05, len(pts)),
                       pts, s=12, color="white", edgecolor=c, lw=0.7, zorder=3)

    if paper:
        ax.axhline(0.058, color="#6B7280", ls="--", lw=1.2, zorder=1)
        ax.set_ylim(0, max(ax.get_ylim()[1], 0.058 * 1.22))
        ax.annotate("논문 LSTM-NN test 0.058 (Table 5)", (-0.42, 0.0585),
                    fontsize=8.5, va="bottom", ha="left", color="#6B7280")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("설정 간 비교 (막대 = 시드 평균 ± SD, 점 = 개별 시드)", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save(fig, "compare_runs")


# --------------------------------------------------------------------------- #
def main():
    setup_style()
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["curves", "genotype", "history", "height",
                                    "compare", "all"])
    p.add_argument("paths", nargs="+")
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--kinship", default=None)
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--reference", type=int, default=106,
                   help="kinship 정렬의 기준 유전자형")
    a = p.parse_args()

    if a.what == "compare":
        plot_compare(a.paths, a.labels)
        return

    if a.what == "all":
        base = a.paths[0].rstrip("_")
        plot_curves(f"{base}_test_predictions.npz", a.kinship)
        plot_genotype(f"{base}_test_predictions.npz", a.kinship)
        if os.path.exists(f"{base}_history.csv"):
            plot_history(f"{base}_history.csv", f"{base}_summary.csv")
        try:
            plot_height(f"{base}_test_predictions.npz", a.data, a.kinship, a.split)
        except Exception as e:
            print(f"  [알림] height 그림 생략: {e}")
        return

    fn = {"curves": lambda q: plot_curves(q, a.kinship),
          "genotype": lambda q: plot_genotype(q, a.kinship),
          "history": lambda q: plot_history(q, q.replace("_history", "_summary")),
          "height": lambda q: plot_height(q, a.data, a.kinship, a.split)}[a.what]
    for q in a.paths:
        fn(q)


if __name__ == "__main__":
    main()
