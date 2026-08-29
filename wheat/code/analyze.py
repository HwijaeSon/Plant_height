"""
analyze.py -- 학습 결과 진단.

train.py 가 저장한 `*_test_predictions.npz` 를 읽어 다음을 보고한다.
  1. 시드별 test RMSE vs 시드 앙상블(예측 평균) RMSE
  2. 유전자형별 RMSE (논문 Fig. 4 대응)
  3. 생육 단계별 오차 분해 — 오차가 급신장 구간에 몰렸는지, 정체기에 몰렸는지
  4. 예측 곡선의 단조성 위반 정도

사용:
  python analyze.py results/latent_ode_split0_test_predictions.npz
  python analyze.py results/a.npz results/b.npz --labels 기본 GDD추가
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def masked_rmse(pred, y, mask):
    err = ((pred - y) ** 2) * mask
    denom = np.clip(mask.sum(axis=-1), 1e-8, None)
    return np.sqrt(err.sum(axis=-1) / denom)


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["pred"], d["y"], d["mask"], d["genotype"]


def phase_breakdown(pred, y, mask, n_bins=3):
    """관측된 높이의 분위수로 생육 단계를 나눠 오차를 분해한다.

    구간 1 = 신장 개시 전/초기, 구간 2 = 급신장, 구간 3 = 최대 높이 도달 후.
    """
    obs = mask > 0
    rel = np.where(obs, y / np.clip(y.max(axis=-1, keepdims=True), 1e-6, None), np.nan)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = obs & (rel >= lo) & (rel <= hi if i == n_bins - 1 else rel < hi)
        sel_b = np.broadcast_to(sel, pred.shape)
        if sel_b.sum() == 0:
            continue
        e = (pred - y[None]) ** 2
        rows.append({
            "구간": f"상대높이 {lo:.2f}–{hi:.2f}",
            "관측점 수": int(sel.sum()),
            "RMSE": float(np.sqrt(e[sel_b].mean())),
            "평균 편향": float((pred - y[None])[sel_b].mean()),
        })
    return pd.DataFrame(rows)


def report(path, label=None):
    pred, y, mask, geno = load(path)   # pred [S, N, T]
    label = label or path
    print(f"\n{'=' * 68}\n{label}\n{'=' * 68}")

    per_seed = np.stack([masked_rmse(pred[s], y, mask) for s in range(pred.shape[0])])
    seed_mean = per_seed.mean(axis=1)
    ens = masked_rmse(pred.mean(axis=0), y, mask)

    print(f"시드 개별 test RMSE : {seed_mean.mean():.4f} ± {seed_mean.std(ddof=1):.4f}")
    print(f"  (시드별: {', '.join(f'{v:.4f}' for v in seed_mean)})")
    print(f"시드 앙상블 test RMSE: {ens.mean():.4f}   "
          f"(개별 평균 대비 {100 * (ens.mean() / seed_mean.mean() - 1):+.1f}%)")

    df = (pd.DataFrame({"genotype": geno,
                        "rmse_mean": per_seed.mean(axis=0),
                        "rmse_sd": per_seed.std(axis=0, ddof=1),
                        "rmse_ens": ens})
          .sort_values("rmse_ens"))
    print("\n[유전자형별 RMSE] 좋은 5개 / 나쁜 5개")
    print(pd.concat([df.head(5), df.tail(5)]).to_string(index=False,
                                                        float_format=lambda v: f"{v:.4f}"))
    print(f"\n유전자형 간 RMSE 범위: {df['rmse_ens'].min():.4f} ~ {df['rmse_ens'].max():.4f} "
          f"(중앙값 {df['rmse_ens'].median():.4f})")

    print("\n[생육 단계별 오차 분해]  편향 > 0 이면 과대예측")
    print(phase_breakdown(pred, y, mask).to_string(index=False,
                                                   float_format=lambda v: f"{v:.4f}"))

    d = np.diff(pred, axis=-1)
    print(f"\n단조성 위반: 전체 시간 스텝의 {100 * (d < 0).mean():.1f}% 에서 예측이 감소, "
          f"누적 감소량 평균 {np.clip(-d, 0, None).sum(axis=-1).mean():.4f} m")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+")
    p.add_argument("--labels", nargs="*", default=None)
    a = p.parse_args()
    labels = a.labels or [None] * len(a.npz)
    for path, label in zip(a.npz, labels):
        report(path, label)


if __name__ == "__main__":
    main()