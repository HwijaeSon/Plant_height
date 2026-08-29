"""
train.py -- Latent ODE 식물 높이 예측 모델 학습/평가.

논문(Shao et al., 2026)의 평가 프로토콜을 그대로 따른다.
  - 연도 분할 (학습 2년 / 검증 1년 / 테스트 1년)
  - 학습은 replicate 단위, 평가는 (genotype, year) 평균 곡선
  - masked RMSE (Eq. 3), 검증 RMSE 최소 시점의 체크포인트 선택
  - 랜덤 시드 5개 반복 후 평균 ± 표준편차 보고

사용 예:
  python train.py --data ../align_height_env_same_length.csv \
                  --kinship ../kinship_matrix_astle.csv \
                  --split 0 --seeds 1 2 3 4 5 --epochs 2000

  # z0 를 유전자형만으로 만드는 ablation
  python train.py ... --init_mode genotype
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from data import Dataset, make_dataset, DEFAULT_DATA_FILE, DEFAULT_KINSHIP_FILE
from model import (
    LatentODEHeightModel,
    count_parameters,
    kl_divergence,
    masked_rmse_loss,
    physics_losses,
)


# --------------------------------------------------------------------------- #
def to_tensors(seq, device):
    return {
        "y": torch.as_tensor(seq.y, dtype=torch.float32, device=device),
        "mask": torch.as_tensor(seq.mask, dtype=torch.float32, device=device),
        "env": torch.as_tensor(seq.env, dtype=torch.float32, device=device),
        "s": torch.as_tensor(seq.s, dtype=torch.float32, device=device),
        "ds": torch.as_tensor(seq.ds, dtype=torch.float32, device=device),
        "g_idx": torch.as_tensor(seq.g_idx, dtype=torch.long, device=device),
    }


@torch.no_grad()
def evaluate(model, batch, encoder_days, per_sample=False):
    """관측된 전 구간(170일)에 대해 masked RMSE 를 계산한다 (논문 Eq. 3)."""
    model.eval()
    out = model(batch["g_idx"], batch["env"], batch["s"], batch["ds"], encoder_days)
    pred = out["pred"]
    m = batch["mask"]
    err = ((pred - batch["y"]) ** 2) * m
    denom = m.sum(dim=-1).clamp_min(1e-8)
    rmse = torch.sqrt(err.sum(dim=-1) / denom)
    if per_sample:
        return rmse.cpu().numpy(), pred.cpu().numpy()
    return float(rmse.mean())


# --------------------------------------------------------------------------- #
def train_one_seed(ds: Dataset, args, seed: int, device, verbose: bool = True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = to_tensors(ds.train, device)
    val = to_tensors(ds.val, device)
    test = to_tensors(ds.test, device)
    train_eval = to_tensors(ds.train_eval, device)

    n_t = ds.train.n_t
    model = LatentODEHeightModel(
        n_genotypes=len(ds.genotypes),
        env_dim=ds.train.env.shape[-1],
        latent_dim=args.latent_dim,
        g_embed_dim=args.g_embed_dim,
        ode_hidden=args.ode_hidden,
        ode_layers=args.ode_layers,
        dec_hidden=args.dec_hidden,
        enc_hidden=args.enc_hidden,
        init_mode=args.init_mode,
        time_mode=args.time_mode,
        genetic_encoding=args.genetic_encoding,
        kinship=ds.kinship,
        variational=args.variational,
        n_substeps=args.n_substeps,
        days_per_tau=float(n_t - 1),
        use_physics=max(args.weight_physic, args.weight_ymax, args.weight_r) > 0,
    ).to(device)

    # 순환층에만 직교 초기화 (논문과 동일한 관행). ODEFunc 마지막 층과
    # ODEParameterHead 의 의도적인 소규모 초기화는 그대로 유지한다.
    for name, p in model.named_parameters():
        if "lstm" in name and "weight" in name and p.dim() >= 2:
            torch.nn.init.orthogonal_(p)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    weights = {"physic": args.weight_physic, "r": args.weight_r,
               "ymax": args.weight_ymax, "mono": args.weight_mono}
    need_physics = model.use_physics or args.weight_mono > 0
    n_train = len(ds.train)
    batch_size = args.batch_size if args.batch_size > 0 else n_train

    best = {"val": float("inf"), "epoch": -1, "state": None}
    history = []
    recent_val = []          # 검증 곡선 평활화용 이동창
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            b = {k: v[idx] for k, v in train.items()}

            out = model(b["g_idx"], b["env"], b["s"], b["ds"], args.encoder_days)
            loss = masked_rmse_loss(out["pred"], b["y"], b["mask"])

            if need_physics:
                for v in physics_losses(model, out, b["env"], weights).values():
                    loss = loss + v
            if model.variational and args.kl_weight > 0:
                loss = loss + args.kl_weight * kl_divergence(out["mu"], out["logvar"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            epoch_loss += loss.detach().item() * len(idx)
        sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            v = evaluate(model, val, args.encoder_days)
            recent_val.append(v)
            if len(recent_val) > args.val_smooth:
                recent_val.pop(0)
            # 검증 곡선의 일시적 저점(노이즈)에 체크포인트가 끌려가지 않도록
            # 최근 val_smooth 회 평가의 이동평균으로 모델을 선택한다.
            v_smooth = float(np.mean(recent_val))

            row = {"epoch": epoch, "train_loss": epoch_loss / n_train,
                   "val_rmse": v, "val_rmse_smooth": v_smooth}
            # 학습/테스트 RMSE 는 기록용으로만 계산한다. 체크포인트 선택에는
            # 오직 val_rmse_smooth 만 쓴다 (아래 조건문 참고).
            if args.track_all:
                row["train_rmse"] = evaluate(model, train_eval, args.encoder_days)
                row["test_rmse"] = evaluate(model, test, args.encoder_days)
            history.append(row)

            if epoch >= args.min_epoch and len(recent_val) == args.val_smooth \
                    and v_smooth < best["val"]:
                best = {
                    "val": v_smooth,
                    "epoch": epoch,
                    "state": {k: t.detach().clone() for k, t in model.state_dict().items()},
                }
            if verbose and epoch % (args.eval_every * args.log_every) == 0:
                msg = (f"  seed {seed} | ep {epoch:5d} | loss {epoch_loss / n_train:.4f}")
                if args.track_all:
                    msg += (f" | train {row['train_rmse']:.4f}"
                            f" | val {v:.4f} | test {row['test_rmse']:.4f}")
                else:
                    msg += f" | val {v:.4f}"
                msg += f" | best(val smooth) {best['val']:.4f} @{best['epoch']}"
                print(msg)

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    rmse_test, pred_test = evaluate(model, test, args.encoder_days, per_sample=True)
    rmse_val, pred_val = evaluate(model, val, args.encoder_days, per_sample=True)
    rmse_train, _ = evaluate(model, train_eval, args.encoder_days, per_sample=True)

    result = {
        "seed": seed,
        "best_epoch": best["epoch"],
        "train_rmse": float(rmse_train.mean()),
        "val_rmse": float(rmse_val.mean()),
        "test_rmse": float(rmse_test.mean()),
        "n_params": count_parameters(model),
        "seconds": time.time() - t0,
    }
    per_genotype = pd.DataFrame(
        {
            "seed": seed,
            "genotype": [ds.genotypes[i] for i in ds.test.g_idx],
            "test_rmse": rmse_test,
            "val_rmse": rmse_val,
        }
    )
    return model, result, per_genotype, pred_test, pd.DataFrame(history)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Latent ODE for plant height prediction")
    # 데이터
    p.add_argument("--data", default=None,
                   help=f"기본값: {DEFAULT_DATA_FILE}. 파일명만 주면 현재 디렉터리와 "
                        f"환경변수 PLANT_DATA_DIR 의 디렉터리들에서 찾는다.")
    p.add_argument("--kinship", default=None,
                   help=f"기본값: {DEFAULT_KINSHIP_FILE}")
    p.add_argument("--split", type=int, default=0, help="0..5, 0 = train(2018,2019)")
    p.add_argument("--start_day", type=int, default=115)
    p.add_argument("--env_cols", nargs="+", default=["Air_temperature_2_m"])
    p.add_argument("--no_fill_na_start", action="store_true")
    p.add_argument("--add_gdd", action="store_true",
                   help="누적 생육도일(GDD) 채널을 환경 입력에 추가")
    p.add_argument("--gdd_base", type=float, default=0.0)
    p.add_argument("--genotypes", type=int, nargs="*", default=None,
                   help="사용할 유전자형 목록 (예: --genotypes 106). "
                        "하나만 주면 논문 Table 4 의 단일 유전자형 설정이 된다. "
                        "비우면 19개 전부 (Table 5 설정).")
    p.add_argument("--val_year", type=int, default=None)
    p.add_argument("--test_year", type=int, default=None)
    # 모델
    p.add_argument("--init_mode", choices=["env_encoder", "genotype"], default="env_encoder")
    p.add_argument("--encoder_days", type=int, default=0,
                   help="인코더가 읽는 기온 구간의 일수. 0 = 전 구간(기본). "
                        "양수로 주면 앞 N일만 읽는 인과적 설정이 된다.")
    p.add_argument("--time_mode",
                   choices=["calendar", "thermal_feature", "thermal_rate"],
                   default="calendar",
                   help="발육을 끄는 시간축. calendar=달력일(기본), "
                        "thermal_feature=시간 특징을 열시간으로 교체, "
                        "thermal_rate=열시간 완전 재매개화")
    p.add_argument("--genetic_encoding", choices=["one_hot", "kinship"], default="one_hot")
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--g_embed_dim", type=int, default=4)
    p.add_argument("--ode_hidden", type=int, default=32)
    p.add_argument("--ode_layers", type=int, default=2)
    p.add_argument("--dec_hidden", type=int, default=32)
    p.add_argument("--enc_hidden", type=int, default=16)
    p.add_argument("--n_substeps", type=int, default=1)
    p.add_argument("--variational", action="store_true")
    # 손실 가중치
    p.add_argument("--weight_physic", type=float, default=2.0, help="0 이면 순수 latent ODE")
    p.add_argument("--weight_r", type=float, default=0.0)
    p.add_argument("--weight_ymax", type=float, default=0.1)
    p.add_argument("--kl_weight", type=float, default=0.0)
    p.add_argument("--weight_mono", type=float, default=0.0,
                   help="단조 증가 벌점 mean(relu(-dy/dt)). dy/dt~0.02 스케일이라 "
                        "1 ~ 100 범위를 훑어보길 권함.")
    # 최적화
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--min_epoch", type=int, default=600)
    p.add_argument("--val_smooth", type=int, default=10,
                   help="체크포인트 선택 시 검증 RMSE 이동평균 창 크기 (평가 횟수 기준)")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--log_every", type=int, default=10,
                   help="eval_every 의 몇 배마다 한 줄씩 출력할지 (기본 10 -> 100 에폭마다)")
    p.add_argument("--track_all", dest="track_all", action="store_true", default=True,
                   help="학습 중 train/test RMSE 도 함께 계산해 출력 (기본 켜짐)")
    p.add_argument("--no_track_all", dest="track_all", action="store_false",
                   help="val 만 계산해 학습을 빠르게")
    p.add_argument("--batch_size", type=int, default=0, help="0 = full batch")
    p.add_argument("--grad_clip", type=float, default=1.0)
    # 실행
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--out_dir", default="results")
    p.add_argument("--tag", default="latent_ode")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.init_mode == "genotype" and args.encoder_days != 0:
        print("[알림] init_mode=genotype 에서는 encoder_days 가 무시됩니다.")
        args.encoder_days = 0

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = make_dataset(
        args.data,
        args.kinship,
        split=args.split,
        env_cols=tuple(args.env_cols),
        start_day=args.start_day,
        n_t=285 - args.start_day,
        fill_in_na_at_start=not args.no_fill_na_start,
        add_gdd=args.add_gdd,
        gdd_base=args.gdd_base,
        val_year=args.val_year,
        test_year=args.test_year,
        genotypes=args.genotypes,
    )
    print(f"[데이터] genotypes={len(ds.genotypes)} {ds.genotypes if len(ds.genotypes) <= 5 else ''}"
          f"  n_t={ds.train.n_t}  "
          f"train {ds.train_years} (N={len(ds.train)}) / val {ds.val_year} / test {ds.test_year}")
    if len(ds.genotypes) == 1:
        print("[비교] 단일 유전자형 설정입니다. 논문 Table 4 와 비교하세요 "
              "(Logi-PINN test 0.057±0.008, LSTM-NN 0.072±0.022).")
    if args.time_mode != "calendar":
        print(f"[열시간] 정규화 상수 {ds.gdd_scale:.0f} °C·day (학습 연도 평균) | "
              f"s(끝) 학습 {ds.train.s[:, -1].mean():.3f} / "
              f"검증 {ds.val.s[:, -1].mean():.3f} / 테스트 {ds.test.s[:, -1].mean():.3f}")
    print(f"[설정] time_mode={args.time_mode} init_mode={args.init_mode} encoder_days="
          f"{args.encoder_days or '전 구간'} "
          f"encoding={args.genetic_encoding} physics_weight={args.weight_physic}")
    print("[과제] 학습에 쓰지 않은 연도의 곡선 전체를 시간·기온·유전자형만으로 예측합니다. "
          "테스트 연도의 높이 관측은 모델에 입력되지 않습니다.")

    results, per_geno, preds, hists = [], [], {}, []
    for seed in args.seeds:
        model, res, pg, pred, hist = train_one_seed(ds, args, seed, device, not args.quiet)
        results.append(res)
        per_geno.append(pg)
        preds[seed] = pred
        hist["seed"] = seed
        hists.append(hist)
        print(f"  seed {seed} 완료 -> train {res['train_rmse']:.4f} | "
              f"val {res['val_rmse']:.4f} | test {res['test_rmse']:.4f}  "
              f"(best epoch {res['best_epoch']}, {res['seconds']:.0f}s, "
              f"params {res['n_params']})\n")
        torch.save(model.state_dict(),
                   os.path.join(args.out_dir, f"{args.tag}_split{args.split}_seed{seed}.pt"))

    res_df = pd.DataFrame(results)
    base = os.path.join(args.out_dir, f"{args.tag}_split{args.split}")
    res_df.to_csv(f"{base}_summary.csv", index=False)
    pd.concat(per_geno).to_csv(f"{base}_per_genotype.csv", index=False)
    pd.concat(hists).to_csv(f"{base}_history.csv", index=False)
    np.savez(
        f"{base}_test_predictions.npz",
        pred=np.stack([preds[s] for s in args.seeds]),
        y=ds.test.y,
        mask=ds.test.mask,
        genotype=np.array([ds.genotypes[i] for i in ds.test.g_idx]),
    )
    with open(f"{base}_args.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    n = len(args.seeds)
    print("=" * 62)
    print(f"시드별 결과 (시드 {n}개)")
    print("=" * 62)
    print(f"{'seed':>6} {'best_ep':>8} {'train':>9} {'val':>9} {'test':>9}")
    print("-" * 62)
    for r in results:
        print(f"{r['seed']:>6} {r['best_epoch']:>8} {r['train_rmse']:>9.4f} "
              f"{r['val_rmse']:>9.4f} {r['test_rmse']:>9.4f}")
    print("-" * 62)
    sd = res_df.std(ddof=1) if n > 1 else res_df.std(ddof=0) * 0
    print(f"{'평균±SD':>6} {'':>8} "
          f"{res_df.train_rmse.mean():>9.4f} {res_df.val_rmse.mean():>9.4f} "
          f"{res_df.test_rmse.mean():>9.4f}")
    print(f"{'':>6} {'':>8} "
          f"{'±' + format(sd.train_rmse, '.4f'):>9} "
          f"{'±' + format(sd.val_rmse, '.4f'):>9} "
          f"{'±' + format(sd.test_rmse, '.4f'):>9}")
    print("=" * 62)
    if n == 1:
        print("[주의] 시드가 1개라 표준편차가 0 입니다. 논문처럼 5개를 쓰세요.")

    if len(ds.genotypes) == 1:
        print("논문 Table 4 (단일 유전자형, split 0): "
              "Logi-PINN test 0.057 ± 0.008, LSTM-NN 0.072 ± 0.022")
    else:
        print("논문 Table 5 (다중 유전자형, year split): "
              "LSTM-NN test 0.058 ± 0.012, Logi-PINN 0.061 ± 0.007")
    print(f"결과 저장: {base}_*.csv / _test_predictions.npz")
    print(f"진단      : python analyze.py {base}_test_predictions.npz")


if __name__ == "__main__":
    main()