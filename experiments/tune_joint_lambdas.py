"""Frozen two-dimensional validation search, using only GPUs 2 and 3 by default."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

from tune_lambda_ode import ROOT, OLD, BASE, sha, atomic

PREVIOUS = ROOT / "experiments/results/lambda_ode_tuning_20260908"
DEFAULT_OUT = ROOT / "experiments/results/joint_lambda_tuning_20260908"
BASE_K = dict(wheat=.1, maize=.5, arabidopsis=.1)
ODE_GRID = dict(wheat=[0., .1, 1., 3.16227766, 10., 100.],
                maize=[0., .5, 5., 50., 500., 5000.],
                arabidopsis=[0., .05, .5, 2., 20., 200.])
K_FACTORS = [0., .01, .1, 1., 10.]


def read(path):
    return json.loads(Path(path).read_text())


def rounded(value):
    return float(format(value, ".10g"))


def pair(row):
    return row["lambda_ode"], row["lambda_k"]


def job_key(row):
    return (*pair(row), row["seed"])


def key(ode, k):
    return f"ode_{ode:.10g}_k_{k:.10g}".replace(".", "p").replace("-", "m")


def imported(dataset):
    rows = []
    for old in read(PREVIOUS / dataset / "validation_records.json"):
        row = {k: old[k] for k in ("dataset", "seed", "lambda_ode", "train_rmse", "val_rmse",
                                   "best_epoch", "checkpoint", "checkpoint_sha256")}
        row.update(lambda_k=BASE_K[dataset], origin="reused ODE-only search validation record",
                   source=str((PREVIOUS / dataset / "validation_records.json").relative_to(ROOT)),
                   prior_test_may_have_been_evaluated=True, test_evaluations_this_search=0)
        rows.append(row)
    for seed in (1, 2, 3):
        folder = OLD / dataset / f"no_biological_loss_seed{seed}"
        old = read(folder / "result.json")
        rows.append(dict(dataset=dataset, seed=seed, lambda_ode=0., lambda_k=0.,
            train_rmse=old["metrics"]["train"]["rmse"], val_rmse=old["metrics"]["val"]["rmse"],
            best_epoch=old["best_epoch"], checkpoint=str((folder / "checkpoint.pt").relative_to(ROOT)),
            checkpoint_sha256=old["checkpoint_sha256"], origin="reused both-zero no-physics control",
            source=str(folder.relative_to(ROOT)), prior_test_may_have_been_evaluated=True,
            test_evaluations_this_search=0))
    assert len({job_key(r) for r in rows}) == len(rows)
    return rows


def refinement_plan(records):
    screen = sorted((r for r in records if r["seed"] == 1 and min(pair(r)) > 0),
                    key=lambda r: (r["val_rmse"], *pair(r)))
    center = pair(screen[0])
    neighbors = []
    for axis, bounds in enumerate(((1e-4, 50000.), (1e-5, 50.))):
        grid = sorted({pair(r)[axis] for r in records if r["seed"] == 1 and pair(r)[axis] > 0})
        value, i = center[axis], grid.index(center[axis])
        left = value / 10 if i == 0 else math.sqrt(grid[i - 1] * value)
        right = value * 10 if i == len(grid) - 1 else math.sqrt(value * grid[i + 1])
        neighbors.append(sorted({rounded(max(bounds[0], min(bounds[1], v))) for v in (left, right)}))
    seen = {pair(r) for r in records if r["seed"] == 1}
    candidates = sorted(set(itertools.product(*neighbors)) - seen)
    return dict(based_on="seed-1 validation only", center=list(center), pairs=[list(p) for p in candidates])


def finalist_plan(records):
    screen = sorted((r for r in records if r["seed"] == 1), key=lambda r: (r["val_rmse"], *pair(r)))
    positive = [list(pair(r)) for r in screen if min(pair(r)) > 0][:3]
    k_only = next(list(pair(r)) for r in screen if r["lambda_ode"] == 0 and r["lambda_k"] > 0)
    ode_only = next(list(pair(r)) for r in screen if r["lambda_ode"] > 0 and r["lambda_k"] == 0)
    return dict(based_on="seed-1 validation only", positive=positive, boundaries=[k_only, ode_only],
                pairs=positive + [k_only, ode_only])


def rank(records):
    groups = {}
    for row in records:
        groups.setdefault(pair(row), {})[row["seed"]] = row
    ranking = []
    for (ode, k), group in groups.items():
        if set(group) == {1, 2, 3}:
            vals = [group[s]["val_rmse"] for s in (1, 2, 3)]
            ranking.append(dict(lambda_ode=ode, lambda_k=k, val_mean=float(np.mean(vals)),
                                val_sd=float(np.std(vals, ddof=1))))
    ranking.sort(key=lambda r: (r["val_mean"], r["val_sd"], *pair(r)))
    return ranking, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--short-gpu", type=int, default=2)
    parser.add_argument("--long-gpu", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    grids = {d: [list(p) for p in itertools.product(ODE_GRID[d], [rounded(BASE_K[d] * f) for f in K_FACTORS])]
             for d in BASE}
    if args.resume:
        protocol = read(out / "protocol.json")
        assert protocol["grids"] == grids
        assert protocol["gpu_assignment"] == dict(wheat=args.short_gpu, arabidopsis=args.short_gpu, maize=args.long_gpu)
    else:
        if (out / "protocol.json").exists():
            raise FileExistsError("An existing search requires --resume")
        sources = [ROOT / "experiments" / name for name in
                   ("joint_lambda_trial.py", "tune_joint_lambdas.py", "physics_ablation.py",
                    "lambda_ode_trial.py", "tune_lambda_ode.py")]
        sources += [ROOT / p for p in read(PREVIOUS / "protocol.json")["source_sha256"]]
        for d in BASE:
            (out / d).mkdir(exist_ok=True)
            snapshot = out / d / "imported_validation_records.json"
            atomic(snapshot, imported(d))
            sources.append(snapshot)
            sources.append(PREVIOUS / d / "validation_records.json")
        protocol = dict(started_unix=time.time(), grids=grids, seed_screen=[1], confirmation_seeds=[1, 2, 3],
            refinement="One four-corner geometric refinement around the best both-positive seed-1 pair; nearest positive coordinate values or a decade outward; ODE bounds [0.0001,50000], K bounds [0.00001,50]",
            confirmation="Top three both-positive pairs, best K-only pair and best ODE-only pair by seed-1 validation; reuse completed seeds",
            fixed="Architecture, initialization, optimizer, weight decay, epochs, schedule, masks and per-dataset checkpoint selection are unchanged",
            candidate_selection="Mean final validation RMSE over three seeds; ties use sample SD, lambda_ODE, lambda_K. All complete historical candidates remain eligible.",
            no_physics_definition="lambda_ODE = lambda_K = 0, with the same latent ODE and optimizer weight decay",
            phyto_definition="Both lambda_ODE and lambda_K strictly positive; also disclose the overall validation winner, including boundaries",
            gpu_assignment=dict(wheat=args.short_gpu, arabidopsis=args.short_gpu, maize=args.long_gpu),
            scheduling="One wheat and one Arabidopsis on the short GPU; two wheat trials after Arabidopsis selection freezes. Two maize trials on the long GPU.",
            resumption="Save model, optimizer, scheduler, validation window and all RNG states every 100 epochs and on graceful interruption; preserve the original schedule",
            test_policy="Freeze all three dataset selections first. Evaluate best both-positive and, only if different, overall winner, once per seed. Never evaluate nonselected new candidates on test.",
            evaluation_limitation="Previously inspected test sets are reused. These are follow-up test evaluations, not independent confirmation; no test-based coefficient selection.",
            source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(sources))})
        atomic(out / "protocol.json", protocol)
    for p, expected in protocol["source_sha256"].items():
        assert sha(ROOT / p) == expected, p
    active, lock, stopped = set(), threading.Lock(), threading.Event()
    wheat_slots = threading.Semaphore(1)

    def stop(_signum, _frame):
        stopped.set()
        with lock:
            for process in list(active):
                process.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def invoke(command, gpu, logfile):
        env = os.environ | dict(CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
        with logfile.open("a") as log:
            with lock:
                if stopped.is_set():
                    raise InterruptedError("Search interrupted")
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active.add(process)
            code = process.wait()
            with lock:
                active.discard(process)
        if stopped.is_set():
            raise InterruptedError("Search interrupted")
        if code:
            raise RuntimeError(f"Run exited with {code}; see {logfile}")

    def command(dataset, ode, k, seed, destination):
        return [sys.executable, "-u", str(ROOT / "experiments/joint_lambda_trial.py"), "train",
                "--dataset", dataset, "--seed", str(seed), "--lambda-ode", str(ode), "--lambda-k", str(k),
                "--output", str(destination)]

    def verify(dataset, gpu):
        folder = out / "checks" / dataset
        if (folder / "verification.json").exists():
            assert read(folder / "verification.json")["status"] == "passed"
            return
        for label, ode, k, epochs in (("full_direct", BASE[dataset], BASE_K[dataset], 20),
                                      ("full_resumed", BASE[dataset], BASE_K[dataset], 10),
                                      ("none", 0., 0., 10)):
            destination = folder / "diagnostics" / label
            if not (destination / "paused.json").exists():
                cmd = command(dataset, ode, k, 1, destination) + ["--stop-after", str(epochs)]
                if (destination / "resume_state.pt").exists():
                    cmd.append("--resume")
                invoke(cmd, gpu, out / "logs" / f"verify_{dataset}_{label}.log")
            reference = "no_biological_loss" if label == "none" else "full"
            current = pd.read_csv(destination / "history.csv").iloc[0]
            old = pd.read_csv(OLD / dataset / f"{reference}_seed1/history.csv").iloc[0]
            for metric in ("data_loss", "objective", "val_rmse"):
                np.testing.assert_allclose(current[metric], old[metric], rtol=1e-7, atol=1e-9)
        destination = folder / "diagnostics/full_resumed"
        if read(destination / "paused.json")["epoch"] < 20:
            invoke(command(dataset, BASE[dataset], BASE_K[dataset], 1, destination) +
                   ["--resume", "--stop-after", "20"], gpu, out / "logs" / f"verify_{dataset}_resume.log")
        direct = folder / "diagnostics/full_direct"
        assert read(destination / "paused.json")["final_state_sha256"] == read(direct / "paused.json")["final_state_sha256"]
        pd.testing.assert_frame_equal(pd.read_csv(destination / "history.csv"), pd.read_csv(direct / "history.csv"), check_exact=True)
        atomic(folder / "verification.json", dict(status="passed", full_and_both_zero_prefix="matches previous controls",
            resumed_20_epoch_state_and_history="bitwise identical to uninterrupted training", test_evaluations=0))

    def search(dataset, gpu):
        folder = out / dataset
        if (folder / "selected.json").exists():
            if dataset == "arabidopsis":
                wheat_slots.release()
            return read(folder / "selected.json")
        verify(dataset, gpu)
        records = {job_key(r): r for r in read(folder / "imported_validation_records.json")}
        for p in (folder / "trials").glob("*/*/result.json"):
            r = read(p)
            assert r["status"] == "complete" and r["test_evaluations"] == 0
            records[job_key(r)] = r

        def persist():
            atomic(folder / "validation_records.json", [records[k] for k in sorted(records)])
        persist()

        def trial(job):
            ode, k, seed = job
            acquired = False
            try:
                if dataset == "wheat":
                    while not acquired:
                        if stopped.is_set():
                            raise InterruptedError("Search interrupted")
                        acquired = wheat_slots.acquire(timeout=1)
                destination = folder / "trials" / key(ode, k) / f"seed{seed}"
                cmd = command(dataset, ode, k, seed, destination)
                if (destination / "resume_state.pt").exists():
                    cmd.append("--resume")
                print(f"START GPU {gpu}: {dataset} ODE={ode:g} K={k:g} seed={seed}", flush=True)
                invoke(cmd, gpu, out / "logs" / f"{dataset}_{key(ode, k)}_seed{seed}.log")
                r = read(destination / "result.json")
                assert r["status"] == "complete" and r["test_evaluations"] == 0 and job_key(r) == job
                print(f"DONE {dataset} ODE={ode:g} K={k:g} seed={seed} val={r['val_rmse']:.8f}", flush=True)
                return r
            finally:
                if acquired:
                    wheat_slots.release()

        def stage(name, jobs):
            pending = [job for job in jobs if job not in records]
            atomic(folder / "phase.json", dict(phase=name, jobs=[list(j) for j in jobs], pending_at_start=len(pending)))
            with ThreadPoolExecutor(max_workers=1 if dataset == "arabidopsis" else 2) as pool:
                for future in as_completed([pool.submit(trial, j) for j in pending]):
                    r = future.result()
                    records[job_key(r)] = r
                    persist()

        stage("screen", [(ode, k, 1) for ode, k in grids[dataset]])
        path = folder / "refinement.json"
        if not path.exists():
            atomic(path, refinement_plan(list(records.values())))
        refinement = read(path)
        stage("refine", [(ode, k, 1) for ode, k in refinement["pairs"]])
        path = folder / "finalists.json"
        if not path.exists():
            atomic(path, finalist_plan(list(records.values())))
        finalists = read(path)
        stage("confirm", [(ode, k, seed) for ode, k in finalists["pairs"] for seed in (2, 3)])
        ranking, groups = rank(list(records.values()))
        phyto = next(r.copy() for r in ranking if min(pair(r)) > 0)
        overall = ranking[0].copy()
        atomic(folder / "coefficient_selected.json", dict(selected_phyto=phyto, selected_overall=overall, ranking=ranking))
        for chosen in (phyto, overall):
            for seed in (1, 2, 3):
                row = groups[pair(chosen)][seed]
                if not row.get("checkpoint") or not (ROOT / row["checkpoint"]).exists():
                    new = trial((*pair(chosen), seed))
                    np.testing.assert_allclose(new["val_rmse"], row["val_rmse"], rtol=1e-6, atol=2e-6)
                    records[job_key(new)] = new
                    groups[pair(chosen)][seed] = new
                    persist()
            chosen["checkpoints"] = {str(s): dict(path=groups[pair(chosen)][s]["checkpoint"],
                sha256=groups[pair(chosen)][s]["checkpoint_sha256"]) for s in (1, 2, 3)}
            for cp in chosen["checkpoints"].values():
                assert sha(ROOT / cp["path"]) == cp["sha256"]
        selected = dict(dataset=dataset, selected_phyto=phyto, selected_overall=overall, ranking=ranking,
                        frozen_unix=time.time(), new_candidate_test_evaluations=0)
        atomic(folder / "selected.json", selected)
        atomic(folder / "phase.json", dict(phase="selection_frozen"))
        print(f"FROZEN {dataset}: PhytoODE={pair(phyto)} val={phyto['val_mean']:.8f}; overall={pair(overall)}", flush=True)
        if dataset == "arabidopsis":
            wheat_slots.release()
        return selected

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {d: pool.submit(search, d, args.long_gpu if d == "maize" else args.short_gpu) for d in BASE}
        selections = [futures[d].result() for d in BASE]
    for p, expected in protocol["source_sha256"].items():
        assert sha(ROOT / p) == expected, p
    frozen_path = out / "all_selections_frozen.json"
    if not frozen_path.exists():
        atomic(frozen_path, dict(frozen_unix=time.time(), datasets=selections, new_candidate_test_evaluations=0,
            selection_sha256={d: sha(out / d / "selected.json") for d in BASE}))
    assert read(frozen_path)["datasets"] == selections

    def final(dataset):
        selected = read(out / dataset / "selected.json")
        roles = ["selected_phyto"]
        if pair(selected["selected_overall"]) != pair(selected["selected_phyto"]):
            roles.append("selected_overall")
        for role in roles:
            for seed in (1, 2, 3):
                destination = out / dataset / "final" / role / f"seed{seed}"
                if (destination / "result.json").exists():
                    assert read(destination / "result.json")["selection_sha256"] == sha(out / dataset / "selected.json")
                    continue
                cmd = [sys.executable, "-u", str(ROOT / "experiments/joint_lambda_trial.py"), "evaluate",
                       "--dataset", dataset, "--seed", str(seed), "--role", role,
                       "--selection", str(out / dataset / "selected.json"), "--output", str(destination)]
                invoke(cmd, args.long_gpu if dataset == "maize" else args.short_gpu,
                       out / "logs" / f"final_{dataset}_{role}_seed{seed}.log")
        atomic(out / dataset / "phase.json", dict(phase="complete"))
        return 3 * len(roles)
    with ThreadPoolExecutor(max_workers=3) as pool:
        count = sum(pool.map(final, BASE))
    atomic(out / "completed.json", dict(status="complete", finished_unix=time.time(),
        seconds=time.time() - protocol["started_unix"], selected_final_test_evaluations=count))
    print("JOINT LAMBDA SEARCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
