"""Validation-only coefficient search: wheat/Arabidopsis on GPU 2, maize on GPU 3."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "experiments/results/physics_ablation_20260908"
DEFAULT_OUT = ROOT / "experiments/results/lambda_ode_tuning_20260908"
GRID = [.01, .1, .5, 1., 2., 5., 10., 50., 100., 500.]
BASE = dict(wheat=2., maize=.5, arabidopsis=2.)


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(value):
    return "lambda_"+format(value, ".10g").replace(".", "p").replace("-", "m")


def imported_candidates(dataset):
    # Project historical records onto train/validation fields only. Their test
    # results were inspected in the preceding work, as disclosed in protocol.
    if dataset == "wheat":
        path = ROOT / "wheat/results/tuned_latent_ode_seed1_3.csv"
        frame = pd.read_csv(path, usecols=["seed", "best_epoch", "train_rmse", "val_rmse"])
    elif dataset == "maize":
        path = ROOT / "maize/results/tuning_20260904/final/all_runs.csv"
        frame = pd.read_csv(path, usecols=["seed", "best_epoch", "train_rmse", "val_rmse"])
    else:
        path = ROOT / "arabidopsis/results/all_runs_seed1_3.csv"
        frame = pd.read_csv(path, usecols=["model", "seed", "best_epoch", "train_rmse_m", "val_rmse_m"])
        frame = frame[frame.model.eq("Latent Neural ODE (ours)")].copy()
        frame["train_rmse"] = frame.train_rmse_m*100
        frame["val_rmse"] = frame.val_rmse_m*100
    records = []
    for row in frame.to_dict("records"):
        seed = int(row["seed"])
        checkpoint = OLD / dataset / f"full_seed{seed}/checkpoint.pt"
        if not checkpoint.exists() and dataset == "wheat":
            checkpoint = ROOT / f"wheat/results/checkpoints/tuned_latent_ode_split0_seed{seed}.pt"
        if not checkpoint.exists() and dataset == "maize":
            checkpoint = ROOT / f"maize/results/tuning_20260904/trials/l160/seed{seed}/checkpoint.pt"
        records.append(dict(dataset=dataset, lambda_ode=BASE[dataset], seed=seed,
            train_rmse=row["train_rmse"], val_rmse=row["val_rmse"], best_epoch=int(row["best_epoch"]),
            checkpoint=str(checkpoint.relative_to(ROOT)) if checkpoint.exists() else None,
            checkpoint_sha256=sha(checkpoint) if checkpoint.exists() else None,
            source=str(path.relative_to(ROOT)), origin="frozen previous full-model result",
            prior_test_evaluated=True, test_evaluations_this_search=0))
    for seed in (1, 2, 3):
        folder = OLD / dataset / f"no_ode_residual_seed{seed}"
        result = json.loads((folder / "result.json").read_text())
        records.append(dict(dataset=dataset, lambda_ode=0., seed=seed,
            train_rmse=result["metrics"]["train"]["rmse"], val_rmse=result["metrics"]["val"]["rmse"],
            best_epoch=result["best_epoch"], checkpoint=str((folder/"checkpoint.pt").relative_to(ROOT)),
            checkpoint_sha256=sha(folder/"checkpoint.pt"), source=str(folder.relative_to(ROOT)),
            origin="frozen previous zero-residual result", prior_test_evaluated=True,
            test_evaluations_this_search=0))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--short-gpu", type=int, default=2)
    parser.add_argument("--long-gpu", type=int, default=3)
    parser.add_argument("--maize-workers", type=int, default=2)
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / "protocol.json").exists():
        raise FileExistsError("Use a fresh output directory for an independent search")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    sources = [ROOT / "experiments/lambda_ode_trial.py", Path(__file__), ROOT / "experiments/physics_ablation.py"]
    for dataset in BASE:
        sources += list((ROOT / dataset / "code").glob("*.py"))
    sources += [ROOT / p for p in ("wheat/results/final_config.json", "wheat/data/align_height_env_same_length.csv",
        "wheat/data/kinship_matrix_astle.csv", "maize/results/tuning_20260904/selected.json",
        "maize/data/processed/height_observations.csv", "maize/data/processed/weather_daily.csv",
        "maize/data/processed/genotypes.csv", "maize/data/processed/year_splits.json",
        "arabidopsis/data/processed/stem_length_long.csv", "paper/relative_errors/denominators.csv")]
    protocol = dict(started_unix=time.time(), grid=GRID, seed_screen=[1], confirmation_seeds=[1, 2, 3],
        refinement="Once around the best positive seed-1 validation candidate; geometric midpoints to neighbors, or one decade outward at a boundary",
        refinement_bounds=[.001, 5000.], confirmation_top_positive=3,
        gpu_assignment=dict(wheat=args.short_gpu, arabidopsis=args.short_gpu, maize=args.long_gpu),
        concurrent_dataset_workers=dict(wheat=1, arabidopsis=1, maize=args.maize_workers),
        fixed="Architecture, initialization, maximum-height coefficient, optimizer, schedule, epochs and split-specific checkpoint selection",
        checkpoint_selection="Unchanged per-dataset validation rule from the manuscript",
        candidate_selection="Mean final validation RMSE over seeds 1--3; ties use SD then smaller lambda; zero is included in overall ranking",
        test_policy="No new candidate test evaluations. Freeze selections for all three datasets before evaluating the selected positive coefficient once per seed.",
        evaluation_limitation="The existing test-set ablation results were already inspected before this follow-up search. Final test scores are follow-up evaluations on reused test sets, not a fresh independent confirmation.",
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(sources))})
    atomic(out / "protocol.json", protocol)
    active, lock, stopped = set(), threading.Lock(), threading.Event()

    def stop(_signum, _frame):
        stopped.set()
        with lock:
            for process in list(active):
                process.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def invoke(command, gpu, logfile):
        if stopped.is_set():
            raise InterruptedError("Search interrupted")
        env = os.environ | dict(CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
        with logfile.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                active.add(process)
            code = process.wait()
            with lock:
                active.discard(process)
        if stopped.is_set():
            raise InterruptedError("Search interrupted")
        if code:
            raise RuntimeError(f"Run exited with {code}; see {logfile}")

    def search(dataset, gpu, workers):
        folder = out / dataset
        folder.mkdir(exist_ok=True)
        check = out / "checks" / dataset / "verification.json"
        if not check.exists():
            cmd = [sys.executable, "-u", str(ROOT / "experiments/lambda_ode_trial.py"), "train",
                   "--dataset", dataset, "--seed", "1", "--lambda-ode", str(BASE[dataset]),
                   "--verify-prefix", "--output", str(check.parent)]
            invoke(cmd, gpu, out / "logs" / f"verify_{dataset}.log")
        assert json.loads(check.read_text())["status"] == "passed"
        records = imported_candidates(dataset)
        atomic(folder / "imported_validation_records.json", records)

        def trial(value_seed):
            value, seed = value_seed
            destination = folder / "trials" / key(value) / f"seed{seed}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            result_path = destination / "result.json"
            if not result_path.exists():
                cmd = [sys.executable, "-u", str(ROOT / "experiments/lambda_ode_trial.py"), "train",
                       "--dataset", dataset, "--seed", str(seed), "--lambda-ode", str(value), "--output", str(destination)]
                print(f"START GPU {gpu}: {dataset} lambda={value:g} seed={seed}", flush=True)
                invoke(cmd, gpu, out / "logs" / f"{dataset}_{key(value)}_seed{seed}.log")
            result = json.loads(result_path.read_text())
            assert result["status"] == "complete" and result["test_evaluations"] == 0
            print(f"DONE {dataset} lambda={value:g} seed={seed} val={result['val_rmse']:.8f}", flush=True)
            return result

        def stage(name, jobs):
            atomic(folder / "phase.json", dict(phase=name, jobs=[dict(lambda_ode=v, seed=s) for v, s in jobs]))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(trial, jobs):
                    records.append(result)
                    atomic(folder / "validation_records.json", records)

        # Existing base and zero results are reused, not retrained for screening.
        stage("screen", [(value, 1) for value in GRID if value != BASE[dataset]])
        screen = sorted([r for r in records if r["seed"] == 1 and r["lambda_ode"] > 0],
                        key=lambda r: (r["val_rmse"], r["lambda_ode"]))
        best_value = screen[0]["lambda_ode"]
        grid = sorted(set(GRID))
        i = grid.index(best_value)
        left = best_value/10 if i == 0 else math.sqrt(grid[i-1]*best_value)
        right = best_value*10 if i == len(grid)-1 else math.sqrt(best_value*grid[i+1])
        refinement = sorted(set(float(format(v, ".10g")) for v in (left, right)) - set(grid))
        atomic(folder / "refinement.json", dict(based_on="seed-1 validation only",
            center=best_value, lambdas=refinement, initial_screen=screen))
        stage("refine", [(value, 1) for value in refinement])
        screen = sorted([r for r in records if r["seed"] == 1 and r["lambda_ode"] > 0],
                        key=lambda r: (r["val_rmse"], r["lambda_ode"]))
        finalists = [r["lambda_ode"] for r in screen[:3]]
        atomic(folder / "finalists.json", dict(based_on="seed-1 validation only", lambdas=finalists))
        available = {(r["lambda_ode"], r["seed"]) for r in records}
        stage("confirm", [(value, seed) for value in finalists for seed in (2, 3) if (value, seed) not in available])
        groups = {}
        for row in records:
            groups.setdefault(row["lambda_ode"], {})[row["seed"]] = row
        ranking = []
        for value, by_seed in groups.items():
            if set(by_seed) == {1, 2, 3}:
                vals = [by_seed[s]["val_rmse"] for s in (1, 2, 3)]
                ranking.append(dict(lambda_ode=value, val_mean=float(np.mean(vals)), val_sd=float(np.std(vals, ddof=1))))
        ranking.sort(key=lambda r: (r["val_mean"], r["val_sd"], r["lambda_ode"]))
        positive = next(row for row in ranking if row["lambda_ode"] > 0).copy()
        winner = positive["lambda_ode"]
        atomic(folder / "coefficient_selected.json", dict(selected_positive=positive, overall=ranking[0], ranking=ranking))
        # Rarely needed: the original Arabidopsis script did not retain all full
        # checkpoints. Reproduce a missing selected checkpoint with fixed lambda.
        for seed in (1, 2, 3):
            row = groups[winner][seed]
            if row.get("checkpoint") is None:
                materialized = trial((winner, seed))
                np.testing.assert_allclose(materialized["val_rmse"], row["val_rmse"], rtol=1e-6, atol=2e-6)
                groups[winner][seed] = materialized
                records = [r for r in records if (r["lambda_ode"], r["seed"]) != (winner, seed)]
                records.append(materialized)
                atomic(folder / "validation_records.json", records)
        positive["checkpoints"] = {str(seed): dict(path=groups[winner][seed]["checkpoint"],
            sha256=groups[winner][seed]["checkpoint_sha256"]) for seed in (1, 2, 3)}
        selected = dict(dataset=dataset, selected_positive=positive, selected_overall=ranking[0],
            ranking=ranking, frozen_unix=time.time(), new_candidate_test_evaluations=0,
            positive_beats_zero_on_validation=positive["val_mean"] < next(r["val_mean"] for r in ranking if r["lambda_ode"] == 0))
        atomic(folder / "selected.json", selected)
        atomic(folder / "phase.json", dict(phase="selection_frozen", selected_positive=winner))
        print(f"FROZEN {dataset}: positive lambda={winner:g} val={positive['val_mean']:.8f}; overall={ranking[0]['lambda_ode']:g}", flush=True)
        return selected

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(search, "wheat", args.short_gpu, 1),
                   pool.submit(search, "arabidopsis", args.short_gpu, 1),
                   pool.submit(search, "maize", args.long_gpu, args.maize_workers)]
        selections = [future.result() for future in futures]
    for path, expected in protocol["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Source changed during search: {path}")
    atomic(out / "all_selections_frozen.json", dict(frozen_unix=time.time(), datasets=selections,
        new_candidate_test_evaluations=0))

    def final(dataset, gpu):
        for seed in (1, 2, 3):
            destination = out / dataset / "final" / f"seed{seed}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-u", str(ROOT / "experiments/lambda_ode_trial.py"), "evaluate",
                   "--dataset", dataset, "--seed", str(seed), "--selection", str(out/dataset/"selected.json"),
                   "--output", str(destination)]
            invoke(cmd, gpu, out / "logs" / f"final_{dataset}_seed{seed}.log")
        atomic(out / dataset / "phase.json", dict(phase="complete"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(final, dataset, args.long_gpu if dataset == "maize" else args.short_gpu) for dataset in BASE]
        for future in futures:
            future.result()
    atomic(out / "completed.json", dict(status="complete", finished_unix=time.time(),
        seconds=time.time()-protocol["started_unix"], selected_final_test_evaluations=9))
    print("LAMBDA SEARCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
