"""Run a fixed loss-ablation matrix across explicitly selected GPUs.

First reproduce seed 1 of each published full model, then run both ablations
for seeds 1--3. Completed runs are verified and reused when restarting.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments/results/physics_ablation_20260908"
DATASETS = ("wheat", "maize", "arabidopsis")
VARIANTS = ("no_ode_residual", "no_biological_loss")


def references():
    rows = []
    for dataset, path in (
        ("wheat", "wheat/results/tuned_latent_ode_seed1_3.csv"),
        ("maize", "maize/results/tuning_20260904/final/all_runs.csv"),
        ("arabidopsis", "arabidopsis/results/all_runs_seed1_3.csv"),
    ):
        frame = pd.read_csv(ROOT / path)
        if dataset == "maize":
            frame = frame[frame.model_key.eq("latent_tuned")]
        if dataset == "arabidopsis":
            frame = frame[frame.model.eq("Latent Neural ODE (ours)")]
        assert sorted(frame.seed.tolist()) == [1, 2, 3]
        for original in frame.to_dict("records"):
            item = dict(dataset=dataset, variant="full", seed=int(original["seed"]),
                        best_epoch=int(original["best_epoch"]), source=path)
            for split in ("train", "val", "test"):
                item[split+"_rmse"] = original[split+"_rmse"+("_m" if dataset == "arabidopsis" else "")] * (100 if dataset == "arabidopsis" else 1)
            rows.append(item)
    return rows


def verify_control(output, dataset):
    result = json.loads((output / dataset / "full_seed1/result.json").read_text())
    ref = next(row for row in references() if row["dataset"] == dataset and row["seed"] == 1)
    # Existing tables use float32 curve reductions; the new exports use float64.
    tolerance = {"wheat": 2e-7, "arabidopsis": 2e-5, "maize": .002}[dataset]
    differences = {split: result["metrics"][split]["rmse"]-ref[split+"_rmse"]
                   for split in ("train", "val", "test")}
    passed = result["best_epoch"] == ref["best_epoch"] and max(map(abs, differences.values())) <= tolerance
    record = dict(dataset=dataset, passed=passed, differences=differences,
                  tolerance=tolerance, actual_epoch=result["best_epoch"], expected_epoch=ref["best_epoch"])
    (output / dataset / "control_verification.json").write_text(json.dumps(record, indent=2)+"\n")
    if not passed:
        raise RuntimeError(f"Full-model reproduction mismatch: {record}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", type=int, nargs=3, default=[0, 2, 3])
    parser.add_argument("--controls-running", action="store_true",
                        help="Wait for separately launched full/seed1 controls on the three GPUs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log_dir = args.output / "logs"
    log_dir.mkdir(exist_ok=True)
    paths = list((ROOT / "experiments").glob("*.py"))
    for dataset in DATASETS:
        paths += list((ROOT / dataset / "code").glob("*.py"))
    paths += [ROOT / row["source"] for row in references()]
    paths += [ROOT / path for path in (
        "wheat/results/final_config.json", "wheat/data/align_height_env_same_length.csv",
        "wheat/data/kinship_matrix_astle.csv", "maize/results/tuning_20260904/selected.json",
        "maize/data/processed/height_observations.csv", "maize/data/processed/weather_daily.csv",
        "maize/data/processed/genotypes.csv", "maize/data/processed/year_splits.json",
        "arabidopsis/data/processed/stem_length_long.csv", "paper/relative_errors/denominators.csv")]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths))}
    manifest = dict(datasets=list(DATASETS), variants=list(VARIANTS), seeds=[1, 2, 3],
        gpus=args.gpus, full_model_control_seeds=[1],
        full_model_comparison="Frozen manuscript results for seeds 1--3",
        selection="Unchanged dataset-specific validation rule; no ablation retuning",
        source_sha256=hashes, references=references())
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["source_sha256"] != hashes:
            raise RuntimeError("Sources changed since experiment manifest was frozen")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    jobs = queue.Queue()
    # Longest jobs first so GPUs can share the maize work once controls finish.
    for dataset in ("maize", "wheat", "arabidopsis"):
        for seed in (1, 2, 3):
            for variant in VARIANTS:
                if not (args.output / dataset / f"{variant}_seed{seed}/result.json").exists():
                    jobs.put((dataset, variant, seed))

    def run(gpu, dataset, variant, seed):
        folder = args.output / dataset / f"{variant}_seed{seed}"
        if (folder / "result.json").exists():
            return
        env = os.environ | dict(CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
        command = [sys.executable, "-u", str(ROOT / "experiments/physics_ablation.py"),
                   "--dataset", dataset, "--variant", variant, "--seed", str(seed), "--output", str(folder)]
        print(f"START GPU {gpu}: {dataset}/{variant}/seed{seed}", flush=True)
        with (log_dir / f"{dataset}_{variant}_seed{seed}.log").open("w") as log:
            subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"DONE GPU {gpu}: {dataset}/{variant}/seed{seed}", flush=True)

    def worker(gpu, control_dataset):
        control = args.output / control_dataset / "full_seed1/result.json"
        if args.controls_running:
            deadline = time.monotonic()+1200
            while not control.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Control did not complete: {control_dataset}")
                time.sleep(2)
        else:
            run(gpu, control_dataset, "full", 1)
        print("CONTROL " + json.dumps(verify_control(args.output, control_dataset)), flush=True)
        while True:
            try:
                dataset, variant, seed = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                run(gpu, dataset, variant, seed)
            finally:
                jobs.task_done()

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, gpu, dataset) for gpu, dataset in zip(args.gpus, DATASETS)]
        for future in futures:
            future.result()
    print("ALL ABLATIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
