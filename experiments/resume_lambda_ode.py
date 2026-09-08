"""Recover an interrupted confirmation phase without changing frozen search rules.

Completed results and selected datasets are reused. An incomplete training
attempt is preserved for diagnosis, then restarted from the identical seed and
full original schedule. Its recorded prefix must match the replacement run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from tune_lambda_ode import ROOT, DEFAULT_OUT, BASE, atomic, sha, key


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = read(out/"protocol.json")
    if (out/"completed.json").exists() or (out/"all_selections_frozen.json").exists():
        raise RuntimeError("This recovery command only supports interrupted confirmation before final tests")
    if (out/"recovery.json").exists():
        raise RuntimeError("A recovery attempt already exists; inspect its state before proceeding")
    assert not list(out.glob("*/final/*/result.json"))

    def verify_sources():
        for path, expected in protocol["source_sha256"].items():
            assert sha(ROOT/path) == expected, path
    verify_sources()
    jobs = []
    for dataset in BASE:
        folder = out/dataset
        if (folder/"selected.json").exists():
            assert read(folder/"phase.json")["phase"] == "selection_frozen"
            continue
        phase = read(folder/"phase.json")
        assert phase["phase"] == "confirm", (dataset, phase)
        finalists = read(folder/"finalists.json")["lambdas"]
        for job in phase["jobs"]:
            assert job["lambda_ode"] in finalists and job["seed"] in (2, 3)
            destination = folder/"trials"/key(job["lambda_ode"])/f"seed{job['seed']}"
            if (destination/"result.json").exists():
                assert read(destination/"result.json")["status"] == "complete"
                continue
            job = job | dict(dataset=dataset, output=str(destination.relative_to(ROOT)),
                             gpu=protocol["gpu_assignment"][dataset])
            if destination.exists():
                interrupted = read(destination/"interrupted.json")
                assert interrupted["test_evaluations"] == 0
                history = pd.read_csv(destination/"history.csv")
                history = history[history.epoch.mod(10).eq(0)]
                job.update(interrupted_epoch=interrupted["epoch"],
                    interrupted_prefix=history[["epoch", "data_loss", "objective", "val_rmse"]].to_dict("records"),
                    interrupted_config_sha256=sha(destination/"config.json"),
                    interrupted_history_sha256=sha(destination/"history.csv"))
            jobs.append(job)
    assert jobs, "No incomplete confirmation jobs"
    recovery = dict(status="planned", started_unix=time.time(), protocol_sha256=sha(out/"protocol.json"),
        recovery_code_sha256=sha(Path(__file__)),
        reason="Previous controller and a training process received termination signals; sender not identified. No model numerical/OOM error was recorded.",
        policy="Reuse completed trials and frozen finalists. Restart interrupted jobs from the same seed with the entire original schedule; match the recorded prefix. Preserve selection and test policies.",
        jobs=jobs, new_candidate_test_evaluations=0)
    atomic(out/"recovery.json", recovery)
    active, stopped = [], False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
        for process in active:
            process.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def invoke(command, gpu, logfile):
        if stopped:
            raise InterruptedError("Recovery interrupted")
        env = os.environ | dict(CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
        with logfile.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append(process)
            code = process.wait()
            active.remove(process)
        if code or stopped:
            raise RuntimeError(f"Recovery run did not complete; see {logfile}")

    for job in jobs:
        destination = ROOT/job["output"]
        if destination.exists():
            diagnostic = out/"diagnostics"/f"{job['dataset']}_{key(job['lambda_ode'])}_seed{job['seed']}_interrupted"
            diagnostic.parent.mkdir(exist_ok=True)
            assert not diagnostic.exists()
            destination.rename(diagnostic)
        command = [sys.executable, "-u", str(ROOT/"experiments/lambda_ode_trial.py"), "train",
                   "--dataset", job["dataset"], "--seed", str(job["seed"]),
                   "--lambda-ode", str(job["lambda_ode"]), "--output", str(destination)]
        print(f"RECOVER {job['dataset']} lambda={job['lambda_ode']:g} seed={job['seed']} on GPU {job['gpu']}", flush=True)
        invoke(command, job["gpu"], out/"logs"/f"recovery_{job['dataset']}_{key(job['lambda_ode'])}_seed{job['seed']}.log")
        result = read(destination/"result.json")
        assert result["status"] == "complete" and result["test_evaluations"] == 0
        if "interrupted_prefix" in job:
            prefix = pd.DataFrame(job["interrupted_prefix"]).set_index("epoch")
            actual = pd.read_csv(destination/"history.csv").set_index("epoch")
            np.testing.assert_allclose(actual.loc[prefix.index, prefix.columns], prefix, rtol=1e-7, atol=1e-9)
            job["prefix_verification"] = "passed"
        job["status"] = "complete"
        atomic(out/"recovery.json", recovery)
        print(f"DONE {job['dataset']} seed={job['seed']} val={result['val_rmse']:.8f}", flush=True)

    selections = []
    for dataset in BASE:
        folder = out/dataset
        if (folder/"selected.json").exists():
            selections.append(read(folder/"selected.json"))
            continue
        records = read(folder/"validation_records.json")
        by_key = {(r["lambda_ode"], r["seed"]): r for r in records}
        for result_path in (folder/"trials").glob("*/*/result.json"):
            result = read(result_path)
            assert result["status"] == "complete" and result["test_evaluations"] == 0
            by_key[result["lambda_ode"], result["seed"]] = result
        records = list(by_key.values())
        atomic(folder/"validation_records.json", records)
        groups = {}
        for row in records:
            groups.setdefault(row["lambda_ode"], {})[row["seed"]] = row
        ranking = []
        for value, by_seed in groups.items():
            if set(by_seed) == {1, 2, 3}:
                vals = [by_seed[s]["val_rmse"] for s in (1, 2, 3)]
                ranking.append(dict(lambda_ode=value, val_mean=float(np.mean(vals)), val_sd=float(np.std(vals, ddof=1))))
        ranking.sort(key=lambda r: (r["val_mean"], r["val_sd"], r["lambda_ode"]))
        positive = next(r for r in ranking if r["lambda_ode"] > 0).copy()
        atomic(folder/"coefficient_selected.json", dict(selected_positive=positive, overall=ranking[0], ranking=ranking))
        positive["checkpoints"] = {str(seed): dict(path=groups[positive["lambda_ode"]][seed]["checkpoint"],
            sha256=groups[positive["lambda_ode"]][seed]["checkpoint_sha256"]) for seed in (1, 2, 3)}
        for record in positive["checkpoints"].values():
            assert record["path"] and sha(ROOT/record["path"]) == record["sha256"]
        selected = dict(dataset=dataset, selected_positive=positive, selected_overall=ranking[0], ranking=ranking,
            frozen_unix=time.time(), new_candidate_test_evaluations=0,
            positive_beats_zero_on_validation=positive["val_mean"] < next(r["val_mean"] for r in ranking if r["lambda_ode"] == 0))
        atomic(folder/"selected.json", selected)
        atomic(folder/"phase.json", dict(phase="selection_frozen", selected_positive=positive["lambda_ode"]))
        selections.append(selected)
        print(f"FROZEN {dataset}: positive lambda={positive['lambda_ode']:g}; overall={ranking[0]['lambda_ode']:g}", flush=True)
    verify_sources()
    atomic(out/"all_selections_frozen.json", dict(frozen_unix=time.time(), datasets=selections,
        new_candidate_test_evaluations=0, recovery_manifest_sha256=sha(out/"recovery.json")))
    for dataset in BASE:
        for seed in (1, 2, 3):
            destination = out/dataset/"final"/f"seed{seed}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(ROOT/"experiments/lambda_ode_trial.py"), "evaluate",
                "--dataset", dataset, "--seed", str(seed), "--selection", str(out/dataset/"selected.json"),
                "--output", str(destination)]
            invoke(command, protocol["gpu_assignment"][dataset], out/"logs"/f"final_{dataset}_seed{seed}.log")
        atomic(out/dataset/"phase.json", dict(phase="complete"))
    atomic(out/"recovery_completed.json", dict(status="complete", finished_unix=time.time(),
        recovery_manifest_sha256=sha(out/"recovery.json"), prefix_verification="passed for each interrupted attempt"))
    atomic(out/"completed.json", dict(status="complete", finished_unix=time.time(),
        seconds=time.time()-protocol["started_unix"], selected_final_test_evaluations=9, recovered_after_interruption=True))
    print("LAMBDA SEARCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
