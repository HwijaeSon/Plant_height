"""Time-bounded, three-GPU search with validation-only ranking and a frozen final choice."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from tune_trial import BASE, ROOT, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/tuning_20260904")
    parser.add_argument("--gpus", nargs=3, default=["0", "1", "2"])
    parser.add_argument("--minutes", type=float, default=200.)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"protocol.json").exists():
        raise FileExistsError("Search output already initialized; use a fresh directory")
    started = time.time()
    source_paths = [Path(__file__), ROOT/"code/tune_trial.py", ROOT/"code/data.py", ROOT/"code/run_experiment.py",
        ROOT.parent/"wheat/code/model.py", ROOT.parent/"arabidopsis/code/models.py",
        ROOT/"data/processed/height_observations.csv", ROOT/"data/processed/weather_daily.csv"]
    source_hashes = {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    gpu_lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    gpu_map = {}
    for line in gpu_lines.splitlines():
        index, uuid, used = [p.strip() for p in line.split(",")]
        if index in args.gpus:
            if int(used) > 1024:
                raise RuntimeError(f"Requested GPU {index} is in use ({used} MiB)")
            gpu_map[index] = uuid
    if len(gpu_map) != 3:
        raise ValueError("Three distinct available GPUs are required")

    rng = random.Random(20260904)
    configs, signatures = {}, set()
    def add_config(config, prefix="c"):
        signature = json.dumps(config, sort_keys=True)
        if signature in signatures:
            return None
        name = f"{prefix}{len(configs):03d}"
        configs[name] = config.copy()
        signatures.add(signature)
        (out/"configs").mkdir(exist_ok=True)
        atomic_json(out/"configs"/f"{name}.json", config)
        return name
    add_config(BASE)
    for lr in [.001, .003, .005, .01]:
        for physic in [0., .1, .5, 2.]:
            add_config(BASE | dict(lr=lr, physic=physic))
    space = dict(latent_dim=[8, 16, 32], g_embed_dim=[4, 8, 16], ode_hidden=[16, 32, 64],
        ode_layers=[1, 2], dec_hidden=[16, 32], enc_hidden=[8, 16, 32],
        time_mode=["calendar", "thermal_feature", "thermal_rate"],
        lr=[.0005, .001, .003, .005, .01], weight_decay=[0., 1e-6, 1e-5, 1e-4, 1e-3],
        physic=[0., .01, .1, .5, 2., 5.], ymax=[0., .01, .1, .5], mono=[0.])
    while len(configs) < 160:
        add_config(BASE | {key: rng.choice(values) for key, values in space.items()})
    protocol = dict(started_unix=started, deadline_unix=started+args.minutes*60, minutes=args.minutes,
        gpu_indices=args.gpus, gpu_uuids=gpu_map, split="chronological", train_years=[2018, 2019], val_year=2020, test_year=2021,
        phases=[dict(name="screen", minutes=90, seeds=[1], epochs=1500, max_candidates=160),
                dict(name="replicate", minutes=50, top_candidates=8, seeds=[1, 2, 3]),
                dict(name="longer", minutes=45, top_candidates=3, epochs=[3000, 4500], seeds=[1, 2, 3]),
                dict(name="final", minutes=15)],
        ranking="mean validation RMSE over seeds 1,2,3; ties use SD then parameter count",
        checkpoint="minimum validation RMSE every 10 epochs, no smoothing or minimum-epoch floor",
        test_usage="no candidate test scores; only the frozen final winner is evaluated",
        search_space=space, base_config=BASE, source_sha256=source_hashes)
    atomic_json(out/"protocol.json", protocol)
    active, completed, failures, skipped = {}, [], [], []
    stopped = False
    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
        for state in active.values():
            state["proc"].terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def ranking(required=1):
        groups = {}
        for row in completed:
            groups.setdefault(row["candidate"], []).append(row)
        ranking_rows = []
        for name, rows in groups.items():
            selected = [r for r in rows if r["seed"] in ([1] if required == 1 else [1, 2, 3])]
            if len(selected) != required:
                continue
            values = [r["val_rmse"] for r in selected]
            mean = sum(values)/len(values)
            sd = (sum((v-mean)**2 for v in values)/(len(values)-1))**.5 if len(values)>1 else 0.
            ranking_rows.append(dict(candidate=name, val_mean=mean, val_sd=sd, seeds=sorted(r["seed"] for r in selected),
                n_params=rows[0]["n_params"], config=configs[name]))
        return sorted(ranking_rows, key=lambda r: (r["val_mean"], r["val_sd"], r["n_params"]))

    def update_status(phase):
        running = []
        for gpu, state in active.items():
            progress_path = state["folder"]/"progress.json"
            try:
                progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            except (OSError, json.JSONDecodeError):
                progress = {}
            running.append(dict(gpu=gpu, candidate=state["name"], seed=state["seed"], pid=state["proc"].pid) | progress)
        atomic_json(out/"status.json", dict(phase=phase, elapsed_minutes=(time.time()-started)/60,
            deadline_unix=protocol["deadline_unix"], completed_runs=len(completed), failed_runs=len(failures),
            skipped_runs=len(skipped), active=running, best_single_seed=ranking(1)[:5], best_three_seed=ranking(3)[:5], test_evaluations=0))
        atomic_json(out/"completed.json", completed)
        atomic_json(out/"failures.json", failures)

    def estimate(name):
        c = configs[name]
        # Use measured normalized throughput; guard large candidates at phase boundaries.
        rate = sum(r["seconds"]/r["epochs"] for r in completed)/len(completed) if completed else .3
        return max(80., rate*c["epochs"]*1.35)

    def run_jobs(jobs, phase, end_minute):
        queue = list(jobs)
        end = started + end_minute*60
        while (queue or active) and not stopped:
            now = time.time()
            if now >= protocol["deadline_unix"]-180:
                stop(None, None)
                break
            for gpu in args.gpus:
                if gpu in active or not queue:
                    continue
                launch_index = next((i for i, (name, seed) in enumerate(queue) if now+estimate(name) < end), None)
                if launch_index is None:
                    continue
                name, seed = queue.pop(launch_index)
                folder = out/"trials"/name/f"seed{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                handle = (folder/"train.log").open("w")
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu_map[gpu], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
                proc = subprocess.Popen([sys.executable, "-u", str(ROOT/"code/tune_trial.py"),
                    "--config", str(out/"configs"/f"{name}.json"), "--seed", str(seed), "--output", str(folder)],
                    cwd=ROOT.parent, env=env, stdout=handle, stderr=subprocess.STDOUT)
                active[gpu] = dict(proc=proc, handle=handle, name=name, seed=seed, folder=folder, started=time.time())
                print(f"START {phase} {name} seed={seed} gpu={gpu}", flush=True)
            for gpu, state in list(active.items()):
                code = state["proc"].poll()
                if code is None:
                    continue
                state["handle"].close()
                del active[gpu]
                result_path = state["folder"]/"result.json"
                if code == 0 and result_path.exists():
                    result = json.loads(result_path.read_text())
                    completed.append(result | dict(candidate=state["name"], phase=phase))
                    print(f"DONE {phase} {state['name']} seed={state['seed']} val={result['val_rmse']:.5f}", flush=True)
                else:
                    failures.append(dict(candidate=state["name"], seed=state["seed"], phase=phase, exit_code=code))
                    print(f"FAILED {state['name']} seed={state['seed']} exit={code}", flush=True)
            update_status(phase)
            if not active and (not queue or all(time.time()+estimate(name) >= end for name, seed in queue)):
                break
            time.sleep(5)
        skipped.extend(dict(candidate=name, seed=seed, phase=phase, reason="phase wall budget") for name, seed in queue)
        update_status(phase)

    run_jobs([(name, 1) for name in list(configs)], "screen", min(90., args.minutes-70))
    top = ranking(1)[:8]
    if not top or stopped:
        update_status("interrupted")
        return
    atomic_json(out/"screen_selection.json", top)
    run_jobs([(row["candidate"], seed) for row in top for seed in [2, 3]], "replicate", min(140., args.minutes-45))
    top = ranking(3)[:3]
    if not top or stopped:
        update_status("incomplete")
        return
    atomic_json(out/"replicate_selection.json", top)
    longer = []
    for epochs in [3000, 4500]:
        for row in top:
            name = add_config(row["config"] | dict(epochs=epochs), prefix="l")
            if name:
                longer.extend((name, seed) for seed in [1, 2, 3])
    run_jobs(longer, "longer", args.minutes-15)
    final_ranking = ranking(3)
    if not final_ranking or stopped:
        update_status("incomplete")
        return
    for path, digest in source_hashes.items():
        if hashlib.sha256((ROOT.parent/path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed during search: {path}")
    winner = final_ranking[0]
    winner["frozen_unix"] = time.time()
    winner["selection_basis"] = "validation only, mean of seeds 1,2,3"
    winner["checkpoint_sha256"] = {str(seed): hashlib.sha256((out/"trials"/winner["candidate"]/f"seed{seed}"/"checkpoint.pt").read_bytes()).hexdigest() for seed in [1, 2, 3]}
    atomic_json(out/"selected.json", winner)
    atomic_json(out/"ranking_three_seed.json", final_ranking)
    atomic_json(out/"skipped.json", skipped)
    update_status("selection_frozen")
    print(f"FROZEN {winner['candidate']} val={winner['val_mean']:.5f} +/- {winner['val_sd']:.5f}", flush=True)
    # Reporting is a separate program and runs only after selection is irrevocably saved.
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu_map[args.gpus[0]], OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
    result = subprocess.run([sys.executable, str(ROOT/"code/tune_report.py"), "--results", str(out)],
                            cwd=ROOT.parent, env=env)
    update_status("complete" if result.returncode == 0 else "report_failed")
    status = json.loads((out/"status.json").read_text())
    status["test_evaluations"] = 3 if result.returncode == 0 else None
    status["finished_unix"] = time.time()
    atomic_json(out/"status.json", status)
    print(f"SEARCH FINISHED elapsed={(time.time()-started)/60:.1f} min report_exit={result.returncode}", flush=True)


if __name__ == "__main__":
    main()
