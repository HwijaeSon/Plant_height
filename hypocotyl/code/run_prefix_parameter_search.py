"""Search lambda_ode on validation, then evaluate a fixed PhytoODE across missingness."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from queue import Queue, Empty
import shutil
import subprocess
import sys
import threading
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'configs/hypocotyl_prefix_parameters_20260911.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen record changed: {path}')
    else:
        path.write_text(json.dumps(value, indent=2) + '\n')


def lambda_tag(value):
    return f'{value:g}'.replace('.', 'p')


def run_jobs(jobs, output, gpus, workers):
    queue = Queue()
    for job in jobs:
        dest = output / job['path']
        if (dest / 'result.json').exists():
            result = json.loads((dest / 'result.json').read_text())
            if result['status'] == 'trained_validation_only':
                assert result['test_evaluations'] == 0
                continue
        if dest.exists():
            raise FileExistsError(f'Inspect incomplete run before restarting: {dest}')
        queue.put(job)
    failures = []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                return
            dest = output / job['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(ROOT / 'hypocotyl/code/prefix_parameter_trial.py'), 'train',
                '--lambda-ode', str(job['lambda_ode']), '--seed', str(job['seed']),
                '--drop-fraction', str(job['drop_fraction']), '--device', 'cuda:0', '--output', str(dest)]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
            print(f"START {job['path']} GPU={gpu}", flush=True)
            with (dest.parent / (dest.name + '.log')).open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            print(f"FINISH {job['path']} code={result.returncode}", flush=True)
            if result.returncode:
                with lock:
                    failures.append(job)

    with ThreadPoolExecutor(max_workers=len(gpus)*workers) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus for _ in range(workers)]
        for future in futures:
            future.result()
    if failures:
        raise RuntimeError(f'Failed runs: {failures}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--workers-per-gpu', type=int, default=2)
    p.add_argument('--output', type=Path, default=ROOT / 'hypocotyl/results/prefix_parameters_20260911')
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    profile = json.loads(PROFILE.read_text())
    sources = ['configs/hypocotyl_prefix_parameters_20260911.json',
        'hypocotyl/code/prefix_parameter_model.py', 'hypocotyl/code/prefix_parameter_trial.py',
        'hypocotyl/code/run_prefix_parameter_search.py', 'hypocotyl/code/forecast_data.py',
        'hypocotyl/code/forecast_models.py', 'hypocotyl/code/forecast_trial.py',
        'hypocotyl/code/single_condition_models.py', 'wheat/code/model.py',
        'hypocotyl/data/processed/prefix_forecast_20260911/config.json']
    jobs = [dict(lambda_ode=value, seed=seed, drop_fraction=0.,
                 path=f'search/lambda_{lambda_tag(value)}/seed{seed}')
            for value in profile['lambda_ode_grid'] for seed in profile['seeds']]
    plan = dict(profile=profile, source_sha256={name: sha(ROOT/name) for name in sources},
                search_jobs=jobs, n_search_fits=len(jobs), n_transfer_fits=6,
                selection_and_test_policy=profile['test_policy'])
    frozen(args.output / 'plan.json', plan)
    run_jobs(jobs, args.output, args.gpus, args.workers_per_gpu)
    candidates = []
    for value in profile['lambda_ode_grid']:
        runs = [json.loads((args.output / f'search/lambda_{lambda_tag(value)}/seed{seed}/result.json').read_text())
                for seed in profile['seeds']]
        assert all(r['test_evaluations'] == 0 and set(r['metrics']) == {'train', 'val'} for r in runs)
        vals = [r['metrics']['val']['rmse'] for r in runs]
        candidates.append(dict(lambda_ode=value, lambda_k=0., val_rmse_mean=float(np.mean(vals)),
            val_rmse_sd=float(np.std(vals, ddof=1)), seeds=profile['seeds'],
            validation_rmse_by_seed=vals, best_epochs=[r['best_epoch'] for r in runs]))
    candidates.sort(key=lambda row: (row['val_rmse_mean'], row['lambda_ode']))
    winner = candidates[0]
    chosen = winner['lambda_ode']
    selection = dict(selected_lambda_ode=chosen, lambda_k=0., selection_drop_fraction=0.,
        candidate_selection=profile['candidate_selection'], candidates=candidates,
        test_evaluations_before_selection=0, plan_sha256=sha(args.output/'plan.json'))
    frozen(args.output / 'lambda_selection.json', selection)
    pd.DataFrame(candidates).to_csv(args.output / 'validation_search.csv', index=False)
    print(f"SELECTED lambda_ode={chosen} validation={winner['val_rmse_mean']:.6f}", flush=True)
    transfer = [dict(lambda_ode=chosen, seed=seed, drop_fraction=drop,
                     path=f'transfer/drop_{round(drop*100)}/seed{seed}')
                for drop in [.25, .5] for seed in profile['seeds']]
    run_jobs(transfer, args.output, args.gpus, args.workers_per_gpu)
    test_jobs = []
    for drop in profile['evaluation_drop_fractions']:
        for seed in profile['seeds']:
            source = (f'search/lambda_{lambda_tag(chosen)}/seed{seed}' if drop == 0
                      else f'transfer/drop_{round(drop*100)}/seed{seed}')
            checkpoint = args.output / source / 'checkpoint.pt'
            result = json.loads((checkpoint.parent/'result.json').read_text())
            assert result['test_evaluations'] == 0 and set(result['metrics']) == {'train', 'val'}
            test_jobs.append(dict(drop_fraction=drop, seed=seed, training_path=source,
                checkpoint_sha256=sha(checkpoint), output=f'selected/drop_{round(drop*100)}/seed{seed}'))
    final_plan = dict(lambda_selection_sha256=sha(args.output/'lambda_selection.json'), jobs=test_jobs,
        note='All nine checkpoints and the coefficient are frozen before test scoring.')
    frozen(args.output / 'test_evaluation_plan.json', final_plan)
    # Import only after all candidate training and selection are complete.
    from prefix_parameter_trial import evaluate_checkpoint
    for job in test_jobs:
        training = args.output / job['training_path']
        dest = args.output / job['output']
        if (dest / 'result.json').exists():
            result = json.loads((dest/'result.json').read_text())
            assert result['checkpoint_sha256'] == job['checkpoint_sha256']
            continue
        evaluate_checkpoint(training/'checkpoint.pt', dest, device='cpu',
                            selection_record=sha(args.output/'test_evaluation_plan.json'))
        for name in ['checkpoint.pt', 'config.json', 'history.csv']:
            shutil.copy2(training/name, dest/name)
        shutil.copy2(training/'selection.json', dest/'checkpoint_selection.json')
        shutil.copy2(training/'result.json', dest/'training_result.json')
        print(f"EVALUATED {job['output']}", flush=True)
    frozen(args.output/'completion.json', dict(search_fits=len(jobs), transfer_fits=len(transfer),
        selected_test_evaluations=len(test_jobs), selected_lambda_ode=chosen, lambda_k=0., failures=[]))


if __name__ == '__main__':
    main()
