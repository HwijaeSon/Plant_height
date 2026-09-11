"""Expand the validation search while reusing the preceding frozen candidates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Queue, Empty
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from run_no_light_prefix_search import frozen, sha, lambda_tag

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT/'configs/hypocotyl_expanded_no_light_lambda_20260911.json'
PREVIOUS = ROOT/'hypocotyl/results/prefix_parameters_no_light_20260911'
RESULTS = ROOT/'hypocotyl/results/expanded_no_light_lambda_20260911'


def location(output, job):
    return (ROOT if job['source']=='previous' else output)/job['path']


def jobs_for(values, drops, seeds):
    return [dict(lambda_ode=value, seed=seed, drop_fraction=drop, source='current',
                 path=(f'search/lambda_{lambda_tag(value)}/seed{seed}' if drop==0 else
                       f'transfer/lambda_{lambda_tag(value)}/drop_{round(100*drop)}/seed{seed}'))
            for value in values for drop in drops for seed in seeds]


def run_jobs(jobs, output, gpus, workers):
    pending = Queue()
    for job in jobs:
        folder = location(output, job)
        if (folder/'result.json').exists():
            r = json.loads((folder/'result.json').read_text())
            assert r['status']=='trained_validation_only' and r['test_evaluations']==0
            assert r['config']['lambda_ode']==job['lambda_ode']
            continue
        if folder.exists():
            raise FileExistsError(f'Inspect incomplete training output: {folder}')
        pending.put(job)
    def worker(gpu):
        failures = []
        while True:
            try:
                job = pending.get_nowait()
            except Empty:
                return failures
            folder = location(output, job)
            folder.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(ROOT/'hypocotyl/code/expanded_no_light_trial.py'), 'train',
                       '--lambda-ode', str(job['lambda_ode']), '--seed', str(job['seed']),
                       '--drop-fraction', str(job['drop_fraction']), '--device', 'cuda:0', '--output', str(folder)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
            print(f"START {job['path']} GPU={gpu}", flush=True)
            with (folder.parent/(folder.name+'.log')).open('w') as log:
                fit = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f"FINISH {job['path']} code={fit.returncode}", flush=True)
            if fit.returncode:
                failures.append(job)
    with ThreadPoolExecutor(max_workers=len(gpus)*workers) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus for _ in range(workers)]
        failures = [job for future in futures for job in future.result()]
    if failures:
        raise RuntimeError(f'Failed fits: {failures}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', type=int, nargs='+', default=[1, 2])
    p.add_argument('--workers-per-gpu', type=int, default=2)
    p.add_argument('--output', type=Path, default=RESULTS)
    args = p.parse_args()
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(PROFILE.read_text())
    old_plan = json.loads((PREVIOUS/'plan.json').read_text())
    for name, digest in old_plan['source_sha256'].items():
        assert sha(ROOT/name)==digest, name
    sources = list(old_plan['source_sha256'])+[
        'configs/hypocotyl_expanded_no_light_lambda_20260911.json',
        'hypocotyl/code/expanded_no_light_trial.py', 'hypocotyl/code/run_expanded_no_light_search.py']
    added = jobs_for(cfg['added_lambda_ode_grid'], [0.], cfg['seeds'])
    reused = []
    reused_hashes = {}
    for job in jobs_for(cfg['reused_lambda_ode_grid'], [0.], cfg['seeds']):
        folder = PREVIOUS/job['path']
        r = json.loads((folder/'result.json').read_text())
        assert r['status']=='trained_validation_only' and r['test_evaluations']==0
        assert set(r['metrics'])=={'train','val'}
        reused.append(dict(job, source='previous', path=str(folder.relative_to(ROOT))))
        for name in ['result.json','checkpoint.pt','config.json','history.csv','predictions.npz','selection.json']:
            f=folder/name;reused_hashes[str(f.relative_to(ROOT))]=sha(f)
    plan = dict(profile=cfg, source_sha256={f:sha(ROOT/f) for f in sources},
                added_search_jobs=added, reused_search_jobs=reused, reused_files_sha256=reused_hashes)
    frozen(output/'plan.json', plan)
    run_jobs(added, output, args.gpus, args.workers_per_gpu)
    candidates = []
    for value in cfg['lambda_ode_grid']:
        jobs = [job for job in reused+added if job['lambda_ode']==value]
        runs = [json.loads((location(output,job)/'result.json').read_text()) for job in jobs]
        assert len(runs)==3 and all(r['test_evaluations']==0 and set(r['metrics'])=={'train','val'} for r in runs)
        vals = [r['metrics']['val']['rmse'] for r in runs]
        candidates.append(dict(lambda_ode=value, lambda_k=0., val_rmse_mean=float(np.mean(vals)),
            val_rmse_sd=float(np.std(vals,ddof=1)), seeds=cfg['seeds'],
            validation_rmse_by_seed=vals, best_epochs=[r['best_epoch'] for r in runs],
            origin=jobs[0]['source']))
    candidates.sort(key=lambda r:(r['val_rmse_mean'],r['lambda_ode']))
    chosen = candidates[0]['lambda_ode']
    frozen(output/'lambda_selection.json', dict(selected_lambda_ode=chosen, lambda_k=0.,
        candidates=candidates, candidate_selection=cfg['candidate_selection'],
        plan_sha256=sha(output/'plan.json'), new_test_evaluations_before_selection=0,
        prior_test_exposure='The prior coefficient-100 results were reported before this expanded validation-only search.'))
    pd.DataFrame(candidates).to_csv(output/'validation_search.csv', index=False)
    print(f"SELECTED lambda_ode={chosen} validation={candidates[0]['val_rmse_mean']:.6f}", flush=True)
    transfer = [] if chosen==100 else jobs_for([chosen], [.25,.5], cfg['seeds'])
    frozen(output/'transfer_plan.json', dict(selected_lambda_ode=chosen, jobs=transfer,
        reuse_previous_coefficient100=chosen==100))
    run_jobs(transfer, output, args.gpus, args.workers_per_gpu)
    test_jobs = []
    for drop in cfg['evaluation_drop_fractions']:
        for seed in cfg['seeds']:
            if drop==0:
                job = next(j for j in reused+added if j['lambda_ode']==chosen and j['seed']==seed)
            elif chosen==100:
                path = PREVIOUS/f'transfer/lambda_100/drop_{round(100*drop)}/seed{seed}'
                job = dict(source='previous', path=str(path.relative_to(ROOT)), lambda_ode=chosen,
                           seed=seed, drop_fraction=drop)
            else:
                job = next(j for j in transfer if j['seed']==seed and j['drop_fraction']==drop)
            folder = location(output, job)
            entry = dict(training=job, checkpoint_sha256=sha(folder/'checkpoint.pt'),
                         output=f'selected/drop_{round(100*drop)}/seed{seed}')
            if chosen==100:
                previous_eval = PREVIOUS/f'evaluated/lambda_100/drop_{round(100*drop)}/seed{seed}'
                entry['previous_evaluation'] = str(previous_eval.relative_to(ROOT))
                entry['previous_result_sha256'] = sha(previous_eval/'result.json')
            test_jobs.append(entry)
    frozen(output/'test_evaluation_plan.json', dict(jobs=test_jobs,
        lambda_selection_sha256=sha(output/'lambda_selection.json'),
        note='All selected checkpoint hashes fixed before any new test evaluation; already reported coefficient-100 evaluations are reused if selected again.'))
    from no_light_prefix_trial import evaluate_checkpoint
    for job in test_jobs:
        dest = output/job['output'];training = location(output,job['training'])
        if (dest/'result.json').exists():
            assert json.loads((dest/'result.json').read_text())['checkpoint_sha256']==job['checkpoint_sha256']
            continue
        if 'previous_evaluation' in job:
            shutil.copytree(ROOT/job['previous_evaluation'], dest)
        else:
            evaluate_checkpoint(training/'checkpoint.pt', dest, device='cpu',
                selection_record=sha(output/'test_evaluation_plan.json'))
            for name in ['checkpoint.pt','config.json','history.csv']:
                shutil.copy2(training/name,dest/name)
            shutil.copy2(training/'selection.json',dest/'checkpoint_selection.json')
            shutil.copy2(training/'result.json',dest/'training_result.json')
        print(f"FINALIZED {job['output']}", flush=True)
    frozen(output/'completion.json', dict(added_search_fits=len(added), reused_search_fits=len(reused),
        total_candidates=len(candidates), transfer_fits=len(transfer), selected_lambda_ode=chosen, lambda_k=0.,
        new_test_evaluations=0 if chosen==100 else 9, reused_test_evaluations=9 if chosen==100 else 0, failures=[]))


if __name__ == '__main__':
    main()
