"""Repeat the initial search protocol on 12L12D-only means without environmental inputs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'hypocotyl/results/single_condition_20260909'
MODELS = ['phytoode', 'latent_ode', 'logistic_pinn', 'lstm', 'rf', 'logistic']
PROTOCOLS = ['replicate', 'time_holdout']


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates():
    catalog={}
    for model in MODELS:
        if model in ['phytoode','logistic_pinn']:
            configs=[dict(model=model,lr=lr,lambda_ode=ode,lambda_k=k,epochs=1500,weight_decay=1e-4)
                     for lr,ode,k in itertools.product([.003,.01],[.5,5.,50.,500.],[.01,.1])]
        elif model in ['latent_ode','lstm']:
            configs=[dict(model=model,lr=lr,lambda_ode=0.,lambda_k=0.,epochs=1500,weight_decay=1e-4)
                     for lr in [.003,.01]]
        elif model=='rf': configs=[dict(model=model,min_samples_leaf=k) for k in [1,2,4]]
        else: configs=[dict(model=model)]
        for i,config in enumerate(configs): catalog[f'{model}_{i:02d}']=config
    assert len(catalog)==40
    return catalog


def initialize_worker(gpu_queue):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu_queue.get()), OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')


def train_worker(job):
    from single_condition_trial import train
    return train(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs=2, default=[8,9])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists() and not args.resume: raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); (out/'configs').mkdir(exist_ok=True)
    catalog = candidates()
    sources = [ROOT/'hypocotyl/code'/name for name in [
        'prepare_single_condition.py','single_condition_data.py','single_condition_models.py',
        'single_condition_baselines.py','single_condition_trial.py','run_single_condition.py']]
    sources += [ROOT/'wheat/code/model.py']
    sources += sorted(p for p in (ROOT/'hypocotyl/data/processed/single_condition_20260909').rglob('*') if p.is_file())
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        assert protocol['source_sha256'] == hashes and protocol['candidates'] == catalog
    else:
        protocol = dict(started_unix=time.time(), gpus=args.gpus, workers_per_gpu=2,
            protocols=PROTOCOLS, models=MODELS, seeds=[1,2,3], candidates=catalog,
            condition='12L12D',excluded_condition='cR',input_features=['genotype','elapsed_time'],
            environmental_feature_count=0, target_type='split_specific_replicate_means_12L12D_only',
            loss='Mean of five genotype RMSEs against training replicate means; absent times have no target.',
            metric='Mean genotype RMSE / pooled mean of scored split-specific replicate means * 100; raw individual RMSE is secondary.',
            selection='Initial-version budget: 40 seed-1 candidates per protocol; confirm top 2 per stochastic family with seeds 2 and 3, then select three-seed mean validation RMSE.',
            matched_control='Same selected PhytoODE learning rate, architecture, initial weights, schedule and checkpoint rule; both physics coefficients zero.',
            stopping='Fixed 80 screening + 40 confirmation fits. Matched controls reuse already confirmed zero-loss fits. No extension after test scoring.',
            fresh_fits=True, reused_checkpoints=False, prior_test_inspected=True,
            test_access='Both protocols and all models freeze before new scoring. The original previously inspected partitions are filtered, not reassigned.',
            source_sha256=hashes)
        atomic(protocol_path, protocol)
    for key,config in catalog.items():
        path = out/'configs'/f'{key}.json'
        if path.exists(): assert json.loads(path.read_text()) == config
        else: atomic(path, config)

    def folder(p,k,s): return out/p/'trials'/k/f'seed{s}'
    def result(p,k,s): return json.loads((folder(p,k,s)/'result.json').read_text())
    ctx = multiprocessing.get_context('spawn'); gpu_queue = ctx.Queue()
    for gpu in args.gpus:
        for _ in range(2): gpu_queue.put(gpu)
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx, initializer=initialize_worker,
                             initargs=(gpu_queue,)) as pool:
        def execute(jobs, stage):
            pending = {}; done = 0
            for p,k,s in jobs:
                destination = folder(p,k,s)
                if (destination/'result.json').exists(): done += 1; continue
                if destination.exists():
                    diagnostic = out/'diagnostics'/f'{p}_{k}_{s}_{time.time_ns()}'
                    diagnostic.parent.mkdir(exist_ok=True); destination.rename(diagnostic)
                job = dict(protocol=p, key=k, seed=s, config_path=str(out/'configs'/f'{k}.json'), output=str(destination))
                pending[pool.submit(train_worker,job)] = job
            while pending:
                finished,_ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = pending.pop(future)
                    try: value = future.result()
                    except BaseException as error:
                        atomic(out/'failure.json', dict(stage=stage, job=job, error=repr(error)))
                        raise
                    done += 1
                    print(f'{stage} {done}/{len(jobs)}: {value}', flush=True)
                atomic(out/'status.json', dict(stage=stage, completed=done, total=len(jobs),
                    remaining=len(pending), elapsed_seconds=time.time()-protocol['started_unix']))
        # Alternate protocols within the initial-version candidate catalog.
        execute([(p,k,1) for k in catalog for p in PROTOCOLS], 'screening')
        finalists_path = out/'finalists.json'
        if finalists_path.exists(): finalists = json.loads(finalists_path.read_text())
        else:
            finalists = {}
            for p in PROTOCOLS:
                finalists[p] = {}
                for model in MODELS:
                    keys = [k for k,c in catalog.items() if c['model'] == model]
                    keys.sort(key=lambda k:(result(p,k,1)['metrics']['val']['rmse'],k))
                    top = 2
                    finalists[p][model] = keys[:top]
            atomic(finalists_path, finalists)
        jobs = [(p,k,s) for p in PROTOCOLS for m in MODELS if m not in ['logistic']
                for k in finalists[p][m] for s in [2,3]]
        assert len(jobs) == 40
        execute(jobs, 'confirmation')
        selected_path = out/'selected_configs.json'
        if selected_path.exists(): selected = json.loads(selected_path.read_text())
        else:
            selected = {}
            for p in PROTOCOLS:
                selected[p] = {}
                for m in MODELS:
                    seeds = [1] if m in ['logistic'] else [1,2,3]
                    ranking = []
                    for k in finalists[p][m]:
                        values = [result(p,k,s)['metrics']['val']['rmse'] for s in seeds]
                        ranking.append(dict(candidate=k, val_mean=sum(values)/len(values), per_seed_validation=values))
                    ranking.sort(key=lambda r:(r['val_mean'],r['candidate']))
                    key = ranking[0]['candidate']
                    selected[p][m] = dict(config=catalog[key], selected_candidate=key, ranking=ranking)
            atomic(selected_path, selected)
        matched_jobs = []
        for p in PROTOCOLS:
            full_config = selected[p]['phytoode']['config']
            matched = full_config | dict(model='latent_ode', lambda_ode=0., lambda_k=0.)
            key = next(k for k,c in catalog.items() if c == matched)
            selected[p]['latent_ode_matched'] = dict(config=matched, selected_candidate=key,
                selection_basis='All selected PhytoODE settings with both physics coefficients zero.')
            matched_jobs.extend((p,key,s) for s in [1,2,3])
        execute(matched_jobs, 'matched_control')
    runs = []
    for p in PROTOCOLS:
        for m,choice in selected[p].items():
            seeds = [1] if m in ['logistic'] else [1,2,3]
            for s in seeds:
                source = folder(p, choice['selected_candidate'], s)
                runs.append(dict(protocol=p, model=m, seed=s, path=str(source.relative_to(ROOT)),
                                 result_sha256=sha(source/'result.json')))
        for s in [1,2,3]:
            full = result(p,selected[p]['phytoode']['selected_candidate'],s)
            pure = result(p,selected[p]['latent_ode_matched']['selected_candidate'],s)
            assert full['initial_state_sha256'] == pure['initial_state_sha256']
    selection_path = out/'selections.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        assert selection['runs'] == runs and selection['selected'] == selected
    else:
        selection = dict(status='all_validation_selections_frozen', frozen_unix=time.time(),
            target_type='split_specific_replicate_means_12L12D_only', prior_test_inspected=True, selected=selected, runs=runs)
        atomic(selection_path, selection)
    assert hashes == {p:sha(ROOT/p) for p in hashes}
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    from single_condition_trial import evaluate
    for record in runs:
        destination = out/record['protocol']/'final'/record['model']/f"seed{record['seed']}"
        if not (destination/'result.json').exists(): evaluate(selection_path, record, destination)
    fits = len(list(out.glob('*/trials/*/seed*/result.json')))
    atomic(out/'completed.json', dict(status='complete', seconds=time.time()-protocol['started_unix'],
        full_training_trials=fits, final_evaluations=len(runs), selection_sha256=sha(selection_path)))
    atomic(out/'status.json', dict(stage='complete', elapsed_seconds=time.time()-protocol['started_unix']))
    print(f'Completed {fits} fresh fits and {len(runs)} frozen-choice evaluations.', flush=True)


if __name__ == '__main__': main()
