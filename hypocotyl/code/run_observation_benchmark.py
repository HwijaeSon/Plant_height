"""Fixed-budget, paired-capacity validation searches using unaveraged observations."""
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
OUT = ROOT/'hypocotyl/results/individual_observations_20260909'
MODELS = ['phytoode', 'latent_ode', 'light_pinn', 'lstm', 'rf', 'light_logistic', 'logistic']
PROTOCOLS = ['replicate', 'time_holdout']


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates():
    arches = [dict(latent_dim=d, g_embed_dim=4, ode_hidden=w, ode_layers=1,
                   dec_hidden=w, enc_hidden=d) for d,w in [(8,16),(12,24),(16,32)]]
    bases = [dict(lr=lr, weight_decay=wd, epochs=2000, architecture=arch)
             for arch,lr,wd in itertools.product(arches, [.001,.003,.006,.01], [1e-4,1e-3])]
    # Include the preceding validation-selected architecture/optimizer as one
    # predeclared starting point; every model is nevertheless fitted from scratch.
    bases[0] = dict(lr=.0015527014849295356, weight_decay=.00036574216923513037,
                    epochs=2000, architecture=arches[1])
    rng = random.Random(20260910)
    ode = [10**(-2+(4*(i+rng.random())/24)) for i in range(24)]
    capacity = [10**(-5+(4*(i+rng.random())/24)) for i in range(24)]
    rng.shuffle(ode); rng.shuffle(capacity)
    ode[0] = .12644235454360606; capacity[0] = .002702328745365828
    catalog = {}
    for i,base in enumerate(bases):
        catalog[f'phytoode_{i:02d}'] = dict(model='phytoode', **base, lambda_ode=ode[i], lambda_k=capacity[i])
        catalog[f'latent_ode_{i:02d}'] = dict(model='latent_ode', **base, lambda_ode=0., lambda_k=0.)
    for i,(lr,(ode,k)) in enumerate(itertools.product([.001,.003,.01], [(.05,.0001),(.5,.003),(5.,.01),(50.,.1)])):
        catalog[f'light_pinn_{i:02d}'] = dict(model='light_pinn', lr=lr, lambda_ode=ode,
                                            lambda_k=k, epochs=2000, weight_decay=1e-4)
    for i,lr in enumerate([.001,.003,.006,.01]):
        catalog[f'lstm_{i:02d}'] = dict(model='lstm', lr=lr, lambda_ode=0., lambda_k=0.,
                                      epochs=2000, weight_decay=1e-4)
    for i,leaf in enumerate([1,2,4,8]):
        catalog[f'rf_{i:02d}'] = dict(model='rf', min_samples_leaf=leaf)
    for m in ['logistic','light_logistic']: catalog[m+'_00'] = dict(model=m)
    assert len(catalog) == 70
    return catalog


def initialize_worker(gpu_queue):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu_queue.get()), OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')


def train_worker(job):
    from observation_trial import train
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
        'observation_data.py','observation_baselines.py','observation_trial.py','run_observation_benchmark.py',
        'models.py','retune_model.py','data.py','process_baselines.py','trial.py']]
    sources += [ROOT/'wheat/code/model.py']
    sources += sorted((ROOT/'hypocotyl/data/processed/splits').rglob('*.csv'))
    sources += sorted((ROOT/'hypocotyl/data/processed/splits').rglob('*.json'))
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        assert protocol['source_sha256'] == hashes and protocol['candidates'] == catalog
    else:
        protocol = dict(started_unix=time.time(), gpus=args.gpus, workers_per_gpu=2,
            protocols=PROTOCOLS, models=MODELS, seeds=[1,2,3], candidates=catalog,
            target_type='Every individual observed measurement, no phenotype imputation or replicate averaging.',
            loss='Pooled RMSE over raw measured training cells after training-only height scaling.',
            metric='Pooled individual-observation RMSE / mean observed length * 100.',
            selection='Screen all 70 candidates at seed 1 per protocol; confirm top 3 PhytoODE and latent ODE, top 2 Light-PINN, LSTM and RF at seeds 2 and 3; select lowest three-seed validation RMSE.',
            matched_control='Final selected PhytoODE architecture and optimizer, both loss coefficients zero; three seeds with identical initialization.',
            stopping='Fixed 140 screening + 48 confirmation + at most 6 new matched-control fits. No extension after test scoring.',
            fresh_fits=True, reused_mean_target_checkpoints=False, prior_test_inspected=True,
            test_access='All models and both protocols freeze before new test scoring. Same previously inspected partitions; evaluation is not an independent new cohort.',
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
        # Alternating protocols share exactly the same architecture/optimizer catalog.
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
                    top = 3 if model in ['phytoode','latent_ode'] else 2
                    finalists[p][model] = keys[:top]
            atomic(finalists_path, finalists)
        jobs = [(p,k,s) for p in PROTOCOLS for m in MODELS if m not in ['logistic','light_logistic']
                for k in finalists[p][m] for s in [2,3]]
        assert len(jobs) == 48
        execute(jobs, 'confirmation')
        selected_path = out/'selected_configs.json'
        if selected_path.exists(): selected = json.loads(selected_path.read_text())
        else:
            selected = {}
            for p in PROTOCOLS:
                selected[p] = {}
                for m in MODELS:
                    seeds = [1] if m in ['logistic','light_logistic'] else [1,2,3]
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
            seeds = [1] if m in ['logistic','light_logistic'] else [1,2,3]
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
            target_type='individual_observed_cells', prior_test_inspected=True, selected=selected, runs=runs)
        atomic(selection_path, selection)
    assert hashes == {p:sha(ROOT/p) for p in hashes}
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    from observation_trial import evaluate
    for record in runs:
        destination = out/record['protocol']/'final'/record['model']/f"seed{record['seed']}"
        if not (destination/'result.json').exists(): evaluate(selection_path, record, destination)
    fits = len(list(out.glob('*/trials/*/seed*/result.json')))
    atomic(out/'completed.json', dict(status='complete', seconds=time.time()-protocol['started_unix'],
        full_training_trials=fits, final_evaluations=len(runs), selection_sha256=sha(selection_path)))
    atomic(out/'status.json', dict(stage='complete', elapsed_seconds=time.time()-protocol['started_unix']))
    print(f'Completed {fits} fresh fits and {len(runs)} frozen-choice evaluations.', flush=True)


if __name__ == '__main__': main()
