"""Fixed-budget, validation-selected follow-up search on the primary split only."""
from __future__ import annotations
import argparse
from collections import deque
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

from retune_model import DEFAULT_ARCHITECTURE
from trial import atomic, sha, evaluate

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'hypocotyl/results/light_growth_20260909'
OUT = ROOT/'hypocotyl/results/primary_retuning_20260909'


def candidates():
    configs = []
    for lr in [.001, .003, .006, .01]:
        for ode in [.05, .5, 5., 50.]:
            for capacity in [.0001, .003]:
                configs.append(dict(model='phytoode', lr=lr, lambda_ode=ode,
                    lambda_k=capacity, epochs=3000, weight_decay=1e-4,
                    architecture=DEFAULT_ARCHITECTURE.copy()))
    architectures = [
        dict(latent_dim=4, g_embed_dim=2, ode_hidden=8, ode_layers=1, dec_hidden=8, enc_hidden=4),
        dict(latent_dim=4, g_embed_dim=4, ode_hidden=16, ode_layers=1, dec_hidden=16, enc_hidden=8),
        DEFAULT_ARCHITECTURE.copy(),
        dict(latent_dim=12, g_embed_dim=4, ode_hidden=24, ode_layers=1, dec_hidden=24, enc_hidden=12),
        dict(latent_dim=16, g_embed_dim=4, ode_hidden=32, ode_layers=1, dec_hidden=32, enc_hidden=16),
        dict(latent_dim=8, g_embed_dim=4, ode_hidden=24, ode_layers=2, dec_hidden=24, enc_hidden=8)]
    rng = random.Random(20260909)
    # Stratified log-space coverage, specified before any new runs.
    factors = {}
    for name, lo, hi in [('lr', -3.3, -1.7), ('lambda_ode', -2.3, 2.7),
                         ('lambda_k', -5., -1.5), ('weight_decay', -6., -2.)]:
        values = [10**(lo+(hi-lo)*(i+rng.random())/48) for i in range(48)]
        rng.shuffle(values); factors[name] = values
    schedules = [2000, 3000, 4500]*16; rng.shuffle(schedules)
    for i in range(48):
        configs.append(dict(model='phytoode', **{k: v[i] for k,v in factors.items()},
            epochs=schedules[i], architecture=architectures[i % len(architectures)].copy()))
    assert len(configs) == 80
    return {f'phytoode_{i:03d}': config for i, config in enumerate(configs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs=2, type=int, default=[8, 9])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists() and not args.resume: raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    catalog = candidates()
    sources = [ROOT/'hypocotyl/code'/name for name in ['retune_model.py', 'retune_trial.py',
        'retune_primary.py', 'models.py', 'data.py', 'trial.py']] + [ROOT/'wheat/code/model.py']
    sources += list((ROOT/'hypocotyl/data/processed/splits/replicate').glob('*'))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources if p.is_file()}
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        assert protocol['source_sha256'] == hashes
        assert protocol['candidates'] == catalog
    else:
        protocol = dict(started_unix=time.time(), protocol='replicate', gpus=args.gpus,
            seeds=[1,2,3], candidates=catalog, source_sha256=hashes,
            screening='80 fresh candidates at seed 1; confirm the best 8 with seeds 2 and 3.',
            selection='Lowest mean validation RMSE across all three seeds; also retain both fully confirmed original PhytoODE candidates as eligible incumbents.',
            matched_control='Selected architecture, learning rate, weight decay, epochs, initialization and checkpoint rule, with both physics coefficients zero; no separate search.',
            stopping='Fixed 80 screening + 16 confirmation + 3 matched-control fits; do not extend or switch candidates based on test performance.',
            prior_test_inspected=True,
            interpretation='Exploratory follow-up on an already inspected test partition, not a fresh blinded confirmation. Baselines retain their original searches; compute budgets differ.',
            test_access='No new test scores until PhytoODE and matched-control configurations are frozen.')
        atomic(protocol_path, protocol)
    (out/'configs').mkdir(exist_ok=True); (out/'logs').mkdir(exist_ok=True)
    for key, config in catalog.items():
        path = out/'configs'/f'{key}.json'
        if path.exists(): assert json.loads(path.read_text()) == config
        else: atomic(path, config)
    def folder(key, seed): return out/'trials'/key/f'seed{seed}'
    def result(key, seed): return json.loads((folder(key, seed)/'result.json').read_text())
    def execute(jobs, stage):
        queue = deque(jobs); running = {}; done = 0; total = len(queue)
        while queue or running:
            for gpu in args.gpus:
                if gpu in running: continue
                while queue:
                    key, seed = queue.popleft(); target = folder(key, seed)
                    if (target/'result.json').exists(): done += 1; continue
                    if target.exists():
                        destination = out/'diagnostics'/f'{key}_seed{seed}_{time.time_ns()}'
                        destination.parent.mkdir(exist_ok=True); target.rename(destination)
                    log = (out/'logs'/f'{key}_seed{seed}.log').open('w')
                    cmd = [sys.executable, str(ROOT/'hypocotyl/code/retune_trial.py'),
                        '--config', str(out/'configs'/f'{key}.json'), '--seed', str(seed), '--output', str(target)]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                    process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    running[gpu] = (process, log, key, seed)
                    break
            for gpu, (process, log, key, seed) in list(running.items()):
                code = process.poll()
                if code is None: continue
                log.close(); del running[gpu]
                if code != 0:
                    atomic(out/'failure.json', dict(stage=stage, key=key, seed=seed, exit_code=code))
                    raise RuntimeError(f'{key} seed {seed} failed: see log')
                done += 1
                print(f'{stage}: {done}/{total}, {key} seed {seed}, validation {result(key,seed)["metrics"]["val"]["rmse"]:.6f}', flush=True)
            atomic(out/'status.json', dict(stage=stage, completed=done, total=total,
                elapsed_seconds=time.time()-protocol['started_unix'],
                running=[dict(gpu=g, pid=r[0].pid, candidate=r[2], seed=r[3]) for g,r in running.items()]))
            if queue or running: time.sleep(2)
    execute([(key,1) for key in catalog], 'screening')
    finalists_path = out/'finalists.json'
    if finalists_path.exists(): finalists = json.loads(finalists_path.read_text())
    else:
        ranking = sorted(catalog, key=lambda k: (result(k,1)['metrics']['val']['rmse'], k))
        finalists = dict(candidates=ranking[:8], seed1_ranking=[dict(candidate=k,
            validation_rmse=result(k,1)['metrics']['val']['rmse']) for k in ranking])
        atomic(finalists_path, finalists)
    execute([(key,s) for key in finalists['candidates'] for s in [2,3]], 'confirmation')
    chosen_path = out/'chosen_phytoode.json'
    if chosen_path.exists(): chosen = json.loads(chosen_path.read_text())
    else:
        options = []
        for key in finalists['candidates']:
            paths = [folder(key,s) for s in [1,2,3]]
            vals = [json.loads((p/'result.json').read_text())['metrics']['val']['rmse'] for p in paths]
            options.append(dict(candidate=key, config=catalog[key], val_mean=sum(vals)/3,
                per_seed_validation=vals, paths=[str(p.relative_to(ROOT)) for p in paths]))
        for key in ['phytoode_08', 'phytoode_12']:
            paths = [OLD/'replicate/trials'/key/f'seed{s}' for s in [1,2,3]]
            records = [json.loads((p/'result.json').read_text()) for p in paths]
            config = records[0]['config'] | dict(weight_decay=1e-4, architecture=DEFAULT_ARCHITECTURE)
            vals = [r['metrics']['val']['rmse'] for r in records]
            options.append(dict(candidate='original_'+key, config=config, val_mean=sum(vals)/3,
                per_seed_validation=vals, paths=[str(p.relative_to(ROOT)) for p in paths]))
        options.sort(key=lambda r: (r['val_mean'], r['candidate']))
        chosen = dict(frozen_unix=time.time(), selected=options[0], ranking=options,
                      test_evaluations_during_search=0)
        atomic(chosen_path, chosen)
    matched = chosen['selected']['config'] | dict(model='latent_ode', lambda_ode=0., lambda_k=0.)
    matched_path = out/'configs/latent_ode_matched.json'
    if matched_path.exists(): assert json.loads(matched_path.read_text()) == matched
    else: atomic(matched_path, matched)
    execute([('latent_ode_matched',s) for s in [1,2,3]], 'matched_control')
    selection_path = out/'selections.json'
    if selection_path.exists(): selection = json.loads(selection_path.read_text())
    else:
        runs = []
        for model in ['phytoode', 'latent_ode_matched']:
            for seed in [1,2,3]:
                p = (ROOT/chosen['selected']['paths'][seed-1] if model == 'phytoode'
                     else folder('latent_ode_matched', seed))
                runs.append(dict(protocol='replicate', model=model, seed=seed,
                    path=str(p.relative_to(ROOT)), result_sha256=sha(p/'result.json')))
        for seed in [1,2,3]:
            full = json.loads((ROOT/chosen['selected']['paths'][seed-1]/'result.json').read_text())
            pure = result('latent_ode_matched', seed)
            assert full['initial_state_sha256'] == pure['initial_state_sha256']
        selection = dict(status='all_validation_selections_frozen', frozen_unix=time.time(),
            prior_test_inspected=True, chosen_sha256=sha(chosen_path), runs=runs,
            selected=dict(replicate=dict(phytoode=chosen['selected'], latent_ode_matched=dict(config=matched))))
        atomic(selection_path, selection)
    assert hashes == {p: sha(ROOT/p) for p in hashes}
    for record in selection['runs']:
        destination = out/'replicate/final'/record['model']/f"seed{record['seed']}"
        if (destination/'result.json').exists(): continue
        evaluate(SimpleNamespace(selection=selection_path, protocol='replicate',
            model=record['model'], seed=record['seed'], output=destination))
    atomic(out/'completed.json', dict(status='complete', seconds=time.time()-protocol['started_unix'],
        full_training_trials=99, final_evaluations=6, selection_sha256=sha(selection_path)))
    atomic(out/'status.json', dict(stage='complete', elapsed_seconds=time.time()-protocol['started_unix']))
    print('Fixed-budget follow-up and frozen-choice evaluations complete.', flush=True)


if __name__ == '__main__': main()
