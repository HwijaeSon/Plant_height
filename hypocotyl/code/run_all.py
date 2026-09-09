"""Run validation-selected light-growth comparisons on two explicitly chosen GPUs."""
from __future__ import annotations

import argparse
from collections import deque
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from trial import atomic, sha

ROOT=Path(__file__).resolve().parents[2]
MODELS=['phytoode','latent_ode','light_pinn','lstm','rf','light_logistic','logistic']
PROTOCOLS=['replicate','time_holdout']


def candidates(model):
    if model in ('phytoode','light_pinn'):
        return [dict(model=model,lr=lr,lambda_ode=ode,lambda_k=k,epochs=1500)
                for lr,ode,k in itertools.product([.003,.01],[.5,5.,50.,500.],[.01,.1])]
    if model in ('latent_ode','lstm'):
        return [dict(model=model,lr=lr,lambda_ode=0.,lambda_k=0.,epochs=1500) for lr in [.003,.01]]
    if model=='rf':return [dict(model=model,min_samples_leaf=leaf) for leaf in [1,2,4]]
    return [dict(model=model)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',nargs=2,type=int,default=[8,9])
    parser.add_argument('--output',type=Path,default=ROOT/'hypocotyl/results/light_growth_20260909')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() and not args.resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    (out/'logs').mkdir(exist_ok=True)
    if args.gpus[0]==args.gpus[1]:raise ValueError('Use two distinct GPU indices')
    code_paths=[ROOT/'hypocotyl/code'/name for name in ('prepare_data.py','make_splits.py','data.py',
        'models.py','process_baselines.py','trial.py','run_all.py','verify_model.py')]+[ROOT/'wheat/code/model.py']
    data_paths=list((ROOT/'hypocotyl/data/processed/splits').rglob('*.csv'))+list((ROOT/'hypocotyl/data/processed/splits').rglob('*.json'))+[ROOT/'hypocotyl/data/metadata.json']
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in code_paths+data_paths}
    if args.resume:
        protocol=json.loads((out/'protocol.json').read_text())
        assert protocol['source_sha256']==source_hashes
    else:
        protocol=dict(started_unix=time.time(),gpus=args.gpus,seeds=[1,2,3],models=MODELS,
            protocols=PROTOCOLS,candidates={m:candidates(m) for m in MODELS},
            screening='Seed 1; confirm two leading candidates with seeds 2 and 3; select by three-seed mean validation RMSE.',
            test_access='All selections for both protocols and all models freeze before any held-out scoring.',
            source_sha256=source_hashes)
        atomic(out/'protocol.json',protocol)
    catalog={}
    for model in MODELS:
        for index,config in enumerate(candidates(model)):
            key=f'{model}_{index:02d}'
            catalog[key]=config
            path=out/'configs'/f'{key}.json';path.parent.mkdir(exist_ok=True)
            if path.exists():assert json.loads(path.read_text())==config
            else:atomic(path,config)

    def trial_path(protocol,key,seed):return out/protocol/'trials'/key/f'seed{seed}'

    def execute(jobs,stage):
        queue=deque(jobs);running={};done=0
        while queue or running:
            for gpu in args.gpus:
                if gpu in running:continue
                while queue:
                    protocol,key,seed=queue.popleft();folder=trial_path(protocol,key,seed)
                    if (folder/'result.json').exists():done+=1;continue
                    if folder.exists():
                        diagnostic=out/'diagnostics'/f'{protocol}_{key}_seed{seed}_{time.time_ns()}'
                        diagnostic.parent.mkdir(exist_ok=True);folder.rename(diagnostic)
                    log=(out/'logs'/f'{protocol}_{key}_seed{seed}.log').open('w')
                    command=[sys.executable,str(ROOT/'hypocotyl/code/trial.py'),'train','--protocol',protocol,
                             '--seed',str(seed),'--config',str(out/'configs'/f'{key}.json'),'--output',str(folder)]
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    running[gpu]=(process,log,(protocol,key,seed),folder)
                    break
            for gpu,(process,log,job,folder) in list(running.items()):
                status=process.poll()
                if status is None:continue
                log.close();del running[gpu]
                if status!=0:
                    atomic(out/'failed.json',dict(stage=stage,job=job,returncode=status))
                    raise RuntimeError(f'Trial failed: {job}; see its log. Other already-running trials may finish.')
                done+=1
                result=json.loads((folder/'result.json').read_text())
                print(f"{stage}: {done}/{len(jobs)} {job} val={result['metrics']['val']['rmse']:.5f}",flush=True)
            atomic(out/'status.json',dict(stage=stage,completed=done,total=len(jobs),
                running={str(g):dict(pid=p.pid,protocol=j[0],candidate=j[1],seed=j[2],path=str(f)) for g,(p,l,j,f) in running.items()},
                elapsed_seconds=time.time()-protocol_start))
            time.sleep(2)

    protocol_start=protocol['started_unix']
    screens=[(p,key,1) for key in catalog for p in PROTOCOLS]
    execute(screens,'screening')
    finalist_path=out/'finalists.json'
    if finalist_path.exists():finalists=json.loads(finalist_path.read_text())
    else:
        finalists={}
        for p in PROTOCOLS:
            finalists[p]={}
            for model in MODELS:
                options=[key for key,c in catalog.items() if c['model']==model]
                options.sort(key=lambda key:(json.loads((trial_path(p,key,1)/'result.json').read_text())['metrics']['val']['rmse'],key))
                finalists[p][model]=options[:2]
        atomic(finalist_path,finalists)
    confirmations=[(p,key,seed) for p in PROTOCOLS for model in MODELS
        if model not in ('logistic','light_logistic') for key in finalists[p][model] for seed in [2,3]]
    execute(confirmations,'confirmation')
    selection_path=out/'selections.json'
    if selection_path.exists():selection=json.loads(selection_path.read_text())
    else:
        selected={};runs=[]
        for p in PROTOCOLS:
            selected[p]={}
            for model in MODELS:
                seeds=[1] if model in ('logistic','light_logistic') else [1,2,3]
                ranking=[]
                for key in finalists[p][model]:
                    values=[json.loads((trial_path(p,key,s)/'result.json').read_text())['metrics']['val']['rmse'] for s in seeds]
                    ranking.append(dict(candidate=key,val_mean=sum(values)/len(values),per_seed_validation=values))
                ranking.sort(key=lambda r:(r['val_mean'],r['candidate']))
                key=ranking[0]['candidate']
                selected[p][model]=dict(config=catalog[key],ranking=ranking,selected_candidate=key)
                for seed in seeds:
                    folder=trial_path(p,key,seed)
                    runs.append(dict(protocol=p,model=model,seed=seed,path=str(folder.relative_to(ROOT)),result_sha256=sha(folder/'result.json')))
            # Also retain a strictly matched loss-removal control at PhytoODE's
            # selected learning rate. Both zero-loss learning rates already have
            # three completed seeds, so this needs no additional training.
            lr=selected[p]['phytoode']['config']['lr']
            key=next(key for key,c in catalog.items() if c['model']=='latent_ode' and c['lr']==lr)
            selected[p]['latent_ode_matched']=dict(config=catalog[key],selected_candidate=key,
                selection_basis='Same learning rate, architecture, initialization and schedule as selected PhytoODE; both loss coefficients zero.')
            for seed in [1,2,3]:
                folder=trial_path(p,key,seed)
                runs.append(dict(protocol=p,model='latent_ode_matched',seed=seed,path=str(folder.relative_to(ROOT)),result_sha256=sha(folder/'result.json')))
        selection=dict(status='all_validation_selections_frozen',frozen_unix=time.time(),selected=selected,runs=runs)
        atomic(selection_path,selection)
    for record in selection['runs']:
        folder=out/record['protocol']/'final'/record['model']/f"seed{record['seed']}"
        if (folder/'result.json').exists():continue
        command=[sys.executable,str(ROOT/'hypocotyl/code/trial.py'),'evaluate','--protocol',record['protocol'],
            '--model',record['model'],'--seed',str(record['seed']),'--selection',str(selection_path),'--output',str(folder)]
        subprocess.run(command,cwd=ROOT,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    atomic(out/'completed.json',dict(status='complete',seconds=time.time()-protocol_start,
        full_training_trials=len(screens)+len(confirmations),final_evaluations=len(selection['runs']),selection_sha256=sha(selection_path)))
    atomic(out/'status.json',dict(stage='complete',elapsed_seconds=time.time()-protocol_start))
    print('All experiments and frozen-selection held-out evaluations complete.',flush=True)


if __name__=='__main__':main()
