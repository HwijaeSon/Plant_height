"""Retrain the six manuscript predictors on the four approved genotypes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[2]
MODELS=['phytoode','latent_ode','logistic_pinn','lstm','rf','logistic']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0])
    parser.add_argument('--workers-per-gpu',type=int,default=2)
    parser.add_argument('--output',type=Path,default=ROOT/'hypocotyl/results/four_genotypes_20260915')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    jobs=[dict(model=model,seed=seed,drop=drop,path=f'drop_{round(100*drop)}/{model}/seed{seed}')
          for drop in [0.,.25,.5] for model in MODELS
          for seed in ([1] if model=='logistic' and drop==0 else [1,2,3])]
    files=['hypocotyl/code/single_condition_models.py','wheat/code/model.py','run.py','hypocotyl/code/forecast_data.py','hypocotyl/code/forecast_models.py',
           'hypocotyl/code/forecast_trial.py','hypocotyl/code/no_light_prefix_model.py',
           'hypocotyl/code/no_light_prefix_trial.py','hypocotyl/code/prefix_parameter_model.py',
           'configs/hypocotyl_forecast.json','configs/hypocotyl_prefix_parameters_no_light_20260911.json',
           'hypocotyl/data/processed/four_genotypes_20260915/config.json']
    plan=dict(protocol='four_genotypes_20260915',jobs=jobs,
              settings='Manuscript settings fixed before retraining; no previous weights reused.',
              code_data_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    plan_path=output/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Frozen execution plan changed')
    else:plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    queues={k:Queue() for k in ['cpu','gpu']}
    for job in jobs:
        out=output/job['path'];final=out/'evaluation/result.json' if job['model']=='phytoode' else out/'result.json'
        if final.exists() and json.loads(final.read_text())['status']=='complete':continue
        if out.exists():raise FileExistsError(f'Inspect incomplete run before restarting: {out}')
        queues['cpu' if job['model'] in ['rf','logistic'] else 'gpu'].put(job)
    failures=[];lock=threading.Lock()
    def worker(kind,gpu=None):
        while True:
            try:job=queues[kind].get_nowait()
            except Empty:return
            out=output/job['path'];out.parent.mkdir(parents=True,exist_ok=True)
            device='cpu' if gpu is None else 'cuda:0'
            common=['--dataset','hypocotyl','--model',job['model'],'--seed',str(job['seed']),
                    '--drop-fraction',str(job['drop']),'--device',device]
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                     CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
            print(f'START {job["path"]} GPU={gpu}',flush=True)
            with (out.parent/f'{out.name}.log').open('w') as log:
                result=subprocess.run([sys.executable,'run.py','train',*common,'--output',str(out)],
                                      env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode==0 and job['model']=='phytoode':
                    result=subprocess.run([sys.executable,'run.py','evaluate',*common,
                        '--checkpoint',str(out/'checkpoint.pt'),'--output',str(out/'evaluation')],
                        env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            print(f'FINISH {job["path"]} status={result.returncode}',flush=True)
            if result.returncode:
                with lock:failures.append(job)
    with ThreadPoolExecutor(max_workers=len(args.gpus)*args.workers_per_gpu+2) as pool:
        futures=[pool.submit(worker,'gpu',gpu) for gpu in args.gpus for _ in range(args.workers_per_gpu)]
        futures += [pool.submit(worker,'cpu') for _ in range(2)]
        for future in futures:future.result()
    (output/'completion.json').write_text(json.dumps(dict(expected_runs=len(jobs),failures=failures),indent=2)+'\n')
    if failures:raise RuntimeError(f'{len(failures)} failed runs')


if __name__=='__main__':main()
