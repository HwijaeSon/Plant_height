"""Run the frozen prefix-forecast and paired missingness benchmark on selected GPUs."""
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

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--workers-per-gpu',type=int,default=2)
    p.add_argument('--output',type=Path,default=ROOT/'hypocotyl/results/prefix_forecast_20260911')
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    profile=json.loads((ROOT/'configs/hypocotyl_forecast.json').read_text())
    jobs=[]
    for drop in profile['conditions']:
        for model in ['phytoode','latent_ode_light','latent_ode','logistic_pinn','lstm','rf','logistic']:
            for seed in ([1] if model=='logistic' and drop==0 else profile['seeds']):
                jobs.append(dict(model=model,seed=seed,drop_fraction=drop,mask_seed=20260910+seed,
                    path=f'drop_{round(drop*100)}/{model}/seed{seed}'))
    files=['hypocotyl/code/forecast_data.py','hypocotyl/code/forecast_models.py','hypocotyl/code/forecast_trial.py','configs/hypocotyl_forecast.json']
    plan=dict(jobs=jobs,code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    plan_path=args.output/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Frozen plan changed')
    else:plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    queues={key:Queue() for key in ['cpu','gpu']}
    for job in jobs:
        out=args.output/job['path']
        if (out/'result.json').exists():
            if json.loads((out/'result.json').read_text())['status']=='complete':continue
        if out.exists():raise FileExistsError(f'Incomplete run requires inspection before restart: {out}')
        queues['cpu' if job['model'] in ['rf','logistic'] else 'gpu'].put(job)
    failures=[];lock=threading.Lock()
    def worker(kind,gpu=None):
        while True:
            try:job=queues[kind].get_nowait()
            except Empty:return
            out=args.output/job['path'];out.parent.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,str(ROOT/'hypocotyl/code/forecast_trial.py'),'train','--model',job['model'],
                '--seed',str(job['seed']),'--drop-fraction',str(job['drop_fraction']),'--mask-seed',str(job['mask_seed']),
                '--device','cpu' if gpu is None else 'cuda:0','--output',str(out)]
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
            env['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
            print(f'START {job["path"]} GPU={gpu}',flush=True)
            with (out.parent/f'{out.name}.log').open('w') as log:
                result=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            print(f'FINISH {job["path"]} code={result.returncode}',flush=True)
            if result.returncode:
                with lock:failures.append(job)
    with ThreadPoolExecutor(max_workers=len(args.gpus)*args.workers_per_gpu+2) as pool:
        futures=[pool.submit(worker,'gpu',gpu) for gpu in args.gpus for _ in range(args.workers_per_gpu)]
        futures += [pool.submit(worker,'cpu') for _ in range(2)]
        for future in futures:future.result()
    (args.output/'completion.json').write_text(json.dumps(dict(expected_runs=len(jobs),failures=failures),indent=2)+'\n')
    if failures:raise RuntimeError(f'{len(failures)} failed runs')

if __name__=='__main__':main()
