"""Run unchanged frozen screening trials in persistent GPU workers on Linux.

Only our screening controller is paused; active children finish normally.
The original controller resumes in finally and performs selection/evaluation.
No training code, configurations, partitions, or selection rules are changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(task_file):
    import gc
    import torch
    from retune_trial import train
    for task in json.loads(task_file.read_text()):
        output=ROOT/task['output']
        if (output/'result.json').exists(): continue
        train(ROOT/task['config'], output, 1)
        print('Completed '+task['candidate'],flush=True)
        gc.collect(); torch.cuda.empty_cache()


def main(args):
    out=args.output.resolve(); protocol=json.loads((out/'protocol.json').read_text())
    assert args.controller is not None
    process=Path(f'/proc/{args.controller}')
    assert process.stat().st_uid==os.getuid()
    assert 'hypocotyl/code/retune_primary.py' in (process/'cmdline').read_bytes().decode().replace('\0',' ')
    assert (process/'cwd').resolve()==ROOT
    assert not (out/'finalists.json').exists()
    assert all(digest(ROOT/p)==h for p,h in protocol['source_sha256'].items())
    runtime=dict(controller=args.controller,started_unix=time.time(),status='waiting_for_active_trials',
        workers_per_gpu=2,gpus=protocol['gpus'],protocol_sha256=digest(out/'protocol.json'),
        training_source_sha256=digest(ROOT/'hypocotyl/code/retune_trial.py'),
        note='Identical frozen training function and configurations; only interpreter reuse and worker concurrency change.')
    def save():
        path=out/'screening_runtime.json'; temporary=path.with_suffix('.json.part')
        temporary.write_text(json.dumps(runtime,indent=2)+'\n'); temporary.replace(path)
    workers=[]
    os.kill(args.controller,signal.SIGSTOP)
    try:
        status=json.loads((out/'status.json').read_text())
        assert status['stage']=='screening'
        save()
        for running in status['running']:
            pid=running['pid']; proc=Path(f'/proc/{pid}')
            while proc.exists():
                stat=(proc/'stat').read_text().split()
                if stat[2]=='Z': break
                time.sleep(1)
            assert (out/'trials'/running['candidate']/'seed1/result.json').exists()
        remaining=[(key,c) for key,c in protocol['candidates'].items()
                   if not (out/'trials'/key/'seed1/result.json').exists()]
        assignments=[[] for _ in range(4)]; costs=[0]*4
        for key,config in sorted(remaining,key=lambda item:(-item[1]['epochs'],item[0])):
            target=min(range(4),key=lambda i:(costs[i],i))
            assignments[target].append(dict(candidate=key,
                config=str((out/'configs'/f'{key}.json').relative_to(ROOT)),
                output=str((out/'trials'/key/'seed1').relative_to(ROOT))))
            costs[target]+=config['epochs']
        runtime.update(status='screening',candidates=[k for k,c in remaining],workers=[])
        for index,tasks in enumerate(assignments):
            task_file=out/f'screening_worker_{index}.json'
            task_file.write_text(json.dumps(tasks,indent=2)+'\n')
            gpu=protocol['gpus'][index%2]
            log=(out/'logs'/f'persistent_screening_{index}.log').open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(task_file)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            workers.append((child,log))
            runtime['workers'].append(dict(pid=child.pid,gpu=gpu,candidates=[t['candidate'] for t in tasks]))
        save()
        while any(p.poll() is None for p,log in workers):
            runtime['completed_candidates']=sum((out/'trials'/k/'seed1/result.json').exists() for k,c in remaining)
            runtime['elapsed_seconds']=time.time()-runtime['started_unix']; save()
            time.sleep(2)
        assert all(p.returncode==0 for p,log in workers)
        assert all((out/'trials'/key/'seed1/result.json').exists() for key in protocol['candidates'])
        assert all(digest(ROOT/p)==h for p,h in protocol['source_sha256'].items())
        runtime.update(status='complete',completed_candidates=len(remaining),seconds=time.time()-runtime['started_unix'])
        save()
    finally:
        # Never leave our controller suspended, including on interruption/failure.
        for child,log in workers:
            if child.poll() is None:
                child.terminate(); child.wait()
            log.close()
        os.kill(args.controller,signal.SIGCONT)
    print('Persistent-worker screening complete; original controller resumed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--controller',type=int)
    parser.add_argument('--output',type=Path,default=ROOT/'hypocotyl/results/primary_retuning_20260909')
    args=parser.parse_args()
    if args.worker: worker(args.worker)
    else: main(args)
