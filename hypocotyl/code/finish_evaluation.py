"""Finish frozen CPU evaluations in one interpreter, then resume their controller.

Optional runtime optimization for an already completed training/search phase.
It calls the unchanged trial.evaluate function, with the same frozen selection
and saved predictions. No model fitting or selection occurs here.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path
import signal
import time

import torch
from trial import evaluate, atomic, sha

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--controller-pid',type=int,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    out=ROOT/'hypocotyl/results/light_growth_20260909'
    selection_path=out/'selections.json'
    selections=json.loads(selection_path.read_text())
    assert selections['status']=='all_validation_selections_frozen'
    assert not (out/'completed.json').exists()
    protocol=json.loads((out/'protocol.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in protocol['source_sha256'].items())
    command=Path(f'/proc/{args.controller_pid}/cmdline').read_bytes().split(b'\0')
    assert Path(os.readlink(f'/proc/{args.controller_pid}/cwd'))==ROOT
    assert any(c.decode().endswith('hypocotyl/code/run_all.py') for c in command)
    torch.set_num_threads(2)
    started=time.time(); completed=[]
    os.kill(args.controller_pid,signal.SIGSTOP)
    try:
        # Let its already-started evaluation finish before taking remaining jobs.
        children=Path(f'/proc/{args.controller_pid}/task/{args.controller_pid}/children').read_text().split()
        for child in children:
            stat=Path(f'/proc/{child}/stat')
            cmd=Path(f'/proc/{child}/cmdline').read_bytes()
            assert b'trial.py' in cmd and b'evaluate' in cmd
            while stat.exists() and stat.read_text().split(') ',1)[1][0] != 'Z':
                if time.time()-started>180: raise TimeoutError('The current evaluation has not exited')
                time.sleep(1)
        def perform(record):
            p,m,seed=record['protocol'],record['model'],record['seed']
            final=out/p/'final'/m/f'seed{seed}'
            assert not final.exists()
            temporary=out/'parallel_work'/p/m/f'seed{seed}'
            evaluate(Namespace(protocol=p,model=m,seed=seed,selection=selection_path,output=temporary))
            final.parent.mkdir(parents=True,exist_ok=True)
            temporary.rename(final)
            return dict(protocol=p,model=m,seed=seed,result_sha256=sha(final/'result.json'))
        pending=[r for r in selections['runs'] if not (out/r['protocol']/'final'/r['model']/f"seed{r['seed']}"/'result.json').exists()]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(perform,r) for r in pending]):
                completed.append(future.result())
                print(f'Frozen evaluation {len(completed)}/{len(pending)} complete',flush=True)
        atomic(out/'evaluation_runtime.json',dict(status='complete',workers=args.workers,
            selection_sha256=sha(selection_path),evaluator_sha256=sha(ROOT/'hypocotyl/code/trial.py'),
            orchestration_sha256=sha(Path(__file__)),completed_records=completed,
            seconds=time.time()-started,
            note='After all choices froze, the existing controller was paused after its active evaluation. Remaining evaluations called the unchanged evaluator in one CPU interpreter, then the controller resumed. Sources, predictions and selection were unchanged.'))
    finally:
        os.kill(args.controller_pid,signal.SIGCONT)


if __name__=='__main__':main()
