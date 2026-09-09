"""Add and tune PhytoODE with binary light; retain prior baseline fits unchanged."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import random
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'hypocotyl/results/light_input_20260910'
PREVIOUS=ROOT/'hypocotyl/results/single_condition_20260909'
PROTOCOLS=['replicate','time_holdout']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temporary.replace(path)


def candidates():
    configs=[dict(model='phytoode_light',lr=lr,lambda_ode=ode,lambda_k=k,weight_decay=1e-4,epochs=1500)
             for lr,ode,k in itertools.product([.003,.01],[.5,5.,50.,500.],[.01,.1])]
    rng=random.Random(20260910)
    def latin(lo,hi):
        values=[10**(lo+(hi-lo)*(i+rng.random())/16) for i in range(16)]
        rng.shuffle(values);return values
    lrs=latin(-3.,-1.7); odes=latin(-2.,3.); ks=latin(-4.,-.5); wds=latin(-6.,-3.)
    wds[0]=wds[8]=0.
    configs += [dict(model='phytoode_light',lr=lr,lambda_ode=ode,lambda_k=k,weight_decay=wd,epochs=1500)
                for lr,ode,k,wd in zip(lrs,odes,ks,wds)]
    return {f'light_{i:02d}':config for i,config in enumerate(configs)}


def initialize_worker(queue):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(queue.get()),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')


def worker(job):
    from light_input_trial import train
    return train(job)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',type=int,nargs=2,default=[8,9])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); out=OUT
    assert len(set(args.gpus))==2
    if out.exists() and not args.resume: raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True); (out/'configs').mkdir(exist_ok=True)
    catalog=candidates(); prior=json.loads((PREVIOUS/'selections.json').read_text())
    anchors={p:next(k for k,c in catalog.items() if c==(prior['selected'][p]['phytoode']['config']|dict(model='phytoode_light'))) for p in PROTOCOLS}
    sources=[ROOT/'hypocotyl/code'/name for name in ['light_input_model.py','light_input_trial.py','run_light_input.py',
              'single_condition_data.py','single_condition_models.py']]+[ROOT/'wheat/code/model.py',PREVIOUS/'selections.json',PREVIOUS/'completed.json']
    sources += sorted(p for p in (ROOT/'hypocotyl/data/processed/single_condition_20260909').rglob('*') if p.is_file())
    # Freeze the comparison sources too; no baseline is refitted or replaced.
    comparison_sources=[ROOT/'hypocotyl/reports/single_condition_20260909/comparison.csv']
    comparison_sources += sorted(PREVIOUS.glob('*/final/*/seed*/result.json'))
    comparison_sources += sorted(PREVIOUS.glob('*/final/*/seed*/predictions.npz'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources+comparison_sources}
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text())
        assert protocol['source_sha256']==hashes and protocol['candidates']==catalog and protocol['anchors']==anchors
    else:
        protocol=dict(started_unix=time.time(),gpus=args.gpus,workers_per_gpu=2,protocols=PROTOCOLS,
            model='phytoode_light',candidates=catalog,anchors=anchors,seeds=[1,2,3],
            condition='12L12D',excluded_condition='cR',input_features=['genotype','elapsed_time','binary_light'],
            target_type='split_specific_replicate_means_12L12D_only',height_scale_mm=13.719,
            architecture='Original hidden sizes; one binary input channel added to encoder and vector field. 1103 parameters.',
            physics='Ordinary logistic, one constant r and K per genotype; original capacity penalty and 23 interior residual nodes retained. At light switches the right-hand derivative is used.',
            forcing='Known 12 h on / 12 h off schedule, first 0-12 h on. Each RK4 interval keeps its light state through its right-end stage.',
            selection='32 seed-1 candidates per protocol. Confirm the top three plus the prior-selected no-input hyperparameter anchor with seeds 2 and 3. Choose lowest three-seed mean validation RMSE among these confirmed candidates.',
            stopping='64 screening and 12-16 confirmation fits, then freeze both protocols and score test once per reported configuration and seed. No extension based on test results.',
            comparison='Retained previous baselines; this PhytoODE-only follow-up has additional search budget. Anchor isolates adding the feature at fixed hyperparameters, not identical weight matrices.',
            prior_test_inspected=True,independent_new_test_cohort=False,baseline_refits=0,source_sha256=hashes)
        atomic(protocol_path,protocol)
    for key,config in catalog.items():
        path=out/'configs'/f'{key}.json'
        if path.exists(): assert json.loads(path.read_text())==config
        else: atomic(path,config)
    def folder(p,k,s): return out/p/'trials'/k/f'seed{s}'
    def result(p,k,s): return json.loads((folder(p,k,s)/'result.json').read_text())
    ctx=multiprocessing.get_context('spawn'); queue=ctx.Queue()
    for gpu in args.gpus:
        for _ in range(2): queue.put(gpu)
    with ProcessPoolExecutor(max_workers=4,mp_context=ctx,initializer=initialize_worker,initargs=(queue,)) as pool:
        def execute(jobs,stage):
            pending={}; done=0
            for p,k,s in jobs:
                destination=folder(p,k,s)
                if (destination/'result.json').exists(): done+=1;continue
                if destination.exists():
                    diagnostic=out/'diagnostics'/f'{p}_{k}_{s}_{time.time_ns()}'
                    diagnostic.parent.mkdir(exist_ok=True); destination.rename(diagnostic)
                job=dict(protocol=p,key=k,seed=s,config_path=str(out/'configs'/f'{k}.json'),output=str(destination))
                pending[pool.submit(worker,job)]=job
            while pending:
                finished,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for future in finished:
                    job=pending.pop(future)
                    try: value=future.result()
                    except BaseException as error:
                        atomic(out/'failure.json',dict(stage=stage,job=job,error=repr(error)));raise
                    done+=1;print(f'{stage} {done}/{len(jobs)}: {value}',flush=True)
                atomic(out/'status.json',dict(stage=stage,completed=done,total=len(jobs),remaining=len(pending),elapsed_seconds=time.time()-protocol['started_unix']))
        execute([(p,k,1) for k in catalog for p in PROTOCOLS],'screening')
        finalists_path=out/'finalists.json'
        if finalists_path.exists(): finalists=json.loads(finalists_path.read_text())
        else:
            finalists={p:sorted(catalog,key=lambda k:(result(p,k,1)['metrics']['val']['rmse'],k))[:3] for p in PROTOCOLS}
            for p in PROTOCOLS:
                if anchors[p] not in finalists[p]: finalists[p].append(anchors[p])
            atomic(finalists_path,finalists)
        execute([(p,k,s) for p in PROTOCOLS for k in finalists[p] for s in [2,3]],'confirmation')
    selected={}; runs=[]
    for p in PROTOCOLS:
        ranking=[]
        for key in finalists[p]:
            values=[result(p,key,s)['metrics']['val']['rmse'] for s in [1,2,3]]
            ranking.append(dict(candidate=key,val_mean=sum(values)/3,per_seed_validation=values))
        ranking.sort(key=lambda r:(r['val_mean'],r['candidate']))
        selected[p]=dict(phytoode_light=dict(selected_candidate=ranking[0]['candidate'],config=catalog[ranking[0]['candidate']],ranking=ranking),
            phytoode_light_anchor=dict(selected_candidate=anchors[p],config=catalog[anchors[p]],selection_basis='All prior-selected no-input PhytoODE hyperparameters, with binary light input added.'))
        for label,choice in selected[p].items():
            for seed in [1,2,3]:
                source=folder(p,choice['selected_candidate'],seed)
                runs.append(dict(protocol=p,model=label,seed=seed,path=str(source.relative_to(ROOT)),result_sha256=sha(source/'result.json')))
    selection_path=out/'selections.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text());assert selection['selected']==selected and selection['runs']==runs
    else:
        selection=dict(status='all_validation_selections_frozen',frozen_unix=time.time(),selected=selected,runs=runs,prior_test_inspected=True)
        atomic(selection_path,selection)
    assert hashes=={p:sha(ROOT/p) for p in hashes}
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    from light_input_trial import evaluate
    for record in runs:
        destination=out/record['protocol']/'final'/record['model']/f"seed{record['seed']}"
        if not (destination/'result.json').exists(): evaluate(selection_path,record,destination)
    count=len(list(out.glob('*/trials/*/seed*/result.json')))
    atomic(out/'completed.json',dict(status='complete',full_training_trials=count,final_evaluations=len(runs),
        seconds=time.time()-protocol['started_unix'],selection_sha256=sha(selection_path),baseline_refits=0))
    atomic(out/'status.json',dict(stage='complete',elapsed_seconds=time.time()-protocol['started_unix']))
    print(f'Completed {count} PhytoODE-only fits and {len(runs)} evaluations; no baseline refits.',flush=True)


if __name__=='__main__': main()
