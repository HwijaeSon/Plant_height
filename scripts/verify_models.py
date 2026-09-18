"""Restore every manuscript PhytoODE/control checkpoint and hypocotyl reference fit.

Requires downloaded/prepared external data. CPU predictions are compared with
recorded arrays at float32 tolerances; this does not retrain any model.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['all','hypocotyl','wheat','maize','arabidopsis'],default='all');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);profile=json.loads((ROOT/'configs/paper.json').read_text());records=[]
 for ds in (['wheat','maize','arabidopsis','hypocotyl'] if a.dataset=='all' else [a.dataset]):
  tasks=[]
  for model,info in profile['datasets'][ds]['models'].items():
   if ds=='hypocotyl':
    if model in ['phytoode','latent_ode']:
     for drop,seeds in info['checkpoints_by_missingness'].items():
      for seed,r in seeds.items():tasks.append((model,int(seed),int(drop)/100,ROOT/r['path']))
    else:
     for folder in sorted((ROOT/'hypocotyl/results/four_genotypes_20260915').glob(f'drop_*/{model}/seed*')):
      name={'rf':'forest.joblib','logistic':'parameters.json'}.get(model,'checkpoint.pt')
      tasks.append((model,int(folder.name[4:]),int(folder.parents[1].name[5:])/100,folder/name))
   else:
    for seed,r in info['checkpoints'].items():tasks.append((model,int(seed),0,ROOT/r['path']))
  for model,seed,drop,checkpoint in tasks:
   dest=a.output/f'{ds}-{model}-drop{round(100*drop)}-seed{seed}'
   command=[sys.executable,str(ROOT/'run.py'),'evaluate','--dataset',ds,'--model',model,'--seed',str(seed),'--checkpoint',str(checkpoint),'--output',str(dest)]
   if ds=='hypocotyl':command+=['--drop-fraction',str(drop)]
   done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
   if done.returncode:raise RuntimeError(done.stdout+done.stderr)
   if ds=='hypocotyl' or model!='phytoode':reference=checkpoint.parent/'predictions.npz'
   elif ds=='maize':reference=ROOT/f'maize/results/tuning_20260904/final/runs/latent_seed{seed}/predictions.npz'
   else:reference=ROOT/f'experiments/results/joint_lambda_tuning_20260908/{ds}/final/selected_phyto/seed{seed}/predictions.npz'
   actual=np.load(dest/'predictions.npz');expected=np.load(reference);keys=['prediction'] if ds=='hypocotyl' else ['train_eval','val','test'];differences={}
   for key in keys:
    np.testing.assert_allclose(actual[key],expected[key],rtol=2e-5,atol=2e-6,err_msg=f'{ds}/{model}/{seed}/{drop}/{key}')
    differences[key]=float(np.max(np.abs(actual[key]-expected[key])))
   record=dict(dataset=ds,model=model,seed=seed,drop=drop,max_absolute_difference=differences);records.append(record)
   print(f'PASS {ds}/{model} seed={seed} drop={drop:g}',flush=True)
 (a.output/'verification.json').write_text(json.dumps(dict(status='passed',n_fits=len(records),fits=records),indent=2)+'\n')
if __name__=='__main__':main()
