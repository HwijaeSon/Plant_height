"""Evaluate retained classical and reference-neural fits without retraining."""
import argparse,json,sys
from pathlib import Path
import joblib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from layout import artifact

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True,choices=['wheat','maize','arabidopsis']);p.add_argument('--model',required=True,choices=['logistic','temperature','rf','lstm','pinn']);p.add_argument('--seed',type=int,default=1);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 torch.set_num_threads(2);sys.path.insert(0,str(ROOT/'src'/a.dataset))
 if a.dataset=='wheat' and a.model in ['lstm','pinn']:
  p.error('The wheat reference authors released predictions/summaries, not checkpoints in this release. See REPRODUCTION.md.')
 if a.dataset=='arabidopsis' and a.model in ['lstm','pinn','rf']:
  p.error('Original Arabidopsis baseline weights were not saved. Predictions and retraining code are provided; see REPRODUCTION.md.')
 a.output.mkdir(parents=True,exist_ok=False);predictions={};metrics={}
 if a.dataset in ['wheat','maize']:
  if a.dataset=='wheat':
   import run_additional_baselines as impl
   cfg=json.loads((ROOT/'src/configs/wheat.json').read_text())
   ds=impl.make_dataset(data_path=ROOT/cfg['data'],kinship_path=ROOT/cfg['kinship'],split=cfg['split'],start_day=cfg['start_day'],env_cols=cfg['env_cols'],fill_in_na_at_start=not cfg['no_fill_na_start'],val_year=cfg['val_year'],test_year=cfg['test_year'],add_gdd=cfg['add_gdd'],gdd_base=cfg['gdd_base'],genotypes=cfg['genotypes'])
   folder=ROOT/'checkpoints/wheat';scale=1.
  else:
   import run_experiment as impl
   ds=impl.make_dataset(split='chronological');folder=ROOT/'checkpoints/maize';scale=ds.y_scale
  if a.model in ['logistic','temperature']:params=json.loads((folder/f'{a.model}.json').read_text())
  elif a.model=='rf':model=joblib.load(artifact(a.dataset,'rf',a.seed,'forest.joblib'))
  else:
   params=json.loads((folder/'logistic.json').read_text());model=impl.build_model(a.model,ds,params)
   saved=torch.load(artifact(a.dataset,a.model,a.seed,'checkpoint.pt'),map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict']);model.eval()
  for split,attr in [('train','train_eval'),('val','val'),('test','test')]:
   batch=getattr(ds,attr)
   if a.dataset=='wheat':
    pred=impl.forest_predict(model,batch,len(ds.genotypes)) if a.model=='rf' else impl.process_predict(batch,ds,params)
   elif a.model in ['lstm','pinn']:
    with torch.no_grad():pred=impl.forward(a.model,model,impl.tensor_batch(batch,'cpu'),torch.linspace(0.,1.,len(ds.days)))['prediction'].numpy()
   elif a.model=='rf':
    rows,cols=np.indices(batch.y.shape);rows=rows.ravel();cols=cols.ravel();values=[];n=len(ds.genotypes)
    for start in range(0,len(rows),4096):
     rr=rows[start:start+4096];cc=cols[start:start+4096];x=np.zeros((len(rr),n+2),dtype=np.float32);x[np.arange(len(rr)),batch.g_idx[rr]]=1;x[:,n]=batch.env[rr,cc,0];x[:,n+1]=ds.days[cc]/ds.days[-1];values.append(model.predict(x))
    pred=np.concatenate(values).reshape(batch.y.shape)
   else:
    elapsed=ds.days[None,:]
    if a.model=='temperature':
     temp=batch.env[:,:,0]*ds.env_std[0]+ds.env_mean[0];resp=impl.temperature_response(temp,params['lower_k'],params['upper_k']);elapsed=np.concatenate([np.zeros((len(temp),1)),np.cumsum((resp[:,:-1]+resp[:,1:])/2,axis=1)],axis=1)
    pred=impl.logistic(elapsed,np.asarray(params['r'])[batch.g_idx,None],np.asarray(params['K'])[batch.g_idx,None])
   predictions[attr]=pred;mask=batch.mask.astype(bool);errors=(pred.astype(float)-batch.y.astype(float))*scale;rmse=np.sqrt((errors**2*mask).sum(1)/mask.sum(1)).mean();mean=batch.y[mask].astype(float).mean()*scale
   metrics[split]=dict(rmse=float(rmse),relative_error=float(100*rmse/mean))
  seed=0 if a.model in ['logistic','temperature'] else a.seed
  expected=np.load(artifact(a.dataset,a.model,seed,'predictions.npz'))
  for key,val in predictions.items():np.testing.assert_allclose(val,expected[key],rtol=2e-5,atol=1e-5)
 else:
  import run_experiment as impl
  ds=impl.load_dataset(impl.DEFAULT_DATA);label={'logistic':'Logi-ODE','temperature':'Temp-ODE'}[a.model]
  params=json.loads((ROOT/'checkpoints/arabidopsis/process_parameters.json').read_text())[label]
  for split,attr in [('train','train_eval'),('val','val'),('test','test')]:
   b=getattr(ds,attr);g,temp,days=impl.batch_metadata(b,ds);multiplier=impl.paper_temperature_response(temp,params['lower_k'][0],params['upper_k'][0]) if a.model=='temperature' else np.ones(len(temp))
   pred=np.stack([impl.logistic_solution(days,params['r'][g[i]],params['K'][g[i]],multiplier[i]) for i in range(len(g))]).astype(np.float32)
   predictions[attr]=pred;mask=b.mask.numpy().astype(bool);target=b.target.numpy();error=(pred.astype(float)-target.astype(float))*100;rmse=np.sqrt((error**2*mask).sum(1)/mask.sum(1)).mean();metrics[split]=dict(rmse=float(rmse),relative_error=float(100*rmse/(target[mask].astype(float).mean()*100)))
 np.savez_compressed(a.output/'predictions.npz',**predictions);(a.output/'result.json').write_text(json.dumps(dict(dataset=a.dataset,model=a.model,seed=a.seed,metrics=metrics),indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
