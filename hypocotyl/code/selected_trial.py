"""Train on 0--36 h and select with 48 h; future scoring is a separate action."""
from pathlib import Path
import argparse,json,os,time,hashlib
import numpy as np
import pandas as pd
import torch
from selected_model import DEFAULTS,initialize,augment_inputs,load,data,FIXED_SCALE
from forecast_trial import state_sha

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def contrasts(out,x):
 means={g:float(out['a'][x['g_idx']==i].detach().mean()) for i,g in enumerate(data.GENOTYPES)}
 absolute=min(abs(means[g]) for g in ['Col-0','MLB'])-max(abs(means[g]) for g in ['hy5','phyAB'])
 signed=min(means[g] for g in ['Col-0','MLB'])-max(means[g] for g in ['hy5','phyAB'])
 return dict(a=means,absolute_gap=absolute,signed_gap=signed,absolute_gate=absolute>.05,signed_gate=signed>.05)
def evaluate(model,x,cfg,out,include_test=False):
 dc,plants,check,batches,_=load(include_test=include_test,drop_fraction=cfg['additional_prefix_drop'],mask_seed=cfg['mask_seed'])
 for key in x:torch.testing.assert_close(x[key].detach().cpu(),check[key],rtol=0,atol=0)
 model.eval()
 with torch.no_grad():output=model(**x);pred=output['pred'].detach().cpu().numpy()
 metrics={split:data.score(pred,batch,FIXED_SCALE) for split,batch in batches.items()}
 arrays=dict(prediction=pred,plant_ids=plants.plant_id.to_numpy(dtype=str),genotype=plants.genotype.to_numpy(dtype=str),hours=data.HOURS,prefix_y=check['prefix_y'].numpy(),prefix_mask=check['prefix_mask'].numpy())
 for split,batch in batches.items():arrays[split+'_target']=batch['target'].numpy();arrays[split+'_mask']=batch['mask'].numpy()
 np.savez_compressed(out/'predictions.npz',**arrays)
 pars=plants[['plant_id','genotype','source_row']].copy()
 for key in ['r','r_light','r_dark','a']:pars[key]=output[key].detach().cpu().numpy()
 if 'K' in output:pars['K_mm']=output['K'].detach().cpu().numpy()*FIXED_SCALE
 pars.to_csv(out/'individual_parameters.csv',index=False)
 return metrics,contrasts(output,x)
def fitloss(pred,batch,kind):
 if kind=='rmse':return data.masked_rmse(pred,batch)
 residual=pred-batch['target'];mask=batch['mask'];counts=mask.sum(1);valid=counts>0
 if kind=='mse':value=residual.square()
 elif kind=='huber':value=torch.nn.functional.huber_loss(pred,batch['target'],delta=.1,reduction='none')
 else:raise ValueError(kind)
 return ((value*mask).sum(1)[valid]/counts[valid]).mean()
def train(profile,seed,drop,output,device):
 cfg=dict(DEFAULTS,**read(profile));cfg.update(seed=seed,additional_prefix_drop=drop,mask_seed=20260910+seed,
  experiment='phytoode2_reproduction_search_20260915',device=device,cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),torch_version=torch.__version__,profile_sha256=sha(profile))
 torch.set_num_threads(2)
 dc,plants,x,batches,_=load(device=device,include_test=False,drop_fraction=drop,mask_seed=cfg['mask_seed'])
 assert len(plants)==141 and set(plants.genotype)=={'Col-0','MLB','hy5','phyAB'}
 model=initialize(cfg,seed,device);cfg.update(data=dc,height_scale_mm=FIXED_SCALE,n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),initial_state_sha256=state_sha(model))
 out=Path(output);out.mkdir(parents=True,exist_ok=False);write(out/'config.json',cfg)
 rates=model.rate_parameters();capacities=model.capacity_parameters();special={id(p) for p in rates+capacities}
 regular=[p for p in model.parameters() if p.requires_grad and id(p) not in special]
 groups=[dict(params=regular,lr=cfg['lr'],weight_decay=cfg['weight_decay'])]
 if rates:groups.append(dict(params=rates,lr=cfg['lr']*cfg['beta_lr_factor'],weight_decay=0.))
 if capacities:groups.append(dict(params=capacities,lr=cfg['lr']*cfg['physical_lr_factor'],weight_decay=0.))
 optimizer=(torch.optim.Adam if cfg['optimizer']=='adam' else torch.optim.AdamW)(groups)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs']) if cfg['scheduler']=='cosine' else None
 rng=torch.Generator(device=device).manual_seed(710000+seed);recent=[];history=[];best=float('inf');best_epoch=None;best_state=None;started=time.monotonic()
 for epoch in range(1,cfg['epochs']+1):
  model.train();inputs=augment_inputs(x,cfg['input_dropout'],rng);prediction=model(**inputs)
  fit=fitloss(prediction['pred'],batches['train'],cfg['fit_type']);physics=model.physics_loss(prediction,inputs)
  interval=model.phase_loss(prediction,batches['train']) if cfg['interval_weight'] else fit*0
  coefficient=cfg['lambda_ode']*min(1.,epoch/max(1,cfg['physics_warmup']))
  penalty=prediction['a'].square().mean()
  loss=fit+coefficient*physics+cfg['interval_weight']*interval+cfg['contrast_penalty']*penalty
  if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite epoch {epoch}')
  optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);optimizer.step()
  if scheduler:scheduler.step()
  if epoch%cfg['eval_every']==0:
   model.eval()
   with torch.no_grad():valout=model(**x);val=float(data.masked_rmse(valout['pred'],batches['val']))
   c=contrasts(valout,x);recent=(recent+[val])[-cfg['val_smooth']:];smooth=float(np.mean(recent))
   if epoch>=cfg['min_epoch'] and len(recent)==cfg['val_smooth'] and smooth<best:
    best,best_epoch=smooth,epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   history.append(dict(epoch=epoch,fit_loss=float(fit.detach()),physics_loss=float(physics.detach()),interval_loss=float(interval.detach()),objective=float(loss.detach()),val_rmse=val*FIXED_SCALE,selection_rmse=smooth*FIXED_SCALE,best_epoch=best_epoch,absolute_gap=c['absolute_gap'],signed_gap=c['signed_gap'],**{f'a_{g}':v for g,v in c['a'].items()}))
  if epoch%100==0:
   pd.DataFrame(history).to_csv(out/'history.csv',index=False);write(out/'progress.json',dict(epoch=epoch,epochs=cfg['epochs'],seconds=time.monotonic()-started,test_evaluations=0))
 assert best_state is not None
 model.load_state_dict(best_state);torch.save(dict(state_dict=best_state,config=cfg,epoch=best_epoch),out/'checkpoint.pt')
 write(out/'selection.json',dict(best_epoch=best_epoch,selection_rmse=best*FIXED_SCALE,checkpoint_sha256=sha(out/'checkpoint.pt'),test_evaluations=0))
 metrics,contrast=evaluate(model,x,cfg,out)
 pd.DataFrame(history).to_csv(out/'history.csv',index=False)
 write(out/'result.json',dict(status='trained_validation_only',config=cfg,best_epoch=best_epoch,metrics=metrics,contrast=contrast,seconds=time.monotonic()-started,test_evaluations=0))
def score(checkpoint,output,freeze_record):
 torch.set_num_threads(2);saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['config'];model=initialize(cfg,cfg['seed']);model.load_state_dict(saved['state_dict'])
 _,_,x,_,_=load(include_test=False,drop_fraction=cfg['additional_prefix_drop'],mask_seed=cfg['mask_seed']);out=Path(output);out.mkdir(parents=True,exist_ok=False)
 metrics,contrast=evaluate(model,x,cfg,out,include_test=True)
 write(out/'result.json',dict(status='evaluated',config=cfg,best_epoch=saved['epoch'],metrics=metrics,contrast=contrast,checkpoint_sha256=sha(checkpoint),freeze_record=str(freeze_record),test_evaluations=1))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--drop',type=float,default=0.);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();train(a.profile,a.seed,a.drop,a.output,a.device)
