"""Train prefix-conditioned models, freeze validation selection, then score future times."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
import forecast_data as data
import forecast_models as models

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/hypocotyl_forecast.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def state_sha(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(model,x,config,out,fixed_prediction=None,include_test=True):
    dc,plants,check_x,batches,raw=data.load(device='cpu',include_test=include_test,
        drop_fraction=config['additional_prefix_drop'],mask_seed=config['mask_seed'])
    for key in x:torch.testing.assert_close(x[key].detach().cpu(),check_x[key],rtol=0,atol=0)
    if model is not None:
        model.eval()
        with torch.no_grad():prediction=model(**x)['pred'].detach().cpu().numpy()
    else:prediction=np.asarray(fixed_prediction)
    metrics={split:data.score(prediction,batch,dc['height_scale_mm']) for split,batch in batches.items()}
    arrays=dict(prediction=prediction,plant_ids=plants.plant_id.to_numpy(dtype=str),genotype=plants.genotype.to_numpy(dtype=str),hours=data.HOURS,
        prefix_y=check_x['prefix_y'].numpy(),prefix_mask=check_x['prefix_mask'].numpy())
    for split,batch in batches.items():
        arrays[split+'_target']=batch['target'].numpy();arrays[split+'_mask']=batch['mask'].numpy()
    np.savez_compressed(out/'predictions.npz',**arrays)
    if include_test:
        horizon={}
        for hour in [60,72]:
            j=int(hour/3);m=arrays['test_mask'][:,j].astype(bool)
            error=(prediction[m,j].astype(float)-arrays['test_target'][m,j].astype(float))*dc['height_scale_mm']
            horizon[str(hour)]=dict(pooled_rmse=float(np.sqrt(np.mean(error**2))),n_plants=int(m.sum()))
        metrics['test_horizons']=horizon
    return metrics


def run(args):
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    profile=json.loads(CONFIG.read_text())
    if args.model not in profile['models']:raise ValueError(args.model)
    cfg=dict(profile['common'],**profile['models'][args.model]);cfg.update(model=args.model,seed=args.seed,
        additional_prefix_drop=args.drop_fraction,mask_seed=args.mask_seed,protocol=data.PROTOCOL,
        experiment_config_sha256=sha(CONFIG))
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    neural=args.model not in ['rf','logistic'];device=torch.device(args.device if neural else 'cpu')
    dc,plants,x,batches,raw=data.load(device=device,drop_fraction=args.drop_fraction,mask_seed=args.mask_seed)
    cfg.update(data=dc,device=str(device),torch_version=torch.__version__,cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
        n_plants=len(plants),height_scale_mm=dc['height_scale_mm'],data_config_sha256=sha(data.PREPARED/'config.json'))
    model=None
    if neural:
        model=models.build(args.model).to(device)
        for name,value in model.named_parameters():
            if 'lstm' in name and 'weight' in name and value.ndim>=2:torch.nn.init.orthogonal_(value)
        cfg.update(n_params=sum(p.numel() for p in model.parameters()),initial_state_sha256=state_sha(model))
    else:cfg['n_params']=len(plants)+2*len(data.GENOTYPES) if args.model=='logistic' else None
    out.mkdir(parents=True);write(out/'config.json',cfg)
    started=time.monotonic()
    if args.mode=='evaluate':
        if not neural or not args.checkpoint:raise ValueError('Evaluation requires a neural checkpoint')
        saved=torch.load(args.checkpoint,map_location=device,weights_only=False)
        if saved['config']['protocol']!=data.PROTOCOL or saved['config']['model']!=args.model:raise ValueError('Incompatible checkpoint')
        for key in ['additional_prefix_drop','mask_seed','height_scale_mm']:
            if saved['config'][key]!=cfg[key]:raise ValueError(f'Checkpoint {key} differs from requested input protocol')
        model.load_state_dict(saved['state_dict'])
        metrics=evaluate(model,x,cfg,out)
        write(out/'result.json',dict(config=cfg,status='evaluated',metrics=metrics,checkpoint_sha256=sha(args.checkpoint)))
        return
    if not neural:
        if args.mode=='smoke':raise ValueError('Use train for classical baselines')
        if args.model=='rf':
            import joblib
            pred,fitted=models.fit_forest(x,batches['train'],args.seed);joblib.dump(fitted,out/'forest.joblib')
        else:
            pred,params=models.fit_logistic(x,batches['train']);write(out/'parameters.json',dict(parameters=params))
        write(out/'selection.json',dict(fitted_on='0--36 h observations only',test_evaluations=0))
        metrics=evaluate(None,x,cfg,out,fixed_prediction=pred)
        write(out/'result.json',dict(config=cfg,status='complete',metrics=metrics,seconds=time.monotonic()-started))
        return
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'])
    steps=args.steps if args.mode=='smoke' else cfg['epochs'];best=float('inf');best_state=None;best_epoch=None
    recent=[];history=[]
    for epoch in range(1,steps+1):
        model.train();prediction=model(**x);data_loss=data.masked_rmse(prediction['pred'],batches['train']);loss=data_loss
        ode,capacity=data_loss.new_zeros(()),data_loss.new_zeros(())
        if cfg['lambda_ode']>0 or cfg['lambda_k']>0:
            ode,capacity=models.physics_losses(model,prediction,x)
            loss=loss+cfg['lambda_ode']*ode+cfg['lambda_k']*capacity
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if epoch==1 and cfg['lambda_ode']==cfg['lambda_k']==0 and hasattr(model,'ode_param_head'):
            assert all(p.grad is None for p in model.ode_param_head.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step()
        if epoch%cfg['eval_every']==0:
            model.eval()
            with torch.no_grad():value=float(data.masked_rmse(model(**x)['pred'],batches['val']))
            recent=(recent+[value])[-cfg['val_smooth']:];smooth=float(np.mean(recent))
            if epoch>=cfg['min_epoch'] and len(recent)==cfg['val_smooth'] and smooth<best:
                best=smooth;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            history.append(dict(epoch=epoch,data_loss=float(data_loss.detach()),objective=float(loss.detach()),
                ode=float(ode.detach()),capacity=float(capacity.detach()),val_rmse=value*dc['height_scale_mm'],
                selection_rmse=smooth*dc['height_scale_mm'],best_epoch=best_epoch))
        if epoch%100==0:
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            write(out/'progress.json',dict(epoch=epoch,epochs=steps,best_epoch=best_epoch,seconds=time.monotonic()-started,test_evaluations=0))
            print(f'{args.model}/drop{args.drop_fraction}/seed{args.seed}: {epoch}/{steps} val={value*dc["height_scale_mm"]:.4f}',flush=True)
    if args.mode=='smoke':
        metrics=evaluate(model,x,cfg,out,include_test=False);status='smoke_only_not_a_paper_result'
    else:
        if best_state is None:raise RuntimeError('No eligible validation checkpoint')
        model.load_state_dict(best_state)
        torch.save(dict(state_dict=best_state,config=cfg,epoch=best_epoch),out/'checkpoint.pt')
        write(out/'selection.json',dict(best_epoch=best_epoch,selection_rmse=best*dc['height_scale_mm'],
            checkpoint_sha256=sha(out/'checkpoint.pt'),test_evaluations=0))
        metrics=evaluate(model,x,cfg,out);status='complete'
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    write(out/'result.json',dict(config=cfg,status=status,epochs=steps,best_epoch=best_epoch,metrics=metrics,
        seconds=time.monotonic()-started,test_evaluations=int(args.mode!='smoke')))
    print(json.dumps(dict(status=status,metrics={k:v['rmse'] for k,v in metrics.items() if 'rmse' in v}),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','evaluate','smoke']);p.add_argument('--model',default='phytoode')
    p.add_argument('--seed',type=int,default=1);p.add_argument('--device',default='cpu')
    p.add_argument('--drop-fraction',type=float,default=0.);p.add_argument('--mask-seed',type=int,default=20260911)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
