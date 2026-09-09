"""PhytoODE-only tuning with complete validation histories and deferred test scoring."""
from pathlib import Path
import hashlib
import json
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from single_condition_data import load,curve_rmse,score,save_targets
from light_input_model import build,inputs,physics_losses
from run_light_input import atomic,sha


def state_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(job):
    config=json.loads(Path(job['config_path']).read_text()); out=Path(job['output'])
    out.mkdir(parents=True,exist_ok=False)
    seed=job['seed']; protocol=job['protocol']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(2)
    data_config,_,batches=load(protocol,device='cuda'); x=inputs(device='cuda')
    scale=data_config['height_scale_mm']; started=time.monotonic()
    model=build(config).cuda()
    for name,parameter in model.named_parameters():
        if 'lstm' in name and 'weight' in name and parameter.ndim==2: torch.nn.init.orthogonal_(parameter)
    initial_hash=state_hash(model); n_params=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'])
    history=[]; recent=[]; best=float('inf'); best_epoch=None; best_state=None
    for epoch in range(1,config['epochs']+1):
        model.train(); output=model(**x)
        data_loss=curve_rmse(output['pred'],batches['train']); ode,capacity=physics_losses(model,output,x)
        loss=data_loss+config['lambda_ode']*ode+config['lambda_k']*capacity
        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        if epoch%20==0:
            model.eval()
            with torch.no_grad(): val=float(curve_rmse(model(**x)['pred'],batches['val']))
            recent=(recent+[val])[-3:]; smooth=sum(recent)/len(recent)
            if epoch>=200 and len(recent)==3 and smooth<best:
                best=smooth; best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            history.append(dict(epoch=epoch,data_loss=float(data_loss.detach()),ode_loss=float(ode.detach()),
                k_loss=float(capacity.detach()),objective=float(loss.detach()),val_rmse=val*scale,
                selection_rmse=smooth*scale,val_rmse_normalized=val,selection_rmse_normalized=smooth))
        if epoch%200==0 or epoch==config['epochs']:
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            atomic(out/'progress.json',dict(epoch=epoch,epochs=config['epochs'],seconds=time.monotonic()-started,
                                           best_epoch=best_epoch,test_evaluations=0))
    assert best_state is not None and history[-1]['epoch']==config['epochs']
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): prediction=model(**x)['pred'].cpu().numpy()
    torch.save(dict(state_dict=best_state,config=config,epoch=best_epoch),out/'checkpoint.pt')
    metrics={s:score(prediction,b,scale) for s,b in batches.items()}
    np.savez_compressed(out/'predictions.npz',prediction=prediction,**save_targets(batches))
    result=dict(protocol=protocol,model=config['model'],config=config,seed=seed,
        target_type='split_specific_replicate_means_12L12D_only',input_features=['genotype','elapsed_time','binary_light'],
        n_params=n_params,best_epoch=best_epoch,best_smoothed_validation_normalized=best,
        metrics=metrics,seconds=time.monotonic()-started,initial_state_sha256=initial_hash,
        checkpoint_sha256=sha(out/'checkpoint.pt'),history_sha256=sha(out/'history.csv'),
        history_last_epoch=history[-1]['epoch'],test_evaluations=0,cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'))
    atomic(out/'result.json',result)
    return dict(protocol=protocol,key=job['key'],seed=seed,val_rmse=metrics['val']['rmse'],seconds=result['seconds'])


def evaluate(selection_path,record,out):
    root=Path(__file__).resolve().parents[2]
    selection=json.loads(selection_path.read_text())
    assert selection['status']=='all_validation_selections_frozen' and record in selection['runs']
    source=root/record['path']; trained=json.loads((source/'result.json').read_text())
    assert trained['test_evaluations']==0 and sha(source/'result.json')==record['result_sha256']
    config,_,batches=load(record['protocol'],splits=('train','val','test'))
    with np.load(source/'predictions.npz') as saved:
        assert not any(k.startswith('test_') for k in saved.files)
        prediction=saved['prediction']
    metrics={s:score(prediction,b,config['height_scale_mm']) for s,b in batches.items()}
    for split in ['train','val']: assert metrics[split]==trained['metrics'][split]
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'predictions.npz',prediction=prediction,**save_targets(batches))
    atomic(out/'result.json',dict(protocol=record['protocol'],model=record['model'],seed=record['seed'],
        metrics=metrics,target_type='split_specific_replicate_means_12L12D_only',test_evaluations=1,
        selection_sha256=sha(selection_path),source_result_sha256=sha(source/'result.json')))
