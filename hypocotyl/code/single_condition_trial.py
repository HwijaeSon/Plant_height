"""Fresh 12L12D-only fits to split-specific replicate means; no environmental features."""
from __future__ import annotations
import json
import os
from pathlib import Path
import random
import time
import joblib
import numpy as np
import pandas as pd
import torch
from single_condition_models import build, physics_losses
from single_condition_data import load, curve_rmse, score, save_targets
from single_condition_baselines import fit_forest, fit_logistic
from run_single_condition import atomic, sha
import hashlib


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(job):
    config_path, out = Path(job['config_path']), Path(job['output'])
    config = json.loads(config_path.read_text())
    protocol, seed = job['protocol'], job['seed']
    out.mkdir(parents=True, exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(2)
    neural = config['model'] in ['phytoode', 'latent_ode', 'logistic_pinn', 'lstm']
    device = 'cuda' if neural else 'cpu'
    data_config, x, batches = load(protocol, device=device)
    scale = data_config['height_scale_mm']
    start = time.monotonic(); initial_hash = None; checkpoint_hash = None
    best_epoch = None; history = []; best = float('inf'); best_state = None; recent = []
    if not neural:
        if config['model'] == 'rf':
            predictions, forest = fit_forest(batches['train'], seed, config['min_samples_leaf'])
            joblib.dump(forest, out/'forest.joblib'); n_params = None
        else:
            predictions, parameters = fit_logistic(batches['train'])
            atomic(out/'parameters.json', dict(model=config['model'], parameters=parameters))
            n_params = 15
    else:
        model = build(config).to(device)
        for name, parameter in model.named_parameters():
            if 'lstm' in name and 'weight' in name and parameter.ndim == 2:
                torch.nn.init.orthogonal_(parameter)
        n_params = sum(p.numel() for p in model.parameters())
        initial_hash = state_hash(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
        for epoch in range(1, config['epochs']+1):
            model.train(); output = model(**x)
            data_loss = curve_rmse(output['pred'], batches['train'])
            ode = capacity = torch.zeros((), device=device)
            if config['lambda_ode'] > 0 or config['lambda_k'] > 0:
                ode, capacity = physics_losses(model, output, x)
            loss = data_loss + config['lambda_ode']*ode + config['lambda_k']*capacity
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if config['model'] == 'latent_ode':
                assert all(p.grad is None for p in model.ode_param_head.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step()
            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    val = float(curve_rmse(model(**x)['pred'], batches['val']))
                recent = (recent+[val])[-3:]; smooth = sum(recent)/len(recent)
                if epoch >= 200 and len(recent) == 3 and smooth < best:
                    best = smooth; best_epoch = epoch
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                history.append(dict(epoch=epoch, data_loss=float(data_loss.detach()),
                    ode_loss=float(ode.detach()), k_loss=float(capacity.detach()),
                    objective=float(loss.detach()), val_rmse=val*scale, selection_rmse=smooth*scale))
            if epoch % 200 == 0:
                pd.DataFrame(history).to_csv(out/'history.csv', index=False)
                atomic(out/'progress.json', dict(epoch=epoch, epochs=config['epochs'],
                    seconds=time.monotonic()-start, test_evaluations=0))
        assert best_state is not None
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad(): predictions = model(**x)['pred'].cpu().numpy()
        torch.save(dict(state_dict=best_state, config=config, epoch=best_epoch), out/'checkpoint.pt')
        checkpoint_hash = sha(out/'checkpoint.pt')
    metrics = {name:score(predictions, b, scale) for name,b in batches.items()}
    np.savez_compressed(out/'predictions.npz', prediction=predictions, **save_targets(batches))
    atomic(out/'result.json', dict(protocol=protocol, model=config['model'], config=config, seed=seed,
        target_type='split_specific_replicate_means_12L12D_only', n_params=n_params, best_epoch=best_epoch,
        metrics=metrics, seconds=time.monotonic()-start, initial_state_sha256=initial_hash,
        checkpoint_sha256=checkpoint_hash, test_evaluations=0,
        cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES')))
    return dict(protocol=protocol, key=job['key'], seed=seed, val_rmse=metrics['val']['rmse'],
                seconds=time.monotonic()-start)


def evaluate(selection_path, record, out):
    root = Path(__file__).resolve().parents[2]
    selection = json.loads(selection_path.read_text())
    assert selection['status'] == 'all_validation_selections_frozen'
    assert record in selection['runs']
    source = root/record['path']
    assert sha(source/'result.json') == record['result_sha256']
    trained = json.loads((source/'result.json').read_text())
    assert trained['test_evaluations'] == 0
    config, x, batches = load(record['protocol'], splits=('train', 'val', 'test'))
    saved = np.load(source/'predictions.npz')
    assert not any(k.startswith('test_') for k in saved.files)
    predictions = saved['prediction']
    metrics = {name:score(predictions, b, config['height_scale_mm']) for name,b in batches.items()}
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'predictions.npz', prediction=predictions, **save_targets(batches))
    atomic(out/'result.json', dict(protocol=record['protocol'], model=record['model'], seed=record['seed'],
        metrics=metrics, target_type='split_specific_replicate_means_12L12D_only', test_evaluations=1,
        selection_sha256=sha(selection_path), source_result_sha256=sha(source/'result.json')))
