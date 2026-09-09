"""Train a fresh primary-protocol candidate using training and validation only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from data import load, score, curve_rmse
from models import physics_losses
from retune_model import build
from trial import atomic, sha, state_hash


def train(config_path, output_path, seed):
    config = json.loads(config_path.read_text())
    output_path.mkdir(parents=True, exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    data_config, x, batches = load('replicate', device=device)
    model = build(config).to(device)
    for name, value in model.named_parameters():
        if 'lstm' in name and 'weight' in name and value.ndim == 2:
            torch.nn.init.orthogonal_(value)
    initial_hash = state_hash(model)
    started = time.monotonic()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    history, recent = [], []
    best = float('inf'); best_epoch = None; best_state = None
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        output = model(**x)
        data_loss = curve_rmse(output['pred'], batches['train']['target'], batches['train']['mask'])
        ode = capacity = torch.zeros((), device=device)
        if config['lambda_ode'] > 0 or config['lambda_k'] > 0:
            ode, capacity = physics_losses(model, output, x)
        loss = data_loss + config['lambda_ode'] * ode + config['lambda_k'] * capacity
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config['model'] == 'latent_ode':
            assert all(p.grad is None for p in model.ode_param_head.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(**x)['pred']
                val = float(curve_rmse(pred, batches['val']['target'], batches['val']['mask']))
            recent = (recent + [val])[-3:]
            smooth = float(np.mean(recent))
            if epoch >= 200 and len(recent) == 3 and smooth < best:
                best = smooth; best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append(dict(epoch=epoch, data_loss=float(data_loss.detach()),
                ode_loss=float(ode.detach()), k_loss=float(capacity.detach()),
                objective=float(loss.detach()), val_rmse=val*data_config['height_scale_mm'],
                selection_rmse=smooth*data_config['height_scale_mm'], best_epoch=best_epoch))
        if epoch % 100 == 0:
            pd.DataFrame(history).to_csv(output_path/'history.csv', index=False)
            atomic(output_path/'progress.json', dict(epoch=epoch, epochs=config['epochs'],
                seconds=time.monotonic()-started, best_epoch=best_epoch, test_evaluations=0))
        if epoch % 500 == 0:
            print(f"{config['model']} seed {seed}: {epoch}/{config['epochs']}, val {val*data_config['height_scale_mm']:.6f}", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        predictions = model(**x)['pred'].cpu().numpy()
    torch.save(dict(state_dict=best_state, config=config, epoch=best_epoch), output_path/'checkpoint.pt')
    np.savez_compressed(output_path/'predictions.npz', prediction=predictions,
        **{f'{name}_{key}': batch[key].cpu().numpy() for name, batch in batches.items()
           for key in ('target', 'mask')})
    metrics = {name: score(predictions, batch, data_config['height_scale_mm']) for name, batch in batches.items()}
    atomic(output_path/'config.json', dict(config=config, seed=seed, protocol='replicate',
        data_config=data_config, cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
        gpu_name=torch.cuda.get_device_name(), initial_state_sha256=initial_hash,
        test_targets_loaded=False))
    atomic(output_path/'result.json', dict(protocol='replicate', config=config, seed=seed,
        best_epoch=best_epoch, n_params=sum(p.numel() for p in model.parameters()), metrics=metrics,
        seconds=time.monotonic()-started, initial_state_sha256=initial_hash,
        checkpoint_sha256=sha(output_path/'checkpoint.pt'), test_evaluations=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    train(args.config, args.output, args.seed)
