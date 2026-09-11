"""Train individual-parameter PhytoODE without loading or scoring test targets."""
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
from forecast_models import physics_losses
from forecast_trial import evaluate, sha, state_sha, write
from prefix_parameter_model import PrefixParameterPhytoODE

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'configs/hypocotyl_prefix_parameters_20260911.json'


def initialize(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PrefixParameterPhytoODE().to(device)
    for name, value in model.named_parameters():
        if 'lstm' in name and 'weight' in name and value.ndim >= 2:
            torch.nn.init.orthogonal_(value)
    return model


def evaluate_checkpoint(checkpoint, output, device='cpu', selection_record=None):
    """Explicit final scoring after the search runner freezes its selection."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = saved['config']
    if cfg['model'] != 'phytoode_prefix_parameters' or cfg['lambda_k'] != 0:
        raise ValueError('Incompatible model configuration')
    torch.set_num_threads(2)
    model = initialize(cfg['seed'], device)
    model.load_state_dict(saved['state_dict'])
    dc, plants, x, _, _ = data.load(device=device, drop_fraction=cfg['additional_prefix_drop'],
                                   mask_seed=cfg['mask_seed'])
    if dc['height_scale_mm'] != cfg['height_scale_mm']:
        raise ValueError('Input scale differs from checkpoint')
    metrics = evaluate(model, x, cfg, output, include_test=True)
    with torch.no_grad():
        prediction = model(**x)
    parameters = plants[['plant_id', 'genotype', 'source_row']].copy()
    parameters['r_per_hour'] = prediction['r'].cpu().numpy()
    parameters['K_mm'] = prediction['K'].cpu().numpy() * cfg['height_scale_mm']
    parameters['prefix_observations'] = x['prefix_mask'].sum(1).cpu().numpy().astype(int)
    parameters.to_csv(output / 'individual_parameters.csv', index=False)
    write(output / 'result.json', dict(status='complete', config=cfg, best_epoch=saved['epoch'],
        checkpoint_sha256=sha(checkpoint), selection_record=selection_record,
        metrics=metrics, test_evaluations=1))


def run(args):
    if args.mode == 'evaluate':
        if args.checkpoint is None:
            raise ValueError('Specify --checkpoint')
        evaluate_checkpoint(args.checkpoint, args.output, args.device)
        return
    profile = json.loads(PROFILE.read_text())
    if args.lambda_ode not in profile['lambda_ode_grid']:
        raise ValueError('Coefficient outside the frozen search grid')
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    cfg = dict(profile['common'], model=profile['model'], experiment=profile['experiment'],
        protocol=data.PROTOCOL, seed=args.seed, lambda_ode=args.lambda_ode, lambda_k=0.,
        additional_prefix_drop=args.drop_fraction, mask_seed=20260910+args.seed,
        parameter_head_inputs=profile['parameter_head_inputs'],
        experiment_config_sha256=sha(PROFILE), device=args.device,
        torch_version=torch.__version__, cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'))
    dc, plants, x, batches, _ = data.load(device=args.device, drop_fraction=args.drop_fraction,
                                         mask_seed=cfg['mask_seed'])
    cfg.update(data=dc, height_scale_mm=dc['height_scale_mm'], n_plants=len(plants),
               data_config_sha256=sha(data.PREPARED / 'config.json'))
    model = initialize(args.seed, args.device)
    cfg.update(n_params=sum(p.numel() for p in model.parameters()),
               initial_state_sha256=state_sha(model))
    out.mkdir(parents=True)
    write(out / 'config.json', cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    epochs = args.steps if args.mode == 'smoke' else cfg['epochs']
    best, best_epoch, best_state = float('inf'), None, None
    recent, history = [], []
    started = time.monotonic()
    for epoch in range(1, epochs+1):
        model.train()
        prediction = model(**x)
        data_loss = data.masked_rmse(prediction['pred'], batches['train'])
        ode, capacity_diagnostic = physics_losses(model, prediction, x)
        # The finite-window capacity term is deliberately absent from the objective.
        loss = data_loss + cfg['lambda_ode'] * ode
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss: epoch {epoch}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if epoch == 1:
            grad = model.ode_param_head.net[0].weight.grad[:, 4:]
            if grad is None or not torch.isfinite(grad).all() or not (grad.abs().sum() > 0):
                raise AssertionError('No physics gradient reaches the prefix parameter head')
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if epoch % cfg['eval_every'] == 0:
            model.eval()
            with torch.no_grad():
                value = float(data.masked_rmse(model(**x)['pred'], batches['val']))
            recent = (recent+[value])[-cfg['val_smooth']:]
            smooth = float(np.mean(recent))
            if epoch >= cfg['min_epoch'] and len(recent) == cfg['val_smooth'] and smooth < best:
                best, best_epoch = smooth, epoch
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            history.append(dict(epoch=epoch, data_loss=float(data_loss.detach()),
                objective=float(loss.detach()), ode=float(ode.detach()),
                capacity_diagnostic=float(capacity_diagnostic.detach()), capacity_penalty_used=False,
                val_rmse=value*cfg['height_scale_mm'], selection_rmse=smooth*cfg['height_scale_mm'],
                best_epoch=best_epoch))
        if epoch % 100 == 0:
            pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
            write(out / 'progress.json', dict(epoch=epoch, epochs=epochs, best_epoch=best_epoch,
                seconds=time.monotonic()-started, test_evaluations=0))
            print(f"lambda={cfg['lambda_ode']} drop={args.drop_fraction} seed={args.seed}: {epoch}/{epochs}", flush=True)
    if args.mode == 'smoke':
        status = 'smoke_only_not_a_paper_result'
    else:
        if best_state is None:
            raise RuntimeError('No eligible validation checkpoint')
        model.load_state_dict(best_state)
        torch.save(dict(state_dict=best_state, config=cfg, epoch=best_epoch), out / 'checkpoint.pt')
        write(out / 'selection.json', dict(best_epoch=best_epoch,
            selection_rmse=best*cfg['height_scale_mm'], checkpoint_sha256=sha(out / 'checkpoint.pt'),
            test_evaluations=0))
        status = 'trained_validation_only'
    metrics = evaluate(model, x, cfg, out, include_test=False)
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    write(out / 'result.json', dict(status=status, config=cfg, best_epoch=best_epoch,
        metrics=metrics, seconds=time.monotonic()-started, test_evaluations=0))
    print(json.dumps(dict(status=status, validation_rmse=metrics['val']['rmse'])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'smoke', 'evaluate'])
    p.add_argument('--lambda-ode', type=float, default=500.)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--drop-fraction', type=float, default=0.)
    p.add_argument('--device', default='cpu')
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
