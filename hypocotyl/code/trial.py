"""A validation-only training trial; held-out evaluation is a separate operation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch

from data import load, inputs, score, curve_rmse
from models import build, physics_losses
from process_baselines import fit_logistic, fit_forest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(args):
    config=json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    neural=config["model"] in ("phytoode","latent_ode","lstm","light_pinn")
    device=torch.device("cuda" if neural else "cpu")
    data_config, x, batches=load(args.protocol, device=device)
    meta=dict(config=config, seed=args.seed, protocol=args.protocol,
              data_config=data_config, cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
              test_targets_loaded=False)
    started=time.monotonic()
    if not neural:
        if config["model"]=="rf":
            predictions,forest=fit_forest(batches["train"],args.seed,config["min_samples_leaf"])
            joblib.dump(forest,args.output/"forest.joblib")
            n_params=None
        else:
            predictions,parameters=fit_logistic(batches["train"],config["model"]=="light_logistic")
            atomic(args.output/"parameters.json", dict(model=config["model"],parameters=parameters))
            n_params=25 if config["model"]=="light_logistic" else 30
        selected_epoch=None
    else:
        model=build(config["model"]).to(device)
        for name,value in model.named_parameters():
            if "lstm" in name and "weight" in name and value.ndim==2:
                torch.nn.init.orthogonal_(value)
        meta["initial_state_sha256"]=state_hash(model)
        meta["gpu_name"]=torch.cuda.get_device_name()
        n_params=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.Adam(model.parameters(), lr=config["lr"],weight_decay=1e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["epochs"])
        best=float("inf"); selected_epoch=-1; best_state=None
        history=[]; recent=[]
        for epoch in range(1,config["epochs"]+1):
            model.train()
            output=model(**x)
            data_loss=curve_rmse(output["pred"],batches["train"]["target"],batches["train"]["mask"])
            loss=data_loss
            ode=capacity=torch.zeros((),device=device)
            if config.get("lambda_ode",0)>0 or config.get("lambda_k",0)>0:
                ode,capacity=physics_losses(model,output,x)
                loss=loss+config["lambda_ode"]*ode+config["lambda_k"]*capacity
            if not torch.isfinite(loss):raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config["model"]=="latent_ode":
                assert all(p.grad is None for p in model.ode_param_head.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); schedule.step()
            if epoch % 20==0:
                model.eval()
                with torch.no_grad():
                    pred=model(**x)["pred"]
                    val=float(curve_rmse(pred,batches["val"]["target"],batches["val"]["mask"]))
                recent=(recent+[val])[-3:]
                smooth=float(np.mean(recent))
                if epoch>=200 and len(recent)==3 and smooth<best:
                    best=smooth; selected_epoch=epoch
                    best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
                history.append(dict(epoch=epoch,data_loss=float(data_loss.detach()),
                    ode_loss=float(ode.detach()),k_loss=float(capacity.detach()),
                    objective=float(loss.detach()),val_rmse=val*data_config["height_scale_mm"],
                    selection_rmse=smooth*data_config["height_scale_mm"],best_epoch=selected_epoch))
            if epoch%100==0:
                pd.DataFrame(history).to_csv(args.output/"history.csv",index=False)
                atomic(args.output/"progress.json",dict(epoch=epoch,epochs=config["epochs"],
                    seconds=time.monotonic()-started,best_epoch=selected_epoch,test_evaluations=0))
                print(f"{args.protocol} {config['model']} seed={args.seed} {epoch}/{config['epochs']} val={val*data_config['height_scale_mm']:.5f}",flush=True)
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():predictions=model(**x)["pred"].cpu().numpy()
        torch.save(dict(state_dict=best_state,config=config,epoch=selected_epoch),args.output/"checkpoint.pt")
        meta["checkpoint_sha256"]=sha(args.output/"checkpoint.pt")
    metrics={name:score(predictions,batch,data_config["height_scale_mm"]) for name,batch in batches.items()}
    np.savez_compressed(args.output/"predictions.npz",prediction=predictions,
        **{f"{name}_{key}":batch[key].detach().cpu().numpy() for name,batch in batches.items() for key in ("target","mask")})
    atomic(args.output/"config.json",meta)
    atomic(args.output/"result.json",dict(protocol=args.protocol,config=config,seed=args.seed,
        best_epoch=selected_epoch,n_params=n_params,metrics=metrics,seconds=time.monotonic()-started,
        initial_state_sha256=meta.get("initial_state_sha256"),checkpoint_sha256=meta.get("checkpoint_sha256"),
        test_evaluations=0))


def evaluate(args):
    if args.output.exists():raise FileExistsError(args.output)
    selections=json.loads(args.selection.read_text())
    assert selections["status"]=="all_validation_selections_frozen"
    record=next(r for r in selections["runs"] if r["protocol"]==args.protocol and r["model"]==args.model and r["seed"]==args.seed)
    folder=Path(record["path"])
    result=json.loads((folder/"result.json").read_text())
    config=result["config"]
    assert sha(folder/"result.json")==record["result_sha256"]
    data_config,x,batches=load(args.protocol,splits=("train","val","test"),device="cpu")
    with np.load(folder/"predictions.npz") as saved:predictions=saved["prediction"]
    args.output.mkdir(parents=True)
    # Validation trials already saved complete time-grid predictions without test targets.
    # Evaluate those exact arrays after all selections have frozen.
    metrics={name:score(predictions,batch,data_config["height_scale_mm"]) for name,batch in batches.items()}
    for split in ("train","val"):
        np.testing.assert_allclose(metrics[split]["rmse"],result["metrics"][split]["rmse"],rtol=1e-12)
    np.savez_compressed(args.output/"predictions.npz",prediction=predictions,
        **{f"{name}_{key}":batch[key].numpy() for name,batch in batches.items() for key in ("target","mask")})
    atomic(args.output/"result.json",dict(protocol=args.protocol,model=args.model,seed=args.seed,config=config,
        metrics=metrics,unit="mm",selection_sha256=sha(args.selection),source_result_sha256=sha(folder/"result.json"),
        n_params=result["n_params"],best_epoch=result["best_epoch"],test_evaluations=1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("train","evaluate"))
    parser.add_argument("--protocol",required=True,choices=("replicate","time_holdout"))
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--selection",type=Path)
    parser.add_argument("--model")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    (train if args.mode=="train" else evaluate)(args)


if __name__=="__main__":main()
