"""Train or evaluate the manuscript's fixed PhytoODE configurations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
DATASETS = ("wheat", "maize", "arabidopsis", "hypocotyl")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Problem:
    def __init__(self, dataset, name, device, seed):
        self.dataset, self.name, self.device = dataset, name, device
        profile = json.loads((ROOT / "configs/paper.json").read_text())["datasets"][dataset]
        if name not in profile["models"]:
            raise ValueError(f"{dataset}: choose a model from {list(profile['models'])}")
        self.reference = profile["models"][name]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if dataset == "hypocotyl":
            sys.path.insert(0, str(ROOT / "hypocotyl/code"))
            import single_condition_data as data
            import single_condition_models as models
            import light_input_model as light
            self.data_module, self.models_module = data, models
            self.data_config, self.x, self.batches = data.load("replicate", device=device)
            self.scale = self.data_config["height_scale_mm"]
            self.config = dict(self.reference["config"])
            self.neural = self.config["model"] not in ("rf", "logistic")
            if self.config["model"] == "phytoode_light":
                self.x, self.model = light.inputs(device=device), light.build(self.config)
            elif self.neural:
                self.model = models.build(self.config)
            self.config.update(eval_every=20, min_epoch=200, val_smooth=3, unit="mm")
        else:
            sys.path.insert(0, str(ROOT / "experiments"))
            import physics_ablation as original
            self.original = original
            self.core, self.loss_fn, model_args, self.config, self.raw, self.batches = original.setup(dataset, device)
            self.scale = self.config["scale"]
            self.neural = True
            self.model = self.core.LatentODEHeightModel(**model_args)
            self.config.update(lambda_ode=profile["lambda_ode"], lambda_k=profile["lambda_k"])
            if name == "latent_ode":
                self.config.update(lambda_ode=0., lambda_k=0.)
            elif name == "capacity_only":
                self.config.update(lambda_ode=0.)
        if self.neural:
            # Preserve CPU recurrent initialization for maize and device-side
            # recurrent initialization for the other manuscript experiments.
            if dataset != "maize":
                self.model = self.model.to(device)
            for key, value in self.model.named_parameters():
                if "lstm" in key and "weight" in key and value.ndim >= 2:
                    torch.nn.init.orthogonal_(value)
            if dataset == "maize":
                self.model = self.model.to(device)
            self.n_params = sum(p.numel() for p in self.model.parameters())
        else:
            self.n_params = None
        self.config["scale"] = self.scale
        if self.neural:
            self.config["effective_weights"] = dict(physic=self.config["lambda_ode"],
                ymax=self.config["lambda_k"], r=0., mono=0.)
            if self.dataset != "hypocotyl":
                self.config["physic"] = self.config["lambda_ode"]
                self.config["ymax"] = self.config["lambda_k"]
        self.config.update(dataset=dataset, model_name=name, seed=seed, device=str(device),
                           torch_version=torch.__version__, n_params=self.n_params)

    def forward(self, batch=None):
        if self.dataset == "hypocotyl":
            return self.model(**self.x)
        return self.original.forward(self.model, batch)

    def data_loss(self, output, batch):
        if self.dataset == "hypocotyl":
            return self.data_module.curve_rmse(output["pred"], batch)
        return self.loss_fn(output["pred"], batch["y"], batch["mask"])

    def objective(self, output, batch):
        data_loss = self.data_loss(output, batch)
        ode, capacity = self.config["lambda_ode"], self.config["lambda_k"]
        total = data_loss
        if ode > 0 or capacity > 0:
            if self.dataset == "hypocotyl":
                residual, maximum = self.models_module.physics_losses(self.model, output, self.x)
                total = total + ode * residual + capacity * maximum
            else:
                weights = dict(physic=ode, ymax=capacity, r=0., mono=0.)
                components = self.core.physics_losses(self.model, output, batch["env"], weights)
                total = total + sum(components.values())
        return data_loss, total

    def training_batch(self):
        batch = self.batches["train"]
        if self.dataset != "hypocotyl" and self.config["permute_training"]:
            index = torch.randperm(len(batch["y"]), device=self.device)
            batch = {key: value[index] for key, value in batch.items()}
        return batch

    def evaluate(self, out, include_test=True, fixed_prediction=None):
        splits = ("train", "val", "test") if include_test else ("train", "val")
        predictions, metrics = {}, {}
        if self.neural:
            self.model.eval()
        with torch.no_grad():
            if self.dataset == "hypocotyl":
                _, _, batches = self.data_module.load("replicate", splits=splits)
                prediction = fixed_prediction if fixed_prediction is not None else self.forward()["pred"].cpu().numpy()
                metrics = {key: self.data_module.score(prediction, batch, self.scale) for key, batch in batches.items()}
                predictions = dict(prediction=prediction, **self.data_module.save_targets(batches))
            else:
                for split in splits:
                    key = "train_eval" if split == "train" else split
                    raw = self.raw[key]
                    batch = {name: torch.as_tensor(raw[name], dtype=torch.long if name == "g_idx" else torch.float32,
                                                   device=self.device) for name in ("g_idx", "env", "s", "ds")}
                    prediction = self.forward(batch)["pred"].cpu().numpy()
                    metrics[split], _ = self.original.score(prediction, raw, self.scale)
                    predictions[key] = prediction
                    for name in ("y", "mask", "genotype"):
                        predictions[key + ("_target" if name == "y" else "_" + name)] = raw[name]
        np.savez_compressed(out / "predictions.npz", **predictions)
        return metrics


def train(problem, args):
    cfg, out = problem.config, args.output
    if not problem.neural:
        if args.mode == "smoke":
            raise ValueError("Use train for the inexpensive process/RF baselines")
        import single_condition_baselines as baselines
        if cfg["model"] == "rf":
            import joblib
            prediction, fitted = baselines.fit_forest(problem.batches["train"], args.seed, cfg["min_samples_leaf"])
            joblib.dump(fitted, out / "forest.joblib")
        else:
            prediction, parameters = baselines.fit_logistic(problem.batches["train"])
            write_json(out / "parameters.json", {"parameters": parameters})
        metrics = problem.evaluate(out, fixed_prediction=prediction)
        write_json(out / "result.json", dict(config=cfg, status="complete", metrics=metrics))
        return
    epochs = args.steps if args.mode == "smoke" else cfg["epochs"]
    optimizer = torch.optim.Adam(problem.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    history, recent, best, best_state, best_epoch = [], [], float("inf"), None, None
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        problem.model.train()
        batch = problem.training_batch()
        data_loss, loss = problem.objective(problem.forward(batch), batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["lambda_ode"] == cfg["lambda_k"] == 0 and hasattr(problem.model, "ode_param_head"):
            assert all(p.grad is None for p in problem.model.ode_param_head.parameters())
        torch.nn.utils.clip_grad_norm_(problem.model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if epoch % cfg["eval_every"] == 0:
            problem.model.eval()
            with torch.no_grad():
                value = float(problem.data_loss(problem.forward(problem.batches["val"]), problem.batches["val"]))
            recent = (recent + [value])[-cfg["val_smooth"]:]
            smooth = float(np.mean(recent))
            if epoch >= cfg["min_epoch"] and len(recent) == cfg["val_smooth"] and smooth < best:
                best, best_epoch = smooth, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in problem.model.state_dict().items()}
                torch.save(dict(state_dict=best_state, config=cfg, epoch=epoch), out / "checkpoint.pt")
            history.append(dict(epoch=epoch, data_loss=float(data_loss.detach()), objective=float(loss.detach()),
                                val_rmse=value * problem.scale, selection_rmse=smooth * problem.scale))
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if epoch % 100 == 0:
            print(f"{problem.dataset}/{problem.name} seed={args.seed} epoch={epoch}/{epochs}", flush=True)
    if args.mode == "smoke":
        metrics = problem.evaluate(out, include_test=False)
        status = "smoke_only_not_a_paper_result"
    else:
        if best_state is None:
            raise RuntimeError("No checkpoint satisfied the validation-selection rule")
        problem.model.load_state_dict(best_state)
        write_json(out / "selection.json", dict(best_epoch=best_epoch,
                   checkpoint_sha256=sha(out / "checkpoint.pt"), test_evaluations=0))
        metrics = problem.evaluate(out)
        status = "complete"
    write_json(out / "result.json", dict(config=cfg, status=status, epochs=epochs,
               best_epoch=best_epoch, seconds=time.monotonic()-started, metrics=metrics,
               test_evaluations=int(args.mode != "smoke")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate", "smoke"))
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", default="phytoode")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N (visible device numbering)")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a newly trained checkpoint instead of the bundled one")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2, help="Training iterations in smoke mode only")
    args = parser.parse_args()
    if args.seed < 0 or args.steps < 1:
        parser.error("seed must be nonnegative and steps positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or install a matching CUDA PyTorch build")
    torch.set_num_threads(2)
    problem = Problem(args.dataset, args.model, device, args.seed)
    if not problem.neural and device.type != "cpu":
        parser.error("Process and random-forest baselines require --device cpu")
    args.output.mkdir(parents=True)
    write_json(args.output / "config.json", problem.config)
    if args.mode == "evaluate":
        if not problem.neural:
            parser.error("Use train to fit the process/RF baselines; their saved predictions are bundled")
        if args.checkpoint is None:
            record = problem.reference["checkpoints"].get(str(args.seed))
            if record is None:
                parser.error("No bundled checkpoint for this seed; pass --checkpoint")
            checkpoint = ROOT / record["path"]
            if sha(checkpoint) != record["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {checkpoint}")
        else:
            checkpoint = args.checkpoint
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        problem.model.load_state_dict(saved.get("state_dict", saved))
        metrics = problem.evaluate(args.output)
        write_json(args.output / "result.json", dict(config=problem.config, status="evaluated",
                   checkpoint_sha256=sha(checkpoint), metrics=metrics))
    else:
        train(problem, args)
    result = json.loads((args.output / "result.json").read_text())
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
