from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor

from models import (
    ReferenceLSTM,
    SequenceBatch,
    curve_mae,
    curve_rmse,
    initialize_logistic_head,
    logi_pinn_losses,
    parameter_count,
    reference_l2_loss,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA = ROOT / "data/processed/stem_length_long.csv"
DEFAULT_RESULTS = ROOT / "results/reproduced"
DAYS = np.arange(27, 50, dtype=np.int64)
TRAIN_PLANTS = (1, 2, 3, 4, 5, 6)
VAL_PLANTS = (7, 8)
TEST_PLANTS = (9, 10)
MODEL_ORDER = [
    "Logi-ODE",
    "Temp-ODE",
    "RF",
    "LSTM-NN",
    "Logi-PINN",
    "Latent Neural ODE (ours)",
]


@dataclass
class Dataset:
    train: SequenceBatch
    train_eval: SequenceBatch
    val: SequenceBatch
    test: SequenceBatch
    genotypes: list[str]
    temperature_mean: float
    temperature_std: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sequence_batch(
    frame: pd.DataFrame,
    genotypes: list[str],
    plant_numbers: tuple[int, ...],
    average_replicates: bool,
    temperature_mean: float,
    temperature_std: float,
) -> SequenceBatch:
    selected = frame[frame["plant_number"].isin(plant_numbers)].copy()
    keys = ["genotype", "condition"]
    if not average_replicates:
        keys.append("plant_number")

    genotype_lookup = {genotype: index for index, genotype in enumerate(genotypes)}
    genotype_indices: list[int] = []
    genotype_names: list[str] = []
    conditions: list[str] = []
    environments: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for key, group in selected.groupby(keys, sort=True):
        genotype, condition = key[:2]
        if average_replicates:
            curve = group.groupby("day_after_sowing", as_index=False).agg(
                length_m=("length_m", "mean"), temperature_c=("temperature_c", "mean")
            )
        else:
            curve = group[["day_after_sowing", "length_m", "temperature_c"]].copy()
        target = np.zeros(len(DAYS), dtype=np.float32)
        mask = np.zeros(len(DAYS), dtype=np.float32)
        for row in curve.itertuples(index=False):
            position = int(row.day_after_sowing) - int(DAYS[0])
            if not 0 <= position < len(DAYS):
                raise ValueError(f"Day {row.day_after_sowing} is outside the common grid")
            target[position] = float(row.length_m)
            mask[position] = 1.0
        temperature = float(curve["temperature_c"].iloc[0])
        environment = np.full(
            len(DAYS), (temperature - temperature_mean) / temperature_std, dtype=np.float32
        )
        genotype_indices.append(genotype_lookup[str(genotype)])
        genotype_names.append(str(genotype))
        conditions.append(str(condition))
        environments.append(environment)
        targets.append(target)
        masks.append(mask)

    return SequenceBatch(
        genotype_index=torch.tensor(genotype_indices, dtype=torch.long),
        condition=conditions,
        genotype=genotype_names,
        environment=torch.tensor(np.stack(environments), dtype=torch.float32),
        time=torch.linspace(0.0, 1.0, len(DAYS), dtype=torch.float32),
        target=torch.tensor(np.stack(targets), dtype=torch.float32),
        mask=torch.tensor(np.stack(masks), dtype=torch.float32),
    )


def load_dataset(path: Path) -> Dataset:
    frame = pd.read_csv(path)
    required = {
        "genotype",
        "condition",
        "plant_number",
        "day_after_sowing",
        "temperature_c",
        "length_m",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Processed data is missing columns: {sorted(missing)}")
    if len(frame) != 720:
        raise ValueError(f"Expected 720 rows, found {len(frame)}")
    genotypes = sorted(frame["genotype"].astype(str).unique().tolist())
    training = frame[frame["plant_number"].isin(TRAIN_PLANTS)]
    temperature_mean = float(training["temperature_c"].mean())
    temperature_std = max(float(training["temperature_c"].std(ddof=0)), 1e-6)

    def make(ids: tuple[int, ...], average: bool) -> SequenceBatch:
        return _sequence_batch(
            frame, genotypes, ids, average, temperature_mean, temperature_std
        )

    dataset = Dataset(
        train=make(TRAIN_PLANTS, False),
        train_eval=make(TRAIN_PLANTS, True),
        val=make(VAL_PLANTS, True),
        test=make(TEST_PLANTS, True),
        genotypes=genotypes,
        temperature_mean=temperature_mean,
        temperature_std=temperature_std,
    )
    if len(dataset.train.genotype) != 108:
        raise ValueError("Expected 108 training individuals")
    for split in [dataset.train_eval, dataset.val, dataset.test]:
        if len(split.genotype) != 18 or not torch.all(split.mask.sum(dim=1) == 4):
            raise ValueError("Each averaged evaluation split must have 18 four-point curves")
    return dataset


def evaluate_prediction(prediction: torch.Tensor, batch: SequenceBatch) -> tuple[float, float]:
    prediction = prediction.detach().cpu()
    return (
        float(curve_rmse(prediction, batch.target.cpu(), batch.mask.cpu())),
        float(curve_mae(prediction, batch.target.cpu(), batch.mask.cpu())),
    )


def logistic_solution(days: np.ndarray, r: np.ndarray, K: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    h0 = 1e-4
    exponent = np.clip(-r * multiplier * days, -80.0, 80.0)
    return K / (1.0 + (K / h0 - 1.0) * np.exp(exponent))


def paper_temperature_response(temperature_c: np.ndarray, lower_k: float, upper_k: float) -> np.ndarray:
    temperature_k = temperature_c + 273.15
    lower_arrhenius = 2000.0
    upper_arrhenius = 60000.0
    low = np.exp(np.clip(lower_arrhenius / temperature_k - lower_arrhenius / lower_k, -60, 60))
    high = np.exp(np.clip(upper_arrhenius / upper_k - upper_arrhenius / temperature_k, -60, 60))
    return 1.0 / (1.0 + low + high)


def batch_metadata(batch: SequenceBatch, dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    genotype = batch.genotype_index.cpu().numpy()
    temperatures = batch.environment[:, 0].cpu().numpy() * dataset.temperature_std + dataset.temperature_mean
    days = DAYS.astype(np.float64)
    return genotype, temperatures, days


def fit_process_baseline(
    dataset: Dataset, temperature_informed: bool
) -> tuple[dict[str, torch.Tensor], int, dict[str, np.ndarray]]:
    batch = dataset.train
    genotype, temperatures, days = batch_metadata(batch, dataset)
    observed_rows: list[tuple[int, float, float, float]] = []
    target = batch.target.numpy()
    mask = batch.mask.numpy().astype(bool)
    for row in range(len(batch.genotype)):
        for column in np.where(mask[row])[0]:
            observed_rows.append(
                (int(genotype[row]), float(temperatures[row]), float(days[column]), float(target[row, column]))
            )
    obs = np.asarray(observed_rows, dtype=np.float64)
    n_genotypes = len(dataset.genotypes)

    if temperature_informed:
        initial = np.concatenate(
            [np.full(n_genotypes, 0.25), np.full(n_genotypes, 0.45), [292.0, 303.0]]
        )
        lower = np.concatenate(
            [np.full(n_genotypes, 1e-4), np.full(n_genotypes, 0.05), [275.0, 295.0]]
        )
        upper = np.concatenate(
            [np.full(n_genotypes, 2.0), np.full(n_genotypes, 1.0), [300.0, 330.0]]
        )

        def residual(parameters: np.ndarray) -> np.ndarray:
            r = parameters[:n_genotypes]
            K = parameters[n_genotypes : 2 * n_genotypes]
            lower_k, upper_k = parameters[-2:]
            g = obs[:, 0].astype(int)
            response = paper_temperature_response(obs[:, 1], lower_k, upper_k)
            prediction = logistic_solution(obs[:, 2], r[g], K[g], response)
            ordering_penalty = np.repeat(max(lower_k - upper_k + 1.0, 0.0), 10)
            return np.concatenate([prediction - obs[:, 3], ordering_penalty])

        fitted = least_squares(residual, initial, bounds=(lower, upper), max_nfev=20000)
        parameters = fitted.x
        r = parameters[:n_genotypes]
        K = parameters[n_genotypes : 2 * n_genotypes]
        lower_k, upper_k = parameters[-2:]
        parameter_total = 2 * n_genotypes + 2
    else:
        initial = np.concatenate([np.full(n_genotypes, 0.15), np.full(n_genotypes, 0.45)])
        lower = np.concatenate([np.full(n_genotypes, 1e-4), np.full(n_genotypes, 0.05)])
        upper = np.concatenate([np.full(n_genotypes, 2.0), np.full(n_genotypes, 1.0)])

        def residual(parameters: np.ndarray) -> np.ndarray:
            r = parameters[:n_genotypes]
            K = parameters[n_genotypes:]
            g = obs[:, 0].astype(int)
            prediction = logistic_solution(obs[:, 2], r[g], K[g], np.ones(len(obs)))
            return prediction - obs[:, 3]

        fitted = least_squares(residual, initial, bounds=(lower, upper), max_nfev=10000)
        parameters = fitted.x
        r = parameters[:n_genotypes]
        K = parameters[n_genotypes:]
        lower_k = upper_k = np.nan
        parameter_total = 2 * n_genotypes

    predictions: dict[str, torch.Tensor] = {}
    for name in ["train_eval", "val", "test"]:
        split = getattr(dataset, name)
        g, split_temperatures, split_days = batch_metadata(split, dataset)
        if temperature_informed:
            multiplier = paper_temperature_response(split_temperatures, lower_k, upper_k)
        else:
            multiplier = np.ones(len(split_temperatures))
        values = np.stack(
            [
                logistic_solution(split_days, r[g[row]], K[g[row]], multiplier[row])
                for row in range(len(g))
            ]
        )
        predictions[name] = torch.tensor(values, dtype=torch.float32)
    fitted_parameters = {
        "r": np.asarray(r, dtype=np.float32),
        "K": np.asarray(K, dtype=np.float32),
    }
    if temperature_informed:
        fitted_parameters["lower_k"] = np.asarray([lower_k], dtype=np.float32)
        fitted_parameters["upper_k"] = np.asarray([upper_k], dtype=np.float32)
    return predictions, parameter_total, fitted_parameters


def fit_random_forest(dataset: Dataset, seed: int) -> tuple[dict[str, torch.Tensor], int]:
    n_genotypes = len(dataset.genotypes)

    def features(batch: SequenceBatch, observed_only: bool) -> tuple[np.ndarray, np.ndarray | None]:
        rows: list[np.ndarray] = []
        targets: list[float] = []
        for sequence in range(len(batch.genotype)):
            columns = np.where(batch.mask[sequence].numpy() > 0)[0] if observed_only else np.arange(len(DAYS))
            one_hot = np.zeros(n_genotypes, dtype=np.float64)
            one_hot[int(batch.genotype_index[sequence])] = 1.0
            for column in columns:
                rows.append(
                    np.concatenate(
                        [one_hot, [float(batch.environment[sequence, column]), float(batch.time[column])]]
                    )
                )
                if observed_only:
                    targets.append(float(batch.target[sequence, column]))
        return np.stack(rows), np.asarray(targets) if observed_only else None

    x_train, y_train = features(dataset.train, True)
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=seed,
        n_jobs=1,
        criterion="squared_error",
    )
    model.fit(x_train, y_train)
    predictions: dict[str, torch.Tensor] = {}
    for name in ["train_eval", "val", "test"]:
        split = getattr(dataset, name)
        x, _ = features(split, False)
        values = model.predict(x).reshape(len(split.genotype), len(DAYS))
        predictions[name] = torch.tensor(values, dtype=torch.float32)
    return predictions, -1


def load_current_latent_module():
    # Keep an exact local copy of the wheat latent ODE implementation so this
    # dataset directory remains independently reproducible.
    path = HERE / "latent_ode_model.py"
    spec = importlib.util.spec_from_file_location("current_latentode_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load current latent model from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def predict_reference(model: ReferenceLSTM, batch: SequenceBatch) -> torch.Tensor:
    model.eval()
    time_batch = batch.time.unsqueeze(0).expand(len(batch.genotype), -1)
    return model(batch.genotype_index, batch.environment, time_batch)["prediction"]


def train_reference_model(
    dataset: Dataset,
    seed: int,
    device: torch.device,
    epochs: int,
    min_epochs: int,
    eval_every: int,
    val_smooth: int,
    physics: bool,
    physics_weight: float,
    logistic_initialization: dict[str, np.ndarray] | None = None,
    model_lr: float = 1e-3,
    ode_lr: float | None = None,
    physics_warmup_epochs: int = 0,
    direct_ode_parameters: bool = False,
) -> tuple[dict, dict[str, torch.Tensor]]:
    seed_everything(seed)
    train = dataset.train.to(device)
    train_eval = dataset.train_eval.to(device)
    val = dataset.val.to(device)
    test = dataset.test.to(device)
    model = ReferenceLSTM(
        len(dataset.genotypes),
        physics=physics,
        direct_ode_parameters=direct_ode_parameters,
    ).to(device)
    for name, parameter in model.named_parameters():
        if "lstm" in name and "weight" in name and parameter.ndim >= 2:
            torch.nn.init.orthogonal_(parameter)
    # Reference repository: orthogonal weights, zero biases, Adam(lr=1e-3),
    # and its explicit normalized L2 term with weight 1.0.
    for name, parameter in model.named_parameters():
        if "weight" in name and parameter.ndim >= 2:
            torch.nn.init.orthogonal_(parameter)
        elif "bias" in name:
            torch.nn.init.zeros_(parameter)
    if physics:
        if logistic_initialization is None:
            raise ValueError("Logi-PINN requires fitted logistic initialization")
        initialize_logistic_head(
            model,
            torch.from_numpy(logistic_initialization["r"]),
            torch.from_numpy(logistic_initialization["K"]),
        )
    slow_parameters: list[torch.nn.Parameter] = []
    if physics:
        if model.direct_ode_parameters:
            slow_parameters = [model.r_raw, model.K_raw]
        else:
            slow_parameters = list(model.genotype_embedding.parameters()) + list(
                model.ode_head.parameters()
            )
    if physics and ode_lr is not None:
        # r and K depend on both the genotype embedding and ode_head.  Keep
        # both in the slow group so the data-fitted initialization is not
        # immediately erased by the prediction network's learning rate.
        slow_ids = {id(parameter) for parameter in slow_parameters}
        fast_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in slow_ids
        ]
        optimizer = torch.optim.Adam(
            [
                {"params": fast_parameters, "lr": model_lr},
                {"params": slow_parameters, "lr": ode_lr},
            ]
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=model_lr)
    best_value = float("inf")
    best_epoch = -1
    best_state = None
    recent: list[float] = []
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        time_batch = train.time.unsqueeze(0).expand(len(train.genotype), -1).clone()
        physics_active = physics and epoch > physics_warmup_epochs
        if physics_active:
            time_batch.requires_grad_(True)
        # The PINN loss differentiates the LSTM output with respect to time and
        # then backpropagates through that derivative. cuDNN RNN kernels do not
        # implement this double backward, so use PyTorch's differentiable RNN
        # path only for Logi-PINN training. Evaluation can still use cuDNN.
        with torch.backends.cudnn.flags(enabled=not physics_active):
            output = model(train.genotype_index, train.environment, time_batch)
        loss = curve_rmse(output["prediction"], train.target, train.mask)
        loss = loss + reference_l2_loss(model, weight=1.0)
        if physics_active:
            for item in logi_pinn_losses(
                output,
                time_batch,
                train.target,
                duration_days=float(DAYS[-1] - DAYS[0]),
                physics_weight=physics_weight,
            ).values():
                loss = loss + item
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if physics and not physics_active:
            # Preserve fitted r/K exactly during data-only warm-up.  The
            # remaining sequence and prediction layers learn a reasonable
            # trajectory before the physics objective is introduced.
            for parameter in slow_parameters:
                parameter.grad = None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % eval_every == 0 or epoch == epochs:
            val_prediction = predict_reference(model, val)
            val_rmse = float(curve_rmse(val_prediction, val.target, val.mask))
            recent.append(val_rmse)
            if len(recent) > val_smooth:
                recent.pop(0)
            smooth = float(np.mean(recent))
            if epoch >= min_epochs and len(recent) == val_smooth and smooth < best_value:
                best_value = smooth
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        best_epoch = epochs
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    predictions = {
        name: predict_reference(model, split).detach().cpu()
        for name, split in [("train_eval", train_eval), ("val", val), ("test", test)]
    }
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "n_params": parameter_count(model),
        "seconds": time.time() - started,
        "model_lr": model_lr,
        "ode_lr": ode_lr if ode_lr is not None else model_lr,
        "physics_warmup_epochs": physics_warmup_epochs,
        "direct_ode_parameters": direct_ode_parameters,
    }
    if physics and logistic_initialization is not None:
        model.eval()
        with torch.no_grad():
            genotype = torch.arange(len(dataset.genotypes), device=device)
            environment = torch.zeros((len(dataset.genotypes), len(DAYS)), device=device)
            time_batch = train.time.unsqueeze(0).expand(len(dataset.genotypes), -1)
            fitted = model(genotype, environment, time_batch)
            final_r = fitted["r"].detach().cpu().numpy()
            final_K = fitted["K"].detach().cpu().numpy()
        initial_r = logistic_initialization["r"]
        initial_K = logistic_initialization["K"]
        result["r_drift_rmse"] = float(np.sqrt(np.mean((final_r - initial_r) ** 2)))
        result["K_drift_rmse_m"] = float(np.sqrt(np.mean((final_K - initial_K) ** 2)))
    return result, predictions


@torch.no_grad()
def predict_latent(model, batch: SequenceBatch) -> torch.Tensor:
    model.eval()
    environment = batch.environment.unsqueeze(-1)
    zeros = torch.zeros_like(batch.environment)
    return model(batch.genotype_index, environment, zeros, zeros, 0)["pred"]


def train_latent_model(
    dataset: Dataset,
    seed: int,
    device: torch.device,
    epochs: int,
    min_epochs: int,
    eval_every: int,
    val_smooth: int,
) -> tuple[dict, dict[str, torch.Tensor]]:
    module = load_current_latent_module()
    seed_everything(seed)
    train = dataset.train.to(device)
    train_eval = dataset.train_eval.to(device)
    val = dataset.val.to(device)
    test = dataset.test.to(device)
    model = module.LatentODEHeightModel(
        n_genotypes=len(dataset.genotypes),
        env_dim=1,
        latent_dim=16,
        g_embed_dim=4,
        ode_hidden=32,
        ode_layers=2,
        dec_hidden=32,
        enc_hidden=16,
        init_mode="env_encoder",
        time_mode="calendar",
        genetic_encoding="one_hot",
        variational=False,
        n_substeps=1,
        days_per_tau=float(DAYS[-1] - DAYS[0]),
        use_physics=True,
    ).to(device)
    for name, parameter in model.named_parameters():
        if "lstm" in name and "weight" in name and parameter.ndim >= 2:
            torch.nn.init.orthogonal_(parameter)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_value = float("inf")
    best_epoch = -1
    best_state = None
    recent: list[float] = []
    started = time.time()
    environment = train.environment.unsqueeze(-1)
    zeros = torch.zeros_like(train.environment)

    for epoch in range(1, epochs + 1):
        model.train()
        output = model(train.genotype_index, environment, zeros, zeros, 0)
        loss = module.masked_rmse_loss(output["pred"], train.target, train.mask)
        weights = {"physic": 2.0, "r": 0.0, "ymax": 0.1, "mono": 0.0}
        for item in module.physics_losses(model, output, environment, weights).values():
            loss = loss + item
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % eval_every == 0 or epoch == epochs:
            val_prediction = predict_latent(model, val)
            val_rmse = float(curve_rmse(val_prediction, val.target, val.mask))
            recent.append(val_rmse)
            if len(recent) > val_smooth:
                recent.pop(0)
            smooth = float(np.mean(recent))
            if epoch >= min_epochs and len(recent) == val_smooth and smooth < best_value:
                best_value = smooth
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        best_epoch = epochs
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    predictions = {
        name: predict_latent(model, split).detach().cpu()
        for name, split in [("train_eval", train_eval), ("val", val), ("test", test)]
    }
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "n_params": module.count_parameters(model),
        "seconds": time.time() - started,
    }
    return result, predictions


def add_metrics(result: dict, predictions: dict[str, torch.Tensor], dataset: Dataset) -> dict:
    for name in ["train_eval", "val", "test"]:
        batch = getattr(dataset, name)
        rmse, mae = evaluate_prediction(predictions[name], batch)
        label = "train" if name == "train_eval" else name
        result[f"{label}_rmse_m"] = rmse
        result[f"{label}_mae_m"] = mae
    return result


def save_prediction_rows(
    model: str,
    seed: int,
    predictions: dict[str, torch.Tensor],
    dataset: Dataset,
) -> list[dict]:
    rows: list[dict] = []
    for split_name in ["train_eval", "val", "test"]:
        batch = getattr(dataset, split_name)
        values = predictions[split_name].numpy()
        for sequence, (genotype, condition) in enumerate(zip(batch.genotype, batch.condition)):
            for column, day in enumerate(DAYS):
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "split": "train" if split_name == "train_eval" else split_name,
                        "genotype": genotype,
                        "condition": condition,
                        "day_after_sowing": int(day),
                        "observed": int(batch.mask[sequence, column].item()),
                        "observed_length_m": float(batch.target[sequence, column]),
                        "predicted_length_m": float(values[sequence, column]),
                    }
                )
    return rows


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        group = runs[runs["model"] == model]
        if group.empty:
            continue
        row: dict[str, object] = {
            "model": model,
            "n_runs": len(group),
            "n_params": int(group["n_params"].iloc[0]),
            "best_epoch_mean": float(group["best_epoch"].mean()),
            "seconds_mean": float(group["seconds"].mean()),
        }
        for split in ["train", "val", "test"]:
            for metric in ["rmse", "mae"]:
                values = group[f"{split}_{metric}_m"] * 100.0
                row[f"{split}_{metric}_cm_mean"] = float(values.mean())
                row[f"{split}_{metric}_cm_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("test_rmse_cm_mean").reset_index(drop=True)


def markdown_table(summary: pd.DataFrame) -> str:
    lines = [
        "| Model | Train RMSE (cm) | Val RMSE (cm) | Test RMSE (cm) | Test MAE (cm) | Runs | Params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.train_rmse_cm_mean:.3f} ± {row.train_rmse_cm_sd:.3f} "
            f"| {row.val_rmse_cm_mean:.3f} ± {row.val_rmse_cm_sd:.3f} "
            f"| {row.test_rmse_cm_mean:.3f} ± {row.test_rmse_cm_sd:.3f} "
            f"| {row.test_mae_cm_mean:.3f} ± {row.test_mae_cm_sd:.3f} "
            f"| {row.n_runs} | {row.n_params if row.n_params >= 0 else 'n/a'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=1500, help="Latent Neural ODE epochs")
    parser.add_argument(
        "--reference-epochs",
        type=int,
        default=3000,
        help="LSTM/Logi-PINN epochs; 3000 matches the authors' GitHub code",
    )
    parser.add_argument("--physics-warmup-epochs", type=int, default=500)
    parser.add_argument(
        "--direct-ode-parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use decoupled genotype-specific r/K instead of a shared genotype embedding head",
    )
    parser.add_argument("--min-epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--val-smooth", type=int, default=10)
    parser.add_argument("--reference-lr", type=float, default=1e-3)
    parser.add_argument(
        "--ode-lr",
        type=float,
        default=1e-4,
        help="Learning rate for Logi-PINN genotype embedding and ODE head",
    )
    args = parser.parse_args()
    if args.min_epochs > args.epochs:
        raise ValueError("--min-epochs cannot exceed --epochs")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data)
    device = torch.device(args.device)
    print(
        f"data: 9 genotypes x 2 temperatures x 10 plants x 4 observations\n"
        f"split: train={len(dataset.train.genotype)} individuals, "
        f"val/test=18 averaged curves each | device={device}"
    )

    runs: list[dict] = []
    prediction_rows: list[dict] = []

    logistic_initialization: dict[str, np.ndarray] | None = None
    fitted_process_parameters: dict[str, dict[str, list[float]]] = {}
    for model_name, temperature_informed in [("Logi-ODE", False), ("Temp-ODE", True)]:
        started = time.time()
        predictions, n_params, fitted_parameters = fit_process_baseline(
            dataset, temperature_informed
        )
        fitted_process_parameters[model_name] = {
            key: value.astype(float).tolist() for key, value in fitted_parameters.items()
        }
        if model_name == "Logi-ODE":
            logistic_initialization = fitted_parameters
        result = add_metrics(
            {
                "model": model_name,
                "seed": 0,
                "best_epoch": 0,
                "n_params": n_params,
                "seconds": time.time() - started,
            },
            predictions,
            dataset,
        )
        runs.append(result)
        prediction_rows.extend(save_prediction_rows(model_name, 0, predictions, dataset))
        print(f"{model_name:26s} test RMSE={result['test_rmse_m'] * 100:.3f} cm")

    if logistic_initialization is None:
        raise RuntimeError("Logi-ODE fit did not produce PINN initialization")
    print("Logi-PINN init r:", np.round(logistic_initialization["r"], 5).tolist())
    print("Logi-PINN init K:", np.round(logistic_initialization["K"], 5).tolist())

    for seed in args.seeds:
        started = time.time()
        predictions, n_params = fit_random_forest(dataset, seed)
        result = add_metrics(
            {
                "model": "RF",
                "seed": seed,
                "best_epoch": 0,
                "n_params": n_params,
                "seconds": time.time() - started,
            },
            predictions,
            dataset,
        )
        runs.append(result)
        prediction_rows.extend(save_prediction_rows("RF", seed, predictions, dataset))
        print(f"RF seed {seed:2d}                 test RMSE={result['test_rmse_m'] * 100:.3f} cm")

    for model_name, physics in [("LSTM-NN", False), ("Logi-PINN", True)]:
        for seed in args.seeds:
            result, predictions = train_reference_model(
                dataset,
                seed,
                device,
                args.reference_epochs,
                max(args.min_epochs, 1500) if args.reference_epochs >= 1500 else args.min_epochs,
                args.eval_every,
                args.val_smooth,
                physics,
                physics_weight=2.0,
                logistic_initialization=logistic_initialization if physics else None,
                model_lr=args.reference_lr,
                ode_lr=args.ode_lr if physics else None,
                physics_warmup_epochs=args.physics_warmup_epochs if physics else 0,
                direct_ode_parameters=args.direct_ode_parameters if physics else False,
            )
            result["model"] = model_name
            result = add_metrics(result, predictions, dataset)
            runs.append(result)
            prediction_rows.extend(save_prediction_rows(model_name, seed, predictions, dataset))
            print(
                f"{model_name:10s} seed {seed:2d} ep={result['best_epoch']:4d} "
                f"train={result['train_rmse_m'] * 100:.3f} "
                f"val={result['val_rmse_m'] * 100:.3f} "
                f"test={result['test_rmse_m'] * 100:.3f} cm"
            )

    for seed in args.seeds:
        result, predictions = train_latent_model(
            dataset,
            seed,
            device,
            args.epochs,
            args.min_epochs,
            args.eval_every,
            args.val_smooth,
        )
        result["model"] = "Latent Neural ODE (ours)"
        result = add_metrics(result, predictions, dataset)
        runs.append(result)
        prediction_rows.extend(
            save_prediction_rows("Latent Neural ODE (ours)", seed, predictions, dataset)
        )
        print(
            f"Latent ODE seed {seed:2d} ep={result['best_epoch']:4d} "
            f"train={result['train_rmse_m'] * 100:.3f} "
            f"val={result['val_rmse_m'] * 100:.3f} "
            f"test={result['test_rmse_m'] * 100:.3f} cm"
        )

    runs_frame = pd.DataFrame(runs)
    predictions_frame = pd.DataFrame(prediction_rows)
    summary = summarise(runs_frame)
    runs_frame.to_csv(args.output / "all_runs.csv", index=False)
    predictions_frame.to_csv(args.output / "predictions.csv", index=False)
    summary.to_csv(args.output / "comparison.csv", index=False)
    (args.output / "comparison.md").write_text(markdown_table(summary), encoding="utf-8")
    (args.output / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (args.output / "fitted_process_parameters.json").write_text(
        json.dumps(fitted_process_parameters, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n" + markdown_table(summary))
    print(f"saved results -> {args.output}")


if __name__ == "__main__":
    main()
