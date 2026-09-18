from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class ReferenceLSTM(nn.Module):
    """Dataset-adapted copy of the GitHub multiple-genotype architecture.

    The source classes are `genotype_code_temperature_input_height_prediction_fc`
    and `pinn_genotype_embedding_fc` in the reference authors' companion code
    `MultipleGenotypeModel.py` module. PyTorch uses batch-first tensors here, but the
    layer graph and selected paper hyperparameters are the same.
    """

    def __init__(
        self,
        n_genotypes: int,
        hidden_size: int = 5,
        genotype_size: int = 5,
        head_size: int = 5,
        physics: bool = False,
        direct_ode_parameters: bool = False,
    ) -> None:
        super().__init__()
        self.physics = physics
        self.direct_ode_parameters = direct_ode_parameters
        self.n_genotypes = n_genotypes
        # nn.Linear(one_hot, embedding) is the exact encoding used upstream.
        self.genotype_embedding = nn.Linear(n_genotypes, genotype_size)
        self.lstm1 = nn.LSTM(2, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, 3, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(3 + genotype_size, head_size),
            nn.LeakyReLU(0.01),
            nn.Linear(head_size, 1),
        )
        self.output = nn.LeakyReLU(0.01)
        if physics:
            if direct_ode_parameters:
                self.r_raw = nn.Parameter(torch.zeros(n_genotypes))
                self.K_raw = nn.Parameter(torch.zeros(n_genotypes))
            else:
                self.ode_head = nn.Linear(genotype_size, 2)

    def forward(
        self,
        genotype_index: torch.Tensor,
        environment: torch.Tensor,
        time: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if time.ndim == 1:
            time = time.unsqueeze(0).expand(environment.shape[0], -1)
        features = torch.stack([environment, time], dim=-1)
        hidden, _ = self.lstm1(features)
        hidden, _ = self.lstm2(hidden)
        hidden = nn.functional.leaky_relu(hidden, negative_slope=0.01)
        one_hot = nn.functional.one_hot(
            genotype_index, num_classes=self.n_genotypes
        ).to(environment.dtype)
        genotype_raw = self.genotype_embedding(one_hot)
        genotype = torch.tanh(genotype_raw)
        repeated = genotype.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        prediction = self.output(self.head(torch.cat([hidden, repeated], dim=-1))).squeeze(-1)
        result = {"prediction": prediction, "genotype": genotype}
        if self.physics:
            if self.direct_ode_parameters:
                result["r"] = torch.sigmoid(self.r_raw[genotype_index])
                result["K"] = torch.tanh(self.K_raw[genotype_index]) + 1.0
            else:
                raw = self.ode_head(genotype_raw)
                result["r"] = torch.sigmoid(raw[:, 0])
                result["K"] = torch.tanh(raw[:, 1]) + 1.0
        return result


@torch.no_grad()
def initialize_logistic_head(
    model: ReferenceLSTM,
    fitted_r: torch.Tensor,
    fitted_K: torch.Tensor,
) -> None:
    """Initialize the PINN's genotype-dependent ODE head from fitted curves.

    The reference architecture maps a learned genotype vector through sigmoid
    and tanh transforms.  Store the inverse-transformed fitted parameters in
    the first two genotype coordinates and make the ODE head read those
    coordinates directly.  The remaining coordinates retain their random
    initialization for the height-prediction head.
    """
    if not model.physics:
        raise ValueError("Logistic initialization requires physics=True")
    if fitted_r.shape != (model.n_genotypes,) or fitted_K.shape != (model.n_genotypes,):
        raise ValueError("Expected one fitted r and K value per genotype")

    r = fitted_r.to(model.genotype_embedding.weight).clamp(1e-5, 1.0 - 1e-5)
    K = fitted_K.to(model.genotype_embedding.weight).clamp(1e-5, 2.0 - 1e-5)
    raw_r = torch.logit(r)
    raw_K = torch.atanh((K - 1.0).clamp(-1.0 + 1e-5, 1.0 - 1e-5))

    if model.direct_ode_parameters:
        model.r_raw.copy_(raw_r)
        model.K_raw.copy_(raw_K)
    else:
        model.genotype_embedding.weight[0].copy_(raw_r)
        model.genotype_embedding.weight[1].copy_(raw_K)
        model.genotype_embedding.bias[0:2].zero_()
        model.ode_head.weight.zero_()
        model.ode_head.bias.zero_()
        model.ode_head.weight[0, 0] = 1.0
        model.ode_head.weight[1, 1] = 1.0


def logi_pinn_losses(
    output: dict[str, torch.Tensor],
    time: torch.Tensor,
    target: torch.Tensor,
    duration_days: float,
    physics_weight: float,
) -> dict[str, torch.Tensor]:
    prediction = output["prediction"]
    derivative_tau = torch.autograd.grad(
        prediction.sum(), time, create_graph=True, retain_graph=True
    )[0]
    derivative_day = derivative_tau / duration_days
    r = output["r"].unsqueeze(1)
    K = output["K"].unsqueeze(1)
    rhs = r * prediction * (1.0 - prediction / K.clamp_min(1e-4))
    return {
        "physics": physics_weight * (derivative_day - rhs).square().mean(),
        "r_penalty": (0.1 * (1e-5 ** output["r"])).mean(),
        "ymax": (output["K"] - target.max(dim=1).values).square().mean().sqrt(),
    }


def reference_l2_loss(model: nn.Module, weight: float = 1.0) -> torch.Tensor:
    """Exact normalized non-bias L2 term used by the authors' GitHub code."""
    weights = [value for name, value in model.named_parameters() if "bias" not in name]
    count = sum(value.numel() for value in weights)
    return weight * sum(value.square().sum() for value in weights) / max(count, 1)


@dataclass
class SequenceBatch:
    genotype_index: torch.Tensor
    condition: list[str]
    genotype: list[str]
    environment: torch.Tensor
    time: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device) -> "SequenceBatch":
        return SequenceBatch(
            genotype_index=self.genotype_index.to(device),
            condition=self.condition,
            genotype=self.genotype,
            environment=self.environment.to(device),
            time=self.time.to(device),
            target=self.target.to(device),
            mask=self.mask.to(device),
        )


def curve_rmse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared = (prediction - target).square() * mask
    observed = mask.sum(dim=1).clamp_min(1.0)
    return torch.sqrt(squared.sum(dim=1) / observed).mean()


def curve_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    absolute = (prediction - target).abs() * mask
    observed = mask.sum(dim=1).clamp_min(1.0)
    return (absolute.sum(dim=1) / observed).mean()
