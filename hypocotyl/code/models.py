"""Light-conditioned latent ODE and transparent neural comparison models."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

source = Path(__file__).resolve().parents[2] / "wheat/code/model.py"
spec = importlib.util.spec_from_file_location("hypocotyl_latent_core", source)
CORE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CORE)


class LightParameterHead(nn.Module):
    """Genotype-specific positive light/dark hourly rates and scaled capacity."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = CORE.mlp([embedding_dim, 8, 3])
        with torch.no_grad():
            self.net[-1].weight.mul_(.01)
            self.net[-1].bias.copy_(torch.logit(torch.tensor([.1, .1, .4])))

    def forward(self, embedding):
        raw = torch.sigmoid(self.net(embedding))
        return .5 * raw[..., 0], .5 * raw[..., 1], 2. * raw[..., 2]


class LightLatentODE(CORE.LatentODEHeightModel):
    """Same latent architecture, with piecewise-constant light during each RK4 step."""

    def __init__(self):
        super().__init__(n_genotypes=5, env_dim=2, latent_dim=8, g_embed_dim=4,
            ode_hidden=16, ode_layers=1, dec_hidden=16, enc_hidden=8,
            use_physics=False, days_per_tau=72.)
        # The inherited chain-rule routine divides by duration; here its unit is hours.
        self.ode_param_head = LightParameterHead(4)

    def forward(self, g_idx, env, time, encoder_env=None, encoder_time=None):
        B, T, _ = env.shape
        embedding = self.g_enc(g_idx)
        times = time.unsqueeze(0).expand(B, -1)
        encoding = env if encoder_env is None else encoder_env
        encoding_times = times if encoder_time is None else encoder_time.unsqueeze(0).expand(B, -1)
        z, _, _ = self.get_z0(embedding, encoding, encoding_times)
        trajectory = [z]
        step = 1. / (T-1)
        for i in range(T-1):
            # All grids align with 12-h transitions. Use the old interval's light
            # at its right endpoint; the next interval starts with the new light.
            forcing = env[:, i]
            def field(t, state):
                tf = torch.full((B, 1), t, dtype=state.dtype, device=state.device)
                return self.ode_func(tf, state, forcing, embedding)
            t = i * step
            k1 = field(t, z)
            k2 = field(t + step/2, z + step/2*k1)
            k3 = field(t + step/2, z + step/2*k2)
            k4 = field(t + step, z + step*k3)
            z = z + step/6*(k1+2*k2+2*k3+k4)
            trajectory.append(z)
        latent = torch.stack(trajectory, dim=1)
        r_light, r_dark, K = self.ode_param_head(embedding)
        return dict(pred=self.decode(latent), z=latent, g_emb=embedding, t_feat=times,
                    r_light=r_light, r_dark=r_dark, K=K)

    def derivative(self, output, inputs):
        return self.dydt(output["z"], inputs["env"], output["g_emb"], output["t_feat"])[0]


class LSTMNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 4)
        self.lstm1 = nn.LSTM(3, 16, batch_first=True)
        self.lstm2 = nn.LSTM(16, 8, batch_first=True)
        self.head = nn.Sequential(nn.Linear(12, 16), nn.LeakyReLU(.01),
                                  nn.Linear(16, 1), nn.LeakyReLU(.01))

    def forward(self, g_idx, env, time):
        t = time.unsqueeze(0).expand(env.shape[0], -1)
        hidden, _ = self.lstm1(torch.cat([env, t.unsqueeze(-1)], dim=-1))
        hidden, _ = self.lstm2(hidden)
        embedding = self.embedding(g_idx).unsqueeze(1).expand(-1, env.shape[1], -1)
        return dict(pred=self.head(torch.cat([hidden, embedding], dim=-1)).squeeze(-1))


class LightPINN(nn.Module):
    """Coordinate MLP PINN; a local time derivative is exact for this architecture.

    The coordinate network takes genotype, time and the known light duty cycle.
    Instantaneous binary light enters the residual, preserving continuous height
    at switching times. This is not a reproduction of the older LSTM Logi-PINN.
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 4)
        self.head = CORE.mlp([6, 32, 32, 1])
        self.output = nn.LeakyReLU(.01)
        self.ode_param_head = LightParameterHead(4)

    def forward(self, g_idx, env, time):
        t = time.unsqueeze(0).expand(env.shape[0], -1).detach().clone()
        t.requires_grad_(torch.is_grad_enabled())
        embedding = self.embedding(g_idx)
        repeated = embedding.unsqueeze(1).expand(-1, env.shape[1], -1)
        features = torch.cat([t.unsqueeze(-1), env[..., 1:2], repeated], dim=-1)
        pred = self.output(self.head(features)).squeeze(-1)
        r_light, r_dark, K = self.ode_param_head(embedding)
        return dict(pred=pred, time=t, r_light=r_light, r_dark=r_dark, K=K)

    def derivative(self, output, inputs):
        return torch.autograd.grad(output["pred"].sum(), output["time"], create_graph=True)[0] / 72.


def physics_losses(model, output, inputs):
    derivative = model.derivative(output, inputs)
    light = inputs["env"][..., 0]
    rate = output["r_light"].unsqueeze(1) * light + output["r_dark"].unsqueeze(1) * (1-light)
    rhs = rate * output["pred"] * (1-output["pred"] / output["K"].unsqueeze(1).clamp_min(1e-4))
    # The derivative is undefined at a switch. All non-switch interior nodes
    # are collocation points, independently of the phenotype observation mask.
    hours = inputs["time"] * 72.
    collocation = ~torch.isclose(torch.remainder(hours, 12.), torch.zeros_like(hours), atol=1e-4)
    ode = (derivative[:, collocation] - rhs[:, collocation]).square().mean()
    capacity = (output["pred"].max(dim=1).values - output["K"]).abs().mean()
    return ode, capacity


def build(model):
    if model in ("phytoode", "latent_ode"):
        return LightLatentODE()
    if model == "lstm":
        return LSTMNN()
    if model == "light_pinn":
        return LightPINN()
    raise ValueError(model)
