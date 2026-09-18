"""PhytoODE with individual logistic parameters conditioned on the observed prefix."""
import torch
from torch import nn
from forecast_models import PrefixLatentODE


class PrefixParameterPhytoODE(PrefixLatentODE):
    """Retain the latent dynamics and add masked prefix inputs to the r/K head.

    The four genotype columns of the original parameter head are preserved;
    eight additional columns receive four masked lengths and four masks.
    These new columns start at zero, so the initial r/K values retain the
    genotype-only reference. RNG restoration preserves core initialization.
    """
    def __init__(self):
        super().__init__(light=True)
        original = self.ode_param_head.net[0]
        with torch.random.fork_rng(devices=[]):
            expanded = nn.Linear(12, original.out_features)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :4].copy_(original.weight)
                expanded.bias.copy_(original.bias)
        self.ode_param_head.net[0] = expanded

    def forward(self, g_idx, time, env, prefix_y, prefix_mask, prefix_time):
        n = len(g_idx)
        embedding = self.g_enc(g_idx)
        z = self.encoder.encode(embedding, prefix_y, prefix_mask, prefix_time)
        trajectory = [z]
        # Same 3-hour RK4 integration and illumination switch convention as
        # the frozen prefix benchmark. No phenotype after 36 h is an input.
        for i in range(len(time) - 1):
            step = 1. / (len(time) - 1)
            t = i * step
            forcing = env[:, i]

            def field(at, state):
                return self.ode_func(state.new_full((n, 1), at), state, forcing, embedding)

            k1 = field(t, z)
            k2 = field(t + step / 2, z + step * k1 / 2)
            k3 = field(t + step / 2, z + step * k2 / 2)
            k4 = field(t + step, z + step * k3)
            z = z + step * (k1 + 2*k2 + 2*k3 + k4) / 6
            trajectory.append(z)
        latent = torch.stack(trajectory, dim=1)
        context = torch.cat([embedding, prefix_y * prefix_mask, prefix_mask], dim=-1)
        rate, capacity = self.ode_param_head(context)
        return dict(pred=self.decode(latent), z=latent, g_emb=embedding,
                    t_feat=time[None, :].expand(n, -1), r=rate, K=capacity)
