"""Individual-parameter PhytoODE with no illumination in encoder or vector field."""
import torch
from torch import nn
from forecast_models import PrefixLatentODE
from prefix_parameter_model import PrefixParameterPhytoODE


class NoLightPrefixParameterPhytoODE(PrefixParameterPhytoODE):
    def __init__(self):
        # This constructs the same no-light backbone as the existing latent ODE.
        PrefixLatentODE.__init__(self, light=False)
        original = self.ode_param_head.net[0]
        with torch.random.fork_rng(devices=[]):
            expanded = nn.Linear(12, original.out_features)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :4].copy_(original.weight)
                expanded.bias.copy_(original.bias)
        self.ode_param_head.net[0] = expanded

    def forward(self, g_idx, time, env, prefix_y, prefix_mask, prefix_time):
        # Only tensor dtype/device are used; illumination values are ignored.
        no_environment = env.new_empty((len(g_idx), len(time), 0))
        return super().forward(g_idx, time, no_environment, prefix_y, prefix_mask, prefix_time)
