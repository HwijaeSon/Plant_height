"""Configurable capacity for the follow-up search; the original benchmark is frozen."""
from models import CORE, LightLatentODE, LightParameterHead


DEFAULT_ARCHITECTURE = dict(latent_dim=8, g_embed_dim=4, ode_hidden=16,
                            ode_layers=1, dec_hidden=16, enc_hidden=8)


class TunableLightLatentODE(LightLatentODE):
    def __init__(self, architecture=None):
        settings = DEFAULT_ARCHITECTURE | (architecture or {})
        CORE.LatentODEHeightModel.__init__(self, n_genotypes=5, env_dim=2,
            use_physics=False, days_per_tau=72., **settings)
        self.ode_param_head = LightParameterHead(settings['g_embed_dim'])


def build(config):
    assert config['model'] in ('phytoode', 'latent_ode')
    if config['model'] == 'latent_ode':
        assert config['lambda_ode'] == config['lambda_k'] == 0
    return TunableLightLatentODE(config.get('architecture'))
