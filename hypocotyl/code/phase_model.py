"""PhytoODE2 and matched controls, with a fixed physical normalization scale."""
from pathlib import Path
import importlib.util
import json
import random
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
PRIOR=HERE
PROPOSAL=HERE


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    return loaded


prior=module('prior_configurable_phytoode',PRIOR/'architecture.py')
law=module('light_dark_logistic_law',PROPOSAL/'growth_law.py')
data=prior.data
from prefix_parameter_model import PrefixParameterPhytoODE

VARIANTS=['phytoode','phytoode_light','phytoode2']
LABELS={'phytoode':'PhytoODE (matched control)',
        'phytoode_light':'PhytoODE + light input', 'phytoode2':'PhytoODE2'}
FIXED_SCALE=float(json.loads((data.PREPARED/'config.json').read_text())['height_scale_mm'])
DEFAULTS=dict(prior.DEFAULTS)


def load(device='cpu',include_test=False,drop_fraction=0.,mask_seed=20260911):
    """Reuse the approved masks; construct all targets directly at one scale."""
    cfg,plants,x,batches,raw=data.load(device=device,include_test=include_test,
        drop_fraction=drop_fraction,mask_seed=mask_seed)
    n=len(plants)
    for split,frame in raw.items():
        target=np.zeros((n,len(data.HOURS)),dtype=np.float32)
        cols=np.searchsorted(data.HOURS,frame.elapsed_hours.to_numpy())
        target[frame.plant_index,cols]=frame.length_mm.to_numpy()/FIXED_SCALE
        batches[split]['target']=torch.tensor(target,device=device)
    frame=raw['train'];prefix=np.zeros((n,4),dtype=np.float32)
    cols=np.searchsorted(data.PREFIX_HOURS,frame.elapsed_hours.to_numpy())
    prefix[frame.plant_index,cols]=frame.length_mm.to_numpy()/FIXED_SCALE
    x['prefix_y']=torch.tensor(prefix,device=device)
    cfg.update(height_scale_mm=FIXED_SCALE,
        scale_fit_on='Original 0--36 h training observations, before controlled removal; common to all masks.',
        normalization='Fixed original-training maximum, not recomputed after removal.')
    return cfg,plants,x,batches,raw


class PhytoODE2(prior.PhytoODE):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.variant=cfg['variant']
        if self.variant not in VARIANTS:raise ValueError(self.variant)
        self.light=self.variant!='phytoode'
        if self.light:
            original=self.ode_func.net[0]
            z=cfg['latent_dim']
            # CORE concatenates [latent, environment, genotype, time]. Insert
            # one initially zero light column without changing common weights
            # or the random state used for subsequent LSTM initialization.
            with torch.random.fork_rng(devices=[]):
                expanded=nn.Linear(original.in_features+1,original.out_features)
                with torch.no_grad():
                    expanded.weight.zero_()
                    expanded.weight[:,:z].copy_(original.weight[:,:z])
                    expanded.weight[:,z+1:].copy_(original.weight[:,z:])
                    expanded.bias.copy_(original.bias)
            self.ode_func.net[0]=expanded
        self.growth_law=law.LightDarkLogistic(len(data.GENOTYPES))
        self.growth_law.beta.requires_grad_(self.variant=='phytoode2')
        self.register_buffer('hours',torch.tensor(data.HOURS,dtype=torch.float32))
        # Use identical 18 non-switch interior nodes in every matched variant.
        self.register_buffer('collocation',torch.tensor((data.HOURS%12)!=0))

    def forward(self,g_idx,time,env,prefix_y,prefix_mask,prefix_time):
        forcing=env if self.light else env.new_empty((len(g_idx),len(time),0))
        # Its RK4 implementation holds forcing constant over each 3-h step,
        # including the endpoint stage; switch points align with step borders.
        out=PrefixParameterPhytoODE.forward(self,g_idx,time,forcing,prefix_y,prefix_mask,prefix_time)
        out['a']=self.growth_law.contrast(g_idx)
        out['r_light'],out['r_dark']=self.growth_law.phase_rates(out['r'],g_idx)
        return out

    def physics_loss(self,out,x):
        derivative=self.derivative(out,x)
        rhs=self.growth_law.rhs(out['pred'],out['r'],out['K'].clamp_min(1e-4),x['g_idx'],self.hours)
        return (derivative[:,self.collocation]-rhs[:,self.collocation]).square().mean()


def initialize(cfg,seed,device='cpu'):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=PhytoODE2(cfg).to(device)
    for name,value in model.named_parameters():
        if 'lstm' in name and 'weight' in name and value.ndim>=2:nn.init.orthogonal_(value)
    return model
