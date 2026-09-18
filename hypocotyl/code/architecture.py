"""Configurable PhytoODE preserving the released model at default dimensions."""
import sys
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn

RELEASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(RELEASE/'hypocotyl/code'))
import forecast_data as data
from single_condition_models import CORE
from no_light_prefix_model import NoLightPrefixParameterPhytoODE

DEFAULTS=dict(latent_dim=8,g_embed_dim=4,ode_hidden=16,ode_layers=1,dec_hidden=16,
              enc_hidden=8,param_hidden=8,ode_scale=3.,lambda_ode=100.,lambda_k=0.,
              lr=.003,weight_decay=0.,optimizer='adam',scheduler='cosine',epochs=1500,
              min_epoch=200,eval_every=20,val_smooth=3,gradient_clip=1.,physics_warmup=0)


class Encoder(CORE.EnvEncoder):
    def __init__(self,cfg):
        super().__init__(env_dim=2,g_dim=cfg['g_embed_dim'],hidden_size=cfg['enc_hidden'],latent_dim=cfg['latent_dim'])
        self.light=False

    def encode(self,embedding,prefix_y,prefix_mask,prefix_time):
        inputs=torch.stack([prefix_y*prefix_mask,prefix_mask],dim=-1)
        return self.forward(inputs,prefix_time[None,:].expand_as(prefix_y),embedding)[0]


class ParameterHead(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        g=cfg['g_embed_dim']
        self.net=CORE.mlp([g,cfg['param_hidden'],2])
        with torch.no_grad():
            self.net[-1].weight.mul_(.01)
            self.net[-1].bias.copy_(torch.logit(torch.tensor([.1,.4])))
        original=self.net[0]
        with torch.random.fork_rng(devices=[]):
            expanded=nn.Linear(g+8,original.out_features)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:,:g].copy_(original.weight)
                expanded.bias.copy_(original.bias)
        self.net[0]=expanded

    def forward(self,context):
        values=torch.sigmoid(self.net(context))
        return .5*values[...,0],2.*values[...,1]


class PhytoODE(NoLightPrefixParameterPhytoODE):
    def __init__(self,cfg):
        CORE.LatentODEHeightModel.__init__(self,n_genotypes=len(data.GENOTYPES),env_dim=0,
            latent_dim=cfg['latent_dim'],g_embed_dim=cfg['g_embed_dim'],ode_hidden=cfg['ode_hidden'],
            ode_layers=cfg['ode_layers'],dec_hidden=cfg['dec_hidden'],enc_hidden=cfg['enc_hidden'],
            use_physics=False,days_per_tau=72.)
        self.light=False
        self.encoder=Encoder(cfg)
        self.ode_param_head=ParameterHead(cfg)
        self.ode_func.out_scale=cfg['ode_scale']


def initialize(cfg,seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=PhytoODE(cfg).to(device)
    for name,value in model.named_parameters():
        if 'lstm' in name and 'weight' in name and value.ndim>=2:
            torch.nn.init.orthogonal_(value)
    return model
