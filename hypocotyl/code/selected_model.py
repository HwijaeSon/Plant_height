"""Auditable factorial PhytoODE2 variants; no genotype ordering prior."""
from pathlib import Path
import importlib.util,random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('original_phytoode2',HERE/'phase_model.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
data,load,FIXED_SCALE=original.data,original.load,original.FIXED_SCALE
DEFAULTS=dict(original.DEFAULTS,light_input=True,rate_level='individual',capacity_level='individual',
 rate_form='mean_contrast',contrast_enabled=True,initial_rate=.05,initial_capacity_scaled=.8,
 capacity_upper_scaled=2.,beta_lr_factor=1.,physical_lr_factor=1.,input_dropout=0.,
 contrast_penalty=0.,input_transform='identity',encoder_type='lstm',output_mode='neural',
 physics_type='absolute',physics_horizon='all',fit_type='rmse',interval_weight=0.,
 anchor_weight=0.,parameter_context='prefix',initial_contrast=0.,lambda_k=0.)

class SelectedHead(nn.Module):
 def __init__(self,old,cfg,rows):
  super().__init__();self.net=old.net;self.rows=rows;self.cfg=cfg
  with torch.no_grad():self.net[-1].bias.copy_(torch.logit(torch.tensor([cfg['initial_rate']/.5,cfg['initial_capacity_scaled']/cfg['capacity_upper_scaled']])))
  if rows!=[0,1]:
   layer=self.net[-1]
   with torch.random.fork_rng(devices=[]):
    replacement=nn.Linear(layer.in_features,len(rows))
    with torch.no_grad():replacement.weight.copy_(layer.weight[rows]);replacement.bias.copy_(layer.bias[rows])
   self.net[-1]=replacement
 def forward(self,context):
  v=torch.sigmoid(self.net(context));return {r:v[:,j]*(.5 if r==0 else self.cfg['capacity_upper_scaled']) for j,r in enumerate(self.rows)}

class FactorialPhytoODE2(original.PhytoODE2):
 def __init__(self,cfg):
  super().__init__(dict(cfg,variant='phytoode2' if cfg['light_input'] else 'phytoode'))
  self.cfg=cfg;self.light=cfg['light_input'];ng=len(data.GENOTYPES)
  self.growth_law.beta.requires_grad_(cfg['contrast_enabled'] and cfg['rate_form']=='mean_contrast')
  with torch.no_grad():self.growth_law.beta.fill_(np.arctanh(cfg['initial_contrast']) if cfg['contrast_enabled'] else 0.)
  rows=([0] if cfg['rate_level']=='individual' else [])+([1] if cfg['capacity_level']=='individual' else [])
  if rows:self.parameter_head=SelectedHead(self.ode_param_head,cfg,rows)
  del self.ode_param_head
  raw_rate=torch.log(torch.expm1(torch.tensor(cfg['initial_rate'])))
  if cfg['rate_level']!='individual':
   count=ng if cfg['rate_level']=='genotype' else 1
   self.raw_rate=nn.Parameter(raw_rate.expand(count,2 if cfg['rate_form']=='direct' and cfg['contrast_enabled'] else 1).clone())
  elif cfg['rate_form']=='direct':raise ValueError('Direct phase-rate coordinates require genotype/shared mean rates.')
  if cfg['capacity_level'] in ['genotype','shared']:
   count=ng if cfg['capacity_level']=='genotype' else 1
   self.raw_capacity=nn.Parameter(torch.logit(torch.tensor(cfg['initial_capacity_scaled']/cfg['capacity_upper_scaled'])).expand(count).clone())
  if cfg['encoder_type']=='mlp':
   self.encoder=nn.Sequential(nn.Linear(cfg['g_embed_dim']+8,cfg['enc_hidden']),nn.Tanh(),nn.Linear(cfg['enc_hidden'],cfg['latent_dim']))
  if cfg['physics_horizon']=='train':self.collocation &= self.hours<=36.
  if cfg['output_mode']=='exact':
   # No unused neural vector field or decoder in this hard-law baseline.
   del self.ode_func;del self.decoder
   self.initial_head=nn.Linear(cfg['latent_dim'],1)
   nn.init.zeros_(self.initial_head.weight);nn.init.constant_(self.initial_head.bias,-2.5)

 def context(self,embedding,prefix_y,prefix_mask):
  y=prefix_y*prefix_mask
  if self.cfg['input_transform']=='log1p':y=torch.log1p(10*y)/10
  if self.cfg['parameter_context']=='genotype':return torch.cat([embedding,torch.zeros_like(y),torch.zeros_like(prefix_mask)],-1)
  return torch.cat([embedding,y,prefix_mask],-1)

 def forward(self,g_idx,time,env,prefix_y,prefix_mask,prefix_time):
  cfg=self.cfg;n=len(g_idx);emb=self.g_enc(g_idx);context=self.context(emb,prefix_y,prefix_mask)
  y=prefix_y if cfg['input_transform']=='identity' else torch.log1p(10*prefix_y)/10
  z=self.encoder.encode(emb,y,prefix_mask,prefix_time) if cfg['encoder_type']=='lstm' else self.encoder(torch.cat([emb,y*prefix_mask,prefix_mask],-1))
  pars=self.parameter_head(context) if hasattr(self,'parameter_head') else {}
  if cfg['rate_level']=='individual':r=pars[0]
  else:
   rates=F.softplus(self.raw_rate);rates=rates.expand(len(data.GENOTYPES),-1) if cfg['rate_level']=='shared' else rates
   rates=rates[g_idx];r=rates.mean(-1)
  if cfg['rate_form']=='direct' and cfg['contrast_enabled']:
   rL,rD=rates[:,0],rates[:,1];a=(rL-rD)/(rL+rD)
  else:
   a=self.growth_law.contrast(g_idx);rL,rD=r*(1+a),r*(1-a)
  K=None
  if cfg['capacity_level']=='individual':K=pars[1]
  elif cfg['capacity_level'] in ['genotype','shared']:
   caps=cfg['capacity_upper_scaled']*torch.sigmoid(self.raw_capacity)
   K=caps.expand(len(data.GENOTYPES))[g_idx] if cfg['capacity_level']=='shared' else caps[g_idx]
  elif cfg['capacity_level']=='fixed':K=r.new_full(r.shape,cfg['initial_capacity_scaled'])
  out=dict(r=r,r_light=rL,r_dark=rD,a=a,g_emb=emb,t_feat=time[None,:].expand(n,-1))
  if K is not None:out['K']=K
  if cfg['output_mode']=='exact':
   raw=self.initial_head(z).squeeze(-1);t=time*72.;TL=12*torch.floor(t/24.)+torch.remainder(t,24.).clamp(max=12.)
   cumulative=rL[:,None]*TL+rD[:,None]*(t-TL)
   if K is None:pred=F.softplus(raw)[:,None]*torch.exp(cumulative.clamp(max=30))
   else:pred=K[:,None]*torch.sigmoid(raw[:,None]+cumulative)
   out.update(pred=pred,z=z);return out
  trajectory=[z]
  for i in range(len(time)-1):
   step=1./(len(time)-1);t=i*step;forcing=env[:,i] if self.light else env.new_empty((n,0))
   def field(at,state):return self.ode_func(state.new_full((n,1),at),state,forcing,emb)
   k1=field(t,z);k2=field(t+step/2,z+step*k1/2);k3=field(t+step/2,z+step*k2/2);k4=field(t+step,z+step*k3)
   z=z+step*(k1+2*k2+2*k3+k4)/6;trajectory.append(z)
  latent=torch.stack(trajectory,dim=1);pred=self.decode(latent)
  if cfg['anchor_weight']:
   first=prefix_mask.argmax(1);rows=torch.arange(n,device=z.device)
   shift=(prefix_y[rows,first]-pred[rows,first*4])*cfg['anchor_weight'];pred=pred+shift[:,None]
  out.update(pred=pred,z=latent);return out

 def physics_loss(self,out,x):
  if self.cfg['output_mode']=='exact':return out['pred'].sum()*0
  d=self.derivative(out,x);h=out['pred'];light=(self.hours%24<12).to(h.dtype)
  rate=out['r_light'][:,None]*light+out['r_dark'][:,None]*(1-light)
  factor=1-h/out['K'][:,None].clamp_min(1e-4) if 'K' in out else torch.ones_like(h)
  if self.cfg['physics_type']=='relative':residual=d/h.detach().clamp_min(.03)-rate*factor
  else:residual=d-rate*h*factor
  return residual[:,self.collocation].square().mean()

 def phase_loss(self,out,batch):
  # Observed adjacent 12-hour intervals only; no desired genotype ranking.
  ids=torch.tensor([0,4,8,12],device=out['pred'].device)
  available=batch['mask'][:,ids];valid=available[:,:-1]*available[:,1:]
  observed=batch['target'][:,ids];pred=out['pred'][:,ids]
  residual=(pred[:,1:]-pred[:,:-1])-(observed[:,1:]-observed[:,:-1])
  if not bool(valid.any()):return pred.sum()*0
  return (residual.square()*valid).sum()/valid.sum()

 def rate_parameters(self):
  return [p for p in self.growth_law.parameters() if p.requires_grad]+([self.raw_rate] if hasattr(self,'raw_rate') else [])

 def capacity_parameters(self):return [self.raw_capacity] if hasattr(self,'raw_capacity') else []

def initialize(cfg,seed,device='cpu'):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 model=FactorialPhytoODE2(dict(DEFAULTS,**cfg)).to(device)
 for name,value in model.named_parameters():
  if 'lstm' in name and 'weight' in name and value.ndim>=2:nn.init.orthogonal_(value)
 return model

def augment_inputs(x,probability,generator):
 if probability==0:return x
 original_mask=x['prefix_mask'];mask=original_mask*(torch.rand(original_mask.shape,device=original_mask.device,generator=generator)>=probability)
 empty=mask.sum(1)==0;first=original_mask.argmax(1);mask[empty,first[empty]]=1.
 return dict(x,prefix_mask=mask,prefix_y=x['prefix_y']*mask)
