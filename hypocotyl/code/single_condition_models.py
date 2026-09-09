"""No environmental inputs: ordinary logistic regularization of genotype/time models."""
import importlib.util
from pathlib import Path
import torch
from torch import nn

source=Path(__file__).resolve().parents[2]/'wheat/code/model.py'
spec=importlib.util.spec_from_file_location('single_condition_core',source)
CORE=importlib.util.module_from_spec(spec); spec.loader.exec_module(CORE)


class LogisticParameterHead(nn.Module):
    def __init__(self):
        super().__init__(); self.net=CORE.mlp([4,8,2])
        with torch.no_grad():
            self.net[-1].weight.mul_(.01)
            self.net[-1].bias.copy_(torch.logit(torch.tensor([.1,.4])))

    def forward(self,embedding):
        values=torch.sigmoid(self.net(embedding))
        return .5*values[...,0],2.*values[...,1]


class GenotypeLatentODE(CORE.LatentODEHeightModel):
    def __init__(self):
        super().__init__(n_genotypes=5,env_dim=0,latent_dim=8,g_embed_dim=4,
            ode_hidden=16,ode_layers=1,dec_hidden=16,enc_hidden=8,
            use_physics=False,days_per_tau=72.)
        self.ode_param_head=LogisticParameterHead()

    def forward(self,g_idx,time,encoder_time=None):
        B,T=len(g_idx),len(time)
        # This tensor has zero channels, not a constant or masked environment feature.
        empty=time.new_empty(B,T,0); embedding=self.g_enc(g_idx)
        times=time.unsqueeze(0).expand(B,-1)
        enc_t=time if encoder_time is None else encoder_time
        z0,_,_=self.get_z0(embedding,time.new_empty(B,len(enc_t),0),enc_t.unsqueeze(0).expand(B,-1))
        latent=CORE.odeint_rk4(self.ode_func,z0,T,CORE.Interp(empty),embedding,self.n_substeps)
        r,K=self.ode_param_head(embedding)
        return dict(pred=self.decode(latent),z=latent,g_emb=embedding,t_feat=times,r=r,K=K)

    def derivative(self,output,inputs):
        B,T=output['pred'].shape
        return self.dydt(output['z'],output['pred'].new_empty(B,T,0),output['g_emb'],output['t_feat'])[0]


class LogisticPINN(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(5,4)
        self.head=CORE.mlp([5,32,32,1]); self.output=nn.LeakyReLU(.01)
        self.ode_param_head=LogisticParameterHead()

    def forward(self,g_idx,time):
        t=time.unsqueeze(0).expand(len(g_idx),-1).detach().clone()
        t.requires_grad_(torch.is_grad_enabled())
        embedding=self.embedding(g_idx)
        repeated=embedding.unsqueeze(1).expand(-1,len(time),-1)
        pred=self.output(self.head(torch.cat([t.unsqueeze(-1),repeated],dim=-1))).squeeze(-1)
        r,K=self.ode_param_head(embedding)
        return dict(pred=pred,time=t,r=r,K=K)

    def derivative(self,output,inputs):
        return torch.autograd.grad(output['pred'].sum(),output['time'],create_graph=True)[0]/72.


class LSTMNN(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(5,4)
        self.lstm1=nn.LSTM(1,16,batch_first=True); self.lstm2=nn.LSTM(16,8,batch_first=True)
        self.head=nn.Sequential(nn.Linear(12,16),nn.LeakyReLU(.01),nn.Linear(16,1),nn.LeakyReLU(.01))

    def forward(self,g_idx,time):
        t=time.unsqueeze(0).expand(len(g_idx),-1)
        hidden,_=self.lstm1(t.unsqueeze(-1)); hidden,_=self.lstm2(hidden)
        embedding=self.embedding(g_idx).unsqueeze(1).expand(-1,len(time),-1)
        return dict(pred=self.head(torch.cat([hidden,embedding],dim=-1)).squeeze(-1))


def physics_losses(model,output,inputs):
    derivative=model.derivative(output,inputs)
    rhs=output['r'][:,None]*output['pred']*(1-output['pred']/output['K'][:,None].clamp_min(1e-4))
    # A smooth ordinary logistic law: all 23 interior nodes are collocation points.
    ode=(derivative[:,1:-1]-rhs[:,1:-1]).square().mean()
    capacity=(output['pred'].max(dim=1).values-output['K']).abs().mean()
    return ode,capacity


def build(config):
    if config['model'] in ['phytoode','latent_ode']:
        if config['model']=='latent_ode': assert config['lambda_ode']==config['lambda_k']==0
        return GenotypeLatentODE()
    if config['model']=='logistic_pinn': return LogisticPINN()
    if config['model']=='lstm': return LSTMNN()
    raise ValueError(config['model'])
