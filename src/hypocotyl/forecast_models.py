"""Prefix-conditioned neural and process baselines for longitudinal forecasting."""
import numpy as np
import torch
from torch import nn
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from single_condition_models import CORE,LogisticParameterHead
from forecast_data import HOURS,GENOTYPES


class PrefixEncoder(CORE.EnvEncoder):
    """Reverse LSTM of prefix length, presence mask, optional light, and time."""
    def __init__(self,light):
        super().__init__(env_dim=2+int(light),g_dim=4,hidden_size=8,latent_dim=8)
        self.light=light
    def encode(self,embedding,prefix_y,prefix_mask,prefix_time):
        features=[prefix_y*prefix_mask,prefix_mask]
        if self.light:
            features.append(((prefix_time*72)%24<12).float()[None,:].expand_as(prefix_y))
        return self.forward(torch.stack(features,dim=-1),prefix_time[None,:].expand_as(prefix_y),embedding)[0]


class PrefixLatentODE(CORE.LatentODEHeightModel):
    def __init__(self,light=True):
        super().__init__(n_genotypes=len(GENOTYPES),env_dim=int(light),latent_dim=8,g_embed_dim=4,
            ode_hidden=16,ode_layers=1,dec_hidden=16,enc_hidden=8,use_physics=False,days_per_tau=72.)
        self.light=light
        self.encoder=PrefixEncoder(light)
        self.ode_param_head=LogisticParameterHead()
    def forward(self,g_idx,time,env,prefix_y,prefix_mask,prefix_time):
        n=len(g_idx); embedding=self.g_enc(g_idx)
        z=self.encoder.encode(embedding,prefix_y,prefix_mask,prefix_time)
        trajectory=[z]
        for i in range(len(time)-1):
            step=1./(len(time)-1); t=i*step
            forcing=env[:,i] if self.light else env.new_empty((n,0))
            def field(t,z): return self.ode_func(z.new_full((n,1),t),z,forcing,embedding)
            k1=field(t,z);k2=field(t+step/2,z+step*k1/2);k3=field(t+step/2,z+step*k2/2);k4=field(t+step,z+step*k3)
            z=z+step*(k1+2*k2+2*k3+k4)/6;trajectory.append(z)
        latent=torch.stack(trajectory,dim=1);r,K=self.ode_param_head(embedding)
        return dict(pred=self.decode(latent),z=latent,g_emb=embedding,t_feat=time[None,:].expand(n,-1),r=r,K=K)
    def derivative(self,out,x):
        env=x['env'] if self.light else x['env'].new_empty((len(x['g_idx']),len(x['time']),0))
        return self.dydt(out['z'],env,out['g_emb'],out['t_feat'])[0]


class PrefixPINN(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(len(GENOTYPES),4)
        self.head=CORE.mlp([13,32,32,1]);self.output=nn.LeakyReLU(.01)
        self.ode_param_head=LogisticParameterHead()
    def forward(self,g_idx,time,env,prefix_y,prefix_mask,prefix_time):
        t=time[None,:].expand(len(g_idx),-1).detach().clone();t.requires_grad_(torch.is_grad_enabled())
        g=self.embedding(g_idx);context=torch.cat([g,prefix_y*prefix_mask,prefix_mask],-1)
        pred=self.output(self.head(torch.cat([t[...,None],context[:,None,:].expand(-1,len(time),-1)],-1))).squeeze(-1)
        r,K=self.ode_param_head(g)
        return dict(pred=pred,time=t,r=r,K=K)
    def derivative(self,out,x):return torch.autograd.grad(out['pred'].sum(),out['time'],create_graph=True)[0]/72.


class PrefixLSTM(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(len(GENOTYPES),4)
        self.lstm1=nn.LSTM(9,16,batch_first=True);self.lstm2=nn.LSTM(16,8,batch_first=True)
        self.head=nn.Sequential(nn.Linear(12,16),nn.LeakyReLU(.01),nn.Linear(16,1),nn.LeakyReLU(.01))
    def forward(self,g_idx,time,env,prefix_y,prefix_mask,prefix_time):
        context=torch.cat([prefix_y*prefix_mask,prefix_mask],-1)
        inp=torch.cat([time[None,:,None].expand(len(g_idx),-1,-1),context[:,None,:].expand(-1,len(time),-1)],-1)
        z,_=self.lstm1(inp);z,_=self.lstm2(z)
        g=self.embedding(g_idx)[:,None,:].expand(-1,len(time),-1)
        return dict(pred=self.head(torch.cat([z,g],-1)).squeeze(-1))


def build(name):
    if name in ['phytoode','latent_ode_light']:return PrefixLatentODE(light=True)
    if name=='latent_ode':return PrefixLatentODE(light=False)
    if name=='logistic_pinn':return PrefixPINN()
    if name=='lstm':return PrefixLSTM()
    raise ValueError(name)


def physics_losses(model,out,x):
    derivative=model.derivative(out,x)
    rhs=out['r'][:,None]*out['pred']*(1-out['pred']/out['K'][:,None].clamp_min(1e-4))
    ode=(derivative[:,1:-1]-rhs[:,1:-1]).square().mean()
    capacity=(out['pred'].max(1).values-out['K']).abs().mean()
    return ode,capacity


def forest_features(x):
    g=x['g_idx'].cpu().numpy();prefix=(x['prefix_y']*x['prefix_mask']).cpu().numpy();mask=x['prefix_mask'].cpu().numpy()
    context=np.concatenate([np.eye(len(GENOTYPES))[g],prefix,mask],1)
    time=x['time'].cpu().numpy()
    return np.concatenate([np.broadcast_to(context[:,None,:],(len(g),len(time),context.shape[1])),
                           np.broadcast_to(time[None,:,None],(len(g),len(time),1))],2)


def fit_forest(x,batch,seed):
    features=forest_features(x);mask=batch['mask'].cpu().numpy().astype(bool)
    model=RandomForestRegressor(n_estimators=300,min_samples_leaf=1,random_state=seed,n_jobs=2)
    model.fit(features[mask],batch['target'].cpu().numpy()[mask])
    return model.predict(features.reshape(-1,features.shape[-1])).reshape(features.shape[:2]),model


def fit_logistic(x,batch):
    # Shared genotype-specific r and K, with a fitted initial length per plant.
    g=x['g_idx'].cpu().numpy();target=batch['target'].cpu().numpy();mask=batch['mask'].cpu().numpy().astype(bool)
    prediction=np.empty_like(target,dtype=float);params=[]
    hours=x['time'].cpu().numpy().astype(float)*72
    for gi,genotype in enumerate(GENOTYPES):
        selected=np.flatnonzero(g==gi);local_mask=mask[selected]
        pi,ti=np.where(local_mask);y=target[selected][local_mask].astype(float);t=hours[ti];n=len(selected)
        def evaluate(p):
            r,K=p[:2];f=p[2:][pi];e=np.exp(-r*t);b=1/f-1;den=1+b*e
            return K/den,e,b,den,f
        def residual(p):return evaluate(p)[0]-y
        def jacobian(p):
            h,e,b,den,f=evaluate(p);r,K=p[:2]
            jac=np.zeros((len(y),2+n));jac[:,0]=K*b*t*e/den**2;jac[:,1]=1/den
            jac[np.arange(len(y)),2+pi]=K*e/(den**2*f**2)
            return jac
        first=local_mask.argmax(1);initial=target[selected,np.asarray(first)]
        best=None
        for rate in [.025,.05,.1,.2]:
            for factor in [1.1,1.6]:
                K=np.clip(float(y.max())*factor,1e-3,1.9)
                f=1/(1+(K/np.maximum(initial,1e-6)-1)*np.exp(rate*hours[first]))
                p=np.r_[rate,K,np.clip(f,1e-5+1e-6,.99999-1e-6)]
                fit=least_squares(residual,p,jac=jacobian,bounds=(np.r_[1e-5,1e-4,np.full(n,1e-5)],
                    np.r_[.5,2.,np.full(n,.99999)]),max_nfev=5000,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                if best is None or fit.cost<best.cost:best=fit
        r,K=best.x[:2];f=best.x[2:]
        prediction[selected]=K/(1+(1/f[:,None]-1)*np.exp(-r*hours[None,:]))
        params.append(dict(genotype=genotype,r_per_hour=float(r),K_scaled=float(K),
            plant_indices=selected.tolist(),initial_fractions=f.tolist(),success=bool(best.success),cost=float(best.cost)))
    return prediction,params
