"""Add one known binary illumination input; retain ordinary logistic physics."""
import numpy as np
import torch
from single_condition_data import HOURS, inputs as time_inputs
from single_condition_models import CORE, LogisticParameterHead, physics_losses


def inputs(hours=HOURS, device='cpu'):
    hours=np.asarray(hours,dtype=float)
    base=time_inputs(hours,device=device)
    assert np.isclose(12./(hours[1]-hours[0]),round(12./(hours[1]-hours[0]))), 'Grid must include every light switch.'
    state=(np.remainder(hours,24.)<12.).astype(float)
    base['env']=torch.tensor(np.broadcast_to(state[None,:,None],(5,len(hours),1)).copy(),
                             dtype=torch.float32,device=device)
    return base


class LightInputPhytoODE(CORE.LatentODEHeightModel):
    def __init__(self):
        super().__init__(n_genotypes=5,env_dim=1,latent_dim=8,g_embed_dim=4,
            ode_hidden=16,ode_layers=1,dec_hidden=16,enc_hidden=8,
            use_physics=False,days_per_tau=72.)
        self.ode_param_head=LogisticParameterHead()

    def forward(self,g_idx,time,env,encoder_time=None,encoder_env=None):
        B,T,C=env.shape
        assert C==1 and B==len(g_idx) and T==len(time)
        embedding=self.g_enc(g_idx); times=time.unsqueeze(0).expand(B,-1)
        enc_t=time if encoder_time is None else encoder_time
        enc_env=env if encoder_env is None else encoder_env
        z,_,_=self.get_z0(embedding,enc_env,enc_t.unsqueeze(0).expand(B,-1))
        trajectory=[z]; step=1./(T-1)
        for i in range(T-1):
            # The right-end RK4 stage uses the current interval's state. The
            # next step uses the new state, without a ramp across a switch.
            forcing=env[:,i]
            def field(t,state):
                tf=torch.full((B,1),t,dtype=state.dtype,device=state.device)
                return self.ode_func(tf,state,forcing,embedding)
            t=i*step
            k1=field(t,z); k2=field(t+step/2,z+step*k1/2)
            k3=field(t+step/2,z+step*k2/2); k4=field(t+step,z+step*k3)
            z=z+step*(k1+2*k2+2*k3+k4)/6
            trajectory.append(z)
        latent=torch.stack(trajectory,dim=1); r,K=self.ode_param_head(embedding)
        return dict(pred=self.decode(latent),z=latent,g_emb=embedding,t_feat=times,r=r,K=K)

    def derivative(self,output,inputs):
        # At switching boundaries this evaluates the right-hand derivative.
        # The ordinary-logistic residual retains the same 23 interior nodes as
        # the existing no-input experiment; no light/dark rate switch is added.
        return self.dydt(output['z'],inputs['env'],output['g_emb'],output['t_feat'])[0]


def build(config):
    assert config['model']=='phytoode_light'
    assert config['lambda_ode']>0 and config['lambda_k']>0
    return LightInputPhytoODE()
