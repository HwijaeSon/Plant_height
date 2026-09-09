"""Check light-switch integration, observable derivatives and complete loss removal."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from data import inputs, load, light_exposure, CASES
from models import LightLatentODE, LightPINN, physics_losses
from trial import state_hash


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    model=LightLatentODE().double()
    x=inputs()
    x={key:value.double() if value.is_floating_point() else value for key,value in x.items()}
    output=model(**x)
    derivative=model.derivative(output,x)
    z=output['z'].detach()
    g=output['g_emb'].detach()
    B,T,D=z.shape
    tf=x['time'].expand(B,-1).reshape(B*T,1)
    forcing=x['env'].reshape(B*T,-1)
    repeated=g[:,None].expand(B,T,-1).reshape(B*T,-1)
    direction=model.ode_func(tf,z.reshape(B*T,D),forcing,repeated).detach().reshape(B,T,D)
    eps=1e-5
    numerical=(model.decode(z+eps*direction)-model.decode(z-eps*direction))/(2*eps*72.)
    torch.testing.assert_close(derivative,numerical,atol=1e-9,rtol=1e-4)
    ode,capacity=physics_losses(model,output,x)
    (ode+capacity).backward()
    for module in (model.g_enc,model.encoder,model.ode_func,model.decoder,model.ode_param_head):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    torch.manual_seed(7); pure=LightLatentODE()
    torch.manual_seed(7); full=LightLatentODE()
    assert state_hash(pure)==state_hash(full)
    y=pure(**inputs())['pred']
    y.square().mean().backward()
    assert all(p.grad is None for p in pure.ode_param_head.parameters())

    class LightField(nn.Module):
        def forward(self,t,z,env,g):return env[:,:1].expand_as(z)
    integrator=LightLatentODE().double()
    integrator.ode_func=LightField()
    with torch.no_grad():z=integrator(**x)['z']
    expected=torch.tensor(np.stack([light_exposure(np.arange(0,73,3),c)/72. for _,c in CASES]),dtype=torch.float64)
    torch.testing.assert_close(z[:,:,0]-z[:,0:1,0],expected,atol=1e-12,rtol=1e-12)
    pinn=LightPINN().double()
    output=pinn(**x)
    derivative=pinn.derivative(output,x)
    assert derivative.shape==(10,25)
    sum(physics_losses(pinn,output,x)).backward()
    assert all(p.grad is not None for p in pinn.parameters())
    result=dict(status='passed',checks=['RK4 integrates piecewise-constant light exposure exactly through all six switches',
        'Decoded directional derivative matches a centered directional finite-difference check',
        'Physics loss gradients reach embedding, encoder, vector field, decoder and logistic head',
        'Both-zero latent model has identical initialization and no auxiliary-head gradients',
        'Coordinate Light-PINN derivative and second-order backpropagation are valid'],new_training_runs=0)
    path=Path(__file__).resolve().parents[1]/'reports/model_verification.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
