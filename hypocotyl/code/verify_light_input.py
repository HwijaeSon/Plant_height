"""Check the new illumination input, switch integration and logistic derivative."""
from pathlib import Path
import json
import numpy as np
import torch
from single_condition_data import load,curve_rmse,HOURS
from light_input_model import inputs,build,physics_losses
from run_light_input import candidates


def main():
    torch.set_num_threads(2); torch.manual_seed(3)
    catalog=candidates(); assert len(catalog)==32
    counts={}
    for protocol in ['replicate','time_holdout']:
        config,old,batches=load(protocol)
        x=inputs()
        for key in old: torch.testing.assert_close(x[key],old[key])
        assert set(batches)=={'train','val'}
        assert all(batch['observations'].sheet.eq('12L12D').all() for batch in batches.values())
        pred=torch.randn(5,25,requires_grad=True)
        loss=curve_rmse(pred,batches['train'])
        gradient=torch.autograd.grad(loss,pred)[0]
        used=torch.zeros_like(pred,dtype=torch.bool)
        used[batches['train']['case'],batches['train']['time']]=True
        assert not bool((gradient[~used]!=0).any())
        counts[protocol]=dict(raw_counts=config['counts'],mean_counts=config['mean_target_counts'])
    assert set(x['env'].unique().tolist())=={0.,1.}
    np.testing.assert_array_equal(x['env'][0,::4,0],[1,0,1,0,1,0,1])
    net=build(catalog['light_00'])
    assert net.encoder.lstm.input_size==2 and net.ode_func.net[0].in_features==14
    assert net.ode_param_head.net[-1].out_features==2
    assert sum(p.numel() for p in net.parameters())==1103
    out=net(**x); ode,k=physics_losses(net,out,x)
    (curve_rmse(out['pred'],batches['train'])+.5*ode+.01*k).backward()
    grads=[p.grad for p in net.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert net.ode_func.net[0].weight.grad[:,8].abs().sum()>0
    assert net.encoder.lstm.weight_ih_l0.grad[:,0].abs().sum()>0
    changed=net(**(x|dict(env=1-x['env'])))['pred']
    input_effect=float((changed-out['pred']).abs().max().detach())
    assert input_effect>0
    net=build(catalog['light_00']).double()
    xd={k:v.double() if v.is_floating_point() else v for k,v in x.items()}
    out=net(**xd); exact=net.derivative(out,xd).detach()
    z=out['z'].detach(); embedding=out['g_emb'].detach(); B,T,D=z.shape
    with torch.no_grad():
        field=net.ode_func(xd['time'][None,:,None].expand(B,-1,-1).reshape(-1,1),z.reshape(-1,D),
            xd['env'].reshape(B*T,1),embedding[:,None,:].expand(-1,T,-1).reshape(B*T,-1)).reshape(B,T,D)
        eps=1e-5
        numerical=(net.decode(z+eps*field)-net.decode(z-eps*field))/(2*eps*72.)
    np.testing.assert_allclose(exact,numerical,rtol=1e-5,atol=1e-10)
    derivative_error=float((exact-numerical).abs().max())
    # A field dz/dtau=L has an exact integral at every grid node. In particular,
    # RK4 must not mix the new state into the final stage before a switch.
    class Exposure(torch.nn.Module):
        def forward(self,time,z,env,embedding): return env.expand_as(z)
    net.ode_func=Exposure()
    with torch.no_grad():
        result=net(**xd); growth=result['z']-result['z'][:,:1]
        expected=np.r_[0,np.cumsum(xd['env'][0,:-1,0].numpy())/24]
        np.testing.assert_allclose(growth.numpy(),np.broadcast_to(expected[None,:,None],growth.shape),atol=1e-14)
    report=dict(status='passed',binary_light=True,condition='12L12D',excluded_condition='cR',
        target_type='split_specific_replicate_means',test_targets_loaded=False,
        existing_split_indices_unchanged=True,missing_target_gradient_zero=True,
        parameter_count=1103,ordinary_logistic_head_outputs=['r','K'],
        light_gradients_nonzero=True,schedule_effect_on_initial_model=input_effect,
        chain_rule_derivative_max_abs_error=derivative_error,
        piecewise_constant_RK4_exposure_integral_checked=True,protocol_counts=counts)
    dest=Path(__file__).resolve().parents[1]/'reports/light_input_20260910'
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
