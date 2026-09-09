"""Verify filtering, split-only means, missing-target gradients and logistic derivatives."""
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from single_condition_data import DATA,GENOTYPES,HOURS,load,curve_rmse,inputs
from single_condition_models import build,physics_losses
from single_condition_baselines import logistic_curve,forest_features
from run_single_condition import OUT,candidates,sha


def main():
    torch.set_num_threads(2); checks={}; catalog=candidates()
    for protocol in ['replicate','time_holdout']:
        config,x,batches=load(protocol)
        assert set(x)=={'g_idx','time'}
        for split,batch in batches.items():
            source=pd.read_csv(DATA.parent/'splits'/protocol/f'{split}.csv')
            source=source[source.sheet.eq('12L12D')]
            assert set(source.observation_id)==set(batch['observations'].observation_id)
            calculated=source.groupby(['genotype','elapsed_hours']).length.mean().sort_index()
            stored=batch['means'].set_index(['genotype','elapsed_hours']).mean_length_mm.sort_index()
            np.testing.assert_allclose(calculated,stored,atol=1e-14,rtol=1e-14)
            prediction=torch.randn(5,25,requires_grad=True)
            value=curve_rmse(prediction,batch)
            gradient=torch.autograd.grad(value,prediction)[0]
            used=torch.zeros_like(prediction,dtype=torch.bool); used[batch['case'],batch['time']]=True
            assert not bool((gradient[~used]!=0).any())
            altered=prediction.detach().clone(); altered[~used]=float('nan')
            torch.testing.assert_close(value,curve_rmse(altered,batch))
            if protocol=='time_holdout': assert not set(batch['means'].elapsed_hours)&{36,60}
        assert config['height_scale_mm']==batches['train']['observations'].length.max()
        checks[protocol]=dict(counts=config['counts'],mean_targets=config['mean_target_counts'],
            height_scale_mm=config['height_scale_mm'],train_val_means_verified=True,missing_target_gradient_zero=True)
    config,x,batches=load('replicate')
    details={}
    for name in ['phytoode','latent_ode','logistic_pinn','lstm']:
        torch.manual_seed(1); net=build(catalog[name+'_00']); out=net(**x)
        assert out['pred'].shape==(5,25)
        assert set(inspect.signature(net.forward).parameters)<= {'g_idx','time','encoder_time'}
        if name in ['phytoode','latent_ode']:
            assert net.encoder.lstm.input_size==1
            assert net.ode_func.net[0].in_features==8+4+1
        loss=curve_rmse(out['pred'],batches['train'])
        if name in ['phytoode','logistic_pinn']:
            ode,capacity=physics_losses(net,out,x); loss=loss+.5*ode+.01*capacity
        loss.backward()
        grads=[p.grad for p in net.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        if name=='latent_ode': assert all(p.grad is None for p in net.ode_param_head.parameters())
        details[name]=dict(n_parameters=sum(p.numel() for p in net.parameters()),backward_finite=True)
    # Independent central difference along the learned latent vector field.
    torch.manual_seed(2); net=build(catalog['phytoode_00']).double()
    xd={k:(v.double() if v.is_floating_point() else v) for k,v in x.items()}
    output=net(**xd); exact=net.derivative(output,xd).detach()
    z=output['z'].detach(); embedding=output['g_emb'].detach(); B,T,D=z.shape
    with torch.no_grad():
        field=net.ode_func(xd['time'][None,:,None].expand(B,-1,-1).reshape(-1,1),z.reshape(-1,D),
            z.new_empty(B*T,0),embedding[:,None,:].expand(-1,T,-1).reshape(B*T,-1)).reshape(B,T,D)
        eps=1e-5
        numerical=(net.decode(z+eps*field)-net.decode(z-eps*field))/(2*eps*72.)
    np.testing.assert_allclose(exact,numerical,rtol=1e-5,atol=1e-10)
    derivative_error=float((exact-numerical).abs().max())
    t=np.array([6.,18.,42.,66.]); eps=1e-4; r,K,q=.07,1.2,.1
    y=logistic_curve(t,r,K,q)
    numerical=(logistic_curve(t+eps,r,K,q)-logistic_curve(t-eps,r,K,q))/(2*eps)
    np.testing.assert_allclose(numerical,r*y*(1-y/K),rtol=1e-8,atol=1e-10)
    assert forest_features().shape==(125,6)
    record=dict(status='passed',condition='12L12D',environmental_feature_count=0,
        test_targets_loaded=False,replicate_means_used=True,phenotype_imputation=False,
        derivative_max_abs_difference=derivative_error,
        logistic_solution_checked=True,models=details,protocols=checks)
    dest=OUT.parents[1]/'reports/single_condition_20260909'; dest.mkdir(parents=True,exist_ok=True)
    (dest/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
