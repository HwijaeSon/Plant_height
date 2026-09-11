"""Verify the expanded search, reused validation runs, and selected checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import forecast_data as data
from no_light_prefix_trial import initialize
from run_expanded_no_light_search import ROOT, PREVIOUS, RESULTS, location, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=RESULTS)
    p.add_argument('--output', type=Path, default=ROOT/'hypocotyl/reports/expanded_no_light_lambda_20260911/artifact_audit.json')
    args = p.parse_args();root=args.results
    torch.set_num_threads(2)
    plan=json.loads((root/'plan.json').read_text())
    selection=json.loads((root/'lambda_selection.json').read_text())
    transfer=json.loads((root/'transfer_plan.json').read_text())
    final=json.loads((root/'test_evaluation_plan.json').read_text())
    completion=json.loads((root/'completion.json').read_text())
    for name,digest in dict(plan['source_sha256'],**plan['reused_files_sha256']).items():
        assert sha(ROOT/name)==digest,name
    assert selection['plan_sha256']==sha(root/'plan.json')
    assert final['lambda_selection_sha256']==sha(root/'lambda_selection.json')
    assert len(plan['added_search_jobs'])==36 and len(plan['reused_search_jobs'])==30
    assert len(selection['candidates'])==22 and len(final['jobs'])==9
    chosen=min(selection['candidates'],key=lambda r:(r['val_rmse_mean'],r['lambda_ode']))['lambda_ode']
    assert chosen==selection['selected_lambda_ode']==completion['selected_lambda_ode']
    assert len(transfer['jobs'])==(0 if chosen==100 else 6)
    cache={};records=[];scores={}
    for job in plan['reused_search_jobs']+plan['added_search_jobs']+transfer['jobs']:
        folder=location(root,job);run=json.loads((folder/'result.json').read_text());cfg=run['config']
        assert run['status']=='trained_validation_only' and run['test_evaluations']==0
        assert set(run['metrics'])=={'train','val'}
        assert cfg['lambda_ode']==job['lambda_ode'] and cfg['lambda_k']==0 and cfg['n_params']==1183
        for key in ['seed','additional_prefix_drop']:
            assert cfg[key]==job['seed' if key=='seed' else 'drop_fraction']
        old=json.loads((PREVIOUS/f'search/lambda_100/seed{cfg["seed"]}/config.json').read_text())
        assert cfg['initial_state_sha256']==old['initial_state_sha256']
        for key,value in plan['profile']['common'].items():
            assert cfg[key]==value
        hist=pd.read_csv(folder/'history.csv')
        assert len(hist)==75 and hist.epoch.iloc[-1]==1500 and not hist.capacity_penalty_used.any()
        np.testing.assert_allclose(hist.objective,hist.data_loss+cfg['lambda_ode']*hist.ode,rtol=2e-6,atol=1e-7)
        np.testing.assert_allclose(hist.selection_rmse.iloc[2:],hist.val_rmse.rolling(3).mean().iloc[2:],rtol=1e-6)
        eligible=hist[hist.epoch.ge(200)]
        assert run['best_epoch']==int(eligible.loc[eligible.selection_rmse.idxmin(),'epoch'])
        ck=(cfg['additional_prefix_drop'],cfg['mask_seed'])
        if ck not in cache:
            cache[ck]=data.load(drop_fraction=ck[0],mask_seed=ck[1],include_test=True)
        dc,plants,x,batches,raw=cache[ck]
        assert cfg['data']['removed_training_observation_ids']==dc['removed_training_observation_ids']
        assert cfg['height_scale_mm']==dc['height_scale_mm']
        with np.load(folder/'predictions.npz') as saved:
            assert not any(k.startswith('test') for k in saved.files)
            for split in ['train','val']:
                np.testing.assert_array_equal(saved[split+'_mask'],batches[split]['mask'])
                np.testing.assert_array_equal(saved[split+'_target'],batches[split]['target'])
                score=data.score(saved['prediction'],batches[split],dc['height_scale_mm'])
                assert abs(score['rmse']-run['metrics'][split]['rmse'])<1e-12
        if job['drop_fraction']==0:
            scores.setdefault(job['lambda_ode'],[]).append(run['metrics']['val']['rmse'])
        records.append(dict(path=job['path'],source=job['source'],validation_only=True,
                            unchanged_initialization=True,objective_and_checkpoint_epoch_verified=True))
    for candidate in selection['candidates']:
        values=scores[candidate['lambda_ode']]
        assert len(values)==3 and abs(np.mean(values)-candidate['val_rmse_mean'])<1e-12
    checks=[]
    for job in final['jobs']:
        folder=root/job['output'];run=json.loads((folder/'result.json').read_text());cfg=run['config']
        assert cfg['lambda_ode']==chosen and cfg['lambda_k']==0
        assert sha(location(root,job['training'])/'checkpoint.pt')==job['checkpoint_sha256']
        assert sha(folder/'checkpoint.pt')==job['checkpoint_sha256']==run['checkpoint_sha256']
        reused='previous_evaluation' in job
        if reused:
            assert chosen==100 and sha(folder/'result.json')==job['previous_result_sha256']
        else:
            assert run['selection_record']==sha(root/'test_evaluation_plan.json')
        saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        model=initialize(cfg['seed'],'cpu').eval();model.load_state_dict(saved['state_dict'])
        ck=(cfg['additional_prefix_drop'],cfg['mask_seed'])
        if ck not in cache:
            cache[ck]=data.load(drop_fraction=ck[0],mask_seed=ck[1],include_test=True)
        dc,plants,x,batches,raw=cache[ck]
        with torch.no_grad():out=model(**x)
        with np.load(folder/'predictions.npz') as values:
            difference=float(np.max(np.abs(out['pred'].numpy()-values['prediction']))*cfg['height_scale_mm'])
            assert difference<1e-5
            for split in ['train','val','test']:
                np.testing.assert_array_equal(values[split+'_mask'],batches[split]['mask'])
                np.testing.assert_array_equal(values[split+'_target'],batches[split]['target'])
                score=data.score(values['prediction'],batches[split],dc['height_scale_mm'])
                assert abs(score['rmse']-run['metrics'][split]['rmse'])<1e-12
        changed={k:v.clone() for k,v in x.items()}
        changed['env']=torch.randn_like(x['env'])*100
        changed['prefix_y'][x['prefix_mask']==0]=1234.
        with torch.no_grad():after=model(**changed)
        for key in ['pred','r','K']:
            torch.testing.assert_close(out[key],after[key],rtol=0,atol=0)
        checks.append(dict(path=job['output'],checkpoint_max_abs_mm=difference,
            prior_test_evaluation_reused=reused,light_and_masked_placeholder_invariant=True))
    assert not completion['failures']
    report=dict(completion,source_hashes_verified=True,all_candidate_training_validation_only=True,
        minimum_three_seed_validation_selected=True,training_checks=records,checkpoint_checks=checks)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['training_checks','checkpoint_checks']},indent=2))


if __name__=='__main__':
    main()
