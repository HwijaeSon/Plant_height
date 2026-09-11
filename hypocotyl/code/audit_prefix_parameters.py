"""Check source hashes, selection rules, absence of test scoring during search, and checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import forecast_data as data
from forecast_trial import sha
from prefix_parameter_trial import initialize

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'hypocotyl/results/prefix_parameters_20260911')
    parser.add_argument('--output', type=Path, default=ROOT/'hypocotyl/reports/prefix_parameters_20260911/artifact_audit.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads((args.results/'plan.json').read_text())
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    selection = json.loads((args.results/'lambda_selection.json').read_text())
    chosen = min(selection['candidates'], key=lambda x:(x['val_rmse_mean'], x['lambda_ode']))
    assert selection['selected_lambda_ode'] == chosen['lambda_ode']
    records = []
    cache = {}
    search = list((args.results/'search').glob('*/seed*/result.json'))
    transfer = list((args.results/'transfer').glob('*/seed*/result.json'))
    assert len(search) == 30 and len(transfer) == 6
    for path in search+transfer:
        run = json.loads(path.read_text())
        cfg = run['config']
        assert cfg['lambda_k'] == 0 and cfg['n_params'] == 1231
        assert run['test_evaluations'] == 0 and set(run['metrics']) == {'train', 'val'}
        history = pd.read_csv(path.parent/'history.csv')
        error = np.max(np.abs(history.objective-history.data_loss-cfg['lambda_ode']*history.ode))
        assert error < 1e-6, (path,error)
        assert not history.capacity_penalty_used.any()
        eligible = history[history.epoch.ge(200)]
        assert run['best_epoch'] == int(eligible.loc[eligible.selection_rmse.idxmin(),'epoch'])
        with np.load(path.parent/'predictions.npz') as saved:
            assert not any(key.startswith('test') for key in saved.files)
            key = (cfg['additional_prefix_drop'], cfg['mask_seed'])
            if key not in cache:
                cache[key] = data.load(drop_fraction=key[0], mask_seed=key[1], include_test=True)
            dc,plants,x,batches,raw = cache[key]
            for split in ['train','val']:
                np.testing.assert_array_equal(saved[split+'_mask'],batches[split]['mask'])
                np.testing.assert_array_equal(saved[split+'_target'],batches[split]['target'])
                score = data.score(saved['prediction'],batches[split],dc['height_scale_mm'])
                assert abs(score['rmse']-run['metrics'][split]['rmse'])<1e-12
        records.append(dict(path=str(path.parent.relative_to(args.results)), validation_only=True,
                            no_capacity_term_in_objective=True, selected_epoch_verified=True))
    for candidate in selection['candidates']:
        values = [json.loads(path.read_text())['metrics']['val']['rmse'] for path in search
                  if json.loads(path.read_text())['config']['lambda_ode']==candidate['lambda_ode']]
        assert len(values)==3 and abs(np.mean(values)-candidate['val_rmse_mean'])<1e-12
    final_plan = json.loads((args.results/'test_evaluation_plan.json').read_text())
    selected = []
    for job in final_plan['jobs']:
        folder = args.results/job['output']
        run = json.loads((folder/'result.json').read_text())
        cfg = run['config']
        assert cfg['lambda_ode'] == chosen['lambda_ode'] and run['test_evaluations']==1
        assert sha(folder/'checkpoint.pt') == job['checkpoint_sha256'] == run['checkpoint_sha256']
        assert run['selection_record'] == sha(args.results/'test_evaluation_plan.json')
        checkpoint = torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        model = initialize(cfg['seed'],'cpu').eval()
        model.load_state_dict(checkpoint['state_dict'])
        dc,plants,x,batches,raw = cache[(cfg['additional_prefix_drop'],cfg['mask_seed'])]
        with torch.no_grad():
            result = model(**x)
        with np.load(folder/'predictions.npz') as saved:
            prediction = result['pred'].numpy()
            difference = float(np.max(np.abs(prediction-saved['prediction']))*dc['height_scale_mm'])
            assert difference < 1e-5
            for split in ['train','val','test']:
                np.testing.assert_array_equal(saved[split+'_target'],batches[split]['target'])
                np.testing.assert_array_equal(saved[split+'_mask'],batches[split]['mask'])
                score = data.score(saved['prediction'],batches[split],dc['height_scale_mm'])
                assert abs(score['rmse']-run['metrics'][split]['rmse'])<1e-12
        changed={k:v.clone() for k,v in x.items()}
        changed['prefix_y'][changed['prefix_mask']==0]=98765.
        with torch.no_grad():
            after=model(**changed)
        for field in ['pred','r','K']:
            torch.testing.assert_close(result[field],after[field],rtol=0,atol=0)
        selected.append(dict(path=job['output'], checkpoint_max_abs_mm=difference,
            masked_placeholder_invariant=True, coefficient_shared_across_missingness=True,
            learned_prefix_column_norm=float(model.ode_param_head.net[0].weight[:,4:].detach().norm())))
    report=dict(search_fits=30,transfer_fits=6,selected_test_evaluations=9,
        selected_lambda_ode=chosen['lambda_ode'],lambda_k=0.,source_sha_verified=True,
        minimum_three_seed_validation_selected=True,training_checks=records,checkpoint_checks=selected)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['training_checks','checkpoint_checks']},indent=2))


if __name__ == '__main__':
    main()
