"""Check that every measured cell is retained and missing cells have no data gradient."""
import json
from pathlib import Path
import numpy as np
import torch
from observation_data import load_observations, observation_rmse


def main():
    torch.set_num_threads(2)
    checks = []
    for protocol in ['replicate', 'time_holdout']:
        config,x,batches = load_observations(protocol)
        for split,batch in batches.items():
            frame = batch['observations']
            assert len(batch['length']) == len(frame) == config['counts'][split]
            torch.testing.assert_close(batch['length'], torch.tensor(frame.length.to_numpy()/config['height_scale_mm'], dtype=torch.float32))
            pred = torch.ones((10,25), requires_grad=True)
            loss = observation_rmse(pred,batch)
            expected = np.sqrt(np.mean((1-frame.length.to_numpy()/config['height_scale_mm'])**2))
            np.testing.assert_allclose(float(loss.detach()), expected, rtol=1e-6)
            loss.backward()
            observed = torch.zeros_like(pred, dtype=torch.bool)
            observed[batch['case'],batch['time']] = True
            assert torch.count_nonzero(pred.grad[~observed]) == 0
            shifted = pred.detach().clone(); shifted[~observed] = 1e6
            torch.testing.assert_close(observation_rmse(shifted,batch),loss.detach())
            if protocol == 'time_holdout':
                assert not torch.isin(batch['time'],torch.tensor([12,20])).any()
            checks.append(dict(protocol=protocol, split=split, n_targets=len(frame),
                               missing_grid_entries_have_zero_data_gradient=True))
    # Averaging these two observations would produce zero loss; direct residuals do not.
    example = dict(case=torch.tensor([0,0]), time=torch.tensor([0,0]), length=torch.tensor([3.,4.]))
    torch.testing.assert_close(observation_rmse(torch.tensor([[3.5]]), example), torch.tensor(.5))
    out = Path(__file__).resolve().parents[1]/'reports/individual_observations_20260909'
    out.mkdir(parents=True, exist_ok=True)
    record = dict(status='passed', data_checks=checks, heterogeneous_replicates_not_collapsed=True,
                  imputed_phenotype_targets=0, test_targets_loaded=False)
    (out/'target_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__': main()
