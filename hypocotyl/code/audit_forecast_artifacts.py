"""Recompute all scores and reload every neural checkpoint, with paired-mask checks."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'hypocotyl/code'))
import forecast_data as data,forecast_models as models

torch.set_num_threads(2)
root=ROOT;results=root/'hypocotyl/results/prefix_forecast_20260911'
plan=json.loads((results/'plan.json').read_text())
for path,digest in plan['code_sha256'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
checks=[];cached={}
for job in plan['jobs']:
 out=results/job['path'];record=json.loads((out/'result.json').read_text());cfg=record['config']
 assert record['status']=='complete'
 key=(job['drop_fraction'],job['mask_seed'])
 if key not in cached:cached[key]=data.load(include_test=True,drop_fraction=key[0],mask_seed=key[1])
 dc,plants,x,batches,raw=cached[key]
 assert cfg['data']['removed_training_observation_ids']==dc['removed_training_observation_ids']
 assert cfg['height_scale_mm']==dc['height_scale_mm']
 saved=np.load(out/'predictions.npz');prediction=saved['prediction']
 np.testing.assert_array_equal(saved['plant_ids'],plants.plant_id)
 for split in ['train','val','test']:
  np.testing.assert_array_equal(saved[split+'_mask'],batches[split]['mask'])
  np.testing.assert_array_equal(saved[split+'_target'],batches[split]['target'])
  metric=data.score(prediction,batches[split],dc['height_scale_mm'])
  assert abs(metric['rmse']-record['metrics'][split]['rmse'])<1e-12
 check=dict(path=job['path'],scores_recomputed=True,paired_mask_verified=True)
 if job['model'] not in ['logistic','rf']:
  checkpoint=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
  model=models.build(job['model']).eval();model.load_state_dict(checkpoint['state_dict'])
  with torch.no_grad():reloaded=model(**x)['pred'].numpy()
  error=float(np.max(np.abs(reloaded-prediction))*dc['height_scale_mm'])
  assert error<5e-4,(out,error)
  selection=json.loads((out/'selection.json').read_text())
  assert selection['best_epoch']==record['best_epoch']==checkpoint['epoch']
  assert selection['test_evaluations']==0
  assert selection['checkpoint_sha256']==hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest()
  check['cpu_checkpoint_max_abs_mm']=error
 elif job['model']=='logistic':
  params=json.loads((out/'parameters.json').read_text())['parameters']
  assert all(p['success'] for p in params)
  check['all_genotypes_converged']=True
 checks.append(check)
value=dict(expected_fits=61,completed_fits=len(checks),frozen_code_sha_verified=True,checks=checks,
 max_cpu_checkpoint_error_mm=max(c.get('cpu_checkpoint_max_abs_mm',0.) for c in checks))
(results.parent.parent/'reports/prefix_forecast_20260911/artifact_audit.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:v for k,v in value.items() if k!='checks'},indent=2))
