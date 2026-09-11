"""Audit source-cell extraction, temporal partitions, and missing-input invariance."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'hypocotyl/code'))
import forecast_data as data, forecast_models as models
from forecast_trial import state_sha
from tempfile import TemporaryDirectory
import pandas as pd

torch.set_num_threads(2)
root=ROOT
report={}
with TemporaryDirectory() as temp:
    data.prepare(output=temp)
    for path in Path(temp).iterdir():assert path.read_bytes()==(data.PREPARED/path.name).read_bytes()
report['workbook_reextraction_exact']=True
base,plants,x,batches,raw=data.load(include_test=True)
assert set(x)=={'g_idx','time','env','prefix_y','prefix_mask','prefix_time'}
assert set(raw['train'].elapsed_hours)=={0,12,24,36}
assert set(raw['val'].elapsed_hours)=={48} and set(raw['test'].elapsed_hours)=={60,72}
for a,b in [('train','val'),('train','test'),('val','test')]:
    assert not set(raw[a].observation_id)&set(raw[b].observation_id)
report['counts']={s:dict(plants=f.plant_id.nunique(),observations=len(f)) for s,f in raw.items()}
report['masked_input_invariance']={}
for model_name in ['phytoode','latent_ode_light','latent_ode','logistic_pinn','lstm']:
    torch.manual_seed(1);model=models.build(model_name).eval()
    changed={k:v.clone() for k,v in x.items()}
    changed['prefix_y'][changed['prefix_mask']==0]=12345
    with torch.no_grad():
        before=model(**x)['pred'];after=model(**changed)['pred']
    assert torch.equal(before,after)
    report['masked_input_invariance'][model_name]=dict(exact=True,n_parameters=sum(p.numel() for p in model.parameters()))
report['mask_checks']=[]
for seed in [1,2,3]:
    removed=set()
    for drop in [0,.25,.5]:
        cfg,ps,xx,bb,rr=data.load(include_test=True,drop_fraction=drop,mask_seed=20260910+seed)
        assert cfg['retained_training_observations']==551-round(551*drop)
        assert set(cfg['removed_training_observation_ids'])>=removed
        removed=set(cfg['removed_training_observation_ids'])
        assert (xx['prefix_mask'].sum(1)>0).all()
        assert rr['val'].equals(raw['val']) and rr['test'].equals(raw['test'])
        report['mask_checks'].append(dict(seed=seed,drop=drop,retained_prefix_observations=len(rr['train']),
            missing_fraction=cfg['prefix_missing_fraction'],height_scale_mm=cfg['height_scale_mm']))
(root/'hypocotyl/reports/prefix_forecast_20260911/data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
