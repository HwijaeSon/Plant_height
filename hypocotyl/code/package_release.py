"""Prepare a depositable data/code package; this does not publish or assign a license."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import zipfile

import torch

ROOT=Path(__file__).resolve().parents[2]


def main():
    output=ROOT/'hypocotyl/release'; output.mkdir(exist_ok=True)
    environment=dict(python=platform.python_version(),platform=platform.platform(),
        torch_cuda=torch.version.cuda,packages={p:importlib.metadata.version(p) for p in
        ['numpy','pandas','torch','scipy','scikit-learn','matplotlib','openpyxl','joblib']})
    (output/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    requirements='\n'.join(f'{p}=={v}' for p,v in environment['packages'].items())+'\n'
    (output/'requirements-pinned.txt').write_text(requirements)
    paths=[]
    for folder in ['hypocotyl/data','hypocotyl/code','hypocotyl/reports']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths += [ROOT/p for p in ['hypocotyl/README.md','hypocotyl/DATA_DICTIONARY.md',
        'hypocotyl/reports/model_rationale.md','hypocotyl/reports/model_verification.json',
        'wheat/code/model.py','requirements.txt','hypocotyl/release/environment.json',
        'hypocotyl/release/requirements-pinned.txt']]
    paths=sorted(set(paths))
    manifest=dict(status='prepared_for_deposit_not_published',data_license='Not yet assigned',
        public_url=None,doi=None,
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    paths += [output/'manifest.json']
    archive=output/'hypocotyl_data_code_20260909.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            item=zipfile.ZipInfo(str(p.relative_to(ROOT)),date_time=(2026,9,9,0,0,0))
            item.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(item,p.read_bytes())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path,digest in manifest['files'].items():
            assert hashlib.sha256(z.read(path)).hexdigest()==digest
    (output/'SHA256SUMS').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
    print(f'Prepared and verified {archive}: {len(paths)} files; no public deposit performed.')


if __name__=='__main__':main()
