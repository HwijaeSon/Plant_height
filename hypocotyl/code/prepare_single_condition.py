"""Freeze the 12L12D subset of the existing partitions and split-specific means."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'data/processed/single_condition_20260909'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists(): raise FileExistsError(DEST)
    DEST.mkdir()
    manifest = dict(condition='12L12D', excluded_condition='cR',
        input_features=['genotype', 'elapsed_time'], target_type='split_specific_replicate_means',
        prior_test_inspected=True, original_partitions='Retained without reassignment; filter sheet == 12L12D.',
        source_sha256={}, protocols={})
    for protocol in ['replicate', 'time_holdout']:
        folder=DEST/protocol; folder.mkdir()
        config=dict(protocol=protocol, condition='12L12D', counts={}, mean_target_counts={},
                    hours={}, split_seed=20260909, target_type='split_specific_replicate_means')
        for split in ['train','val','test']:
            source=ROOT/'data/processed/splits'/protocol/f'{split}.csv'
            manifest['source_sha256'][str(source.relative_to(ROOT))]=sha(source)
            frame=pd.read_csv(source); frame=frame[frame.sheet.eq('12L12D')].copy()
            assert frame.observation_id.is_unique and frame.length.notna().all()
            frame.to_csv(folder/f'{split}.csv',index=False)
            means=frame.groupby(['genotype','elapsed_hours'],sort=True).length.agg(
                mean_length_mm='mean',n_observations='size').reset_index()
            means.to_csv(folder/f'{split}_means.csv',index=False)
            config['counts'][split]=len(frame); config['mean_target_counts'][split]=len(means)
            config['hours'][split]=sorted(frame.elapsed_hours.unique().astype(int).tolist())
            if split=='train': config['height_scale_mm']=float(frame.length.max())
        if protocol=='time_holdout':
            assert config['hours']['test']==[36,60]
            assert set(config['hours']['train']).isdisjoint([36,60])
            assert set(config['hours']['val']).isdisjoint([36,60])
        (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        manifest['protocols'][protocol]=config
    manifest['files_sha256']={str(p.relative_to(ROOT)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['protocols'],indent=2))


if __name__=='__main__': main()
