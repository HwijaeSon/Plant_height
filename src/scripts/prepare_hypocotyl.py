"""Verify approved-cell extraction and the longitudinal prefix forecast partitions."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/hypocotyl'))
from forecast_data import prepare,PREPARED

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,help='Re-extract into a new directory; default verifies bundled data')
    args=p.parse_args()
    if args.output and args.output.exists():p.error('Output already exists')
    with TemporaryDirectory() as temporary:
        output=args.output or Path(temporary)
        config=prepare(output=output)
        for path in output.iterdir():
            if path.read_bytes()!=(PREPARED/path.name).read_bytes():raise ValueError(f'Extraction differs: {path.name}')
    print(f"Verified {config['source_measurements']} released cells; {config['n_plants']} plants and {config['retained_measurements']} retained observations. No averaging or imputation.")

if __name__=='__main__':main()
