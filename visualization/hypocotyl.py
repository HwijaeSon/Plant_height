"""Regenerate the current individual-plant and missingness figures."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'hypocotyl/code'))
from report_adopted_hypocotyl import export

def make_figure(output_dir):
    export(Path(output_dir)/'hypocotyl_tables',figures=output_dir)
