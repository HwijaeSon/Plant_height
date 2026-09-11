"""Regenerate the current individual-plant and missingness figures."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'hypocotyl/code'))
from report_forecast import make_figures

def make_figure(output_dir):make_figures(output_dir)
