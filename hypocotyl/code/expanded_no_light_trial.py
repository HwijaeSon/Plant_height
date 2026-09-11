"""Run the unchanged no-light trainer with the expanded coefficient profile."""
from pathlib import Path
import no_light_prefix_trial as trial

trial.PROFILE = Path(__file__).resolve().parents[2]/'configs/hypocotyl_expanded_no_light_lambda_20260911.json'

if __name__ == '__main__':
    trial.main()
