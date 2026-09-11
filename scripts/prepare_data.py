"""Prepare the external datasets or verify the bundled hypocotyl targets."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate_wheat():
    # Use the exact manuscript configuration, including its year overrides.
    # The internal data.py diagnostic has different default validation years.
    import torch
    sys.path.insert(0, str(ROOT / "experiments"))
    from physics_ablation import setup
    _, _, _, config, raw, _ = setup("wheat", torch.device("cpu"))
    for split, batch in raw.items():
        years = sorted(set(int(year) for year in batch["year"]))
        print(f"wheat {split}: years={years}, trajectories={len(batch['y'])}, "
              f"grid_points={config['n_times']}, scored_targets={int(batch['mask'].sum())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "wheat", "maize", "arabidopsis", "hypocotyl"), default="all")
    args = parser.parse_args()
    commands = {
        "wheat": [],
        "maize": [["maize/code/prepare_data.py"], ["maize/code/data.py", "--split", "chronological"]],
        "arabidopsis": [["arabidopsis/code/prepare_data.py"]],
        "hypocotyl": [["scripts/prepare_hypocotyl.py"]],
    }
    for name in (list(commands) if args.dataset == "all" else [args.dataset]):
        if name == "wheat":
            validate_wheat()
            continue
        for command in commands[name]:
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
