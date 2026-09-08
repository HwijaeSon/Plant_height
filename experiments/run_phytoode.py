"""Train one seed with the currently adopted PhytoODE loss coefficients."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/phytoode_config.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("wheat", "maize", "arabidopsis"))
    parser.add_argument("--seed", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show the command without training")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())["datasets"][args.dataset]
    if args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    for key in ("lambda_ode", "lambda_k"):
        if not math.isfinite(config[key]) or config[key] < 0:
            parser.error(f"Invalid {key} in {CONFIG}")
    command = [sys.executable, str(ROOT / "experiments/joint_lambda_trial.py"), "train",
               "--dataset", args.dataset, "--seed", str(args.seed),
               "--lambda-ode", str(config["lambda_ode"]),
               "--lambda-k", str(config["lambda_k"]), "--output", str(args.output.resolve())]
    if args.resume:
        command.append("--resume")
    print(f"CUDA_VISIBLE_DEVICES={args.gpu} {shlex.join(command)}", flush=True)
    if not args.dry_run:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
