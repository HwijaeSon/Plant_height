"""Read-only progress summary; does not inspect any test predictions or scores."""
import argparse
import json
from pathlib import Path
import time

from tune_lambda_ode import DEFAULT_OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output
    protocol = json.loads((out/"protocol.json").read_text())
    print(f"Elapsed: {(time.time()-protocol['started_unix'])/60:.1f} min")
    for dataset in ("wheat", "arabidopsis", "maize"):
        folder = out/dataset
        phase = json.loads((folder/"phase.json").read_text())["phase"]
        completed = list((folder/"trials").glob("*/*/result.json"))
        active = []
        for path in (folder/"trials").glob("*/*/progress.json"):
            if not (path.parent/"result.json").exists():
                progress = json.loads(path.read_text())
                active.append(f"{path.parent.parent.name}/{path.parent.name} {progress['epoch']}/{progress['epochs']}")
        print(f"{dataset}: {len(completed)} new trials complete; {phase}; active: {', '.join(active) or 'none'}")
        selected = folder/"selected.json"
        if selected.exists():
            record = json.loads(selected.read_text())
            print(f"  Frozen positive λ={record['selected_positive']['lambda_ode']:g}; "
                  f"overall validation winner λ={record['selected_overall']['lambda_ode']:g}")


if __name__ == "__main__":
    main()
