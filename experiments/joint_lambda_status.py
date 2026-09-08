"""Compact, read-only progress for the joint coefficient search."""
import argparse
from pathlib import Path
import time

from tune_joint_lambdas import DEFAULT_OUT, BASE, read, pair, job_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output
    protocol = read(out / "protocol.json")
    print(f"Elapsed: {(time.time() - protocol['started_unix']) / 60:.1f} min")
    for dataset in BASE:
        folder = out / dataset
        phase = read(folder / "phase.json") if (folder / "phase.json").exists() else {"phase": "verification"}
        completed = list((folder / "trials").glob("*/*/result.json"))
        print(f"{dataset}: {phase['phase']}; {len(completed)} new full trials complete")
        if phase.get("jobs") and (folder / "validation_records.json").exists():
            available = {job_key(r) for r in read(folder / "validation_records.json")}
            done = sum(tuple(j) in available for j in phase["jobs"])
            print(f"  current phase: {done}/{len(phase['jobs'])} jobs complete (including reused results)")
        if (folder / "selected.json").exists():
            chosen = read(folder / "selected.json")
            for role in ("selected_phyto", "selected_overall"):
                row = chosen[role]
                print(f"  {role}: ODE,K={pair(row)}, validation={row['val_mean']:.8f} +/- {row['val_sd']:.8f}")
        for path in sorted((folder / "trials").glob("*/*/progress.json")):
            if (path.parent / "result.json").exists():
                continue
            progress = read(path)
            print(f"  {path.parent.parent.name}/{path.parent.name}: {progress['epoch']}/{progress['epochs']}")
    if (out / "completed.json").exists():
        print("COMPLETE: " + str(read(out / "completed.json")))


if __name__ == "__main__":
    main()
