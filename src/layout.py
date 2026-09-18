"""Flat file names for the immutable manuscript artifacts."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def artifact(dataset, model, seed, kind, drop=0.):
    stem = f"{model}_seed{seed}"
    if dataset == "hypocotyl":
        stem += f"_drop{round(100*drop)}"
    if kind in ("checkpoint.pt", "forest.joblib", "parameters.json"):
        return ROOT / "checkpoints" / dataset / (stem + Path(kind).suffix)
    return ROOT / "results" / dataset / f"{stem}_{kind}"
