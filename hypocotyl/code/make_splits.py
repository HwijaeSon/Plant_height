"""Freeze conservative replicate partitions and an unseen-time interpolation test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260909


def main():
    out = ROOT / "data/processed/splits"
    if out.exists():
        raise FileExistsError("Frozen splits already exist; do not overwrite them")
    frame = pd.read_csv(ROOT / "data/processed/observations.csv")
    rng = np.random.default_rng(SEED)
    assignments = []
    for genotype, group in frame.groupby("genotype", sort=True):
        rows = np.sort(group.source_row.unique())
        for attempt in range(10000):
            shuffled = rng.permutation(rows)
            ntrain, nval = int(.6 * len(rows)), int(.2 * len(rows))
            mapping = dict(zip(shuffled, ["train"] * ntrain + ["val"] * nval
                               + ["test"] * (len(rows) - ntrain - nval)))
            trial = group.assign(split=group.source_row.map(mapping))
            counts = trial.groupby(["sheet", "elapsed_hours", "split"]).size()
            if len(counts) == 2 * 7 * 3:
                break
        else:
            raise RuntimeError(f"No viable partition for {genotype}")
        assignments.extend(dict(genotype=genotype, source_row=int(row), split=mapping[row]) for row in rows)
    assignment = pd.DataFrame(assignments)
    frame = frame.merge(assignment, on=["genotype", "source_row"], validate="many_to_one")
    out.mkdir(parents=True)
    assignment.to_csv(out / "row_assignments.csv", index=False)
    protocols = {}
    for protocol in ("replicate", "time_holdout"):
        folder = out / protocol
        folder.mkdir()
        if protocol == "replicate":
            selected = frame.copy()
        else:
            # No phenotype at 36 or 60 h enters training or validation.
            selected = frame[(frame.split != "test") | frame.elapsed_hours.isin([36, 60])].copy()
            selected.loc[selected.elapsed_hours.isin([36, 60]), "split"] = "test"
        counts = {}
        for split in ("train", "val", "test"):
            part = selected[selected.split == split]
            assert len(part.groupby(["sheet", "genotype", "elapsed_hours"])) == (
                70 if protocol == "replicate" else (20 if split == "test" else 50))
            part.to_csv(folder / f"{split}.csv", index=False)
            counts[split] = len(part)
        training = selected[selected.split == "train"]
        scale = float(training.length.max())
        config = dict(protocol=protocol, split_seed=SEED, counts=counts, height_scale_mm=scale,
            train_hours=[0, 12, 24, 36, 48, 60, 72] if protocol == "replicate" else [0, 12, 24, 48, 72],
            test_hours=[0, 12, 24, 36, 48, 60, 72] if protocol == "replicate" else [36, 60],
            grouping="Same genotype/source-row bookkeeping groups remain together across all times and sheets in the replicate protocol. Row groups are not verified plant IDs.",
            scoring="Mean per-genotype/condition curve RMSE against split-specific replicate means; rRMSE divides by the pooled mean of those scored group-mean targets. Individual-observation RMSE is secondary.")
        (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        protocols[protocol] = config
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(status="frozen_before_training", split_seed=SEED, protocols=protocols,
        source_sha256=sha(ROOT / "data/processed/observations.csv"),
        files_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        note="Partition retries check only group coverage, never height values.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(protocols, indent=2))


if __name__ == "__main__":
    main()
