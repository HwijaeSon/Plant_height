"""Restore the public Sweet et al. (2024) tabular data; no UAV rasters.

Uses Europe PMC's public supplementary-files API, DRUM's public DSpace API,
and a pinned author-repository commit. Existing downloads are checked against
the saved manifest. Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
MANIFEST = ROOT / "data/download_manifest.json"
COMMIT = "8fed4415f99f7ee4f8e43fb088319622bedc81a3"
REPO = "https://raw.githubusercontent.com/HirschLabUMN/WiDiv_Drone_Height/"
DRUM_ITEM = "https://conservancy.umn.edu/server/api/core/items/2ea4b4e4-26a6-4a6a-bc45-f24c10f7402b"
DRUM_FILES = "https://conservancy.umn.edu/server/api/core/bundles/b2fa20e5-7c26-4c80-8cb0-2b3e68e154ad/bitstreams?size=1000"
SUPPLEMENT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11629746/supplementaryFiles"
TABLE_FILES = {1: 11, 2: 5, 3: 9, 4: 3, 5: 13, 6: 14, 7: 6, 8: 10,
               9: 8, 10: 4, 11: 12, 12: 16, 13: 1, 14: 2, 15: 15}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    old = {x["path"]: x for x in previous.get("files", [])}
    records = []

    def record(path: Path, url: str, **extra) -> None:
        content = path.read_bytes()
        rel = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(content).hexdigest()
        if rel in old and digest != old[rel]["sha256"]:
            raise ValueError(f"Checksum changed: {rel}")
        if "expected_size" in extra and len(content) != extra["expected_size"]:
            raise ValueError(f"Incomplete download: {rel}")
        if "drum_md5" in extra and hashlib.md5(content).hexdigest() != extra["drum_md5"]:
            raise ValueError(f"DRUM checksum mismatch: {rel}")
        records.append(dict(path=rel, url=url, bytes=len(content), sha256=digest, **extra))

    def fetch(path: Path, url: str, **extra) -> None:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(url, timeout=60) as response:
                content = response.read()
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(content)
            temporary.replace(path)
        record(path, url, **extra)

    if args.verify_only:
        if not old:
            raise ValueError("No download manifest; run the downloader first")
        for item in old.values():
            record(ROOT / item["path"], item["url"])
        print(f"Verified {len(records)} source files")
        return

    fetch(RAW / "source/drum_item.json", DRUM_ITEM)
    fetch(RAW / "source/drum_bitstreams.json", DRUM_FILES)
    inventory = json.loads((RAW / "source/drum_bitstreams.json").read_text())
    if inventory["page"]["totalPages"] != 1:
        raise ValueError("DRUM inventory now requires pagination")
    for item in inventory["_embedded"]["bitstreams"]:
        if item["sizeBytes"] >= 1_000_000 or item["name"].endswith("_plotboundaries.zip"):
            continue
        checksum = item.get("checkSum", {})
        extra = {"expected_size": item["sizeBytes"], "license": "CC0-1.0"}
        if checksum.get("checkSumAlgorithm", "").upper() == "MD5":
            extra["drum_md5"] = checksum["value"]
        fetch(RAW / "drum" / item["name"], item["_links"]["content"]["href"], **extra)

    archive = RAW / "supplementary/PMC11629746_supplementary.zip"
    fetch(archive, SUPPLEMENT_URL, license="CC-BY-4.0 (article)")
    with ZipFile(archive) as z:
        for name in [f"TPJ-120-1969-s{n:03}.xlsx" for n in TABLE_FILES.values()] + ["TPJ-120-1969-s007.pdf"]:
            path = archive.parent / name
            # Read exact allowlisted members; never extract arbitrary archive paths.
            if not path.exists():
                path.write_bytes(z.read(name))
            record(path, SUPPLEMENT_URL, archive_member=name, license="CC-BY-4.0 (article)")

    paths = ["README.md", "Raw_Data/README.md", "WiDiv_analysis.Rmd"]
    paths += [f"Raw_Data/{y}/{name}_{y}.csv" for y in range(2018, 2022)
              for name in ["Temperature_Data_StPaul", "Precipitation"]]
    paths += [f"Raw_Data/All/{name}.csv" for name in
              ["Plots2GenotypeAllYears", "Our_Widiv_Genotypes", "FloweringTimeHeteroticGroup_Widiv"]]
    for name in paths:
        fetch(RAW / "github" / name, REPO + COMMIT + "/" + name, github_commit=COMMIT)
    fetch(RAW / "source/article.html", "https://pmc.ncbi.nlm.nih.gov/articles/PMC11629746/")
    fetch(RAW / "source/github_tree.json",
          f"https://api.github.com/repos/HirschLabUMN/WiDiv_Drone_Height/git/trees/{COMMIT}?recursive=1")
    MANIFEST.write_text(json.dumps({
        "paper_doi": "10.1111/tpj.17092", "dataset_doi": "10.13020/SKJN-QX31",
        "retrieved_date": "2026-09-04", "github_commit": COMMIT,
        "table_to_pmc_file": {f"S{k}": f"TPJ-120-1969-s{v:03}.xlsx" for k, v in TABLE_FILES.items()},
        "files": records,
    }, indent=2) + "\n")
    print(f"Ready: {len(records)} files, {sum(x['bytes'] for x in records):,} bytes")


if __name__ == "__main__":
    main()
