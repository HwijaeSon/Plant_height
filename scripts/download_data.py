"""Download the exact external inputs used by the manuscript."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "wheat", "maize", "arabidopsis"), default="all")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT, help="Destination repository root")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/data_sources.json").read_text())
    records = {entry["path"]: entry for group in manifest.values() for entry in group}
    verified = set()

    def restore(name):
        if name in verified:
            return
        entry = records[name]
        target = args.root / name
        if not target.exists():
            if args.verify_only:
                raise FileNotFoundError(f"Missing input: {target}. Run without --verify-only.")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            if "archive" in entry:
                restore(entry["archive"])
                with ZipFile(args.root / entry["archive"]) as archive:
                    temporary.write_bytes(archive.read(entry["member"]))
            else:
                for attempt in range(3):
                    try:
                        request = Request(entry["url"], headers={"User-Agent": "PhytoODE-reproducibility/1.0"})
                        with urlopen(request, timeout=45) as response, temporary.open("wb") as out:
                            while chunk := response.read(1024 * 1024):
                                out.write(chunk)
                        break
                    except OSError:
                        temporary.unlink(missing_ok=True)
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
            if entry.get("sha256") and digest(temporary) != entry["sha256"]:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Downloaded bytes differ from the paper input: {name}")
            temporary.replace(target)
        if entry.get("sha256") and digest(target) != entry["sha256"]:
            raise ValueError(f"Input checksum mismatch: {target}; existing file was not overwritten")
        verified.add(name)
        print(f"{'Verified' if entry.get('sha256') else 'Archive available'} {name}", flush=True)

    groups = list(manifest) if args.dataset == "all" else [args.dataset]
    for group in groups:
        for record in manifest[group]:
            restore(record["path"])
    hashed = sum(bool(records[name].get("sha256")) for name in verified)
    print(f"Ready: {hashed} hash-verified data files and {len(verified)-hashed} archive containers. "
          f"Next: python scripts/prepare_data.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()
