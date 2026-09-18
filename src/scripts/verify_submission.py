"""Verify the immutable submission files against their SHA-256 manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    errors = []
    for name, record in manifest["files"].items():
        path = ROOT / name
        if not path.is_file():
            errors.append(f"Missing: {name}")
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            errors.append(f"Checksum mismatch: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(manifest['files'])} submission files. External downloads and local outputs are not part of this manifest.")


if __name__ == "__main__":
    main()
