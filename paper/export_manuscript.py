"""Export the edited LaTeX sources without regenerating manuscript prose."""
from pathlib import Path
import argparse
from datetime import date
import hashlib
import json
import re
import zipfile


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    args = parser.parse_args()
    if not re.fullmatch(r"\d{8}", args.date):
        parser.error("--date must have the form YYYYMMDD")

    sources = [PAPER / "main.tex"]

    def expand(match):
        path = PAPER / (match[1] + ".tex")
        sources.append(path)
        return path.read_text()

    flat = re.sub(r"\\input\{([^}]+)\}", expand, sources[0].read_text())
    (PAPER / "main_standalone.tex").write_text(flat)
    figures = sorted(set(re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", flat)))
    archive = PAPER / f"overleaf_{args.date}.zip"
    members = {"main.tex": flat.encode(),
               "references.bib": (PAPER / "references.bib").read_bytes()}
    members.update({name: (PAPER / name).read_bytes() for name in figures})
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in members.items():
            # Stable metadata keeps repeated exports identical.
            item = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(item, content)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert all(bundle.read(name) == value for name, value in members.items())

    # Retain experiment provenance while recording the current editorial sources.
    path = PAPER / "hypocotyl_manuscript_provenance_20260910.json"
    provenance = json.loads(path.read_text())
    for source in sources:
        key = str(source.relative_to(ROOT))
        if key in provenance["sources_sha256"]:
            provenance["sources_sha256"][key] = sha(source)
    provenance["manuscript_sources_sha256"] = {
        str(source.relative_to(ROOT)): sha(source) for source in sources
    }
    provenance["overleaf_sha256"] = sha(archive)
    provenance["editorial_exporter"] = str(Path(__file__).relative_to(ROOT))
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported edited manuscript: {archive}")


if __name__ == "__main__":
    main()
