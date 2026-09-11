"""Regenerate the four manuscript figures without rewriting manuscript prose."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    temperature = load("submission_temperature_figures", ROOT / "paper/make_figures.py")
    temperature.OUT = args.output.resolve()
    temperature.wheat_examples()
    temperature.arabidopsis_examples()
    temperature.maize_examples()
    hypocotyl = load("submission_hypocotyl_figure", ROOT / "paper/hypocotyl_figure.py")
    # The existing helper writes PAPER/figures. Point only that output root at
    # the requested directory; data and prediction roots remain unchanged.
    hypocotyl.make_figure(output_dir=args.output.resolve())
    print(f"Saved the four manuscript figures to {args.output}")


if __name__ == "__main__":
    main()
