from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parents[1] / "data/arabidopsis"
DEFAULT_INPUT = EXPERIMENT_DIR / "raw/12870_2024_5394_MOESM5_ESM.xlsx"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "processed/stem_length_long.csv"

SHEETS = {
    "stem length_nAT": ("nAT", 21.0, 18.0),
    "stem length_hAT": ("hAT", 28.0, 24.0),
}


def load_sheet(path: Path, sheet: str, condition: str, day_temp: float, night_temp: float) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet)
    frame = frame.rename(
        columns={
            "Plant ID": "genotype",
            "plant #": "plant_number",
            "time point": "day_after_sowing",
            "length": "length_cm",
        }
    )
    required = ["genotype", "plant_number", "day_after_sowing", "length_cm"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{sheet} is missing columns: {missing}")
    frame = frame[required].copy()
    for column in ["plant_number", "day_after_sowing", "length_cm"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required)
    frame["plant_number"] = frame["plant_number"].astype(int)
    frame["day_after_sowing"] = frame["day_after_sowing"].astype(int)
    frame["condition"] = condition
    frame["day_temperature_c"] = day_temp
    frame["night_temperature_c"] = night_temp
    frame["temperature_c"] = (day_temp + night_temp) / 2.0
    frame["length_m"] = frame["length_cm"] / 100.0
    return frame


def prepare(input_path: Path, output_path: Path) -> pd.DataFrame:
    frames = [
        load_sheet(input_path, sheet, condition, day_temp, night_temp)
        for sheet, (condition, day_temp, night_temp) in SHEETS.items()
    ]
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(
        ["condition", "genotype", "plant_number", "day_after_sowing"]
    ).reset_index(drop=True)

    duplicates = data.duplicated(
        ["condition", "genotype", "plant_number", "day_after_sowing"]
    )
    if duplicates.any():
        raise ValueError(f"Found {int(duplicates.sum())} duplicate phenotype rows")
    if len(data) != 720:
        raise ValueError(f"Expected 720 measurements, found {len(data)}")
    if data["genotype"].nunique() != 9:
        raise ValueError("Expected nine genotypes")
    counts = data.groupby(["condition", "genotype", "plant_number"]).size()
    if set(counts.unique()) != {4} or len(counts) != 180:
        raise ValueError("Each of 180 individual trajectories must have four measurements")
    if (data["length_m"] < 0).any():
        raise ValueError("Negative stem lengths are not expected")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = prepare(args.input, args.output)
    print(
        f"saved {len(data)} rows -> {args.output}\n"
        f"genotypes={data.genotype.nunique()} conditions={data.condition.nunique()} "
        f"individuals={data.groupby(['condition','genotype','plant_number']).ngroups}"
    )


if __name__ == "__main__":
    main()
