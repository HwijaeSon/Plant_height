"""Audit the released supplements and prepare observed maize height targets.

No phenotype interpolation, additional peak filtering, or model fitting.
All-year cohort selection is based on coverage, never height magnitude.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/maize/raw"
# Supplement table number to publisher attachment number.
TABLE_FILES = {1: 11, 2: 5, 3: 9, 4: 3, 5: 13, 6: 14, 7: 6, 8: 10,
               9: 8, 10: 4, 11: 12, 12: 16, 13: 1, 14: 2, 15: 15}

OUT = ROOT / "data/maize/processed"
REPORTS = ROOT / "outputs/maize-data-report"
YEARS = (2018, 2019, 2020, 2021)
PLANTING = {2018: "2018-05-14", 2019: "2019-05-30", 2020: "2020-05-07", 2021: "2021-05-06"}


def read_table(number: int) -> pd.DataFrame:
    path = RAW / f"supplementary/TPJ-120-1969-s{TABLE_FILES[number]:03}.xlsx"
    return pd.read_excel(path, header=1).dropna(axis=1, how="all").dropna(how="all")


def genotype_label(value) -> str:
    if pd.isna(value):
        raise ValueError("Missing genotype label")
    if isinstance(value, (int, float, np.number)):
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def write_csv(data: pd.DataFrame, name: str) -> None:
    data.to_csv(OUT / name, index=False, float_format="%.10g")


def make_weather(last_day: int) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        source = RAW / f"github/Raw_Data/{year}/Temperature_Data_StPaul_{year}.csv"
        d = pd.read_csv(source).rename(columns={"Date": "date", "Max Deg F": "tmax_f", "Min Deg F": "tmin_f"})
        d["date"] = pd.to_datetime(d.date, format="%m/%d/%Y")
        d["year"] = year
        d["dap"] = (d.date - pd.Timestamp(PLANTING[year])).dt.days
        d = d[d.dap.between(0, last_day)].sort_values("dap").copy()
        if d.dap.tolist() != list(range(last_day + 1)) or d.isna().any().any():
            raise ValueError(f"Incomplete daily station temperature for {year}")
        d["tmax_c"] = (d.tmax_f - 32) * 5 / 9
        d["tmin_c"] = (d.tmin_f - 32) * 5 / 9
        d["temperature_c"] = (d.tmax_c + d.tmin_c) / 2
        # The author's R code gates on the uncapped mean, then caps Tmax at
        # 86 F and floors Tmin at 50 F. Preserve that exact daily definition.
        d["daily_gdd_f"] = np.where((d.tmax_f + d.tmin_f) / 2 > 50,
                                    (d.tmax_f.clip(upper=86) + d.tmin_f.clip(lower=50)) / 2 - 50, 0)
        d["daily_gdd_c"] = d.daily_gdd_f * 5 / 9
        d["station_gdd_f"] = d.daily_gdd_f.where(d.dap > 0, 0).cumsum()
        d["station_gdd_c"] = d.station_gdd_f * 5 / 9
        # Consistent with the existing model's linear rate interpolation/RK4:
        # trapezoidal integral is separately named, not called published GDD.
        rates = d.daily_gdd_c.to_numpy()
        d["thermal_integral_c"] = np.r_[0, np.cumsum((rates[:-1] + rates[1:]) / 2)]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    flights = read_table(1).rename(columns={"Year": "year", "Date": "source_excel_date",
                                           "GDD after Planting": "published_gdd_f", "Notes": "notes"})
    # S1 stores every Excel date in 2022; Year supplies the actual trial year.
    flights["date"] = [pd.Timestamp(year=int(y), month=d.month, day=d.day)
                       for y, d in zip(flights.year, flights.source_excel_date)]
    flights["dap"] = [(d - pd.Timestamp(PLANTING[int(y)])).days for y, d in zip(flights.year, flights.date)]
    flights["is_ground_flight"] = flights.notes.fillna("").str.contains("no plant material", case=False)
    if flights.date.duplicated().any():
        raise ValueError("Duplicate flight date")
    weather = make_weather(int(flights.dap.max()))
    flights = flights.merge(weather[["date", "station_gdd_f"]], on="date", validate="one_to_one")
    flights["gdd_difference_f"] = flights.published_gdd_f - flights.station_gdd_f
    write_csv(flights, "flights.csv")
    write_csv(weather, "weather_daily.csv")

    aux = read_table(5).rename(columns={"env": "year", "YYYYMMDD": "date"})
    aux["dap"] = [(d - pd.Timestamp(PLANTING[int(y)])).days for y, d in zip(aux.year, aux.date)]
    if not (aux.dap == aux.daysFromStart - 1).all():
        raise ValueError("Unexpected S5 day numbering")
    write_csv(aux, "weather_envirotype_daily.csv")

    loess = read_table(4).rename(columns={"Plot": "plot_id", "Genotype": "genotype", "Year": "year",
                                         "Rep": "replicate", "Block": "block", "Entry": "entry",
                                         "Stand": "stand", "Range": "field_range", "Row": "field_row"})
    loess["genotype"] = loess.genotype.map(genotype_label)
    meta_cols = ["plot_id", "genotype", "year", "replicate", "block", "entry", "stand", "field_range", "field_row"]
    meta = loess[meta_cols].copy()
    if meta.plot_id.duplicated().any():
        raise ValueError("S4 repeats a physical plot ID")
    keys = ["genotype", "replicate", "year"]
    meta["ambiguous_key"] = meta.duplicated(keys, keep=False)
    height_cols = [c for c in loess.columns if re.fullmatch(r"\d+_GDD", c)]
    meta["has_loess_height"] = loess[height_cols].notna().any(axis=1)
    write_csv(meta, "plot_metadata.csv")
    write_csv(loess, "loess_reference_wide.csv.gz")

    tables = {}
    for number in [2, 3]:
        table = read_table(number).rename(columns={"Genotype": "genotype", "Rep": "replicate"})
        table["genotype"] = table.genotype.map(genotype_label)
        table["source_row"] = np.arange(3, len(table) + 3)
        date_cols = [c for c in table.columns if re.fullmatch(r"\d{8}", str(c))]
        long = table.melt(id_vars=["genotype", "replicate", "source_row"], value_vars=date_cols,
                          var_name="flight_code", value_name="height_raw" if number == 2 else "height_relative")
        value_col = "height_raw" if number == 2 else "height_relative"
        long[value_col] = pd.to_numeric(long[value_col], errors="raise")
        long["date"] = pd.to_datetime(long.flight_code, format="%m%d%Y")
        long = long.merge(flights[["date", "year", "dap", "published_gdd_f", "station_gdd_f", "is_ground_flight"]],
                          on="date", how="left", validate="many_to_one")
        if long.year.isna().any():
            raise ValueError("Height date absent from S1")
        long["trajectory_id"] = long.year.astype(str) + ":S3row" + long.source_row.astype(str)
        tables[number] = long
    raw, clean = tables[2], tables[3]
    join = ["genotype", "replicate", "source_row", "date"]
    clean = clean.merge(raw[join + ["height_raw"]], on=join, validate="one_to_one")
    unique_meta = meta.loc[~meta.ambiguous_key]
    clean = clean.merge(unique_meta.drop(columns="ambiguous_key"), on=keys, how="left", validate="many_to_one")
    mapping_counts = meta.groupby(keys).size().rename("mapping_count").reset_index()
    clean = clean.merge(mapping_counts, on=keys, how="left", validate="many_to_one")
    clean["ambiguous_plot_mapping"] = clean.mapping_count > 1
    clean["unmapped_plot_metadata"] = clean.mapping_count.isna()
    clean.loc[clean.plot_id.notna(), "trajectory_id"] = clean.loc[clean.plot_id.notna(), "plot_id"]
    clean["observed"] = clean.height_relative.notna()
    clean = clean.sort_values(["year", "source_row", "date"]).reset_index(drop=True)
    write_csv(clean, "height_all_slots.csv.gz")

    # Separate exclusions from published missing values; no implicit row drops.
    trajectories = clean.groupby(["trajectory_id", "genotype", "replicate", "year"], sort=True).agg(
        source_row=("source_row", "first"), plot_id=("plot_id", "first"),
        ambiguous_plot_mapping=("ambiguous_plot_mapping", "first"),
        unmapped_plot_metadata=("unmapped_plot_metadata", "first"),
        n_slots=("observed", "size"), n_observed=("observed", "sum"),
    ).reset_index()
    plant_counts = clean.loc[~clean.is_ground_flight].groupby("trajectory_id")["observed"].sum()
    trajectories["n_plant_observed"] = trajectories.trajectory_id.map(plant_counts).astype(int)
    eligible = trajectories.loc[~trajectories.ambiguous_plot_mapping & ~trajectories.unmapped_plot_metadata &
                                (trajectories.n_plant_observed >= 4)]
    coverage = eligible.groupby("genotype").year.nunique()
    common = sorted(coverage[coverage == len(YEARS)].index)
    trajectories["in_common_genotypes"] = trajectories.genotype.isin(common)
    trajectories["exclusion_reason"] = np.select(
        [trajectories.ambiguous_plot_mapping, trajectories.unmapped_plot_metadata, trajectories.n_observed == 0,
         trajectories.n_plant_observed < 4, ~trajectories.in_common_genotypes],
        ["ambiguous_genotype_replicate_to_plot", "genotype_label_absent_from_plot_metadata", "no_published_clean_observations",
         "fewer_than_4_plant_observations", "genotype_not_observed_in_all_4_years"], default="")
    trajectories["selected"] = trajectories.exclusion_reason == ""
    write_csv(trajectories, "trajectory_inventory.csv")
    primary = clean.loc[clean.observed & ~clean.is_ground_flight &
                        clean.trajectory_id.isin(trajectories.loc[trajectories.selected, "trajectory_id"])].copy()
    primary = primary.merge(weather[["date", "temperature_c", "tmin_c", "tmax_c", "daily_gdd_c", "thermal_integral_c"]],
                            on="date", validate="many_to_one")
    primary["genotype_index"] = primary.genotype.map({g: i for i, g in enumerate(common)})
    primary["split"] = primary.year.map({2018: "train", 2019: "train", 2020: "val", 2021: "test"})
    if primary.duplicated(["plot_id", "date"]).any() or primary.height_relative.isna().any():
        raise ValueError("Invalid primary target table")
    if primary.temperature_c.isna().any() or not np.isfinite(primary.height_relative).all():
        raise ValueError("Invalid model observations")
    write_csv(primary, "height_observations.csv")
    mean = primary.groupby(["year", "genotype", "genotype_index", "date", "dap", "split"]).agg(
        height_relative=("height_relative", "mean"), n_replicates=("height_relative", "size"),
        temperature_c=("temperature_c", "first"), station_gdd_f=("station_gdd_f", "first"),
    ).reset_index()
    write_csv(mean, "height_genotype_means.csv")
    pd.DataFrame({"genotype": common, "genotype_index": range(len(common))}).to_csv(OUT / "genotypes.csv", index=False)

    manual = read_table(14).rename(columns={"Plot": "plot_id", "Date": "flight_code"})
    manual["date"] = pd.to_datetime(manual.flight_code.astype(int).astype(str).str.zfill(8), format="%m%d%Y")
    manual = manual.melt(id_vars=["plot_id", "date"], value_vars=[f"Height{i}" for i in range(1, 6)],
                         var_name="plant_position", value_name="height_cm").dropna(subset=["height_cm"])
    manual["height_cm"] = pd.to_numeric(manual.height_cm, errors="raise")
    manual["height_m"] = manual.height_cm / 100
    manual = manual.merge(meta[meta_cols], on="plot_id", how="left", validate="many_to_one")
    write_csv(manual, "manual_height_reference.csv")

    fold_list = [{"name": "chronological", "train_years": [2018, 2019], "val_year": 2020, "test_year": 2021}]
    for test in YEARS:
        for val in YEARS:
            if val == test or (test == 2021 and val == 2020):
                continue
            fold_list.append({"name": f"test{test}_val{val}", "train_years": [y for y in YEARS if y not in (test, val)],
                              "val_year": val, "test_year": test})
    (OUT / "year_splits.json").write_text(json.dumps(fold_list, indent=2) + "\n")

    audit = []
    for year in YEARS:
        f, c, t, p = (x[x.year == year] for x in [flights, clean, trajectories, primary])
        active = c[c.observed]
        meta_y = meta[meta.year == year]
        audit.append({
            "year": year, "paper_flights": {2018: 14, 2019: 27, 2020: 12, 2021: 11}[year],
            "published_flights": len(f), "ground_flights": int(f.is_ground_flight.sum()),
            "s3_plots_with_observations": active.trajectory_id.nunique(), "s3_genotypes": active.genotype.nunique(),
            "s3_observations": len(active), "s3_missing_slots": int((~c.observed).sum()),
            "s4_plots_with_loess": int(meta_y.has_loess_height.sum()),
            "primary_genotypes": p.genotype.nunique(), "primary_plots": p.plot_id.nunique(),
            "primary_observations": len(p), "primary_mean_observations": int((mean.year == year).sum()),
            "primary_flights": p.date.nunique(), "primary_min_observations_per_plot": int(p.groupby("plot_id").size().min()),
            "primary_max_observations_per_plot": int(p.groupby("plot_id").size().max()),
            "first_dap": int(p.dap.min()), "last_dap": int(p.dap.max()),
            "height_min": float(p.height_relative.min()), "height_max": float(p.height_relative.max()),
        })
    audit = pd.DataFrame(audit)
    audit.to_csv(REPORTS / "yearly_counts.csv", index=False)
    trajectories.groupby(["year", "exclusion_reason"], dropna=False).size().rename("n_trajectories").to_csv(REPORTS / "exclusions.csv")
    flights.loc[flights.gdd_difference_f.abs() > 0.01].to_csv(REPORTS / "gdd_discrepancies.csv", index=False)
    summary = {
        "n_genotypes_primary": len(common), "n_plots_primary": primary.plot_id.nunique(),
        "n_observations_primary": len(primary), "n_genotype_year_curves": len(common) * 4,
        "n_published_s3_observations": int(clean.observed.sum()),
        "n_published_s3_active_trajectories": int((trajectories.n_observed > 0).sum()),
        "n_published_s4_active_trajectories": int(meta.has_loess_height.sum()),
        "ambiguous_genotypes": sorted(meta.loc[meta.ambiguous_key, "genotype"].unique()),
        "unmapped_genotype_labels": sorted(trajectories.loc[trajectories.unmapped_plot_metadata, "genotype"].unique()),
        "n_ground_observations_removed_before_cohort_filter": int((clean.observed & clean.is_ground_flight).sum()),
        "day_grid": [0, int(flights.dap.max())], "n_time_grid": int(flights.dap.max()) + 1,
        "n_manual_measurements": len(manual), "manual_plots": manual.plot_id.nunique(),
        "manual_years": sorted(manual.year.dropna().astype(int).unique().tolist()),
        "target_unit": "relative UAV height; not cm or m",
        "primary_protocol": fold_list[0], "yearly": audit.to_dict(orient="records"),
    }
    (REPORTS / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(audit.to_string(index=False))
    print(f"Primary: {len(common)} genotypes, {primary.plot_id.nunique()} plots, {len(primary)} observations")


if __name__ == "__main__":
    main()
