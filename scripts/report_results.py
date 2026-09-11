"""Export recorded comparison metrics as CSV and Markdown."""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(ROOT / "results/comparison_temperature.csv")
    hyp = pd.read_csv(ROOT / "hypocotyl/reports/prefix_forecast_20260911/comparison.csv")
    hyp[hyp.additional_prefix_drop.ne(0)].to_csv(args.output / "hypocotyl_missingness.csv", index=False)
    hyp = hyp[hyp.additional_prefix_drop.eq(0)].copy()
    hyp["model"] = hyp.model.replace(dict(phytoode="PhytoODE", latent_ode="Latent ODE (no light)",
        latent_ode_light="Latent ODE (+ light)", logistic_pinn="Logistic-PINN", lstm="LSTM-NN", rf="RF", logistic="Logi-ODE"))
    columns = ["dataset", "model", "split", "rmse_mean", "rmse_sd", "relative_error_mean", "relative_error_sd", "n_seeds"]
    frame = pd.concat([frame[columns], hyp[columns]], ignore_index=True)
    frame.to_csv(args.output / "comparison.csv", index=False)
    markdown = ["# Submitted results", "", "Each cell is RMSE / relative RMSE (%). Bold marks the lowest unrounded mean in each partition.", ""]
    for dataset in ["wheat", "maize", "arabidopsis", "hypocotyl"]:
        part = frame[frame.dataset.eq(dataset)]
        decimals = {"wheat": 5, "maize": 2, "arabidopsis": 3, "hypocotyl": 3}[dataset]
        markdown += [f"## {dataset}", "", "| Model | Train | Validation | Test |", "|---|---:|---:|---:|"]
        order = ["Logi-ODE", "Temp-ODE", "RF", "LSTM-NN", "Logi-PINN", "Logistic-PINN", "Latent ODE (no physics)", "Latent ODE (no light)", "Latent ODE (+ light)", "PhytoODE"]
        for model in [name for name in order if name in set(part.model)]:
            md_cells = []
            for split in ["train", "val", "test"]:
                rows = part[part.split.eq(split)]
                row = rows[rows.model.eq(model)].iloc[0]
                rmse, rel = f"{row.rmse_mean:.{decimals}f}", f"{row.relative_error_mean:.2f}"
                if row.n_seeds > 1:
                    rmse += f" ± {row.rmse_sd:.{decimals}f}"
                    rel += f" ± {row.relative_error_sd:.2f}"
                cell = rmse + " / " + rel
                if row.rmse_mean == rows.rmse_mean.min():
                    cell = "**" + cell + "**"
                md_cells.append(cell)
            markdown.append("| " + model + " | " + " | ".join(md_cells) + " |")
        markdown.append("")
    (args.output / "comparison.md").write_text("\n".join(markdown))
    print(f"Exported {len(frame)} model/partition rows to {args.output}")


if __name__ == "__main__":
    main()
