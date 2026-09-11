"""Export the submitted comparison tables from the recorded metrics."""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(ROOT / "paper/current_results.csv")
    hyp = pd.read_csv(ROOT / "hypocotyl/reports/light_input_20260910/comparison.csv")
    hyp = hyp[hyp.protocol.eq("replicate") & hyp.model.isin([
        "phytoode_light", "latent_ode", "logistic_pinn", "lstm", "rf", "logistic"])].copy()
    hyp["dataset"] = "hypocotyl"
    hyp["model"] = hyp.model.replace(dict(phytoode_light="PhytoODE", latent_ode="Latent ODE (no physics)",
        logistic_pinn="Logistic-PINN", lstm="LSTM-NN", rf="RF", logistic="Logi-ODE"))
    columns = ["dataset", "model", "split", "rmse_mean", "rmse_sd", "relative_error_mean", "relative_error_sd", "n_seeds"]
    frame = pd.concat([frame[columns], hyp[columns]], ignore_index=True)
    frame.to_csv(args.output / "comparison.csv", index=False)
    markdown = ["# Submitted results", "", "Each cell is RMSE / relative RMSE (%). Bold marks the lowest unrounded mean in each partition.", ""]
    for dataset in ["wheat", "maize", "arabidopsis", "hypocotyl"]:
        part = frame[frame.dataset.eq(dataset)]
        decimals = {"wheat": 5, "maize": 2, "arabidopsis": 3, "hypocotyl": 3}[dataset]
        markdown += [f"## {dataset}", "", "| Model | Train | Validation | Test |", "|---|---:|---:|---:|"]
        units = {"wheat": "m", "maize": "relative UAV-height units", "arabidopsis": "cm", "hypocotyl": "mm"}
        latex = [r"\begin{table}[t]", r"\centering", r"\small",
                 r"\caption{" + dataset.capitalize() + " prediction errors: RMSE (" + units[dataset] + r") / relative RMSE (\%).}",
                 r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lccc}", r"\toprule",
                 r"Model & Train & Validation & Test \\", r"\midrule"]
        order = ["Logi-ODE", "Temp-ODE", "RF", "LSTM-NN", "Logi-PINN", "Logistic-PINN", "Latent ODE (no physics)", "PhytoODE"]
        for model in [name for name in order if name in set(part.model)]:
            md_cells, tex_cells = [], []
            for split in ["train", "val", "test"]:
                rows = part[part.split.eq(split)]
                row = rows[rows.model.eq(model)].iloc[0]
                rmse, rel = f"{row.rmse_mean:.{decimals}f}", f"{row.relative_error_mean:.2f}"
                if row.n_seeds > 1:
                    rmse += f" ± {row.rmse_sd:.{decimals}f}"
                    rel += f" ± {row.relative_error_sd:.2f}"
                cell = rmse + " / " + rel
                tex = cell.replace(" ± ", r"\pm").replace(" / ", r"\,/\,")
                if row.rmse_mean == rows.rmse_mean.min():
                    cell, tex = "**" + cell + "**", r"\mathbf{" + tex + "}"
                md_cells.append(cell)
                tex_cells.append("$" + tex + "$")
            markdown.append("| " + model + " | " + " | ".join(md_cells) + " |")
            latex.append(model + " & " + " & ".join(tex_cells) + r" \\")
        latex += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
        (args.output / f"{dataset}.tex").write_text("\n".join(latex))
        markdown.append("")
    (args.output / "comparison.md").write_text("\n".join(markdown))
    print(f"Exported {len(frame)} model/partition rows to {args.output}")


if __name__ == "__main__":
    main()
