"""Export the manuscript's retained no-light lambda=100 model and six baselines."""
import argparse
from pathlib import Path
import pandas as pd
import report_forecast as baseline

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'hypocotyl/reports/adopted_lambda100_20260911'
SOURCE=ROOT/'hypocotyl/reports/prefix_parameters_no_light_20260911/comparison.csv'
ORDER=['logistic','rf','lstm','logistic_pinn','latent_ode','latent_ode_light','phytoode']
LABELS=dict(logistic='Logistic ODE',rf='Random forest',lstm='LSTM-NN',logistic_pinn='Logistic-PINN',
            latent_ode='Latent ODE (no light)',latent_ode_light='Latent ODE (+ light)',phytoode='PhytoODE')


def directory(drop,model,seed):
    if model=='phytoode':
        return ROOT/f'hypocotyl/results/prefix_parameters_no_light_20260911/evaluated/lambda_100/drop_{round(100*drop)}/seed{seed}'
    return ROOT/f'hypocotyl/results/prefix_forecast_20260911/drop_{round(100*drop)}/{model}/seed{seed}'


def export(output=REPORT,figures=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(SOURCE)
    frame=frame[frame.model.isin(ORDER)].copy();frame['label']=frame.model.map(LABELS)
    assert len(frame)==63
    frame.to_csv(output/'comparison.csv',index=False)
    md=['# Retained manuscript configuration: no-light PhytoODE, lambda_ODE=100, lambda_K=0','',
        'The individual r/K head receives genotype, masked prefix lengths and masks. '
        'The six baseline results are unchanged. The original ten-coefficient validation search selected 100; '
        'it was retained after inspection of a later expanded search and its test results. '
        'That expanded search selected 10000 on validation. The manuscript configuration is therefore '
        'not the expanded validation optimum, and its retention does not constitute independent test confirmation.','',
        'Cells: mean per-plant RMSE (mm) / relative RMSE (%), mean ± sample SD. '
        'Train: 0–36 h; validation: 48 h; test: 60/72 h. Bold marks the smallest unrounded mean in each column.','']
    for drop in [0.,.25,.5]:
        part=frame[frame.additional_prefix_drop.eq(drop)]
        md += [f'## Additional removal of available prefix measurements: {drop:.0%}','','| Model | Train | Validation | Test |','|---|---:|---:|---:|']
        for model in ORDER:
            cells=[]
            for split in ['train','val','test']:
                sub=part[part.split.eq(split)];r=sub[sub.model.eq(model)].iloc[0]
                cell=f'{r.rmse_mean:.3f} / {r.relative_error_mean:.2f}%'
                if r.n_seeds>1:cell=f'{r.rmse_mean:.3f} ± {r.rmse_sd:.3f} / {r.relative_error_mean:.2f} ± {r.relative_error_sd:.2f}%'
                if r.rmse_mean==sub.rmse_mean.min():cell='**'+cell+'**'
                cells.append(cell)
            md.append('| '+LABELS[model]+' | '+' | '.join(cells)+' |')
        md.append('')
    (output/'comparison.md').write_text('\n'.join(md).rstrip()+'\n')
    if figures is not None:
        baseline.ORDER=ORDER;baseline.LABELS=LABELS;baseline.REPORT=output
        baseline.directory=directory
        baseline.make_figures(Path(figures))
    return frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=REPORT);p.add_argument('--figures',type=Path)
    args=p.parse_args();frame=export(args.output,args.figures)
    print(f'Exported {len(frame)} retained manuscript rows')


if __name__=='__main__':main()
