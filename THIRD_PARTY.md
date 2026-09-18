# Data and reference-model sources

The datasets and reference models used in this study are listed below.
The original authors and data providers retain their respective rights.
Consult the source publications, repositories and file-level notices for reuse terms.

## Published data

- Wheat inputs are downloaded from the pinned
  [Shao et al. companion repository](https://github.com/YingjieShao/PINN_for_plant_height_forecasting/tree/3da92f51f42d3fde5e06f6fc8ce8f233490a389c).
  They derive from the ETH field phenotyping experiments cited in the manuscript.
  The pinned repository has no top-level licence file. Source links
  and a downloader are provided for its input CSVs.
- Maize supplementary data are associated with
  [Sweet et al.](https://doi.org/10.1111/tpj.17092), and the related data deposit is
  [DRUM](https://doi.org/10.13020/SKJN-QX31). The necessary station data are retrieved
  from the pinned
  [HirschLabUMN repository](https://github.com/HirschLabUMN/WiDiv_Drone_Height/tree/8fed4415f99f7ee4f8e43fb088319622bedc81a3).
- Arabidopsis stem-length measurements come from
  [Ebrahimi Naghani et al.](https://doi.org/10.1186/s12870-024-05394-w).
  The supplementary workbook is downloaded from Europe PMC.

The download script retrieves these external inputs and verifies their file
checksums. Preparation scripts derive the model inputs locally.

## Reference models and results

Logi-PINN and LSTM reference architectures follow Shao et al.; process models and
baseline adaptations are described in the manuscript. Wheat reference summaries
and figure predictions include source-file attribution in
`results/wheat/reference_test_predictions_seed1_3_provenance.json`.
The accompanying paper provides the full research citations.

## Author-collected hypocotyl data and code

The Arabidopsis hypocotyl measurements were collected by the study authors.
The repository includes the measurement workbook, individual-observation tables,
experimental metadata and forecasting partitions. The
[dataset documentation](data/hypocotyl/README.md) describes plant identifiers,
measurement units, missing entries and collection metadata.

No additional code or dataset licence is specified in this repository.
