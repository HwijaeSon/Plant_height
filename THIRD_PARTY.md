# Sources, attribution, and scope of distribution

This submission contains the authors' model implementation, analysis scripts,
selected experiment outputs, and original hypocotyl dataset. It does not assign
a blanket licence to third-party work or grant additional reuse rights.
The original authors and data providers retain their respective rights.

## Published data

- Wheat inputs are downloaded from the pinned
  [Shao et al. companion repository](https://github.com/YingjieShao/PINN_for_plant_height_forecasting/tree/3da92f51f42d3fde5e06f6fc8ce8f233490a389c).
  They derive from the ETH field phenotyping experiments cited in the manuscript.
  The pinned repository has no top-level licence file. The submission includes
  source links and a downloader, rather than redistributing its input CSVs.
- Maize supplementary data are associated with
  [Sweet et al.](https://doi.org/10.1111/tpj.17092), and the related data deposit is
  [DRUM](https://doi.org/10.13020/SKJN-QX31). The necessary station data are retrieved
  from the pinned
  [HirschLabUMN repository](https://github.com/HirschLabUMN/WiDiv_Drone_Height/tree/8fed4415f99f7ee4f8e43fb088319622bedc81a3).
  This submission does not redistribute that repository's source snapshot.
- Arabidopsis stem-length measurements come from
  [Ebrahimi Naghani et al.](https://doi.org/10.1186/s12870-024-05394-w).
  The supplementary workbook is downloaded from Europe PMC.

Consult the original publications, deposits, and file-level notices for their
terms. The downloads and derived local data are ignored by Git in this branch.

## Reference models and results

Logi-PINN and LSTM reference architectures follow Shao et al.; process models and
baseline adaptations are described in the manuscript. Wheat reference summaries
and compact figure predictions retain source-file attribution in the adjacent
provenance JSON. These data are scientific result records, not a claim of ownership
of the reference authors' work. `paper/references.bib` contains the full research
citations, including foundational Neural ODE and physics-informed learning papers.

## Original hypocotyl data and code

The workbook was supplied by the study authors and is included with their
authorization for this submission. Its original bytes, cell identifiers, missing
metadata, and fixed partitions are preserved. No DOI or additional code/data
licence has been assigned in this snapshot. Public availability of these files
does not relabel upstream material or establish a new third-party licence.
