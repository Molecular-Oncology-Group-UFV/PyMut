# PyMut Validation & Benchmark Scripts

Reproducible validation workflows comparing **PyMut** (pure-Python mutation
analysis toolkit) against **maftools** (R/Bioconductor reference
implementation) on two TCGA cohorts (TCGA-LAML and TCGA-PAAD), GRCh37/hg19.
Each folder validates one analytical feature independently.

> These scripts live **outside** the PyMut package itself: they are the
> empirical evidence that PyMut reproduces the reference tool's results.

## Structure

| Folder | Validates | Scripts |
|--------|-----------|---------|
| [`filter_concordance/`](filter_concordance/) | Variant filtering (chromosome + gene locus) against `maftools::subsetMaf`, plus the PAAD tissue-expression filter figure | `flow_pymut.py`, `flow_maftools.R`, `compare_and_plot.py`, `paad_tissue_expression_filter_figure.py` |
| [`pfam_domain_correlation/`](pfam_domain_correlation/) | Pfam domain annotation against `maftools::pfamDomains()` | `run_maftools_pfam.R`, `correlate_pfam_vs_maftools.py` |
| [`tmb_equivalence/`](tmb_equivalence/) | Tumor Mutational Burden (per-sample and cohort level) | `compare_TMB_maftools_vs_pymut.R` |
| [`roundtrip/`](roundtrip/) | MAF/VCF read -> export -> re-read preservation | `multi_source_roundtrip_test.py` |

Every pipeline writes its outputs to a `results/` folder **inside its own
directory** (reports, figures and tables).

## Quick start

Run each pipeline individually following the README in its folder.

## Conventions

- **Paths are resolved relative to each script's own location** (not the
  working directory): raw inputs are read from `<project_root>/data/` and
  outputs are written to `<pipeline>/results/`. Scripts can be launched
  from anywhere.
- Inputs for the TMB pipeline (`results_<cohort>_<tool>/` folders at the
  project root) are produced beforehand by the two reference notebooks:
  `PyMut_notebook.ipynb` (PyMut side) and `maftools_analysis.Rmd`
  (maftools side).

## Results

Key numbers obtained on TCGA-LAML / TCGA-PAAD (fill in after running):

| Check | Result |
|-------|--------|
| Filter concordance, Jaccard (4 cases) | ... |
| Pfam variants/domain, Pearson r (log10) | ... |
| Pfam genes/domain, Pearson r (log10) | ... |
| TMB per-sample, Pearson r | ... |

The full machine-readable reports are kept in each `results/` folder
(`report.json` / `report.md`, `correlation_report.txt`, and the
`Figure_S4_*_summary.csv`).
