# Pfam domain annotation correlation (PyMut vs maftools)

Validates PyMut's re-implementation of `maftools::pfamDomains()` (gene
symbol + amino-acid position matched against the same bundled Pfam/CDD
domain table maftools uses), for the PAAD-TB and TCGA-LAML cohorts.

## Usage

Two-step pipeline (step 1 needs R/maftools and cannot be done from Python):

```bash
# Step 1 -- reference output with the actual R/Bioconductor maftools:
Rscript run_maftools_pfam.R
#   -> results/<cohort>_maftools_domainSummary.csv
#   -> results/<cohort>_maftools_proteinSummary.csv

# Step 2 -- pyMut's own annotate_pfam(), merged with step 1:
python correlate_pfam_vs_maftools.py
```

Paths are resolved relative to this script folder: MAFs from
`<project_root>/data/`, all outputs (and the step-1 CSVs) in `results/`
here. Both steps can be overridden with `--base-dir` / `--results-dir`.

## Inputs

| File | Used by |
|------|---------|
| `<project_root>/data/PAAD-TB.final_analysis.maf` | step 2 |
| `<project_root>/data/tcga_laml.maf.gz` | step 2 |
| `results/<cohort>_maftools_domainSummary.csv` | step 2 (from step 1) |

## Outputs (`results/`)

| File | Content |
|------|---------|
| `<cohort>_maftools_{domain,protein}Summary.csv` | Reference maftools tables |
| `<cohort>_pfam_vs_maftools_merged.csv` | Per-domain merged counts, pyMut vs maftools |
| `pfam_vs_maftools_correlation.{png,pdf}` | 4-panel log-log concordance figure |
| `correlation_report.txt` | Pearson/Spearman statistics + domains found by only one tool |

Correlations are computed on log10(1 + count) (Pearson) and on ranks
(Spearman), restricted to domains recovered by both tools; domains found by
only one tool are listed for QC. Minor count differences are expected from
the two tools' different variant de-duplication defaults (see the header
comments of `run_maftools_pfam.R`).
