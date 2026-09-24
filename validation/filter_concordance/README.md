# Filter concordance (PyMut vs maftools)

Validates that PyMut's filtering methods select exactly the same variant
records as maftools' `subsetMaf()`, and produces the PAAD tissue-expression
filter figure (Figure S4).

## What is compared

- **Filter 1 (shared, both cohorts):** keep chromosomes 1, 2, 3, 17.
- **Filter 2 (gene locus, GRCh37):** DNMT3A `chr2:25,455,845-25,565,459`
  for TCGA-LAML; TP53 `chr17:7,571,720-7,590,868` for TCGA-PAAD.

DNMT3A/TP53 were chosen because they are recurrently mutated, clinically
relevant genes in AML / pancreatic adenocarcinoma, which keeps the region
filter non-trivial in both cohorts.

## Usage

```bash
# Option 1 -- step by step:
python flow_pymut.py            # PyMut filters   -> results/tcga_*_py.maf
Rscript flow_maftools.R         # maftools filters -> results/tcga_*_maftools.maf
python compare_and_plot.py      # compares the 4 cases, writes reports + figure

# Option 2 -- single command (runs the two flows first):
python compare_and_plot.py --run-py --run-r

# PAAD tissue-expression filter figure (Figure S4), independent step:
python paad_tissue_expression_filter_figure.py
```

Paths are resolved relative to this script folder: inputs are read from
`<project_root>/data/`, outputs written to `results/` here.

## Inputs

| File | Used by |
|------|---------|
| `<project_root>/data/tcga_laml.maf.gz` | flows, comparison |
| `<project_root>/data/PAAD-TB.final_analysis.clean.maf` | flows |
| `<project_root>/data/PAAD-TB.final_analysis.maf` | expression filter figure |

## Outputs (`results/`)

| File | Content |
|------|---------|
| `tcga_*_{py,maftools}.maf` | Filtered MAFs (intermediate) |
| `report.json` / `report.md` | Per-case row counts, shared/exclusive keys, Jaccard, PASS/FAIL |
| `diff_<case>.tsv` | Discordant variant keys, only if non-empty |
| `pymut_vs_maftools_filter_concordance.{png,pdf}` | Publication figure (3 panels) |
| `Figure_S4_PAAD_expression_filter.{png,pdf}` | Tissue-expression filter figure |
| `Figure_S4_PAAD_expression_filter_summary.csv` | Counts/percentages/threshold |

`compare_and_plot.py` exits non-zero unless every case passes the Jaccard
threshold (`--jaccard-min`, default 0.995), which makes it easy to chain
after the two flow scripts.

## Fair-comparison notes

`flow_maftools.R` strips `chr` prefixes and forces `vc_nonSyn` to include
every `Variant_Classification` in `@data`, so no variant is silently
routed to maftools' `@maf.silent` slot; PyMut keeps everything in a single
table. `flow_pymut.py` carries a no-op-safe pyarrow compatibility shim for
older pyarrow releases.
