# TMB equivalence (PyMut vs maftools)

Compares Tumor Mutational Burden estimates produced independently by
maftools (`tmb()`) and PyMut (`calculate_tmb_analysis()`) on TCGA-LAML and
TCGA-PAAD, with three complementary figures:

- **Figure 1** -- per-sample scatter, maftools vs PyMut, with `y = x` line
  and Pearson `r` (numerical equivalence).
- **Figure 2** -- Bland-Altman plots (difference vs mean, 95 % limits of
  agreement).
- **Figure 3** -- cohort summary statistics (mean +/- SD) by tool.

## Usage

```bash
Rscript compare_TMB_maftools_vs_pymut.R
```

Paths are resolved relative to this script folder: figures go to
`results/` here; inputs are read from the project root (see below).

## Inputs

Produced beforehand by the two reference notebooks (not by this script),
expected at the **project root**:

```
<project_root>/
├── results_laml_maftools/
│   ├── TMB_por_muestra.csv     # Tumor_Sample_Barcode, total, total_perMB, total_perMB_log
│   └── TMB_statistics.csv      # Metric, Count, Median, Mean, Min, Max, Q1.25., Q3.75., Std
├── results_tcga_laml_pymut/
│   ├── TMB_analysis.tsv        # same per-sample columns, tab-separated
│   └── TMB_statistics.tsv
├── results_paad_maftools/      # same two files as results_laml_maftools
└── results_tcga_paad_pymut/    # same two files as results_tcga_laml_pymut
```

Column names must match exactly; only samples present in both tools are
used for the paired comparison. When adapting the reference notebooks to a
new cohort, make sure their `results_dir` / `cohort_name` match these
folder names.

## Outputs (`results/`)

`Figure1_TMB_numerical_equivalence.{png,pdf}`,
`Figure2_TMB_differences_bland_altman.{png,pdf}`,
`Figure3_TMB_summary_statistics.{png,pdf}`, plus the same panels as
individual files under `results/individual_panels/`.

Extreme hypermutated outliers (Tukey's rule, `k = 3*IQR` on pooled values)
are excluded from Figures 1-2 only -- reported on the console and noted in
each panel's subtitle; Figure 3 always uses all samples. Both behaviours
are controlled by `EXCLUDE_OUTLIERS_IN_PLOTS` and `OUTLIER_IQR_MULTIPLIER`
at the top of the script.
