#!/usr/bin/env Rscript
#
# run_maftools_pfam.R
# ---------------------------------------------------------------------------
# Runs the reference (R) maftools::pfamDomains() analysis on the two cohorts
# under study and exports the resulting domain-level summary tables as CSV.
#
# This is step 1 of a two-step pipeline:
#
#   1. run_maftools_pfam.R              (this script, run with Rscript)
#        -> writes <cohort>_maftools_domainSummary.csv
#        -> writes <cohort>_maftools_proteinSummary.csv
#
#   2. correlate_pfam_vs_maftools.py    (companion Python script)
#        -> runs pyMut's own Pfam annotation on the same MAF files
#        -> merges it with the CSVs produced here
#        -> computes correlation statistics and draws the comparison figures
#
# The two steps are split into separate scripts/languages on purpose: pyMut
# is a pure-Python package (no R runtime dependency), while the ground-truth
# reference implementation this validation is checked against is the actual
# R/Bioconductor "maftools" package. Keeping them decoupled means this
# validation works regardless of whether rpy2 (or any other R<->Python
# bridge) is available in the analyst's environment - only a working R
# installation with maftools is required to run this file.
#
# Paths (resolved relative to this script, regardless of the working dir):
#   - Input MAFs:   <project_root>/data/
#   - Output CSVs:  validation/pfam_domain_correlation/results/
#
# Requirements
# ------------
#   - R (>= 4.0) with the Bioconductor package "maftools" installed:
#       if (!requireNamespace("BiocManager", quietly = TRUE))
#           install.packages("BiocManager")
#       BiocManager::install("maftools")
#
# Usage
# -----
#   Rscript run_maftools_pfam.R
#
# All paths are configured in the "Configuration" section below. Edit them
# if your cohort files live somewhere else.
#
# Notes on comparability with pyMut
# ----------------------------------
# pyMut's default `annotate_pfam()` strategy (gene symbol + amino acid
# position, i.e. no UniProt/VEP annotation needed) is a Python re-
# implementation of maftools::pfamDomains(), matching against the very same
# bundled gene -> Pfam domain table maftools ships internally
# (protein_domains.RDs), grouped by domain label exactly as maftools does
# (`prot.sum[, .(nMut = sum(N)), by = DomainLabel]`).
#
# To make the comparison as fair as possible, this script mirrors pyMut's
# defaults:
#   - AACol            = "Protein_Change" (pyMut's default aa-change column
#                         for the gene-symbol strategy is resolved in the
#                         same priority order maftools uses: HGVSp_Short,
#                         Protein_Change, AAChange - matches automatically
#                         via AACol = NULL, kept explicit here for clarity).
#   - summarizeBy       = "AAPos"      (pyMut's summarize_by='PfamDomain'
#                         groups at the same AAPos-derived domain level).
#   - varClass          = "nonSyn"     (pyMut's include_synonymous=False,
#                         the default of both tools).
#
# Known, expected source of minor discrepancy
# ---------------------------------------------
# read.maf() (maftools) and read_maf() (pyMut, called with
# consolidate_variants=False in this project's notebooks) do not necessarily
# apply identical default variant de-duplication rules. If the two variant
# counts differ by a handful of mutations for some domains, this is the
# most likely explanation and is not a bug in either tool - it is reported
# explicitly in the correlation report produced by the Python step.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  if (!requireNamespace("maftools", quietly = TRUE)) {
    stop(
      "The 'maftools' package is not installed. Install it with:\n",
      "  if (!requireNamespace('BiocManager', quietly = TRUE)) install.packages('BiocManager')\n",
      "  BiocManager::install('maftools')"
    )
  }
  library(maftools)
})

## ---------------------------------------------------------------------------
## Configuration - edit if your files live somewhere else
## ---------------------------------------------------------------------------

# Resolve the folder that contains this script, so all relative paths below
# work no matter where Rscript is invoked from (Rscript passes the script
# path via --file=...; when sourced interactively, fall back to getwd()).
.cmd_args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .cmd_args, value = TRUE)
script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}

data_dir    <- file.path(script_dir, "..", "..", "data")
results_dir <- file.path(script_dir, "results")

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

cohorts <- list(
  list(
    key      = "paad_tb",
    label    = "PAAD-TB",
    maf_path = file.path(data_dir, "PAAD-TB.final_analysis.maf")
  ),
  list(
    key      = "tcga_laml",
    label    = "TCGA-LAML",
    maf_path = file.path(data_dir, "tcga_laml.maf.gz")
  )
)

## ---------------------------------------------------------------------------
## Analysis
## ---------------------------------------------------------------------------

run_cohort <- function(cohort) {
  message(sprintf("\n=== %s ===", cohort$label))

  if (!file.exists(cohort$maf_path)) {
    stop(sprintf("MAF file not found for cohort '%s': %s", cohort$label, cohort$maf_path))
  }

  message(sprintf("Reading MAF: %s", cohort$maf_path))
  maf_obj <- read.maf(maf = cohort$maf_path, verbose = FALSE)

  message("Running maftools::pfamDomains() ...")
  pfam_result <- pfamDomains(
    maf         = maf_obj,
    AACol       = "Protein_Change",
    summarizeBy = "AAPos",
    varClass    = "nonSyn",
    top         = 10
  )

  domain_summary  <- pfam_result$domainSummary
  protein_summary <- pfam_result$proteinSummary

  domain_csv  <- file.path(results_dir, sprintf("%s_maftools_domainSummary.csv", cohort$key))
  protein_csv <- file.path(results_dir, sprintf("%s_maftools_proteinSummary.csv", cohort$key))

  write.csv(domain_summary, domain_csv, row.names = FALSE)
  write.csv(protein_summary, protein_csv, row.names = FALSE)

  message(sprintf(
    "  %d domains / %d protein-domain rows written to:\n    %s\n    %s",
    nrow(domain_summary), nrow(protein_summary), domain_csv, protein_csv
  ))

  invisible(list(domain_summary = domain_summary, protein_summary = protein_summary))
}

results <- lapply(cohorts, run_cohort)

message("\nDone. Now run the companion Python script:")
message("  python correlate_pfam_vs_maftools.py")
