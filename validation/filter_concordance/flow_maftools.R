#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# flow_maftools.R
#
# maftools side of the filter-concordance benchmark. For EACH of two
# independent TCGA cohorts (TCGA-LAML and TCGA-PAAD), this script applies
# the SAME two filters as flow_pymut.py, using maftools' own API
# (read.maf + subsetMaf), not a raw data.table re-implementation:
#
#   1) Chromosome-based filter: keep chromosomes 1, 2, 3, 17 (same filter,
#      applied independently to each cohort).
#   2) Gene-locus / position-based filter (GRCh37 coordinates):
#        - TCGA-LAML  -> DNMT3A locus, chromosome 2
#        - TCGA-PAAD  -> TP53 locus,   chromosome 17
#      A different gene was chosen per cohort because it is a clinically
#      relevant, recurrently mutated gene for that tumor type (DNMT3A in
#      AML, TP53 in pancreatic adenocarcinoma), which keeps the region
#      filter non-trivial (i.e. it actually removes variants) in both
#      cohorts.
#
# Counterpart: flow_pymut.py (same cohorts, same two filters, done with
# pyMut).
# Comparison + figure: compare_and_plot.py (combines both cohorts into a
# single publication figure).
#
# Paths (resolved relative to this script, regardless of the working dir):
#   - Input MAFs:   <project_root>/data/
#   - Output MAFs:  validation/filter_concordance/results/
#
# Requirements: R (>= 4.0), maftools, data.table
#   install.packages("BiocManager")
#   BiocManager::install("maftools")
#   install.packages("data.table")

suppressPackageStartupMessages({
  library(maftools)
  library(data.table)
})

# --------------------------------------------------------------------------
# Paths (relative to the validation/ folder -- run from there)
# --------------------------------------------------------------------------

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

# Directory containing the raw input MAF files for both cohorts. Same
# location used by flow_pymut.py, so both sides read identical input data.
DATA_DIR <- file.path(script_dir, "..", "..", "data")

# Directory where the filtered MAF files are written. Shared with
# flow_pymut.py and consumed by compare_and_plot.py.
OUTPUT_DIR <- file.path(script_dir, "results")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Chromosome-based filter: identical set of chromosomes for both cohorts.
# Keep this in sync with flow_pymut.py (CHROM_FILTER).
CHROM_FILTER <- c("1", "2", "3", "17")

# --------------------------------------------------------------------------
# Cohort definitions
# --------------------------------------------------------------------------
# GRCh37/hg19 gene-body coordinates used for the region filter:
#   - DNMT3A: chr2:25,455,845-25,565,459  (RefSeq NM_022552, minus strand)
#   - TP53:   chr17:7,571,720-7,590,868   (RefSeq NM_000546, minus strand)
#
# Keep the values below in sync with the matching entry in the `COHORTS`
# list in flow_pymut.py.
COHORTS <- list(
  list(
    name = "laml",
    label = "TCGA-LAML",
    maf_input = file.path(DATA_DIR, "tcga_laml.maf.gz"),
    region_gene = "DNMT3A",
    region_chrom = "2",
    region_start = 25455845,
    region_end = 25565459
  ),
  list(
    name = "paad",
    label = "TCGA-PAAD",
    maf_input = file.path(DATA_DIR, "PAAD-TB.final_analysis.clean.maf"),
    region_gene = "TP53",
    region_chrom = "17",
    region_start = 7571720,
    region_end = 7590868
  )
)

# --------------------------------------------------------------------------
# Per-cohort processing
# --------------------------------------------------------------------------

process_cohort <- function(cohort) {

  cat(sprintf("\n=== %s (maftools) ===\n", cohort$label))

  if (!file.exists(cohort$maf_input)) {
    stop(sprintf("Input MAF not found for cohort '%s': %s", cohort$label, cohort$maf_input))
  }

  # --- Load and normalize -------------------------------------------------
  # Read the raw MAF once to (a) strip any 'chr' prefix from Chromosome so
  # it matches pyMut's normalization, and (b) collect every
  # Variant_Classification value present. That second step is passed to
  # read.maf() as vc_nonSyn so that ALL variants land in the MAF object's
  # @data slot -- by default maftools silently reroutes "silent"
  # classifications (e.g. Silent, Intron, 3'UTR) into a separate
  # @maf.silent slot, which would otherwise make this comparison against
  # pyMut's full variant table (which keeps everything in one table) unfair.
  raw <- fread(cohort$maf_input, sep = "\t", header = TRUE, na.strings = c("", "NA"))
  raw[, Chromosome := sub("^chr", "", as.character(Chromosome), ignore.case = TRUE)]

  tmp_maf_path <- file.path(tempdir(), sprintf("tcga_%s_normalized.maf", cohort$name))
  fwrite(raw, tmp_maf_path, sep = "\t", na = "", quote = "auto")

  all_classifications <- sort(unique(as.character(raw$Variant_Classification)))
  cat(sprintf("Variant_Classification values found (%d): %s\n",
              length(all_classifications), paste(all_classifications, collapse = ", ")))

  maf_obj <- read.maf(maf = tmp_maf_path, vc_nonSyn = all_classifications)

  if (nrow(maf_obj@maf.silent) != 0) {
    warning(sprintf(
      "%s: maf.silent has %d rows after forcing vc_nonSyn -- some variants may be excluded from the filters below.",
      cohort$label, nrow(maf_obj@maf.silent)
    ))
  }
  cat(sprintf("Total variants loaded into @data: %d\n", nrow(maf_obj@data)))

  # --- Chromosome-based filter (maftools::subsetMaf) -----------------------
  cat(sprintf("Applying chromosome filter: %s\n", paste(CHROM_FILTER, collapse = ", ")))
  chrom_query <- sprintf("Chromosome %%in%% c(%s)",
                          paste(sprintf("'%s'", CHROM_FILTER), collapse = ", "))
  chrom_maf <- subsetMaf(maf = maf_obj, query = chrom_query, mafObj = TRUE)
  chrom_dt <- chrom_maf@data
  cat(sprintf("Variants after chromosome filter: %d\n", nrow(chrom_dt)))

  chrom_out <- file.path(OUTPUT_DIR, sprintf("tcga_%s_chr_filter_maftools.maf", cohort$name))
  fwrite(chrom_dt, chrom_out, sep = "\t", na = "", quote = "auto")
  cat(sprintf("Written: %s\n", chrom_out))

  # --- Gene-locus / position-based filter (maftools::subsetMaf) ------------
  cat(sprintf("Applying %s region filter: chr%s:%d-%d\n",
              cohort$region_gene, cohort$region_chrom, cohort$region_start, cohort$region_end))
  region_query <- sprintf("Chromosome == '%s' & Start_Position >= %d & Start_Position <= %d",
                           cohort$region_chrom, cohort$region_start, cohort$region_end)
  region_maf <- subsetMaf(maf = maf_obj, query = region_query, mafObj = TRUE)
  region_dt <- region_maf@data
  cat(sprintf("Variants after %s region filter: %d\n", cohort$region_gene, nrow(region_dt)))

  region_out <- file.path(
    OUTPUT_DIR,
    sprintf("tcga_%s_%s_region_maftools.maf", cohort$name, tolower(cohort$region_gene))
  )
  fwrite(region_dt, region_out, sep = "\t", na = "", quote = "auto")
  cat(sprintf("Written: %s\n", region_out))

  invisible(NULL)
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

for (cohort in COHORTS) {
  process_cohort(cohort)
}

cat("\nDone. Now run compare_and_plot.py.\n")
