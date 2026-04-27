#!/usr/bin/env Rscript

# ---------------------------------------------------------------------------
# Simple script: compare your signatures (W: 96xK) against COSMIC v3.4 (GRCh37).
# Tries maftools::compareSignatures; if it fails, computes cosine similarity.
# Paths are fixed as requested by the user.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  if (!requireNamespace("maftools", quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org")
    BiocManager::install("maftools", ask = FALSE, update = FALSE)
  }
  library(maftools)
})

# --- Paths (edit here if needed) ---
COSMIC_FILE <- "/home/luisruimore/Escritorio/TFG/src/pyMut/data/examples/COSMIC_catalogue-signatures_SBS96_v3.4/COSMIC_v3.4_SBS_GRCh37.txt"
W_CSV      <- "/home/luisruimore/Escritorio/TFG/validation/mutational_signature/extract_signatures/R_signatures_W_96x3.csv"
OUT_DIR   <- "/home/luisruimore/Escritorio/TFG/validation/mutational_signature/compare_signatures"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
OUT_PREFIX <- file.path(OUT_DIR, "R_cosmic_cmp")

# --- Read ---
read_W <- function(path){
  W <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE))
  storage.mode(W) <- "double"
  if (any(!is.finite(W))) stop("W contains NA/Inf or non-numeric values.")
  if (nrow(W) != 96) stop(sprintf("W must have 96 rows (has %d).", nrow(W)))
  W
}

read_COSMIC <- function(path){
  # Typical format: TSV with 'Type' column and SBS1..SBSxx columns (values in [0,1])
  DB <- as.matrix(read.delim(path, row.names = 1, check.names = FALSE))
  storage.mode(DB) <- "double"
  if (any(!is.finite(DB))) stop("COSMIC contains NA/Inf or non-numeric values.")
  if (nrow(DB) != 96) stop(sprintf("COSMIC must have 96 rows (has %d).", nrow(DB)))
  DB
}

W  <- read_W(W_CSV)          # 96 x K
DB <- read_COSMIC(COSMIC_FILE) # 96 x S (SBS1..)

# --- Context alignment ---
missing_in_db <- setdiff(rownames(W), rownames(DB))
if (length(missing_in_db) > 0) {
  stop(paste0("W contexts not present in COSMIC: ", paste(missing_in_db, collapse=", "),
              "
Check that you use labels like 'A[C>A]A'."))
}
# Reorder DB to match W
DB <- DB[rownames(W), , drop = FALSE]

# --- Attempt 1: use maftools::compareSignatures with custom DB ---
res <- NULL
try({
  nmfRes <- list(signatures = W)
  # maftools accepts 'sig_db' as a matrix with the same rows
  res <- maftools::compareSignatures(nmfRes = nmfRes, sig_db = DB, verbose = TRUE)
}, silent = TRUE)

# --- If it fails: manual cosine calculation ---
manual_mode <- is.null(res)
if (manual_mode) {
  message("compareSignatures failed with custom database; computing cosines manually...")
  # Normalize columns to L2 and multiply: cosines = t(Wn) %*% DBn
  l2norm <- function(A){
    d <- sqrt(colSums(A*A)); d[d==0] <- 1
    sweep(A, 2, d, "/")
  }
  Wn  <- l2norm(W)
  DBn <- l2norm(DB)
  COS <- t(Wn) %*% DBn   # K x S
  # 'best match' for each signature (column of W)
  best_idx <- apply(COS, 1, which.max)
  best_cos <- mapply(function(i, j) COS[i, j], seq_len(nrow(COS)), best_idx)
  best_df <- data.frame(
    signature = colnames(W),
    cosmic    = colnames(DB)[best_idx],
    cosine    = as.numeric(best_cos),
    stringsAsFactors = FALSE
  )
  cosine_mat <- COS
} else {
  cosine_mat <- res$cosine_similarities  # K x S
  # Build best_df from 'best_match'
  # Build best_df from 'best_match'
  best_df <- do.call(rbind, lapply(names(res$best_match), function(nm){
    bm <- res$best_match[[nm]]
    cosmic_name <- sub(".*Best match: ", "", bm$best_match)
    cosmic_name <- sub(" \\[cosine-similarity:.*", "", cosmic_name)  # <-- escaped bracket
    cos_val <- sub(".*cosine-similarity: ", "", bm$best_match)
    cos_val <- as.numeric(sub("\\].*", "", cos_val))
    data.frame(
      signature = nm,
      cosmic    = cosmic_name,
      cosine    = cos_val,
      aetiology = as.character(bm$aetiology),
      stringsAsFactors = FALSE
    )
  }))
  rownames(best_df) <- NULL
}

# --- Outputs ---
cos_csv  <- file.path(OUT_DIR, "R_cosine_matrix.csv")
best_csv <- file.path(OUT_DIR, "R_summary_compare_signatures.csv")

write.csv(cosine_mat, file = cos_csv, quote = TRUE)
write.csv(best_df,  file = best_csv,  row.names = FALSE)

cat(sprintf("OK: %s
OK: %s
", cos_csv, best_csv))
if (manual_mode) cat("(Manual cosine mode; compareSignatures was not used)\n")
cat("Done.\n")
