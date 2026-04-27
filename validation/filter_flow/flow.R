# --- Configuration ---
get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  i <- grep(file_arg, args)
  if (length(i) > 0) {
    return(normalizePath(sub(file_arg, "", args[i][1])))
  }
  # When sourced (e.g., in RStudio)
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile))
  }
  # Fallback to current working directory
  return(normalizePath("."))
}
script_file <- get_script_path()
script_dir <- dirname(script_file)
REPO_ROOT <- normalizePath(file.path(script_dir, "..", ".."))

maf_file <- file.path(REPO_ROOT, "src", "pyMut", "data", "examples", "MAF", "tcga_laml.maf.gz")

chroms_keep <- c("1", "2", "3")
pos_min <- 20818769L
pos_max <- 133551224L

# Output file (same directory, with suffix)
out_file <- file.path(
  dirname(maf_file),
  paste0(
    tools::file_path_sans_ext(basename(maf_file)),
    "_filtered_R.maf"
  )
)

# --- Packages ---
library(data.table)

# --- 1) Read MAF and ensure valid/unique column names ---
dt <- fread(maf_file, data.table = TRUE, check.names = TRUE)
# Warn if duplicates were found
dups <- names(dt)[duplicated(names(dt))]
if (length(dups) > 0) message("Se detectaron columnas duplicadas y fueron renombradas: ",
                              paste(unique(dups), collapse = ", "))

# --- 2) Check required columns ---
req_cols <- c("Chromosome", "Start_Position")
missing <- setdiff(req_cols, names(dt))
if (length(missing) > 0) {
  stop("Faltan columnas en el MAF: ", paste(missing, collapse = ", "))
}

# --- 3) Normalize chromosome field and filter ---
# (if values are like 'chr1', remove the prefix to compare)
dt[, Chromosome := sub("^chr", "", as.character(Chromosome), ignore.case = TRUE)]

dt_filt <- dt[
  Chromosome %chin% chroms_keep &
    Start_Position >= pos_min &
    Start_Position <= pos_max
]

if (nrow(dt_filt) == 0) stop("El filtro no devolvió variantes.")

# --- 4) Write output in MAF (TSV) format ---
fwrite(dt_filt, out_file, sep = "\t", quote = FALSE, na = "")

cat("MAF filtrado escrito en:\n  ", normalizePath(out_file), "\n")