# R/generate_trinuc_matrix.R
# Uso:
#   Rscript trinucleotide.R --maf /home/luisruimore/Escritorio/TFG/src/pyMut/data/examples/MAF/tcga_laml.maf.gz --genome hg19 --out r_trinuc_matrix.csv
#   Rscript trinucleotide.R --maf /home/luisruimore/Escritorio/TFG/src/pyMut/data/examples/MAF/tcga_laml.maf.gz --genome hs37d5 --out r_trinuc_matrix.csv

#
# Requiere (ejemplos):
#   BiocManager::install(c("maftools", "BSgenome"))
#   BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")    # UCSC/hg19 (GRCh37)
#   BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")    # UCSC/hg38 (GRCh38)
#   BiocManager::install("BSgenome.Hsapiens.1000genomes.hs37d5")  # Ensembl/GRCh37 (hs37d5)

#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("optparse", quietly = TRUE)) install.packages("optparse", repos="https://cloud.r-project.org")
  library(optparse)
})

opt_list <- list(
  make_option("--maf", type="character", help="Ruta al MAF (.maf o .maf.gz)"),
  make_option("--genome", type="character", default="hg19",
              help="hg19|hg38|hs37d5 o nombre BSgenome (p.ej. BSgenome.Hsapiens.UCSC.hg19)"),
  make_option("--out", type="character", default="r_trinuc_matrix.csv",
              help="Salida CSV (96 x N)")
)
opt <- parse_args(OptionParser(option_list = opt_list))
stopifnot(!is.null(opt$maf))

# Resolve BSgenome from common aliases
map_genome <- function(x){
  x0 <- tolower(x)
  if (x0 %in% c("hg19","grch37"))        return("BSgenome.Hsapiens.UCSC.hg19")   # UCSC
  if (x0 %in% c("hg38","grch38"))        return("BSgenome.Hsapiens.UCSC.hg38")   # UCSC
  if (x0 %in% c("hs37d5","grch37d5"))    return("BSgenome.Hsapiens.1000genomes.hs37d5") # Ensembl-like
  return(x) # already a BSgenome name
}
bs_pkg <- map_genome(opt$genome)

ensure_pkg <- function(pkg, bioc=FALSE){
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (bioc) {
      if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org")
      BiocManager::install(pkg, ask=FALSE, update=FALSE)
    } else {
      install.packages(pkg, repos="https://cloud.r-project.org")
    }
  }
}

ensure_pkg("maftools", bioc = TRUE)
ensure_pkg("BSgenome", bioc = TRUE)
ensure_pkg(bs_pkg, bioc = TRUE)

suppressPackageStartupMessages({
  library(maftools)
})

# Detect genome naming style from BSgenome name
detect_style <- function(bs_pkg){
  if (grepl("UCSC", bs_pkg, ignore.case = TRUE)) return("UCSC")
  if (grepl("1000genomes\\.hs37d5", bs_pkg, ignore.case = TRUE)) return("Ensembl")
  # default: UCSC
  return("UCSC")
}
style <- detect_style(bs_pkg)
cat(sprintf("Usando BSgenome: %s  (estilo: %s)\n", bs_pkg, style))

cat("Leyendo MAF…\n")
laml <- maftools::read.maf(maf = opt$maf, verbose = TRUE)

# Normalize chromosome naming to the reference style
chr_vec <- as.character(laml@data$Chromosome)
has_chr <- any(grepl("^chr", chr_vec))

if (style == "UCSC") {
  # Target: 'chr*' and 'chrM'
  # - If no 'chr', use prefix='chr'
  # - Mitochondria: MT -> M so prefix='chr' yields 'chrM'
  chr_vec <- sub("^MT$", "M", chr_vec)
  chr_vec <- sub("^chrMT$", "chrM", chr_vec)  # in case it already had chrMT
  laml@data$Chromosome <- chr_vec
  prefix <- if (has_chr) NULL else "chr"
} else { # Ensembl (hs37d5)
  # Target: no 'chr' and 'MT'
  chr_vec <- sub("^chr", "", chr_vec)         # remove prefix if present
  chr_vec <- sub("^M$", "MT", chr_vec)        # M -> MT
  chr_vec <- sub("^chrM$", "MT", chr_vec)     # chrM -> MT
  laml@data$Chromosome <- chr_vec
  prefix <- NULL  # never add 'chr' for Ensembl
}

cat("Construyendo matriz 96×N con maftools::trinucleotideMatrix…\n")
tnm <- maftools::trinucleotideMatrix(
  maf = laml,
  ref_genome = bs_pkg,   # BSgenome name
  prefix = prefix,       # add 'chr' only when needed (UCSC and MAF without 'chr')
  add = TRUE,
  useSyn = TRUE          # include all SNVs
)

nmf_mat <- tnm$nmf_matrix # N x 96
mat_96xN <- t(nmf_mat)    # 96 x N

# Ensure unique, sortable column names
colnames(mat_96xN) <- make.unique(colnames(mat_96xN))

write.csv(mat_96xN, file = opt$out, quote = TRUE)
cat(sprintf("OK: %s (dim %dx%d)\n", opt$out, nrow(mat_96xN), ncol(mat_96xN)))