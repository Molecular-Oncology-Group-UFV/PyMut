# --- Packages ---
# BiocManager::install("maftools")
# install.packages(c("dplyr","readr"))
library(maftools)
library(dplyr)
library(readr)

# ===== Parameters =====
maf_file <- "/home/luisruimore/Escritorio/TFG/src/pyMut/data/examples/MAF/tcga_laml.maf.gz"
genome_size_bp <- 60456963
capture_size_mb <- genome_size_bp / 1e6  # 60.456963

# Non-synonymous (same as in your Python; maftools uses CamelCase)
vc_nonsyn <- c(
  "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
  "Nonstop_Mutation","Translation_Start_Site","Splice_Site",
  "In_Frame_Del","In_Frame_Ins",
  "Start_Codon_SNP","Start_Codon_Del","Start_Codon_Ins",
  "Stop_Codon_Del","Stop_Codon_Ins",
  "De_novo_Start_In_Frame","De_novo_Start_Out_Frame"
)

# ===== Read MAF with maftools =====
# NOTE: 'read.maf' stores non-synonymous in @data and SYNONYMOUS in @maf.silent.
# 'vc_nonSyn' ensures it counts exactly the above as non-synonymous.
m <- read.maf(maf = maf_file, vc_nonSyn = vc_nonsyn, verbose = FALSE)  # :contentReference[oaicite:1]{index=1}

# ===== 1) Non-synonymous TMB using tmb() =====
# tmb() uses getSampleSummary() (non-synonymous) and produces 'total_perMB' = non-syn/Mb. :contentReference[oaicite:2]{index=2}
tmb_ns <- tmb(maf = m, captureSize = capture_size_mb, logScale = FALSE) %>%
  select(Tumor_Sample_Barcode, total_perMB) %>%
  rename(TMB_Non_Synonymous_Normalized = total_perMB)

# ===== 2) Counts per sample: NON-SYN and SYN to obtain 'Total_Mutations' =====
ns_counts <- getSampleSummary(m) %>%               # only non-synonymous. :contentReference[oaicite:3]{index=3}
  select(Tumor_Sample_Barcode, total) %>%
  rename(Non_Synonymous_Mutations = total)

# Synonymous are in the @maf.silent slot of the MAF object. :contentReference[oaicite:4]{index=4}
syn_df <- m@maf.silent
syn_counts <- if (!is.null(syn_df) && nrow(syn_df) > 0) {
  syn_df %>% count(Tumor_Sample_Barcode, name = "Synonymous_Mutations")
} else {
  tibble(Tumor_Sample_Barcode = character(), Synonymous_Mutations = integer())
}

# Combine and calculate totals + total TMB
per_sample <- full_join(ns_counts, syn_counts, by = "Tumor_Sample_Barcode") %>%
  mutate(
    Non_Synonymous_Mutations = coalesce(Non_Synonymous_Mutations, 0L),
    Synonymous_Mutations     = coalesce(Synonymous_Mutations, 0L),
    Total_Mutations          = Non_Synonymous_Mutations + Synonymous_Mutations
  ) %>%
  select(Tumor_Sample_Barcode, Total_Mutations, Non_Synonymous_Mutations)

# Add non-synonymous TMB (from tmb()) and compute total TMB
per_sample <- per_sample %>%
  left_join(tmb_ns, by = c("Tumor_Sample_Barcode" = "Tumor_Sample_Barcode")) %>%
  mutate(TMB_Total_Normalized = Total_Mutations / capture_size_mb)

# ===== EXACT outputs with 6 decimals =====
fmt6 <- function(x) sprintf("%.6f", round(x, 6))

TMB_analysis <- per_sample %>%
  transmute(
    Sample = Tumor_Sample_Barcode,
    Total_Mutations = as.integer(Total_Mutations),
    Non_Synonymous_Mutations = as.integer(Non_Synonymous_Mutations),
    TMB_Total_Normalized = fmt6(TMB_Total_Normalized),
    TMB_Non_Synonymous_Normalized = fmt6(TMB_Non_Synonymous_Normalized)
  )

write_tsv(TMB_analysis, "TMB_analysis.tsv", na = "")

# ===== Statistics (for TMB_statistics.tsv) =====
x_total      <- per_sample$Total_Mutations
x_nonsyn     <- per_sample$Non_Synonymous_Mutations
x_tmb_total  <- as.numeric(TMB_analysis$TMB_Total_Normalized)
x_tmb_nonsyn <- as.numeric(TMB_analysis$TMB_Non_Synonymous_Normalized)

stats_vec <- function(x) c(
  Count  = length(x),
  Mean   = mean(x),
  Median = median(x),
  Min    = min(x),
  Max    = max(x),
  Q1     = as.numeric(quantile(x, 0.25, names = FALSE, type = 7)),
  Q3     = as.numeric(quantile(x, 0.75, names = FALSE, type = 7)),
  Std    = sd(x)
)

S <- rbind(
  Total_Mutations                = stats_vec(x_total),
  Non_Synonymous_Mutations       = stats_vec(x_nonsyn),
  TMB_Total_Normalized           = stats_vec(x_tmb_total),
  TMB_Non_Synonymous_Normalized  = stats_vec(x_tmb_nonsyn)
)

TMB_statistics <- tibble::tibble(
  Metric = rownames(S),
  Count  = as.integer(S[, "Count"]),
  Mean   = fmt6(S[, "Mean"]),
  Median = fmt6(S[, "Median"]),
  Min    = fmt6(S[, "Min"]),
  Max    = fmt6(S[, "Max"]),
  Q1     = fmt6(S[, "Q1"]),
  Q3     = fmt6(S[, "Q3"]),
  Std    = fmt6(S[, "Std"])
)

write_tsv(TMB_statistics, "TMB_statistics.tsv", na = "")

cat("Done:\n- TMB_analysis.tsv\n- TMB_statistics.tsv\n")
