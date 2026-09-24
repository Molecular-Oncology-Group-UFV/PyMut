# =============================================================================
# Title:   Comparison of TMB calculations between maftools and PyMut
# Purpose: Build publication-ready figures comparing Tumor Mutational Burden
#          (TMB) estimates produced by maftools vs. PyMut, for two tumor
#          cohorts (LAML and PAAD).
#
#          Two complementary figures are generated:
#            (1) Scatter-plot comparison  -> "numerical equivalence" check
#                (per-sample TMB, maftools vs. PyMut, with y = x reference
#                 line and Pearson correlation)
#            (2) Bland-Altman plots       -> "Differences in TMB calculations
#                 between both tools" (per-sample difference vs. mean)
#
#          A third, optional summary figure compares the overall descriptive
#          statistics (Mean +/- SD) reported by each tool.
#
# Output:  PNG + PDF files saved under validation/tmb_equivalence/results/
#
# Usage:   Rscript compare_TMB_maftools_vs_pymut.R
#          (paths are resolved relative to this script, so it can be run
#          from any working directory)
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Required packages
# -----------------------------------------------------------------------------
# NOTE: we install only the specific packages actually used by this script,
# instead of the full "tidyverse" meta-package. This avoids pulling in heavy,
# unrelated dependencies (rmarkdown, googledrive, reprex, etc.) that can fail
# to compile on some systems (e.g. missing system library 'libuv' for 'fs').
required_packages <- c("dplyr", "readr", "tidyr", "purrr", "tibble",
                        "ggplot2", "patchwork", "scales")

installed <- rownames(installed.packages())
missing_pkgs <- setdiff(required_packages, installed)
if (length(missing_pkgs) > 0) {
  install.packages(missing_pkgs, repos = "https://cloud.r-project.org")
}

library(dplyr)      # data manipulation (filter, mutate, select, join, etc.)
library(readr)      # fast reading of csv/tsv files
library(tidyr)      # pivot_wider, drop_na
library(purrr)      # map_dfr (looping over tumors/tools)
library(tibble)     # modern data frames
library(ggplot2)    # plotting
library(patchwork)  # combine multiple ggplot panels into one figure
library(scales)     # nicer axis label formatting

# -----------------------------------------------------------------------------
# 1. Paths and general configuration
# -----------------------------------------------------------------------------

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

# Base directory with the input data: the project root (two levels above
# this script). The four per-tool/per-tumor result folders
# (results_laml_maftools, results_tcga_laml_pymut, ...) are expected at
# the project root (e.g. <project_root>/results_laml_maftools/...).
base_dir <- file.path(script_dir, "..", "..")

# Directory where figures will be saved (created if it does not exist):
# results/ inside this script's folder.
fig_dir <- file.path(script_dir, "results")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# Per-tool, per-tumor result folders
paths <- list(
  laml = list(
    maftools = file.path(base_dir, "results_laml_maftools"),
    pymut    = file.path(base_dir, "results_tcga_laml_pymut")
  ),
  paad = list(
    maftools = file.path(base_dir, "results_paad_maftools"),
    pymut    = file.path(base_dir, "results_tcga_paad_pymut")
  )
)

# File names (constant across tumor types)
file_names <- list(
  maftools = list(per_sample = "TMB_por_muestra.csv", summary = "TMB_statistics.csv"),
  pymut    = list(per_sample = "TMB_analysis.tsv",     summary = "TMB_statistics.tsv")
)

# Tumor labels as they should appear in the figures (uppercase, as requested)
tumor_labels <- c(laml = "LAML", paad = "PAAD")

# Tool labels as they should appear in the figures
tool_labels <- c(maftools = "maftools", pymut = "PyMut")

# -----------------------------------------------------------------------------
# Outlier handling (Figures 1 and 2 only)
# -----------------------------------------------------------------------------
# Some cohorts (e.g. PAAD) may contain one or a few hypermutated samples whose
# TMB values are so much larger than the rest that they compress the scale of
# every other point in the scatter / Bland-Altman plots.
#
# EXCLUDE_OUTLIERS_IN_PLOTS: if TRUE, such samples are left out of Figures 1
#   and 2 ONLY (the underlying data and Figure 3 summary statistics are NOT
#   affected). Excluded samples are reported in the console and noted in the
#   plot subtitle so nothing is silently hidden.
#
# OUTLIER_IQR_MULTIPLIER: how extreme a value must be to be excluded, using
#   Tukey's rule: outside [Q1 - k*IQR, Q3 + k*IQR]. The conservative default
#   (k = 3) only removes clearly extreme points, not just the usual upper tail
#   (k = 1.5 would be more aggressive and typically removes more points).
EXCLUDE_OUTLIERS_IN_PLOTS <- TRUE
OUTLIER_IQR_MULTIPLIER    <- 3

# Common ggplot theme: font sizes are increased for readability
theme_tmb <- theme_bw(base_size = 14) +
  theme(
    plot.title       = element_text(size = 15, face = "bold", hjust = 0.5),
    plot.subtitle    = element_text(size = 12, color = "grey30", hjust = 0.5),
    axis.title       = element_text(size = 13),
    axis.text        = element_text(size = 11),
    strip.text       = element_text(size = 13, face = "bold"),
    legend.title     = element_text(size = 12),
    legend.text      = element_text(size = 11),
    panel.grid.minor = element_blank()
  )

# -----------------------------------------------------------------------------
# 2. Helper functions to read and standardize the data
# -----------------------------------------------------------------------------

#' Read the per-sample TMB table for a given tool and tumor type
#'
#' Both tools are expected to report the same columns:
#'   Tumor_Sample_Barcode, total, total_perMB, total_perMB_log
#' maftools files are comma-separated (.csv); PyMut files are tab-separated (.tsv)
read_persample <- function(tumor, tool) {
  folder   <- paths[[tumor]][[tool]]
  fname    <- file_names[[tool]]$per_sample
  filepath <- file.path(folder, fname)

  if (!file.exists(filepath)) {
    stop(sprintf("File not found: %s", filepath))
  }

  reader <- if (tool == "maftools") readr::read_csv else readr::read_tsv

  df <- reader(filepath, show_col_types = FALSE) %>%
    select(Tumor_Sample_Barcode, total, total_perMB, total_perMB_log) %>%
    # NOTE: we keep the raw lowercase tool key here (not the pretty label),
    # because this column feeds pivot_wider() below and the resulting column
    # suffixes (e.g. "total_maftools", "total_pymut") must match exactly what
    # make_scatter_panel() / make_bland_altman_panel() expect.
    mutate(tool = tool, tumor = tumor_labels[[tumor]])

  df
}

#' Read the summary statistics table for a given tool and tumor type
#' Expected columns: Metric, Count, Median, Mean, Min, Max, Q1.25., Q3.75., Std
read_summary <- function(tumor, tool) {
  folder   <- paths[[tumor]][[tool]]
  fname    <- file_names[[tool]]$summary
  filepath <- file.path(folder, fname)

  if (!file.exists(filepath)) {
    stop(sprintf("File not found: %s", filepath))
  }

  df <- readr::read_csv(filepath, show_col_types = FALSE) %>%
    mutate(tool = tool_labels[[tool]], tumor = tumor_labels[[tumor]])

  df
}

# -----------------------------------------------------------------------------
# 3. Load all data
# -----------------------------------------------------------------------------

tumors <- c("laml", "paad")
tools  <- c("maftools", "pymut")

# Per-sample data: one row per sample/tool/tumor
persample_long <- map_dfr(tumors, function(tm) {
  map_dfr(tools, function(tl) read_persample(tm, tl))
})

# Wide format: one row per sample, per tumor, with maftools/PyMut side by side
# This is the table used for the numerical-equivalence and Bland-Altman plots
persample_wide <- persample_long %>%
  pivot_wider(
    id_cols     = c(Tumor_Sample_Barcode, tumor),
    names_from  = tool,
    values_from = c(total, total_perMB, total_perMB_log)
  ) %>%
  # keep only samples present in BOTH tools, to allow a fair paired comparison
  drop_na()

# General summary statistics (Mean, Std, etc.) reported by each tool
summary_long <- map_dfr(tumors, function(tm) {
  map_dfr(tools, function(tl) read_summary(tm, tl))
})

# -----------------------------------------------------------------------------
# 4. Figure 1 - Numerical equivalence (scatter plots, maftools vs PyMut)
# -----------------------------------------------------------------------------

#' Identify extreme outlier rows for a maftools/PyMut pair of columns
#'
#' Uses Tukey's rule on the pooled values of both tools (x and y), so a
#' sample is flagged if EITHER its maftools OR its PyMut value falls far
#' outside the bulk of the distribution.
#'
#' @param d data frame with columns x, y
#' @param k IQR multiplier (see OUTLIER_IQR_MULTIPLIER above)
#' @return logical vector, TRUE = outlier row
flag_outlier_rows <- function(d, k = OUTLIER_IQR_MULTIPLIER) {
  pooled <- c(d$x, d$y)
  q      <- quantile(pooled, probs = c(0.25, 0.75), na.rm = TRUE)
  iqr    <- unname(q[2] - q[1])
  lower  <- q[1] - k * iqr
  upper  <- q[2] + k * iqr
  (d$x < lower | d$x > upper) | (d$y < lower | d$y > upper)
}

#' Prepare the (x, y) data for one panel, optionally removing extreme outliers
#'
#' Returns a list with the (possibly filtered) data frame and a short label
#' describing how many samples were excluded (empty string if none), so it
#' can be appended to the plot subtitle for full transparency.
prepare_panel_data <- function(data, tumor_lab, metric, panel_name) {

  x_col <- paste0(metric, "_maftools")
  y_col <- paste0(metric, "_pymut")

  d <- data %>%
    filter(tumor == tumor_lab) %>%
    rename(x = all_of(x_col), y = all_of(y_col))

  excluded_note <- ""

  if (EXCLUDE_OUTLIERS_IN_PLOTS) {
    is_outlier <- flag_outlier_rows(d)
    n_excluded <- sum(is_outlier)

    if (n_excluded > 0) {
      excluded_ids <- d$Tumor_Sample_Barcode[is_outlier]
      message(sprintf(
        "[%s] %s: excluding %d extreme outlier sample(s) from the plot: %s",
        tumor_lab, panel_name, n_excluded, paste(excluded_ids, collapse = ", ")
      ))
      excluded_note <- "Extreme outliers excluded"
      d <- d[!is_outlier, ]
    }
  }

  list(data = d, note = excluded_note)
}

#' Build one scatter panel comparing maftools vs PyMut for a given metric/tumor
#'
#' @param data      persample_wide data frame
#' @param tumor_lab tumor label to filter on ("LAML" or "PAAD")
#' @param metric    "total" or "total_perMB"
#' @param axis_lab  axis label to display (e.g. "Total mutation count")
make_scatter_panel <- function(data, tumor_lab, metric, axis_lab) {

  prep <- prepare_panel_data(data, tumor_lab, metric, axis_lab)
  d    <- prep$data

  # Pearson correlation coefficient as a simple numerical-equivalence metric
  # (computed AFTER outlier removal, so it reflects the bulk of the data
  # shown in the plot; see console messages for excluded samples)
  r_value  <- cor(d$x, d$y, method = "pearson")
  r_label  <- paste0("r = ", sprintf("%.3f", r_value))

  max_val <- max(c(d$x, d$y), na.rm = TRUE)

  ggplot(d, aes(x = x, y = y)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                color = "grey40", linewidth = 0.6) +
    geom_point(alpha = 0.7, size = 2, color = "#2c7fb8") +
    annotate("text", x = -Inf, y = Inf, label = r_label,
             hjust = -0.15, vjust = 1.5, size = 4.5, fontface = "italic") +
    coord_equal(xlim = c(0, max_val * 1.05), ylim = c(0, max_val * 1.05)) +
    labs(
      title    = paste0(tumor_lab, " - ", axis_lab),
      subtitle = if (nzchar(prep$note)) prep$note else NULL,
      x = paste0(axis_lab, " (maftools)"),
      y = paste0(axis_lab, " (PyMut)")
    ) +
    theme_tmb
}

scatter_total_laml    <- make_scatter_panel(persample_wide, "LAML", "total",       "Total mutations")
scatter_total_paad    <- make_scatter_panel(persample_wide, "PAAD", "total",       "Total mutations")
scatter_perMB_laml    <- make_scatter_panel(persample_wide, "LAML", "total_perMB", "TMB (mutations/Mb)")
scatter_perMB_paad    <- make_scatter_panel(persample_wide, "PAAD", "total_perMB", "TMB (mutations/Mb)")

figure1 <- (scatter_total_laml | scatter_total_paad) /
  (scatter_perMB_laml | scatter_perMB_paad) +
  plot_annotation(
    title    = "TMB calculations compared for numerical equivalence",
    subtitle = "Per-sample values from maftools vs. PyMut (dashed line = perfect agreement, y = x)",
    theme    = theme(
      plot.title    = element_text(size = 17, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 13, color = "grey30", hjust = 0.5)
    )
  )

# -----------------------------------------------------------------------------
# 5. Figure 2 - Bland-Altman plots (differences between both tools)
# -----------------------------------------------------------------------------

#' Build one Bland-Altman panel for a given metric/tumor
#' x-axis: mean of the two tools' values per sample
#' y-axis: difference (maftools - PyMut) per sample
make_bland_altman_panel <- function(data, tumor_lab, metric, axis_lab) {

  prep <- prepare_panel_data(data, tumor_lab, metric, axis_lab)

  d <- prep$data %>%
    rename(maftools_val = x, pymut_val = y) %>%
    mutate(
      mean_val = (maftools_val + pymut_val) / 2,
      diff_val = maftools_val - pymut_val
    )

  mean_diff <- mean(d$diff_val, na.rm = TRUE)
  sd_diff   <- sd(d$diff_val, na.rm = TRUE)
  upper_loa <- mean_diff + 1.96 * sd_diff
  lower_loa <- mean_diff - 1.96 * sd_diff

  ggplot(d, aes(x = mean_val, y = diff_val)) +
    geom_hline(yintercept = 0, color = "grey70", linewidth = 0.5) +
    geom_hline(yintercept = mean_diff, color = "#d95f02", linewidth = 0.7) +
    geom_hline(yintercept = c(upper_loa, lower_loa),
               linetype = "dashed", color = "#d95f02", linewidth = 0.6) +
    geom_point(alpha = 0.7, size = 2, color = "#2c7fb8") +
    labs(
      title    = paste0(tumor_lab, " - ", axis_lab),
      subtitle = if (nzchar(prep$note)) prep$note else NULL,
      x = paste0("Mean of maftools and PyMut (", axis_lab, ")"),
      y = "Difference (maftools - PyMut)"
    ) +
    theme_tmb
}

ba_total_laml <- make_bland_altman_panel(persample_wide, "LAML", "total",       "Total mutations")
ba_total_paad <- make_bland_altman_panel(persample_wide, "PAAD", "total",       "Total mutations")
ba_perMB_laml <- make_bland_altman_panel(persample_wide, "LAML", "total_perMB", "TMB (mutations/Mb)")
ba_perMB_paad <- make_bland_altman_panel(persample_wide, "PAAD", "total_perMB", "TMB (mutations/Mb)")

figure2 <- (ba_total_laml | ba_total_paad) /
  (ba_perMB_laml | ba_perMB_paad) +
  plot_annotation(
    title    = "Differences in TMB calculations between both tools",
    subtitle = "Bland-Altman plots: solid line = mean difference, dashed lines = 95% limits of agreement",
    theme    = theme(
      plot.title    = element_text(size = 17, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 13, color = "grey30", hjust = 0.5)
    )
  )

# -----------------------------------------------------------------------------
# 6. Figure 3 (optional) - Summary statistics comparison (Mean +/- Std)
# -----------------------------------------------------------------------------

metric_labels <- c(total = "Total mutations", total_perMB = "TMB (mutations/Mb)")

summary_plot_data <- summary_long %>%
  filter(Metric %in% c("total", "total_perMB")) %>%
  mutate(metric_label = recode(Metric, !!!metric_labels))

figure3 <- ggplot(summary_plot_data, aes(x = tool, y = Mean, fill = tool)) +
  geom_col(width = 0.6, alpha = 0.85) +
  geom_errorbar(aes(ymin = Mean - Std, ymax = Mean + Std),
                width = 0.15, linewidth = 0.6) +
  facet_grid(metric_label ~ tumor, scales = "free_y") +
  scale_fill_manual(values = c("maftools" = "#7570b3", "PyMut" = "#1b9e77")) +
  labs(
    title    = "Overall TMB summary statistics by tool",
    subtitle = "Bars show Mean, error bars show +/- 1 Std. Dev.",
    x = NULL, y = "Value", fill = "Tool"
  ) +
  theme_tmb +
  theme(legend.position = "top")

# -----------------------------------------------------------------------------
# 7. Save all figures (PNG for quick viewing, PDF for publication quality)
# -----------------------------------------------------------------------------

# --- 7a. Combined multi-panel figures (as before) ---------------------------
ggsave(file.path(fig_dir, "Figure1_TMB_numerical_equivalence.png"),
       figure1, width = 11, height = 9, dpi = 300)
ggsave(file.path(fig_dir, "Figure1_TMB_numerical_equivalence.pdf"),
       figure1, width = 11, height = 9)

ggsave(file.path(fig_dir, "Figure2_TMB_differences_bland_altman.png"),
       figure2, width = 11, height = 9, dpi = 300)
ggsave(file.path(fig_dir, "Figure2_TMB_differences_bland_altman.pdf"),
       figure2, width = 11, height = 9)

ggsave(file.path(fig_dir, "Figure3_TMB_summary_statistics.png"),
       figure3, width = 9, height = 8, dpi = 300)
ggsave(file.path(fig_dir, "Figure3_TMB_summary_statistics.pdf"),
       figure3, width = 9, height = 8)

# --- 7b. Individual panels (each saved on its own, PNG + PDF) ---------------
# Useful when a single panel (e.g. only PAAD, or only the total_perMB metric)
# is needed on its own, for a slide or a specific section of a manuscript.
individual_panels <- list(
  Figure1_panel_LAML_total       = scatter_total_laml,
  Figure1_panel_PAAD_total       = scatter_total_paad,
  Figure1_panel_LAML_total_perMB = scatter_perMB_laml,
  Figure1_panel_PAAD_total_perMB = scatter_perMB_paad,
  Figure2_panel_LAML_total       = ba_total_laml,
  Figure2_panel_PAAD_total       = ba_total_paad,
  Figure2_panel_LAML_total_perMB = ba_perMB_laml,
  Figure2_panel_PAAD_total_perMB = ba_perMB_paad
)

# Subfolder to keep individual panels separate from the combined figures
individual_dir <- file.path(fig_dir, "individual_panels")
dir.create(individual_dir, showWarnings = FALSE, recursive = TRUE)

walk2(individual_panels, names(individual_panels), function(panel, fname) {
  ggsave(file.path(individual_dir, paste0(fname, ".png")),
         panel, width = 6, height = 5.5, dpi = 300)
  ggsave(file.path(individual_dir, paste0(fname, ".pdf")),
         panel, width = 6, height = 5.5)
})

message("All figures were saved in: ", fig_dir)
message("Individual panels were saved in: ", individual_dir)

# Display the figures interactively if running in RStudio / an interactive session
if (interactive()) {
  print(figure1)
  print(figure2)
  print(figure3)
}
