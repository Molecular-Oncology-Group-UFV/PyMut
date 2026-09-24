#!/usr/bin/env python3
"""
correlate_pfam_vs_maftools.py
==============================================================================
Correlate pyMut's Pfam-domain mutation counts against the reference R
maftools::pfamDomains() output, for the PAAD-TB and TCGA-LAML cohorts.

Purpose
-------
pyMut re-implements maftools' domain-level Pfam annotation (gene symbol +
amino acid position matched against the same bundled Pfam/CDD domain table
maftools uses internally). This script provides an empirical, per-cohort
validation of that re-implementation by comparing, for every Pfam domain:

    n_variants  (pyMut)   vs   nMuts   (maftools)
    n_genes     (pyMut)   vs   nGenes  (maftools)

and reporting Pearson/Spearman correlation coefficients together with a
publication-quality scatter-plot figure for each cohort.

Two-step pipeline
------------------
This script is the SECOND step of a two-step pipeline. maftools is an R/
Bioconductor package, so its reference output cannot be produced from pure
Python: it must first be generated with a small R script.

    1. Rscript run_maftools_pfam.R
           -> writes <cohort>_maftools_domainSummary.csv to the results dir

    2. python correlate_pfam_vs_maftools.py
           -> runs pyMut's own annotate_pfam() / pfam_domains() on the same
              MAF files, merges the result with the CSVs from step 1, and
              writes the correlation figures/tables/report to the results dir.

Run step 1 first. This script will raise a clear error, naming the missing
file, if it is executed before step 1 has produced the required CSVs.

Paths (resolved relative to this script, regardless of the working dir):
  - Input MAFs:    <project_root>/data/
  - Input CSVs:    validation/pfam_domain_correlation/results/ (from step 1)
  - Outputs:       validation/pfam_domain_correlation/results/

Inputs (edit in `build_default_cohorts()` below if paths differ)
------------------------------------------------------------------
    <project_root>/data/PAAD-TB.final_analysis.maf
    <project_root>/data/tcga_laml.maf.gz

Outputs (all written to validation/pfam_domain_correlation/results/)
------------------------------------------------------------------
    pfam_vs_maftools_correlation.png / .pdf
        2x2 publication figure: rows = {variants per domain, genes per
        domain}, columns = {PAAD-TB, TCGA-LAML}. Each panel is a log-log
        scatter plot with the y = x concordance line, an ordinary-least-
        squares fit on log10(1 + count), and Pearson/Spearman statistics
        annotated on the panel.

    <cohort>_pfam_vs_maftools_merged.csv
        Per-domain merged table (pyMut and maftools counts side by side)
        used to produce the figure, for the raw numbers.

    correlation_report.txt
        Plain-text summary: correlation statistics, number of domains
        compared, and any domains recovered by only one of the two tools
        (useful for spotting edge cases / QC).

Usage
-----
    python correlate_pfam_vs_maftools.py
    python correlate_pfam_vs_maftools.py --base-dir /custom/data/dir
    python correlate_pfam_vs_maftools.py --results-dir /custom/output/dir
==============================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger("correlate_pfam_vs_maftools")

# A very large top_n so pyMut's pfam_domains() returns every annotated
# domain rather than truncating the ranking.
_ALL_DOMAINS = 10 ** 9


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class CohortConfig:
    """Everything needed to analyse one cohort and locate its maftools export."""

    key: str            # short machine-friendly id, e.g. "paad_tb" (must match run_maftools_pfam.R)
    display_name: str   # human-readable label used in titles/legends, e.g. "PAAD-TB"
    maf_path: Path       # path to the cohort's MAF file
    assembly: str        # genome assembly passed to pyMut's read_maf()
    maftools_domain_csv: Path  # domainSummary CSV produced by run_maftools_pfam.R


def build_default_cohorts(base_dir: Path, results_dir: Path) -> list[CohortConfig]:
    """
    Cohort definitions used in this project's notebooks PAAD-TB and TCGA-LAML, both GRCh37/hg19.
    """
    return [
        CohortConfig(
            key="paad_tb",
            display_name="TCGA - PAAD",
            maf_path=base_dir / "PAAD-TB.final_analysis.maf",
            assembly="37",
            maftools_domain_csv=results_dir / "paad_tb_maftools_domainSummary.csv",
        ),
        CohortConfig(
            key="tcga_laml",
            display_name="TCGA-LAML",
            maf_path=base_dir / "tcga_laml.maf.gz",
            assembly="37",
            maftools_domain_csv=results_dir / "tcga_laml_maftools_domainSummary.csv",
        ),
    ]


# =============================================================================
# pyMut side: run the package's own Pfam annotation
# =============================================================================

def run_pymut_pfam_summary(cohort: CohortConfig) -> pd.DataFrame:
    """
    Load a cohort's MAF with pyMut and return its domain-level Pfam summary.

    Mirrors the exact calls used in Standard_pymut_updated.ipynb:
        read_maf(maf_path, assembly=..., consolidate_variants=False)
        mutations.annotate_pfam()
        annotated.pfam_domains(summarize_by='PfamDomain', top_n=<all>)

    Returns
    -------
    DataFrame with (at least) columns: pfam_id, pfam_name, n_genes, n_variants.
    `pfam_name` is the domain label pyMut groups by, equivalent to
    maftools' `DomainLabel`.
    """
    # Imported lazily so this module can be inspected/imported without pyMut
    # installed (e.g. for documentation tooling); the actual run needs it.
    from pyMut import read_maf

    LOGGER.info("[%s] Reading MAF: %s", cohort.display_name, cohort.maf_path)
    if not cohort.maf_path.exists():
        raise FileNotFoundError(
            f"MAF file not found for cohort '{cohort.display_name}': {cohort.maf_path}"
        )

    mutations = read_maf(cohort.maf_path, assembly=cohort.assembly, consolidate_variants=False)
    LOGGER.info("[%s] %s variants loaded", cohort.display_name, f"{len(mutations.data):,}")

    LOGGER.info("[%s] Running pyMut annotate_pfam() ...", cohort.display_name)
    annotated = mutations.annotate_pfam()

    LOGGER.info("[%s] Summarizing Pfam domains ...", cohort.display_name)
    summary = annotated.pfam_domains(
        summarize_by="PfamDomain",
        top_n=_ALL_DOMAINS,
        include_synonymous=False,  # matches maftools' varClass='nonSyn' default
    )

    if len(summary) == 0:
        raise RuntimeError(
            f"pyMut returned an empty Pfam domain summary for cohort "
            f"'{cohort.display_name}'. Check that annotate_pfam() found "
            f"Hugo_Symbol/Protein_Change columns in this MAF."
        )

    LOGGER.info("[%s] pyMut: %d Pfam domains annotated", cohort.display_name, len(summary))
    return summary


# =============================================================================
# maftools side: load the CSV produced by run_maftools_pfam.R
# =============================================================================

def load_maftools_domain_summary(csv_path: Path, cohort_label: str) -> pd.DataFrame:
    """
    Load and normalize the domainSummary table exported by
    run_maftools_pfam.R (columns: DomainLabel, nMuts, nGenes, pfam,
    Description - see maftools::pfamDomains() source, R/domainSummary.R).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"maftools domain summary not found for cohort '{cohort_label}': {csv_path}\n"
            f"Run `Rscript run_maftools_pfam.R` first - see the module docstring "
            f"of this script for the full two-step pipeline."
        )

    df = pd.read_csv(csv_path)
    required = {"DomainLabel", "nMuts", "nGenes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"'{csv_path}' is missing expected column(s) {sorted(missing)}. "
            f"Was it produced by run_maftools_pfam.R?"
        )

    df["DomainLabel"] = df["DomainLabel"].astype(str).str.strip()
    LOGGER.info("[%s] maftools: %d Pfam domains loaded from %s", cohort_label, len(df), csv_path.name)
    return df


# =============================================================================
# Merge + statistics
# =============================================================================

def merge_domain_summaries(pymut_summary: pd.DataFrame, maftools_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join the two per-domain summaries on the domain label (pyMut's
    `pfam_name` == maftools' `DomainLabel` - both group by the same key from
    the shared bundled domain table). Domains recovered by only one tool are
    kept (with NaN on the other side) so they can be reported for QC, but are
    excluded from the correlation statistics computed downstream.
    """
    left = pymut_summary[["pfam_name", "pfam_id", "n_variants", "n_genes"]].copy()
    left = left.rename(columns={
        "pfam_name": "domain_label",
        "n_variants": "pymut_n_variants",
        "n_genes": "pymut_n_genes",
    })
    left["domain_label"] = left["domain_label"].astype(str).str.strip()

    right = maftools_summary[["DomainLabel", "nMuts", "nGenes"]].copy()
    right = right.rename(columns={
        "DomainLabel": "domain_label",
        "nMuts": "maftools_n_variants",
        "nGenes": "maftools_n_genes",
    })

    merged = left.merge(right, on="domain_label", how="outer")
    merged["in_both"] = merged["pymut_n_variants"].notna() & merged["maftools_n_variants"].notna()
    merged = merged.sort_values("domain_label", kind="mergesort").reset_index(drop=True)
    return merged


@dataclass(frozen=True)
class CorrelationStats:
    n_domains: int
    pearson_r_log: float
    pearson_p_log: float
    spearman_rho: float
    spearman_p: float
    n_exact_match: int  # domains where pyMut and maftools report the identical count


def compute_correlation(merged: pd.DataFrame, pymut_col: str, maftools_col: str) -> CorrelationStats:
    """
    Compute Pearson correlation on log10(1 + count) (appropriate for
    heavily right-skewed mutation-count data spanning several orders of
    magnitude) and Spearman rank correlation (scale-free, robust to the
    same skew) between the two tools' per-domain counts, restricted to
    domains recovered by both.
    """
    common = merged.loc[merged["in_both"], [pymut_col, maftools_col]].dropna()

    if len(common) < 2:
        return CorrelationStats(len(common), np.nan, np.nan, np.nan, np.nan, 0)

    x_log = np.log10(1 + common[pymut_col].to_numpy(dtype=float))
    y_log = np.log10(1 + common[maftools_col].to_numpy(dtype=float))

    pearson_r, pearson_p = stats.pearsonr(x_log, y_log)
    spearman_rho, spearman_p = stats.spearmanr(common[pymut_col], common[maftools_col])
    n_exact = int((common[pymut_col] == common[maftools_col]).sum())

    return CorrelationStats(
        n_domains=len(common),
        pearson_r_log=pearson_r,
        pearson_p_log=pearson_p,
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        n_exact_match=n_exact,
    )


# =============================================================================
# Plotting
# =============================================================================
#
# Visual language for this figure:
#   - A muted, warm off-white panel background (not stark white) with soft
#     dashed gridlines, so the eye rests on the data rather than the axes.
#   - Points colored by a per-cohort accent color, sized by the larger of
#     the two counts (bigger dot = more mutated domain) with a thin white
#     rim so overlapping points stay legible.
#   - A shaded +/-2x-fold concordance band around the y = x line, in
#     addition to the line itself, so near-exact agreement is visible at a
#     glance rather than having to read the stats box.
#   - Deliberately large type throughout (titles, axis labels, tick labels,
#     stats box, domain callouts) so the figure reads well at poster/slide
#     size, not only at print column width.

_PANEL_BG = "#f7f5f0"
_GRID_COLOR = "#c9c2b4"
_ACCENT_PALETTE = ["#1f7a8c", "#e8743b"]     # PAAD-TB, TCGA-LAML (teal / coral)
_LINE_COLOR = "#8a8577"
_TEXT_DARK = "#2b2b2b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
})


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL_BG)
    ax.grid(True, which="major", linestyle="--", linewidth=0.9, color=_GRID_COLOR, alpha=0.6, zorder=0)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.5, color=_GRID_COLOR, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#8a8577")
        ax.spines[spine].set_linewidth(1.1)
    ax.tick_params(axis="both", which="major", labelsize=24, length=7, width=1.4, colors=_TEXT_DARK)


def _declutter_labels(
    fig: plt.Figure,
    ax: plt.Axes,
    text_objs: list,
    iterations: int = 400,
    push_px: float = 2.5,
) -> None:
    """
    Lightweight, dependency-free label de-overlap pass.

    Repeatedly finds pairs of text bounding boxes (in pixel/display space,
    so it works correctly on log-scaled axes) that overlap and nudges each
    member of the pair apart along the line connecting their centers, then
    converts the new pixel position back to data coordinates. This is a
    much smaller version of what the `adjustText` package does, kept
    in-repo so this script has no extra runtime dependency.
    """
    if len(text_objs) < 2:
        return

    fig.canvas.draw()  # required once so get_window_extent() returns real boxes
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    for _ in range(iterations):
        boxes = [t.get_window_extent(renderer=renderer) for t in text_objs]
        moved = False
        for i in range(len(text_objs)):
            for j in range(i + 1, len(text_objs)):
                bi, bj = boxes[i], boxes[j]
                if not bi.overlaps(bj):
                    continue
                moved = True
                cxi, cyi = bi.x0 + bi.width / 2, bi.y0 + bi.height / 2
                cxj, cyj = bj.x0 + bj.width / 2, bj.y0 + bj.height / 2
                dx, dy = cxi - cxj, cyi - cyj
                if dx == 0 and dy == 0:
                    dy = 1.0
                dist = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
                ux, uy = dx / dist, dy / dist

                for k, sign in ((i, 1.0), (j, -1.0)):
                    xd, yd = text_objs[k].get_position()
                    px, py = ax.transData.transform((xd, yd))
                    new_px = (px + sign * ux * push_px, py + sign * uy * push_px)
                    text_objs[k].set_position(tuple(inv.transform(new_px)))

                boxes[i] = text_objs[i].get_window_extent(renderer=renderer)
                boxes[j] = text_objs[j].get_window_extent(renderer=renderer)
        if not moved:
            break


def _add_top_domain_labels(
    fig: plt.Figure,
    ax: plt.Axes,
    common: pd.DataFrame,
    pymut_col: str,
    maftools_col: str,
    color: str,
    top_n: int = 5,
) -> None:
    """
    Label the `top_n` largest domains, then de-overlap the labels and draw a
    thin leader line back to the original point wherever a label had to be
    moved more than a few pixels away from it.
    """
    if not len(common):
        return

    top_domains = common.reindex(
        common[[pymut_col, maftools_col]].max(axis=1).sort_values(ascending=False).index
    ).head(top_n)

    anchors: list[tuple[float, float]] = []
    text_objs = []
    for _, row in top_domains.iterrows():
        x0, y0 = float(row[pymut_col]), float(row[maftools_col])
        anchors.append((x0, y0))
        # Small initial offset (in pixel space) so labels don't start
        # stacked exactly on top of their marker before de-overlapping.
        px, py = ax.transData.transform((x0, y0))
        x_init, y_init = ax.transData.inverted().transform((px + 8, py + 8))
        text_objs.append(
            ax.text(
                x_init, y_init, row["domain_label"],
                fontsize=24, color=_TEXT_DARK, fontweight="medium", zorder=5,
            )
        )

    _declutter_labels(fig, ax, text_objs)

    # Leader lines: only drawn when de-overlapping actually moved a label
    # far enough from its point that the connection is no longer obvious.
    for (x0, y0), text in zip(anchors, text_objs):
        xt, yt = text.get_position()
        p0 = ax.transData.transform((x0, y0))
        p1 = ax.transData.transform((xt, yt))
        pixel_dist = ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5
        if pixel_dist > 14:
            ax.plot(
                [x0, xt], [y0, yt], color="#9a9a9a", linewidth=0.7,
                zorder=4, solid_capstyle="round",
            )


def _scatter_panel(
    ax: plt.Axes,
    merged: pd.DataFrame,
    pymut_col: str,
    maftools_col: str,
    corr: CorrelationStats,
    metric_label: str,
    cohort_label: str,
    point_color: str,
) -> None:
    """Draw one log-log concordance scatter panel (pyMut vs maftools)."""
    common = merged.loc[merged["in_both"], [pymut_col, maftools_col, "domain_label"]].dropna()

    x = common[pymut_col].to_numpy(dtype=float)
    y = common[maftools_col].to_numpy(dtype=float)

    lo = max(1, min(x.min() if len(x) else 1, y.min() if len(y) else 1))
    hi = max(x.max() if len(x) else 1, y.max() if len(y) else 1)
    lo_lim, hi_lim = lo * 0.75, hi * 1.4

    # Shaded 2-fold concordance band around y = x, drawn first (behind data).
    band_x = np.geomspace(lo_lim, hi_lim, 200)
    ax.fill_between(
        band_x, band_x / 2, band_x * 2, color=_LINE_COLOR, alpha=0.12, zorder=1,
        label="±2-fold band",
    )
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color=_LINE_COLOR,
             linestyle="--", linewidth=1.6, zorder=2), #label="y = x (exact match)")

    # Point size scales gently with domain size (largest counts get bigger
    # markers) so heavily-mutated domains stand out visually.
    magnitude = np.maximum(x, y)
    sizes = 60 + 55 * (np.log1p(magnitude) / np.log1p(magnitude.max() if len(magnitude) else 1))

    ax.scatter(
        x, y, s=sizes, color=point_color, alpha=0.82, edgecolor="white",
        linewidth=1.1, zorder=3,
    )

    # Scale/limits/labels must be set BEFORE placing text labels: the
    # de-overlap pass below relies on ax.transData, which depends on the
    # final axis scale and limits.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_ylim(lo_lim, hi_lim)
    ax.set_xlabel(f"{metric_label} — pyMut", fontsize=30, labelpad=12)
    ax.set_ylabel(f"{metric_label} — maftools", fontsize=30, labelpad=12)
    ax.set_title(cohort_label, fontsize=33, fontweight="bold", color=point_color, pad=16, linespacing=1.4)
    _style_axes(ax)

    # Label the handful of domains with the largest counts, for
    # orientation, with automatic de-overlapping (see _add_top_domain_labels).
    _add_top_domain_labels(ax.figure, ax, common, pymut_col, maftools_col, point_color)

    stats_text = (
        f"Pearson r (log₁₀) = {corr.pearson_r_log:.3f}\n"
        f"Spearman ρ = {corr.spearman_rho:.3f}\n"
        f"n domains = {corr.n_domains}\n"
        f"exact match = {corr.n_exact_match}/{corr.n_domains}"
    )
    ax.text(
        0.04, 0.96, stats_text, transform=ax.transAxes, ha="left", va="top",
        fontsize=22, color=_TEXT_DARK, linespacing=1.7, fontweight="medium",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                   edgecolor=point_color, linewidth=1.6, alpha=0.94),
    )


def plot_correlation_grid(
    per_cohort: dict[str, tuple[pd.DataFrame, CorrelationStats, CorrelationStats]],
    save_path_png: Path,
    save_path_pdf: Path,
) -> None:
    """
    Single-row publication figure with all four panels side by side:
      [variants-per-domain, cohort 1] [variants-per-domain, cohort 2]
      [genes-per-domain,    cohort 1] [genes-per-domain,    cohort 2]
    `per_cohort[label] = (merged_df, variants_corr_stats, genes_corr_stats)`
    """
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    cohort_labels = list(per_cohort.keys())
    metrics = [
        ("pymut_n_variants", "maftools_n_variants", "Variants per domain"),
        ("pymut_n_genes", "maftools_n_genes", "Genes per domain"),
    ]
    # One panel per (metric, cohort) combination, all in a single row.
    panels = [
        (cohort_label, metric_idx, pymut_col, maftools_col, metric_label)
        for metric_idx, (pymut_col, maftools_col, metric_label) in enumerate(metrics)
        for cohort_label in cohort_labels
    ]
    n_panels = len(panels)

    fig, axes = plt.subplots(1, n_panels, figsize=(9.5 * n_panels, 11.5))
    fig.patch.set_facecolor("white")
    if n_panels == 1:
        axes = [axes]

    for ax, (cohort_label, metric_idx, pymut_col, maftools_col, metric_label) in zip(axes, panels):
        cohort_idx = cohort_labels.index(cohort_label)
        merged, variants_corr, genes_corr = per_cohort[cohort_label]
        corr = variants_corr if metric_idx == 0 else genes_corr
        color = _ACCENT_PALETTE[cohort_idx % len(_ACCENT_PALETTE)]

        _scatter_panel(
            ax, merged, pymut_col, maftools_col, corr, metric_label,
            f"{cohort_label}\n{metric_label}", color,
        )

    # Single shared legend for the reference line / band, placed once
    # beneath the whole figure to avoid cluttering every panel.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False,
        fontsize=24, bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle(
        "pyMut vs. maftools — Pfam domain-level mutation counts",
        fontsize=45, fontweight="bold", y=1.06, color=_TEXT_DARK,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    fig.savefig(save_path_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(save_path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("Saved figure: %s", save_path_png)
    LOGGER.info("Saved figure: %s", save_path_pdf)


# =============================================================================
# Report
# =============================================================================

def write_report(
    per_cohort: dict[str, tuple[pd.DataFrame, CorrelationStats, CorrelationStats]],
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("Pfam domain analysis: pyMut vs. maftools - correlation report")
    lines.append("=" * 72)
    lines.append("")

    for cohort_label, (merged, variants_corr, genes_corr) in per_cohort.items():
        only_pymut = merged.loc[merged["pymut_n_variants"].notna() & merged["maftools_n_variants"].isna()]
        only_maftools = merged.loc[merged["pymut_n_variants"].isna() & merged["maftools_n_variants"].notna()]

        lines.append(f"Cohort: {cohort_label}")
        lines.append("-" * 72)
        lines.append(f"  Domains found by pyMut only:      {len(only_pymut)}")
        lines.append(f"  Domains found by maftools only:   {len(only_maftools)}")
        lines.append(f"  Domains found by both tools:      {variants_corr.n_domains}")
        lines.append("")
        lines.append("  Variants per domain:")
        lines.append(f"    Pearson r (log10 scale)  = {variants_corr.pearson_r_log:.4f}  (p = {variants_corr.pearson_p_log:.2e})")
        lines.append(f"    Spearman rho             = {variants_corr.spearman_rho:.4f}  (p = {variants_corr.spearman_p:.2e})")
        lines.append(f"    Exact count matches      = {variants_corr.n_exact_match}/{variants_corr.n_domains}")
        lines.append("")
        lines.append("  Genes per domain:")
        lines.append(f"    Pearson r (log10 scale)  = {genes_corr.pearson_r_log:.4f}  (p = {genes_corr.pearson_p_log:.2e})")
        lines.append(f"    Spearman rho             = {genes_corr.spearman_rho:.4f}  (p = {genes_corr.spearman_p:.2e})")
        lines.append(f"    Exact count matches      = {genes_corr.n_exact_match}/{genes_corr.n_domains}")
        lines.append("")

        if len(only_pymut):
            lines.append(f"  Domains only in pyMut ({len(only_pymut)}): "
                          + ", ".join(only_pymut["domain_label"].head(20)))
            lines.append("")
        if len(only_maftools):
            lines.append(f"  Domains only in maftools ({len(only_maftools)}): "
                          + ", ".join(only_maftools["domain_label"].head(20)))
            lines.append("")

    lines.append(
        "Note: any small discrepancies are most likely explained by differing "
        "default variant de-duplication behaviour between pyMut's read_maf(...,"
        " consolidate_variants=False) and maftools::read.maf() - see the header "
        "comment of run_maftools_pfam.R for details."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Saved report: %s", report_path)


# =============================================================================
# Main
# =============================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-dir", type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "data",
        help="Directory containing the cohort MAF files (default: "
             "<project_root>/data, resolved from this script's location)",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Output directory for figures, tables and the report (default: "
             "results/ next to this script). Must also contain the CSVs "
             "produced by run_maftools_pfam.R.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    cohorts = build_default_cohorts(args.base_dir, results_dir)

    per_cohort: dict[str, tuple[pd.DataFrame, CorrelationStats, CorrelationStats]] = {}

    for cohort in cohorts:
        try:
            pymut_summary = run_pymut_pfam_summary(cohort)
            maftools_summary = load_maftools_domain_summary(cohort.maftools_domain_csv, cohort.display_name)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            LOGGER.error(str(exc))
            return 1

        merged = merge_domain_summaries(pymut_summary, maftools_summary)
        merged_csv = results_dir / f"{cohort.key}_pfam_vs_maftools_merged.csv"
        merged.to_csv(merged_csv, index=False)
        LOGGER.info("[%s] Saved merged per-domain table: %s", cohort.display_name, merged_csv)

        variants_corr = compute_correlation(merged, "pymut_n_variants", "maftools_n_variants")
        genes_corr = compute_correlation(merged, "pymut_n_genes", "maftools_n_genes")

        LOGGER.info(
            "[%s] variants/domain: Pearson r(log)=%.3f, Spearman rho=%.3f (n=%d)",
            cohort.display_name, variants_corr.pearson_r_log, variants_corr.spearman_rho, variants_corr.n_domains,
        )

        per_cohort[cohort.display_name] = (merged, variants_corr, genes_corr)

    plot_correlation_grid(
        per_cohort,
        save_path_png=results_dir / "pfam_vs_maftools_correlation.png",
        save_path_pdf=results_dir / "pfam_vs_maftools_correlation.pdf",
    )
    write_report(per_cohort, results_dir / "correlation_report.txt")

    LOGGER.info("Done. All outputs written to: %s", results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
