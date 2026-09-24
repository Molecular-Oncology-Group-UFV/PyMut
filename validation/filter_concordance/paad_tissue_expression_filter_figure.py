#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAAD tissue-expression filter — figure generation
===================================================

This script reproduces, in isolation, the figure that illustrates the effect
of PyMut's tissue-expression filter (`filter_by_tissue_expression`) on the
PAAD (Pancreatic Adenocarcinoma) MAF file used in the PyMut validation
notebook ("Figure S3A").

What the script does
---------------------
1. Loads the PAAD MAF file with PyMut (`read_maf`), keeping one row per
   original record (`consolidate_variants=False`).
2. Applies PyMut's tissue-expression filter
   (`filter_by_tissue_expression`) with `keep_expressed=True`: a mutation
   is kept only if its gene has a PAAD expression value that is greater
   than or equal to `PAAD_EXPRESSION_THRESHOLD` in PyMut's internal
   expression resource. Genes that are not present in that resource are
   also removed, since they cannot demonstrate sufficient expression.
3. Counts how many mutations are retained vs. removed, and the associated
   percentage.
4. Draws a two-bar figure (retained vs. removed) with counts and
   percentages annotated on top of each bar, and exports it as both PNG
   (300 dpi) and PDF.
5. Saves a small CSV summary with the same numbers, for traceability.

Requirements
------------
- PyMut must be installed in the active Python environment
  (`pip install PyMut-Library`). The script imports `read_maf` from the `pyMut` package.
- matplotlib, numpy and pandas.

Usage
-----
    python paad_tissue_expression_filter_figure.py

All paths and the expression threshold are defined as constants right
below the imports, so they can be edited without touching the rest of
the script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# PyMut must be installed in the environment running this script.
from pyMut import read_maf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Folder containing this script (validation/filter_concordance/) and the
# project root (two levels up). Paths are resolved from here, so the
# script works no matter which directory it is launched from.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Folder where the figure and its summary CSV will be written.
RESULTS_DIR = SCRIPT_DIR / "results"

# PAAD MAF file used as input for the tissue-expression filter.
PAAD_MAF_PATH = PROJECT_ROOT / "data" / "PAAD-TB.final_analysis.maf"

# Genome assembly used to read the MAF (must match the file's coordinates).
PAAD_MAF_ASSEMBLY = "37"

# Minimum PAAD expression value (from PyMut's internal expression
# resource) required to keep a mutation. This is an analytical choice
# and should be reported alongside the figure.
PAAD_EXPRESSION_THRESHOLD = 0.5

# Bar colors: [retained, removed].
BAR_COLORS = ["#457b9d", "#e76f51"]


def load_paad_mutations(maf_path: Path, assembly: str):
    """Read the PAAD MAF file with PyMut.

    Parameters
    ----------
    maf_path : Path
        Path to the PAAD MAF file.
    assembly : str
        Genome assembly matching the MAF coordinates (e.g. "37").

    Returns
    -------
    pyMut.PyMutation
        PyMut mutation object with one row per original MAF record
        (`consolidate_variants=False`, i.e. variants are not merged).

    Raises
    ------
    FileNotFoundError
        If `maf_path` does not exist.
    """
    if not maf_path.exists():
        raise FileNotFoundError(f"PAAD MAF file not found: {maf_path}")

    return read_maf(
        maf_path,
        assembly=assembly,
        consolidate_variants=False,
    )


def apply_expression_filter(paad_mutations, threshold: float):
    """Apply PyMut's PAAD tissue-expression filter.

    A mutation is kept only if its gene has a PAAD expression value
    greater than or equal to ``threshold`` in PyMut's internal
    expression resource (`keep_expressed=True`). Genes absent from that
    resource are removed as well, since expression cannot be confirmed
    for them.

    Parameters
    ----------
    paad_mutations : pyMut.PyMutation
        Mutation object obtained from :func:`load_paad_mutations`.
    threshold : float
        Minimum PAAD expression value required to keep a mutation.

    Returns
    -------
    dict
        Dictionary with the initial/retained/removed counts, the
        removed percentage, and the filtered PyMut object.
    """
    initial_count = len(paad_mutations.data)

    filtered_mutations = paad_mutations.filter_by_tissue_expression(
        tissues=[("PAAD", threshold)],
        keep_expressed=True,
    )

    retained_count = len(filtered_mutations.data)
    removed_count = initial_count - retained_count
    removed_percentage = (
        100 * removed_count / initial_count if initial_count else np.nan
    )

    return {
        "initial_count": initial_count,
        "retained_count": retained_count,
        "removed_count": removed_count,
        "removed_percentage": removed_percentage,
        "filtered_mutations": filtered_mutations,
    }


def save_summary_csv(filter_results: dict, threshold: float, output_dir: Path) -> Path:
    """Write a small CSV summarizing the filter's effect.

    Parameters
    ----------
    filter_results : dict
        Output of :func:`apply_expression_filter`.
    threshold : float
        Expression threshold used, stored for traceability.
    output_dir : Path
        Directory where the CSV file will be written.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    initial_count = filter_results["initial_count"]
    retained_count = filter_results["retained_count"]
    removed_count = filter_results["removed_count"]

    summary = pd.DataFrame(
        {
            "status": ["Expressed gene", "Low/no expression"],
            "retained": [retained_count, 0],
            "removed": [0, removed_count],
            "count": [retained_count, removed_count],
            "percentage_of_initial": [
                100 * retained_count / initial_count if initial_count else np.nan,
                filter_results["removed_percentage"],
            ],
            "expression_threshold": [threshold] * 2,
        }
    )

    csv_path = output_dir / "Figure_S4_PAAD_expression_filter_summary.csv"
    summary.to_csv(csv_path, index=False)
    return csv_path


def plot_expression_filter_figure(
    filter_results: dict,
    threshold: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Draw and save the PAAD tissue-expression filter figure.

    The figure shows two bars (retained vs. removed mutations), each
    annotated with its count and the percentage relative to the initial
    number of mutations, plus a caption stating the exclusion
    percentage and the threshold used.

    Parameters
    ----------
    filter_results : dict
        Output of :func:`apply_expression_filter`.
    threshold : float
        Expression threshold used, shown in the figure caption.
    output_dir : Path
        Directory where the PNG and PDF files will be written.

    Returns
    -------
    tuple[Path, Path]
        Paths to the saved PNG and PDF files, respectively.
    """
    initial_count = filter_results["initial_count"]
    retained_count = filter_results["retained_count"]
    removed_count = filter_results["removed_count"]
    removed_percentage = filter_results["removed_percentage"]

    labels = ["Retained\n(expressed)", "Removed\n(low/no expression)"]
    counts = [retained_count, removed_count]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, counts, color=BAR_COLORS, width=0.62)

    ax.set_title(
        "PAAD: tissue-expression filtering", fontsize=18, fontweight="bold", pad=14
    )
    ax.set_ylabel("Number of mutations", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count in zip(bars, counts):
        percentage = 100 * count / initial_count if initial_count else np.nan
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    fig.text(
        0.5,
        0.01,
        f"Excluded below PAAD expression threshold {threshold:g}: "
        f"{removed_percentage:.2f}%",
        ha="center",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    png_path = output_dir / "Figure_S4_PAAD_expression_filter.png"
    pdf_path = output_dir / "Figure_S4_PAAD_expression_filter.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def main() -> None:
    """Run the full PAAD tissue-expression filter figure pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading PAAD MAF from: {PAAD_MAF_PATH}")
    paad_mutations = load_paad_mutations(PAAD_MAF_PATH, PAAD_MAF_ASSEMBLY)
    print(f"Initial mutations: {len(paad_mutations.data):,}")

    print(
        f"Applying PAAD tissue-expression filter "
        f"(threshold = {PAAD_EXPRESSION_THRESHOLD:g})..."
    )
    filter_results = apply_expression_filter(paad_mutations, PAAD_EXPRESSION_THRESHOLD)

    print(f"Retained mutations: {filter_results['retained_count']:,}")
    print(f"Removed mutations: {filter_results['removed_count']:,}")
    print(f"Removed percentage: {filter_results['removed_percentage']:.2f}%")

    csv_path = save_summary_csv(filter_results, PAAD_EXPRESSION_THRESHOLD, RESULTS_DIR)
    png_path, pdf_path = plot_expression_filter_figure(
        filter_results, PAAD_EXPRESSION_THRESHOLD, RESULTS_DIR
    )

    print("\nFiles written:")
    print(f"- {csv_path}")
    print(f"- {png_path}")
    print(f"- {pdf_path}")


if __name__ == "__main__":
    main()
