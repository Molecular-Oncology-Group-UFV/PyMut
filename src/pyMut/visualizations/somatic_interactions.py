"""
Module for creating somatic interactions plots.

This module contains functions for creating heatmaps that visualize
co-occurrence and mutual exclusivity patterns between mutated genes
using Fisher's exact test.

Somatic interactions plots are essential in cancer genomics for:
- Identifying gene pairs that are mutated together (co-occurrence)
- Detecting mutually exclusive mutations
- Understanding functional relationships between genes
- Discovering potential synthetic lethal interactions

Main functions:
- _build_mutation_matrix(): Creates binary mutation matrix (samples x genes)
- _prepare_top_genes_matrix(): Shared data-prep step (filtering + top-k gene selection)
- _build_interactions_table(): Pairwise Fisher's exact test table, with FDR correction
- _compute_fisher_interactions(): Gene x gene matrices derived from the table above
- _signed_log10_matrix(): Signed -log10(p-value) matrix used for heatmap coloring
- _get_somatic_interactions_table(): Public-facing table builder (used by PyMutation.somatic_interactions_table)
- _create_somatic_interactions_plot(): Main function to create the heatmap

Multiple-testing correction:
Every pairwise comparison table (whether printed directly via
`_get_somatic_interactions_table` / `PyMutation.somatic_interactions_table`, or
computed internally for the heatmap) includes a False Discovery Rate (FDR,
Benjamini-Hochberg by default) correction. The correction is always computed
over exactly the C(top_genes, 2) pairwise tests that were actually performed,
so it automatically adapts to whatever `top_genes` (top-k) value is selected.
The heatmap itself only *displays* the adjusted p-value when the caller opts
in via `fdr_correction=True`; by default it keeps showing the raw p-value to
preserve backward-compatible behavior.
"""

import itertools
import logging
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from ..core import PyMutation

from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_SOMATIC_FIGSIZE = (12, 10)
DEFAULT_TOP_GENES = 25


def _build_mutation_matrix(data: pd.DataFrame,
                          gene_column: str = "Hugo_Symbol",
                          sample_column: str = "Tumor_Sample_Barcode",
                          all_samples: np.ndarray = None) -> pd.DataFrame:
    """
    Build binary mutation matrix (samples x genes).
    
    IMPORTANT: This function includes ALL samples from the dataset,
    even those without mutations in the selected genes. This is critical
    for correct Fisher's exact test calculations.
    
    Creates a matrix where:
    - Rows = ALL samples (including those with no mutations)
    - Columns = genes
    - Values = 1 if sample has mutation in gene, 0 otherwise
    
    Args:
        data: DataFrame with mutation data
        gene_column: Name of column containing gene names
        sample_column: Name of column containing sample IDs
        all_samples: Array of ALL sample IDs (including those without mutations
                    after filtering). Must be provided to ensure correct sample count.
        
    Returns:
        DataFrame: Binary mutation matrix (samples x genes)
    """
    # Validate columns exist
    if gene_column not in data.columns:
        raise ValueError(f"Column '{gene_column}' not found in data")
    if sample_column not in data.columns:
        raise ValueError(f"Column '{sample_column}' not found in data")
    
    # Remove rows with missing gene or sample values
    valid_data = data[[sample_column, gene_column]].dropna()
    
    # Create binary matrix: 1 if sample has mutation in gene
    # Using pivot_table with aggfunc='size' counts mutations, then clip to binary
    matrix = valid_data.groupby([sample_column, gene_column]).size().unstack(fill_value=0)
    matrix = (matrix > 0).astype(int)
    
    if all_samples is not None:
        # Add rows for samples that have no mutations in any of the genes
        # This is CRITICAL for correct Fisher's exact test calculations
        missing_samples = set(all_samples) - set(matrix.index)
        if missing_samples:
            # Sort missing samples for deterministic order (fixes non-reproducible results)
            missing_samples_sorted = sorted(list(missing_samples))
            missing_df = pd.DataFrame(0, index=missing_samples_sorted, columns=matrix.columns)
            matrix = pd.concat([matrix, missing_df], axis=0)
    
    return matrix


def _build_interactions_table(mutation_matrix: pd.DataFrame,
                              genes: List[str],
                              fdr_method: str = "fdr_bh") -> pd.DataFrame:
    """
    Build the tidy (long-format) pairwise Fisher's exact test table for a
    set of genes, including False Discovery Rate (FDR) correction.

    For each pair of genes:
    - Constructs the 2x2 contingency table (both mutated, gene1 only,
      gene2 only, neither)
    - Runs Fisher's exact test to get the odds ratio and raw p-value
    - Clips raw p-values at 1e-10 (avoids issues with log10(0) downstream)

    The FDR correction is computed ONCE, across exactly the C(len(genes), 2)
    pairwise tests performed here. This means the strength of the correction
    automatically follows whatever top-k gene set (`genes`) was selected
    upstream: a larger `top_genes` -> more pairwise tests -> stronger
    correction, and vice versa.

    This function is the single source of truth for the pairwise statistics:
    both the printable table (`_get_somatic_interactions_table`) and the
    heatmap matrices (`_compute_fisher_interactions`) are derived from it, so
    the two are always guaranteed to be consistent with each other.

    Args:
        mutation_matrix: Binary mutation matrix (samples x genes).
        genes: List of gene names to analyze (defines the top-k gene set;
            also determines the number of pairwise tests used for FDR).
        fdr_method: Multiple-testing correction method forwarded to
            `statsmodels.stats.multitest.multipletests` (default: "fdr_bh",
            i.e. Benjamini-Hochberg). Any method supported by that function
            can be used (e.g. "bonferroni", "holm").

    Returns:
        pd.DataFrame: One row per gene pair, sorted by FDR-adjusted p-value
        ascending (ties broken by raw p-value ascending). Columns:
            - Gene 1, Gene 2
            - Altered samples gene 1, Altered samples gene 2
            - Both altered, Only gene 1 altered, Only gene 2 altered, Neither altered
            - Odds ratio
            - P-value: raw p-value from Fisher's exact test (clipped at 1e-10)
            - P-value adjusted (FDR): p-value corrected via `fdr_method`
              across the C(len(genes), 2) tests performed here
            - Significant (FDR < 0.05): boolean convenience flag
            - Association: "co-occurrence" / "mutual_exclusivity" / "neutral"
    """
    matrix_vals = mutation_matrix[genes].values  # shape: (n_samples, n_genes)
    records = []

    for i, j in itertools.combinations(range(len(genes)), 2):
        vec1 = matrix_vals[:, i]
        vec2 = matrix_vals[:, j]

        both_mut = int(np.sum((vec1 == 1) & (vec2 == 1)))
        gene1_only = int(np.sum((vec1 == 1) & (vec2 == 0)))
        gene2_only = int(np.sum((vec1 == 0) & (vec2 == 1)))
        neither = int(np.sum((vec1 == 0) & (vec2 == 0)))

        contingency = [[both_mut, gene1_only], [gene2_only, neither]]
        try:
            or_val, p_val = fisher_exact(contingency)
            if p_val < 1e-10:
                p_val = 1e-10
        except Exception as e:
            logger.warning(f"Fisher test failed for {genes[i]}-{genes[j]}: {e}")
            or_val, p_val = 1.0, 1.0

        records.append({
            "Gene 1": genes[i],
            "Gene 2": genes[j],
            "Altered samples gene 1": int(vec1.sum()),
            "Altered samples gene 2": int(vec2.sum()),
            "Both altered": both_mut,
            "Only gene 1 altered": gene1_only,
            "Only gene 2 altered": gene2_only,
            "Neither altered": neither,
            "Odds ratio": or_val,
            "P-value": p_val,
        })

    interactions_table = pd.DataFrame.from_records(records)

    # --- FDR correction across the C(len(genes), 2) tests performed above ---
    if not interactions_table.empty:
        reject, p_adj, _, _ = multipletests(
            interactions_table["P-value"], method=fdr_method
        )
        interactions_table["P-value adjusted (FDR)"] = p_adj
        interactions_table["Significant (FDR < 0.05)"] = reject
    else:
        interactions_table["P-value adjusted (FDR)"] = pd.Series(dtype=float)
        interactions_table["Significant (FDR < 0.05)"] = pd.Series(dtype=bool)

    interactions_table["Association"] = np.select(
        [
            interactions_table["Odds ratio"] > 1,
            interactions_table["Odds ratio"] < 1,
        ],
        ["co-occurrence", "mutual_exclusivity"],
        default="neutral",
    )

    interactions_table = interactions_table.sort_values(
        ["P-value adjusted (FDR)", "P-value", "Odds ratio"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return interactions_table


def _compute_fisher_interactions(mutation_matrix: pd.DataFrame,
                                 genes: List[str],
                                 fdr_method: str = "fdr_bh"
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute Fisher's exact test for all gene pairs and return both the
    square (genes x genes) matrices needed for the heatmap and the tidy
    pairwise table (with FDR correction) needed for reporting.

    Internally delegates the per-pair calculation to `_build_interactions_table`,
    so the heatmap and the exported table are always built from the exact
    same numbers.

    Results interpretation:
    - Odds ratio > 1: Co-occurrence (genes mutated together more than expected)
    - Odds ratio < 1: Mutual exclusivity (genes rarely mutated together)
    - Raw P-value < 0.05: Nominally significant interaction
    - FDR-adjusted P-value < 0.05: Significant after correcting for the
      C(len(genes), 2) pairwise comparisons performed on this gene set

    Args:
        mutation_matrix: Binary mutation matrix (samples x genes)
        genes: List of gene names to analyze (in desired order). This is the
            top-k gene set: the number of pairwise tests -- and therefore the
            strength of the FDR correction -- depends directly on len(genes).
        fdr_method: Multiple-testing correction method forwarded to
            `statsmodels.stats.multitest.multipletests` (default: "fdr_bh").

    Returns:
        Tuple of four DataFrames:
        - pvalues: Raw p-values (genes x genes, symmetric, diagonal = NaN)
        - pvalues_adj: FDR-adjusted p-values (genes x genes, symmetric, diagonal = NaN)
        - oddsratios: Odds ratios (genes x genes, symmetric, diagonal = NaN)
        - interactions_table: Tidy table with one row per gene pair, including
          both raw and FDR-adjusted p-values (see `_build_interactions_table`)
    """
    interactions_table = _build_interactions_table(
        mutation_matrix, genes, fdr_method=fdr_method
    )

    pvalues = pd.DataFrame(np.nan, index=genes, columns=genes)
    pvalues_adj = pd.DataFrame(np.nan, index=genes, columns=genes)
    oddsratios = pd.DataFrame(np.nan, index=genes, columns=genes)

    gene_pos = {gene: idx for idx, gene in enumerate(genes)}
    for _, row in interactions_table.iterrows():
        i = gene_pos[row["Gene 1"]]
        j = gene_pos[row["Gene 2"]]
        pvalues.iloc[i, j] = pvalues.iloc[j, i] = row["P-value"]
        pvalues_adj.iloc[i, j] = pvalues_adj.iloc[j, i] = row["P-value adjusted (FDR)"]
        oddsratios.iloc[i, j] = oddsratios.iloc[j, i] = row["Odds ratio"]

    return pvalues, pvalues_adj, oddsratios, interactions_table


def _signed_log10_matrix(pvalues: pd.DataFrame, oddsratios: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a genes x genes p-value matrix into a signed -log10(p-value)
    matrix used to color the somatic interactions heatmap.

    Sign convention:
        - Positive values -> co-occurrence (odds ratio > 1)
        - Negative values -> mutual exclusivity (odds ratio < 1)
        - NaN on the diagonal (gene vs itself)

    Args:
        pvalues: Square DataFrame of p-values (either raw or FDR-adjusted),
            genes x genes, symmetric. Values are clipped at 1e-10 before
            taking the log to avoid -inf.
        oddsratios: Square DataFrame of odds ratios, same shape/index as `pvalues`.

    Returns:
        pd.DataFrame: Signed -log10(p-value) matrix, same shape/index as inputs.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        p_clipped = pvalues.clip(lower=1e-10)
        signed = np.where(
            oddsratios.to_numpy() > 1,
            -np.log10(p_clipped.to_numpy()),
            np.log10(p_clipped.to_numpy()),
        )

    signed = np.array(signed, copy=True)  # ensure a writeable array (np.where can return a read-only view)
    np.fill_diagonal(signed, np.nan)
    return pd.DataFrame(signed, index=pvalues.index, columns=pvalues.columns)


def _prepare_top_genes_matrix(py_mut: 'PyMutation',
                              top_genes: int,
                              gene_column: str,
                              sample_column: str
                              ) -> Tuple[pd.DataFrame, List[str], pd.Series, int, int]:
    """
    Shared data-preparation step used by both the somatic interactions
    heatmap and the somatic interactions table.

    Filters out silent and non-coding variants, ranks genes by number of
    altered samples, selects the top-k genes, and builds the binary mutation
    matrix restricted to that gene set (including ALL samples, mutated or
    not, which is required for correct Fisher's exact test calculations).

    Centralizing this logic guarantees that the heatmap and the table use
    IDENTICAL gene filtering and gene selection for the same `top_genes`
    value.

    Args:
        py_mut: PyMutation object containing mutation data.
        top_genes: Number of most frequently mutated genes to include.
        gene_column: Name of the column containing gene symbols.
        sample_column: Name of the column containing sample identifiers.

    Returns:
        Tuple containing:
            - mutation_matrix_top: Binary mutation matrix (samples x top genes)
            - top_gene_list: Selected genes, in display order (AlteredSamples
              desc, gene name asc)
            - gene_counts: Series mapping gene -> number of altered samples
            - n_variants: Number of variants retained after filtering
            - n_samples: Total number of samples considered

    Raises:
        ValueError: If fewer than 2 genes remain mutated in the dataset.
    """
    # 1. Use ALL variants from py_mut.data
    # NOTE: We exclude Silent variants and non-coding variants
    # to focus on functionally relevant mutations.
    # PyMut's read_maf() keeps all variants, so we must filter here.
    data_for_analysis = py_mut.data.copy()

    exclude_variants = [
        "SILENT",
        "INTRON",
        "IGR",
        "3'UTR",
        "5'UTR",
        "3'FLANK",
        "5'FLANK",
        "RNA",
    ]
    if "Variant_Classification" in data_for_analysis.columns:
        vc_upper = data_for_analysis["Variant_Classification"].str.upper()
        data_for_analysis = data_for_analysis[~vc_upper.isin(exclude_variants)]

    n_variants = len(data_for_analysis)

    # Get ALL samples from the original unfiltered data
    # This is critical for correct Fisher's exact test calculations
    all_samples = py_mut.data[sample_column].dropna().unique()
    n_samples = len(all_samples)

    # 2. Calculate gene summary with proper tie-breaking
    # Use a TWO-PHASE sorting approach:
    #   Phase 1 (SELECTION): Sort by AlteredSamples desc, then total desc to SELECT top N genes
    #   Phase 2 (DISPLAY): Resort selected genes by AlteredSamples desc, gene name asc for display
    gene_summary = (
        data_for_analysis.groupby(gene_column)
        .agg(
            {
                sample_column: "nunique",  # AlteredSamples (unique samples)
                gene_column: "count",  # total (total events)
            }
        )
        .rename(columns={sample_column: "AlteredSamples", gene_column: "total"})
        .reset_index()
    )

    gene_summary_for_selection = gene_summary.sort_values(
        by=["AlteredSamples", "total", gene_column], ascending=[False, False, True]
    )
    top_gene_list_unsorted = (
        gene_summary_for_selection[gene_column].head(top_genes).tolist()
    )

    top_genes_df = gene_summary[gene_summary[gene_column].isin(top_gene_list_unsorted)]
    top_genes_sorted = top_genes_df.sort_values(
        by=["AlteredSamples", gene_column], ascending=[False, True]
    )
    top_gene_list = top_genes_sorted[gene_column].tolist()
    gene_counts = gene_summary.set_index(gene_column)["AlteredSamples"]

    if len(top_gene_list) < 2:
        raise ValueError("Need at least 2 mutated genes to compute interactions")

    # 3. Build mutation matrix (with ALL samples from original data)
    mutation_matrix = _build_mutation_matrix(
        data_for_analysis, gene_column, sample_column, all_samples=all_samples
    )
    mutation_matrix_top = mutation_matrix[top_gene_list]

    return mutation_matrix_top, top_gene_list, gene_counts, n_variants, n_samples


def _get_somatic_interactions_table(py_mut: 'PyMutation',
                                    top_genes: int = DEFAULT_TOP_GENES,
                                    gene_column: str = "Hugo_Symbol",
                                    sample_column: str = "Tumor_Sample_Barcode",
                                    fdr_method: str = "fdr_bh") -> pd.DataFrame:
    """
    Build the somatic interactions table: pairwise Fisher's exact test
    results for the top-k mutated genes, including FDR-adjusted p-values.

    Uses `_prepare_top_genes_matrix` for gene filtering/selection, so the
    table is always consistent with what `_create_somatic_interactions_plot`
    displays for the same `top_genes` value. The FDR correction is
    recomputed over the C(top_genes, 2) pairwise tests actually performed,
    so it automatically adapts whenever `top_genes` changes.

    Args:
        py_mut: PyMutation object containing mutation data.
        top_genes: Number of most frequently mutated genes to include (default: 25).
        gene_column: Name of column containing gene symbols (default: "Hugo_Symbol").
        sample_column: Name of column containing sample identifiers
            (default: "Tumor_Sample_Barcode").
        fdr_method: Multiple-testing correction method forwarded to
            `statsmodels.stats.multitest.multipletests` (default: "fdr_bh",
            Benjamini-Hochberg).

    Returns:
        pd.DataFrame: One row per gene pair, sorted by FDR-adjusted p-value
        ascending. See `_build_interactions_table` for the full column
        description.

    Example:
        >>> table = _get_somatic_interactions_table(py_mut, top_genes=25)
        >>> print(table.head())
    """
    start_time = time.time()

    mutation_matrix_top, top_gene_list, _gene_counts, n_variants, n_samples = (
        _prepare_top_genes_matrix(py_mut, top_genes, gene_column, sample_column)
    )

    interactions_table = _build_interactions_table(
        mutation_matrix_top, top_gene_list, fdr_method=fdr_method
    )

    elapsed_time = time.time() - start_time
    logger.info(
        f"Somatic interactions table: {n_variants:,} variants, {n_samples} samples, "
        f"{len(top_gene_list)} genes, {len(interactions_table)} pairwise tests "
        f"in {elapsed_time:.2f}s"
    )

    return interactions_table


def _create_somatic_interactions_plot(py_mut: 'PyMutation',
                                     top_genes: int = DEFAULT_TOP_GENES,
                                     gene_column: str = "Hugo_Symbol",
                                     sample_column: str = "Tumor_Sample_Barcode",
                                     figsize: Tuple[int, int] = DEFAULT_SOMATIC_FIGSIZE,
                                     title: Optional[str] = None,
                                     vmin: float = -3.0,
                                     vmax: float = 3.0,
                                     pvalue: Tuple[float, float] = (0.05, 0.1),
                                     show_counts: bool = True,
                                     fdr_correction: bool = False,
                                     fdr_method: str = "fdr_bh") -> Figure:
    """
    Create somatic interactions heatmap.
    
    Visualizes co-occurrence and mutual exclusivity patterns between top mutated genes
    using Fisher's exact test. The heatmap shows:
    - Brown/orange: Co-occurrence (genes mutated together)
    - Blue/green: Mutual exclusivity (genes rarely mutated together)
    - Asterisks (*): Highly significant interactions (p < 0.1)
    - Dots (·): Significant interactions (p < 0.05)
    
    Args:
        py_mut: PyMutation object containing mutation data
        top_genes: Number of top mutated genes to include
        gene_column: Name of column containing gene names
        sample_column: Name of column containing sample IDs
        figsize: Figure size (width, height) in inches
        title: Plot title (default: "Somatic Interactions")
        vmin: Minimum value for color scale (default: -3)
        vmax: Maximum value for color scale (default: 3)
        pvalue: Tuple of p-value thresholds (upper, lower) for significance markers.
            These thresholds are always applied to whichever p-value is being
            displayed (raw or FDR-adjusted, depending on `fdr_correction`).
        show_counts: Whether to show sample counts for each gene in labels
        fdr_correction: If False (default), the heatmap colors and significance
            markers are based on the RAW Fisher's exact test p-value, exactly
            as before. If True, both are based on the FDR-adjusted p-value
            (Benjamini-Hochberg by default) computed across the
            C(top_genes, 2) pairwise tests performed on this gene set. The
            colorbar label, tick labels and legend are updated automatically
            to reflect which p-value is being shown.
        fdr_method: Multiple-testing correction method forwarded to
            `statsmodels.stats.multitest.multipletests` (default: "fdr_bh").
            Only used when `fdr_correction=True`.
        
    Returns:
        matplotlib.figure.Figure: The somatic interactions heatmap
        
    Example:
        >>> fig = py_mut.somatic_interactions(top_genes=25)
        >>> py_mut.save_figure(fig, "somatic_interactions.png")

        >>> # Color/mark significance using the FDR-adjusted p-value instead
        >>> fig = py_mut.somatic_interactions(top_genes=25, fdr_correction=True)
    """
    start_time = time.time()

    # 1-4. Filter variants, select top-k genes and build the mutation matrix.
    # Shared with `_get_somatic_interactions_table` so both stay consistent.
    mutation_matrix_top, top_gene_list, gene_counts, n_variants, n_samples = (
        _prepare_top_genes_matrix(py_mut, top_genes, gene_column, sample_column)
    )

    # Set title (note whether FDR correction is being displayed)
    if title is None:
        title = f"Somatic Interactions (Top {top_genes} Mutated Genes)"
        if fdr_correction:
            title += " [FDR-corrected]"

    # Create gene labels with counts (format: "GENE [count]")
    gene_labels = {}
    if show_counts:
        for gene in top_gene_list:
            count = int(gene_counts[gene])
            gene_labels[gene] = f"{gene} [{count}]"
    else:
        gene_labels = {gene: gene for gene in top_gene_list}

    # 5. Compute Fisher's exact test for all gene pairs (raw + FDR-adjusted)
    pvalues, pvalues_adj, oddsratios, _interactions_table = _compute_fisher_interactions(
        mutation_matrix_top, top_gene_list, fdr_method=fdr_method
    )

    # Select which p-value matrix drives the heatmap coloring and the
    # significance markers, based on the `fdr_correction` flag.
    pvalues_for_plot = pvalues_adj if fdr_correction else pvalues
    pvalue_label = "FDR-adjusted P-value" if fdr_correction else "P-value"
    significance_prefix = "FDR" if fdr_correction else "P"

    # Build the signed -log10(p-value) matrix used for coloring, from
    # whichever p-value matrix was selected above.
    results = _signed_log10_matrix(pvalues_for_plot, oddsratios)

    # 6. Configure visual layout:
    #
    # KEY INSIGHT: Use the following axis ordering:
    #   - X-axis (columns): most → least mutated (left → right)
    #   - Y-axis (rows): least → most mutated (top → bottom)
    #
    # This means rows are REVERSED from columns!
    # The diagonal (same gene vs same gene) becomes the ANTIDIAGONAL.
    # Therefore, we must mask relative to the antidiagonal, not the main diagonal.

    # --- ORDER AXES FOR VISUALIZATION ---
    cols_order = top_gene_list  # most→least (X, left→right)
    rows_order = list(reversed(top_gene_list))  # least→most (Y, top→bottom)

    # Reindex matrices for visualization
    results_display = results.loc[rows_order, cols_order].copy()
    pvalues_display = pvalues_for_plot.loc[rows_order, cols_order].copy()

    # 7. Mask relative to ANTIDIAGONAL (to show upper-left triangle)
    # The antidiagonal runs from top-right to bottom-left
    # We want to hide cells to the RIGHT of the antidiagonal
    n = results_display.shape[0]
    row_idx = np.arange(n)[:, None]
    col_idx = np.arange(n)[None, :]
    mask = col_idx > (n - 1 - row_idx)  # True = hide cells to the right of antidiagonal

    # 8. Create figure and heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Apply gene labels with counts to results matrix
    results_labeled = results_display.copy()
    results_labeled.index = [gene_labels[g] for g in results_display.index]
    results_labeled.columns = [gene_labels[g] for g in results_display.columns]

    # Create heatmap with improved styling
    sns.heatmap(
        results_labeled,
        mask=mask,
        cmap="BrBG",
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={
            "label": f"-log10({pvalue_label})",
            "orientation": "vertical",
            "ticks": [-3, -2, -1, 0, 1, 2, 3],
        },
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=True,
        yticklabels=True,
        ax=ax,
    )

    # Move X-axis labels to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Rotate labels for better readability
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=90, ha="left", fontsize=9, style="italic"
    )
    ax.set_yticklabels(
        ax.get_yticklabels(), rotation=0, ha="right", fontsize=9, style="italic"
    )

    # 9. Add significance markers
    for i_display, gene1 in enumerate(results_display.index):
        for j, gene2 in enumerate(results_display.columns):
            # Skip if masked (lower triangle or diagonal)
            if mask[i_display, j]:
                continue

            p_val = pvalues_display.loc[gene1, gene2]

            # Use p-value thresholds where:
            # - min(pvalue) = 0.05 gets asterisk (*)
            # - max(pvalue) = 0.1 gets dot (·)

            # Check thresholds from most to least stringent
            if p_val < min(pvalue):
                # Asterisk for highly significant (p < 0.05)
                ax.scatter(
                    j + 0.5, i_display + 0.5, marker="*", s=200, c="black", zorder=10
                )
            elif p_val < max(pvalue):
                # Dot for significant (p < 0.1)
                ax.scatter(
                    j + 0.5, i_display + 0.5, marker=".", s=150, c="black", zorder=10
                )

    # 10. Add legend for significance markers
    # Create a proper legend with actual marker symbols
    # Note: max(pvalue) is less stringent (0.05), min(pvalue) is more stringent (0.1)
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label=f"{significance_prefix} < {min(pvalue)}",
            markerfacecolor="black",
            markersize=12,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{significance_prefix} < {max(pvalue)}",
            markerfacecolor="black",
            markersize=8,
            linestyle="None",
        ),
    ]

    # Add legend in lower right corner
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.9,
        fontsize=10,
        edgecolor="black",
        facecolor="white",
    )

    # 11. Improve colorbar styling
    cbar = fig.axes[-1]
    # Move label to left side
    cbar.set_ylabel(f"-log10({pvalue_label})", rotation=90, labelpad=20, fontsize=11)
    cbar.yaxis.set_label_position("left")

    # Update colorbar tick labels
    # Get current tick positions and labels
    current_ticks = cbar.get_yticks()
    new_labels = []
    for tick in current_ticks:
        if tick <= vmin:
            new_labels.append(f"< {int(vmin)} (Mutually exclusive)")
        elif tick >= vmax:
            new_labels.append(f"> {int(vmax)} (Co-occurrence)")
        else:
            new_labels.append(f"{int(tick)}")

    cbar.set_yticklabels(new_labels, fontsize=9)

    # 12. Set title and layout
    ax.set_title(title, fontsize=16, pad=20)
    plt.tight_layout()

    elapsed_time = time.time() - start_time
    logger.info(
        f"Somatic interactions plot: {n_variants:,} variants, {n_samples} samples, "
        f"{len(top_gene_list)} genes analyzed in {elapsed_time:.2f}s"
    )
    
    return fig
