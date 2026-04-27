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
- _compute_fisher_interactions(): Calculates Fisher's exact test for gene pairs
- _create_somatic_interactions_plot(): Main function to create the heatmap
"""

import logging
import time
from typing import TYPE_CHECKING, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact

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


def _compute_fisher_interactions(mutation_matrix: pd.DataFrame,
                                 genes: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute Fisher's exact test for all gene pairs.
    
    For each pair of genes:
    - Constructs 2x2 contingency table (both mutated, gene1 only, gene2 only, neither)
    - Runs Fisher's exact test
    - Calculates odds ratio and p-value
    
    Results interpretation:
    - Odds ratio > 1: Co-occurrence (genes mutated together more than expected)
    - Odds ratio < 1: Mutual exclusivity (genes rarely mutated together)
    - P-value < 0.05: Statistically significant interaction
    
    Args:
        mutation_matrix: Binary mutation matrix (samples x genes)
        genes: List of gene names to analyze (in desired order)
        
    Returns:
        Tuple of three DataFrames (all genes x genes):
        - results: Signed -log10(p-value) for visualization
                  (positive = co-occurrence, negative = mutual exclusivity)
        - pvalues: Raw p-values from Fisher's test
        - oddsratios: Odds ratios from Fisher's test
    """
    n_genes = len(genes)
    results = pd.DataFrame(np.zeros((n_genes, n_genes)), index=genes, columns=genes)
    pvalues = pd.DataFrame(np.ones((n_genes, n_genes)), index=genes, columns=genes)
    oddsratios = pd.DataFrame(np.ones((n_genes, n_genes)), index=genes, columns=genes)

    # Pre-extract all vectors as a numpy array for faster access
    matrix_vals = mutation_matrix[genes].values  # shape: (n_samples, n_genes)

    for i in range(n_genes):
        results.iloc[i, i] = np.nan
        for j in range(i + 1, n_genes):  # Only upper triangle
            vec1 = matrix_vals[:, i]
            vec2 = matrix_vals[:, j]

            both_mut = np.sum((vec1 == 1) & (vec2 == 1))
            gene1_only = np.sum((vec1 == 1) & (vec2 == 0))
            gene2_only = np.sum((vec1 == 0) & (vec2 == 1))
            neither = np.sum((vec1 == 0) & (vec2 == 0))

            contingency = np.array([[both_mut, gene1_only],
                                    [gene2_only, neither]])
            try:
                or_val, p_val = fisher_exact(contingency)
                if p_val < 1e-10:
                    p_val = 1e-10

                pvalues.iloc[i, j] = p_val
                pvalues.iloc[j, i] = p_val
                oddsratios.iloc[i, j] = or_val
                oddsratios.iloc[j, i] = or_val

                signed_log = -np.log10(p_val) if or_val > 1 else np.log10(p_val)
                results.iloc[i, j] = signed_log
                results.iloc[j, i] = signed_log

            except Exception as e:
                logger.warning(f"Fisher test failed for {genes[i]}-{genes[j]}: {e}")
                pvalues.iloc[i, j] = 1.0
                pvalues.iloc[j, i] = 1.0
                oddsratios.iloc[i, j] = 1.0
                oddsratios.iloc[j, i] = 1.0

    return results, pvalues, oddsratios


def _create_somatic_interactions_plot(py_mut: 'PyMutation',
                                     top_genes: int = DEFAULT_TOP_GENES,
                                     gene_column: str = "Hugo_Symbol",
                                     sample_column: str = "Tumor_Sample_Barcode",
                                     figsize: Tuple[int, int] = DEFAULT_SOMATIC_FIGSIZE,
                                     title: Optional[str] = None,
                                     vmin: float = -3.0,
                                     vmax: float = 3.0,
                                     pvalue: Tuple[float, float] = (0.05, 0.1),
                                     show_counts: bool = True) -> Figure:
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
        pvalue: Tuple of p-value thresholds (upper, lower) for significance markers
        show_counts: Whether to show sample counts for each gene in labels
        
    Returns:
        matplotlib.figure.Figure: The somatic interactions heatmap
        
    Example:
        >>> fig = py_mut.somatic_interactions(top_genes=25)
        >>> py_mut.save_figure(fig, "somatic_interactions.png")
    """
    # Set title
    if title is None:
        title = f"Somatic Interactions (Top {top_genes} Mutated Genes)"

    start_time = time.time()

    # 1. Use ALL variants from py_mut.data
    # NOTE: We exclude Silent variants and non-coding variants
    # to focus on functionally relevant mutations.
    # PyMut's read_maf() keeps all variants, so we must filter here.
    data_for_analysis = py_mut.data.copy()

    # Filter out silent and non-coding variants
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

    # Phase 1: Select top genes by AlteredSamples desc, total desc, gene name asc (for determinism)
    gene_summary_for_selection = gene_summary.sort_values(
        by=["AlteredSamples", "total", gene_column], ascending=[False, False, True]
    )
    top_gene_list_unsorted = (
        gene_summary_for_selection[gene_column].head(top_genes).tolist()
    )

    # Phase 2: Resort selected genes alphabetically within each sample count for display
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

    # Create gene labels with counts (format: "GENE [count]")
    gene_labels = {}
    if show_counts:
        for gene in top_gene_list:
            count = int(gene_counts[gene])
            gene_labels[gene] = f"{gene} [{count}]"
    else:
        gene_labels = {gene: gene for gene in top_gene_list}

    # 4. Filter mutation matrix to top genes
    mutation_matrix_top = mutation_matrix[top_gene_list]

    # 5. Compute Fisher's exact test for all gene pairs
    results, pvalues, oddsratios = _compute_fisher_interactions(
        mutation_matrix_top, top_gene_list
    )

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
    pvalues_display = pvalues.loc[rows_order, cols_order].copy()

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
            "label": "-log10(P-value)",
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
            label=f"P < {min(pvalue)}",
            markerfacecolor="black",
            markersize=12,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"P < {max(pvalue)}",
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
    cbar.set_ylabel("-log10(P-value)", rotation=90, labelpad=20, fontsize=11)
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
