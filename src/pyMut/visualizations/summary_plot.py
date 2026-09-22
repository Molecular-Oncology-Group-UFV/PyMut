"""
Module for generating summary charts.

This module contains functions for creating summary visualizations
that show different statistics from mutation data.
"""

import logging
from typing import Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core import PyMutation
from .oncoplot import DEFAULT_NON_SYNONYMOUS_CLASSIFICATIONS

logger = logging.getLogger(__name__)


def _filter_non_synonymous(
    data: pd.DataFrame,
    variant_column: str,
    include_silent: bool,
    non_syn_classifications: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Filter a mutation DataFrame down to "non-synonymous" variant
    classifications, matching maftools' default read.maf() behavior
    (the `vc_nonSyn` whitelist) exactly. This is the single source of
    truth used by every panel in this module, instead of each function
    keeping its own copy of the classification list.

    Args:
        data: DataFrame with a variant classification column.
        variant_column: Name of the column holding the variant
            classification (e.g. "Variant_Classification").
        include_silent: If True, `data` is returned unchanged -- every
            variant classification counts as a mutation, nothing is
            filtered. If False (default elsewhere in this module), only
            rows whose classification is in the whitelist are kept.
        non_syn_classifications: Custom whitelist to use instead of
            DEFAULT_NON_SYNONYMOUS_CLASSIFICATIONS. Ignored when
            include_silent=True.

    Returns:
        The filtered (or original) DataFrame. Comparison against the
        whitelist is case-insensitive and whitespace-trimmed, so it isn't
        thrown off by casing differences between datasets (e.g.
        "Missense_Mutation" vs "MISSENSE_MUTATION").
    """
    if include_silent:
        return data
    whitelist = (
        non_syn_classifications
        if non_syn_classifications is not None
        else DEFAULT_NON_SYNONYMOUS_CLASSIFICATIONS
    )
    whitelist_lower = {str(v).strip().lower() for v in whitelist}
    mask = data[variant_column].astype(str).str.strip().str.lower().isin(whitelist_lower)
    return data[mask]


def _count_variants_from_samples(
    data: pd.DataFrame, variant_column: str
) -> Dict[str, int]:
    sample_cols = [c for c in data.columns if str(c).startswith("TCGA-")]
    if not sample_cols:
        sample_cols = [
            c for c in data.columns if isinstance(c, str) and c.count("-") >= 2
        ]
    if not sample_cols:
        logger.warning("No sample columns detected, using simple row counts")
        return data[variant_column].value_counts().to_dict()

    logger.debug(f"Counting variants across {len(sample_cols)} sample columns")

    # Build a boolean mask for all sample columns at once: True = has mutation
    # A sample has a mutation if it is not NaN and not equal to REF|REF
    ref_genotype = data["REF"].astype(str) + "|" + data["REF"].astype(str)
    sample_df = data[sample_cols]

    # Broadcasting: compare each sample column against the ref genotype per row
    not_na = sample_df.notna()
    not_ref = sample_df.ne(ref_genotype, axis=0)
    has_mutation = not_na & not_ref  # DataFrame of booleans (rows × samples)

    # Sum across sample columns to get number of mutated samples per row
    mutation_count_per_row = has_mutation.sum(axis=1)

    # Group by variant classification and sum
    vc = data[variant_column].copy()
    valid = vc.notna()
    counts = mutation_count_per_row[valid].groupby(vc[valid]).sum()

    return counts.to_dict()


def _create_variant_classification_plot(
    py_mut: PyMutation,
    variant_column: str = "Variant_Classification",
    ax: Optional[Axes] = None,
    color_map: Optional[Dict] = None,
    set_title: bool = True,
    include_silent: bool = False,
    non_syn_classifications: Optional[Set[str]] = None,
) -> Axes:
    """
    Create a horizontal bar chart showing the distribution of variant classifications.

    This visualization displays the count of each variant classification type present
    in the mutation data, sorted by frequency. By default, silent/synonymous mutations
    and non-coding variants are excluded (matching maftools' default vc_nonSyn whitelist).

    Args:
        py_mut: PyMutation object with mutation data.
        variant_column: Name of the column containing the variant classification.
        ax: Matplotlib axis to draw on. If None, a new one is created.
        color_map: Optional dictionary mapping variant classifications to colors.
        set_title: Whether to set the title on the plot.
        include_silent: Whether to include silent/synonymous and non-coding variants.
                       Default is False.
        non_syn_classifications: Custom whitelist of variant classifications to
                       treat as non-synonymous, mirroring maftools' vc_nonSyn.
                       If None, uses the maftools default. Ignored when
                       include_silent=True.

    Returns:
        Matplotlib axis with the visualization.
    """
    data = py_mut.data

    data_filtered = _filter_non_synonymous(
        data, variant_column, include_silent, non_syn_classifications
    )

    # Count variants properly from sample columns (handles consolidated format)
    variant_counts = _count_variants_from_samples(data_filtered, variant_column)
    variant_counts = dict(sorted(variant_counts.items(), key=lambda item: item[1]))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if color_map:
        colors = [
            color_map.get(variant, plt.colormaps["tab20"](i % 20))
            for i, variant in enumerate(variant_counts.keys())
        ]
    else:
        cmap = plt.colormaps["tab20"]
        colors = [cmap(i % 20) for i in range(len(variant_counts))]

    bars = ax.barh(
        list(variant_counts.keys()), list(variant_counts.values()), color=colors
    )

    if set_title:
        ax.set_title("Variant Classification", fontsize=14, fontweight="bold")

    is_in_summary_plot = hasattr(ax.figure, "axes") and len(ax.figure.axes) > 1
    ax.set_xlabel("Number of variants")
    if not is_in_summary_plot:
        ax.set_ylabel("Variant Classification")
    else:
        ax.set_ylabel("")

    for bar in bars:
        ax.text(
            bar.get_width() + 10,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width())}",
            va="center",
            fontsize=10,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return ax


def _create_variant_type_plot(
    py_mut: PyMutation,
    variant_column: str = "Variant_Type",
    ax: Optional[Axes] = None,
    set_title: bool = True,
    include_silent: bool = False,
    non_syn_classifications: Optional[Set[str]] = None,
) -> Axes:
    """
    Create a horizontal bar chart showing the distribution of variant types.

    This visualization displays the count of each variant type (SNP, INS, DEL, etc.)
    present in the mutation data, sorted by frequency. By default, excludes silent/synonymous
    mutations (matching maftools' default vc_nonSyn whitelist).

    Args:
        py_mut: PyMutation object with mutation data.
        variant_column: Name of the column containing the variant type.
        ax: Matplotlib axis to draw on. If None, a new one is created.
        set_title: Whether to set the title on the plot.
        include_silent: Whether to include silent/synonymous and non-coding variants.
                       Default is False.
        non_syn_classifications: Custom whitelist of variant classifications to
                       treat as non-synonymous, mirroring maftools' vc_nonSyn.
                       If None, uses the maftools default. Ignored when
                       include_silent=True.

    Returns:
        Matplotlib axis with the visualization.
    """
    data = py_mut.data

    # Filtering is always based on Variant_Classification, regardless of what
    # variant_column points to here (which is Variant_Type for this plot).
    data_filtered = _filter_non_synonymous(
        data, "Variant_Classification", include_silent, non_syn_classifications
    )

    # Count variants properly from sample columns (handles consolidated format)
    variant_counts = _count_variants_from_samples(data_filtered, variant_column)
    variant_counts = dict(sorted(variant_counts.items(), key=lambda item: item[1]))
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#D3C4E7', '#FFFACD', '#87CEEB']
    if len(variant_counts) > len(colors):
        cmap = plt.colormaps['tab20']
        colors = list(cmap(range(len(variant_counts))))
    else:
        colors = colors[:len(variant_counts)]
    
    bars = ax.barh(list(variant_counts.keys()), list(variant_counts.values()), color=colors)
    
    if set_title:
        ax.set_title("Variant Type", fontsize=14, fontweight='bold')
    
    is_in_summary_plot = hasattr(ax.figure, "axes") and len(ax.figure.axes) > 1
    ax.set_xlabel("Number of variants")
    if not is_in_summary_plot:
        ax.set_ylabel("Variant Type")
    else:
        ax.set_ylabel("")
    
    for bar in bars:
        ax.text(bar.get_width() + 10,
                 bar.get_y() + bar.get_height()/2,
                 f'{int(bar.get_width())}',
                 va='center', fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return ax


def _normalize_to_pyrimidine(ref: str, alt: str) -> str:
    """
    Normalize SNV to pyrimidine base (C or T) as reference.

    This follows the convention used in mutational signature analysis where
    all mutations are represented with pyrimidines (C or T) as the reference base.
    Purines (A or G) are converted to their complementary pyrimidine representation.

    Args:
        ref: Reference allele (A, T, C, or G).
        alt: Alternative allele (A, T, C, or G).

    Returns:
        Normalized SNV class string (e.g., 'C>T').
    """
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

    if ref in ["C", "T"]:
        return f"{ref}>{alt}"
    else:
        # Convert to complement (purine to pyrimidine)
        return f"{complement[ref]}>{complement[alt]}"


def _create_snv_class_plot(
        py_mut: PyMutation,
        ref_column: str = "REF",
        alt_column: str = "ALT",
        ax: Optional[Axes] = None,
        set_title: bool = True,
        normalize_pyrimidine: bool = True,
    include_silent: bool = True,
    non_syn_classifications: Optional[Set[str]] = None,
) -> Axes:
    """
    Create a horizontal bar chart showing the distribution of SNV classes.
    """
    data = _filter_non_synonymous(
        py_mut.data,
        "Variant_Classification",
        include_silent,
        non_syn_classifications,
    )

    if ref_column not in data.columns or alt_column not in data.columns:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data available for SNV Class",
                ha='center', va='center', fontsize=12)
        if set_title:
            ax.set_title("SNV Class", fontsize=14, fontweight='bold')
        ax.axis('off')
        return ax

    # Filter to only SNPs (single nucleotide variants)
    df_snp = data[
        (data[ref_column].isin(["A", "T", "C", "G"]))
        & (data[alt_column].isin(["A", "T", "C", "G"]))
        ].copy()

    if len(df_snp) == 0:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No SNV data available", ha="center", va="center", fontsize=12)
        if set_title:
            ax.set_title("SNV Class", fontsize=14, fontweight="bold")
        ax.axis("off")
        return ax

    # FIX VECTORIZADO: Usamos un diccionario y .map() en lugar de iterar fila por fila
    raw_snv = df_snp[ref_column] + ">" + df_snp[alt_column]

    if normalize_pyrimidine:
        # Diccionario estándar de firmas mutacionales para convertir purinas a pirimidinas
        transitions = {
            'A>G': 'T>C', 'T>C': 'T>C',
            'G>A': 'C>T', 'C>T': 'C>T',
            'A>T': 'T>A', 'T>A': 'T>A',
            'A>C': 'T>G', 'T>G': 'T>G',
            'C>A': 'C>A', 'G>T': 'C>A',
            'C>G': 'C>G', 'G>C': 'C>G'
        }
        df_snp["SNV_Class"] = raw_snv.map(transitions).fillna(raw_snv)
    else:
        df_snp["SNV_Class"] = raw_snv

    # Count from sample columns
    snv_counts = _count_variants_from_samples(df_snp, "SNV_Class")
    snv_counts = dict(sorted(snv_counts.items(), key=lambda item: item[1]))

    if not snv_counts:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No SNV data available", ha="center", va="center", fontsize=12)
        if set_title:
            ax.set_title("SNV Class", fontsize=14, fontweight="bold")
        ax.axis("off")
        return ax

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    colors = ["#FF8C00", "#9ACD32", "#FFD700", "#FF4500", "#4169E1", "#1E90FF"]
    if len(snv_counts) > len(colors):
        cmap = plt.colormaps["tab20"]
        colors = list(cmap(range(len(snv_counts))))
    else:
        colors = colors[: len(snv_counts)]

    bars = ax.barh(list(snv_counts.keys()), list(snv_counts.values()), color=colors)

    if set_title:
        ax.set_title("SNV Class", fontsize=14, fontweight="bold")

    is_in_summary_plot = hasattr(ax.figure, "axes") and len(ax.figure.axes) > 1
    ax.set_xlabel("Number of variants")
    if not is_in_summary_plot:
        ax.set_ylabel("SNV Class")
    else:
        ax.set_ylabel("")

    for bar in bars:
        ax.text(
            bar.get_width() + 10,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width())}",
            va="center",
            fontsize=10,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return ax


def _create_variants_per_sample_plot(
        py_mut: PyMutation,
        variant_column: str = "Variant_Classification",
        sample_column: str = "Tumor_Sample_Barcode",
        ax: Optional[Axes] = None,
        color_map: Optional[Dict] = None,
        set_title: bool = True,
        max_samples: Optional[int] = 200,
        include_silent: bool = False,
        non_syn_classifications: Optional[Set[str]] = None,
) -> Axes:
    """
    Create a stacked bar plot showing variants per sample (tumor mutation burden).
    """
    data = py_mut.data

    if variant_column not in data.columns:
        logger.warning(f"Column not found: {variant_column}")
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No data available\nMissing column: {variant_column}", ha="center", va="center", fontsize=12)
        if set_title:
            ax.set_title("Variants per Sample", fontsize=14, fontweight="bold")
        ax.axis("off")
        return ax

    data = _filter_non_synonymous(data, variant_column, include_silent, non_syn_classifications)

    samples_as_columns = sample_column not in data.columns

    if samples_as_columns:
        potential_sample_cols = [col for col in data.columns if
                                 col.startswith('TCGA-') or '|' in str(data[col].iloc[0])]

        if not potential_sample_cols:
            logger.warning("No sample columns found")
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No sample columns detected", ha='center', va='center', fontsize=12)
            if set_title:
                ax.set_title("Variants per Sample", fontsize=14, fontweight='bold')
            ax.axis('off')
            return ax

        # FIX VECTORIZADO: Máscara booleana y Groupby transpuesto
        ref_col = "Reference_Allele" if "Reference_Allele" in data.columns else "REF"
        if ref_col in data.columns:
            ref_geno = data[ref_col].astype(str) + "|" + data[ref_col].astype(str)
            sample_df = data[potential_sample_cols]
            has_mutation = sample_df.notna() & sample_df.ne(ref_geno, axis=0)
        else:
            no_mut_vals = {"", ".", "0", "0/0", "0|0", "./.", ".|.", "NA", "NaN"}
            sample_df = data[potential_sample_cols]
            has_mutation = sample_df.notna() & ~sample_df.isin(no_mut_vals)

        # Agrupamos los booleanos (True=1) usando la columna de variantes y transponemos
        variant_counts = has_mutation.groupby(data[variant_column]).sum().T
        variant_counts.index.name = 'Sample'

    else:
        # Formato long estándar (Ya era rápido)
        variant_counts = data.groupby([sample_column, variant_column]).size().unstack(fill_value=0)

    variant_counts['total'] = variant_counts.sum(axis=1)
    variant_counts = variant_counts.sort_values('total', ascending=False)

    if max_samples is not None and len(variant_counts) > max_samples:
        variant_counts = variant_counts.iloc[:max_samples]

    median_tmb = variant_counts['total'].median()

    if 'total' in variant_counts.columns:
        variant_counts = variant_counts.drop('total', axis=1)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if color_map is not None:
        colors = [color_map.get(variant, plt.colormaps['tab20'](i % 20)) for i, variant in
                  enumerate(variant_counts.columns)]
    else:
        cmap = plt.colormaps['tab20']
        colors = [cmap(i % cmap.N) for i in range(len(variant_counts.columns))]

    variant_counts.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)

    ax.axhline(y=median_tmb, color='red', linestyle='--', linewidth=1)

    if set_title:
        ax.set_title("Variants per Sample", fontsize=14, fontweight='bold')

    ax.set_ylabel("Nº Variants")
    ax.set_xlabel("Samples")
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', bottom=False)

    is_in_summary_plot = hasattr(ax.figure, 'axes') and len(ax.figure.axes) > 1

    if is_in_summary_plot:
        ax.legend(title="Variant Classification", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.text(0.5, 0.92, f"Median: {median_tmb:.1f}", transform=ax.transAxes, ha='center', fontsize=12)
    else:
        legend_title = f"$\\mathbf{{Variant Classification}}$\n\n$\\mathbf{{Median:}}$ {median_tmb:.1f}"
        ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))

    return ax


def _create_variant_classification_summary_plot(
        py_mut: PyMutation,
        variant_column: str = "Variant_Classification",
        sample_column: str = "Tumor_Sample_Barcode",
        ax: Optional[Axes] = None,
        color_map: Optional[Dict] = None,
        set_title: bool = True,
        include_silent: bool = False,
        non_syn_classifications: Optional[Set[str]] = None,
) -> Axes:
    """
    Create a boxplot showing the distribution of variant counts per sample.
    """
    data = py_mut.data

    if variant_column not in data.columns:
        logger.warning(f"Column not found: {variant_column}")
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Missing column: {variant_column}", ha="center", va="center", fontsize=12)
        if set_title:
            ax.set_title("Variant Classification Summary", fontsize=14, fontweight="bold")
        ax.axis("off")
        return ax

    data = _filter_non_synonymous(data, variant_column, include_silent, non_syn_classifications)

    samples_as_columns = sample_column not in data.columns

    # FIX VECTORIZADO: Directo al DataFrame final en ambos casos, sin diccionarios intermedios lentos
    if samples_as_columns:
        potential_sample_cols = [col for col in data.columns if
                                 col.startswith("TCGA-") or (isinstance(col, str) and col.count("-") >= 2)]

        if not potential_sample_cols:
            if ax is None:
                _, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "No sample columns detected", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return ax

        ref_col = "Reference_Allele" if "Reference_Allele" in data.columns else "REF"
        if ref_col in data.columns:
            ref_geno = data[ref_col].astype(str) + "|" + data[ref_col].astype(str)
            sample_df = data[potential_sample_cols]
            has_mutation = sample_df.notna() & sample_df.ne(ref_geno, axis=0)
        else:
            no_mut_vals = {"", ".", "0", "0/0", "0|0", "./.", ".|.", "NA", "NaN"}
            sample_df = data[potential_sample_cols]
            has_mutation = sample_df.notna() & ~sample_df.isin(no_mut_vals)

        df = has_mutation.groupby(data[variant_column]).sum().T

        # Eliminamos la posible clase "Unknown" si existiese en el índice de columnas
        if "Unknown" in df.columns:
            df = df.drop(columns=["Unknown"])

    else:
        # Reemplazo de .iterrows() por .crosstab() ultrarrápido (en C)
        valid_data = data.dropna(subset=[sample_column, variant_column])
        valid_data = valid_data[valid_data[variant_column] != "Unknown"]

        if valid_data.empty:
            df = pd.DataFrame()
        else:
            df = pd.crosstab(valid_data[sample_column], valid_data[variant_column])

    if df.empty:
        logger.warning("No data found for analysis after processing")
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No data found for analysis", ha="center", va="center", fontsize=12)
        if set_title:
            ax.set_title("Variant Classification Summary", fontsize=14, fontweight="bold")
        ax.axis("off")
        return ax

    col_order = df.sum(axis=0).sort_values(ascending=False).index.tolist()
    df = df[col_order]
    df = df.loc[:, df.sum() > 0]

    variant_types = df.columns.tolist()
    data_to_plot = [np.asarray(df[vt].to_numpy()) for vt in variant_types]

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    bp = ax.boxplot(
        data_to_plot, patch_artist=True,
        medianprops=dict(color="red", linewidth=1.5),
        boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5),
        showfliers=False, widths=0.7, #Cambio temporal buscando fix
    )

    if color_map:
        colors = [color_map.get(vt, plt.colormaps["tab20"](i % 20)) for i, vt in enumerate(variant_types)]
    else:
        cmap = plt.colormaps["tab20"]
        colors = [cmap(i % 20) for i in range(len(variant_types))]

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    is_in_summary_plot = hasattr(ax.figure, "axes") and len(ax.figure.axes) > 1

    if is_in_summary_plot:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(variant_types, rotation=45, ha="right", fontsize=10)

    #ymin = 0
    #ymax = max(max(d) if len(d) > 0 else 0 for d in data_to_plot) * 1.1
    #if ymax == 0:
    #    ymax = 1
    #ax.set_ylim(ymin, ymax)

    ax.set_ylim(bottom=0)

    if set_title:
        ax.set_title("Variant Classification Summary", fontsize=14, fontweight="bold")

    ax.set_ylabel("Mutations / Sample")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    if is_in_summary_plot:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_position(("outward", 5))
        ax.spines["bottom"].set_position(("outward", 5))
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return ax


def _create_top_mutated_genes_plot(
        py_mut: PyMutation,
        mode: str = "variants",
        variant_column: str = "Variant_Classification",
        gene_column: str = "Hugo_Symbol",
        sample_column: str = "Tumor_Sample_Barcode",
        count: int = 10,
        ax: Optional[Axes] = None,
        color_map: Optional[Dict] = None,
        set_title: bool = True,
        include_silent: bool = False,
        non_syn_classifications: Optional[Set[str]] = None,
) -> Axes:
    """
    Create a horizontal bar plot showing the most mutated genes.

    By default, excludes silent/synonymous mutations (matching maftools'
    default vc_nonSyn whitelist), same as every other panel in this module.
    """
    data = py_mut.data

    if gene_column not in data.columns or variant_column not in data.columns:
        if ax is None: _, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Missing required columns", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return ax

    sample_cols = [col for col in data.columns if
                   str(col).startswith("TCGA-") or (isinstance(col, str) and col.count("-") >= 2)]
    samples_as_columns = len(sample_cols) > 0

    if not samples_as_columns and sample_column not in data.columns:
        if ax is None: _, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No samples detected", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return ax

    if ax is None: _, ax = plt.subplots(figsize=(10, 8))

    # Filtrado inicial
    data_filtered = data.dropna(subset=[gene_column, variant_column]).copy()
    data_filtered = data_filtered[
        (data_filtered[gene_column] != "Unknown") & (data_filtered[variant_column] != "Unknown")]

    data_filtered = _filter_non_synonymous(
        data_filtered, variant_column, include_silent, non_syn_classifications
    )

    if data_filtered.empty:
        ax.text(0.5, 0.5, "No data available after filtering", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return ax

    # =========================================================================
    # EL FIX DEFINITIVO: CERO BUCLES PYTHON. TODO MATRICIAL Y GROUPBY.
    # =========================================================================

    if samples_as_columns:
        total_samples_in_dataset = len(sample_cols)

        # 1. Crear máscara booleana de todo el DataFrame
        ref_col = "Reference_Allele" if "Reference_Allele" in data_filtered.columns else "REF"
        if ref_col in data_filtered.columns:
            ref_geno = data_filtered[ref_col].astype(str) + "|" + data_filtered[ref_col].astype(str)
            sample_df = data_filtered[sample_cols]
            mutation_mask = sample_df.notna() & sample_df.ne(ref_geno, axis=0)
        else:
            no_mut_vals = {"", ".", "0", "0/0", "0|0", "./.", ".|.", "NA", "NaN", "<NA>"}
            mutation_mask = data_filtered[sample_cols].notna() & ~data_filtered[sample_cols].isin(no_mut_vals)

        # 2. Preparar columnas para agrupar
        group_keys = [data_filtered[gene_column].rename('Gene'), data_filtered[variant_column].rename('Variant')]

        # --- CÁLCULOS MODO VARIANTS ---
        # Sumamos cuántas mutaciones hay en cada fila a lo largo de las muestras
        row_mutation_counts = mutation_mask.sum(axis=1)
        # Agrupamos por Gen y Variante y sumamos los conteos
        gene_variant_counts = row_mutation_counts.groupby(group_keys).sum().unstack(fill_value=0)

        # --- CÁLCULOS MODO SAMPLES ---
        # Usamos .any() para ver si la muestra tiene AL MENOS UNA mutación para esa combinación Gen-Variante
        sample_any_var = mutation_mask.groupby(group_keys).any()
        gene_variant_sample_counts = sample_any_var.sum(axis=1).unstack(fill_value=0)

        # --- TOTALES GLOBALES POR GEN ---
        gene_sample_counts = mutation_mask.groupby(data_filtered[gene_column]).any().sum(axis=1)
        gene_total_variants = gene_variant_counts.sum(axis=1)

    else:
        # Formato MAF Estándar (Long format)
        total_samples_in_dataset = data_filtered[sample_column].nunique()

        # --- CÁLCULOS MODO VARIANTS --- (Conteo de filas)
        gene_variant_counts = data_filtered.groupby([gene_column, variant_column]).size().unstack(fill_value=0)

        # --- CÁLCULOS MODO SAMPLES --- (Muestras únicas)
        gene_variant_sample_counts = data_filtered.groupby([gene_column, variant_column])[
            sample_column].nunique().unstack(fill_value=0)

        # --- TOTALES GLOBALES POR GEN ---
        gene_sample_counts = data_filtered.groupby(gene_column)[sample_column].nunique()
        gene_total_variants = gene_variant_counts.sum(axis=1)

    # =========================================================================
    # SELECCIÓN Y ORDENACIÓN
    # =========================================================================

    # Elegir el DataFrame base según el modo
    df_counts = gene_variant_counts if mode == "variants" else gene_variant_sample_counts

    # Asegurar que todas las métricas compartan los mismos índices
    valid_genes = df_counts.index.intersection(gene_sample_counts.index).intersection(gene_total_variants.index)
    df_counts = df_counts.loc[valid_genes]

    # Criterios de ordenación (primario y secundario para empates)
    if mode == "variants":
        sort_primary = gene_total_variants.loc[valid_genes]
        sort_secondary = gene_sample_counts.loc[valid_genes]
    else:
        sort_primary = gene_sample_counts.loc[valid_genes]
        sort_secondary = gene_total_variants.loc[valid_genes]

    # Ordenar de mayor a menor
    sorted_genes = sorted(valid_genes, key=lambda g: (-sort_primary[g], -sort_secondary[g]))
    top_genes = sorted_genes[:count]

    # Filtramos el top y lo invertimos [::-1] para que en el barh (horizontal) el más alto quede arriba
    df_plot_final = df_counts.loc[top_genes[::-1]].copy().astype(float)

    # En el modo samples, escalamos las barras apiladas para que la longitud total = Nº de muestras únicas
    if mode == "samples":
        for gene in df_plot_final.index:
            row_sum = df_plot_final.loc[gene].sum()
            unique_s = gene_sample_counts[gene]
            if row_sum > 0 and row_sum != unique_s:
                df_plot_final.loc[gene] = df_plot_final.loc[gene] * (unique_s / row_sum)

    # Quitar columnas que hayan quedado a 0 en el top
    df_plot_final = df_plot_final.loc[:, (df_plot_final != 0).any(axis=0)]

    if df_plot_final.empty:
        ax.text(0.5, 0.5, "No genes to display", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return ax

    # =========================================================================
    # RENDERIZADO DEL PLOT
    # =========================================================================

    variant_types_in_plot = df_plot_final.columns.tolist()
    cmap_instance = plt.colormaps.get_cmap('tab20')
    if color_map is None:
        colors_for_plot = [cmap_instance(i % cmap_instance.N) for i in range(len(variant_types_in_plot))]
    else:
        colors_for_plot = [color_map.get(vt, cmap_instance(i % cmap_instance.N)) for i, vt in
                           enumerate(variant_types_in_plot)]

    df_plot_final.plot(kind='barh', stacked=True, ax=ax, color=colors_for_plot, width=0.65)
    ax.margins(x=0.05, y=0.01)

    # Etiquetas de porcentaje (a la derecha de las barras)
    for i, gene in enumerate(df_plot_final.index):
        num_samples = gene_sample_counts[gene]
        percentage = (num_samples / total_samples_in_dataset) * 100 if total_samples_in_dataset > 0 else 0
        bar_length = df_plot_final.loc[gene].sum()
        offset = 0.01 * ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 0.1

        # En variants usamos número entero, en samples %
        if mode == "variants":
            ax.text(bar_length + offset, i, f'{percentage:.0f}%', va='center', fontsize=10)
        else:
            ax.text(bar_length + offset, i, f'{percentage:.1f}%', va='center', fontsize=10)

    # Títulos y ejes
    title_suffix = "(variants)" if mode == "variants" else "(Freq)"
    if set_title:
        ax.set_title(f"Top {count} Mutated Genes {title_suffix}", fontsize=14, fontweight='bold')

    ax.set_xlabel("Number of variants" if mode == "variants" else "Number of samples")
    ax.set_ylabel("")

    # Eje secundario decorativo derecho para "% samples"
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    ax2.set_ylabel("% samples", rotation=270, labelpad=15)

    for spine in ["top", "right", "bottom", "left"]:
        ax2.spines[spine].set_visible(False)
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=True)

    # Leyenda
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = dict(zip(labels, handles))
    valid_handles = [legend_elements[label] for label in df_plot_final.columns if label in legend_elements]
    valid_labels = [label for label in df_plot_final.columns if label in legend_elements]

    if valid_labels:
        ncol_legend = min(len(valid_labels), 4)
        num_rows = (len(valid_labels) + ncol_legend - 1) // ncol_legend
        y_offset = -0.25 - (0.06 * num_rows)
        ax.legend(
            valid_handles, valid_labels, title="Variant Classification", loc="lower center",
            bbox_to_anchor=(0.5, y_offset), ncol=ncol_legend, fontsize="small", title_fontsize="medium"
        )
    elif ax.get_legend() is not None:
        ax.get_legend().remove()

    return ax
    

def _create_summary_plot(py_mut: PyMutation,
                      figsize: Tuple[int, int] = (16, 12),
                      title: str = "Mutation Summary",
                      max_samples: Optional[int] = 200,
                      top_genes_count: int = 10,
                      include_silent: bool = False,
                      non_syn_classifications: Optional[Set[str]] = None) -> Figure:
    """
    Create a multi-panel summary plot with comprehensive mutation visualizations.

    This plot combines six individual visualizations into a single figure:
    - Variant classification distribution
    - Variant type distribution  
    - SNV classification (trinucleotide context)
    - Variants per sample distribution
    - Variant classification summary (boxplot)
    - Top mutated genes

    Args:
        py_mut: PyMutation object with mutation data.
        figsize: Figure size as (width, height) in inches.
        title: Main title for the summary plot.
        max_samples: Maximum number of samples to display in variants per sample plot.
                    If None, all samples are shown.
        top_genes_count: Number of top genes to display in the mutated genes plot.
        include_silent: If False (default, matches maftools' default vc_nonSyn
                    whitelist), silent/synonymous and non-coding variants are
                    excluded from every panel except SNV Class (which, like
                    maftools' TiTv view, always shows all SNPs regardless of
                    coding consequence). If True, every variant classification
                    counts everywhere.
        non_syn_classifications: Custom whitelist of "non-synonymous" variant
                    classifications, mirroring maftools' vc_nonSyn argument.
                    If None, uses maftools' own default whitelist. Ignored
                    when include_silent=True.

    Returns:
        Figure object containing all summary visualizations.
    """
    data = py_mut.data

    fig, axs = plt.subplots(2, 3, figsize=figsize, gridspec_kw={'width_ratios': [1.5, 1.5, 1.5], 'height_ratios': [1, 1]})
    fig.suptitle(title, fontsize=16)
    
    variant_classification_col = "Variant_Classification"
    sample_column = "Tumor_Sample_Barcode"
    gene_column = "Hugo_Symbol"
    
    if variant_classification_col not in data.columns:
        for col in data.columns:
            if col.lower() == variant_classification_col.lower():
                variant_classification_col = col
                break
    
    if gene_column not in data.columns:
        for col in data.columns:
            if col.lower() == gene_column.lower():
                gene_column = col
                break
    
    unique_variants = data[variant_classification_col].unique()
    cmap = plt.colormaps['tab20']
    variant_color_map = {variant: cmap(i % 20) for i, variant in enumerate(unique_variants) if pd.notna(variant)}
    
    var_class_ax = _create_variant_classification_plot(
        py_mut,
        variant_column=variant_classification_col,
        ax=axs[0, 0],
        color_map=variant_color_map,
        set_title=True,
        include_silent=include_silent,
        non_syn_classifications=non_syn_classifications,
    )
    
    _create_variant_type_plot(
        py_mut, ax=axs[0, 1], set_title=True,
        include_silent=include_silent,
        non_syn_classifications=non_syn_classifications,
    )
    
    _create_snv_class_plot(
        py_mut,
        ref_column="REF",
        alt_column="ALT",
        ax=axs[0, 2],
        set_title=True,
        include_silent=include_silent,
        non_syn_classifications=non_syn_classifications,
    )
    
    variants_ax = _create_variants_per_sample_plot(
        py_mut,
        variant_column=variant_classification_col,
        sample_column=sample_column,
        ax=axs[1, 0],
        color_map=variant_color_map,
        set_title=True,
        max_samples=max_samples,
        include_silent=include_silent,
        non_syn_classifications=non_syn_classifications,
    )
    
    var_boxplot_ax = _create_variant_classification_summary_plot(
        py_mut,
        variant_column=variant_classification_col,
        sample_column=sample_column,
        ax=axs[1, 1],
        color_map=variant_color_map,
        set_title=True,
        include_silent=include_silent,
        non_syn_classifications=non_syn_classifications,
    )
    
    top_genes_ax = _create_top_mutated_genes_plot(
        py_mut,
        variant_column=variant_classification_col,
        gene_column=gene_column,
        sample_column=sample_column,
        mode="samples",
        count=top_genes_count,
        ax=axs[1, 2],
        color_map=variant_color_map,
        set_title=True,
        include_silent=include_silent,
        non_syn_classifications=non_syn_classifications,
    )
    
    if var_class_ax.get_legend() is not None:
        var_class_ax.get_legend().remove()
    
    if variants_ax.get_legend() is not None:
        variants_ax.get_legend().remove()
        
    if var_boxplot_ax.get_legend() is not None:
        var_boxplot_ax.get_legend().remove()
        
    if top_genes_ax.get_legend() is not None:
        top_genes_ax.get_legend().remove()
    
    handles = []
    labels = []
    
    variant_counts = data[variant_classification_col].value_counts()
    
    ordered_variants = variant_counts.index.tolist()
    
    for variant in ordered_variants:
        if variant in variant_color_map and not pd.isnull(variant) and variant != "Unknown":
            color = variant_color_map[variant]
            patch = plt.Rectangle((0,0), 1, 1, fc=color)
            handles.append(patch)
            labels.append(variant)
    
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 5),
        bbox_to_anchor=(0.5, 0.00),
        fontsize="small",
        title_fontsize="medium",
    )
    
    plt.tight_layout(pad=2.0)

    plt.subplots_adjust(top=0.9, bottom=0.15, hspace=0.5, left=0.15, right=0.93)
    
    return fig