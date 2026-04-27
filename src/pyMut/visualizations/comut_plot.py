"""
Module for creating CoMut (Complex Mutation) plots.

This module contains functions for creating multi-panel visualizations that combine
multiple aspects of mutation data into a comprehensive view.

CoMut plots are essential in cancer genomics for:
- Visualizing mutation burden across samples
- Showing mutational signatures
- Displaying clinical annotations
- Integrating multiple data types in aligned panels

Main functions:
- _create_comut_mutation_burden_plot(): Panel A - Mutation burden (synonymous/non-synonymous)
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..core import PyMutation

from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_COMUT_FIGSIZE = (14, 1.8)  # Width increased for legend placement outside plot
DEFAULT_TERRITORY_BP = 60456963  # Default exome size in bp
DEFAULT_MAX_SAMPLES = 50  # Default number of samples to display

# Mutation classification mappings (case-insensitive - will normalize to uppercase)
SYNONYMOUS_CLASSES = {
    'SILENT',
    'SYNONYMOUS_SNV',
}

NON_SYNONYMOUS_CLASSES = {
    'MISSENSE_MUTATION',
    'NONSENSE_MUTATION',
    'FRAME_SHIFT_DEL',
    'FRAME_SHIFT_INS',
    'IN_FRAME_DEL',
    'IN_FRAME_INS',
    'SPLICE_SITE',  # Only Splice_Site counts (not Splice_Region)
    'NONSTOP_MUTATION',
    'TRANSLATION_START_SITE',
    'START_CODON_SNP',
}

# Exclude non-coding regions from TMB calculation
# Note: Splice_Region is NOT counted as non-synonymous (can inflate TMB)
EXCLUDED_CLASSES = {
    "3'UTR",
    "5'UTR",
    'IGR',
    'INTRON',
    'RNA',
    'LINCRNA',
    "3'FLANK",
    "5'FLANK",
    'SPLICE_REGION',  # Exclude Splice_Region from TMB (not always counted)
}


def _reload_with_silent_variants(py_mut: 'PyMutation', variant_column: str = "Variant_Classification") -> pd.DataFrame:
    """
    Reload data including silent variants for CoMut plots.
    
    Since read_maf() filters out Silent/Synonymous variants by default,
    we need to reload the original file to include them for CoMut calculations.
    
    Args:
        py_mut: PyMutation object with metadata
        variant_column: Column name for variant classification
        
    Returns:
        DataFrame with coding + silent variants included
    """
    # Check if we have the file path in metadata
    if py_mut.metadata is None or not hasattr(py_mut.metadata, 'file_path'):
        logger.warning("Cannot reload file: metadata missing. Using existing data (may lack synonymous variants)")
        return py_mut.data.copy()
    
    file_path = Path(py_mut.metadata.file_path)
    
    if not file_path.exists():
        logger.warning(f"Cannot reload file: {file_path} not found. Using existing data (may lack synonymous variants)")
        return py_mut.data.copy()
    
    logger.info(f"Reloading MAF with silent variants for CoMut: {file_path}")
    
    # Import here to avoid circular dependency
    import io
    
    # Temporarily patch read_maf to skip silent filtering
    # We'll read the file and apply CoMut-specific filtering
    try:
        # Read raw MAF without going through PyMutation's normal filtering
        from ..input import _open_text_maybe_gzip, _standardise_maf_columns
        from ..utils.format import normalize_variant_classification
        
        # Read file (similar to read_maf but without filtering)
        comments = []
        buf = io.StringIO()
        
        with _open_text_maybe_gzip(file_path) as fh:
            for line in fh:
                if line.startswith("#"):
                    comments.append(line.rstrip("\n"))
                else:
                    buf.write(line)
        buf.seek(0)
        
        # Load dataframe
        try:
            maf = pd.read_csv(buf, sep="\t", dtype_backend="pyarrow", low_memory=False)
        except (ValueError, ImportError):
            buf.seek(0)
            maf = pd.read_csv(buf, engine="c", low_memory=False, sep="\t")
        
        # Standardize columns
        _standardise_maf_columns(maf)
        maf = normalize_variant_classification(maf)
        
        # Apply CoMut-specific filtering (keep coding + silent)
        maf = _filter_for_comut(maf, variant_column)
        
        # Create minimal PyMut-like structure (just the data we need)
        # We need to match the columns from the original py_mut object
        return maf
        
    except Exception as e:
        logger.error(f"Failed to reload MAF with silent variants: {e}")
        logger.warning("Falling back to existing data (may lack synonymous variants)")
        return py_mut.data.copy()


def _filter_for_comut(data: pd.DataFrame, variant_column: str = "Variant_Classification") -> pd.DataFrame:
    """
    Filter variants to include both coding and silent variants for CoMut plots.

    CoMut plots require synonymous variants to calculate mutation burden correctly,
    unlike other pyMut visualizations which typically exclude them.

    This filter keeps:
    - Non-synonymous coding variants (functional impact)
    - Synonymous/Silent variants (for accurate TMB calculation)
    - Excludes non-coding regions (RNA, Intron, IGR, etc.)

    Args:
        data: DataFrame with mutation data
        variant_column: Column name for variant classification

    Returns:
        Filtered DataFrame with coding + silent variants
    """
    if variant_column not in data.columns:
        logger.warning(f"Column '{variant_column}' not found, skipping CoMut filtering")
        return data

    original_count = len(data)

    # Define coding + silent variants to keep for CoMut
    COMUT_VARIANTS = [
        # Non-synonymous (functional impact)
        "FRAME_SHIFT_DEL",
        "FRAME_SHIFT_INS",
        "IN_FRAME_DEL",
        "IN_FRAME_INS",
        "MISSENSE_MUTATION",
        "NONSENSE_MUTATION",
        "NONSTOP_MUTATION",
        "SPLICE_SITE",
        "TRANSLATION_START_SITE",
        # Synonymous (needed for accurate TMB)
        "SILENT",
        "SYNONYMOUS_SNV",
    ]

    # Case-insensitive filtering
    coding_mask = data[variant_column].str.upper().isin(COMUT_VARIANTS)
    filtered_data = data[coding_mask].copy()

    filtered_count = original_count - len(filtered_data)
    if filtered_count > 0:
        logger.info(
            f"CoMut filter: kept {len(filtered_data):,} coding+silent variants, "
            f"excluded {filtered_count:,} non-coding variants (RNA, Intron, IGR, etc.)"
        )

    return filtered_data


def _classify_variant(variant_class: str) -> Optional[str]:
    """
    Classify a variant as synonymous, non-synonymous, or excluded.

    Args:
        variant_class: Variant_Classification value from MAF

    Returns:
        'synonymous', 'non_synonymous', or None (excluded)
    """
    # Normalize to uppercase for case-insensitive matching
    variant_class_upper = variant_class.upper()

    if variant_class_upper in SYNONYMOUS_CLASSES:
        return "synonymous"
    elif variant_class_upper in NON_SYNONYMOUS_CLASSES:
        return "non_synonymous"
    elif variant_class_upper in EXCLUDED_CLASSES:
        return None
    else:
        # Unknown classification - log warning but exclude
        logger.warning(
            f"Unknown variant classification: {variant_class} - excluding from TMB"
        )
        return None


def _calculate_tmb_per_sample(
    data: pd.DataFrame, sample_column: str, variant_column: str, territory_bp: int
) -> pd.DataFrame:
    """
    Calculate TMB (tumor mutation burden) per sample.

    Calculates synonymous and non-synonymous mutation rates normalized to Mb.

    Args:
        data: Filtered MAF data
        sample_column: Column name for sample IDs
        variant_column: Column name for variant classification
        territory_bp: Territory size in base pairs (for normalization)

    Returns:
        DataFrame with columns: sample, n_syn, n_nonsyn, tmb_syn, tmb_nonsyn, tmb_total
    """
    # Classify all variants
    data = data.copy()
    data["_classification"] = data[variant_column].apply(_classify_variant)

    # Remove excluded variants
    data_coding = data[data["_classification"].notna()].copy()

    logger.info(
        f"TMB calculation: {len(data_coding):,} coding variants "
        f"from {len(data):,} total variants"
    )

    # Count by sample and classification
    counts = (
        data_coding.groupby([sample_column, "_classification"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure both columns exist
    if "synonymous" not in counts.columns:
        counts["synonymous"] = 0
    if "non_synonymous" not in counts.columns:
        counts["non_synonymous"] = 0

    # Calculate TMB (mutations per megabase)
    mb = territory_bp / 1e6

    result = pd.DataFrame(
        {
            "sample": counts.index,
            "n_syn": counts["synonymous"].values,
            "n_nonsyn": counts["non_synonymous"].values,
            "tmb_syn": counts["synonymous"].values / mb,
            "tmb_nonsyn": counts["non_synonymous"].values / mb,
        }
    )

    result["tmb_total"] = result["tmb_syn"] + result["tmb_nonsyn"]

    return result


def _order_samples_by_tmb(tmb_df: pd.DataFrame) -> List[str]:
    """
    Order samples by non-synonymous TMB (descending), then by sample ID (ascending).

    Samples are ordered by NON-SYNONYMOUS mutation burden only,
    not total TMB. This ensures the ordering matches the functional impact of mutations.

    This ordering is used consistently across all CoMut panels.

    Args:
        tmb_df: DataFrame from _calculate_tmb_per_sample

    Returns:
        List of sample IDs in display order
    """
    # Order by NON-SYNONYMOUS TMB (not total)
    ordered = tmb_df.sort_values(by=["tmb_nonsyn", "sample"], ascending=[False, True])
    return ordered["sample"].tolist()


def load_sample_order(file_path: str) -> List[str]:
    """
    Load sample order from a text file.

    This utility loads a previously saved sample order to ensure consistent
    ordering across all CoMut panels (A-G).

    Args:
        file_path: Path to the sample order file (one sample ID per line)

    Returns:
        List of sample IDs in display order

    Example:
        >>> sample_order = load_sample_order("comut_sample_order.txt")
        >>> fig_a = py_mut.comut_mutation_burden(sample_ids=sample_order)
        >>> fig_b = py_mut.comut_mutation_signatures_plot(
        ...     signatures_tsv="sig_contribution.tsv",
        ...     sample_order=sample_order
        ... )
    """
    with open(file_path, "r") as f:
        sample_order = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(sample_order)} sample IDs from {file_path}")
    return sample_order


def save_sample_order(
    py_mut: "PyMutation",
    output_path: str = "sample_order.txt",
    sample_column: str = "Tumor_Sample_Barcode",
    variant_column: str = "Variant_Classification",
    territory_bp: int = DEFAULT_TERRITORY_BP,
    max_samples: Optional[int] = DEFAULT_MAX_SAMPLES,
    somatic_only: bool = True,
    pass_only: bool = True,
) -> List[str]:
    """
    Compute sample order for CoMut panels and save to file.

    This utility function calculates TMB, orders samples, and saves the order
    to a text file that can be reused across all CoMut panels (A-G) to ensure
    perfect alignment.

    Args:
        py_mut: PyMutation object containing mutation data
        output_path: Path to save sample order file (default: "sample_order.txt")
        sample_column: Column name for sample IDs
        variant_column: Column name for variant classification
        territory_bp: Territory size in bp for normalization
        max_samples: Maximum number of samples to include (None = all)
        somatic_only: If True, filter to somatic variants
        pass_only: If True, filter to PASS variants

    Returns:
        List of sample IDs in display order

    Example:
        >>> # Compute and save sample order for all CoMut panels
        >>> sample_order = save_sample_order(
        ...     py_mut,
        ...     output_path="comut_sample_order.txt",
        ...     max_samples=50
        ... )
        >>>
        >>> # Use the same order for all panels
        >>> fig_a = py_mut.comut_mutation_burden(sample_ids=sample_order)
        >>> # fig_b = py_mut.comut_panel_b(sample_ids=sample_order)
        >>> # ...
    """
    # Apply same filters as main function
    data = py_mut.data.copy()

    if somatic_only and "Variant_Status" in data.columns:
        data = data[data["Variant_Status"].str.upper() == "SOMATIC"]

    if pass_only and "FILTER" in data.columns:
        data = data[data["FILTER"].isin(["PASS", "."])]

    # Remove duplicates
    dup_cols = []
    for col in [
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
    ]:
        if col in data.columns:
            dup_cols.append(col)

    if dup_cols:
        dup_cols.append(sample_column)
        data = data.drop_duplicates(subset=dup_cols)

    # Calculate TMB
    tmb_df = _calculate_tmb_per_sample(
        data, sample_column, variant_column, territory_bp
    )

    # Order samples
    sample_order = _order_samples_by_tmb(tmb_df)

    # Limit if requested
    if max_samples is not None:
        sample_order = sample_order[:max_samples]

    # Save to file
    with open(output_path, "w") as f:
        for sample in sample_order:
            f.write(f"{sample}\n")

    logger.info(f"Saved {len(sample_order)} sample IDs to {output_path}")

    return sample_order


def _create_comut_mutation_burden_plot(
    py_mut: "PyMutation",
    sample_column: str = "Tumor_Sample_Barcode",
    variant_column: str = "Variant_Classification",
    territory_bp: int = DEFAULT_TERRITORY_BP,
    figsize: Tuple[float, float] = DEFAULT_COMUT_FIGSIZE,
    title: Optional[str] = None,
    hypermutator_threshold: Optional[float] = None,
    sample_ids: Optional[List[str]] = None,
    max_samples: Optional[int] = DEFAULT_MAX_SAMPLES,
    somatic_only: bool = True,
    pass_only: bool = True,
    show_xlabel: bool = True,
) -> Figure:
    """
    Create CoMut Panel A: Mutation Burden visualization.

    Displays a stacked bar chart showing synonymous and non-synonymous mutation
    rates (Muts/Mb) for each sample. Samples are ordered by total TMB.

    This implementation follows these conventions:
    - Shows top N samples (default: 50) for better visualization
    - Filters for somatic variants with PASS status (configurable)
    - Uses strict non-synonymous classification (excludes Splice_Region)
    - Compact figure size optimized for multi-panel layouts

    The visualization shows:
    - Dark bars: Non-synonymous mutations (functional impact)
    - Light bars: Synonymous mutations (silent)
    - Optional: Horizontal line marking hypermutator threshold

    Args:
        py_mut: PyMutation object containing mutation data
        sample_column: Column name for sample IDs
        variant_column: Column name for variant classification
        territory_bp: Territory size in bp for normalization (default: 60,456,963 bp)
        figsize: Figure size (width, height) in inches (default: (14, 1.8))
        title: Plot title (default: "Mutation Burden")
        hypermutator_threshold: TMB threshold for hypermutator line (None = don't show)
        sample_ids: Optional list of specific sample IDs to include (overrides max_samples)
        max_samples: Maximum number of samples to display (default: 50, None = all).
                    Samples are selected by highest TMB.
        somatic_only: If True, filter to Variant_Status == 'Somatic' (default: True)
        pass_only: If True, filter to FILTER in ['PASS', '.'] (default: True)

    Returns:
        matplotlib.figure.Figure: The mutation burden bar chart

    Example:
        >>> # Show top 50 samples by default
        >>> fig = py_mut.comut_mutation_burden()
        >>>
        >>> # Show all samples without filters
        >>> fig = py_mut.comut_mutation_burden(
        ...     max_samples=None,
        ...     somatic_only=False,
        ...     pass_only=False
        ... )
        >>>
        >>> # Use specific sample list (for multi-panel coordination)
        >>> sample_order = ['SAMPLE1', 'SAMPLE2', ...]
        >>> fig = py_mut.comut_mutation_burden(sample_ids=sample_order)
    """
    start_time = time.time()

    if title is None:
        title = "Mutation Burden"

    # Reload data with silent variants (read_maf filters them out by default)
    data = _reload_with_silent_variants(py_mut, variant_column)

    # Validate required columns
    if sample_column not in data.columns:
        raise ValueError(f"Sample column '{sample_column}' not found in data")
    if variant_column not in data.columns:
        raise ValueError(f"Variant column '{variant_column}' not found in data")

    # Apply quality filters
    initial_count = len(data)

    if somatic_only and "Variant_Status" in data.columns:
        data = data[data["Variant_Status"].str.upper() == "SOMATIC"]
        filtered_count = len(data)
        if filtered_count < initial_count:
            logger.info(
                f"Filtered to somatic variants: {filtered_count:,} / {initial_count:,}"
            )

    if pass_only and "FILTER" in data.columns:
        pre_filter = len(data)
        data = data[data["FILTER"].isin(["PASS", "."])]
        filtered_count = len(data)
        if filtered_count < pre_filter:
            logger.info(
                f"Filtered to PASS variants: {filtered_count:,} / {pre_filter:,}"
            )

    if len(data) == 0:
        raise ValueError("No variants remaining after quality filters")

    # Remove duplicates (same genomic position in same sample)
    dup_cols = []
    for col in [
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
    ]:
        if col in data.columns:
            dup_cols.append(col)

    if dup_cols:
        dup_cols.append(sample_column)
        initial_count = len(data)
        data = data.drop_duplicates(subset=dup_cols)
        removed_count = initial_count - len(data)
        if removed_count > 0:
            logger.info(f"Removed {removed_count:,} duplicate variants")

    # Calculate TMB per sample
    tmb_df = _calculate_tmb_per_sample(
        data, sample_column, variant_column, territory_bp
    )

    if len(tmb_df) == 0:
        raise ValueError("No valid samples found for TMB calculation")

    logger.info(
        f"Calculated TMB for {len(tmb_df)} samples "
        f"(range: {tmb_df['tmb_total'].min():.2f}-{tmb_df['tmb_total'].max():.2f} Muts/Mb)"
    )

    # Order samples (respect caller-provided order if given)
    if sample_ids is not None:
        available = set(tmb_df["sample"])
        missing = [s for s in sample_ids if s not in available]
        if missing:
            logger.warning(f"{len(missing)} provided samples not found in TMB data")
        sample_order = [s for s in sample_ids if s in available]
    else:
        sample_order = _order_samples_by_tmb(tmb_df)
        if max_samples is not None:
            sample_order = sample_order[:max_samples]
            logger.info(f"Limited to top {max_samples} samples by TMB")

    # Filter TMB dataframe to selected samples
    tmb_df = tmb_df[tmb_df["sample"].isin(sample_order)].copy()
    tmb_df["sample"] = pd.Categorical(
        tmb_df["sample"], categories=sample_order, ordered=True
    )
    tmb_df = tmb_df.sort_values("sample")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # X positions for bars
    x = np.arange(len(sample_order))

    # Colors
    color_nonsyn = "#2E5C8A"  # Dark blue for non-synonymous
    color_syn = "#A8C5E0"  # Light blue for synonymous

    # Plot stacked bars (non-synonymous at bottom, synonymous on top)
    ax.bar(
        x, tmb_df["tmb_nonsyn"], color=color_nonsyn, label="Nonsynonymous", width=0.8
    )

    ax.bar(
        x,
        tmb_df["tmb_syn"],
        bottom=tmb_df["tmb_nonsyn"],
        color=color_syn,
        label="Synonymous",
        width=0.8,
    )

    # Optional: hypermutator threshold line
    if hypermutator_threshold is not None:
        ax.axhline(
            y=hypermutator_threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Hypermutator ({hypermutator_threshold} Muts/Mb)",
        )

        n_hyper = (tmb_df["tmb_total"] > hypermutator_threshold).sum()
        logger.info(
            f"Hypermutators: {n_hyper}/{len(tmb_df)} samples "
            f"(>{hypermutator_threshold} Muts/Mb)"
        )

    # Styling
    if show_xlabel:
        ax.set_xlabel("Samples", fontsize=10)
    else:
        ax.set_xlabel("")
    ax.set_ylabel("Muts/Mb", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)

    # Hide x-axis labels (too many samples for readability)
    ax.set_xticks(x)
    ax.set_xticklabels([])

    # Y-axis starts at 0
    ax.set_ylim(bottom=0)

    # Legend with Patch objects (matching create_comut_plot style)
    legend_patches = [
        mpatches.Patch(facecolor=color_nonsyn, label="Nonsynonymous"),
        mpatches.Patch(facecolor=color_syn, label="Synonymous"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        title="Mutation Classification",
        title_fontproperties={"size": 9, "weight": "bold"},
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.3,
        alignment="left",
    )

    # Clean appearance (no grid, minimal spines)
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Tight layout with extra space for legend
    plt.tight_layout()

    elapsed_time = time.time() - start_time
    logger.info(f"Panel A created in {elapsed_time:.2f}s")

    plt.close(fig)

    return fig


def _create_comut_mutation_signatures_plot(
    py_mut: "PyMutation",
    signatures_df: Optional[pd.DataFrame] = None,
    signatures_tsv: Optional[str] = None,
    sample_order: Optional[List[str]] = None,
    signature_labels: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[float, float] = (12, 2.0),
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_xlabel: bool = True,
) -> Tuple[Figure, pd.DataFrame]:
    """
    Create CoMut Panel B: Mutational Signatures visualization.

    Displays a stacked bar chart showing the contribution of each mutational signature
    per sample. Each bar represents one sample, with colors indicating different signatures.

    This implementation follows these conventions:
    - Accepts pre-computed signature contributions (like -cmp flag)
    - Stacked bars normalized to [0,1] showing proportions
    - Samples ordered consistently with Panel A for alignment
    - Standard color palette

    The visualization shows:
    - Y-axis: Mutational signatures (proportions, 0-1)
    - X-axis: Samples (no labels for compactness)
    - Stacked bars: Each color represents a different signature
    - Legend: Signature labels

    Args:
        py_mut: PyMutation object (not used for data, only for consistency)
        signatures_df: Pre-computed signature contributions DataFrame.
                      Index = sample IDs, columns = signature names.
                      Values = proportions (should sum to ~1.0 per row).
        signatures_tsv: Alternative path to TSV file with signature contributions.
                       First column should be sample IDs, rest are signatures.
        sample_order: List of sample IDs to display (and their order).
                     Should match Panel A for proper alignment.
                     If None, uses all samples in the order from signatures_df.
        signature_labels: List of signature names in desired display order.
                         If None, uses column order from signatures_df.
        normalize: If True, normalize each row to sum to 1.0 (default: True).
        figsize: Figure size (width, height) in inches (default: (12, 2.0)).
        colors: List of colors for signatures (in order of signature_labels).
        title: Plot title (default: "Mutational Signatures").

    Returns:
        Tuple of (matplotlib.figure.Figure, DataFrame with used data)

    Raises:
        ValueError: If neither signatures_df nor signatures_tsv is provided,
                   or if required data is invalid.

    Example:
        >>> # Load pre-computed signatures
        >>> fig, df = py_mut.comut_mutation_signatures_plot(
        ...     signatures_tsv="sig_contribution.tsv",
        ...     sample_order=sample_order,  # from Panel A
        ...     signature_labels=["Signature 1", "Signature 2", "Signature 3", "Signature 4"]
        ... )
        >>> fig.savefig("comut_panel_b_signatures.png")
        >>>
        >>> # With custom colors
        >>> colors = ['#E0F3FF', '#90EE90', '#4682B4', '#FFB6C1']
        >>> fig, df = py_mut.comut_mutation_signatures_plot(
        ...     signatures_tsv="sig_contribution.tsv",
        ...     sample_order=sample_order,
        ...     colors=colors
        ... )
    """
    start_time = time.time()

    if title is None:
        title = "Mutational Signatures"

    # === 1. Load data ===
    if signatures_df is None and signatures_tsv is None:
        raise ValueError("Must provide either 'signatures_df' or 'signatures_tsv'")

    if signatures_tsv is not None:
        df = pd.read_csv(signatures_tsv, sep="\t")

        # Detect sample column
        sample_col_candidates = [
            "sample",
            "sample_id",
            "tumor_sample_barcode",
            "Tumor_Sample_Barcode",
        ]
        sample_col = next(
            (col for col in sample_col_candidates if col in df.columns), df.columns[0]
        )

        df[sample_col] = df[sample_col].astype(str)
        df = df.set_index(sample_col)
    else:
        df = signatures_df.copy()
        df.index = df.index.astype(str)

    if df.empty:
        raise ValueError("Signature contributions DataFrame is empty")

    # === 2. Handle signature labels ===
    if signature_labels is not None:
        missing_sigs = [sig for sig in signature_labels if sig not in df.columns]
        if missing_sigs:
            logger.warning(f"Adding {len(missing_sigs)} missing signatures with zeros")
            for sig in missing_sigs:
                df[sig] = 0.0
        df = df[signature_labels]
    else:
        signature_labels = list(df.columns)

    # === 3. Filter and order samples ===
    if sample_order is not None:
        available_samples = set(df.index)
        present_samples = [s for s in sample_order if s in available_samples]

        if not present_samples:
            raise ValueError("No samples from sample_order found in signatures data")

        df = df.loc[present_samples].copy()
        df.index = pd.Categorical(df.index, categories=present_samples, ordered=True)
        df = df.sort_index()

    # === 4. Normalize ===
    if normalize:
        row_sums = df.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=0.01):
            df = df.div(row_sums, axis=0).fillna(0).clip(lower=0, upper=1)

    # === 5. Setup colors ===
    if colors is None:
        colors = [
            "#E0F3FF",  # Very light blue (Signature 1)
            "#b7cfb3",  # Light green (Signature 2)
            "#4682B4",  # Steel blue (Signature 3)
            "#FFB6C1",  # Light pink (Signature 4)
        ]

        if len(signature_labels) > len(colors):
            extra_colors = plt.cm.Set3(
                np.linspace(0, 1, len(signature_labels) - len(colors))
            )
            colors.extend([plt.matplotlib.colors.rgb2hex(c) for c in extra_colors])

    if len(colors) < len(signature_labels):
        raise ValueError(
            f"Need {len(signature_labels)} colors but only {len(colors)} provided"
        )

    # === 6. Create figure ===
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    for i, sig_name in enumerate(signature_labels):
        values = df[sig_name].values
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=colors[i],
            label=sig_name,
            width=0.8,
            edgecolor="none",
        )
        bottom += values

    # === 7. Styling ===
    if show_xlabel:
        ax.set_xlabel("Samples", fontsize=10)
    else:
        ax.set_xlabel("")
    ax.set_ylabel("Mutational Signatures", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)

    ax.set_xticks(x)
    ax.set_xticklabels([])

    # Y-axis: fixed [0, 1] range
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])

    # Legend with Patch objects (matching create_comut_plot style)
    legend_patches = [
        mpatches.Patch(facecolor=colors[i], label=sig_name)
        for i, sig_name in enumerate(signature_labels)
    ]
    ax.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        title="Mutation Signature",
        title_fontproperties={"size": 9, "weight": "bold"},
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.3,
        alignment="left",
    )

    # Clean spines
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    elapsed_time = time.time() - start_time
    logger.info(f"Panel B created in {elapsed_time:.2f}s")

    plt.close(fig)

    return fig, df


def _create_comut_purity_plot(
    py_mut: "PyMutation",
    purity_df: Optional[pd.DataFrame] = None,
    purity_tsv: Optional[str] = None,
    sample_order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 0.6),
    title: Optional[str] = None,
) -> Tuple[Figure, pd.Series]:
    """
    Create CoMut Panel C: Purity heatmap visualization.

    Displays a 1-row heatmap showing tumor purity values (0-1) for each sample.
    Each cell represents one sample, with color intensity indicating purity level
    (darker = higher purity).

    This implementation follows these conventions:
    - Accepts pre-computed purity values (like other clinical data)
    - 1xN heatmap with Blues colormap (vmin=0, vmax=1)
    - Samples ordered consistently with Panels A and B for alignment
    - No colorbar (compact design for multi-panel layout)

    The visualization shows:
    - Y-axis: "Purity" label
    - X-axis: No labels (for compactness)
    - Color: Blues gradient (white/light = low purity, dark blue = high purity)
    - Missing samples: White cells (NaN values)

    Args:
        py_mut: PyMutation object (not used for data, only for consistency)
        purity_df: Pre-computed purity DataFrame.
                  Should have at least 2 columns: sample ID and purity value.
                  Or a Series indexed by sample IDs.
        purity_tsv: Alternative path to TSV file with purity data.
                   Expected columns: sample, value (or purity/tumor_purity).
        sample_order: List of sample IDs to display (and their order).
                     Should match Panels A/B for proper alignment.
                     If None, uses all samples in the order from purity_df.
        figsize: Figure size (width, height) in inches (default: (12, 1.5)).
        title: Plot title (default: None, will show "Purity" as Y-label instead).

    Returns:
        Tuple of (matplotlib.figure.Figure, Series with used purity data)

    Raises:
        ValueError: If neither purity_df nor purity_tsv is provided,
                   or if required data is invalid.

    Example:
        >>> # Load pre-computed purity values
        >>> fig, series = py_mut.comut_purity_plot(
        ...     purity_tsv="purity.tsv",
        ...     sample_order=sample_order  # from Panel A
        ... )
        >>> fig.savefig("comut_panel_c_purity.png")
    """
    start_time = time.time()

    # === 1. Load data ===
    if purity_df is None and purity_tsv is None:
        raise ValueError("Must provide either 'purity_df' or 'purity_tsv'")

    if purity_tsv is not None:
        df = pd.read_csv(purity_tsv, sep="\t")

        # Detect sample column
        sample_col_candidates = [
            "sample",
            "sample_id",
            "tumor_sample_barcode",
            "Tumor_Sample_Barcode",
        ]
        sample_col = next(
            (col for col in sample_col_candidates if col in df.columns), df.columns[0]
        )

        # Detect value column
        value_col_candidates = [
            "value",
            "purity",
            "tumor_purity",
            "Tumor_Purity",
            "pur",
        ]
        value_col = next(
            (col for col in value_col_candidates if col in df.columns),
            df.columns[1] if len(df.columns) > 1 else df.columns[0],
        )

        purity_series = df.set_index(sample_col)[value_col].astype(float)
    else:
        if isinstance(purity_df, pd.Series):
            purity_series = purity_df.copy()
            purity_series.index = purity_series.index.astype(str)
        else:
            # Detect columns from DataFrame
            sample_col_candidates = [
                "sample",
                "sample_id",
                "tumor_sample_barcode",
                "Tumor_Sample_Barcode",
            ]
            sample_col = next(
                (col for col in sample_col_candidates if col in purity_df.columns),
                purity_df.columns[0],
            )

            value_col_candidates = [
                "value",
                "purity",
                "tumor_purity",
                "Tumor_Purity",
                "pur",
            ]
            value_col = next(
                (col for col in value_col_candidates if col in purity_df.columns),
                purity_df.columns[1]
                if len(purity_df.columns) > 1
                else purity_df.columns[0],
            )

            purity_series = purity_df.set_index(sample_col)[value_col].astype(float)

    if purity_series.empty:
        raise ValueError("Purity data is empty")

    # === 2. Convert from percentage to [0, 1] if needed ===
    if purity_series.max() > 1.5:
        purity_series = purity_series / 100.0

    # === 3. Clip to [0, 1] range ===
    purity_series = purity_series.clip(0, 1)

    # === 4. Reindex with sample_order ===
    if sample_order is not None:
        purity_aligned = purity_series.reindex(sample_order)

        n_missing = purity_aligned.isna().sum()
        if n_missing > 0:
            logger.warning(
                f"{n_missing}/{len(sample_order)} samples have no purity data"
            )
    else:
        purity_aligned = purity_series

    # === 5. Create figure with cell size matching CNA plot ===
    # Define colormap for purity values (Blues)
    from matplotlib import cm

    fig, ax = plt.subplots(figsize=figsize)

    # Get number of samples
    n_samples = len(purity_aligned)

    blues_cmap = cm.get_cmap("Blues")

    # Draw individual rectangles with white borders (like Panel F)
    for i, sample in enumerate(purity_aligned.index):
        purity_val = purity_aligned[sample]

        # Determine color based on purity value
        if pd.isna(purity_val):
            # White for missing data
            color = "#ffffff"
        else:
            # Map purity value (0-1) to Blues colormap
            color = blues_cmap(purity_val)

        # Draw rectangle with white border for cell separation
        rect = plt.Rectangle(
            (i - 0.5, -0.5),
            1,
            1,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    # === 6. Styling ===
    # Set axis limits - match Panel F coordinate system
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_ylim(0.5, -0.5)  # Single row

    # Y-axis: label only
    ax.set_yticks([0])
    ax.set_yticklabels(["Purity"], fontsize=10)

    # X-axis: no ticks (samples already labeled in Panel A)
    ax.set_xticks([])
    ax.set_xlabel("")

    # Title
    if title is None:
        title = "Purity"
    ax.set_title(title, loc="center", fontweight="bold", fontsize=12, pad=10)

    # Remove spines for clean look
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    elapsed_time = time.time() - start_time
    logger.info(f"Panel C created in {elapsed_time:.2f}s")

    plt.close(fig)

    return fig, purity_aligned


def _create_comut_mutation_type_plot(
    py_mut: "PyMutation",
    mutation_data_df: Optional[pd.DataFrame] = None,
    mutation_data_tsv: Optional[str] = None,
    sample_order: Optional[List[str]] = None,
    gene_order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 2.4),
    title: Optional[str] = None,
) -> Tuple[Figure, pd.DataFrame]:
    """
    Create CoMut Panel D: Mutation Type (Oncoprint) visualization.

    Displays a gene x sample matrix where each cell shows the predominant mutation
    type for that gene-sample pair. This is an oncoprint-style visualization following
    these conventions:

    This implementation follows these conventions:
    - Accepts pre-computed mutation type data (gene, sample, type)
    - 5 mutation categories: Missense, Nonsense, Frameshift indel, Silent, Multiple
    - Gene order: from top to bottom (typically: MAP3K1, CDH1, TP53, PIK3CA)
    - Sample order: consistent with Panels A-C for perfect alignment
    - No borders on cells, white for missing data

    Mutation type resolution (when multiple mutations exist for same gene-sample):
    - If all types identical → use that type
    - If multiple non-silent types → "Multiple"
    - If Silent + one non-silent type → use the non-silent
    - If only Silent → "Silent"

    Args:
        py_mut: PyMutation object (for consistency, not used for data)
        mutation_data_df: Pre-computed mutation type DataFrame.
                         Expected columns: sample (or similar), gene (or category),
                         mutation_type (or value).
        mutation_data_tsv: Alternative path to TSV file with mutation data.
                          Expected columns: sample, category (gene), value (type).
        sample_order: List of sample IDs to display (and their order).
                     Should match Panels A-C for proper alignment.
                     If None, uses all samples in data.
        gene_order: List of genes to display (and their order, top to bottom).
                   If None, uses all genes in data (ordered by frequency).
        figsize: Figure size (width, height) in inches.
                Default: (12, 4.0) for ~4 genes. Cells are rectangular (taller).
        title: Plot title (default: "D  Mutation Type").

    Returns:
        Tuple of (matplotlib.figure.Figure, DataFrame with mutation matrix)

    Raises:
        ValueError: If neither mutation_data_df nor mutation_data_tsv is provided,
                   or if required data is invalid.

    Example:
        >>> # Using pre-computed mutation data
        >>> fig, matrix = py_mut.comut_mutation_type_plot(
        ...     mutation_data_tsv="mutation_data.tsv",
        ...     sample_order=sample_order,  # from Panel A
        ...     gene_order=["MAP3K1", "CDH1", "TP53", "PIK3CA"]
        ... )
        >>> fig.savefig("comut_panel_d_mutation_type.png")
    """
    start_time = time.time()

    if title is None:
        title = "D  Mutation Type"

    # Load data
    if mutation_data_df is None and mutation_data_tsv is None:
        raise ValueError(
            "Must provide either 'mutation_data_df' or 'mutation_data_tsv'"
        )

    if mutation_data_tsv is not None:
        df = pd.read_csv(mutation_data_tsv, sep="\t")
    else:
        df = mutation_data_df.copy()

    # Normalize column names (case-insensitive)
    df.columns = [col.strip().lower() for col in df.columns]

    sample_col = next(
        (c for c in df.columns if c in {"sample", "sample_id", "tumor_sample_barcode"}),
        df.columns[0],
    )

    gene_col = next(
        (c for c in df.columns if c in {"category", "gene", "hugo_symbol"}), "category"
    )

    type_col = next(
        (c for c in df.columns if c in {"value", "type", "mutation_type"}), "value"
    )

    # Standardize mutation type values
    df[type_col] = df[type_col].str.strip()

    type_mapping = {
        "missense": "Missense",
        "nonsense": "Nonsense",
        "frameshift indel": "Frameshift indel",
        "in frame indel": "Frameshift indel",
        "silent": "Silent",
        "multiple": "Multiple",
    }

    df["mutation_type_clean"] = df[type_col].str.lower().map(type_mapping)
    df_valid = df[df["mutation_type_clean"].notna()].copy()

    # Handle duplicates - resolve to single type or tuple for split cells
    df_check = (
        df_valid.groupby([sample_col, gene_col])["mutation_type_clean"]
        .apply(list)
        .reset_index()
    )

    resolved_data = []
    for idx, row in df_check.iterrows():
        sample = row[sample_col]
        gene = row[gene_col]
        types = row["mutation_type_clean"]

        if len(types) == 1:
            resolved_data.append(
                {sample_col: sample, gene_col: gene, "mutation_type_clean": types[0]}
            )
        elif len(types) == 2 and "Multiple" not in types:
            resolved_data.append(
                {
                    sample_col: sample,
                    gene_col: gene,
                    "mutation_type_clean": tuple(sorted(types)),
                }
            )
        else:
            resolved_data.append(
                {sample_col: sample, gene_col: gene, "mutation_type_clean": "Multiple"}
            )

    df_resolved = pd.DataFrame(resolved_data)

    # Determine gene and sample order
    if gene_order is None:
        gene_counts = df_resolved[gene_col].value_counts()
        gene_order = gene_counts.index.tolist()

    if sample_order is None:
        sample_order = df_resolved[sample_col].unique().tolist()

    # Filter to genes and samples of interest
    df_filtered = df_resolved[
        df_resolved[gene_col].isin(gene_order)
        & df_resolved[sample_col].isin(sample_order)
    ].copy()

    # Create gene x sample matrix
    mutation_matrix = df_filtered.pivot(
        index=gene_col, columns=sample_col, values="mutation_type_clean"
    )
    mutation_matrix = mutation_matrix.reindex(index=gene_order, columns=sample_order)

    # Define mutation type colors
    mutation_colors = {
        "Missense": "#1f5a85",
        "Nonsense": "#8e2c2c",
        "Frameshift indel": "#d8b6cf",
        "Silent": "#b7cfb3",
        "Multiple": "#cfe4f6",
    }

    fig, ax = plt.subplots(figsize=figsize)
    n_genes, n_samples = mutation_matrix.shape

    # Draw cells using same coordinate system as Panel E (CNA)
    # Cells centered at integers, from -0.5 to n-0.5
    for i, gene in enumerate(mutation_matrix.index):
        for j, sample in enumerate(mutation_matrix.columns):
            value = mutation_matrix.loc[gene, sample]

            if pd.notna(value):
                if isinstance(value, tuple):
                    # Split cell with two triangles
                    type1, type2 = value
                    color1 = mutation_colors[type1]
                    color2 = mutation_colors[type2]

                    triangle1 = mpatches.Polygon(
                        [
                            (j - 0.5, i - 0.5),
                            (j + 0.5, i - 0.5),
                            (j - 0.5, i + 0.5),
                        ],
                        facecolor=color1,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    ax.add_patch(triangle1)

                    triangle2 = mpatches.Polygon(
                        [
                            (j + 0.5, i - 0.5),
                            (j + 0.5, i + 0.5),
                            (j - 0.5, i + 0.5),
                        ],
                        facecolor=color2,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    ax.add_patch(triangle2)
                else:
                    # Solid rectangle
                    rect = mpatches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=mutation_colors[value],
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    ax.add_patch(rect)

    # Configure axes - match Panel E coordinate system
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_ylim(n_genes - 0.5, -0.5)  # Inverted for top-to-bottom gene order

    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(mutation_matrix.index, fontsize=10, style="italic")
    ax.set_xticks([])

    # Configure spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_visible(False)

    # Add legend
    mutation_legend_elements = [
        mpatches.Patch(facecolor=mutation_colors[mt], edgecolor="none")
        for mt in ["Missense", "Nonsense", "Frameshift indel", "Silent", "Multiple"]
    ]
    ax.legend(
        mutation_legend_elements,
        ["Missense", "Nonsense", "Frameshift indel", "Silent", "Multiple"],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        title="Mutation Type",
        title_fontproperties={"size": 9, "weight": "bold"},
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.3,
        alignment="left",
    )

    ax.set_title(title, loc="center", fontweight="bold", fontsize=12, pad=10)
    plt.tight_layout()

    elapsed_time = time.time() - start_time
    logger.info(f"Panel D created in {elapsed_time:.2f}s")

    plt.close(fig)

    return fig, mutation_matrix


def _create_comut_cna_plot(
    py_mut: "PyMutation",
    cna_df: Optional[pd.DataFrame] = None,
    cna_tsv: Optional[str] = None,
    sample_order: Optional[List[str]] = None,
    gene_order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 2.0),
    title: Optional[str] = None,
) -> Tuple[Figure, pd.DataFrame]:
    """
    Create CoMut Panel E: Copy Number Alteration (CNA) heatmap.

    Displays a gene x sample heatmap showing the allelic CNA state for each
    gene-sample pair. This panel follows these conventions for CNA visualization.

    CNA states (priority order for duplicates):
    1. aCN = 0 (homozygous deletion) - highest priority
    2. Allelic deletion (heterozygous loss)
    3. Allelic amplification (gain)
    4. Baseline (diploid/neutral) - lowest priority

    Args:
        py_mut: PyMutation object (not directly used, for API consistency)
        cna_df: Pre-loaded DataFrame with CNA data.
                Expected columns: sample, category (gene), value (CNA state)
        cna_tsv: Path to TSV file with CNA data (standard format).
                 Columns: sample, category, value
        sample_order: List of sample IDs in desired order (for alignment with other panels).
                     If None, uses all samples found in data.
        gene_order: List of genes to display (top to bottom order).
                   Example: ["ERBB2", "CDKN2A", "MYC"]
                   If None, uses all genes ordered by frequency.
        figsize: Figure size (width, height) in inches.
                Default: (12, 2.0) for 3 genes.
        title: Plot title (default: "E  Copy Number Alteration").

    Returns:
        Tuple[Figure, DataFrame]: (figure, cna_matrix)
            - figure: matplotlib Figure with CNA heatmap
            - cna_matrix: DataFrame with resolved CNA states (genes x samples)

    Example:
        >>> fig, cna_matrix = py_mut._create_comut_cna_plot(
        ...     cna_tsv="cna.tsv",
        ...     sample_order=sample_order,
        ...     gene_order=["ERBB2", "CDKN2A", "MYC"]
        ... )
    """
    start_time = time.time()

    # Load CNA data
    if cna_df is not None:
        data = cna_df.copy()
    elif cna_tsv is not None:
        cna_path = Path(cna_tsv)
        if not cna_path.exists():
            raise FileNotFoundError(f"CNA TSV file not found: {cna_tsv}")
        data = pd.read_csv(cna_path, sep="\t")
    else:
        raise ValueError("Must provide either cna_df or cna_tsv")

    # Normalize column names
    col_mapping = {}
    for col in data.columns:
        col_lower = col.lower().strip()
        if col_lower in ["sample", "sample_id", "tumor_sample_barcode"]:
            col_mapping[col] = "sample"
        elif col_lower in ["category", "gene", "hugo_symbol"]:
            col_mapping[col] = "gene"
        elif col_lower in ["value", "cna_state", "state"]:
            col_mapping[col] = "cna_state"

    data = data.rename(columns=col_mapping)

    required = ["sample", "gene", "cna_state"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(
            f"CNA data missing required columns: {missing}. "
            f"Available: {list(data.columns)}"
        )

    # Normalize CNA states
    data["cna_state"] = data["cna_state"].str.strip().str.lower()

    state_mapping = {
        "acn = 0": "aCN = 0",
        "acn=0": "aCN = 0",
        "allelic deletion": "Allelic deletion",
        "allelic amplification": "Allelic amplification",
        "baseline": "Baseline",
    }
    data["cna_state"] = data["cna_state"].map(state_mapping)

    unknown = data["cna_state"].isna()
    if unknown.any():
        logger.warning(
            f"Found {unknown.sum()} rows with unknown CNA states, dropping them"
        )
        data = data[~unknown].copy()

    # Handle duplicates - keep ALL states (including duplicates) for split cells
    grouped = data.groupby(["sample", "gene"])["cna_state"].apply(list).reset_index()
    grouped.columns = ["sample", "gene", "cna_states"]
    data = grouped

    # Determine sample and gene order
    all_samples = data["sample"].unique().tolist()
    all_genes = data["gene"].unique().tolist()

    if sample_order is not None:
        final_samples = [s for s in sample_order if s in all_samples]
        missing_samples = len(sample_order) - len(final_samples)
        if missing_samples > 0:
            logger.warning(
                f"{missing_samples} samples from sample_order not found in CNA data"
            )
    else:
        final_samples = sorted(all_samples)

    if gene_order is not None:
        final_genes = [g for g in gene_order if g in all_genes]
        missing_genes = len(gene_order) - len(final_genes)
        if missing_genes > 0:
            logger.warning(
                f"{missing_genes} genes from gene_order not found in CNA data"
            )
    else:
        gene_counts = data["gene"].value_counts()
        final_genes = gene_counts.index.tolist()

    n_samples = len(final_samples)
    n_genes = len(final_genes)

    # Create CNA matrix (genes x samples)
    cna_matrix = pd.DataFrame(index=final_genes, columns=final_samples, dtype="object")

    for _, row in data.iterrows():
        sample = row["sample"]
        gene = row["gene"]
        states = row["cna_states"]

        if sample in final_samples and gene in final_genes:
            cna_matrix.loc[gene, sample] = states

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    cna_colors = {
        "Allelic amplification": "#cfe4f6",
        "Baseline": "#e9e9e9",
        "Allelic deletion": "#1f5a85",
        "aCN = 0": "#8b8b8b",
        "NA": "#ffffff",
    }

    from matplotlib.patches import Polygon, Rectangle

    # Draw cells
    for i, gene in enumerate(final_genes):
        for j, sample in enumerate(final_samples):
            cell_value = cna_matrix.loc[gene, sample]

            if cell_value is None or (
                isinstance(cell_value, float) and pd.isna(cell_value)
            ):
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=cna_colors["NA"],
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax.add_patch(rect)
            else:
                states = cell_value if isinstance(cell_value, list) else [cell_value]

                if len(states) == 1:
                    color = cna_colors.get(states[0], cna_colors["NA"])
                    rect = Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    ax.add_patch(rect)
                else:
                    # Sort by priority for consistent split cell ordering
                    priority_map = {
                        "aCN = 0": 4,
                        "Allelic deletion": 3,
                        "Allelic amplification": 2,
                        "Baseline": 1,
                    }
                    sorted_states = sorted(
                        states, key=lambda x: priority_map.get(x, 0), reverse=True
                    )

                    state1 = sorted_states[0]
                    state2 = sorted_states[1] if len(sorted_states) > 1 else state1

                    color1 = cna_colors.get(state1, cna_colors["NA"])
                    color2 = cna_colors.get(state2, cna_colors["NA"])

                    triangle1 = Polygon(
                        [
                            (j - 0.5, i - 0.5),
                            (j + 0.5, i - 0.5),
                            (j - 0.5, i + 0.5),
                        ],
                        facecolor=color1,
                        edgecolor="none",
                    )
                    ax.add_patch(triangle1)

                    triangle2 = Polygon(
                        [
                            (j + 0.5, i - 0.5),
                            (j + 0.5, i + 0.5),
                            (j - 0.5, i + 0.5),
                        ],
                        facecolor=color2,
                        edgecolor="none",
                    )
                    ax.add_patch(triangle2)

                    # Add diagonal divider line
                    from matplotlib.lines import Line2D

                    divider = Line2D(
                        [j - 0.5, j + 0.5],
                        [i + 0.5, i - 0.5],
                        color="white",
                        linewidth=0.5,
                        zorder=10,
                    )
                    ax.add_line(divider)

                    border = Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    ax.add_patch(border)

    # Configure axes
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_ylim(n_genes - 0.5, -0.5)

    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(final_genes, style="italic", fontsize=10)
    ax.set_ylabel("")

    ax.set_xticks([])
    ax.set_xlabel("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    if title is None:
        title = "E  Copy Number Alteration"
    ax.set_title(title, fontsize=12, fontweight="bold", loc="center", pad=10)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=cna_colors["Allelic amplification"],
            edgecolor="none",
            label="Allelic amplification",
        ),
        Patch(facecolor=cna_colors["Baseline"], edgecolor="none", label="Baseline"),
        Patch(
            facecolor=cna_colors["Allelic deletion"],
            edgecolor="none",
            label="Allelic deletion",
        ),
        Patch(facecolor=cna_colors["aCN = 0"], edgecolor="none", label="aCN = 0"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        title="Copy Number Alteration",
        title_fontproperties={"size": 9, "weight": "bold"},
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.3,
        alignment="left",
    )

    plt.tight_layout()

    elapsed_time = time.time() - start_time
    logger.info(f"Panel E created in {elapsed_time:.2f}s")

    plt.close(fig)

    return fig, cna_matrix


def _create_comut_wgd_plot(
    py_mut: "PyMutation",
    wgd_df: Optional[pd.DataFrame] = None,
    wgd_tsv: Optional[str] = None,
    sample_order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 1.2),
    title: Optional[str] = None,
) -> Tuple[Figure, pd.Series]:
    """
    Create CoMut Panel F: Whole Genome Doubling (WGD) track.

    Displays a single-row track with gray rectangles for samples with WGD=Yes
    and blank (white) for samples with WGD=No or missing data.

    This panel follows these conventions for WGD visualization:
    - Gray (#8b8b8b) rectangle: WGD = Yes
    - Blank (white): WGD = No or missing
    - Small gaps between rectangles for visual clarity

    WGD data interpretation (flexible column naming and value encoding):
    - Sample column: sample, sample_id, tumor_sample_barcode (case-insensitive)
    - WGD column: wgd, status, value (case-insensitive)
    - WGD values (normalized to boolean):
      * Yes/1/True/Y/T → True (WGD present)
      * No/0/False/N/F → False (no WGD)
      * Missing/NaN → NaN (drawn as blank)

    For duplicate sample entries, groups by sample and uses any() → True if
    any row indicates WGD.

    Args:
        py_mut: PyMutation object (not directly used, included for API consistency)
        wgd_df: Pre-loaded DataFrame with WGD data.
               Expected columns: sample, category/wgd/status, value
        wgd_tsv: Path to TSV file with WGD data (standard format).
                Expected columns: sample, category, value
                Values: "Yes"/"No" or similar boolean encodings
        sample_order: List of sample IDs in desired order (for alignment with Panels A-E).
                     Should match the order used in mutation burden plot.
                     If None, uses all samples found in data.
        figsize: Figure size (width, height) in inches.
                Default: (12, 0.8) for single-row track. Keep height small.
        title: Plot title (default: "F  Whole Genome Doubling").

    Returns:
        Tuple[Figure, pd.Series]: (WGD figure, WGD status series indexed by sample)

    Example:
        >>> # Align with other CoMut panels
        >>> sample_order = ['SAMPLE1', 'SAMPLE2', ...]  # from Panel A
        >>> fig = py_mut.comut_wgd_plot(
        ...     wgd_tsv="wgd.tsv",
        ...     sample_order=sample_order
        ... )
        >>> fig.savefig("comut_panel_f_wgd.png", dpi=300, bbox_inches='tight')
    """
    start_time = time.time()

    # Load WGD data
    if wgd_df is None and wgd_tsv is None:
        raise ValueError("Either wgd_df or wgd_tsv must be provided")

    if wgd_df is None:
        wgd_path = Path(wgd_tsv)
        if not wgd_path.exists():
            raise FileNotFoundError(f"WGD file not found: {wgd_tsv}")
        wgd_df = pd.read_csv(wgd_path, sep="\t")

    # Validate WGD data
    if wgd_df.empty:
        raise ValueError("WGD data is empty")

    # Identify sample column (case-insensitive)
    sample_col_candidates = ["sample", "sample_id", "tumor_sample_barcode"]
    sample_col = None
    for col in wgd_df.columns:
        if col.lower() in sample_col_candidates:
            sample_col = col
            break

    if sample_col is None:
        raise ValueError(
            f"Could not identify sample column. Expected one of: {sample_col_candidates}. "
            f"Found columns: {wgd_df.columns.tolist()}"
        )

    # Identify WGD status column (case-insensitive)
    # Priority order: value > wgd > status (skip category if it contains constant labels)
    wgd_col_candidates = ["value", "wgd", "status"]
    wgd_col = None
    for candidate in wgd_col_candidates:
        for col in wgd_df.columns:
            if col.lower() == candidate and col != sample_col:
                wgd_col = col
                break
        if wgd_col is not None:
            break

    if wgd_col is None:
        raise ValueError(
            f"Could not identify WGD status column. Expected one of: {wgd_col_candidates}. "
            f"Found columns: {wgd_df.columns.tolist()}"
        )

    # Normalize WGD values to boolean
    # Yes/1/True/Y/T → True; No/0/False/N/F → False; else NaN
    def normalize_wgd_value(val):
        """Convert various WGD encodings to boolean."""
        if pd.isna(val):
            return np.nan

        # Convert to string and normalize
        val_str = str(val).strip().upper()

        # True cases
        if val_str in {"YES", "1", "TRUE", "Y", "T"}:
            return True
        # False cases
        elif val_str in {"NO", "0", "FALSE", "N", "F"}:
            return False
        else:
            logger.warning(f"Unknown WGD value: '{val}', treating as NaN")
            return np.nan

    wgd_df["wgd_normalized"] = wgd_df[wgd_col].apply(normalize_wgd_value)

    # Handle duplicates: group by sample and use any() → True if any row indicates WGD
    wgd_grouped = wgd_df.groupby(sample_col)["wgd_normalized"].apply(
        lambda x: x.any() if x.notna().any() else np.nan
    )

    # If sample_order not provided, use all samples from WGD data
    if sample_order is None:
        sample_order = wgd_grouped.index.tolist()

    # Create WGD series aligned to sample_order (use object dtype to avoid FutureWarning)
    wgd_series = pd.Series(index=sample_order, dtype=object)
    for sample in sample_order:
        if sample in wgd_grouped.index:
            wgd_series[sample] = wgd_grouped[sample]
        else:
            wgd_series[sample] = np.nan

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set title
    if title is None:
        title = "Whole Genome Doubling"
    ax.set_title(title, fontsize=12, weight="bold", loc="center", pad=10)

    # Define colors
    WGD_YES_COLOR = "#8b8b8b"  # Gray for WGD=Yes
    WGD_NO_COLOR = "#ffffff"  # White for WGD=No/Missing
    WGD_BORDER_COLOR = "#e0e0e0"  # Light gray border for "No" cells

    # Draw rectangles using same coordinate system as Panel E (CNA)
    # Cells centered at integers, from -0.5 to n-0.5
    n_samples = len(sample_order)

    for i, sample in enumerate(sample_order):
        wgd_status = wgd_series[sample]

        # Determine color and border
        if pd.isna(wgd_status):
            color = WGD_NO_COLOR
        elif wgd_status:
            color = WGD_YES_COLOR
        else:
            color = WGD_NO_COLOR

        # Draw rectangle - same coordinates as Panel E
        # All cells have white border for separation (matching Panel E)
        rect = plt.Rectangle(
            (i - 0.5, -0.5),
            1,
            1,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    # Set axis limits - match Panel E coordinate system
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_ylim(0.5, -0.5)  # Single row

    # Y-axis: label only
    ax.set_yticks([0])
    ax.set_yticklabels(["WGD"], fontsize=10)

    # X-axis: no ticks (samples already labeled in Panel A)
    ax.set_xticks([])
    ax.set_xlabel("")

    # Remove spines for clean look
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    # Add legend matching CNA style
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=WGD_NO_COLOR, edgecolor=WGD_BORDER_COLOR, linewidth=0.5, label="No"),
        Patch(facecolor=WGD_YES_COLOR, edgecolor="none", label="Yes"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        title="Whole Genome Doubling",
        title_fontproperties={"size": 9, "weight": "bold"},
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.3,
        alignment="left",
    )

    plt.tight_layout()

    elapsed_time = time.time() - start_time
    logger.info(f"Panel F created in {elapsed_time:.2f}s")

    # Close to prevent duplicate display in notebooks
    plt.close(fig)

    return fig, wgd_series


def _create_comut_same_patient_plot(
    py_mut: "PyMutation",
    sp_df: Optional[pd.DataFrame] = None,
    sp_tsv: Optional[str] = None,
    sample_order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 1.2),
    title: Optional[str] = None,
) -> Tuple[Figure, pd.DataFrame]:
    """
    Create Panel G: Same Patient track.

    This panel shows which samples belong to the same patient by drawing:
    - A black dot for each sample (single horizontal track)
    - Horizontal black lines connecting dots of samples from the same patient

    Replicates these conventions for the "Same Patient" panel:
    - All samples have a dot (marker) in a single Y position
    - Samples from the same patient (≥2 samples) are connected by lines
    - Lines connect consecutive samples within the patient group (in sample_order)
    - Samples not in sp.tsv or without a partner show only a dot

    Args:
        py_mut: PyMutation object (for consistency, not directly used)
        sp_df: DataFrame with columns ['sample', 'group'] where 'group' identifies the patient.
              If provided, sp_tsv is ignored.
        sp_tsv: Path to TSV file with columns: sample, group (patient ID)
        sample_order: List of sample IDs in desired order (x-axis).
                     If None, uses all samples from sp data.
        figsize: Figure size (width, height)
        title: Panel title (default: "G Same Patient")

    Returns:
        Tuple[Figure, pd.DataFrame]: (figure, patient_mapping_df)

    Raises:
        ValueError: If neither sp_df nor sp_tsv is provided
        FileNotFoundError: If sp_tsv path doesn't exist

    Example:
        >>> fig, df = py_mut.comut_same_patient_plot(
        ...     sp_tsv="sp.tsv",
        ...     sample_order=sample_order,
        ...     figsize=(12, 1.2)
        ... )
    """
    start_time = time.time()

    # Load data

    # ========================================================================
    # 1. Load and validate patient data
    # ========================================================================
    if sp_df is None and sp_tsv is None:
        raise ValueError("Either sp_df or sp_tsv must be provided")

    if sp_df is None:
        sp_path = Path(sp_tsv)
        if not sp_path.exists():
            raise FileNotFoundError(f"Same patient file not found: {sp_tsv}")
        sp_df = pd.read_csv(sp_path, sep="\t")

    # Validate columns
    required_cols = ["sample", "group"]
    if not all(col in sp_df.columns for col in required_cols):
        raise ValueError(
            f"sp data must contain columns: {required_cols}. Found: {sp_df.columns.tolist()}"
        )

    # Normalize column names to lowercase for case-insensitive matching
    sp_df = sp_df.copy()
    sp_df.columns = sp_df.columns.str.lower()

    # Convert sample IDs to strings
    sp_df["sample"] = sp_df["sample"].astype(str)

    # ========================================================================
    # 2. Determine sample order
    # ========================================================================
    if sample_order is None:
        sample_order = sp_df["sample"].tolist()
    else:
        sample_order = [str(s) for s in sample_order]

    # ========================================================================
    # 3. Create patient groups (only samples in sample_order)
    # ========================================================================
    # Filter sp_df to only include samples in sample_order
    sp_filtered = sp_df[sp_df["sample"].isin(sample_order)].copy()

    if len(sp_filtered) == 0:
        logger.warning(
            "No samples from sp data found in sample_order. Showing only dots."
        )

    # Group samples by patient
    patient_groups = {}
    for patient_id, group_df in sp_filtered.groupby("group"):
        samples_in_group = group_df["sample"].tolist()

        # Sort by sample_order to maintain visual alignment
        samples_sorted = [s for s in sample_order if s in samples_in_group]

        if len(samples_sorted) >= 2:
            patient_groups[patient_id] = samples_sorted

    # ========================================================================
    # 4. Create figure
    # ========================================================================
    fig, ax = plt.subplots(figsize=figsize)

    # Set title
    if title is None:
        title = "Same Patient"
    ax.set_title(title, fontsize=12, fontweight="bold", loc="center", pad=10)

    # ========================================================================
    # 5. Plot dots for all samples (single Y position)
    # ========================================================================
    # Map sample IDs to x positions (0-indexed)
    sample_to_x = {sample: i for i, sample in enumerate(sample_order)}
    n_samples = len(sample_order)

    # Single Y position for all samples
    y_pos = 0.5

    # Draw dots for all samples
    x_positions = list(range(n_samples))
    y_positions = [y_pos] * n_samples

    ax.scatter(
        x_positions,
        y_positions,
        color="black",
        s=30,
        zorder=10,
        edgecolors="none",
        alpha=1.0,
    )

    # ========================================================================
    # 6. Draw connecting lines for patients with ≥2 samples
    # ========================================================================
    for patient_id, samples in patient_groups.items():
        # Get x positions for this patient's samples (in sample_order)
        x_coords = [sample_to_x[s] for s in samples if s in sample_to_x]

        if len(x_coords) < 2:
            continue

        # Draw line segments between consecutive samples
        for i in range(len(x_coords) - 1):
            x1 = x_coords[i]
            x2 = x_coords[i + 1]
            ax.plot(
                [x1, x2],
                [y_pos, y_pos],
                color="black",
                linewidth=1.5,
                zorder=5,
                solid_capstyle="butt",
            )

    # ========================================================================
    # 7. Add legend (positioned outside plot area, to the right)
    # ========================================================================
    # Create dummy elements for legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markersize=4,
            linestyle="-",
            linewidth=1.5,
        )
    ]

    # Add legend matching create_comut_plot style
    ax.legend(
        handles=legend_elements,
        labels=[r"$\mathbf{Same\ Patient}$"],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.3,
        markerfirst=False,
        alignment="left",
    )

    # ========================================================================
    # 8. Configure axes
    # ========================================================================
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_xticks([])
    ax.set_xlabel("")

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("")

    # Remove spines
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    elapsed_time = time.time() - start_time
    logger.info(f"Panel G created in {elapsed_time:.2f}s")

    # Close to prevent duplicate display in notebooks
    plt.close(fig)

    return fig, sp_filtered


def create_comut_plot(
    py_mut: "PyMutation",
    sample_order: Optional[List[str]] = None,
    gene_order: Optional[List[str]] = None,
    cna_gene_order: Optional[List[str]] = None,
    signatures_tsv: Optional[str] = None,
    purity_tsv: Optional[str] = None,
    mutation_data_tsv: Optional[str] = None,
    cna_tsv: Optional[str] = None,
    wgd_tsv: Optional[str] = None,
    sp_tsv: Optional[str] = None,
    territory_bp: int = DEFAULT_TERRITORY_BP,
    max_samples: Optional[int] = DEFAULT_MAX_SAMPLES,
    somatic_only: bool = True,
    pass_only: bool = True,
    signature_labels: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """
    Create complete CoMut plot with panels A-G stacked vertically.

    This function creates a comprehensive multi-panel visualization combining:
    - Panel A: Mutation burden (synonymous/non-synonymous)
    - Panel B: Mutational signatures
    - Panel C: Purity heatmap
    - Panel D: Mutation type (oncoprint)
    - Panel E: Copy Number Alteration (CNA)
    - Panel F: Whole Genome Doubling (WGD)
    - Panel G: Same Patient (samples from same patient connected by lines)

    All panels are aligned by sample order for perfect visual coherence.
    The panels are rendered as images and combined into a single figure.

    Args:
        py_mut: PyMutation object containing mutation data
        sample_order: List of sample IDs in desired order (for all panels).
                     If None, will be calculated from mutation burden.
        gene_order: List of genes to display in Panel D (top to bottom).
                   Example: ["MAP3K1", "CDH1", "TP53", "PIK3CA"]
        cna_gene_order: List of genes to display in Panel E (top to bottom).
                       Example: ["ERBB2", "CDKN2A", "MYC"]
                       If None, uses all genes from CNA data.
        signatures_tsv: Path to TSV file with signature contributions (Panel B).
        purity_tsv: Path to TSV file with purity data (Panel C).
        mutation_data_tsv: Path to TSV file with mutation type data (Panel D).
        cna_tsv: Path to TSV file with CNA data (Panel E).
        wgd_tsv: Path to TSV file with WGD data (Panel F).
        sp_tsv: Path to TSV file with same patient data (Panel G).
               Format: 'sample' and 'group' columns (group = patient ID).
        territory_bp: Territory size in bp for TMB normalization (Panel A).
        max_samples: Maximum number of samples to display (if sample_order=None).
        somatic_only: If True, filter to somatic variants (Panel A).
        pass_only: If True, filter to PASS variants (Panel A).
        signature_labels: Labels for signatures in Panel B.
        figsize: Overall figure size (width, height). If None, auto-calculated based on panels.

    Returns:
        matplotlib.figure.Figure: Complete CoMut plot with all 7 panels stacked vertically

    Example:
        >>> fig = py_mut.create_comut_plot(
        ...     sample_order=sample_order,
        ...     gene_order=["MAP3K1", "CDH1", "TP53", "PIK3CA"],
        ...     cna_gene_order=["ERBB2", "CDKN2A", "MYC"],
        ...     signatures_tsv="sig_contribution.tsv",
        ...     purity_tsv="purity.tsv",
        ...     mutation_data_tsv="mutation_data.tsv",
        ...     cna_tsv="cna.tsv",
        ...     wgd_tsv="wgd.tsv",
        ...     sp_tsv="sp.tsv",
        ...     territory_bp=60456963,
        ...     signature_labels=["Sig 1", "Sig 2", "Sig 3", "Sig 4"]
        ... )
        >>> fig.savefig("comut_plot_complete.png", dpi=300, bbox_inches='tight')
    """
    start_time = time.time()

    logger.info("Creating complete CoMut plot (Panels A-G)")

    # Disable interactive mode to prevent figures from showing automatically
    was_interactive = plt.isinteractive()
    plt.ioff()

    try:
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch, Polygon, Rectangle

        # ============================================================
        # STEP 1: Load and prepare all data first
        # ============================================================

        # --- Panel A: Mutation Burden Data ---
        data_a = _reload_with_silent_variants(py_mut, "Variant_Classification")

        if somatic_only and "Variant_Status" in data_a.columns:
            data_a = data_a[data_a["Variant_Status"].str.upper() == "SOMATIC"]
        if pass_only and "FILTER" in data_a.columns:
            data_a = data_a[data_a["FILTER"].isin(["PASS", "."])]

        # Remove duplicates
        dup_cols = []
        for col in [
            "Chromosome",
            "Start_Position",
            "End_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele2",
        ]:
            if col in data_a.columns:
                dup_cols.append(col)
        if dup_cols:
            dup_cols.append("Tumor_Sample_Barcode")
            data_a = data_a.drop_duplicates(subset=dup_cols)

        tmb_df = _calculate_tmb_per_sample(
            data_a, "Tumor_Sample_Barcode", "Variant_Classification", territory_bp
        )

        # Determine final sample order
        if sample_order is not None:
            available = set(tmb_df["sample"])
            final_sample_order = [s for s in sample_order if s in available]
        else:
            final_sample_order = _order_samples_by_tmb(tmb_df)
            if max_samples is not None:
                final_sample_order = final_sample_order[:max_samples]

        n_samples = len(final_sample_order)
        logger.info(f"Final sample order: {n_samples} samples")

        # Filter TMB data
        tmb_df = tmb_df[tmb_df["sample"].isin(final_sample_order)].copy()
        tmb_df["sample"] = pd.Categorical(
            tmb_df["sample"], categories=final_sample_order, ordered=True
        )
        tmb_df = tmb_df.sort_values("sample")

        # --- Panel B: Signatures Data ---
        if signatures_tsv is not None:
            sig_df = pd.read_csv(signatures_tsv, sep="\t")
            sample_col_candidates = [
                "sample",
                "sample_id",
                "tumor_sample_barcode",
                "Tumor_Sample_Barcode",
            ]
            sample_col = next(
                (col for col in sample_col_candidates if col in sig_df.columns),
                sig_df.columns[0],
            )
            sig_df[sample_col] = sig_df[sample_col].astype(str)
            sig_df = sig_df.set_index(sample_col)

            if signature_labels is not None:
                for sig in signature_labels:
                    if sig not in sig_df.columns:
                        sig_df[sig] = 0.0
                sig_df = sig_df[signature_labels]
            else:
                signature_labels = list(sig_df.columns)

            sig_df = sig_df.reindex(
                [s for s in final_sample_order if s in sig_df.index]
            )
            row_sums = sig_df.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=0.01):
                sig_df = sig_df.div(row_sums, axis=0).fillna(0).clip(lower=0, upper=1)
        else:
            sig_df = None

        # --- Panel C: Purity Data ---
        if purity_tsv is not None:
            pur_df = pd.read_csv(purity_tsv, sep="\t")
            sample_col_candidates = [
                "sample",
                "sample_id",
                "tumor_sample_barcode",
                "Tumor_Sample_Barcode",
            ]
            sample_col = next(
                (col for col in sample_col_candidates if col in pur_df.columns),
                pur_df.columns[0],
            )
            value_col_candidates = [
                "value",
                "purity",
                "tumor_purity",
                "Tumor_Purity",
                "pur",
            ]
            value_col = next(
                (col for col in value_col_candidates if col in pur_df.columns),
                pur_df.columns[1],
            )
            purity_series = pur_df.set_index(sample_col)[value_col].astype(float)
            if purity_series.max() > 1.5:
                purity_series = purity_series / 100.0
            purity_series = purity_series.clip(0, 1)
            purity_aligned = purity_series.reindex(final_sample_order)
        else:
            purity_aligned = None

        # --- Panel D: Mutation Type Data ---
        if mutation_data_tsv is not None:
            mut_df = pd.read_csv(mutation_data_tsv, sep="\t")
            mut_df.columns = [col.strip().lower() for col in mut_df.columns]
            sample_col = next(
                (
                    c
                    for c in mut_df.columns
                    if c in {"sample", "sample_id", "tumor_sample_barcode"}
                ),
                mut_df.columns[0],
            )
            gene_col = next(
                (c for c in mut_df.columns if c in {"category", "gene", "hugo_symbol"}),
                "category",
            )
            type_col = next(
                (c for c in mut_df.columns if c in {"value", "type", "mutation_type"}),
                "value",
            )

            type_mapping = {
                "missense": "Missense",
                "nonsense": "Nonsense",
                "frameshift indel": "Frameshift indel",
                "in frame indel": "Frameshift indel",
                "silent": "Silent",
                "multiple": "Multiple",
            }
            mut_df["mutation_type_clean"] = (
                mut_df[type_col].str.strip().str.lower().map(type_mapping)
            )
            mut_df_valid = mut_df[mut_df["mutation_type_clean"].notna()].copy()

            df_check = (
                mut_df_valid.groupby([sample_col, gene_col])["mutation_type_clean"]
                .apply(list)
                .reset_index()
            )
            resolved_data = []
            for _, row in df_check.iterrows():
                types = row["mutation_type_clean"]
                if len(types) == 1:
                    resolved_data.append(
                        {
                            sample_col: row[sample_col],
                            gene_col: row[gene_col],
                            "mutation_type_clean": types[0],
                        }
                    )
                elif len(types) == 2 and "Multiple" not in types:
                    resolved_data.append(
                        {
                            sample_col: row[sample_col],
                            gene_col: row[gene_col],
                            "mutation_type_clean": tuple(sorted(types)),
                        }
                    )
                else:
                    resolved_data.append(
                        {
                            sample_col: row[sample_col],
                            gene_col: row[gene_col],
                            "mutation_type_clean": "Multiple",
                        }
                    )

            df_resolved = pd.DataFrame(resolved_data)
            final_genes_d = (
                gene_order
                if gene_order
                else df_resolved[gene_col].value_counts().index.tolist()
            )
            df_filtered = df_resolved[
                df_resolved[gene_col].isin(final_genes_d)
                & df_resolved[sample_col].isin(final_sample_order)
            ]
            mutation_matrix = df_filtered.pivot(
                index=gene_col, columns=sample_col, values="mutation_type_clean"
            )
            mutation_matrix = mutation_matrix.reindex(
                index=final_genes_d, columns=final_sample_order
            )
        else:
            mutation_matrix = None
            final_genes_d = []

        # --- Panel E: CNA Data ---
        if cna_tsv is not None:
            cna_data = pd.read_csv(cna_tsv, sep="\t")
            col_mapping = {}
            for col in cna_data.columns:
                col_lower = col.lower().strip()
                if col_lower in ["sample", "sample_id", "tumor_sample_barcode"]:
                    col_mapping[col] = "sample"
                elif col_lower in ["category", "gene", "hugo_symbol"]:
                    col_mapping[col] = "gene"
                elif col_lower in ["value", "cna_state", "state"]:
                    col_mapping[col] = "cna_state"
            cna_data = cna_data.rename(columns=col_mapping)
            cna_data["cna_state"] = cna_data["cna_state"].str.strip().str.lower()
            state_mapping = {
                "acn = 0": "aCN = 0",
                "acn=0": "aCN = 0",
                "allelic deletion": "Allelic deletion",
                "allelic amplification": "Allelic amplification",
                "baseline": "Baseline",
            }
            cna_data["cna_state"] = cna_data["cna_state"].map(state_mapping)
            cna_data = cna_data[cna_data["cna_state"].notna()]

            grouped = (
                cna_data.groupby(["sample", "gene"])["cna_state"]
                .apply(list)
                .reset_index()
            )
            grouped.columns = ["sample", "gene", "cna_states"]

            final_genes_e = (
                cna_gene_order
                if cna_gene_order
                else cna_data["gene"].value_counts().index.tolist()
            )
            cna_matrix = pd.DataFrame(
                index=final_genes_e, columns=final_sample_order, dtype="object"
            )
            for _, row in grouped.iterrows():
                if row["sample"] in final_sample_order and row["gene"] in final_genes_e:
                    cna_matrix.loc[row["gene"], row["sample"]] = row["cna_states"]
        else:
            cna_matrix = None
            final_genes_e = []

        # --- Panel F: WGD Data ---
        if wgd_tsv is not None:
            wgd_df = pd.read_csv(wgd_tsv, sep="\t")
            sample_col_candidates = ["sample", "sample_id", "tumor_sample_barcode"]
            sample_col = next(
                (col for col in wgd_df.columns if col.lower() in sample_col_candidates),
                None,
            )
            wgd_col_candidates = ["value", "wgd", "status"]
            wgd_col = next(
                (
                    candidate
                    for candidate in wgd_col_candidates
                    for col in wgd_df.columns
                    if col.lower() == candidate and col != sample_col
                ),
                None,
            )

            def normalize_wgd_value(val):
                if pd.isna(val):
                    return np.nan
                val_str = str(val).strip().upper()
                if val_str in {"YES", "1", "TRUE", "Y", "T"}:
                    return True
                elif val_str in {"NO", "0", "FALSE", "N", "F"}:
                    return False
                return np.nan

            wgd_df["wgd_normalized"] = wgd_df[wgd_col].apply(normalize_wgd_value)
            wgd_grouped = wgd_df.groupby(sample_col)["wgd_normalized"].apply(
                lambda x: x.any() if x.notna().any() else np.nan
            )
            wgd_series = pd.Series(index=final_sample_order, dtype=object)
            for sample in final_sample_order:
                wgd_series[sample] = wgd_grouped.get(sample, np.nan)
        else:
            wgd_series = None

        # --- Panel G: Same Patient Data ---
        if sp_tsv is not None:
            sp_df = pd.read_csv(sp_tsv, sep="\t")
            sample_col = next(
                (
                    col
                    for col in sp_df.columns
                    if col.lower() in ["sample", "sample_id"]
                ),
                sp_df.columns[0],
            )
            group_col = next(
                (
                    col
                    for col in sp_df.columns
                    if col.lower() in ["group", "patient", "patient_id"]
                ),
                sp_df.columns[1],
            )
            sp_df = sp_df[[sample_col, group_col]].copy()
            sp_df.columns = ["sample", "group"]
            sp_df = sp_df[sp_df["sample"].isin(final_sample_order)]
        else:
            sp_df = None

        # ============================================================
        # STEP 2: Create figure with proper layout
        # ============================================================

        # Calculate number of genes for panels D and E
        n_genes_d = len(final_genes_d) if mutation_matrix is not None else 0
        n_genes_e = len(final_genes_e) if cna_matrix is not None else 0

        # Height ratios based on content
        # Panels A, B: bar charts (fixed height)
        # Panels C, F: single row heatmaps
        # Panels D, E: multi-row heatmaps (height depends on genes)
        # Panel G: same patient track
        h_a, h_b = 1.5, 1.6
        h_c = 0.6
        h_d = max(1.5, n_genes_d * 0.6) if n_genes_d > 0 else 1.5
        h_e = max(1.0, n_genes_e * 0.6) if n_genes_e > 0 else 1.0
        h_f = 0.6
        h_g = 0.8

        height_ratios = [h_a, h_b, h_c, h_d, h_e, h_f, h_g]
        total_height = sum(height_ratios) + 1.5  # Extra space for titles/legends

        if figsize is None:
            figsize = (14, total_height)
        
        fig = plt.figure(figsize=figsize)

        # GridSpec with 2 columns: main plot area (left) + legend area (right)
        gs = GridSpec(
            7,
            2,
            figure=fig,
            height_ratios=height_ratios,
            width_ratios=[0.85, 0.15],
            hspace=0.20,
            wspace=0.02,
        )

        # Create axes for each panel (main plot column)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[1, 0])
        ax_c = fig.add_subplot(gs[2, 0])
        ax_d = fig.add_subplot(gs[3, 0])
        ax_e = fig.add_subplot(gs[4, 0])
        ax_f = fig.add_subplot(gs[5, 0])
        ax_g = fig.add_subplot(gs[6, 0])

        # Legend axes (right column) - separate for each panel
        ax_leg_a = fig.add_subplot(gs[0, 1])
        ax_leg_b = fig.add_subplot(gs[1, 1])
        ax_leg_c = fig.add_subplot(gs[2, 1])  # Empty (no legend for Purity)
        ax_leg_d = fig.add_subplot(gs[3, 1])  # Mutation Type legend
        ax_leg_e = fig.add_subplot(gs[4, 1])  # CNA legend
        ax_leg_f = fig.add_subplot(gs[5, 1])  # WGD legend
        ax_leg_g = fig.add_subplot(gs[6, 1])

        for ax_leg in [
            ax_leg_a,
            ax_leg_b,
            ax_leg_c,
            ax_leg_d,
            ax_leg_e,
            ax_leg_f,
            ax_leg_g,
        ]:
            ax_leg.axis("off")

        # Common X limits for alignment (cell-based coordinate system)
        x_lim = (-0.5, n_samples - 0.5)

        # ============================================================
        # STEP 3: Draw Panel A - Mutation Burden
        # ============================================================
        x = np.arange(n_samples)
        color_nonsyn = "#2E5C8A"
        color_syn = "#A8C5E0"

        ax_a.bar(x, tmb_df["tmb_nonsyn"].values, color=color_nonsyn, width=0.8)
        ax_a.bar(
            x,
            tmb_df["tmb_syn"].values,
            bottom=tmb_df["tmb_nonsyn"].values,
            color=color_syn,
            width=0.8,
        )

        ax_a.set_xlim(x_lim)
        ax_a.set_ylim(bottom=0)
        ax_a.set_ylabel("Muts/Mb", fontsize=10)
        ax_a.set_title(
            "A  Mutation Classification",
            fontsize=12,
            fontweight="bold",
            loc="center",
            pad=6,
        )
        ax_a.set_xticks([])
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)
        ax_a.grid(False)

        # Legend A
        ax_leg_a.legend(
            [Patch(facecolor=color_nonsyn), Patch(facecolor=color_syn)],
            ["Nonsynonymous", "Synonymous"],
            loc="center left",
            frameon=False,
            fontsize=8,
            title="Mutation Classification",
            title_fontproperties={"size": 9, "weight": "bold"},
            handlelength=1.2,
            handletextpad=0.5,
            labelspacing=0.3,
            alignment="left",
        )

        # ============================================================
        # STEP 4: Draw Panel B - Mutational Signatures
        # ============================================================
        if sig_df is not None:
            sig_colors = ["#E0F3FF", "#b7cfb3", "#4682B4", "#FFB6C1"]
            if len(signature_labels) > len(sig_colors):
                extra = plt.cm.Set3(
                    np.linspace(0, 1, len(signature_labels) - len(sig_colors))
                )
                sig_colors.extend([plt.matplotlib.colors.rgb2hex(c) for c in extra])

            bottom = np.zeros(len(sig_df))
            for i, sig_name in enumerate(signature_labels):
                if sig_name in sig_df.columns:
                    values = (
                        sig_df[sig_name].reindex(final_sample_order).fillna(0).values
                    )
                    ax_b.bar(x, values, bottom=bottom, color=sig_colors[i], width=0.8)
                    bottom += values

        ax_b.set_xlim(x_lim)
        ax_b.set_ylim([0, 1])
        ax_b.set_ylabel("Mutational Signatures", fontsize=10)
        ax_b.set_title(
            "B  Mutational Signatures",
            fontsize=12,
            fontweight="bold",
            loc="center",
            pad=6,
        )
        ax_b.set_xticks([])
        ax_b.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        ax_b.grid(False)

        # Legend B
        if sig_df is not None and signature_labels:
            sig_patches = [
                Patch(facecolor=sig_colors[i], label=sig_name)
                for i, sig_name in enumerate(signature_labels)
            ]
            ax_leg_b.legend(
                handles=sig_patches,
                loc="center left",
                frameon=False,
                fontsize=8,
                title="Mutation Signature",
                title_fontproperties={"size": 9, "weight": "bold"},
                handlelength=1.2,
                handletextpad=0.5,
                labelspacing=0.3,
                alignment="left",
            )

        # ============================================================
        # STEP 5: Draw Panel C - Purity (with cell borders)
        # ============================================================
        if purity_aligned is not None:
            # Get Blues colormap for purity values
            blues_cmap = plt.cm.get_cmap("Blues")

            # Draw individual rectangles with white borders (like Panel F)
            for i, sample in enumerate(final_sample_order):
                if sample in purity_aligned.index:
                    purity_val = purity_aligned[sample]

                    # Determine color based on purity value
                    if pd.isna(purity_val):
                        # White for missing data
                        color = "#ffffff"
                    else:
                        # Map purity value (0-1) to Blues colormap
                        color = blues_cmap(purity_val)

                    # Draw rectangle with white border for cell separation
                    rect = Rectangle(
                        (i - 0.5, -0.5),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    ax_c.add_patch(rect)

        ax_c.set_xlim(x_lim)
        ax_c.set_ylim(0.5, -0.5)
        ax_c.set_yticks([0])
        ax_c.set_yticklabels(["Purity"], fontsize=10)
        ax_c.set_title("C  Purity", fontsize=12, fontweight="bold", loc="center", pad=6)
        ax_c.set_xticks([])
        for spine in ax_c.spines.values():
            spine.set_visible(False)

        # ============================================================
        # STEP 6: Draw Panel D - Mutation Type
        # ============================================================
        mutation_colors = {
            "Missense": "#1f5a85",
            "Nonsense": "#8e2c2c",
            "Frameshift indel": "#d8b6cf",
            "Silent": "#b7cfb3",
            "Multiple": "#cfe4f6",
        }

        if mutation_matrix is not None and len(final_genes_d) > 0:
            for i, gene in enumerate(final_genes_d):
                for j, sample in enumerate(final_sample_order):
                    value = (
                        mutation_matrix.loc[gene, sample]
                        if sample in mutation_matrix.columns
                        else None
                    )
                    if pd.notna(value):
                        if isinstance(value, tuple):
                            type1, type2 = value
                            ax_d.add_patch(
                                Polygon(
                                    [
                                        (j - 0.5, i - 0.5),
                                        (j + 0.5, i - 0.5),
                                        (j - 0.5, i + 0.5),
                                    ],
                                    facecolor=mutation_colors[type1],
                                    edgecolor="white",
                                    linewidth=0.5,
                                )
                            )
                            ax_d.add_patch(
                                Polygon(
                                    [
                                        (j + 0.5, i - 0.5),
                                        (j + 0.5, i + 0.5),
                                        (j - 0.5, i + 0.5),
                                    ],
                                    facecolor=mutation_colors[type2],
                                    edgecolor="white",
                                    linewidth=0.5,
                                )
                            )
                        else:
                            ax_d.add_patch(
                                Rectangle(
                                    (j - 0.5, i - 0.5),
                                    1,
                                    1,
                                    facecolor=mutation_colors[value],
                                    edgecolor="white",
                                    linewidth=0.5,
                                )
                            )

        ax_d.set_xlim(x_lim)
        ax_d.set_ylim(len(final_genes_d) - 0.5, -0.5)
        ax_d.set_yticks(range(len(final_genes_d)))
        ax_d.set_yticklabels(final_genes_d, fontsize=10, style="italic")
        ax_d.set_title(
            "D  Mutation Type", fontsize=12, fontweight="bold", loc="center", pad=6
        )
        ax_d.set_xticks([])
        for spine in ax_d.spines.values():
            spine.set_visible(False)

        # ============================================================
        # STEP 7: Draw Panel E - CNA
        # ============================================================
        cna_colors = {
            "Allelic amplification": "#cfe4f6",
            "Baseline": "#e9e9e9",
            "Allelic deletion": "#1f5a85",
            "aCN = 0": "#8b8b8b",
            "NA": "#ffffff",
        }

        if cna_matrix is not None and len(final_genes_e) > 0:
            for i, gene in enumerate(final_genes_e):
                for j, sample in enumerate(final_sample_order):
                    cell_value = (
                        cna_matrix.loc[gene, sample]
                        if sample in cna_matrix.columns
                        else None
                    )
                    if cell_value is None or (
                        isinstance(cell_value, float) and pd.isna(cell_value)
                    ):
                        ax_e.add_patch(
                            Rectangle(
                                (j - 0.5, i - 0.5),
                                1,
                                1,
                                facecolor=cna_colors["NA"],
                                edgecolor="white",
                                linewidth=0.5,
                            )
                        )
                    else:
                        states = (
                            cell_value if isinstance(cell_value, list) else [cell_value]
                        )
                        if len(states) == 1:
                            ax_e.add_patch(
                                Rectangle(
                                    (j - 0.5, i - 0.5),
                                    1,
                                    1,
                                    facecolor=cna_colors.get(states[0], "#ffffff"),
                                    edgecolor="white",
                                    linewidth=0.5,
                                )
                            )
                        else:
                            priority_map = {
                                "aCN = 0": 4,
                                "Allelic deletion": 3,
                                "Allelic amplification": 2,
                                "Baseline": 1,
                            }
                            sorted_states = sorted(
                                states,
                                key=lambda x: priority_map.get(x, 0),
                                reverse=True,
                            )
                            ax_e.add_patch(
                                Polygon(
                                    [
                                        (j - 0.5, i - 0.5),
                                        (j + 0.5, i - 0.5),
                                        (j - 0.5, i + 0.5),
                                    ],
                                    facecolor=cna_colors.get(
                                        sorted_states[0], "#ffffff"
                                    ),
                                    edgecolor="none",
                                )
                            )
                            ax_e.add_patch(
                                Polygon(
                                    [
                                        (j + 0.5, i - 0.5),
                                        (j + 0.5, i + 0.5),
                                        (j - 0.5, i + 0.5),
                                    ],
                                    facecolor=cna_colors.get(
                                        sorted_states[1]
                                        if len(sorted_states) > 1
                                        else sorted_states[0],
                                        "#ffffff",
                                    ),
                                    edgecolor="none",
                                )
                            )
                            ax_e.plot(
                                [j - 0.5, j + 0.5],
                                [i + 0.5, i - 0.5],
                                color="white",
                                linewidth=0.5,
                            )

        ax_e.set_xlim(x_lim)
        ax_e.set_ylim(len(final_genes_e) - 0.5, -0.5)
        ax_e.set_yticks(range(len(final_genes_e)))
        ax_e.set_yticklabels(final_genes_e, fontsize=10, style="italic")
        ax_e.set_title(
            "E  Copy Number Alteration",
            fontsize=12,
            fontweight="bold",
            loc="center",
            pad=6,
        )
        ax_e.set_xticks([])
        for spine in ax_e.spines.values():
            spine.set_visible(False)

        # ============================================================
        # STEP 8: Draw Panel F - WGD
        # ============================================================
        WGD_YES_COLOR = "#8b8b8b"
        WGD_NO_COLOR = "#ffffff"
        WGD_BORDER_COLOR = "#e0e0e0"

        if wgd_series is not None:
            for i, sample in enumerate(final_sample_order):
                wgd_status = wgd_series[sample]
                if pd.isna(wgd_status) or not wgd_status:
                    ax_f.add_patch(
                        Rectangle(
                            (i - 0.5, -0.5),
                            1,
                            1,
                            facecolor=WGD_NO_COLOR,
                            edgecolor="white",
                            linewidth=0.5,
                        )
                    )
                else:
                    ax_f.add_patch(
                        Rectangle(
                            (i - 0.5, -0.5),
                            1,
                            1,
                            facecolor=WGD_YES_COLOR,
                            edgecolor="white",
                            linewidth=0.5,
                        )
                    )

        ax_f.set_xlim(x_lim)
        ax_f.set_ylim(0.5, -0.5)
        ax_f.set_yticks([0])
        ax_f.set_yticklabels(["WGD"], fontsize=10)
        ax_f.set_title(
            "F  Whole Genome Doubling",
            fontsize=12,
            fontweight="bold",
            loc="center",
            pad=6,
        )
        ax_f.set_xticks([])
        for spine in ax_f.spines.values():
            spine.set_visible(False)

        # Legend D - Mutation Type
        mutation_legend_elements = [
            Patch(facecolor=mutation_colors[mt], edgecolor="none")
            for mt in ["Missense", "Nonsense", "Frameshift indel", "Silent", "Multiple"]
        ]
        ax_leg_d.legend(
            mutation_legend_elements,
            ["Missense", "Nonsense", "Frameshift indel", "Silent", "Multiple"],
            loc="center left",
            frameon=False,
            fontsize=8,
            title="Mutation Type",
            title_fontproperties={"size": 9, "weight": "bold"},
            handlelength=1.2,
            handletextpad=0.5,
            labelspacing=0.3,
            alignment="left",
        )

        # Legend E - Copy Number Alteration
        cna_legend_elements = [
            Patch(facecolor=cna_colors[state], edgecolor="none")
            for state in [
                "Allelic amplification",
                "Baseline",
                "Allelic deletion",
                "aCN = 0",
            ]
        ]
        ax_leg_e.legend(
            cna_legend_elements,
            ["Allelic amplification", "Baseline", "Allelic deletion", "aCN = 0"],
            loc="center left",
            frameon=False,
            fontsize=8,
            title="Copy Number Alteration",
            title_fontproperties={"size": 9, "weight": "bold"},
            handlelength=1.2,
            handletextpad=0.5,
            labelspacing=0.3,
            alignment="left",
        )

        # Legend F - WGD
        wgd_legend_elements = [
            Patch(facecolor=WGD_NO_COLOR, edgecolor=WGD_BORDER_COLOR, linewidth=0.5),
            Patch(facecolor=WGD_YES_COLOR, edgecolor="none"),
        ]
        ax_leg_f.legend(
            wgd_legend_elements,
            ["No", "Yes"],
            loc="center left",
            frameon=False,
            fontsize=8,
            title="Whole Genome Doubling",
            title_fontproperties={"size": 9, "weight": "bold"},
            handlelength=1.2,
            handletextpad=0.5,
            labelspacing=0.3,
            alignment="left",
        )

        # ============================================================
        # STEP 9: Draw Panel G - Same Patient
        # ============================================================
        if sp_df is not None and len(sp_df) > 0:
            sample_to_idx = {s: i for i, s in enumerate(final_sample_order)}
            groups = sp_df.groupby("group")["sample"].apply(list)

            for group_name, samples in groups.items():
                indices = sorted(
                    [sample_to_idx[s] for s in samples if s in sample_to_idx]
                )
                for idx in indices:
                    ax_g.plot(idx, 0, "o", color="black", markersize=4)
                if len(indices) > 1:
                    ax_g.plot(
                        [indices[0], indices[-1]],
                        [0, 0],
                        "-",
                        color="black",
                        linewidth=1.5,
                    )

        ax_g.set_xlim(x_lim)
        ax_g.set_ylim(-0.5, 0.5)
        ax_g.set_yticks([])
        ax_g.set_title(
            "G  Same Patient", fontsize=12, fontweight="bold", loc="center", pad=6
        )
        ax_g.set_xticks([])
        for spine in ax_g.spines.values():
            spine.set_visible(False)

        # Legend G
        ax_leg_g.legend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="black",
                    markersize=4,
                    linestyle="-",
                    linewidth=1.5,
                )
            ],
            [r"$\mathbf{Same\ Patient}$"],
            loc="center left",
            frameon=False,
            fontsize=9,
            handlelength=1.2,
            handletextpad=0.5,
            labelspacing=0.3,
            markerfirst=False,
            alignment="left",
        )

        plt.tight_layout()

    finally:
        # Restore interactive mode
        if was_interactive:
            plt.ion()

    elapsed_time = time.time() - start_time
    logger.info(f"Complete CoMut plot created successfully in {elapsed_time:.2f}s")

    return fig