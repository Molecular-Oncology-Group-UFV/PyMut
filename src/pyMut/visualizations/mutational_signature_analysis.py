"""
Module for mutational signature analysis visualizations.

This module provides visualization functions for mutational signature analysis,
including signature profiles, cosine similarity heatmaps, and contribution plots.

The visualizations work with data from the analysis.mutational_signature module:
- trinucleotideMatrix: Generates 96 x n_samples context matrix
- extractSignatures: Performs NMF decomposition to extract signatures

Architecture:
- This module handles ONLY visualization, not data processing or analysis
- Functions receive pre-processed matrices (W, H) from extractSignatures
- Clear separation of concerns: data preparation vs. rendering
"""

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Color palette for signature legend/identification
SIGNATURE_COLORS = [
    '#266199', '#b7d5ea', '#acc6aa', '#E0CADB', '#695D73',
    '#B88655', '#DDDDDD', '#71a0a5', '#841D22', '#E08B69'
]

# SNV substitution colors
SUBSTITUTION_COLORS = {
    'C>A': '#02bdee',  # cyan
    'C>G': '#010101',  # black
    'C>T': '#e32925',  # red
    'T>A': '#cac9c9',  # grey
    'T>C': '#a1cf63',  # green
    'T>G': '#ecc7c4'   # pink
}

# Standard order of 6 substitution types (pyrimidine-centric)
SUBSTITUTION_TYPES = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

# Number of trinucleotide contexts per substitution type
CONTEXTS_PER_SUBSTITUTION = 16

# Total number of trinucleotide contexts
TOTAL_CONTEXTS = 96


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

class SignatureValidator:
    """Centralized validation for mutational signature matrices."""
    
    @staticmethod
    def validate_signature_matrix(W: np.ndarray, matrix_name: str = "W") -> None:
        """
        Validate that the signature matrix has the correct shape and properties.
        
        Parameters
        ----------
        W : np.ndarray
            Signature matrix to validate (should be 96 × k)
        matrix_name : str, default "W"
            Name of the matrix for error messages
        
        Raises
        ------
        ValueError
            If the matrix shape or properties are invalid
        TypeError
            If the input is not a numpy array
        
        Notes
        -----
        - Validates that W is a 2D numpy array
        - Validates that W has exactly 96 rows (trinucleotide contexts)
        - Warns if columns don't sum to 1 (normalized signatures)
        """
        if not isinstance(W, np.ndarray):
            raise TypeError(f"{matrix_name} must be a numpy array, got {type(W)}")
        
        if W.ndim != 2:
            raise ValueError(f"{matrix_name} must be a 2D array, got {W.ndim}D")
        
        if W.shape[0] != TOTAL_CONTEXTS:
            raise ValueError(
                f"{matrix_name} must have {TOTAL_CONTEXTS} rows (trinucleotide contexts), "
                f"got {W.shape[0]}"
            )
        
        # Check if signatures are normalized (each column sums to ~1)
        col_sums = W.sum(axis=0)
        if not np.allclose(col_sums, 1.0, rtol=1e-3):
            logger.debug(
                f"{matrix_name} columns will be normalized (max deviation from 1: {np.max(np.abs(col_sums - 1.0)):.6f})"
            )
    
    @staticmethod
    def validate_exposure_matrix(H: np.ndarray, W: np.ndarray) -> None:
        """
        Validate exposure matrix compatibility with signature matrix.
        
        Parameters
        ----------
        H : np.ndarray
            Exposure matrix (k × n_samples)
        W : np.ndarray
            Signature matrix (96 × k)
        
        Raises
        ------
        ValueError
            If H and W are incompatible
        TypeError
            If inputs are not numpy arrays
        """
        if not isinstance(H, np.ndarray):
            raise TypeError(f"H must be a numpy array, got {type(H)}")
        
        if H.ndim != 2:
            raise ValueError(f"H must be a 2D array, got {H.ndim}D")
        
        if H.shape[0] != W.shape[1]:
            raise ValueError(
                f"Number of signatures in H ({H.shape[0]}) doesn't match "
                f"number of signatures in W ({W.shape[1]})"
            )
    
    @staticmethod
    def validate_sample_names(sample_names: List[str], n_samples: int, matrix_name: str = "H") -> None:
        """
        Validate sample names list matches number of samples.
        
        Parameters
        ----------
        sample_names : List[str]
            List of sample names
        n_samples : int
            Expected number of samples
        matrix_name : str, default "H"
            Name of the matrix for error messages
        
        Raises
        ------
        ValueError
            If lengths don't match
        """
        if len(sample_names) != n_samples:
            raise ValueError(
                f"Number of samples in {matrix_name} ({n_samples}) doesn't match "
                f"sample_names length ({len(sample_names)})"
            )
    
    @staticmethod
    def validate_signature_names(signature_names: List[str], n_signatures: int) -> None:
        """
        Validate signature names list matches number of signatures.
        
        Parameters
        ----------
        signature_names : List[str]
            List of signature names
        n_signatures : int
            Expected number of signatures
        
        Raises
        ------
        ValueError
            If lengths don't match
        """
        if len(signature_names) != n_signatures:
            raise ValueError(
                f"Number of signature_names ({len(signature_names)}) doesn't match "
                f"number of signatures ({n_signatures})"
            )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_trinucleotide_contexts() -> List[str]:
    """
    Generate all 96 trinucleotide contexts in standard SBS order.
    
    The order follows the COSMIC convention:
    - 6 substitution types (C>A, C>G, C>T, T>A, T>C, T>G)
    - 16 contexts per substitution (4 upstream * 4 downstream bases)
    
    Returns
    -------
    List[str]
        List of 96 context labels in format "X[REF>ALT]Z"
        where X is upstream base, Z is downstream base
    
    Examples
    --------
    >>> contexts = _get_trinucleotide_contexts()
    >>> contexts[0]
    'A[C>A]A'
    >>> len(contexts)
    96
    """
    bases = ['A', 'C', 'G', 'T']
    contexts = []
    
    for substitution in SUBSTITUTION_TYPES:
        for upstream in bases:
            for downstream in bases:
                context = f"{upstream}[{substitution}]{downstream}"
                contexts.append(context)
    
    return contexts


# ============================================================================
# SIGNATURE BAR CHART VISUALIZATION
# ============================================================================

def _normalize_signature_matrix(signatures: np.ndarray) -> np.ndarray:
    """
    Normalize signature matrix so each column (signature) sums to 1.
    
    This ensures that signature profiles are represented as proportions,
    which is the standard representation for mutational signatures.
    
    Parameters
    ----------
    signatures : np.ndarray
        Signature matrix of shape (96, n_signatures) where each column
        represents one signature profile
    
    Returns
    -------
    np.ndarray
        Normalized signature matrix where each column sums to 1.0
    
    Notes
    -----
    - Adds a small epsilon (1e-10) to avoid division by zero
    - Input can be either raw counts or already normalized values
    
    Examples
    --------
    >>> sig = np.array([[10, 20], [5, 10], [5, 10]])
    >>> normalized = _normalize_signature_matrix(sig)
    >>> np.allclose(normalized.sum(axis=0), 1.0)
    True
    """
    column_sums = signatures.sum(axis=0, keepdims=True)
    # Add small epsilon to avoid division by zero
    column_sums = np.where(column_sums == 0, 1e-10, column_sums)
    normalized = signatures / column_sums
    return normalized


def _draw_signature_bars(ax: Axes, values: np.ndarray) -> None:
    """Draw the 96 trinucleotide context bars for a signature profile."""
    for i in range(TOTAL_CONTEXTS):
        sub_type_index = i // CONTEXTS_PER_SUBSTITUTION
        sub_type = SUBSTITUTION_TYPES[sub_type_index]
        color = SUBSTITUTION_COLORS[sub_type]
        ax.bar(i, values[i], color=color, width=0.6, edgecolor="white", linewidth=0.3)


def _add_substitution_header(ax: Axes, max_y_val: float) -> None:
    """Add colored header bars showing the 6 substitution types."""
    import matplotlib.patches as mpatches

    y_bottom = max_y_val * 1.02
    y_top = max_y_val * 1.14
    y_text = max_y_val * 1.08
    
    for i, sub_type in enumerate(SUBSTITUTION_TYPES):
        # X coordinates for this substitution section (16 contexts each)
        x_start = i * CONTEXTS_PER_SUBSTITUTION

        # Draw colored rectangle aligned with the bars
        # Rectangle starts at x_start - 0.5 (bar edges) and has width 16
        rect = mpatches.Rectangle(
            (x_start - 0.5, y_bottom),
            CONTEXTS_PER_SUBSTITUTION,
            y_top - y_bottom,
            facecolor=SUBSTITUTION_COLORS[sub_type],
            edgecolor="none",
            alpha=0.8,
            zorder=10,
            clip_on=False,
        )
        ax.add_patch(rect)

        # Add substitution type label centered in the rectangle
        ax.text(
            x_start + 7.5,
            y_text,
            sub_type,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
            zorder=11,
            clip_on=False,
        )


def _add_trinucleotide_footer(ax: Axes, max_y_val: float) -> None:
    """Add trinucleotide context labels at the bottom of the plot."""
    contexts = _get_trinucleotide_contexts()
    y_text_position = -max_y_val * 0.08
    
    for i, context in enumerate(contexts):
        # Determine color based on substitution type
        sub_type_index = i // CONTEXTS_PER_SUBSTITUTION
        sub_type = SUBSTITUTION_TYPES[sub_type_index]
        color = SUBSTITUTION_COLORS[sub_type]
        
        # Format context label (e.g., "A[C>A]T")
        formatted_context = f"{context[0]}{context[2:6]}{context[-1]}"
        
        # Add rotated text label
        ax.text(
            i, 
            y_text_position, 
            formatted_context,
            ha='center', 
            va='top', 
            rotation=90,
            fontsize=7, 
            color=color
        )


def _style_signature_axis(ax: Axes, title: str, max_y_val: float, is_last_plot: bool) -> None:
    """
    Apply consistent styling to a signature profile axis.
    
    Configures titles, labels, limits, and spines for a clean professional look
    consistent with COSMIC signature visualization standards.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axis to style
    title : str
        Title for this specific subplot (e.g., "Signature 1")
    max_y_val : float
        Maximum Y value in the data (used for scaling)
    is_last_plot : bool
        Whether this is the last plot in a multi-panel figure
        (determines if trinucleotide labels are shown)
    
    Notes
    -----
    - Y-axis shows percentage with 5 evenly spaced ticks
    - X-axis ticks are hidden (context labels in footer instead)
    - Only bottom and left spines are visible
    - Trinucleotide footer only added to last panel
    """
    # Set panel title
    ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
    
    # Y-axis configuration
    ax.set_ylabel("Percentage", fontsize=11, labelpad=10, color='black')
    ax.set_ylim(0, max_y_val * 1.2)
    ax.set_yticks(np.linspace(0, max_y_val * 1.2, 5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%' if y > 0 else ''))
    ax.tick_params(axis='y', labelsize=10, colors='black')

    # X-axis configuration - set limits to match bar positions
    ax.set_xlim(-0.5, TOTAL_CONTEXTS - 0.5)
    ax.tick_params(axis='x', length=0, labelbottom=False)

    # Clean styling
    ax.grid(False)
    for spine_pos, spine in ax.spines.items():
        spine.set_visible(spine_pos in ['bottom', 'left'])
        if spine_pos in ['bottom', 'left']:
            spine.set_linewidth(0.5)
            spine.set_color('black')
    
    # Add trinucleotide footer only to last panel
    if is_last_plot:
        _add_trinucleotide_footer(ax, max_y_val)


def create_signature_bar_chart(
    signatures: np.ndarray,
    signature_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    axes: Optional[List[Axes]] = None
) -> Tuple[Figure, List[Axes]]:
    """
    Create a multi-panel bar chart showing mutational signature profiles.
    
    This is the main public function for creating signature bar charts.
    Each signature is displayed in a separate panel with 96 bars representing
    the trinucleotide context distribution.
    
    Parameters
    ----------
    signatures : np.ndarray
        Signature matrix of shape (96, n_signatures) where each column represents
        one signature profile. Can be raw counts or normalized - will be normalized
        internally. This is the 'signatures' output from extractSignatures().
    signature_names : Optional[List[str]], default None
        Custom names for each signature (e.g., ["SBS1", "SBS5", "SBS13"]).
        If None, uses generic names like "Signature 1", "Signature 2", etc.
    figsize : Optional[Tuple[int, int]], default None
        Figure size as (width, height) in inches.
        If None, automatically calculated as (12, 3 * n_signatures)
    axes : Optional[List[Axes]], default None
        List of matplotlib axes to render on (one per signature). If None, creates a new figure.
        Must have exactly n_signatures elements. Use this to embed the charts in a larger figure.
    
    Returns
    -------
    fig : Figure
        Matplotlib figure containing all panels
    axes : List[Axes]
        List of axes, one per signature, for further customization if needed
    
    Raises
    ------
    ValueError
        If signatures matrix is not 96 rows
        If signature_names length doesn't match number of signatures
    
    Notes
    -----
    - Signatures are automatically normalized to sum to 1 (100%)
    - The 96 contexts follow standard COSMIC order (pyrimidine-centric)
    - Colors follow standard SBS color scheme
    - Trinucleotide labels only shown on the last panel
    - Each panel includes colored header with substitution types
    
    Examples
    --------
    >>> # After running extractSignatures
    >>> result = py_mut.extractSignatures(contexts_df, n=3)
    >>> fig, axes = create_signature_bar_chart(
    ...     result['signatures'],
    ...     signature_names=["Signature A", "Signature B", "Signature C"]
    ... )
    >>> plt.show()
    
    See Also
    --------
    trinucleotideMatrix : Generate 96 x n_samples context matrix
    extractSignatures : Perform NMF to extract signatures
    """
    import time
    start_time = time.time()
    
    logger.info("Generating signature bar chart...")
    
    # Validate input
    SignatureValidator.validate_signature_matrix(signatures, "signatures")
    
    n_signatures = signatures.shape[1]
    
    # Generate default names if not provided
    if signature_names is None:
        signature_names = [f"Signature {i+1}" for i in range(n_signatures)]
    else:
        SignatureValidator.validate_signature_names(signature_names, n_signatures)
    
    # Calculate figure size if not provided
    if figsize is None:
        figsize = (12, 3 * n_signatures)
    
    # Normalize signatures (each column sums to 1)
    signatures_norm = _normalize_signature_matrix(signatures)
    
    # Create figure and axes if not provided
    if axes is None:
        # Create our own figure with subplots (one per signature)
        fig = plt.figure(figsize=figsize)
        axes_list = [
            fig.add_subplot(n_signatures, 1, i + 1) for i in range(n_signatures)
        ]
    else:
        # Use provided axes
        if len(axes) != n_signatures:
            raise ValueError(
                f"Number of provided axes ({len(axes)}) must match number of signatures ({n_signatures})"
            )
        axes_list = axes
        fig = axes_list[0].figure  # type: ignore[assignment]

    # Render each signature panel
    for i in range(n_signatures):
        ax = axes_list[i]
        signature_profile = signatures_norm[:, i]
        
        # Convert to percentages for display
        values = signature_profile * 100
        
        # Calculate max Y value for scaling
        # Use minimum of 5% if all values are very small
        max_y_val = float(max(np.max(values), 5.0))
        
        # === Render signature panel ===
        # 1. Draw the 96 bars
        _draw_signature_bars(ax, values)
        
        # 2. Add colored header with substitution types
        _add_substitution_header(ax, max_y_val)
        
        # 3. Apply styling (footer only on last panel)
        is_last_plot = (i == n_signatures - 1)
        _style_signature_axis(ax, signature_names[i], max_y_val, is_last_plot)

    # Note: Layout adjustment is handled by the calling function (core.py)
    # to allow for proper spacing with suptitle
    
    elapsed_time = time.time() - start_time
    logger.info(f"Signature bar chart generated in {elapsed_time:.2f}s")
    
    return fig, axes_list  # type: ignore[return-value]


# ============================================================================
# COSINE SIMILARITY HEATMAP VISUALIZATION
# ============================================================================

def _load_and_validate_cosmic_signatures(cosmic_path: str) -> pd.DataFrame:
    """
    Load COSMIC signature catalog and validate its structure.
    
    Parameters
    ----------
    cosmic_path : str
        Path to COSMIC catalog file (TSV format with 96 rows)
    
    Returns
    -------
    pd.DataFrame
        COSMIC signatures dataframe with:
        - Index: 96 trinucleotide contexts in standard order
        - Columns: COSMIC signature names (e.g., 'SBS1', 'SBS5')
        - Values: Normalized signature profiles (each column sums to 1)
    
    Raises
    ------
    FileNotFoundError
        If the COSMIC file doesn't exist
    ValueError
        If the file format is invalid or contexts don't match
    
    Notes
    -----
    - Automatically filters out artifact signatures
    - Reindexes to match standard trinucleotide context order
    - Normalizes each signature to sum to 1
    """
    try:
        cosmic_df = pd.read_csv(cosmic_path, sep='\t')
    except FileNotFoundError:
        raise FileNotFoundError(f"COSMIC catalog file not found: {cosmic_path}")
    except Exception as e:
        raise ValueError(f"Error reading COSMIC catalog: {e}")
    
    if cosmic_df.shape[0] != TOTAL_CONTEXTS:
        raise ValueError(
            f"COSMIC catalog must have {TOTAL_CONTEXTS} rows (trinucleotide contexts), "
            f"got {cosmic_df.shape[0]}"
        )
    
    context_column = cosmic_df.columns[0]
    contexts = cosmic_df[context_column].values
    cosmic_df = cosmic_df.set_index(context_column)
    
    trinuc_contexts = _get_trinucleotide_contexts()
    missing_contexts = set(trinuc_contexts) - set(contexts)
    if missing_contexts:
        raise ValueError(f"COSMIC catalog is missing required contexts: {missing_contexts}")
    
    try:
        cosmic_df = cosmic_df.reindex(trinuc_contexts)
        if cosmic_df.isnull().any().any():
            raise ValueError("Some standard trinucleotide contexts are missing in COSMIC catalog")
    except Exception as e:
        raise ValueError(f"Failed to align COSMIC catalog contexts: {e}")
    
    # Filter artifact signatures (ending with 'c' or containing 'artifact')
    filtered_columns = [
        col for col in cosmic_df.columns
        if not col.endswith('c') and 'artifact' not in col.lower() and 'artefact' not in col.lower()
    ]
    cosmic_df = cosmic_df[filtered_columns]
    
    # Remove zero-sum signatures
    column_sums = cosmic_df.sum(axis=0)
    zero_sum_mask = column_sums == 0
    if zero_sum_mask.any():
        zero_sum_sigs = cosmic_df.columns[zero_sum_mask].tolist()
        logger.debug(f"Removing {len(zero_sum_sigs)} signatures with zero sum from COSMIC catalog")
        cosmic_df = cosmic_df.loc[:, ~zero_sum_mask]
    
    # Normalize to sum=1
    cosmic_df = cosmic_df / cosmic_df.sum(axis=0)
    
    return cosmic_df


def _calculate_cosine_similarity_matrix(W: np.ndarray, 
                                       cosmic_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate cosine similarity between extracted signatures and COSMIC signatures.
    
    Parameters
    ----------
    W : np.ndarray
        Extracted signature matrix (96 * k) where k is the number of signatures
    cosmic_df : pd.DataFrame
        COSMIC signatures dataframe (96 * m) where m is the number of COSMIC signatures
    
    Returns
    -------
    np.ndarray
        Cosine similarity matrix (k * m) with values in range [0, 1]
        where 1 indicates identical signatures
    
    Notes
    -----
    - Both matrices must have 96 rows in the same order
    - Higher values indicate more similar signatures
    - Uses sklearn's cosine_similarity for efficient computation
    """
    if W.shape[0] != cosmic_df.shape[0]:
        raise ValueError(
            f"Context dimension mismatch: W has {W.shape[0]} rows, "
            f"COSMIC has {cosmic_df.shape[0]} rows"
        )
    
    similarity_matrix = cosine_similarity(W.T, cosmic_df.values.T)
    return similarity_matrix


def _render_cosine_similarity_heatmap(
    similarity_matrix: np.ndarray,
    signature_names: List[str],
    cosmic_names: List[str],
    ax: Axes,
    title: Optional[str] = None,
    cbar_ax: Optional[Axes] = None,
) -> None:
    """
    Render the cosine similarity heatmap on a matplotlib axis.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Cosine similarity matrix (k × m) to visualize
    signature_names : List[str]
        Names for extracted signatures (k names)
    cosmic_names : List[str]
        Names for COSMIC signatures (m names)
    ax : Axes
        Matplotlib axis to draw on
    title : Optional[str], default None
        Title for the heatmap. If None, a default title is used.
    cbar_ax : Optional[Axes], default None
        Separate axis for the colorbar. If None, colorbar is placed within ax.

    Notes
    -----
    - Uses 'Blues' colormap with range [0, 1]
    - Darker blue indicates higher similarity
    - Includes gridlines for better readability
    - X-axis labels are rotated 90 degrees
    """
    df_similarity = pd.DataFrame(
        similarity_matrix,
        index=signature_names,
        columns=cosmic_names
    )

    # Configure colorbar - use manual creation for separate axis
    if cbar_ax is None:
        # Standard colorbar within the plot
        cbar_kws = {
            "label": "Cosine Similarity",
            "orientation": "vertical",
            "shrink": 1.0,
            "pad": 0.02,
        }
        create_cbar = True
    else:
        # Disable automatic colorbar
        cbar_kws = None
        create_cbar = False

    # Call heatmap
    sns.heatmap(
        df_similarity,
        vmin=0,
        vmax=1,
        cmap="Blues",
        linewidths=0.5,
        linecolor="lightgray",
        square=False,
        cbar_kws=cbar_kws if create_cbar else None,
        cbar=create_cbar,  # Only create colorbar if no separate axis
        ax=ax,
        annot=False,
    )

    # Manually create colorbar in separate axis if provided
    if cbar_ax is not None:
        # Get the mappable from the heatmap
        mappable = ax.collections[0]
        # Create colorbar in the separate axis
        cbar = ax.figure.colorbar(mappable, cax=cbar_ax)
        cbar.set_label("Cosine Similarity", rotation=90, labelpad=2)
        cbar_ax.yaxis.set_label_position("left")
        cbar_ax.yaxis.set_ticks_position("right")
        cbar.outline.set_visible(False)

    # Configure colorbar position and labels for standard case
    if cbar_ax is None:
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.outline.set_visible(False)

    ax.set_xlabel("COSMIC SBS Signatures", fontsize=11, labelpad=10)
    ax.set_xticks(np.arange(len(cosmic_names)) + 0.5)
    ax.set_yticks(np.arange(len(signature_names)) + 0.5)
    ax.set_xticklabels(cosmic_names, rotation=90, ha="center", fontsize=9)
    ax.set_yticklabels(signature_names, rotation=0, ha="right", fontsize=10)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    for spine in ax.spines.values():
        spine.set_visible(False)


def create_cosine_similarity_heatmap(
    W: np.ndarray,
    cosmic_path: str,
    signature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 3),
    title: str = "Cosine Similarity: Extracted vs COSMIC Signatures",
    ax: Optional[Axes] = None,
    _cbar_ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Create a heatmap showing cosine similarity between extracted and COSMIC signatures.

    This visualization compares mutational signatures extracted from your data
    with the COSMIC catalog of known signatures. Darker blue indicates higher
    similarity, helping identify which COSMIC signatures are present in your data.

    Parameters
    ----------
    W : np.ndarray
        Extracted signature matrix of shape (96, k) where k is the number of
        signatures. This is the 'signatures' output from extractSignatures().
        Each column represents one signature profile and should sum to 1.
    cosmic_path : str
        Path to COSMIC signature catalog file (TSV format).
        Example: "COSMIC_v3.4_SBS_GRCh38.txt"
        The file should have 96 rows (trinucleotide contexts) and columns
        for each COSMIC signature (e.g., SBS1, SBS2, etc.)
    signature_names : Optional[List[str]], default None
        Custom names for extracted signatures (e.g., ["Sig A", "Sig B"]).
        If None, uses generic names like "Signature 1", "Signature 2", etc.
        Must match the number of columns in W.
    figsize : Tuple[int, int], default (14, 3)
        Figure size as (width, height) in inches.
        Adjust based on the number of signatures and COSMIC references.
    title : str, default "Cosine Similarity: Extracted vs COSMIC Signatures"
        Title for the heatmap. Set to empty string "" to hide the title.
    ax : Optional[Axes], default None
        Matplotlib axis to render on. If None, creates a new figure.
        Use this to embed the heatmap in a larger figure with multiple panels.
    _cbar_ax : Optional[Axes], default None
        (Internal parameter) Separate axis for colorbar placement. Not intended
        for public use. Only used internally by create_complete_signature_analysis.

    Returns
    -------
    fig : Figure
        Matplotlib figure containing the heatmap
    ax : Axes
        Matplotlib axis with the heatmap for further customization

    Raises
    ------
    ValueError
        If W matrix has invalid shape or cosmic_path is invalid
    FileNotFoundError
        If cosmic_path file doesn't exist
    TypeError
        If inputs have incorrect types

    Notes
    -----
    - **Critical**: The 96 trinucleotide contexts must be in the same order
      in both W and the COSMIC catalog. This function automatically handles
      alignment using the standard COSMIC order.
    - Artifact signatures are automatically filtered out from COSMIC catalog
    - The heatmap shows k × m values where k is the number of extracted
      signatures and m is the number of COSMIC signatures
    - Cosine similarity ranges from 0 (completely different) to 1 (identical)
    - Values above 0.6 typically indicate meaningful similarity

    Examples
    --------
    >>> # After running extractSignatures
    >>> result = py_mut.extractSignatures(contexts_df, n=3)
    >>> W = result['signatures'].values  # Get 96 × 3 numpy array
    >>>
    >>> # Create cosine similarity heatmap
    >>> fig, ax = create_cosine_similarity_heatmap(
    ...     W=W,
    ...     cosmic_path="data/COSMIC_v3.4_SBS_GRCh38.txt",
    ...     signature_names=["Signature A", "Signature B", "Signature C"],
    ...     figsize=(16, 4)
    ... )
    >>> plt.show()

    See Also
    --------
    trinucleotideMatrix : Generate 96 * n_samples context matrix
    extractSignatures : Extract signatures using NMF decomposition
    compare_signatures : Detailed comparison with statistics
    create_signature_bar_chart : Visualize individual signature profiles
    """
    import time

    start_time = time.time()

    logger.info("Generating cosine similarity heatmap...")

    SignatureValidator.validate_signature_matrix(W, matrix_name="W")

    n_signatures = W.shape[1]

    if signature_names is None:
        signature_names = [f"Signature {i + 1}" for i in range(n_signatures)]
    else:
        SignatureValidator.validate_signature_names(signature_names, n_signatures)

    cosmic_df = _load_and_validate_cosmic_signatures(cosmic_path)
    similarity_matrix = _calculate_cosine_similarity_matrix(W, cosmic_df)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_figure = True
    else:
        fig = ax.figure  # type: ignore[assignment]
        own_figure = False

    _render_cosine_similarity_heatmap(
        similarity_matrix=similarity_matrix,
        signature_names=signature_names,
        cosmic_names=cosmic_df.columns.tolist(),
        ax=ax,
        title=title,
        cbar_ax=_cbar_ax,
    )

    if own_figure and hasattr(fig, "tight_layout"):
        fig.tight_layout()  # type: ignore

    elapsed_time = time.time() - start_time
    logger.info(f"Cosine similarity heatmap generated in {elapsed_time:.2f}s")

    return fig, ax  # type: ignore[return-value]


def _normalize_contributions_per_sample(H: np.ndarray) -> np.ndarray:
    """
    Normalize contribution matrix so each sample (column) sums to 1.

    Parameters
    ----------
    H : np.ndarray
        Contribution matrix of shape (k_signatures, n_samples)

    Returns
    -------
    np.ndarray
        Normalized matrix where each column sums to 1.0
        Columns with zero sum are set to NaN

    Notes
    -----
    - Non-zero columns are normalized to sum to 1.0
    - Zero-sum columns (samples with no mutations) are set to NaN
    """
    # Calculate sum for each sample (column)
    column_sums = H.sum(axis=0, keepdims=True)

    # Create normalized matrix
    H_normalized = np.zeros_like(H, dtype=float)

    # Normalize non-zero columns
    non_zero_mask = column_sums[0] > 0
    H_normalized[:, non_zero_mask] = H[:, non_zero_mask] / column_sums[:, non_zero_mask]

    # Set zero-sum columns to NaN (samples with no mutations)
    zero_mask = column_sums[0] == 0
    H_normalized[:, zero_mask] = np.nan

    return H_normalized


def create_signature_contribution_heatmap(
    contributions: pd.DataFrame,
    signature_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "Blues",
    show_values: bool = False,
    show_labels: bool = True,
    ax: Optional[Axes] = None,
    _cbar_ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Create a heatmap showing signature contributions normalized per sample.

    Each column (sample) is normalized to sum to 1.0, showing the relative
    contribution of each signature within that sample.

    Parameters
    ----------
    contributions : pd.DataFrame
        Contribution DataFrame from extractSignatures['contributions']
        Rows are signatures, columns are samples
    signature_names : Optional[List[str]], default None
        Custom signature names. If None, uses DataFrame index.
    figsize : Optional[Tuple[float, float]], default None
        Figure size (width, height). If None, auto-calculated.
    cmap : str, default 'Blues'
        Colormap for the heatmap
    show_values : bool, default False
        Whether to show numerical values in each cell
    show_labels : bool, default True
        Whether to show sample labels on X-axis. Set to False in composite plots
        to reduce clutter. Default is True.
    ax : Optional[Axes], default None
        Matplotlib axis to render on. If None, creates a new figure.
        Use this to embed the heatmap in a larger figure with multiple panels.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Raises
    ------
    ValueError
        If input dimensions are invalid

    Examples
    --------
    >>> result = py_mut.extractSignatures(contexts_df, n=3)
    >>> fig, ax = create_signature_contribution_heatmap(
    ...     contributions=result['contributions'],
    ...     signature_names=["SBS1", "SBS5", "SBS13"]
    ... )
    >>> plt.show()

    See Also
    --------
    create_signature_stacked_bar_chart : Alternative visualization
    create_signature_donut_plot : Cohort-level overview
    """
    import time

    import matplotlib.pyplot as plt

    start_time = time.time()

    logger.info("Generating signature contribution heatmap...")

    # Extract data from DataFrame
    H = contributions.values
    sample_names = contributions.columns.tolist()

    # Validate dimensions
    SignatureValidator.validate_sample_names(sample_names, H.shape[1], "contributions")

    n_signatures = H.shape[0]
    n_samples = H.shape[1]

    # Use DataFrame index as signature names if not provided
    if signature_names is None:
        if contributions.index.name or any(
            contributions.index != range(len(contributions))
        ):
            signature_names = contributions.index.tolist()
        else:
            signature_names = [f"Signature {i + 1}" for i in range(n_signatures)]
    else:
        SignatureValidator.validate_signature_names(signature_names, n_signatures)

    # Calculate figure size if not provided
    if figsize is None:
        width = min(20, max(10, n_samples * 0.15))
        height = n_signatures * 0.8 + 1  # Add space for colorbar
        figsize = (width, height)

    # Normalize contributions per sample (columns sum to 1)
    H_normalized = _normalize_contributions_per_sample(H)

    # Check for samples with zero mutations
    zero_samples = np.isnan(H_normalized).all(axis=0)
    n_zero_samples = zero_samples.sum()

    if n_zero_samples > 0:
        logger.debug(
            f"Found {n_zero_samples} sample(s) with zero total mutations (will appear white in heatmap)"
        )

    # Create DataFrame for heatmap
    df_contributions = pd.DataFrame(
        H_normalized, index=signature_names, columns=sample_names
    )

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_figure = True
    else:
        fig = ax.figure  # type: ignore[assignment]
        own_figure = False

    # Configure seaborn style
    sns.set_style("whitegrid", {"axes.grid": False})

    # Configure colorbar - use manual creation for separate axis
    if _cbar_ax is None:
        create_cbar = True
        cbar_kws = {
            "orientation": "vertical",
            "shrink": 1.0,
            "label": "Relative Contribution",
            "pad": 0.04,
        }
    else:
        create_cbar = False
        cbar_kws = None

    # Create heatmap
    sns.heatmap(
        df_contributions,
        vmin=0,
        vmax=1,
        cmap=cmap,
        linewidths=0.5,
        linecolor="lightgray",
        square=False,
        cbar_kws=cbar_kws if create_cbar else None,
        cbar=create_cbar,
        ax=ax,
        annot=show_values,
        fmt=".2f",
    )

    # Manually create colorbar in separate axis if provided
    if _cbar_ax is not None:
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=0, vmax=1)
        cbar = ColorbarBase(_cbar_ax, cmap=cmap, norm=norm, orientation="vertical")

        # Style similar to cosine similarity heatmap colorbar
        cbar.set_label(
            "Relative Contribution", fontsize=9, rotation=90, labelpad=8, va="bottom"
        )
        _cbar_ax.yaxis.set_label_position("left")
        _cbar_ax.yaxis.set_ticks_position("right")
        _cbar_ax.tick_params(labelsize=8)
    elif create_cbar:
        # Configure colorbar label position and size for standard case
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.yaxis.label.set_size(9)  # Smaller font size to prevent overlap

    # Remove Y-axis label
    ax.set_ylabel("")

    # Handle X-axis labels based on show_labels parameter
    if not show_labels:
        # Hide labels completely
        ax.set_xticklabels([])
        ax.set_xlabel(f"Samples (n={n_samples})", fontsize=11, labelpad=10)
    elif n_samples > 50:
        # Too many samples: hide labels and show count
        ax.set_xticklabels([])
        ax.set_xlabel(f"Samples (n={n_samples})", fontsize=11, labelpad=10)
    else:
        # Show individual sample labels
        ax.set_xlabel("Samples", fontsize=11, labelpad=10)
        # Force all tick labels to show
        ax.set_xticks(np.arange(n_samples) + 0.5)
        ax.set_xticklabels(sample_names, rotation=90, ha="center", fontsize=8)

    # Configure Y-axis labels
    ax.set_yticks(np.arange(n_signatures) + 0.5)
    ax.set_yticklabels(signature_names, rotation=0, ha="right", fontsize=10)

    # Hide spines for cleaner look (consistent with cosine similarity heatmap)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Tight layout to prevent label cutoff (only if we created the figure)
    if own_figure and hasattr(fig, "tight_layout"):
        fig.tight_layout()  # type: ignore

    elapsed_time = time.time() - start_time
    logger.info(f"Signature contribution heatmap generated in {elapsed_time:.2f}s")

    return fig, ax  # type: ignore[return-value]


# ============================================================================
# STACKED BAR CHART VISUALIZATION
# ============================================================================


def _sort_samples_by_dominant_signature(H_normalized: np.ndarray) -> np.ndarray:
    """
    Sort samples by their dominant signature for better visualization.

    Sorting strategy:
    1. Group samples by dominant signature (signature with highest contribution)
    2. Within each group, sort by contribution magnitude (descending)

    Parameters
    ----------
    H_normalized : np.ndarray
        Normalized contribution matrix (k * n_samples), columns sum to 1

    Returns
    -------
    np.ndarray
        Array of sample indices in sorted order
    """
    n_signatures, n_samples = H_normalized.shape

    # Find dominant signature for each sample (argmax per column)
    dominant_signatures = np.argmax(H_normalized, axis=0)

    # Get contribution of dominant signature for each sample
    dominant_contributions = np.max(H_normalized, axis=0)

    # Create sorting keys: (dominant_signature, -contribution)
    # This groups by dominant signature and sorts descending within groups
    sort_keys = list(zip(dominant_signatures, -dominant_contributions))

    # Get sorted indices
    sorted_indices = np.array(sorted(range(n_samples), key=lambda i: sort_keys[i]))

    return sorted_indices


def _get_signature_colors_for_plot(n_signatures: int) -> List[str]:
    """
    Get consistent colors for signatures.

    If more signatures than predefined colors, generate additional colors
    using a colormap.

    Parameters
    ----------
    n_signatures : int
        Number of signatures to color

    Returns
    -------
    List[str]
        List of hex color codes
    """
    if n_signatures <= len(SIGNATURE_COLORS):
        return SIGNATURE_COLORS[:n_signatures]

    # Need more colors: extend with colormap
    import matplotlib.cm as cm

    additional_needed = n_signatures - len(SIGNATURE_COLORS)
    cmap = cm.get_cmap("tab20", additional_needed)
    additional_colors = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b, _ in [cmap(i) for i in range(additional_needed)]
    ]

    return SIGNATURE_COLORS + additional_colors


def _render_stacked_bars(
    ax: Axes,
    H_normalized: np.ndarray,
    sample_names: List[str],
    signature_names: List[str],
    colors: List[str],
) -> None:
    """
    Render stacked bars showing signature contributions per sample.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on
    H_normalized : np.ndarray
        Normalized contribution matrix (k * n_samples), columns sum to 1
    sample_names : List[str]
        Sample names (x-axis labels)
    signature_names : List[str]
        Signature names (legend labels)
    colors : List[str]
        Colors for each signature
    """
    n_signatures, n_samples = H_normalized.shape
    x_positions = np.arange(n_samples)

    # Draw stacked bars
    # Start with cumulative sum of 0 for each sample
    cumulative = np.zeros(n_samples)

    for sig_idx in range(n_signatures):
        contributions = H_normalized[sig_idx, :]

        # Draw bars for this signature
        ax.bar(
            x_positions,
            contributions,
            bottom=cumulative,
            color=colors[sig_idx],
            label=signature_names[sig_idx],
            width=0.8,
            edgecolor="white",
            linewidth=0.3,
        )

        # Update cumulative sum for next signature
        cumulative += contributions


def _style_stacked_bar_axis(
    ax: Axes,
    sample_names: List[str],
    n_samples: int,
    title: str = "Signature Contributions per Sample",
    show_labels: bool = True,
    legend_position: str = "right",
) -> None:
    """
    Style the stacked bar chart axis.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to style
    sample_names : List[str]
        Sample names for x-axis labels
    n_samples : int
        Total number of samples
    title : str
        Plot title
    show_labels : bool, default True
        Whether to show sample labels on X-axis
    legend_position : str, default 'right'
        Position of the legend. Options: 'right', 'bottom'
    """
    # Y-axis configuration
    ax.set_ylabel("Relative Contribution", fontsize=12, labelpad=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))

    # X-axis configuration
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_xticks(range(n_samples))

    if not show_labels:
        # Hide labels completely
        ax.set_xticklabels([])
        ax.set_xlabel(f"Samples (n={n_samples})", fontsize=12, labelpad=10)
    else:
        # Show labels - always show all sample names
        ax.set_xlabel("Samples", fontsize=12, labelpad=10)

        # Adjust font size based on number of samples for readability
        if n_samples <= 30:
            fontsize = 8
        elif n_samples <= 60:
            fontsize = 7
        else:
            fontsize = 6

        ax.set_xticklabels(sample_names, rotation=90, ha="center", fontsize=fontsize)

    # Title
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Legend with configurable position
    if legend_position == "bottom":
        # Horizontal legend below the plot (for composite figures)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(6, len(ax.get_legend_handles_labels()[0])),  # Max 6 columns
            frameon=True,
            framealpha=0.95,
            edgecolor="gray",
            fontsize=9,
            title=None,  # No title for bottom legend
        )
    else:
        # Vertical legend on the right side (default for individual plots)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
            framealpha=0.95,
            edgecolor="gray",
            fontsize=9,
            title="Mutational Signatures",
            title_fontsize=10,
        )

    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color("black")


def create_signature_donut_plot(
    contributions_abs: np.ndarray,
    signature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 8),
    title: str = "Relative Contribution of Mutational Signatures",
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Create a donut plot showing the global proportion of each signature across the cohort.

    This visualization shows the relative contribution of each signature based on
    absolute mutation counts summed across all samples. It provides a cohort-level
    overview of mutational processes.

    Parameters
    ----------
    contributions_abs : np.ndarray
        Absolute contribution matrix (k * n_samples) from extractSignatures['contributions_abs']
        Each entry represents the number of mutations attributed to that signature in that sample
    signature_names : Optional[List[str]]
        List of signature names for the legend. If None, defaults to "Signature 1", "Signature 2", etc.
    figsize : Tuple[int, int]
        Figure size (width, height) in inches
    title : str
        Plot title
    ax : Optional[Axes], default None
        Matplotlib axis to render on. If None, creates a new figure.
        Use this to embed the donut plot in a larger figure with multiple panels.

    Returns
    -------
    Tuple[Figure, Axes]
        Matplotlib figure and axes objects

    Raises
    ------
    ValueError
        If input dimensions are invalid

    Examples
    --------
    >>> result = py_mut.extractSignatures(contexts_df, n=3)
    >>> fig, ax = create_signature_donut_plot(
    ...     contributions_abs=result['contributions_abs'].values,
    ...     signature_names=["SBS1-like", "SBS5-like", "SBS13-like"]
    ... )

    Notes
    -----
    The donut plot uses absolute mutation counts (not normalized proportions) to
    calculate the global contribution of each signature. This means:
    - Signatures are summed across all samples: p = contributions_abs.sum(axis=1)
    - Final proportions are: p / p.sum()
    - The plot shows which signatures explain the most mutations cohort-wide
    """
    import time

    start_time = time.time()

    logger.info("Generating signature donut plot...")

    # Validate input
    if not isinstance(contributions_abs, np.ndarray):
        raise TypeError(
            f"contributions_abs must be a numpy array, got {type(contributions_abs)}"
        )

    if contributions_abs.ndim != 2:
        raise ValueError(
            f"contributions_abs must be a 2D array, got {contributions_abs.ndim}D"
        )

    n_signatures = contributions_abs.shape[0]

    # Generate default signature names if not provided
    if signature_names is None:
        signature_names = [f"Signature {i + 1}" for i in range(n_signatures)]
    elif len(signature_names) != n_signatures:
        raise ValueError(
            f"Number of signature_names ({len(signature_names)}) must match "
            f"number of signatures ({n_signatures})"
        )

    # Calculate total contribution per signature (sum across all samples)
    total_per_signature = contributions_abs.sum(axis=1)

    # Check for zero contributions
    if np.all(total_per_signature == 0):
        raise ValueError("All signatures have zero absolute contributions")

    # Calculate proportions
    proportions = total_per_signature / total_per_signature.sum()

    # Get consistent colors
    colors = _get_signature_colors_for_plot(n_signatures)

    # Create figure if axis not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_figure = True
    else:
        fig = ax.figure
        own_figure = False

    # Create donut chart
    wedges, texts, autotexts = ax.pie(
        proportions,
        labels=None,  # Labels will be in legend
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        pctdistance=0.85,
        textprops={"fontsize": 11, "fontweight": "bold", "color": "black"},
    )

    # Style percentage labels
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")

    # Add legend with signature names and percentages
    legend_labels = [
        f"{name}: {prop * 100:.1f}%" for name, prop in zip(signature_names, proportions)
    ]

    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=11,
    )

    # Set title
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Equal aspect ratio ensures circular shape
    ax.axis("equal")

    # Adjust layout to prevent legend cutoff (only if we created the figure)
    if own_figure:
        fig.tight_layout()

    elapsed_time = time.time() - start_time
    logger.info(f"Signature donut plot rendered in {elapsed_time:.2f}s")

    return fig, ax


def create_signature_stacked_bar_chart(
    H: np.ndarray,
    sample_names: List[str],
    signature_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Signature Contributions per Sample",
    sort_samples: bool = True,
    max_samples: Optional[int] = None,
    show_labels: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Create a stacked bar chart showing signature contributions per sample.

    Each bar represents one sample, with stacked colors showing the relative
    contribution of each signature. This visualization shows how signatures
    are distributed across samples.

    Parameters
    ----------
    H : np.ndarray
        Exposure matrix of shape (k, n_samples) where k is the number of signatures.
        This is the 'exposures' output from extractSignatures().
    sample_names : List[str]
        Names for each sample (n_samples names)
    signature_names : Optional[List[str]], default None
        Custom names for each signature. If None, uses generic names.
    figsize : Optional[Tuple[float, float]], default None
        Figure size as (width, height). If None, auto-calculated based on n_samples.
    title : str, default "Signature Contributions per Sample"
        Title for the plot
    sort_samples : bool, default True
        Whether to sort samples by dominant signature for better visualization
    max_samples : Optional[int], default None
        Maximum number of samples to display. If provided and there are more samples,
        only the first max_samples will be shown.
    show_labels : bool, default True
        Whether to show sample labels on X-axis. Set to False in composite plots
        to reduce clutter. Default is True.
    ax : Optional[Axes], default None
        Matplotlib axis to render on. If None, creates a new figure.
        Use this to embed the chart in a larger figure with multiple panels.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axis for further customization

    Raises
    ------
    ValueError
        If H matrix dimensions don't match sample_names length
        If signature_names length doesn't match number of signatures

    Notes
    -----
    - Contributions are normalized to sum to 1 (100%) per sample
    - Samples can be sorted by dominant signature for clearer patterns
    - Uses standard signature color palette
    - Legend shows signature names with color coding

    Examples
    --------
    >>> result = py_mut.extractSignatures(contexts_df, n=3)
    >>> fig, ax = create_signature_stacked_bar_chart(
    ...     result['exposures'],
    ...     sample_names=contexts_df.columns.tolist(),
    ...     signature_names=["SBS1", "SBS5", "SBS13"]
    ... )
    >>> plt.show()
    """
    import time

    start_time = time.time()

    logger.info("Generating signature stacked bar chart...")

    # Validate input dimensions
    SignatureValidator.validate_sample_names(sample_names, H.shape[1], "H")

    n_signatures = H.shape[0]
    n_samples = len(sample_names)

    # Apply max_samples filter if specified
    if max_samples is not None and n_samples > max_samples:
        logger.debug(f"Limiting display to first {max_samples} of {n_samples} samples")
        H = H[:, :max_samples]
        sample_names = sample_names[:max_samples]
        n_samples = max_samples

    # Generate default signature names if not provided
    if signature_names is None:
        signature_names = [f"Signature {i + 1}" for i in range(n_signatures)]
    else:
        SignatureValidator.validate_signature_names(signature_names, n_signatures)

    # Calculate figure size if not provided
    if figsize is None:
        width = max(10, min(30, n_samples * 0.3))
        height = 8
        figsize = (width, height)

    # Normalize contributions per sample
    H_normalized = _normalize_contributions_per_sample(H)

    # Sort samples by dominant signature if requested
    if sort_samples:
        sorted_indices = _sort_samples_by_dominant_signature(H_normalized)
        H_normalized = H_normalized[:, sorted_indices]
        sample_names = [sample_names[i] for i in sorted_indices]

    # Get colors for signatures
    colors = _get_signature_colors_for_plot(n_signatures)

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_figure = True
    else:
        fig = ax.figure  # type: ignore[assignment]
        own_figure = False

    # Render stacked bars
    _render_stacked_bars(ax, H_normalized, sample_names, signature_names, colors)

    # Apply styling with default 'right' legend position for individual plots
    _style_stacked_bar_axis(
        ax, sample_names, n_samples, title, show_labels, legend_position="right"
    )

    # Adjust layout only if we created the figure
    if own_figure and hasattr(fig, "tight_layout"):
        fig.tight_layout()  # type: ignore

    elapsed_time = time.time() - start_time
    logger.info(f"Signature stacked bar chart rendered in {elapsed_time:.2f}s")

    return fig, ax  # type: ignore[return-value]


# ============================================================================
# COMPLETE MUTATIONAL SIGNATURE ANALYSIS VISUALIZATION
# ============================================================================


def create_complete_signature_analysis(
    W: np.ndarray,
    H: np.ndarray,
    sample_names: List[str],
    cosmic_path: str,
    contributions_abs: Optional[np.ndarray] = None,
    signature_names: Optional[List[str]] = None,
    selected_signatures: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (24, 12),
    title: str = "Mutational Signature Analysis",
) -> Tuple[Figure, List[Axes]]:
    """
    Create a comprehensive mutational signature analysis visualization.

    This function combines all five signature visualizations into a single figure:
    A. Signature Bar Charts (96-context profiles for each signature)
    B. Cosine Similarity Heatmap (comparison with COSMIC signatures)
    C. Contribution Heatmap (absolute contributions per sample)
    D. Stacked Bar Chart (relative contributions per sample)
    E. Donut Plot (overall signature distribution)

    This is the main visualization function that provides a complete overview
    of mutational signature analysis results, similar to comprehensive reports
    in cancer genomics publications.

    Parameters
    ----------
    W : np.ndarray
        Signature matrix of shape (96, n_signatures) containing the trinucleotide
        context profiles for each extracted signature.
    H : np.ndarray
        Exposure matrix of shape (n_signatures, n_samples) containing the contribution
        of each signature in each sample. Can be normalized or absolute counts.
    sample_names : List[str]
        Names for each sample (n_samples names)
    cosmic_path : str
        Path to COSMIC signature catalog file (TSV format)
    contributions_abs : Optional[np.ndarray], default None
        Absolute contribution matrix of shape (n_signatures, n_samples) with mutation
        counts (not normalized). Used for the donut plot to calculate correct global
        proportions. If None, uses H (assumes H contains absolute counts).
    signature_names : Optional[List[str]], default None
        Custom names for each signature. If None, uses generic names like
        "Signature 1", "Signature 2", etc.
    selected_signatures : Optional[List[str]], default None
        Subset of signatures to plot (e.g., ["Signature 1", "Signature 3"]).
        If None, all signatures are plotted. This allows filtering to focus
        on specific signatures of interest.
    figsize : Tuple[int, int], default (24, 12)
        Figure size as (width, height) in inches. Optimized for 2-column layout
        with equal width columns (1/2 left, 1/2 right).
    title : str, default "Mutational Signature Analysis"
        Main title for the entire figure

    Returns
    -------
    fig : Figure
        Matplotlib figure containing all five visualization panels
    axes : List[Axes]
        List of all axes for potential further customization

    Raises
    ------
    ValueError
        If matrix dimensions are incompatible
        If signature_names length doesn't match number of signatures
    FileNotFoundError
        If COSMIC catalog file doesn't exist

    Notes
    -----
    Layout structure:
    - Top section (A): Signature bar charts, one per signature
    - Middle-left (B): Cosine similarity heatmap
    - Middle-right (C): Contribution heatmap
    - Bottom-left (D): Stacked bar chart
    - Bottom-right (E): Donut plot

    The function automatically handles:
    - Signature normalization
    - COSMIC catalog loading and validation
    - Color consistency across all panels
    - Proper spacing and layout

    Examples
    --------
    >>> # After extracting signatures
    >>> result = py_mut.extractSignatures(contexts_df, n=3)
    >>> fig, axes = create_complete_signature_analysis(
    ...     W=result['signatures'].values,
    ...     H=result['contributions'].values,
    ...     sample_names=contexts_df.columns.tolist(),
    ...     cosmic_path='path/to/COSMIC_v3.4_SBS_GRCh38.txt',
    ...     contributions_abs=result['contributions_abs'].values,
    ...     signature_names=["Signature A", "Signature B", "Signature C"]
    ... )
    >>> plt.savefig('complete_signature_analysis.png', dpi=300, bbox_inches='tight')
    >>> plt.show()

    See Also
    --------
    create_signature_bar_chart : Individual signature profiles
    create_cosine_similarity_heatmap : COSMIC comparison
    create_signature_contribution_heatmap : Absolute contributions
    create_signature_stacked_bar_chart : Relative contributions
    create_signature_donut_plot : Overall distribution
    """
    import time

    start_time = time.time()

    logger.info("Generating complete mutational signature analysis...")

    # Validate inputs
    SignatureValidator.validate_signature_matrix(W, "W")
    SignatureValidator.validate_exposure_matrix(H, W)
    SignatureValidator.validate_sample_names(sample_names, H.shape[1], "H")

    # Use contributions_abs if provided, otherwise use H (assumes H is absolute)
    if contributions_abs is None:
        contributions_abs = H
    else:
        if contributions_abs.shape != H.shape:
            raise ValueError(
                f"contributions_abs shape {contributions_abs.shape} must match H shape {H.shape}"
            )

    n_signatures = W.shape[1]

    # Generate default signature names if not provided
    if signature_names is None:
        signature_names = [f"Signature {i + 1}" for i in range(n_signatures)]
    else:
        SignatureValidator.validate_signature_names(signature_names, n_signatures)

    # Filter signatures if selected_signatures is provided
    if selected_signatures is not None:
        # Validate that selected signatures exist
        missing_sigs = set(selected_signatures) - set(signature_names)
        if missing_sigs:
            raise ValueError(
                f"Selected signatures not found in signature_names: {missing_sigs}. "
                f"Available signatures: {signature_names}"
            )

        # Get indices of selected signatures
        indices = [signature_names.index(name) for name in selected_signatures]

        # Filter W, H, and contributions_abs matrices
        W = W[:, indices]
        H = H[indices, :]
        contributions_abs = contributions_abs[indices, :]
        signature_names = selected_signatures
        n_signatures = len(selected_signatures)

    # Create figure with 2-column layout: 1/2 left, 1/2 right
    fig = plt.figure(figsize=figsize)

    # Use GridSpec for flexible layout with nested grids
    # Main layout: 2 columns (50-50 split)
    main_gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1, 1],  # 50-50 split
        wspace=0.25,
    )

    # Left column GridSpec: Signature bars (A) + Cosine heatmap (B) with colorbar
    # Use 2 columns: main content + narrow colorbar space
    left_gs = main_gs[0, 0].subgridspec(
        nrows=n_signatures + 1,
        ncols=2,
        width_ratios=[0.98, 0.02],  # 98% for content, 2% for colorbar
        hspace=0.5,
        wspace=0.05,
    )
    
    # Right column GridSpec: C, D, E with custom heights
    # C: Contribution heatmap (0.8x the height of B)
    # D: Stacked bar chart (0.8x the height of B) with legend beside it
    # E: Donut plot (1.5x the height of B for better visibility)
    right_gs = main_gs[0, 1].subgridspec(
        nrows=3,
        ncols=2,
        height_ratios=[0.8, 0.8, 1.5],
        width_ratios=[
            0.98,
            0.02,
        ],  # 98% plot, 2% colorbar/legend (matching left column)
        hspace=0.5,
        wspace=0.07,
    )
    
    all_axes = []
    
    # Panel A: Signature Bar Charts (96-context profiles)
    # Create axes for signature bar charts (one per signature, left column, first col)
    signature_axes = [fig.add_subplot(left_gs[i, 0]) for i in range(n_signatures)]
    all_axes.extend(signature_axes)
    
    # Use the refactored function to render signature bar charts
    _, _ = create_signature_bar_chart(
        signatures=W,
        signature_names=signature_names,
        figsize=None,  # Size already set by parent figure
        axes=signature_axes
    )
    
    # Add panel label "A" only to the first signature bar chart
    signature_axes[0].text(-0.08, 1.05, 'A', transform=signature_axes[0].transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='left')

    # Panel B: Cosine Similarity Heatmap (left column, bottom, spanning first column only)
    ax_cosine = fig.add_subplot(left_gs[n_signatures, 0])
    all_axes.append(ax_cosine)

    # Panel B Colorbar: Separate axis for the colorbar (right of heatmap)
    ax_cosine_cbar = fig.add_subplot(left_gs[n_signatures, 1])
    all_axes.append(ax_cosine_cbar)

    # Use the refactored function to render cosine similarity heatmap WITHOUT colorbar
    _, _ = create_cosine_similarity_heatmap(
        W=W,
        cosmic_path=cosmic_path,
        signature_names=signature_names,
        figsize=(14, 6),  # Dummy size, ax is provided
        title="",  # No individual title
        ax=ax_cosine,
        _cbar_ax=ax_cosine_cbar,  # Pass separate axis for colorbar
    )
    
    # Add panel label
    ax_cosine.text(-0.08, 1.05, 'B', transform=ax_cosine.transAxes,
                   fontsize=14, fontweight='bold', va='top', ha='left')
    
    # Prepare data for Panels C and D - use all samples (no arbitrary limit)
    # The individual functions handle max_samples internally if needed

    # Panel C: Contribution Heatmap (right column, top, first column only)
    ax_heatmap = fig.add_subplot(right_gs[0, 0])
    all_axes.append(ax_heatmap)

    # Panel C Colorbar: Separate axis for the colorbar (right of heatmap)
    ax_heatmap_cbar = fig.add_subplot(right_gs[0, 1])
    all_axes.append(ax_heatmap_cbar)

    # Create DataFrame with contributions (will be normalized by the function)
    contributions_df = pd.DataFrame(H, index=signature_names, columns=sample_names)

    # Use the refactored function to render contribution heatmap
    _, _ = create_signature_contribution_heatmap(
        contributions=contributions_df,
        signature_names=signature_names,
        figsize=None,  # Size already set by parent figure
        cmap="Blues",
        show_values=False,
        show_labels=True,  # Show sample labels on X-axis
        ax=ax_heatmap,
        _cbar_ax=ax_heatmap_cbar,  # Pass separate axis for colorbar
    )

    # Remove title to save space
    ax_heatmap.set_title("", fontsize=11, fontweight="bold", pad=8)
    ax_heatmap.set_ylabel("Signatures", fontsize=10, labelpad=6)
    ax_heatmap.tick_params(axis="y", rotation=0, labelsize=9)

    # Add panel label
    ax_heatmap.text(
        -0.08,
        1.02,
        "C",
        transform=ax_heatmap.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Panel D: Stacked Bar Chart (right column, middle, full width)
    ax_stacked = fig.add_subplot(right_gs[1, 0])
    all_axes.append(ax_stacked)

    # Panel D Legend: Separate space for legend (right column, middle, second column)
    ax_stacked_legend = fig.add_subplot(right_gs[1, 1])
    all_axes.append(ax_stacked_legend)
    ax_stacked_legend.axis("off")  # Hide the axis itself

    # Use the same data as Panel C to ensure consistency
    _, _ = create_signature_stacked_bar_chart(
        H=H,
        sample_names=sample_names,
        signature_names=signature_names,
        figsize=None,  # Size already set by parent figure
        title=None,  # No individual title
        sort_samples=False,  # Keep original MAF order
        max_samples=None,  # No arbitrary limit, function handles internally
        show_labels=True,  # Show sample labels on X-axis
        ax=ax_stacked,
    )
    
    # Adjust styling for compact display
    ax_stacked.set_title("", fontsize=11, fontweight='bold', pad=8)
    ax_stacked.tick_params(
        axis="x", labelsize=7
    )  # Show tick marks and labels on X-axis
    ax_stacked.tick_params(axis="y", labelsize=9)
    ax_stacked.set_ylabel("Relative Contribution", fontsize=9)

    # Move legend to the right outside the plot area
    handles, labels = ax_stacked.get_legend_handles_labels()
    if ax_stacked.get_legend() is not None:
        ax_stacked.get_legend().remove()  # Remove legend from main plot

    # Place legend in the dedicated legend axis
    ax_stacked_legend.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.05, 0.5),
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=7,
        title=None,
    )

    # Add panel label
    ax_stacked.text(
        -0.08,
        1.05,
        "D",
        transform=ax_stacked.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Panel E: Donut Plot (right column, bottom - larger)
    ax_donut = fig.add_subplot(right_gs[2, :])
    all_axes.append(ax_donut)

    # Calculate total contribution per signature (sum across all samples)
    total_per_signature = contributions_abs.sum(axis=1)

    # Check for zero contributions
    if np.all(total_per_signature == 0):
        raise ValueError("All signatures have zero absolute contributions")

    # Calculate proportions
    proportions = total_per_signature / total_per_signature.sum()

    # Get consistent colors
    colors = _get_signature_colors_for_plot(len(signature_names))

    # Create donut chart with adjusted parameters for composite plot
    wedges, texts, autotexts = ax_donut.pie(
        proportions,
        labels=None,  # Labels will be in legend
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        pctdistance=0.75,
        textprops={
            "fontsize": 9,
            "fontweight": "bold",
            "color": "black",
        },  # Smaller font
    )

    # Style percentage labels
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")

    # Add legend with signature names and percentages
    legend_labels = [
        f"{name}: {prop * 100:.1f}%" for name, prop in zip(signature_names, proportions)
    ]

    ax_donut.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=9,
    )

    # Equal aspect ratio ensures circular shape
    ax_donut.axis("equal")

    # Remove title to save space
    ax_donut.set_title("", fontsize=11, fontweight='bold', pad=8)
    
    # Add panel label
    ax_donut.text(
        -0.08,
        1.05,
        "E",
        transform=ax_donut.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )
    
    # Final adjustments
    # Add main title at the top with reduced spacing
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.0)
    
    # Adjust layout with tight spacing - rect=(left, bottom, right, top)
    # Reduce top margin to bring plots closer to title
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    
    elapsed_time = time.time() - start_time
    logger.info(f"Complete mutational signature analysis rendered in {elapsed_time:.2f}s")
    
    return fig, all_axes