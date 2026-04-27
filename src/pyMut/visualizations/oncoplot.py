"""
Functions for creating oncoplots (also known as waterfall plots).

This module contains the necessary functions to generate oncoplots, which are
heatmap visualizations showing mutation patterns across samples and genes.
Oncoplots are fundamental in cancer genomics for visualizing mutational landscapes.

Main functions:
- is_mutated(): Determines if a genotype represents a mutation
- detect_sample_columns(): Automatically detects sample columns in the DataFrame
- create_variant_color_mapping(): Creates color mapping for variant types
- process_mutation_matrix(): Processes mutation data into matrix format
- _create_oncoplot_plot(): Main function to create the oncoplot
"""

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.constants import (
    ALT_COLUMN,
    DEFAULT_ONCOPLOT_FIGSIZE,
    GENE_COLUMN,
    REF_COLUMN,
    VARIANT_CLASSIFICATION_COLUMN,
)

if TYPE_CHECKING:
    from ..core import PyMutation


def is_mutated(genotype: str, ref: str, alt: str) -> bool:
    """
    Determines if a genotype represents a mutation.

    Supports multiple genotype formats:
    - Pipe format: "A|G", "C|T"
    - Slash format: "A/G", "C/T"
    - Other separators: "A:G", "A;G"

    Args:
        genotype: Sample genotype value
        ref: Reference allele
        alt: Alternative allele

    Returns:
        True if genotype represents a mutation, False otherwise

    Examples:
        >>> is_mutated("A|G", "A", "G")  # True
        >>> is_mutated("A|A", "A", "G")  # False
        >>> is_mutated("G|G", "A", "G")  # True (homozygous alternative)
        >>> is_mutated("./.", "A", "G")  # False
    """
    if pd.isna(genotype) or pd.isna(ref) or pd.isna(alt):
        return False

    genotype_str = str(genotype).strip()
    alt_str = str(alt).strip()

    no_mutation_values = {"", ".", "0", "0/0", "0|0", "./.", ".|.", "NA", "NaN"}
    if genotype_str.upper() in no_mutation_values:
        return False

    separators = ["|", "/", ":", ";", ","]
    alleles = [genotype_str]

    for sep in separators:
        if sep in genotype_str:
            alleles = genotype_str.split(sep)
            break

    return alt_str in alleles


def detect_sample_columns(data: pd.DataFrame) -> List[str]:
    """
    Automatically detects columns representing samples in the DataFrame.

    Searches for common sample naming patterns:
    - TCGA format: columns starting with "TCGA-"
    - GT format: columns ending with ".GT"

    Args:
        data: DataFrame with mutation data

    Returns:
        List of column names identified as samples

    Raises:
        ValueError: If no sample columns are detected

    Examples:
        >>> columns = ["Hugo_Symbol", "TCGA-AB-2988", "TCGA-AB-2869", "Variant_Classification"]
        >>> detect_sample_columns(pd.DataFrame(columns=columns))
        ['TCGA-AB-2988', 'TCGA-AB-2869']
    """
    sample_columns = []

    tcga_cols = [col for col in data.columns if str(col).startswith("TCGA-")]
    sample_columns.extend(tcga_cols)

    gt_cols = [col for col in data.columns if str(col).endswith(".GT")]
    sample_columns.extend(gt_cols)

    sample_columns = list(dict.fromkeys(sample_columns))

    if not sample_columns:
        raise ValueError(
            "Could not automatically detect sample columns. "
            "Please ensure columns follow standard formats like 'TCGA-*' or '*.GT', "
            "or specify columns manually."
        )

    return sample_columns


def create_variant_color_mapping(variants: Set[str]) -> Dict[str, np.ndarray]:
    """
    Creates a consistent color mapping for variant types.

    Args:
        variants: Set of unique variant types

    Returns:
        Dictionary mapping variant type to RGB color array

    Examples:
        >>> variants = {"Missense_Mutation", "Nonsense_Mutation", "None"}
        >>> mapping = create_variant_color_mapping(variants)
        >>> len(mapping) == 3
        True
    """
    # Standard color palette for cancer genomics variant classifications
    predefined_colors = {
        "missense_mutation": np.array([34 / 255, 139 / 255, 34 / 255]),
        "nonsense_mutation": np.array([220 / 255, 20 / 255, 60 / 255]),
        "frame_shift_del": np.array([30 / 255, 144 / 255, 255 / 255]),
        "frame_shift_ins": np.array([167 / 255, 0 / 255, 204 / 255]),
        "in_frame_del": np.array([255 / 255, 215 / 255, 0 / 255]),
        "in_frame_ins": np.array([250 / 255, 60 / 255, 88 / 255]),
        "splice_site": np.array([255 / 255, 140 / 255, 0 / 255]),
        "translation_start_site": np.array([255 / 255, 228 / 255, 181 / 255]),
        "nonstop_mutation": np.array([218 / 255, 112 / 255, 214 / 255]),
        "silent": np.array([176 / 255, 196 / 255, 222 / 255]),
        "multi_hit": np.array([0 / 255, 0 / 255, 0 / 255]),
        "none": np.array([245 / 255, 245 / 255, 245 / 255]),
    }

    color_mapping = {}

    for variant in variants:
        # Normalize to lowercase for matching
        variant_lower = str(variant).lower()
        if variant_lower in predefined_colors:
            color_mapping[variant] = predefined_colors[variant_lower]

    unassigned_variants = [v for v in variants if v not in color_mapping]

    if unassigned_variants:
        n_colors = len(unassigned_variants)
        auto_colors = plt.cm.tab20(np.linspace(0, 1, n_colors))

        for i, variant in enumerate(unassigned_variants):
            color_mapping[variant] = auto_colors[i][:3]

    return color_mapping


def process_mutation_matrix(
    data: pd.DataFrame,
    gene_column: str = GENE_COLUMN,
    variant_column: str = VARIANT_CLASSIFICATION_COLUMN,
    ref_column: str = REF_COLUMN,
    alt_column: str = ALT_COLUMN,
    sample_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    required_columns = [gene_column, variant_column]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if sample_columns is None:
        try:
            sample_columns = detect_sample_columns(data)
        except ValueError:
            excluded_cols = {gene_column, variant_column, ref_column, alt_column,
                            "Tumor_Sample_Barcode"}
            potential_samples = [col for col in data.columns if col not in excluded_cols]
            if potential_samples:
                sample_columns = potential_samples
            else:
                raise ValueError("No valid sample columns detected")

    if not sample_columns:
        raise ValueError("No valid sample columns detected")

    unique_genes = data[gene_column].dropna().unique()
    unique_genes = [g for g in unique_genes if str(g) != "nan" and str(g).strip() != ""]
    if not unique_genes:
        raise ValueError("No valid genes found in the data")

    has_ref_alt = ref_column in data.columns and alt_column in data.columns

    # --- Vectorized mutation detection ---
    # For pipe-separated genotypes (VCF-style), a sample is mutated if genotype
    # is not NA and not equal to REF|REF.
    if has_ref_alt:
        ref_geno = data[ref_column].astype(str) + "|" + data[ref_column].astype(str)
    sample_df = data[sample_columns]
    not_na = sample_df.notna()
    no_mutation_vals = {"", ".", "0", "0/0", "0|0", "./.", ".|.", "NA", "NaN", "None"}

    if has_ref_alt:
        not_ref = sample_df.ne(ref_geno, axis=0)
        not_no_mut = ~sample_df.isin(no_mutation_vals)
        has_mutation_mask = not_na & not_ref & not_no_mut
    else:
        not_no_mut = ~sample_df.isin(no_mutation_vals)
        has_mutation_mask = not_na & not_no_mut

    # Melt to long format: (row_idx, sample, is_mutated)
    # Attach gene and variant columns
    melted = has_mutation_mask.copy()
    melted[gene_column] = data[gene_column].values
    melted[variant_column] = data[variant_column].values

    # Filter to rows with valid gene/variant
    valid = melted[gene_column].notna() & melted[variant_column].notna()
    melted = melted[valid]

    # For each gene-sample pair, collect variant types
    mutation_matrix = pd.DataFrame("None", index=unique_genes, columns=sample_columns)
    mutation_counts = {gene: 0 for gene in unique_genes}

    # Stack sample columns into long form efficiently
    gene_col_vals = melted[gene_column].values
    variant_col_vals = melted[variant_column].values
    mask_vals = melted[sample_columns].values  # bool array (rows × samples)

    # Find all (row_idx, sample_idx) where mutation is True
    row_indices, col_indices = np.where(mask_vals)

    # Build a dict of (gene, sample) -> list of variant types
    from collections import defaultdict
    event_counts = defaultdict(list)
    for r, c in zip(row_indices, col_indices):
        gene_str = str(gene_col_vals[r]).strip()
        variant_str = str(variant_col_vals[r]).strip()
        if gene_str and variant_str and gene_str in mutation_counts:
            event_counts[(gene_str, sample_columns[c])].append(variant_str)

    # Populate matrix
    for (gene_str, sample_col), variant_list in event_counts.items():
        if len(variant_list) == 1:
            mutation_matrix.loc[gene_str, sample_col] = variant_list[0]
        else:
            mutation_matrix.loc[gene_str, sample_col] = "Multi_Hit"
        mutation_counts[gene_str] += 1

    return mutation_matrix, mutation_counts


def _create_oncoplot_plot(
    py_mut: "PyMutation",
    gene_column: str = GENE_COLUMN,
    variant_column: str = VARIANT_CLASSIFICATION_COLUMN,
    ref_column: str = REF_COLUMN,
    alt_column: str = ALT_COLUMN,
    top_genes_count: int = 30,
    max_samples: int = 193,
    figsize: Optional[Tuple[int, int]] = None,
    title: str = "Oncoplot",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Creates a complete oncoplot visualization.

    Generates a comprehensive oncoplot that includes:
    - Upper panel: TMB (Tumor Mutation Burden) per sample
    - Main panel: Mutation heatmap (genes x samples)
    - Right side panel: Gene mutation frequency barplot
    - Bottom panel: Legend for variant classifications

    Args:
        py_mut: PyMutation object with mutation data
        gene_column: Name of the gene column (default 'Hugo_Symbol')
        variant_column: Name of the variant classification column
        ref_column: Name of the reference allele column
        alt_column: Name of the alternative allele column
        top_genes_count: Number of top mutated genes to show (default 30)
        max_samples: Maximum number of samples to show (default 193)
        figsize: Figure size (width, height) in inches
        title: Title for the plot
        ax: Existing matplotlib axes (optional, for simple mode without panels)

    Returns:
        matplotlib Figure object with the complete oncoplot

    Raises:
        ValueError: If required columns are missing or no data to visualize
    """
    start_time = time.time()
    data = py_mut.data

    try:
        mutation_matrix, mutation_counts = process_mutation_matrix(
            data, gene_column, variant_column, ref_column, alt_column
        )

        if mutation_matrix.empty:
            raise ValueError("No mutation data to visualize")

        top_genes = sorted(mutation_counts.items(), key=lambda x: x[1], reverse=True)
        top_genes = [gene for gene, count in top_genes[:top_genes_count] if count > 0]

        if not top_genes:
            raise ValueError("No genes with mutations found")

        plot_matrix = mutation_matrix.loc[top_genes].copy()

        # Apply waterfall/cascade sorting algorithm
        unique_variants = set()
        for row in plot_matrix.values.flatten():
            unique_variants.add(row)

        unique_values = sorted(unique_variants)
        value_to_num = {value: i for i, value in enumerate(unique_values)}

        # Convert to numeric matrix for sorting operations
        with pd.option_context("future.no_silent_downcasting", True):
            numeric_matrix = plot_matrix.replace(value_to_num).infer_objects(copy=False)
        waterfall_matrix = numeric_matrix.copy()

        # Sort genes by mutation frequency (fewer non-mutated samples = higher priority)
        # Genes with equal frequency maintain their original order for stability
        if "None" in value_to_num:
            gene_zero_counts = (waterfall_matrix == value_to_num["None"]).sum(axis=1)
        else:
            min_value = min(value_to_num.values())
            gene_zero_counts = (waterfall_matrix == min_value).sum(axis=1)

        # Create sorting DataFrame with tiebreaker based on original order
        gene_sorting_df = pd.DataFrame(
            {
                "gene": waterfall_matrix.index,
                "zero_count": gene_zero_counts.values,
                "original_order": range(len(waterfall_matrix)),
            }
        )

        # Sort by mutation frequency (ascending zero_count), then by original position
        gene_sorting_df = gene_sorting_df.sort_values(
            by=["zero_count", "original_order"], ascending=[True, True]
        )

        sorted_genes = gene_sorting_df["gene"].tolist()
        waterfall_matrix = waterfall_matrix.loc[sorted_genes]

        binary_matrix = (plot_matrix != "None").astype(int)

        # Pre-sort samples alphabetically for consistent tiebreaking
        binary_matrix = binary_matrix[sorted(binary_matrix.columns)]

        # Sort samples using lexicographic ordering by mutation pattern
        # Samples with more mutations in high-priority genes come first
        binary_array = binary_matrix.values  # genes x samples

        # Create sort keys in reverse order for numpy lexsort
        # lexsort processes keys from last to first
        sort_keys = [-binary_array[i, :] for i in range(len(sorted_genes) - 1, -1, -1)]

        # Get sorting indices and apply to sample names
        sort_indices = np.lexsort(sort_keys)
        sorted_samples = [binary_matrix.columns[i] for i in sort_indices]

        if max_samples and len(sorted_samples) > max_samples:
            waterfall_matrix = waterfall_matrix.iloc[:, :max_samples]
            sorted_samples = sorted_samples[:max_samples]

        plot_matrix = plot_matrix.loc[sorted_genes, sorted_samples]

        # Create color mapping
        unique_variants = set()
        for row in plot_matrix.values.flatten():
            unique_variants.add(row)

        color_mapping = create_variant_color_mapping(unique_variants)

        # Convert categorical values to numeric for heatmap
        unique_values = sorted(unique_variants)
        value_to_num = {value: i for i, value in enumerate(unique_values)}

        numeric_matrix = plot_matrix.map(lambda x: value_to_num[x])

        # Configure figure and subplots
        if figsize is None:
            figsize = DEFAULT_ONCOPLOT_FIGSIZE

        if ax is None:
            fig = plt.figure(figsize=figsize)

            # Grid layout: TMB panel, heatmap, gene panel, and legend with more space
            gs = fig.add_gridspec(
                3,
                2,
                height_ratios=[2.5, 10, 2.5],
                width_ratios=[10, 1],
                hspace=0.25,
                wspace=0.02,
            )

            ax_tmb = fig.add_subplot(gs[0, 0])
            ax_main = fig.add_subplot(gs[1, 0])
            ax_genes = fig.add_subplot(gs[1, 1])
            ax_legend = fig.add_subplot(gs[2, :])

            # TMB panel
            # Only include variant types that are present in the color_mapping
            valid_variant_types = set(color_mapping.keys()) - {"None"}
            detected_sample_columns = mutation_matrix.columns.tolist()
            variant_counts_tmb = {}

            # Filter to valid variant types
            valid_data = data[data[variant_column].isin(valid_variant_types)]

            # Build ref genotype for vectorized comparison
            ref_geno = valid_data[ref_column].astype(str) + "|" + valid_data[ref_column].astype(str)

            # Boolean mask: is mutated in each sample column
            sample_df = valid_data[detected_sample_columns]
            has_mutation = sample_df.notna() & sample_df.ne(ref_geno, axis=0)

            # Group by variant type and sum across rows
            variant_types_col = valid_data[variant_column].values
            variant_counts_tmb = {}
            for sample_col in detected_sample_columns:
                col_mask = has_mutation[sample_col].values
                counts = pd.Series(variant_types_col[col_mask]).value_counts().to_dict()
                if counts:
                    variant_counts_tmb[sample_col] = counts

            samples_df_tmb = []
            for sample, variants in variant_counts_tmb.items():
                for var_type, count in variants.items():
                    samples_df_tmb.append(
                        {
                            "Sample": sample,
                            "Variant_Classification": var_type,
                            "Count": count,
                        }
                    )

            if samples_df_tmb:
                processed_df_tmb = pd.DataFrame(samples_df_tmb)
                tmb_df = processed_df_tmb.pivot(
                    index="Sample", columns="Variant_Classification", values="Count"
                ).fillna(0)

                tmb_df = tmb_df.loc[tmb_df.index.intersection(sorted_samples)]
                tmb_df = tmb_df.reindex(sorted_samples, fill_value=0)

                tmb_colors = [
                    color_mapping.get(vt, [0.7, 0.7, 0.7]) for vt in tmb_df.columns
                ]

                tmb_df.plot(
                    kind="bar",
                    stacked=True,
                    ax=ax_tmb,
                    color=tmb_colors,
                    width=0.8,
                    legend=False,
                )

                ax_tmb.set_ylabel("TMB", fontsize=10)
                ax_tmb.set_xlim(-0.5, len(sorted_samples) - 0.5)

                max_tmb = tmb_df.sum(axis=1).max() if not tmb_df.empty else 10
                ax_tmb.set_ylim(0, max_tmb * 1.1)

                # Set Y-axis ticks to show only 0, mid, and max
                mid_tmb = int(max_tmb / 2)
                max_tmb_int = int(max_tmb)
                ax_tmb.set_yticks([0, mid_tmb, max_tmb_int])
                ax_tmb.set_yticklabels([0, mid_tmb, max_tmb_int])

                ax_tmb.tick_params(
                    axis="x", which="both", bottom=False, labelbottom=False
                )
                ax_tmb.set_xlabel("")
                ax_tmb.spines["top"].set_visible(False)
                ax_tmb.spines["right"].set_visible(False)
                ax_tmb.spines["bottom"].set_visible(False)
                ax_tmb.grid(axis="y", alpha=0.3)
            else:
                ax_tmb.text(
                    0.5, 0.5, "TMB not available", ha="center", va="center", fontsize=10
                )
                ax_tmb.set_xlim(0, 1)
                ax_tmb.set_ylim(0, 1)
                ax_tmb.axis("off")

            # Gene panel (right side)
            try:
                # Count unique samples per gene and variant type
                gene_sample_counts = {}

                for gene in sorted_genes:
                    gene_sample_counts[gene] = {}
                    gene_row = plot_matrix.loc[gene]

                    for variant_type in gene_row.value_counts().index:
                        if variant_type != "None":
                            # Count the number of samples with this variant type for this gene
                            count = (gene_row == variant_type).sum()
                            gene_sample_counts[gene][variant_type] = count

                variant_types = set()
                for gene_variants in gene_sample_counts.values():
                    variant_types.update(gene_variants.keys())

                variant_types = sorted(list(variant_types))

                stacked_data = []
                for gene in sorted_genes:
                    row_data = {"gene": gene}
                    for variant_type in variant_types:
                        row_data[variant_type] = gene_sample_counts[gene].get(
                            variant_type, 0
                        )
                    stacked_data.append(row_data)

                df_stacked = pd.DataFrame(stacked_data).set_index("gene")
                df_stacked = df_stacked.iloc[::-1]

                for col in df_stacked.columns:
                    if pd.api.types.is_numeric_dtype(df_stacked[col]):
                        df_stacked[col] = df_stacked[col].astype(float)

                stacked_colors = []
                for variant_type in variant_types:
                    if variant_type in color_mapping:
                        stacked_colors.append(color_mapping[variant_type])
                    else:
                        stacked_colors.append([0.7, 0.7, 0.7])

                df_stacked.plot(
                    kind="barh",
                    stacked=True,
                    ax=ax_genes,
                    color=stacked_colors,
                    width=0.65,
                )

                ax_genes.set_xlabel("Nº of samples", fontsize=10)
                ax_genes.set_ylabel("")

                max_samples_gene = (
                    df_stacked.sum(axis=1).max() if not df_stacked.empty else 50
                )
                ax_genes.set_xlim(0, max_samples_gene * 1.1)
                ax_genes.set_ylim(-0.5, len(sorted_genes) - 0.5)

                # Set X-axis ticks to show only 0, mid, and max
                mid_samples = int(max_samples_gene / 2)
                max_samples_int = int(max_samples_gene)
                ax_genes.set_xticks([0, mid_samples, max_samples_int])
                ax_genes.set_xticklabels([0, mid_samples, max_samples_int])

                ax_genes.set_yticklabels(["" for _ in sorted_genes])

                sample_cols_for_percentage = [
                    col for col in data.columns if str(col).startswith("TCGA-")
                ]
                total_samples_in_dataset = len(sample_cols_for_percentage)

                # Pre-compute ref genotypes
                ref_geno = data[ref_column].astype(str) + "|" + data[ref_column].astype(str)
                sample_df = data[sample_cols_for_percentage]
                has_mut = sample_df.notna() & sample_df.ne(ref_geno, axis=0)

                for i, gene in enumerate(df_stacked.index):
                    gene_in_original_order = sorted_genes[len(sorted_genes) - 1 - i]
                    gene_mask = data[gene_column] == gene_in_original_order
                    num_unique_samples_affected = has_mut.loc[gene_mask].any(axis=0).sum()
                    real_percentage = (num_unique_samples_affected / total_samples_in_dataset) * 100 \
                        if total_samples_in_dataset > 0 else 0
                    if real_percentage > 0:
                        bar_length = df_stacked.loc[gene].sum()
                        offset = max_samples_gene * 0.02
                        ax_genes.text(bar_length + offset, i, f"{real_percentage:.1f}%",
                                      va="center", fontsize=9)

                legend = ax_genes.get_legend()
                if legend is not None:
                    legend.remove()

                ax_genes.spines["top"].set_visible(False)
                ax_genes.spines["right"].set_visible(False)
                ax_genes.spines["left"].set_visible(False)
                ax_genes.tick_params(axis="y", which="both", left=False, right=False)
                ax_genes.tick_params(axis="x", labelsize=9)
                ax_genes.grid(axis="x", alpha=0.3)
                ax_genes.margins(x=0.05, y=0.01)

            except Exception:
                ax_genes.text(
                    0.5,
                    0.5,
                    "Gene panel\nnot available",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                ax_genes.set_xlim(0, 1)
                ax_genes.set_ylim(0, 1)
                ax_genes.axis("off")

            # Legend panel (bottom)
            legend_elements = []
            for variant in sorted(unique_variants):
                if variant != "None":
                    color = color_mapping[variant]
                    label = variant.replace("_", " ")
                    legend_elements.append(
                        plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label)
                    )

            if legend_elements:
                ax_legend.legend(
                    handles=legend_elements,
                    title="Variant Classification",
                    loc="upper center",
                    ncol=min(len(legend_elements), 6),
                    fontsize=10,
                    title_fontsize=11,
                    frameon=True,
                    edgecolor="black",
                    fancybox=False,
                )

            ax_legend.axis("off")

            fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)

        else:
            fig = ax.get_figure()
            ax_main = ax

        # Main heatmap
        colors = [color_mapping[value] for value in unique_values]
        custom_cmap = mcolors.ListedColormap(colors)

        ax_main.imshow(
            numeric_matrix.values,
            cmap=custom_cmap,
            aspect="auto",
            interpolation="nearest",
        )

        if ax is not None:
            ax_main.set_title(title, fontsize=16, fontweight="bold", pad=20)

        # Calculate the number of altered samples (samples with at least one mutation)
        total_samples_count = len(sorted_samples)
        altered_samples = set()
        for sample in sorted_samples:
            sample_has_mutation = False
            for gene in sorted_genes:
                if plot_matrix.loc[gene, sample] != "None":
                    sample_has_mutation = True
                    break
            if sample_has_mutation:
                altered_samples.add(sample)

        n_altered = len(altered_samples)
        pct_altered = (
            (n_altered / total_samples_count) * 100 if total_samples_count > 0 else 0
        )

        ax_main.set_xlabel(
            f"Altered in {n_altered} ({pct_altered:.2f}%) of {total_samples_count} samples",
            fontsize=12,
        )
        ax_main.set_ylabel("Genes", fontsize=12)

        ax_main.set_xticks(range(len(sorted_samples)))
        ax_main.set_yticks(range(len(sorted_genes)))
        ax_main.set_yticklabels(sorted_genes, fontsize=10)

        ax_main.set_xticklabels([""] * len(sorted_samples))
        ax_main.tick_params(axis="x", which="both", bottom=False, top=False)

        ax_main.set_xticks(np.arange(-0.5, len(sorted_samples), 1), minor=True)
        ax_main.set_yticks(np.arange(-0.5, len(sorted_genes), 1), minor=True)
        ax_main.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

        if ax is not None:
            legend_elements = []
            for variant in sorted(unique_variants):
                if variant != "None":
                    color = color_mapping[variant]
                    label = variant.replace("_", " ")
                    legend_elements.append(
                        plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label)
                    )

            if legend_elements:
                ax_main.legend(
                    handles=legend_elements,
                    title="Variant Classification",
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    fontsize=9,
                    title_fontsize=10,
                )

        elapsed_time = time.time() - start_time
        print(
            f"Oncoplot generated: {plot_matrix.shape[0]} genes × {plot_matrix.shape[1]} samples ({elapsed_time:.2f}s)"
        )
        
        return fig 
        
    except Exception as e:
        print(f"Error creating oncoplot: {e}")
        raise