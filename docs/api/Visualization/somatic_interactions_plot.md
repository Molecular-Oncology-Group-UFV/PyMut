# Somatic Interactions

## Overview

The `somatic_interactions()` plot visualizes how pairs of genes are mutated together or separately in a cohort of tumor samples. It answers questions such as:

- Do mutations in gene A and gene B tend to appear in the same samples (co-occurrence)?  
- Or do they tend to avoid each other, with samples usually mutated in only one of them (mutual exclusivity)?  

The plot is a triangular heatmap where each cell represents a pair of genes. The color encodes the strength and direction of the interaction, and optional symbols mark statistically significant interactions. The visualization is based on pairwise Fisher's exact tests applied to a binary mutation matrix.

## Input Data

The visualization requires a `PyMutation` object containing mutation data with at least:

- Sample identifiers (column name specified by `sample_column`, default: `Tumor_Sample_Barcode`)  
- Gene names (column name specified by `gene_column`, default: `Hugo_Symbol`)  

From this data, a binary mutation matrix is constructed where:

- Rows represent tumor samples (all samples from the dataset, including those with no mutations in the selected genes)  
- Columns represent genes (top N most frequently mutated genes)  
- Values indicate presence (1) or absence (0) of mutations in each gene for each sample  

By design, the matrix includes all samples from the dataset, even those without mutations in any of the selected genes. This ensures accurate statistical calculations.

## Gene Selection and Ordering

By default, `somatic_interactions()` selects the top 25 genes based on the number of altered samples. Genes are ranked by:

1. Number of altered samples (descending)  
2. Total number of mutations (descending, for tie-breaking during selection)  
3. Gene name (ascending, for deterministic ordering within tied groups)  

This ensures reproducible results with a clear visual hierarchy.

## Statistical Method

For every pair of genes (G1, G2), all samples are examined and counted according to four possible patterns:

- Neither gene mutated  
- Only G2 mutated  
- Only G1 mutated  
- Both genes mutated  

These counts form a 2×2 contingency table used for Fisher's exact test, which provides:

- A p-value measuring how strongly the pattern deviates from independence  
- An odds ratio (OR) indicating the direction of the relationship:  
      - OR > 1: genes co-occur more often than expected (co-occurrence)  
      - OR < 1: genes are more often found separately than expected (mutual exclusivity)  

To encode both significance and direction in a single value, the p-value is transformed:

- Co-occurring pairs (OR > 1) get positive values: `-log10(p-value)`  
- Mutually exclusive pairs (OR < 1) get negative values: `log10(p-value)`  

P-values smaller than 1e-10 are clipped to avoid numerical issues.

## Plot Layout

The interaction matrix is square (same genes on rows and columns). To avoid duplicate information:

- X-axis (columns): genes ordered from most mutated to least mutated (left to right)  
- Y-axis (rows): genes ordered from least mutated to most mutated (top to bottom)  
- Only the left half of the matrix is shown (cells on or to the left of the anti-diagonal), avoiding symmetric gene pair duplicates  
- The diagonal (same gene vs itself) is hidden  

Each row label on the left and column label on the top corresponds to one gene, optionally formatted as `GENE_NAME [N]` where N is the number of altered samples (controlled by `show_counts`).

## Color Encoding

Each visible cell represents one gene pair. Its color encodes:

- Magnitude: strength of the association (how significant the interaction is)  
- Sign: direction of the association (co-occurrence vs mutual exclusivity)  

A diverging color palette (BrBG - Brown-Blue-Green) is used:

- Blue/green side: co-occurrence (positive values)  
- Brown/orange side: mutual exclusivity (negative values)  
- Pale/white colors near zero: no evidence of interaction (p close to 1)  

The color scale is bounded by default to [-3, 3] (controlled by `vmin` and `vmax`). Values outside this range are displayed at maximum intensity.

## Significance Symbols

Black symbols are overlaid on cells to mark statistically significant pairs. The `pvalue` parameter accepts two thresholds (default: (0.05, 0.1)):

- Asterisk (*): highly significant pairs (p < min(pvalue), typically p < 0.05)  
- Dot (·): moderately significant pairs (min(pvalue) ≤ p < max(pvalue), typically 0.05 ≤ p < 0.1)  

Pairs with p ≥ max(pvalue) are drawn without a symbol. The order of values in the tuple does not matter as the function uses min() and max() internally.

## Color Legend

A vertical color bar is displayed on the right side showing:

- Tick labels: < -3, -2, -1, 0, 1, 2, > 3  
- Bottom (< -3): strong mutual exclusivity  
- Top (> 3): strong co-occurrence  
- Label: `-log10(P-value)` (positioned on the left side of the colorbar)  

This legend allows interpretation of both direction (which side of the scale) and strength (distance from zero) of each interaction.

## Parameters

- `top_genes` (int, default: 25): Number of most frequently mutated genes to include in the analysis. Genes are ranked by number of altered samples.  
- `gene_column` (str, default: "Hugo_Symbol"): Column name containing gene symbols.  
- `sample_column` (str, default: "Tumor_Sample_Barcode"): Column name containing sample identifiers.  
- `figsize` (tuple, default: (12, 10)): Figure size as (width, height) in inches.  
- `title` (str or None, default: None): Custom plot title. If None, auto-generated as "Somatic Interactions (Top N Mutated Genes)".  
- `vmin` (float, default: -3.0): Minimum value for color scale. Controls saturation for mutual exclusivity (negative values).  
- `vmax` (float, default: 3.0): Maximum value for color scale. Controls saturation for co-occurrence (positive values).  
- `pvalue` (tuple, default: (0.05, 0.1)): Tuple of two p-value thresholds for significance markers. The smaller value (typically 0.05) controls asterisks (*), the larger value (typically 0.1) controls dots (·). Applied to whichever p-value metric is shown (raw or FDR-corrected). Order does not matter.  
- `show_counts` (bool, default: True): Whether to display sample counts next to gene names in format "GENE [count]" (e.g., "TP53 [52]").  
- `fdr_correction` (bool, default: False): If `False` (default), heatmap colors and significance markers are based on the raw Fisher's exact test p-values. If `True`, both the colors and significance markers reflect False Discovery Rate (FDR) adjusted p-values, computed across the $C(top\_genes, 2)$ pairwise tests performed. The colorbar label and legend automatically update to indicate whether raw or FDR-adjusted values are shown.
- `fdr_method` (str, default: "fdr_bh"): Multiple-testing correction method forwarded to `statsmodels.stats.multitest.multipletests` (e.g., `"fdr_bh"` for Benjamini-Hochberg, `"bonferroni"`, `"holm"`). Only evaluated when `fdr_correction=True`.

## `somatic_interactions_table()`

In addition to the visual heatmap, pyMut provides the companion function `somatic_interactions_table()` to obtain the underlying numerical statistics in tabular form (`pd.DataFrame`).

```python
table = py_mut.somatic_interactions_table(
    top_genes=25,
    gene_column="Hugo_Symbol",
    sample_column="Tumor_Sample_Barcode",
    fdr_method="fdr_bh"
)
```

### Table Output Columns

- `gene1`, `gene2`: The pair of genes being tested.
- `n_both`: Number of samples mutated in both genes.
- `n_gene1_only`: Number of samples mutated only in gene 1.
- `n_gene2_only`: Number of samples mutated only in gene 2.
- `n_neither`: Number of samples wild-type for both genes.
- `odds_ratio`: Odds ratio from Fisher's exact test ($>1$ indicates co-occurrence, $<1$ indicates mutual exclusivity).
- `p_value`: Raw p-value from Fisher's exact test.
- `fdr`: Adjusted p-value using the specified multiple testing correction method (e.g., Benjamini-Hochberg).
- `event`: Classification string (`"Co-occurrence"`, `"Mutually Exclusive"`, or `"None"`).

Because `somatic_interactions_table()` uses the exact same filtering, sample population, and gene selection criteria as `somatic_interactions()`, the tabular values directly reflect what is displayed in the plot.

## Interpreting the Plot

1. Gene list and frequencies  
      - Gene names are shown on the left (Y-axis) and top (X-axis)  
      - Numbers in brackets (e.g., `DNMT3A [48]`) indicate how many samples have that gene mutated  
      - Genes are ordered by mutation frequency (most to least on X-axis; least to most on Y-axis)  

2. Locate a gene pair  
      - Choose a row gene (Y-axis) and a column gene (X-axis)  
      - The cell at their intersection represents the interaction between the two genes  
      - Only cells on or to the left of the anti-diagonal are visible  

3. Interpret the color  
      - Blue/green: Co-occurrence (genes mutated together more than expected)  
      - Brown/orange: Mutual exclusivity (genes rarely mutated together)  
      - White/pale: No significant interaction (p ≈ 1)  
      - More intense colors indicate stronger evidence (smaller p-values)  

4. Check significance symbols  
      - Asterisk (*): Highly significant (p < 0.05)  
      - Dot (·): Moderately significant (0.05 ≤ p < 0.1)  
      - No symbol: Not significant (p ≥ 0.1)  

5. Use the colorbar for magnitude  
      - The `-log10(P-value)` scale shows interaction strength  
      - Positive values (1, 2, >3): Co-occurrence of increasing strength  
      - Negative values (-1, -2, <-3): Mutual exclusivity of increasing strength  
      - Example: value = 2 means p ≈ 0.01; value = 3 means p ≈ 0.001  

## Example Usage

```python
import pyMut

# Load mutation data
py_mut = pyMut.read_maf("tcga_laml.maf")

# Basic usage (top 25 genes, default parameters)
fig = py_mut.somatic_interactions()
py_mut.save_figure(fig, "somatic_interactions.png")

# Analyze more genes with custom thresholds
fig = py_mut.somatic_interactions(
    top_genes=50,
    pvalue=(0.01, 0.001),  # Stricter significance thresholds
    vmin=-5,
    vmax=5,
    figsize=(14, 12)
)

# Minimal plot without sample counts
fig = py_mut.somatic_interactions(
    top_genes=15,
    show_counts=False,
    title="Gene Interactions in LAML"
)
```
