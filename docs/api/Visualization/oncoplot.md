# Oncoplot Documentation

## Overview

The `oncoplot()` visualization displays somatic mutation patterns across genes and samples in a single comprehensive figure. This plot type (also known as *waterfall plot* or *oncoPrint*) combines multiple coordinated panels to provide a complete view of the mutational landscape in a cancer cohort.

## Structure

The visualization consists of four main components:

- Center panel: Mutation matrix showing gene-sample mutation status  
- Top panel: Tumor Mutation Burden (TMB) per sample  
- Right panel: Gene alteration frequency  
- Bottom panel: Variant classification legend  

### Matrix layout

Rows represent genes (typically the most frequently mutated), columns represent tumor samples, and individual cells show mutation status color-coded by variant classification type.

## What it shows

This visualization provides:

- Which genes are most frequently mutated in the cohort  
- How mutations are distributed across individual samples  
- The variant type composition for each gene and sample  
- Overall mutational patterns and co-occurrence relationships  

## Data requirements

### Required columns

The visualization requires a `PyMutation` object containing mutation data with these columns:

- Gene identifier (default: `Hugo_Symbol`)  
- Variant classification (default: `Variant_Classification`)  
- Reference allele (default: `REF`)  
- Alternative allele (default: `ALT`)  
- Sample genotype columns containing mutation information  

### Sample column formats

Sample columns are automatically detected using these patterns:

- TCGA format: columns starting with `TCGA-` (e.g., `TCGA-AB-2988`)  
- Genotype format: columns ending with `.GT` (e.g., `SAMPLE001.GT`)  

If neither pattern is found, all columns except metadata columns are treated as potential samples.

### Genotype formats

Supported genotype notations:

- Pipe format: `A|G`, `C|T`  
- Slash format: `A/G`, `C/T`  
- Other separators: `A:G`, `A;G`  

No-mutation indicators: `./.`, `.|.`, `0/0`, `0|0`, `.`, `NA`

A genotype is considered mutated when it contains the alternative allele specified in the `ALT` column.

Note: VCF numeric genotypes (e.g., `0/1`, `1/1`) are currently not supported.

### Variant classifications

Mutations are color-coded by variant type. Standard categories include:

- `Missense_Mutation`: Single nucleotide change altering amino acid (green)  
- `Nonsense_Mutation`: Premature stop codon (crimson)  
- `Frame_Shift_Del`: Frameshift deletion (blue)  
- `Frame_Shift_Ins`: Frameshift insertion (purple)  
- `In_Frame_Del`: In-frame deletion (gold)  
- `In_Frame_Ins`: In-frame insertion (coral)  
- `Splice_Site`: Splice site alteration (orange)  
- `Translation_Start_Site`: Start codon mutation (wheat)  
- `Nonstop_Mutation`: Stop codon loss (orchid)  
- `Silent`: Synonymous mutation (light blue)  
- `Multi_Hit`: Multiple mutations in the same gene-sample pair (black, automatically assigned)  

Unrecognized variant types receive automatically generated colors.

### Multi_Hit detection

`Multi_Hit` is not expected in the input data. It is automatically assigned by the visualization when more than one mutation is found for the same gene-sample pair. In such cases, the cell is displayed in black, visually overriding the individual variant classifications.

## Gene and sample selection

### Gene selection

Genes are selected based on mutation frequency (number of samples with at least one mutation). The `top_genes_count` parameter (default: 10) controls how many of the most frequently mutated genes are displayed. Genes with zero mutations are excluded.

### Sample selection

By default, all samples are included. The `max_samples` parameter (default: 180) can limit the number of displayed samples. When the cohort exceeds this limit, only the first N samples after sorting are shown.

## Sorting behavior

The visualization uses waterfall sorting to create informative visual patterns.

### Gene sorting

Genes are sorted by mutation frequency from most to least frequent. Genes with more mutated samples appear at the top of the plot. Genes with equal frequency maintain their original order for stability.

### Sample sorting

Samples are sorted using a lexicographic ordering algorithm based on mutation patterns:

1. A binary matrix is created (0 = no mutation, 1 = mutation present)  
2. Samples are pre-sorted alphabetically for consistent tiebreaking  
3. Lexicographic sorting prioritizes mutations in high-frequency genes  

This creates the characteristic waterfall pattern where samples with mutations in top genes appear on the left, and samples sharing mutation patterns are grouped together.

## Visualization panels

### Main mutation matrix

The central heatmap displays mutation status for each gene-sample pair:

- Cells are color-coded by variant classification  
- Non-mutated cells appear in light gray (#F5F5F5)  
- White grid lines separate cells  
- Gene names are shown on the y-axis  
- X-axis label shows altered samples format: "Altered in X (Y%) of Z samples" (e.g., "Altered in 141 (73.06%) of 193 samples")  
- Individual sample names are hidden to reduce clutter  

### Top panel: Mutation count barplot (TMB)

Shows the total number of mutations for each sample, aligned with matrix columns:

- Stacked bar chart with segments representing variant types  
- **Only variant types present in the displayed mutation matrix are shown** (filters out variant types not in the color mapping)  
- Colors match the main matrix  
- Y-axis labeled as "TMB" with ticks showing 0, midpoint, and maximum values (e.g., 0, 17, 34)  
- Counts mutations only from variant types present in the color mapping  
- Values are raw mutation counts, not normalized by megabases  

### Right panel: Gene alteration barplot

Shows the number of mutated samples per gene, aligned with matrix rows:

- Stacked horizontal bars showing sample counts broken down by variant type  
- Bar lengths represent absolute number of mutated samples (e.g., max value of 52 means the most frequently mutated gene has 52 samples with mutations)  
- X-axis labeled as "Nº of samples" with ticks showing 0, midpoint, and maximum values  
- Percentage labels displayed to the right of each bar (e.g., "27.5%") are computed as `n_mutated_samples / total_cohort_size`  
- Percentages are calculated from the entire cohort, even when `max_samples` limits the displayed columns  
- Multi_Hit counted as a single event per gene-sample pair when calculating both counts and percentages  

### Bottom panel: Legend

Maps variant classifications to colors:

- Only variant types present in the data are shown  
- Non-mutated status (`None`) is excluded  
- Variant names formatted with spaces (e.g., "Missense Mutation")  
- Arranged in up to 6 columns for compact display  
- Displayed with a black border frame for clear visual separation  
- Positioned with adequate spacing below the main matrix for clarity  

## Parameters

- `figsize` (Optional[Tuple[int, int]], default: (16, 10)): Figure size in inches (width, height).
- `title` (str, default: "Oncoplot"): Plot title displayed at the top of the figure.
- `gene_column` (str, default: "Hugo_Symbol"): Column name containing gene symbols.
- `variant_column` (str, default: "Variant_Classification"): Column name containing variant classifications.
- `ref_column` (str, default: "REF"): Column name containing reference alleles.
- `alt_column` (str, default: "ALT"): Column name containing alternative alleles.
- `top_genes_count` (int, default: 10): Number of top mutated genes to display. Only genes with at least one mutation are included.
- `max_samples` (int, default: 180): Maximum number of samples to display. When the cohort exceeds this limit, only the first N samples after sorting are shown.

## Usage

```python
py_mut = PyMutation(data)
fig = py_mut.oncoplot(
    figsize=(20, 12),
    title="TCGA-LAML Cohort",
    gene_column="Hugo_Symbol",
    variant_column="Variant_Classification",
    ref_column="REF",
    alt_column="ALT",
    top_genes_count=20,
    max_samples=100
)
fig.savefig('oncoplot.png', dpi=300)
```