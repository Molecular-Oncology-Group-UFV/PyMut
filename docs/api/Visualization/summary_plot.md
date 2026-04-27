# MAF Summary Dashboard

The `summary_plot()` function provides a compact, high‑level overview of all somatic variants in a cohort. It is designed to answer key questions about the mutational landscape:

- *Which mutation types are most common in this cohort?*  
- *How many variants does each sample have?*  
- *Which genes are mutated most often?*  

The visualization consists of six coordinated panels, all computed from the same input data and sharing a consistent color palette for variant classifications (e.g., Missense_Mutation, Nonsense_Mutation, Splice_Site), ensuring visual coherence across the dashboard.

## Parameters

The `summary_plot()` function accepts the following parameters:  

- `figsize`: Figure size as (width, height) in inches. Default is `(16, 12)`.  
- `title`: Main title for the summary plot. Default is `"Mutation Summary"`.  
- `max_samples`: Maximum number of samples to display in the "Variants per sample" panel. If None, all samples are shown. Default is `200`.  
- `top_genes_count`: Number of top genes to display in the "Top mutated genes" panel. Default is `10`.  

## Input data formats

The function automatically detects and supports two input formats:

Long format (MAF-like): Each row represents one somatic variant. This is the standard MAF structure where the same sample or gene can appear in multiple rows. Sample identifiers are stored in the `Tumor_Sample_Barcode` column.

Wide format (matrix-like): Each row represents one variant with samples as columns (e.g., TCGA-XX-XXXX). Cell values contain genotype information in formats like `A|G` (heterozygous variant), `A|A` or `0|0` (homozygous reference, no variant). The function detects this format when `Tumor_Sample_Barcode` is absent.

The format is detected automatically. Panels 1–3 (classification, type, SNV class) work identically in both formats by counting variant rows. Panels 4–6 (per-sample and per-gene views) adapt their behavior based on the detected format.

## Required data columns

- `Hugo_Symbol`: gene symbol  
- `Variant_Classification`: functional effect of the variant on the coding sequence  
- `Variant_Type`: structural type of variant (SNP, INS, DEL, etc.)  
- `REF`: reference allele  
- `ALT`: alternative allele (used to classify SNVs into substitution classes)  
- `Tumor_Sample_Barcode`: sample identifier (required only in long format)  
- Sample columns: TCGA-XX-XXXX or similar (required only in wide format)  


---

## Panel 1: Variant Classification (top‑left) - `variant_classification_plot()`

This panel displays a horizontal barplot showing the distribution of variant classifications across the entire cohort. Each bar represents a different classification type, with the bar length indicating the total number of mutations of that type.

Typical variant classifications include:

- `Missense_Mutation`: single amino‑acid change in the protein  
- `Nonsense_Mutation`: introduces a premature stop codon, truncating the protein  
- `Splice_Site`: variant near exon–intron boundaries that may disrupt splicing  
- `Frame_Shift_Ins` / `Frame_Shift_Del`: insertions or deletions that change the reading frame  
- `In_Frame_Ins` / `In_Frame_Del`: insertions or deletions that preserve the reading frame  

Bars are sorted by frequency (high to low), with the most common classifications appearing at the top. Each bar is annotated with its exact count.

**Note:** By default, silent (synonymous) mutations and non-coding variants are excluded. Set `include_silent=True` to include all variants.

**Parameters (standalone function only):**

- `figsize`: Figure size. Default: `(10, 6)`
- `title`: Plot title. Default: `"Variant Classification"`
- `include_silent`: Whether to include silent/synonymous and non-coding variants. Default: `False`

Required data: `Variant_Classification` column, one row per somatic variant.

---

## Panel 2: Variant Type (top‑middle) - `variant_type_plot()`

This panel shows a horizontal barplot summarizing variant types based on their structural characteristics:

- `SNP`: single nucleotide polymorphism (change of one base)  
- `INS`: small insertion  
- `DEL`: small deletion  
- `TNP` / `ONP`: tri‑ and oligo‑nucleotide changes (when present in the dataset)  

Each bar represents the total number of variants of that type across all samples. Bars are sorted by frequency (low to high) and annotated with exact counts.

This view focuses on the structural nature of the DNA change, independent of its biological effect on proteins.

**Note:** By default, silent (synonymous) mutations and non-coding variants are excluded. Set `include_silent=True` to include all variants.

**Parameters (standalone function only):**

- `figsize`: Figure size. Default: `(10, 6)`
- `title`: Plot title. Default: `"Variant Type"`
- `include_silent`: Whether to include silent/synonymous and non-coding variants. Default: `False`

Required data: `Variant_Type` column, one row per variant.

---

## Panel 3: SNV Class (top‑right) - `snv_class_plot()`

This panel displays a horizontal barplot for single‑nucleotide variant (SNV) substitution classes. Each bar corresponds to one base substitution pattern (e.g., C>T, G>A, A>G), showing the total number of SNVs of each class across all samples in the cohort.

**Important:** By default, this visualization normalizes all SNVs to pyrimidine context (C or T as reference). This means complementary substitutions are merged into a single class (e.g., both A>C and T>G are counted as A>C when normalized). This follows the convention used in mutational signature analysis and matches maftools behavior, showing 6 canonical substitution classes instead of 12.

To see all 12 substitution types without normalization, use the `normalize_pyrimidine=False` parameter in the standalone function.

**Note:** Unlike other panels, this includes silent mutations by default to show the complete SNV spectrum, which is important for mutational signature analysis.

Bars are sorted by frequency (low to high) and annotated with exact counts.

**Parameters (standalone function only):**

- `figsize`: Figure size. Default: `(10, 6)`
- `title`: Plot title. Default: `"SNV Class"`
- `ref_column`: Column containing reference alleles. Default: `"REF"`
- `alt_column`: Column containing alternative alleles. Default: `"ALT"`
- `normalize_pyrimidine`: Whether to normalize to pyrimidine bases (C/T as reference). Default: `True` (shows 6 canonical classes). Set to `False` to show all 12 possible substitution types.

Required data: `REF` and `ALT` columns. Only rows where both are single nucleotides are included.

---

## Panel 4: Variants per sample (bottom‑left) - `variants_per_sample_plot()`

This panel shows a stacked barplot where each bar represents a sample, sorted by decreasing total variant count. The height of each bar indicates the total number of variants in that sample, and the bar is stacked by variant classification using the same color scheme as Panel 1.

This visualization reveals the tumor mutational burden (TMB) per sample and how that burden is composed (e.g., predominantly missense mutations or many frameshifts). A red dashed horizontal line marks the median number of variants per sample across the cohort.

If the `max_samples` parameter is specified, only the top N samples (by variant count) are displayed.

**Note:** By default, silent (synonymous) mutations and non-coding variants are excluded. Set `include_silent=True` to include all variants.

**Parameters (standalone function only):**

- `figsize`: Figure size. Default: `(12, 6)`
- `title`: Plot title. Default: `"Variants per Sample"`
- `variant_column`: Column containing variant classifications. Default: `"Variant_Classification"`
- `sample_column`: Column containing sample identifiers. Default: `"Tumor_Sample_Barcode"`
- `max_samples`: Maximum number of samples to display. Default: `200`
- `include_silent`: Whether to include silent/synonymous and non-coding variants. Default: `False`

Required data: `Tumor_Sample_Barcode` (in long format) or sample columns (in wide format, e.g., TCGA-XX-XXXX), and `Variant_Classification`.

---

## Panel 5: Variant Classification summary (bottom‑middle) - `variant_classification_summary_plot()`

This panel contains boxplots, one for each variant classification type. Each boxplot summarizes the distribution of per‑sample variant counts for that classification across the entire cohort.

For a given classification (e.g., *Missense_Mutation*), the boxplot shows:

- The median (red line) and interquartile range of variant counts per sample  
- Whiskers extending to the most extreme non‑outlier values  
- Outliers displayed as small gray dots  

This answers questions like "In a typical sample, how many missense mutations are present?" and "Are splice‑site mutations rare in this cohort?"

Boxplots are ordered by total variant count (most common classifications appear first from left to right) and colored using the same palette as other panels.

**Note:** By default, silent (synonymous) mutations and non-coding variants are excluded. Set `include_silent=True` to include all variants.

**Parameters (standalone function only):**

- `figsize`: Figure size. Default: `(10, 6)`
- `title`: Plot title. Default: `"Variant Classification Summary"`
- `variant_column`: Column containing variant classifications. Default: `"Variant_Classification"`
- `sample_column`: Column containing sample identifiers. Default: `"Tumor_Sample_Barcode"`
- `include_silent`: Whether to include silent/synonymous and non-coding variants. Default: `False`

Required data: `Tumor_Sample_Barcode` (in long format) or sample columns (in wide format), and `Variant_Classification`.

---

## Panel 6: Top mutated genes (bottom‑right) - `top_mutated_genes_plot()`

This panel displays a horizontal stacked barplot showing the most frequently mutated genes in the cohort. By default, the top 10 genes are shown (controlled by the `top_genes_count` parameter).

For each gene:

- The y‑axis lists gene symbols (e.g., *FLT3*, *TET2*, *NRAS*), ordered from bottom to top by increasing mutation count  
- The x‑axis shows the count metric (total variants or affected samples, depending on mode)  
- Each bar is stacked by variant classification, using the same color palette as other panels  
- The total count is annotated at the right end of each bar  

Two counting modes are available:

1. `mode='variants'` (default): Counts the total number of variants in each gene across all samples. The annotation shows the total variant count.

2. `mode='samples'`: Counts the number of distinct samples with at least one variant in each gene. The annotation shows the percentage of affected samples (e.g., "25.0%").

**Important note:** In the `summary_plot()` dashboard, this panel always uses `mode='variants'`. The mode parameter is only available when calling the standalone `top_mutated_genes_plot()` function.

**Note:** By default, this panel includes silent (synonymous) mutations to match maftools behavior. Set `include_silent=False` to exclude them.

This panel highlights recurrently mutated genes and reveals which types of mutations they tend to carry.

**Parameters (standalone function only):**

- `figsize`: Figure size. Default: `(10, 6)`
- `title`: Plot title. Default: `"Top Mutated Genes"`
- `mode`: Counting mode. Options: `"variants"` (total variants) or `"samples"` (affected samples). Default: `"variants"`
- `variant_column`: Column containing variant classifications. Default: `"Variant_Classification"`
- `gene_column`: Column containing gene symbols. Default: `"Hugo_Symbol"`
- `sample_column`: Column containing sample identifiers. Default: `"Tumor_Sample_Barcode"`
- `count`: Number of top genes to display. Default: `10`
- `include_silent`: Whether to include silent/synonymous and non-coding variants. Default: `True`

Required data: `Hugo_Symbol`, `Tumor_Sample_Barcode` (in long format) or sample columns (in wide format), and `Variant_Classification`.

---

## Summary

The `summary_plot()` dashboard combines sample‑level, gene‑level, and cohort‑level views into a single comprehensive figure:

- Panels 1–3 describe what kinds of variants exist in the cohort (functional consequence, structural type, and base‑change patterns)  
  - **Panels 1–2** exclude silent mutations by default to focus on protein-altering variants
  - **Panel 3** includes silent mutations by default (important for mutational signature analysis) and normalizes SNVs to pyrimidine context
- Panel 4 shows how many variants each sample carries and how those variants are distributed across classifications (excludes silent mutations by default)
- Panel 5 summarizes per‑sample distributions for each classification type (excludes silent mutations by default)
- Panel 6 highlights which genes are recurrently mutated and what types of mutations they harbor (includes silent mutations by default to match maftools behavior)

Together, these six views provide a rapid assessment of the input data and a compact overview of the mutational landscape.