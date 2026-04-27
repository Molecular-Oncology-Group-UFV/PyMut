# Mutational Signature Analysis

This document explains the complete mutational signature analysis visualization and its component panels. It describes what each panel shows, the data it requires, and how to interpret the results.

## Overview

Somatic single-nucleotide variants (SNVs) arise from different mutational processes such as aging, UV exposure, APOBEC activity, and defective DNA repair mechanisms. Each process leaves a characteristic pattern across 96 trinucleotide mutation contexts, defined by the six base substitutions (C>A, C>G, C>T, T>A, T>C, T>G) combined with the 16 possible combinations of upstream and downstream flanking bases.

The complete analysis consists of five complementary panels:

- Panel A: Signature profiles showing the trinucleotide distribution for each mutational signature  
- Panel B: Cosine similarity heatmap comparing extracted signatures with known COSMIC signatures  
- Panel C: Heatmap showing relative signature contributions per sample  
- Panel D: Stacked bar chart displaying signature distribution across samples  
- Panel E: Donut plot showing overall cohort-level signature proportions  

These panels work together to provide a comprehensive view of mutational processes active in your dataset.

## Data Requirements

All visualization functions are methods of the `PyMutation` class and operate on the mutation data already loaded in the instance (`self.data`). The reference genome is only used to derive trinucleotide contexts when needed.

To generate mutational signature visualizations, you need:

1. Variants: SNVs already loaded in your PyMutation instance
      * Original format: MAF or VCF with chromosome, position, reference allele, alternative allele
      * Sample identifier field: Tumor_Sample_Barcode (or equivalent)
      * Only SNVs are considered; indels are automatically filtered out before trinucleotide context extraction

2. Reference genome FASTA: To extract trinucleotide contexts (GRCh37 or GRCh38)
      * Can be gzipped (.fa.gz)
      * Chromosome names must match your variant file
      * Examples: `hs37d5.fa`, `hg38.fa`, `GRCh38.primary_assembly.genome.fa`
      * Used to extract the upstream and downstream bases for each SNV

3. COSMIC Catalog (optional, for signature comparison):
      * TSV format with 96 rows representing trinucleotide contexts
      * First column: context labels following COSMIC v3.x convention
      * Row order must match COSMIC standard to ensure alignment between panels A and B
      * Subsequent columns: COSMIC signatures (e.g., "SBS1", "SBS2")
      * Download from: https://cancer.sanger.ac.uk/signatures/
      * Recommended files: `COSMIC_v3.4_SBS_GRCh37.txt` or `COSMIC_v3.4_SBS_GRCh38.txt`

4. Number of signatures: Specified via the `n_signatures` parameter
      * Must be set manually based on your dataset characteristics
      * Typical range: 2-6 signatures for most datasets
      * Start with n=3 and adjust based on biological interpretation and signature stability

Important - Pyrimidine Convention:
All mutations are normalized to pyrimidine context (C or T as reference base) before analysis. If the mutated base is A or G, the mutation is converted to its reverse complement. This is why there are exactly 96 contexts (6 substitution types × 16 trinucleotide combinations) instead of 192. This convention ensures compatibility with COSMIC signature definitions.

## Panel A: Signature Bar Chart

The `signature_bar_chart()` function creates a multi-panel visualization where each panel shows one mutational signature as a bar chart with 96 bars.

What it shows:
Each signature is displayed as a profile of 96 trinucleotide contexts following the COSMIC v3.x standard order. The bars are visually grouped under the 6 substitution classes (C>A, C>G, C>T, T>A, T>C, T>G), with each group containing 16 bars representing the different upstream and downstream base combinations. The Y-axis shows the percentage contribution of each trinucleotide motif within that signature.

Each signature is normalized so that all 96 bars sum to 100%, representing it as a probability distribution over trinucleotide contexts.

Color scheme (standard COSMIC colors):  
- C>A: cyan (#02bdee)  
- C>G: black (#010101)  
- C>T: red (#e32925)  
- T>A: gray (#cac9c9)  
- T>C: green (#a1cf63)  
- T>G: pink (#ecc7c4)  

The trinucleotide context labels appear at the bottom of the last panel, showing the specific upstream and downstream bases for each bar (e.g., A[C>A]A, A[C>A]C, etc.).

COSMIC alignment:
If `cosmic_path` is provided, signatures are automatically reordered and renamed based on their best COSMIC match (using cosine similarity ≥ 0.5). This helps with immediate biological interpretation, showing names like "SBS1-like (cos=0.95)" instead of generic "Signature 1". Unmatched signatures retain generic names and are sorted to the end.

## Panel B: Cosine Similarity Heatmap

The `cosine_similarity_heatmap()` function creates a heatmap comparing your extracted signatures with the COSMIC catalog of known mutational signatures.

What it shows:
A matrix where each cell represents the similarity between one of your extracted signatures (Y-axis) and one COSMIC reference signature (X-axis). The cosine similarity metric ranges from 0 (completely different) to 1 (identical). Darker blue cells indicate higher similarity.

Both signature sets are L2-normalized before computing cosine similarity to ensure fair comparison. The calculation follows: cosine(u, v) = (u · v) / (||u|| × ||v||).

Interpretation:
High similarity values (typically ≥ 0.85) suggest that your extracted signature closely matches a known biological process. For example, if Signature 1 shows high similarity to SBS1 and SBS5, it likely represents the aging-related mutational process. This panel helps you identify which biological mechanisms are active in your dataset.

## Panel C: Signature Contribution Heatmap

The `signature_contribution_heatmap()` function shows how much each signature contributes to each sample as a heatmap.

What it shows:
A matrix where signatures are on the Y-axis and samples are on the X-axis. Each cell shows the proportion of mutations in that sample attributable to that signature. Values range from 0 to 1, with darker blue indicating higher contribution.

Normalization:
Column-normalized H (each sample sums to 1): H_normalized = H / H.sum(axis=0), where each sample column sums to 1 (100%).

Interpretation:
This panel reveals sample-level heterogeneity in mutational processes. For instance, you might observe that some samples are dominated by a single signature while others show contributions from multiple signatures. This can help identify subgroups within your cohort based on their active mutational mechanisms.

## Panel D: Stacked Bar Chart per Sample

The `signature_stacked_bar_chart()` function displays signature contributions across samples as stacked bars.

What it shows:
Each vertical bar represents one sample, with colored segments showing the relative proportion of each signature. Every bar has a total height of 1 (100%), making it easy to compare the relative composition of mutational signatures across samples.

Normalization:
Column-normalized H (each sample sums to 1): uses the same normalization as Panel C (H_normalized = H / H.sum(axis=0)), ensuring consistency between panels.

Interpretation:
This panel provides an intuitive view of signature heterogeneity across your cohort. Samples can be optionally sorted by their dominant signature to reveal patterns and clusters. For example, you might see groups of samples with similar signature compositions, suggesting shared etiologies or exposures.

## Panel E: Donut Plot - Cohort-Level Proportions

The `signature_donut_plot()` function shows the overall distribution of mutational signatures across the entire cohort.

What it shows:
A donut chart where each slice represents the proportion of all mutations in the dataset attributable to each signature. Percentages are displayed on the chart showing each signature's contribution to the total mutational burden.

This panel uses absolute contributions summed across all samples: cohort_exposure = contributions_abs.sum(axis=1), then normalized to percentages. This is the ONLY panel that uses absolute counts rather than per-sample normalization.

Interpretation:
This panel answers the question: "Which mutational processes dominate my entire dataset?" For example, if Signature 1 accounts for 47% of all mutations and shows high similarity to SBS1/SBS5, this indicates that aging is the predominant mutational process in your cohort. This cohort-level view complements the per-sample views in Panels C and D.

Important note:
Unlike Panels C and D which show relative proportions per sample, this panel reflects both the activity and prevalence of each signature across the entire cohort. This means samples with more mutations contribute more to the overall percentages (activity × prevalence).

## Complete Mutational Signature Analysis

The `mutational_signature_analysis()` function generates a comprehensive figure containing all five panels described above in a single publication-ready visualization.

Layout:  
- Top section (larger): Panel A showing all signature profiles  
- Bottom section, left to right:  
      - Panel B: Cosine similarity heatmap  
      - Panel C: Sample contribution heatmap  
      - Panel D: Stacked bar chart  
      - Panel E: Donut plot  

This combined visualization provides a complete view of mutational processes in your dataset, from the detailed trinucleotide patterns of each signature to their distribution across samples and overall cohort-level proportions.

## Function Parameters

All functions below are methods of the `PyMutation` class. They automatically use the mutation data loaded in the instance. 

**Important Note on Data Processing:**
Each visualization function internally recomputes the trinucleotide context matrix from `self.data` and the reference genome. The parameters `ref_genome`, `prefix`, `add`, `ignoreChr`, and `useSyn` control this preprocessing step. This means the context matrix is generated fresh each time a visualization is called, allowing you to experiment with different filtering strategies without modifying your original data.

### Parameter Categories

Parameters fall into three categories:

1. **Data Preprocessing** (`ref_genome`, `prefix`, `add`, `ignoreChr`, `useSyn`): Control how variant data is converted to trinucleotide contexts  
2. **Analysis** (`n_signatures`, `cosmic_path`, `signature_names`): Control signature extraction and annotation  
3. **Visualization** (`figsize`, `title`, `cmap`, `show_values`, etc.): Control plot appearance  

---

### `mutational_signature_analysis()`

- `ref_genome` (str): Path to reference genome FASTA file. Required to extract trinucleotide contexts from variant positions in `self.data`.  
- `cosmic_path` (str): Path to COSMIC signature catalog TSV file. Required for Panel B comparison and signature annotation.  
- `n_signatures` (int, default=3): Number of mutational signatures to extract from the data.  
- `prefix` (str, optional): Prefix to add or remove from chromosome names to match the reference genome. Use when chromosome naming differs between your variants and FASTA (e.g., 'chr').  
- `add` (bool, default=True): If True, adds the prefix to chromosome names; if False, removes it.  
- `ignoreChr` (list, optional): List of chromosomes to exclude from analysis (e.g., ['chrM', 'chrY']).  
- `useSyn` (bool, default=True): Include synonymous variants in the analysis. Set to False to exclude them.  
- `signature_names` (list, optional): Custom names for the extracted signatures. If not provided, uses generic names like "Signature 1", "Signature 2", etc.  
- `selected_signatures` (list, optional): Subset of signature indices to display (0-based indexing). Use this to focus on specific signatures of interest.  
- `figsize` (tuple, default=(24, 12)): Figure size as (width, height) in inches.  
- `title` (str, default="Mutational Signature Analysis"): Main title displayed at the top of the figure.  

### `signature_bar_chart()`

- `ref_genome` (str): Path to reference genome FASTA file.  
- `n_signatures` (int, default=3): Number of signatures to extract.  
- `cosmic_path` (str, optional): Path to COSMIC catalog. If provided, signatures are reordered and renamed based on their best COSMIC match.  
- `figsize` (tuple, optional): Figure size. If not specified, automatically calculated as (12, 3 × n_signatures).  
- `title` (str, default="Mutational Signature Profiles"): Figure title.  

### `cosine_similarity_heatmap()`

- `ref_genome` (str): Path to reference genome FASTA file.  
- `cosmic_path` (str): Path to COSMIC signature catalog (required).  
- `n_signatures` (int, default=3): Number of signatures to extract.  
- `prefix` (str, optional): Chromosome name prefix adjustment.  
- `add` (bool, default=True): Whether to add or remove the prefix.  
- `ignoreChr` (list, optional): Chromosomes to exclude.  
- `useSyn` (bool, default=True): Include synonymous variants.  
- `signature_names` (list, optional): Custom signature names.  
- `figsize` (tuple, default=(14, 3)): Figure size.  
- `title` (str, default="Cosine Similarity: Extracted vs COSMIC Signatures"): Plot title.  

### `signature_contribution_heatmap()`

- `ref_genome` (str): Path to reference genome FASTA file.  
- `n_signatures` (int, default=3): Number of signatures to extract.  
- `prefix` (str, optional): Chromosome name prefix adjustment.  
- `add` (bool, default=True): Whether to add or remove the prefix.  
- `ignoreChr` (list, optional): Chromosomes to exclude.  
- `useSyn` (bool, default=True): Include synonymous variants.  
- `signature_names` (list, optional): Custom signature names.  
- `cosmic_path` (str, optional): Path to COSMIC catalog for signature alignment.  
- `figsize` (tuple, optional): Figure size. Auto-calculated if not specified.  
- `cmap` (str, default='Blues'): Colormap name for the heatmap.  
- `show_values` (bool, default=False): Display numerical values in heatmap cells.  
- `show_labels` (bool, default=True): Show sample and signature labels on axes.  
- `title` (str, default="Signature Contributions per Sample"): Plot title.  

### `signature_stacked_bar_chart()`

- `ref_genome` (str): Path to reference genome FASTA file.  
- `n_signatures` (int, default=3): Number of signatures to extract.  
- `prefix` (str, optional): Chromosome name prefix adjustment.  
- `add` (bool, default=True): Whether to add or remove the prefix.  
- `ignoreChr` (list, optional): Chromosomes to exclude.  
- `useSyn` (bool, default=True): Include synonymous variants.  
- `signature_names` (list, optional): Custom signature names.  
- `figsize` (tuple, optional): Figure size. Auto-calculated if not specified.  
- `title` (str, default="Signature Contributions per Sample"): Plot title.  
- `sort_samples` (bool, default=False): Sort samples by their dominant signature for better pattern visualization.  
- `max_samples` (int, optional): Maximum number of samples to display. Shows only the first N samples if specified.  
- `show_labels` (bool, default=True): Show sample labels on X-axis and legend.  

### `signature_donut_plot()`

- `ref_genome` (str): Path to reference genome FASTA file.  
- `n_signatures` (int, default=3): Number of signatures to extract.  
- `prefix` (str, optional): Chromosome name prefix adjustment.  
- `add` (bool, default=True): Whether to add or remove the prefix.  
- `ignoreChr` (list, optional): Chromosomes to exclude.  
- `useSyn` (bool, default=True): Include synonymous variants.  
- `signature_names` (list, optional): Custom signature names.  
- `figsize` (tuple, default=(8, 8)): Figure size.  
- `title` (str, default="Relative Contribution of Mutational Signatures"): Plot title.  
