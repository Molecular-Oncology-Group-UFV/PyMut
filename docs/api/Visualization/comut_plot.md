# CoMut Plot

## Overview

The CoMut plot is a sample-centric multi-panel visualization that integrates multiple genomic and clinical features into a single coordinated view. It displays samples as columns and different types of information as rows (panels), allowing comprehensive analysis of mutation patterns, signatures, and clinical annotations across a cohort.

Note: Throughout this documentation, `PyMutation` refers to the main class in the pyMut library that wraps mutation data (MAF/VCF) and metadata. The library is installed as `pymut-bio` but imported as `pyMut`.

Key characteristics:  
- Columns represent individual samples (tumors from different patients or multiple biopsies from the same patient)  
- Rows represent different data tracks (mutation burden, signatures, purity, gene mutations, copy number alterations, etc.)  
- All panels share the same x-axis (sample order), enabling vertical alignment and cross-panel comparison  
- Each panel can be generated independently or combined in a multi-panel figure  

---

## Sample Ordering and Alignment

Critical for multi-panel coordination: all CoMut panels must share the same sample order on the x-axis for proper visual alignment. The sample order is established by Panel A (Mutation Burden), which orders samples by non-synonymous tumor mutation burden (TMB) in descending order, then by sample ID in ascending order.

The function `save_sample_order()` can be used to compute and save the sample order to a text file, which can then be loaded with `load_sample_order()` and passed to all subsequent panels via the `sample_order` parameter.

### Data Format Conventions

Most panels accept data as TSV files or pandas DataFrames. The expected format varies by panel:

Panels D, E, F, and G use tidy/long format with three columns:  
- `sample`: sample identifier (must be consistent across all panels)  
- `category`: gene name or feature name (for panels D and E)  
- `value`: measurement or label (mutation type, CNA state, Yes/No status, patient ID)  

Panels B and C use wide format:  
- Panel B (signatures): `sample` + one column per signature  
- Panel C (purity): `sample` + `value`  

Column names are case-insensitive and flexible. For example, `sample_id`, `Tumor_Sample_Barcode`, and `sample` are all recognized as the sample identifier column.

---

## Panel A: Mutation Burden

### Description

`comut_mutation_burden()` displays a stacked bar chart of mutation burden per sample, quantifying the number of mutations per megabase (Muts/Mb) of genome examined. This metric is commonly referred to as Tumor Mutation Burden (TMB).

The visualization shows:  
- Dark blue bars: non-synonymous mutations (mutations that alter protein sequence)  
- Light blue bars: synonymous mutations (silent mutations that do not change protein sequence)  
- Stacked bars: total height represents total mutation rate  
- Sample ordering: by non-synonymous TMB (descending), then by sample ID (ascending)  

High mutation burden can indicate hypermutated tumors, which may respond differently to certain therapies (e.g., immunotherapy).

### Parameters

- `sample_column` (str): Column name containing sample identifiers. Default: `"Tumor_Sample_Barcode"`  
- `variant_column` (str): Column name containing variant classifications. Default: `"Variant_Classification"`  
- `territory_bp` (int): Territory size in base pairs for normalization. Default: `60456963` (whole exome sequencing territory)  
- `figsize` (tuple): Figure size as (width, height) in inches. Default: `(14, 1.8)`  
- `title` (str or None): Plot title. Default: `None` (uses "Mutation Burden")  
- `hypermutator_threshold` (float or None): Optional TMB threshold to draw a horizontal reference line. Default: `None`  
- `sample_ids` (list or None): Optional list of specific sample IDs to include in their exact specified order. If provided, samples are filtered to those present in the data but their order is preserved exactly as given (no re-sorting by TMB). This overrides both automatic TMB-based ordering and `max_samples`. Default: `None`  
- `max_samples` (int or None): Maximum number of samples to display when `sample_ids` is not provided. Samples are automatically selected by highest non-synonymous TMB and ordered accordingly. Default: `50`. Set to `None` to show all samples.  
- `somatic_only` (bool): If True, filter to variants with Variant_Status = 'Somatic'. Default: `True`  
- `pass_only` (bool): Filter to variants with FILTER in ['PASS', '.']. Default: `True`  

### Data Requirements

This panel uses mutation data directly from the PyMutation object's MAF data. No external data files are required.

Variant classification categories:
- Non-synonymous: `Missense_Mutation`, `Nonsense_Mutation`, `Frame_Shift_Del`,   `Frame_Shift_Ins`, `In_Frame_Del`, `In_Frame_Ins`, `Splice_Site`, `Nonstop_Mutation`, `Translation_Start_Site`
- Synonymous: `Silent`, `Synonymous_SNV`  
- Excluded from TMB: `3'UTR`, `5'UTR`, `IGR`, `Intron`, `RNA`, `Splice_Region`  

Duplicates (same genomic position in the same sample) are automatically removed.

---

## Panel B: Mutational Signatures

### Description

`comut_mutation_signatures_plot()` displays a stacked bar chart of mutational signatures per sample. Each bar represents one sample, with colored segments showing the relative contribution of different mutational signatures.

Mutational signatures represent distinct biological processes that cause mutations, such as aging, APOBEC activity, or defective DNA repair mechanisms. Each bar has a height of 1 (100%), with colored segments indicating each signature's proportional contribution.

### Parameters

- `signatures_df` (DataFrame or None): DataFrame with samples as index and signatures as columns. Values should be proportions (0-1). Default: `None`  
- `signatures_tsv` (str or None): Path to TSV file containing signature data. Alternative to `signatures_df`. Default: `None`  
- `sample_order` (list or None): List of sample IDs for alignment with other panels. Default: `None`  
- `signature_labels` (list or None): List of signature names in display order. Default: `None`  
- `normalize` (bool): If True, normalize rows to sum to 1. Default: `True`  
- `figsize` (tuple): Figure size as (width, height) in inches. Default: `(12, 2.0)`  
- `colors` (list or None): List of colors for signatures. If None, uses default palette. Default: `None`  
- `title` (str or None): Plot title. Default: `None` (uses "Mutational Signatures")  

### Data Requirements

Data can be provided as a pandas DataFrame or TSV file:

DataFrame format:  
- Index: sample IDs  
- Columns: signature names (e.g., "Signature 1", "Signature 2")  
- Values: proportions (0-1) representing each signature's contribution  

TSV file format:  
- Columns: `sample` (sample ID), followed by signature columns (e.g., `Signature 1`, `Signature 2`)  
- Each row: one sample  
- Values: proportions that should sum to approximately 1.0 per row  

Example TSV:
```
sample       Signature 1  Signature 2  Signature 3
SAMPLE1      0.30         0.50         0.20
SAMPLE2      0.10         0.80         0.10
```

---

## Panel C: Tumor Purity

### Description

`comut_purity_plot()` displays a 1-row heatmap showing tumor purity per sample. Color intensity represents the proportion of tumor cells versus normal cells in each sample.

Color scale:  
- Light blue: low purity (higher proportion of normal cells)  
- Dark blue: high purity (mostly tumor cells)  
- White: missing data  

High purity indicates more reliable genomic data, as the signal is less diluted by normal cell contamination.

### Parameters

- `purity_df` (DataFrame, Series, or None): DataFrame or Series containing purity values. Default: `None`  
- `purity_tsv` (str or None): Path to TSV file containing purity data. Alternative to `purity_df`. Default: `None`  
- `sample_order` (list or None): List of sample IDs for alignment with other panels. Default: `None`  
- `figsize` (tuple): Figure size as (width, height) in inches. Default: `(12, 1.5)`  
- `title` (str or None): Plot title. Default: `None` (uses "Purity" as y-axis label when generated independently; "C  Purity" when part of multi-panel plot)  

### Data Requirements

Data can be provided as a pandas Series, DataFrame, or TSV file:

Series format:  
- Index: sample IDs  
- Values: purity values (0-1 scale)  

TSV file format:  
- Columns: `sample`, `value` (or `purity`, `tumor_purity`)  
- Values: 0-1 scale (or 0-100, which is automatically converted)  

Example TSV:
```
sample    value
SAMPLE1   0.80
SAMPLE2   0.65
```

---

## Panel D: Mutation Type (Oncoprint)

### Description

`comut_mutation_type_plot()` displays an oncoprint-style matrix showing mutation types across key genes and samples. Each cell represents a gene-sample pair, with colors indicating the type of mutation present.

The visualization shows:  
- Rows: genes  
- Columns: samples  
- Cell colors: mutation types  
- Split cells (diagonal): when a gene has 2 different mutation types in the same sample  
- White cells: no mutation detected  

Mutation type categories (5 types):  
- `Missense`: missense mutations (dark blue)  
- `Nonsense`: nonsense mutations (dark red)  
- `Frameshift indel`: frameshift insertions or deletions (pink)  
- `Silent`: synonymous mutations (light green)  
- `Multiple`: more than 2 different mutation types in the same gene-sample (light blue)  

Note: This panel expects mutation types to already be categorized into these five groups in the input data. If starting from raw MAF data with detailed `Variant_Classification` values, you will need to pre-process and map them (e.g., `Frame_Shift_Del` and `Frame_Shift_Ins` → `Frameshift indel`, `Missense_Mutation` → `Missense`, etc.) before creating the TSV file.

### Parameters

- `mutation_data_df` (DataFrame or None): DataFrame containing mutation type data. Default: `None`  
- `mutation_data_tsv` (str or None): Path to TSV file containing mutation data. Alternative to `mutation_data_df`. Default: `None`  
- `sample_order` (list or None): List of sample IDs for alignment with other panels. Default: `None`  
- `gene_order` (list or None): List of gene names specifying display order (top to bottom). Default: `None`  
- `figsize` (tuple): Figure size as (width, height) in inches. Adjust height based on number of genes. Default: `(12, 4.0)`  
- `title` (str or None): Plot title. Default: `None` (uses "D  Mutation Type")  

### Data Requirements

Data provided as TSV file or DataFrame:

TSV file format (tidy format):  
- Columns: `sample`, `category` (gene name), `value` (mutation type)  
- One row per mutation event  
- Multiple rows allowed for same gene-sample pair (indicates multiple mutation types)  

Example TSV:
```
sample    category  value
SAMPLE1   TP53      Missense
SAMPLE1   PIK3CA    Frameshift indel
SAMPLE2   TP53      Nonsense
SAMPLE2   TP53      Missense
```

Duplicate handling:
- 1 mutation type: solid color cell
- 2 mutation types: split cell with diagonal triangles
- 3+ mutation types: "Multiple" category (solid light blue)

---

## Panel E: Copy Number Alteration

### Description

`comut_cna_plot()` displays a gene × sample matrix showing allelic copy number alteration (CNA) states. Each cell represents the CNA state for one gene in one sample, highlighting amplified oncogenes and deleted tumor suppressors.

Important: This panel does not perform CNA calling or segmentation. It expects per-gene CNA states that have been pre-computed by an upstream pipeline (e.g., GISTIC2, ASCAT, Sequenza, or similar tools) and already categorized into the four discrete states below.

CNA state categories (4 states):  
- `Baseline`: normal/diploid copy number (light gray)  
- `Allelic amplification`: copy number gain (light blue)  
- `Allelic deletion`: heterozygous loss (dark blue)  
- `aCN = 0`: homozygous deletion/complete loss (dark gray)  

Split cells (diagonal triangles) indicate multiple CNA events in the same gene-sample pair.

### Parameters

- `cna_df` (DataFrame or None): DataFrame containing CNA data. Default: `None`  
- `cna_tsv` (str or None): Path to TSV file containing CNA data. Alternative to `cna_df`. Default: `None`  
- `sample_order` (list or None): List of sample IDs for alignment with other panels. Default: `None`  
- `gene_order` (list or None): List of gene names specifying display order (top to bottom). Default: `None`  
- `figsize` (tuple): Figure size as (width, height) in inches. Adjust height based on number of genes. Default: `(12, 2.0)`  
- `title` (str or None): Plot title. Default: `None` (uses "E  Copy Number Alteration")  

### Data Requirements

Data provided as TSV file or DataFrame:

TSV file format (tidy format):  
- Columns: `sample`, `category` (gene name), `value` (CNA state)  
- One row per CNA event  

Example TSV:
```
sample    category  value
SAMPLE1   ERBB2     Allelic amplification
SAMPLE1   CDKN2A    Allelic deletion
SAMPLE2   ERBB2     Baseline
SAMPLE2   MYC       aCN = 0
```

Duplicate handling:  
- Multiple CNA states in same gene-sample pair: split cell with diagonal triangles  
- Priority order for split cells: `aCN = 0` > `Allelic deletion` > `Allelic amplification` > `Baseline`  

---

## Panel F: Whole Genome Doubling

### Description

`comut_wgd_plot()` displays whether each sample has undergone whole genome doubling (WGD), which is the duplication of the entire chromosome set. WGD is common in cancer and associated with genomic instability.

Visual encoding:  
- Gray rectangle: WGD present (Yes)  
- White/blank: no WGD (No) or missing data  

### Parameters

- `wgd_df` (DataFrame or None): DataFrame containing WGD status. Default: `None`  
- `wgd_tsv` (str or None): Path to TSV file containing WGD data. Alternative to `wgd_df`. Default: `None`  
- `sample_order` (list or None): List of sample IDs for alignment with other panels. Default: `None`  
- `figsize` (tuple): Figure size as (width, height) in inches. Default: `(12, 1.2)`  
- `title` (str or None): Plot title. Default: `None` (uses "F  Whole Genome Doubling")  

### Data Requirements

Data provided as TSV file or DataFrame:

TSV file format:  
- Columns: `sample`, `value` (or `wgd`, `status`)  
- Values: flexible encoding for Yes/No status  

Accepted values (case-insensitive):  
- Yes: `Yes`, `Y`, `True`, `T`, `1`  
- No: `No`, `N`, `False`, `F`, `0`  

Example TSV:
```
sample    value
SAMPLE1   Yes
SAMPLE2   No
SAMPLE3   Yes
```

---

## Panel G: Same Patient

### Description

`comut_same_patient_plot()` indicates which samples belong to the same patient, useful for identifying paired samples such as primary and relapse tumors, or multiple biopsies from the same individual.

Visual encoding:  
- Black dot: each individual sample  
- Horizontal line: connects samples from the same patient (≥2 samples)  
- Isolated dots: samples from unique patients with no paired samples  

### Parameters

- `sp_df` (DataFrame or None): DataFrame containing patient grouping information. Default: `None`  
- `sp_tsv` (str or None): Path to TSV file containing patient grouping data. Alternative to `sp_df`. Default: `None`  
- `sample_order` (list or None): List of sample IDs for alignment with other panels. Default: `None`  
- `figsize` (tuple): Figure size as (width, height) in inches. Default: `(12, 1.2)`  
- `title` (str or None): Plot title. Default: `None` (uses "G  Same Patient")  

### Data Requirements

Data provided as TSV file or DataFrame:

TSV file format:  
- Columns: `sample`, `group` (patient ID or grouping identifier)  
- All samples with the same `group` value are connected by a line  

Example TSV:
```
sample      group
SAMPLE1_P   PATIENT_A
SAMPLE1_R   PATIENT_A
SAMPLE2_P   PATIENT_B
SAMPLE2_R   PATIENT_B
SAMPLE3     PATIENT_C
```

In this example:  
- `SAMPLE1_P` and `SAMPLE1_R` are connected (same patient A)  
- `SAMPLE2_P` and `SAMPLE2_R` are connected (same patient B)  
- `SAMPLE3` shows only a dot (no paired sample from patient C)  

---

## Complete Multi-Panel Plot

### Description

`create_comut_plot()` generates a complete multi-panel CoMut visualization by combining all available panels (A-G) into a single vertically stacked figure with perfect sample alignment.

### Parameters

- `sample_order` (list or None): List of sample IDs to use for all panels. If None, computed from Panel A. Default: `None`  
- `gene_order` (list or None): List of gene names for Panel D (mutation type). Default: `None`  
- `cna_gene_order` (list or None): List of gene names for Panel E (CNA). Default: `None`  
- `signatures_tsv` (str or None): Path to TSV file for Panel B (signatures). Default: `None`  
- `purity_tsv` (str or None): Path to TSV file for Panel C (purity). Default: `None`  
- `mutation_data_tsv` (str or None): Path to TSV file for Panel D (mutation types). Default: `None`  
- `cna_tsv` (str or None): Path to TSV file for Panel E (CNA). Default: `None`  
- `wgd_tsv` (str or None): Path to TSV file for Panel F (WGD). Default: `None`  
- `sp_tsv` (str or None): Path to TSV file for Panel G (same patient). Default: `None`  
- `territory_bp` (int): Territory size in bp for Panel A TMB calculation. Default: `60456963`  
- `max_samples` (int or None): Maximum number of samples to display. Default: `50`  
- `somatic_only` (bool): Filter to somatic variants for Panel A. Default: `True`  
- `pass_only` (bool): Filter to PASS variants for Panel A. Default: `True`  
- `signature_labels` (list or None): Signature names for Panel B. Default: `None`  
- `figsize` (tuple or None): Overall figure size. If None, computed automatically. Default: `None`  

---

## Summary

The CoMut plot consists of up to seven coordinated panels, all sharing the same x-axis (sample order):

| Panel | Function | Data Source | Description |
|-------|----------|-------------|-------------|
| A | `comut_mutation_burden()` | PyMutation MAF data | Stacked bars of synonymous/non-synonymous TMB |
| B | `comut_mutation_signatures_plot()` | TSV or DataFrame | Stacked bars of signature contributions |
| C | `comut_purity_plot()` | TSV or DataFrame | Heatmap of tumor purity values |
| D | `comut_mutation_type_plot()` | TSV or DataFrame | Oncoprint of mutation types per gene |
| E | `comut_cna_plot()` | TSV or DataFrame | Matrix of copy number alterations |
| F | `comut_wgd_plot()` | TSV or DataFrame | Track showing whole genome doubling status |
| G | `comut_same_patient_plot()` | TSV or DataFrame | Track connecting samples from same patient |

### Workflow Recommendations

1. Generate Panel A first to establish sample order based on non-synonymous TMB  
2. Save the sample order using `save_sample_order()` for reuse  
3. Use the same `sample_order` parameter for all subsequent panels to ensure alignment  
4. Prepare external data (signatures, purity, CNA, etc.) as TSV files in tidy format  
5. Use `gene_order` parameters to control which genes appear in Panels D and E  

### Data Format

External data panels accept different formats depending on the panel:

Wide format (sample as rows, features as columns):  
- Panel B (signatures): `sample` + one column per signature (e.g., `Signature 1`, `Signature 2`)  
- Panel C (purity): `sample` + `value` (or `purity`, `tumor_purity`)  

Tidy/long format (one row per observation):  
- Panel D (mutation type): `sample`, `category` (gene name), `value` (mutation type)  
- Panel E (CNA): `sample`, `category` (gene name), `value` (CNA state)  
- Panel F (WGD): `sample`, `value` (Yes/No status)  
- Panel G (same patient): `sample`, `group` (patient ID)  

Column names are case-insensitive and flexible (e.g., `sample_id` and `Tumor_Sample_Barcode` are both recognized as sample identifiers). All functions return matplotlib Figure objects that can be saved or displayed.
