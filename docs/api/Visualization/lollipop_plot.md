# Lollipop Plot

The `lollipop_plot()` function visualizes somatic mutations mapped onto a protein structure, revealing mutation hotspots and their relationship to functional domains.

## Overview

Somatic mutations in cancer often cluster at specific amino acid positions within proteins, particularly in functionally important domains. This visualization maps mutations onto the linear protein sequence, highlighting patterns that may indicate driver mutations or functionally critical positions.

Use cases:  
- Identify mutation hotspots (recurrently mutated positions)  
- Understand functional consequences (mutations in catalytic sites, binding regions)  
- Compare mutation types at specific amino acid positions  
- Discover positional selection patterns across a cohort  

## What the plot shows

The lollipop plot displays:  
- Mutation positions along the protein sequence (x-axis in amino acid coordinates)  
- Mutation types at each position (color-coded by variant classification)  
- Mutation frequency at each site (stem height, proportional to count)  
- Functional domain architecture (PFAM domain rectangles)  
- Specific amino acid changes at hotspots (labeled positions, e.g., R882C)  
- Gene alteration frequency across the cohort (somatic mutation rate in title)  

## Visual components

Protein backbone:  
- Gray horizontal bar representing the protein from residue 1 to full length (y = 0.3–0.7)  

PFAM domains:  
- Colored rectangles overlaid on the backbone at their start–end amino acid coordinates (y = 0.25–0.75)  
- Sorted by length (longest first) for proper overlapping visualization  
- Domain names labeled when space permits  

Lollipops (mutations):  
- Vertical stems with circles at the top, one per unique amino acid change at each position  
- Circle color encodes variant classification using a fixed color palette:  
    - Missense_Mutation: Green (#008000)  
    - Nonsense_Mutation: Red (#FF0000)  
    - Frame_Shift_Del: Dark magenta (#8B008B)  
    - Frame_Shift_Ins: Dark violet (#9400D3)  
    - In_Frame_Del: Dark turquoise (#00CED1)  
    - In_Frame_Ins: Steel blue (#4682B4)  
    - Splice_Site: Dark orange (#FF8C00)  
    - Other types: Various colors  
- Stem height is proportional to mutation count and linearly rescaled when counts are high to maintain readability  
- Circle size is fixed for visual consistency  
- Aggregation rule: When multiple amino acid changes occur at the same position (e.g., R882H, R882C), we plot separate lollipops. If the same amino acid change appears under multiple classifications, the most frequent classification is used  

Labels:  
- Top N mutated positions labeled with their specific amino acid change (e.g., R882C, D835Y)  
- Positioned directly above lollipops with dotted gray connector lines  

Title and metadata:  
- Gene name (italicized) with Somatic Mutation Rate percentage  
- Subtitle displays the transcript/isoform used (e.g., NM_022552) if specified  

Legend:  
- Maps colors to variant classifications  
- Shows total mutation count for each classification type (e.g., "Missense_Mutation (245)")  

## Required data

A MAF (Mutation Annotation Format) table with the following columns:

| Column | Purpose |
|---|---|
| Hugo_Symbol | Gene symbol used to filter rows for the target gene |
| Tumor_Sample_Barcode | Sample identifier, used to compute gene-level mutation rate and count unique samples |
| Variant_Classification | Category for coloring (e.g., Missense_Mutation, Nonsense_Mutation, Frame_Shift_Del, etc.) |
| Amino acid change | Needed to derive the amino acid position and change. Common field names include HGVSp_Short, Protein_Change, or AAChange. Must contain HGVS notation (examples: p.R882C, p.G646Wfs*12, p.E746_A750del). The parser extracts the left-most numeric coordinate for ranged edits |

Note: Only rows with a parseable amino acid position are plotted; records without one (commonly Splice_Site) are dropped.

Additional resources (automatically queried):

- Protein length: Queried from UniProt for the canonical protein length. If `transcript_id` is specified, attempts to retrieve transcript-specific length from internal database; otherwise uses canonical. If unavailable, estimated from observed positions  
- PFAM domain annotations: Obtained from UniProt or internal database (domain name + start/end amino acid coordinates)  
- Transcript metadata: Optional; user can specify `transcript_id` or `protein_id` to override automatic selection  

Reproducibility note: For reproducible figures across time, prefer using `custom_domains` with a fixed, versioned PFAM mapping (e.g., shipped with the package). Online queries to UniProt can change as the database is updated.

## Interpretation

Hotspots:  
- Tall stems indicate recurrent mutations at specific positions  
- These may represent driver mutations or functionally critical residues  
- Height is proportional to mutation count (scaled for visual clarity)  
- Hotspots within or near conserved domains often indicate selective pressure  

Domain context:  
- Mutations clustered within a domain (e.g., catalytic sites, binding regions) suggest functional importance  
- The relationship between mutation position and domain architecture can reveal mechanisms of oncogenesis  

Mutation types:  
- Missense_Mutation (green): Amino acid substitutions; often cluster at hotspots  
- Nonsense_Mutation (red): Premature stop codons; truncating mutations  
- Frame_Shift_Del/Ins (purple shades): Frameshift mutations; usually loss-of-function  
- In_Frame_Del/Ins (blue shades): In-frame insertions/deletions; may preserve some function  
- Splice_Site (orange): Splice junction mutations; affect RNA splicing  

Somatic Mutation Rate:  
- Calculated as: (number of unique samples with ≥1 non-synonymous mutation in the gene) / (total unique samples in the MAF) × 100  
- High percentage suggests the gene is commonly altered across the cohort  
- Effect size depends on where mutations occur and their type  

Specific changes:  
- Labels show exact amino acid changes (e.g., R882C), not just positions  
- Allows identification of recurrent substitutions that may have specific functional consequences  

## Function parameters

Required:  
- `gene` (str): Gene symbol to visualize (e.g., "FLT3", "TP53", "DNMT3A")  

Optional:  
- `aa_col` (str): Column name containing protein change annotation. Default: "HGVSp_Short"  
- `transcript_id` (str): Specific RefSeq/Ensembl transcript ID to use for isoform selection. If provided, attempts to retrieve transcript-specific domains and protein length. Default: None  
- `protein_id` (str): Specific UniProt protein ID to use. Overrides automatic protein resolution. Default: None  
- `domains_source` (str): Source for domain annotations. Use "pfam" to query PFAM domains, or None to skip domain visualization. Default: "pfam"  
- `custom_domains` (List[Dict]): Custom domain definitions that override automatic domain resolution. Format: [{"start": 100, "end": 200, "name": "Kinase"}, ...]. Recommended for reproducible figures. Default: None  
- `count_by` (str): How to count mutations. Use "mutations" to count all mutation events, or "samples" to count unique samples (deduplicates). Default: "mutations"  
- `label_top_n` (int): Number of top mutated positions to label. Default: 20  
- `show_lollipops` (bool): If True (default), show mutation lollipops; if False, show only protein domains. Default: True  
- `figsize` (Tuple[int, int]): Figure size as (width, height) in inches. Default: (16, 6)  
- `title` (str): Custom plot title. Default: None (auto-generated)  

## Usage examples

Basic usage with default PFAM domains:

```python
fig = pymutation.lollipop_plot(gene="FLT3")
pymutation.save_figure(fig, "FLT3_lollipop.png")
```

Count by unique samples instead of all mutations:

```python
fig = pymutation.lollipop_plot(gene="TP53", count_by="samples", label_top_n=15)
```

Show only protein domains without mutations:

```python
fig = pymutation.lollipop_plot(gene="TP53", show_lollipops=False)
```

Use specific transcript for reproducibility:

```python
fig = pymutation.lollipop_plot(
    gene="DNMT3A",
    transcript_id="NM_022552",
    label_top_n=10
)
```

Use custom domains:

```python
domains = [
    {"start": 100, "end": 300, "name": "DNA Binding"},
    {"start": 320, "end": 355, "name": "Tetramerization"}
]
fig = pymutation.lollipop_plot(gene="TP53", custom_domains=domains)
```

## Important notes

Mutation aggregation:  
- Events are aggregated by (position, amino_acid_change) tuple. Distinct amino acid changes at the same position (e.g., R882H vs R882C) produce separate lollipops  
- When the same amino acid change appears under multiple variant classifications, only the dominant (most frequent) classification is shown in the visualization  

Counting modes:  
- `count_by="mutations"`: Count all mutation events (default behavior)  
- `count_by="samples"`: Count unique samples, deduplicating multiple mutations in the same sample. This typically lowers stem heights at hotspots where the same sample has multiple events  

Filtering and parsing:  
- Only non-synonymous variants are included (excludes Silent, Intron, IGR, 3'UTR, etc.)  
- Variants without parseable amino acid positions are automatically dropped  
- Splice_Site events are included only if they have a valid amino acid position annotation  

Visual encoding:  
- Stem height is proportional to mutation count, with linear rescaling applied for high counts to maintain readability  
- Circle size is fixed for all lollipops to ensure visual consistency  
- Labels are placed above lollipops for clarity  

Color palette:  
- Fixed color scheme (standard palette) ensures consistent visualization across runs  
- Colors follow established variant classification conventions  
- Not customizable via parameters to maintain reproducibility  

Domain resolution (priority order):  
1. `custom_domains` (if provided) - recommended for reproducible figures  
2. UniProt query with PFAM annotations - most comprehensive but may change over time  
3. Internal database lookup (if transcript_id provided)  
4. Full-length fallback (single domain spanning entire protein)  

Note: For reproducibility, we recommend pinning a versioned PFAM mapping (e.g., shipped with the package) instead of live queries. Domains are sorted by length (longest first) for proper overlapping display and validated against protein length

Protein length resolution (priority order):  
1. UniProt query for canonical protein (most reliable)  
2. Internal database (if protein_id or transcript_id provided)  
3. Estimated from observed mutation positions (95th percentile × 1.2)  

## Example output

For gene DNMT3A with transcript NM_022552:  
- Protein length: 912 amino acids  
- Hotspot: R882 often shows high mutation frequency  
- Domains: DNMT3b-related, ADDz_Dnmt3, and C-terminal methyltransferase domains  
- Title displays: "Somatic Mutation Rate: 24.87%" (example percentage)  

