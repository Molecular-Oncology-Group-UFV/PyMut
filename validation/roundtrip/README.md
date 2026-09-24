# Multi-source round-trip test (MAF/VCF preservation)

For each input file (2 bundled example VCFs + 2 project MAFs), verifies
that the PyMut cycle **read -> export -> re-read** preserves 100 % of the
variant IDs (`CHROM_POS_REF_ALT`) and of the per-sample genotypes, and
diagnoses any mismatch (formatting vs real content loss, REF/ALT changes,
indel handling, phasing order swaps).

## Usage

```bash
python multi_source_roundtrip_test.py
```

Paths are resolved relative to this script folder: example VCFs are
located automatically from the installed pyMut package; project MAFs are
read from `<project_root>/data/`; figures are written to `results/` here.

## Inputs

| File | Source |
|------|--------|
| `subset_1k_variants_ALL.chr10...vcf` | pyMut package examples (auto-located) |
| `subset_50k_variants_vep...vcf.gz` | pyMut package examples (auto-located) |
| `<project_root>/data/tcga_laml.maf.gz` | project data |
| `<project_root>/data/PAAD-TB.final_analysis.maf` | project data |

Edit the `DATASETS` list at the top of the script to add/remove inputs.

## Outputs (`results/`)

`roundtrip_preservation_combined.png` (all datasets in one row) plus one
individual panel per dataset in `results/individual/`. The script also
prints detailed per-dataset diagnostics and mismatch bucketing
(indel / order_swap / missing_vs_called / other) to the console.
