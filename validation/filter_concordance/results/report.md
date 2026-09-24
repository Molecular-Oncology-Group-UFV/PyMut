# pyMut vs maftools — Filter Concordance Report

Cohorts: TCGA-LAML, TCGA-PAAD

Key columns: Hugo_Symbol, Chromosome, Start_Position, End_Position, Tumor_Sample_Barcode
Jaccard threshold: >= 0.995

## TCGA-LAML

### TCGA-LAML Chromosome filter
- pyMut rows: 624 (624 unique keys)
- maftools rows: 624 (624 unique keys)
- Shared keys: 624
- pyMut only: 0
- maftools only: 0
- Jaccard index: 100.000%
- PASS (>= threshold): YES
- Exact match (Jaccard = 1, 0 discordant): YES

### TCGA-LAML DNMT3A region
- pyMut rows: 54 (54 unique keys)
- maftools rows: 54 (54 unique keys)
- Shared keys: 54
- pyMut only: 0
- maftools only: 0
- Jaccard index: 100.000%
- PASS (>= threshold): YES
- Exact match (Jaccard = 1, 0 discordant): YES

## TCGA-PAAD

### TCGA-PAAD Chromosome filter
- pyMut rows: 12839 (12839 unique keys)
- maftools rows: 12839 (12839 unique keys)
- Shared keys: 12839
- pyMut only: 0
- maftools only: 0
- Jaccard index: 100.000%
- PASS (>= threshold): YES
- Exact match (Jaccard = 1, 0 discordant): YES

### TCGA-PAAD TP53 region
- pyMut rows: 119 (119 unique keys)
- maftools rows: 119 (119 unique keys)
- Shared keys: 119
- pyMut only: 0
- maftools only: 0
- Jaccard index: 100.000%
- PASS (>= threshold): YES
- Exact match (Jaccard = 1, 0 discordant): YES

**OVERALL: PASS**
