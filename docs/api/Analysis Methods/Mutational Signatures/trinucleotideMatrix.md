### trinucleotideMatrix

#### Short description

Builds the **96-trinucleotide context matrix** for all single-nucleotide variants (SNVs) in a `PyMutation` object, enriching the original data with context annotations to enable downstream signature analysis.

#### Signature

```python
def trinucleotideMatrix(
    self,
    ref_genome,
    prefix=None,
    add=True,
    ignoreChr=None,
    useSyn=True,
    fn=None,
    apobec_window=20,
):
    pass
```

#### Parameters

| Parameter       | Type                 | Required | Description |
| --------------- | -------------------- | -------- | ----------- |
| `ref_genome`    | `str` (path)         | **Yes**  | Path to the reference FASTA used to extract trinucleotide contexts (must be indexed with a .fai). |
| `prefix`        | `str` or `None`      | No       | Chromosome name prefix to add/remove (e.g., `'chr'`). Default: `None`. |
| `add`           | `bool`               | No       | Controls how `prefix` is applied. If `True`, adds the prefix when missing; if `False`, removes it. Default: `True`. |
| `ignoreChr`     | `list[str]` or `None`| No       | Chromosomes to exclude from analysis (e.g., `['chrM']`). Default: `None`. |
| `useSyn`        | `bool`               | No       | Include synonymous variants. Set to `False` to exclude them. Default: `True`. |
| `fn`            | `str` or `None`      | No       | If provided, writes an APOBEC enrichment report to `{fn}.apobec_enrichment.tsv` (approximate background). Default: `None`. |
| `apobec_window` | `int`                | No       | Window size (+/-) used for approximate background TCW motif estimation. Default: `20`. |

#### Return value

A tuple `(contexts_df, enriched_data)`:

* **`contexts_df`** – `pd.DataFrame` of shape **96 × N-samples**. Each row corresponds to one of the 96 canonical trinucleotide mutation classes; each column contains the raw counts for a sample (columns correspond to samples present after filtering). Rows are ordered as in the internal `TRINUCLEOTIDE_CONTEXTS` list.
* **`enriched_data`** – the subset of input SNVs that yielded valid contexts, with extra columns:

  * `trinuc` – the reference trinucleotide (e.g. "ACA").
  * `class96` – class label in the form "A[C>T]A".
  * `idx96` – integer index (0–95) into the standard context order.
  * `norm_alt` – alternative allele after pyrimidine-normalization (useful for wide genotypes).

Both DataFrames are never `None`.

#### Exceptions

* `ImportError` – *pyfaidx* is missing.
* `ValueError` – required columns are absent, no valid SNVs are found, or the FASTA cannot be read.

#### Minimal usage example

```python
import pandas as pd
from pyMut.core import PyMutation

# Load MAF (tab-delimited; lines starting with '#' are comments)
df = pd.read_csv("my_cohort.maf", sep="\t", comment="#", low_memory=False)

pm = PyMutation(df)
contexts_df, enriched = pm.trinucleotideMatrix(
    ref_genome="GRCh37.fa",
    prefix="chr",          # or None, depending on your FASTA/MAF naming
    add=True,
    ignoreChr=["chrM"],    # commonly ignored
    useSyn=True,
    fn=None,
)

# Save the 96xN matrix to CSV
contexts_df.to_csv("trinucleotide_contexts_96xN.csv")
```
