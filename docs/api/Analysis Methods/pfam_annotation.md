### annotate_pfam

#### Short description

Annotates a `PyMutation` dataset with Pfam protein domains using either a standard gene-symbol mapping strategy (`basic`, matching `maftools`) or a comprehensive multi-tier resolution strategy (`extended`, leveraging VEP-derived UniProt IDs and domain fields).

#### Signature

```python
def annotate_pfam(
    self,
    db_conn: Optional[duckdb.DuckDBPyConnection] = None,
    *,
    strategy: str = "basic"
) -> PyMutation:
```

#### Parameters

| Parameter | Type | Required | Description |
| --------- | ---- | -------- | ----------- |
| `db_conn` | `Optional[duckdb.DuckDBPyConnection]` | No | DuckDB connection for querying the internal Pfam and UniProt reference tables. If `None` (default), a database connection is created and closed automatically. |
| `strategy` | `str` | No | Annotation strategy: `"basic"` (default) or `"extended"`. |

#### Strategies: When to Use Each

- **`strategy="basic"` (Default, maftools-equivalent)**:
    - **How it works**: Maps variants using gene symbol (`Hugo_Symbol`) and amino-acid change / position (parsed from `Protein_Change`, `HGVSp_Short`, or `HGVSp`) against the bundled gene-to-Pfam table. Variants that cannot be resolved stay unmapped without fallback.
    - **When to use**: Ideal for any standard MAF or cohort data (such as raw TCGA MAF files) where Ensembl VEP annotations are not present. Requires only `Hugo_Symbol` and a standard protein change column.
    - **Performance**: Fast and deterministic; logs the total count and percentage of successfully mapped variants.

- **`strategy="extended"` (Multi-tier resolution)**:
    - **How it works**: Executes a multi-tier cascade across all variants:
        1. **1a & 1b**: Maps gene symbols and VEP-derived UniProt identifiers simultaneously, prioritizing UniProt-specific matches (1b) when both resolve a variant.
        2. **2**: Cascades unresolved variants into explicit VEP domain annotation fields (such as `DOMAINS` containing Pfam accessions).
        3. **3**: Falls back to loose text parsing for any remaining unmapped variants.
    - **When to use**: Use when working with VEP-annotated MAF datasets (such as `tcga_laml_VEP_annotated.maf.gz`). Requires VEP annotation fields; raises `ValueError` if run on a non-VEP annotated dataset without UniProt/domain fields.
    - **Performance**: Provides maximal domain coverage and logs a detailed breakdown of variants resolved at each tier (1a, 1b, 2, and 3).

#### Return value

Returns a new `PyMutation` object containing the Pfam domain annotations added as columns (e.g. `PfamDomain`, `pfam_id`, `pfam_name`).

#### Exceptions

* `ValueError`: If `strategy` is not `"basic"` or `"extended"`, or if `strategy="extended"` is requested on a non-VEP annotated dataset.
* `PfamAnnotationError`: If database connection fails or required columns cannot be detected.

#### Usage Examples

**Basic strategy (standard MAF):**

```python
from pyMut import read_maf

py_mut = read_maf("src/pyMut/data/examples/MAF/tcga_laml.maf.gz", assembly="37")
annotated_basic = py_mut.annotate_pfam(strategy="basic")
print(annotated_basic.data[["Hugo_Symbol", "PfamDomain"]].head())
```

**Extended strategy (VEP-annotated MAF):**

```python
from pyMut import read_maf

py_mut_vep = read_maf("src/pyMut/data/examples/MAF/tcga_laml_VEP_annotated.maf.gz", assembly="37")
annotated_extended = py_mut_vep.annotate_pfam(strategy="extended")
print(annotated_extended.data[["Hugo_Symbol", "PfamDomain"]].head())
```
