### pfam_domains

#### Short description

Summarises already-annotated Pfam protein domain information in a `PyMutation` object, returning the most frequently mutated domains or residue-level hotspots.

#### Signature

```python
def pfam_domains(
    *,
    aa_column: Optional[str] = None,
    summarize_by: str = "PfamDomain",
    top_n: int = 10,
    include_synonymous: bool = False
) -> pd.DataFrame:
```

#### Parameters

| Parameter | Type | Required | Description |
| --------- | ---- | -------- | ----------- |
| `aa_column` | `Optional[str]` | No | Column containing amino acid positions. If `None` (default), automatically resolved from previous `annotate_pfam()` metadata or standard MAF column names (`aa_pos`, `Protein_Change`, `HGVSp_Short`). |
| `summarize_by` | `str` | No | Aggregation level: `"PfamDomain"` (default, aggregates variants by Pfam domain) or `"AAPos"` (summarises hotspots per amino acid residue). |
| `top_n` | `int` | No | Number of top rows to return, ranked by mutation count. Default: `10`. |
| `include_synonymous` | `bool` | No | If `False` (default), silent/synonymous mutations are excluded. Set to `True` to include all variants. |

#### Return value

`pd.DataFrame` containing domain summaries:
- When `summarize_by="PfamDomain"`: `pfam_id`, `pfam_name`, `n_genes`, `n_variants`, `genes` (with the underlying gene breakdown stored in `df.attrs["gene_breakdown"]`).
- When `summarize_by="AAPos"`: columns per residue position with `n_variants` and `n_genes`.

---

### pfam_domain_protein_breakdown

#### Short description

Zooms into a single Pfam domain to break down variant distribution across the individual proteins/genes that harbor the domain.

#### Signature

```python
def pfam_domain_protein_breakdown(
    self,
    pfam_id: str,
    *,
    aa_column: Optional[str] = None,
    include_synonymous: bool = False
) -> pd.DataFrame:
```

#### Parameters

| Parameter | Type | Required | Description |
| --------- | ---- | -------- | ----------- |
| `pfam_id` | `str` | Yes | Pfam identifier (accession e.g. `PF00096`, or display name e.g. `zf-C2H2`) as reported in `pfam_domains()`. |
| `aa_column` | `Optional[str]` | No | Column with amino acid positions. If `None`, automatically detected. |
| `include_synonymous` | `bool` | No | Whether to include synonymous variants. Default: `False`. |

#### Return value

`pd.DataFrame` with columns: `pfam_id`, `Hugo_Symbol`, `n_variants`, and `pct_of_domain`, sorted by variant count descending.
