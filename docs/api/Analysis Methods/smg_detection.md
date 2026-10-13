# Significantly Mutated Genes (SMG) Detection

`detect_smg_oncodriveclustl()` integrates the OncodriveCLUSTL pipeline into `PyMutation` to detect significantly mutated driver genes displaying spatial clustering of somatic mutations along protein sequences.

## Method Overview

The method automatically handles:
1. Preparation of mutation datasets and regional transcript definitions via `pyensembl`.
2. Format validation and chromosome representation alignment.
3. Computation of optimal permutations tailored to the dataset's SNV count.
4. Execution of OncodriveCLUSTL clustering algorithms.
5. Statistical filtering by empirical false discovery rate threshold (`threshold` on `Q_EMPIRICAL`).

## Signature

```python
def detect_smg_oncodriveclustl(
    self,
    threads: int = 4,
    run_analysis: bool = True,
    threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
```

## Parameters

- `threads` (int, default: 4): Number of parallel CPU threads to use for simulation and clustering.
- `run_analysis` (bool, default: True): Whether to execute the full OncodriveCLUSTL analysis pipeline.
- `threshold` (Optional[float], default: None): Empirical q-value (`Q_EMPIRICAL`) cutoff for identifying significant candidate driver genes. If `None`, raw mutation tables and cluster tables are produced without filtering.

## Return Value

Returns a tuple `(mutations_df, significant_genes_df)`:
- `mutations_df`: DataFrame of prepared mutations (`CHROMOSOME`, `POSITION`, `REF`, `ALT`, `SAMPLE`).
- `significant_genes_df`: DataFrame of driver genes passing the empirical significance threshold.

## Example

```python
mutations_df, smg_results = py_mut.detect_smg_oncodriveclustl(
    threads=4,
    threshold=0.10
)
print(smg_results.head())
```
