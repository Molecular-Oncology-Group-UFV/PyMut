### extractSignatures

#### Short description

Extracts mutational signatures from a **96 × samples** trinucleotide-context matrix using NMF (KL divergence with multiplicative updates). Returns signature profiles and per-sample contributions.

#### Signature

```python
def extractSignatures(
    contexts_df: pd.DataFrame,
    n: int,
    parallel: int = 4,
    pConstant: Optional[float] = None,
) -> Dict:
```

#### Parameters

| Parameter     | Type           | Required | Description                                                                                              |
|---------------|----------------|----------|----------------------------------------------------------------------------------------------------------|
| `contexts_df` | `pd.DataFrame` | **Yes**  | **96 × N** matrix of raw counts produced by `trinucleotideMatrix`.                                       |
| `n`           | `int`          | **Yes**  | Number of signatures to extract.                                                                         |
| `parallel`    | `int`          | No       | Number of cores to use. Currently not used (kept for compatibility).                                     |
| `pConstant`   | `float | None` | No       | Small positive constant added to avoid numerical issues. Must be > 0 if provided. If `None` and there are samples with total mutations = 0, `0.01` is applied automatically. |

Constraints: `n ≥ 1` and `contexts_df` must have exactly 96 rows. If provided, `pConstant` must be > 0.

#### Return value

`dict` with keys:

| Key                  | Type                         | Meaning                                                                                  |
|----------------------|------------------------------|------------------------------------------------------------------------------------------|
| `signatures`         | `pd.DataFrame`               | Scaled signature matrix **96 × n** (each signature sums to 1).                           |
| `contributions`      | `pd.DataFrame`               | Normalized contribution matrix **n × samples** (each sample sums to 1).                  |
| `contributions_abs`  | `pd.DataFrame`               | Absolute contribution matrix **n × samples** (not normalized).                           |
| `nmfObj`             | `sklearn.decomposition.NMF`  | Fitted NMF model object.                                                                 |

#### Exceptions

- `ImportError` – *scikit-learn* missing.
- `ValueError` – Invalid arguments (wrong shapes, `n < 1`, `pConstant <= 0`) or NMF failure.

#### Minimal usage example

```python
# ctx is the 96 × samples DataFrame returned by trinucleotideMatrix
res = extractSignatures(ctx, n=4)

print("signatures:", res["signatures"].shape)            # (96, 4)
print("contributions (norm):", res["contributions"].shape)     # (4, n_samples)
print("contributions (abs):", res["contributions_abs"].shape)  # (4, n_samples)
print(type(res["nmfObj"]))
```