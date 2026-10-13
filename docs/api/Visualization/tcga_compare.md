# TCGA Cohort TMB Comparison

`tcga_compare()` and `calculate_tcga_compare()` place a user cohort tumor mutational burden (TMB) in the context of ~33 TCGA cancer cohorts bundled with `maftools`, reproducing `maftools`' `tcgaCompare()` analysis and plot.

---

## `tcga_compare()` (Visualization)

### Overview

Visualizes cohort TMB alongside reference TCGA cohorts:
- Cohorts are arranged along the x-axis, typically ordered by ascending median TMB.
- Each cohort is represented by a vertical scatter of dots (one per sample, sorted by TMB) with a prominent red horizontal bar indicating the median.
- The input cohort dots are rendered in black for immediate distinction, while TCGA reference cohorts are rendered in gray.

### Method Signature

```python
def tcga_compare(
    self,
    cohort_name: str = "Input",
    capture_size: Optional[float] = 50.0,
    tcga_capture_size: float = 35.8,
    tcga_cohorts: Optional[List[str]] = None,
    primary_site: bool = False,
    logscale: bool = True,
    rm_hyper: bool = False,
    rm_zero: bool = True,
    decreasing: bool = False,
    figsize: Tuple[int, int] = (14, 6),
    title: Optional[str] = None
) -> matplotlib.figure.Figure:
```

### Parameters

- `cohort_name` (str): Label for the user cohort (default: `"Input"`).
- `capture_size` (Optional[float]): Exome capture size in Mb used to normalize the user cohort TMB into mutations/Mb (default: `50.0`). If `None`, raw mutation counts are compared.
- `tcga_capture_size` (float): Target capture size in Mb for TCGA cohorts (default: `35.8` Mb, matching typical TCGA whole-exome sequencing). Ignored if `capture_size` is `None`.
- `tcga_cohorts` (Optional[List[str]]): Specific TCGA cohort codes to include (e.g. `["PAAD", "LAML", "SKCM"]`). If `None`, all ~33 reference cohorts are included.
- `primary_site` (bool): If True, groups TCGA samples by tumor anatomical primary site instead of TCGA project codes. Default: `False`.
- `logscale` (bool): Whether to plot TMB on a log10 scale. Default: `True`.
- `rm_hyper` (bool): Whether to remove hypermutated outlier samples based on Tukey boxplot statistics. Default: `False`.
- `rm_zero` (bool): Whether to drop samples with zero non-synonymous mutations. Default: `True`.
- `decreasing` (bool): Whether to sort cohorts in descending order of median TMB. Default: `False` (ascending).
- `figsize` (Tuple[int, int]): Figure size in inches. Default: `(14, 6)`.
- `title` (Optional[str]): Custom plot title.

---

## `calculate_tcga_compare()` (Statistical Tables)

### Overview

Computes the underlying numeric tables behind `tcga_compare()`, including median TMB metrics, per-sample normalized values, and pairwise two-sample t-tests against TCGA cohorts.

```python
results = py_mut.calculate_tcga_compare(
    cohort_name="LAML-TB",
    capture_size=50.0,
    save_files=False
)
```

### Return Value

Returns a dictionary containing:
- `median_mutation_burden`: Table with `Cohort`, `Cohort_Size`, and `Median_Mutations`.
- `mutation_burden_perSample`: Per-sample mutation counts and normalized `total_perMB`.
- `pairwise_t_test`: Pairwise t-test comparisons between cohorts (`Cohort1`, `Cohort2`, `Pval`).
