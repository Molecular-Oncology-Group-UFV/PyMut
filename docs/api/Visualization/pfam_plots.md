# Pfam Domain Visualizations

pyMut provides dedicated publication-ready visualizations to explore Pfam domain alterations in cancer cohorts:

1. **`plot_pfam_domains()`**: Displays the top mutated Pfam protein domains across the cohort as an adaptive horizontal bar chart.
2. **`plot_pfam_domain_breakdown()`**: Zooms into a specific domain to reveal how mutations are distributed across individual genes carrying that domain (using either horizontal bar plots or donut charts).

---

## `plot_pfam_domains()`

### Overview

`plot_pfam_domains()` plots the top Pfam domains hit by mutations. The chart dimensions automatically adapt to the dataset size:
- Height scales with the number of displayed domains (`top_n`).
- Width scales dynamically with the character length of the domain names.

Each bar displays the domain label (e.g. `PF00096 (zf-C2H2) - 15 genes`), the total variant count, and the percentage of all domain-annotated variants.

### Method Signature

```python
def plot_pfam_domains(
    self,
    *,
    aa_column: Optional[str] = None,
    summarize_by: str = "PfamDomain",
    top_n: int = 15,
    include_synonymous: bool = False,
    color: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    base_fontsize: float = 10.0,
    title_fontsize: float = 14.0,
    subtitle_fontsize: Optional[float] = None
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
```

### Parameters

- `aa_column` (Optional[str]): Column with amino acid positions. Default: `None` (auto-resolved).
- `summarize_by` (str): Aggregation level (`"PfamDomain"` or `"AAPos"`). Default: `"PfamDomain"`.
- `top_n` (int): Number of top domains to display. Default: `15`.
- `include_synonymous` (bool): Whether to include synonymous variants. Default: `False`.
- `color` (Optional[str]): Bar fill color. Default: muted blue.
- `ax` (Optional[Axes]): Existing matplotlib axes to draw on.
- `figsize` (Optional[Tuple[float, float]]): Custom figure dimensions `(width, height)`. If None, automatically computed.
- `save_path` (Optional[str]): Path to export figure (e.g. `"pfam_domains.png"`).
- `base_fontsize` (float): Base font size for labels and ticks. Default: `10.0`.
- `title_fontsize` (float): Font size for main title. Default: `14.0`.

### Example

```python
from pyMut import read_maf

py_mut = read_maf("src/pyMut/data/examples/MAF/tcga_laml.maf.gz", assembly="37")
annotated = py_mut.annotate_pfam(strategy="basic")

fig, ax = annotated.plot_pfam_domains(
    top_n=15,
    figsize=(8, 10),
    save_path="pfam_domains_basic.png"
)
```

---

## `plot_pfam_domain_breakdown()`

### Overview

`plot_pfam_domain_breakdown()` investigates how variants within a single domain are distributed across the different genes that encode it. This resolves whether mutations concentrate in a single driver gene or distribute across an entire gene family sharing the domain motif.

Supports two visualization modes:
- **`kind="bar"`** (default): Horizontal bar chart showing gene counts and percentages.
- **`kind="donut"`**: Ring/donut chart showing gene proportions with a center hole.

Long tails of genes beyond `top_n` are automatically aggregated into a single `"Other (N genes)"` slice to keep the plot clear and readable.

### Method Signature

```python
def plot_pfam_domain_breakdown(
    self,
    pfam_id: str,
    *,
    aa_column: Optional[str] = None,
    include_synonymous: bool = False,
    top_n: int = 10,
    kind: str = "bar",
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    base_fontsize: float = 10.0,
    title_fontsize: float = 14.0,
    donut_width: float = 0.42
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
```

### Parameters

- `pfam_id` (str): Domain identifier to zoom into (e.g. `PF00096`, `7tm_1`, `PKc_like`). Matches both Pfam accessions and display names.
- `aa_column` (Optional[str]): Amino acid position column. Default: `None`.
- `include_synonymous` (bool): Whether to include synonymous mutations. Default: `False`.
- `top_n` (int): Maximum number of individual genes to display before collapsing the tail into `"Other"`. Default: `10`.
- `kind` (str): Chart style: `bar` (default) or `donut`.
- `donut_width` (float): Relative ring thickness when `kind=donut`. Default: `0.42`.
- `save_path` (Optional[str]): File path to save output image.
- `figsize` (Optional[Tuple[float, float]]): Custom figure dimensions.

### Example

```python
# Donut chart breakdown
fig_donut, ax_donut = annotated.plot_pfam_domain_breakdown(
    "PKc_like",
    kind="donut",
    figsize=(6, 6),
    donut_width=0.65,
    save_path="pfam_donut_PKc.png"
)

# Bar chart breakdown
fig_bar, ax_bar = annotated.plot_pfam_domain_breakdown(
    "7tm_1",
    kind="bar",
    figsize=(6, 4),
    save_path="pfam_bar_7tm1.png"
)
```
