"""
Module for creating TCGA cohort comparison plots.

Reproduces maftools' ``tcgaCompare()``: places a cohort's per-sample tumor
mutation burden (TMB) in the context of the ~33 TCGA cohorts bundled with
maftools (``pyMut/data/tcga_cohort.txt.gz``, maftools' own
``tcga_cohort.txt.gz``), as a per-cohort dot plot with a median line, ordered
by median TMB.

Main functions:
- load_tcga_cohort_table(): Loads the bundled TCGA per-sample reference table
- compute_tcga_compare(): Combines this cohort's TMB with the TCGA reference
  table and builds the comparison tables (median/per-sample/pairwise t-test)
- _create_tcga_compare_plot(): Draws the comparison plot
"""

import logging
from importlib import resources
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy import stats

if TYPE_CHECKING:
    from ..core import PyMutation

logger = logging.getLogger(__name__)

_TCGA_COHORT_FILE = "tcga_cohort.txt.gz"

# maftools' own default whole-exome capture size (Mb) used to normalize the
# bundled TCGA cohorts.
DEFAULT_TCGA_CAPTURE_SIZE = 35.8

# TCGA points (gray70), Input points (black) - matches maftools' default `col`.
DEFAULT_COL = ("#B3B3B3", "#000000")
# Alternating per-cohort background stripes - matches maftools' default `bg_col`.
DEFAULT_BG_COL = ("#EDF8B1", "#2C7FB8")
DEFAULT_MEDIAN_COL = "red"


def load_tcga_cohort_table(primary_site: bool = False,
                            tcga_cohorts: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load the bundled TCGA per-sample mutation-count reference table.

    Parameters
    ----------
    primary_site : bool, default False
        Group by tumor primary site (``site`` column) instead of by TCGA
        project code (``cohort`` column).
    tcga_cohorts : list of str, optional
        Restrict to these cohort codes (or site names, if ``primary_site``).

    Returns
    -------
    pd.DataFrame
        Columns: ``cohort``, ``Tumor_Sample_Barcode``, ``total``.
    """
    with resources.files("pyMut.data").joinpath(_TCGA_COHORT_FILE).open("rb") as fh:
        table = pd.read_csv(fh, sep="\t", compression="gzip")

    cohort_col = "site" if primary_site else "cohort"
    table = table[["Tumor_Sample_Barcode", "total", cohort_col]].rename(columns={cohort_col: "cohort"})

    if tcga_cohorts:
        table = table[table["cohort"].isin(tcga_cohorts)]
        if table.empty:
            raise ValueError(
                "No matching TCGA cohorts found for 'tcga_cohorts'. "
                "Check the requested cohort codes (or site names if primary_site=True)."
            )

    return table.reset_index(drop=True)


def _remove_boxplot_outliers(values: pd.Series) -> pd.Series:
    """Tukey's boxplot.stats() rule (coef=1.5), matching R's default outlier definition."""
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return values[(values >= lower) & (values <= upper)]


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment, equivalent to R's p.adjust(method='fdr')."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _pairwise_t_test(cohort_values: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Pairwise t-tests between every pair of cohorts sharing a common pooled
    standard deviation (equivalent to R's
    ``pairwise.t.test(pool.sd=TRUE, p.adjust.method="fdr")``, used internally
    by maftools' ``tcgaCompare()``).
    """
    names = list(cohort_values.keys())
    k = len(names)
    n_total = sum(len(v) for v in cohort_values.values())

    # Pooled within-group variance (one-way ANOVA mean squared error).
    grand_ss = sum(((cohort_values[n] - cohort_values[n].mean()) ** 2).sum() for n in names)
    df_resid = n_total - k
    pooled_var = grand_ss / df_resid if df_resid > 0 else np.nan

    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = cohort_values[names[i]], cohort_values[names[j]]
            n_a, n_b = len(a), len(b)
            if pooled_var == pooled_var and n_a > 0 and n_b > 0:
                se = np.sqrt(pooled_var * (1 / n_a + 1 / n_b))
            else:
                se = np.nan
            if se == se and se > 0:
                t_stat = (a.mean() - b.mean()) / se
                pval = 2 * stats.t.sf(abs(t_stat), df_resid)
            else:
                pval = np.nan
            rows.append({"Cohort1": names[i], "Cohort2": names[j], "Pval": pval})

    result = pd.DataFrame(rows, columns=["Cohort1", "Cohort2", "Pval"])
    valid = result["Pval"].notna()
    if valid.any():
        result.loc[valid, "Pval"] = _bh_adjust(result.loc[valid, "Pval"].to_numpy())
    return result.sort_values("Pval", na_position="last").reset_index(drop=True)


def compute_tcga_compare(py_mut: 'PyMutation',
                          cohort_name: str = "Input",
                          capture_size: Optional[float] = 50.0,
                          tcga_capture_size: float = DEFAULT_TCGA_CAPTURE_SIZE,
                          tcga_cohorts: Optional[List[str]] = None,
                          primary_site: bool = False,
                          rm_hyper: bool = False,
                          rm_zero: bool = True,
                          decreasing: bool = False) -> Dict[str, object]:
    """
    Combine this cohort's per-sample TMB with the bundled TCGA reference
    table and build the tables behind both ``PyMutation.calculate_tcga_compare()``
    and ``PyMutation.tcga_compare()``.

    Returns a dict with the three public tables (``median_mutation_burden``,
    ``mutation_burden_perSample``, ``pairwise_t_test``) plus an internal
    ``plot_data`` DataFrame (cohort, plot_total, TCGA) used by
    ``_create_tcga_compare_plot()``.
    """
    tmb = py_mut.calculate_tmb_analysis(save_files=False)['analysis'][['Tumor_Sample_Barcode', 'total']].copy()

    if rm_zero:
        n_zero = int((tmb['total'] == 0).sum())
        if n_zero:
            logger.warning(f"Removed {n_zero} samples with zero mutations from '{cohort_name}'.")
        tmb = tmb[tmb['total'] != 0]

    if tmb.empty:
        raise ValueError(f"No samples with mutations left for cohort '{cohort_name}'.")

    tmb['cohort'] = cohort_name

    tcga = load_tcga_cohort_table(primary_site=primary_site, tcga_cohorts=tcga_cohorts)

    combined = pd.concat(
        [tcga[['cohort', 'Tumor_Sample_Barcode', 'total']], tmb[['cohort', 'Tumor_Sample_Barcode', 'total']]],
        ignore_index=True,
    )
    combined['total'] = pd.to_numeric(combined['total'])

    if rm_hyper:
        cleaned = []
        for cohort, group in combined.groupby('cohort', sort=False):
            kept = _remove_boxplot_outliers(group['total'])
            n_removed = len(group) - len(kept)
            if n_removed:
                logger.info(f"Removed {n_removed} outliers from '{cohort}'.")
            cleaned.append(group.loc[kept.index])
        combined = pd.concat(cleaned, ignore_index=True)

    if capture_size is None:
        combined['plot_total'] = combined['total']
    else:
        is_input = combined['cohort'] == cohort_name
        combined['total_perMB'] = np.where(is_input, combined['total'] / capture_size,
                                            combined['total'] / tcga_capture_size)
        combined['plot_total'] = combined['total_perMB']

    median_summary = (
        combined.groupby('cohort')['plot_total']
        .agg(Cohort_Size='size', Median_Mutations='median')
        .reset_index()
        .rename(columns={'cohort': 'Cohort'})
        .sort_values('Median_Mutations', ascending=not decreasing)
        .reset_index(drop=True)
    )

    cohort_values = {c: g['plot_total'].to_numpy() for c, g in combined.groupby('cohort', sort=False)}
    pairwise = _pairwise_t_test(cohort_values)

    combined['TCGA'] = np.where(combined['cohort'] == cohort_name, 'Input', 'TCGA')

    per_sample_cols = ['cohort', 'Tumor_Sample_Barcode', 'total']
    if 'total_perMB' in combined.columns:
        per_sample_cols.append('total_perMB')

    return {
        'median_mutation_burden': median_summary,
        'mutation_burden_perSample': combined[per_sample_cols].reset_index(drop=True),
        'pairwise_t_test': pairwise,
        'plot_data': combined,
        'cohort_order': median_summary['Cohort'].tolist(),
    }


def _create_tcga_compare_plot(py_mut: 'PyMutation',
                               cohort_name: str = "Input",
                               capture_size: Optional[float] = 50.0,
                               tcga_capture_size: float = DEFAULT_TCGA_CAPTURE_SIZE,
                               tcga_cohorts: Optional[List[str]] = None,
                               primary_site: bool = False,
                               logscale: bool = True,
                               rm_hyper: bool = False,
                               rm_zero: bool = True,
                               decreasing: bool = False,
                               col: tuple = DEFAULT_COL,
                               bg_col: tuple = DEFAULT_BG_COL,
                               median_col: str = DEFAULT_MEDIAN_COL,
                               cohort_font_size: float = 8,
                               axis_font_size: float = 9,
                               figsize=(14, 7),
                               title: Optional[str] = None) -> Figure:
    """Build the maftools-style per-cohort dot/median TMB comparison plot."""
    import matplotlib.pyplot as plt

    result = compute_tcga_compare(
        py_mut,
        cohort_name=cohort_name,
        capture_size=capture_size,
        tcga_capture_size=tcga_capture_size,
        tcga_cohorts=tcga_cohorts,
        primary_site=primary_site,
        rm_hyper=rm_hyper,
        rm_zero=rm_zero,
        decreasing=decreasing,
    )

    plot_data = result['plot_data']
    cohort_order = result['cohort_order']
    median_summary = result['median_mutation_burden'].set_index('Cohort')
    n_cohorts = len(cohort_order)

    fig, ax = plt.subplots(figsize=figsize)

    # Faint full-plot background, matching maftools' translucent gray canvas.
    ax.set_facecolor((0.5, 0.5, 0.5, 0.1))

    per_cohort = {}
    all_y = []
    for i, cohort in enumerate(cohort_order):
        values = plot_data.loc[plot_data['cohort'] == cohort, 'plot_total'].to_numpy()
        values = np.sort(values)[::-1]  # largest first
        x = np.linspace(i, i + 1, len(values))[::-1] if len(values) > 1 else np.array([i + 0.5])
        is_input = cohort == cohort_name
        per_cohort[cohort] = (x, values, is_input)
        nonzero = values[values > 0]
        all_y.append(np.log10(nonzero) if logscale else values)

    all_y = np.concatenate(all_y) if all_y else np.array([0.0])
    y_min, y_max = float(np.floor(all_y.min())), float(np.ceil(all_y.max()))
    if y_min == y_max:
        y_max += 1
    y_at = np.arange(y_min, y_max + 1)

    # Alternating per-cohort background stripes.
    for i in range(n_cohorts):
        stripe_color = bg_col[i % len(bg_col)]
        ax.add_patch(Rectangle((i, y_min), 1, y_max - y_min, facecolor=stripe_color, alpha=0.2,
                                edgecolor='none', zorder=0))

    ax.hlines(y_at, 0, n_cohorts, linestyles='dashed', colors='gray', linewidth=0.6, zorder=1)

    for i, cohort in enumerate(cohort_order):
        x, values, is_input = per_cohort[cohort]
        with np.errstate(divide='ignore'):
            y = np.log10(values) if logscale else values
        y = np.clip(y, y_min, None)
        point_color = col[1] if is_input else col[0]
        ax.scatter(x, y, s=6, color=point_color, zorder=2)

        median_val = median_summary.loc[cohort, 'Median_Mutations']
        with np.errstate(divide='ignore'):
            median_y = np.log10(median_val) if logscale else median_val
        ax.hlines(median_y, i, i + 1, colors=median_col, linewidth=1.2, zorder=3)

    ax.set_xlim(0, n_cohorts)
    ax.set_ylim(y_min, y_max)

    tick_pos = np.arange(n_cohorts) + 0.5
    ax.set_xticks(tick_pos)
    tick_labels = ax.set_xticklabels(cohort_order, rotation=90, fontsize=cohort_font_size)
    for label, cohort in zip(tick_labels, cohort_order):
        if cohort == cohort_name:
            label.set_fontweight('bold')
            label.set_color(col[1])

    sample_sizes = [len(per_cohort[c][1]) for c in cohort_order]
    ax_top = ax.secondary_xaxis('top')
    ax_top.set_xticks(tick_pos)
    ax_top.set_xticklabels(sample_sizes, rotation=90, fontsize=cohort_font_size, style='italic')
    ax_top.tick_params(length=0)

    ax.set_yticks(y_at)
    if logscale:
        ax.set_yticklabels([f"{10 ** v:g}" for v in y_at], fontsize=axis_font_size)
    else:
        ax.set_yticklabels([f"{v:g}" for v in y_at], fontsize=axis_font_size)
    ax.set_ylabel("TMB (per MB)" if capture_size is not None else "TMB")

    ax.tick_params(axis='x', length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)

    fig.suptitle(title or f"TMB comparison with TCGA cohorts ({cohort_name})", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    return fig
