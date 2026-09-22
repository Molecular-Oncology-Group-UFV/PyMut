import math
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple


def plot_pfam_domains(summary: pd.DataFrame, *, color: Optional[str] = None,
                      figsize: Optional[Tuple[float, float]] = None, ax=None,
                      save_path: Optional[str] = None,
                      base_fontsize: float = 10.0, title_fontsize: float = 14.0,
                      subtitle_fontsize: Optional[float] = None):
    """
    Plot helper that draws the Pfam domains bar chart from a precomputed
    `summary` DataFrame (as returned by `PfamAnnotationMixin.pfam_domains`).
    Returns (fig, ax).
    """
    if len(summary) == 0:
        raise ValueError("Empty summary passed to plot_pfam_domains")

    has_breakdown = 'protein_breakdown' in summary.columns

    if 'display_name' in summary.columns:
        def make_label(row):
            pfam_id_val = row['pfam_id']
            has_id = pd.notna(pfam_id_val) and str(pfam_id_val).strip() != ''
            if not has_id:
                base = row['display_name']
            elif row['display_name'] != pfam_id_val:
                id_str = str(pfam_id_val)
                if len(id_str) > 24:
                    id_str = id_str[:21] + '...'
                base = f"{row['display_name']} ({id_str})"
            else:
                base = row['display_name']
            if has_breakdown:
                genes = row['protein_breakdown'].split(',')[0].strip()
                n_genes = int(row['n_genes']) if 'n_genes' in row else 1
                extra = f" +{n_genes - 1} more" if n_genes > 1 else ""
                base += f"\n{genes}{extra}"
            return base
        labels = summary.apply(make_label, axis=1)
    else:
        labels = summary['pfam_id']

    n_bars = len(summary)
    max_label_len = max((len(str(label).split(chr(10))[0]) for label in labels), default=10)

    per_bar_height = 0.62 if has_breakdown else 0.45
    header_height = 2.0
    fig_height = min(18, header_height + max(1.2, per_bar_height * n_bars))
    fig_width = min(14, max(7.5, 4.5 + max_label_len * 0.12))

    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    else:
        fig = ax.figure

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        pass

    if color:
        bar_colors = [color] * n_bars
    else:
        cmap = plt.get_cmap('viridis')
        bar_colors = [cmap(0.15 + 0.7 * (i / max(n_bars - 1, 1))) for i in range(n_bars)][::-1]

    y_pos = range(n_bars)
    ax.barh(
        y_pos, summary['n_variants'].values[::-1], color=bar_colors,
        edgecolor='white', linewidth=0.6, height=0.68, zorder=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels.values[::-1], fontsize=base_fontsize + 0.5)
    ax.set_xlabel('Number of variants', fontsize=base_fontsize, labelpad=8)

    ax.set_title(f"Top {n_bars} Pfam domains by variant count", fontsize=title_fontsize, fontweight='bold', pad=16)

    total_mapped = summary.attrs.get('total_mapped')
    total_considered = summary.attrs.get('total_considered')
    if total_mapped and total_considered:
        coverage_pct = total_mapped / total_considered * 100
        sf = subtitle_fontsize if subtitle_fontsize is not None else max(8.0, base_fontsize - 1)
        ax.text(
            0.5, 0.975,
            f"{total_mapped:,}/{total_considered:,} variants ({coverage_pct:.2f}%) mapped to a Pfam domain",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=sf, color='#666666',
        )
        try:
            fig.subplots_adjust(top=0.92)
        except Exception:
            pass

    pct_col = 'pct_of_mapped' if 'pct_of_mapped' in summary.columns else None
    reversed_summary = summary.iloc[::-1].reset_index(drop=True)
    max_variants = summary['n_variants'].max()
    for i, (_, row) in enumerate(reversed_summary.iterrows()):
        label = f"{int(row['n_variants']):,}"
        if pct_col:
            label += f" ({row[pct_col]:.2f}%)"
        ax.text(
            row['n_variants'] + max_variants * 0.015, i, label,
            va='center', fontsize=base_fontsize - 0.5, color='#333333',
        )

    ax.set_xlim(0, max_variants * 1.18)
    ax.grid(axis='x', linestyle='--', alpha=0.35, zorder=0)
    ax.grid(axis='y', visible=False)
    ax.set_axisbelow(True)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_pfam_domain_breakdown(breakdown: pd.DataFrame, *, pfam_id: str,
                               kind: str = 'bar', figsize: Optional[Tuple[float, float]] = None,
                               ax=None, save_path: Optional[str] = None,
                               base_fontsize: float = 10.0, title_fontsize: float = 14.0,
                               donut_width: float = 0.42):
    """
    Plot helper that draws the per-domain protein breakdown (bar/donut) from
    a precomputed `breakdown` DataFrame (as returned by
    `PfamAnnotationMixin.pfam_domain_protein_breakdown`). Returns (fig, ax).
    """
    if len(breakdown) == 0:
        raise ValueError("Empty breakdown passed to plot_pfam_domain_breakdown")

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        pass

    n_items = len(breakdown)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i / max(n_items - 1, 1) if n_items > 1 else 0) for i in range(n_items)]

    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig_height = min(10, max(3, 0.55 * n_items + 1.8))
            fig, ax = plt.subplots(figsize=(8, fig_height))
    else:
        fig = ax.figure

    title = f"How domain {pfam_id} is mutated across proteins"

    if kind == 'donut':
        wedges, _, autotexts = ax.pie(
            breakdown['n_variants'], colors=colors, startangle=90,
            autopct=lambda pct: f"{pct:.2f}%" if pct >= 5 else '',
            pctdistance=0.78, wedgeprops=dict(width=donut_width, edgecolor='white', linewidth=1.5),
        )
        for t in autotexts:
            t.set_color('white')
            t.set_fontsize(max(8.5, base_fontsize - 0.5))
            t.set_fontweight('bold')
        # draw external percentage labels for small slices (<5%) with connector lines
        total = float(breakdown['n_variants'].sum())
        pcts = (breakdown['n_variants'] / total) * 100
        for i, w in enumerate(wedges):
            pct = pcts.iloc[i]
            if pct < 5:
                # middle angle of the wedge
                ang = (w.theta2 + w.theta1) / 2.0
                ang_rad = math.radians(ang)
                # start point near the outer edge of the donut
                r_start = 1 - (donut_width / 2.0)
                x_start = math.cos(ang_rad) * r_start
                y_start = math.sin(ang_rad) * r_start
                # end point a bit further out
                r_text = 1.25
                x_text = math.cos(ang_rad) * r_text
                y_text = math.sin(ang_rad) * r_text
                ax.plot([x_start, x_text], [y_start, y_text], color='#777777', linewidth=0.9)
                align = 'left' if x_text >= 0 else 'right'
                ax.text(x_text, y_text, f"{pct:.2f}%", ha=align, va='center', fontsize=base_fontsize - 0.5, color='#333333')
        ax.legend(
            wedges, breakdown['Hugo_Symbol'], title='Gene', loc='center left',
            bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=max(8, base_fontsize - 0.5),
        )
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=8)
        try:
            fig.subplots_adjust(top=0.94)
        except Exception:
            pass
    else:
        y_pos = range(n_items)
        ax.barh(
            y_pos, breakdown['n_variants'].values[::-1], color=colors[::-1],
            edgecolor='white', linewidth=0.6, height=0.65, zorder=3,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(breakdown['Hugo_Symbol'].values[::-1], fontsize=base_fontsize + 0.5)
        max_val = breakdown['n_variants'].max()
        for i, (_, row) in enumerate(breakdown.iloc[::-1].iterrows()):
            ax.text(
                row['n_variants'] + max_val * 0.015, i,
                f"{int(row['n_variants']):,} ({row['pct_of_domain']:.2f}%)",
                va='center', fontsize=base_fontsize - 0.5, color='#333333',
            )
        ax.set_xlim(0, max_val * 1.25)
        ax.set_xlabel('Number of variants', fontsize=base_fontsize, labelpad=8)
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=6)
        ax.grid(axis='x', linestyle='--', alpha=0.35, zorder=0)
        ax.grid(axis='y', visible=False)
        ax.set_axisbelow(True)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='y', length=0)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax
