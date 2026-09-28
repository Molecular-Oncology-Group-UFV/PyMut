import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.pyMut.visualizations.pfam_plots import (
    plot_pfam_domains,
    plot_pfam_domain_breakdown,
)


@pytest.fixture
def summary_df():
    df = pd.DataFrame({
        "pfam_id": ["PF00001", "PF00002", "PF00003"],
        "pfam_name": ["Kinase", "SH2", "PTP"],
        "display_name": ["Kinase", "SH2", "PTP"],
        "n_variants": [30, 20, 10],
        "n_genes": [2, 1, 1],
        "pct_of_mapped": [50.0, 33.33, 16.67],
        "protein_breakdown": ["BRAF,ARAF", "SRC", "PTPN11"],
    })
    df.attrs = {}
    return df


@pytest.fixture
def breakdown_df():
    return pd.DataFrame({
        "Hugo_Symbol": ["BRAF", "ARAF", "KRAS"],
        "n_variants": [20, 10, 5],
        "pct_of_domain": [57.14, 28.57, 14.29],
    })


# ---------------------------------------------------------------------------
# plot_pfam_domains
# ---------------------------------------------------------------------------

def test_plot_pfam_domains_empty_raises():
    with pytest.raises(ValueError, match="Empty summary"):
        plot_pfam_domains(pd.DataFrame())


def test_plot_pfam_domains_returns_figure_with_one_bar_per_domain(summary_df):
    fig, ax = plot_pfam_domains(summary_df)
    try:
        assert isinstance(fig, plt.Figure)
        assert len(ax.patches) == len(summary_df)
        assert "Top 3 Pfam domains" in ax.get_title()
    finally:
        plt.close(fig)


def test_plot_pfam_domains_without_display_name_uses_pfam_id(summary_df):
    df = summary_df.drop(columns=["display_name"])
    fig, ax = plot_pfam_domains(df)
    try:
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert any("PF00001" in lab for lab in labels)
    finally:
        plt.close(fig)


def test_plot_pfam_domains_uses_coverage_subtitle_when_attrs_present(summary_df):
    summary_df.attrs = {"total_mapped": 60, "total_considered": 100}
    fig, ax = plot_pfam_domains(summary_df)
    try:
        texts = [t.get_text() for t in ax.texts]
        assert any("60/100" in t and "60.00%" in t for t in texts)
    finally:
        plt.close(fig)


def test_plot_pfam_domains_with_explicit_ax(summary_df):
    fig0, ax0 = plt.subplots()
    try:
        fig, ax = plot_pfam_domains(summary_df, ax=ax0)
        assert fig is fig0 and ax is ax0
    finally:
        plt.close(fig0)


def test_plot_pfam_domains_save_path(summary_df, tmp_path):
    out = tmp_path / "domains.png"
    fig, _ = plot_pfam_domains(summary_df, save_path=str(out))
    try:
        assert out.exists() and out.stat().st_size > 0
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_pfam_domain_breakdown
# ---------------------------------------------------------------------------

def test_plot_pfam_domain_breakdown_empty_raises(breakdown_df):
    with pytest.raises(ValueError, match="Empty breakdown"):
        plot_pfam_domain_breakdown(pd.DataFrame(), pfam_id="PF00001")


def test_plot_pfam_domain_breakdown_bar(breakdown_df):
    fig, ax = plot_pfam_domain_breakdown(breakdown_df, pfam_id="PF00001", kind="bar")
    try:
        assert isinstance(fig, plt.Figure)
        assert len(ax.patches) == len(breakdown_df)
        assert "PF00001" in ax.get_title()
    finally:
        plt.close(fig)


def test_plot_pfam_domain_breakdown_donut(breakdown_df):
    fig, ax = plot_pfam_domain_breakdown(breakdown_df, pfam_id="PF00001", kind="donut")
    try:
        assert isinstance(fig, plt.Figure)
        assert len(ax.patches) == len(breakdown_df)  # wedges
    finally:
        plt.close(fig)


def test_plot_pfam_domain_breakdown_unknown_kind_falls_back_to_bar(breakdown_df):
    fig, ax = plot_pfam_domain_breakdown(breakdown_df, pfam_id="PF00001", kind="weird")
    try:
        assert len(ax.patches) == len(breakdown_df)
    finally:
        plt.close(fig)


def test_plot_pfam_domain_breakdown_save_path(breakdown_df, tmp_path):
    out = tmp_path / "breakdown.png"
    fig, _ = plot_pfam_domain_breakdown(breakdown_df, pfam_id="PF00001", save_path=str(out))
    try:
        assert out.exists() and out.stat().st_size > 0
    finally:
        plt.close(fig)
