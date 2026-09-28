import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.pyMut.visualizations import tcga_compare_plot as tcc


# ---------------------------------------------------------------------------
# Doubles: a minimal object exposing calculate_tmb_analysis (schema:
# analysis[Tumor_Sample_Barcode, total]), like the real PyMutation.
# ---------------------------------------------------------------------------

class _TmbPM:
    def __init__(self, totals):
        self._analysis = pd.DataFrame({
            "Tumor_Sample_Barcode": [f"S{i}" for i in range(len(totals))],
            "total": list(totals),
        })

    def calculate_tmb_analysis(self, save_files=False):
        return {"analysis": self._analysis.copy()}


@pytest.fixture
def small_tcga(monkeypatch):
    """Replace the bundled TCGA table with a tiny deterministic one."""
    table = pd.DataFrame({
        "cohort": ["ACC", "ACC", "BRCA", "BRCA"],
        "Tumor_Sample_Barcode": ["a", "b", "c", "d"],
        "total": [5, 15, 20, 40],
    })

    def _load(primary_site=False, tcga_cohorts=None):
        df = table.copy()
        if tcga_cohorts:
            df = df[df["cohort"].isin(tcga_cohorts)]
        return df.reset_index(drop=True)

    monkeypatch.setattr(tcc, "load_tcga_cohort_table", _load)
    return table


# ---------------------------------------------------------------------------
# load_tcga_cohort_table (against the real bundled resource)
# ---------------------------------------------------------------------------

def test_load_tcga_cohort_table_defaults():
    df = tcc.load_tcga_cohort_table()
    assert set(df.columns) == {"cohort", "Tumor_Sample_Barcode", "total"}
    assert len(df) > 0


def test_load_tcga_cohort_table_primary_site():
    df = tcc.load_tcga_cohort_table(primary_site=True)
    assert set(df.columns) == {"cohort", "Tumor_Sample_Barcode", "total"}


def test_load_tcga_cohort_table_filter_valid():
    full = tcc.load_tcga_cohort_table()
    some = sorted(full["cohort"].unique())[:2]
    df = tcc.load_tcga_cohort_table(tcga_cohorts=some)
    assert set(df["cohort"].unique()) == set(some)


def test_load_tcga_cohort_table_filter_unknown_raises():
    with pytest.raises(ValueError, match="No matching TCGA cohorts"):
        tcc.load_tcga_cohort_table(tcga_cohorts=["NOT_A_COHORT"])


# ---------------------------------------------------------------------------
# Statistical helpers (pure functions)
# ---------------------------------------------------------------------------

def test_remove_boxplot_outliers():
    values = pd.Series(list(range(1, 11)) + [100])
    kept = tcc._remove_boxplot_outliers(values)
    assert 100 not in kept.values
    assert len(kept) == 10


def test_bh_adjust_single_pvalue_unchanged():
    np.testing.assert_allclose(tcc._bh_adjust(np.array([0.05])), [0.05])


def test_bh_adjust_equal_pvalues():
    np.testing.assert_allclose(tcc._bh_adjust(np.array([0.1, 0.1])), [0.1, 0.1])


def test_bh_adjust_bounded_and_ordered():
    adj = tcc._bh_adjust(np.array([0.01, 0.4, 0.02]))
    assert (adj >= 0).all() and (adj <= 1).all()
    # BH is monotonic in the input order: smaller p -> smaller adjusted
    assert adj[0] <= adj[2] <= adj[1]


def test_pairwise_t_test_structure_and_ordering():
    cohorts = {
        "A": np.array([1.0, 2.0, 3.0]),
        "B": np.array([2.0, 3.0, 4.0]),
        "C": np.array([10.0, 11.0, 12.0]),
    }
    res = tcc._pairwise_t_test(cohorts)
    assert len(res) == 3                     # k*(k-1)/2 pairs
    assert list(res.columns) == ["Cohort1", "Cohort2", "Pval"]
    assert (res["Pval"].dropna().between(0, 1)).all()
    # A vs C must be more significant than A vs B, and table sorted by p
    p_ab = res.set_index(["Cohort1", "Cohort2"]).loc[("A", "B"), "Pval"]
    p_ac = res.set_index(["Cohort1", "Cohort2"]).loc[("A", "C"), "Pval"]
    assert p_ac < p_ab
    assert list(res["Pval"]) == sorted(res["Pval"])


# ---------------------------------------------------------------------------
# compute_tcga_compare
# ---------------------------------------------------------------------------

def test_compute_tcga_compare_tables(small_tcga):
    pm = _TmbPM([10, 0, 30])   # one zero-mutation sample (removed by default)
    res = tcc.compute_tcga_compare(pm, cohort_name="MyCohort", capture_size=50.0)

    assert set(res) == {"median_mutation_burden", "mutation_burden_perSample",
                        "pairwise_t_test", "plot_data", "cohort_order"}

    med = res["median_mutation_burden"].set_index("Cohort")
    assert "MyCohort" in med.index
    assert med.loc["MyCohort", "Cohort_Size"] == 2          # zero sample removed
    # input normalized by capture_size=50: median of [0.2, 0.6]
    assert med.loc["MyCohort", "Median_Mutations"] == pytest.approx(0.4)
    # ACC normalized by the TCGA default (35.8): median of [5, 15]/35.8
    assert med.loc["ACC", "Median_Mutations"] == pytest.approx((10 / 35.8))

    per_sample = res["mutation_burden_perSample"]
    assert "total_perMB" in per_sample.columns
    assert not (per_sample["total"] == 0).any()

    pairwise = res["pairwise_t_test"]
    assert not pairwise.empty
    assert (pairwise["Pval"].dropna().between(0, 1)).all()

    assert set(res["plot_data"]["TCGA"].unique()) == {"Input", "TCGA"}


def test_compute_tcga_compare_all_zero_raises(small_tcga):
    pm = _TmbPM([0, 0])
    with pytest.raises(ValueError, match="No samples with mutations"):
        tcc.compute_tcga_compare(pm, cohort_name="Empty")


def test_compute_tcga_compare_no_normalization(small_tcga):
    pm = _TmbPM([10, 20])
    res = tcc.compute_tcga_compare(pm, cohort_name="Raw", capture_size=None)
    med = res["median_mutation_burden"].set_index("Cohort")
    assert med.loc["Raw", "Median_Mutations"] == pytest.approx(15.0)
    assert "total_perMB" not in res["mutation_burden_perSample"].columns


# ---------------------------------------------------------------------------
# _create_tcga_compare_plot (smoke test with the double + patched table)
# ---------------------------------------------------------------------------

def test_create_tcga_compare_plot_returns_figure(small_tcga):
    pm = _TmbPM([10, 30])
    fig = tcc._create_tcga_compare_plot(pm, cohort_name="MyCohort",
                                        tcga_cohorts=["ACC", "BRCA"])
    try:
        assert isinstance(fig, Figure)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
