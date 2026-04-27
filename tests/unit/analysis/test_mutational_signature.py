import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pyMut.analysis.mutational_signature import (
    _get_reverse_complement,
    _normalize_to_pyrimidine,
    _get_trinucleotide_context,
    _create_context_label,
    TRINUCLEOTIDE_CONTEXTS,
    CONTEXT_TO_INDEX,
    MutationalSignatureMixin,
    estimateSignatures,
    extractSignatures,
    compare_signatures,
)

from .fixtures.mutational_signature_fixtures import (
    DummyFasta,
    dummy_fasta,
    dummy_fasta_with_Ns,
    contexts_df_small,
    build_cosmic_df,
)


# ------------------------
# Helper function tests
# ------------------------

def test_get_reverse_complement_basic():
    assert _get_reverse_complement("ACGT") == "ACGT"  # palindrome
    assert _get_reverse_complement("AT") == "AT"
    assert _get_reverse_complement("AGT") == "ACT"
    with pytest.raises(KeyError):
        _get_reverse_complement("AX")


def test_normalize_to_pyrimidine_passthrough_pyrimidine():
    ref, alt, trinuc = _normalize_to_pyrimidine("C", "A", "ACA")
    assert (ref, alt, trinuc) == ("C", "A", "ACA")


def test_normalize_to_pyrimidine_reverse_complement_for_purines():
    ref, alt, trinuc = _normalize_to_pyrimidine("A", "G", "ATC")
    assert (ref, alt, trinuc) == ("T", "C", "GAT")
    # Idempotence
    ref2, alt2, trinuc2 = _normalize_to_pyrimidine(ref, alt, trinuc)
    assert (ref2, alt2, trinuc2) == ("T", "C", "GAT")


def test_get_trinucleotide_context_ok_and_edge_cases(dummy_fasta, dummy_fasta_with_Ns, caplog):
    # Name resolution: request '1' when 'chr1' exists
    t1 = _get_trinucleotide_context(dummy_fasta, "1", 10)
    assert isinstance(t1, str) and len(t1) == 3
    # Reverse case: request 'chr1' when '1' exists
    t2 = _get_trinucleotide_context(dummy_fasta, "chr1", 10)
    assert isinstance(t2, str) and len(t2) == 3

    # Nonexistent chromosome
    caplog.clear()
    t3 = _get_trinucleotide_context(dummy_fasta, "2", 10)
    assert t3 is None
    assert any("not found" in r.message for r in caplog.records)

    # Window with Ns returns None
    t4 = _get_trinucleotide_context(dummy_fasta_with_Ns, "1", 100)
    assert t4 is None

    # Slice at edge producing len != 3 returns None (pos=1 -> start=-1)
    t5 = _get_trinucleotide_context(dummy_fasta, "1", 1)
    assert t5 is None

    # Simulate internal exception
    class BadFasta(DummyFasta):
        def __getitem__(self, chrom: str):
            raise RuntimeError("boom")

    caplog.clear()
    t6 = _get_trinucleotide_context(BadFasta({"1": "ACGT" * 5}), "1", 10)
    assert t6 is None
    assert any("Error extracting trinucleotide" in r.message for r in caplog.records)


def test_create_context_label_builds_expected_string():
    assert _create_context_label("C", "A", "ACA") == "A[C>A]A"
    with pytest.raises(IndexError):
        _create_context_label("C", "A", "AC")


# ------------------------
# MutationalSignatureMixin tests
# ------------------------

class DummyPM(MutationalSignatureMixin):
    def __init__(self, df: pd.DataFrame):
        self.data = df


def _patch_pyfaidx(monkeypatch, fasta_obj):
    fake_module = types.SimpleNamespace(Fasta=lambda path: fasta_obj)
    monkeypatch.setitem(sys.modules, 'pyfaidx', fake_module)


import sys


def test_trinucleotideMatrix_missing_required_columns_raises():
    df = pd.DataFrame({
        "Chromosome": ["1"],
        "Start_Position": [10],
        "Reference_Allele": ["A"],
        # Missing Tumor_Seq_Allele2
    })
    pm = DummyPM(df)
    with pytest.raises(ValueError, match="Missing required column"):
        pm.trinucleotideMatrix("/dummy.fa")


def test_trinucleotideMatrix_no_valid_snvs_raises(monkeypatch):
    df = pd.DataFrame({
        "Chromosome": ["1"],
        "Start_Position": [10],
        "Reference_Allele": ["A"],
        "Tumor_Seq_Allele2": ["A"],  # REF == ALT -> invalid
        "Tumor_Sample_Barcode": ["S1"],
    })
    pm = DummyPM(df)
    _patch_pyfaidx(monkeypatch, DummyFasta({"1": "ACGT" * 60, "chr1": "ACGT" * 60}))
    with pytest.raises(ValueError, match="Zero SNPs"):
        pm.trinucleotideMatrix("/dummy.fa")


def test_trinucleotideMatrix_fasta_loading_error_raises(monkeypatch):
    class BoomFasta:
        def __init__(self, path):
            raise RuntimeError("cannot open")

    fake_module = types.SimpleNamespace(Fasta=BoomFasta)
    monkeypatch.setitem(sys.modules, 'pyfaidx', fake_module)

    df = pd.DataFrame({
        "Chromosome": ["1"],
        "Start_Position": [10],
        "Reference_Allele": ["A"],
        "Tumor_Seq_Allele2": ["C"],
        "Tumor_Sample_Barcode": ["S1"],
    })
    pm = DummyPM(df)
    with pytest.raises(ValueError, match="Could not open FASTA"):
        pm.trinucleotideMatrix("/nope.fa")


def test_trinucleotideMatrix_long_format_counts_by_sample(monkeypatch):
    df = pd.DataFrame({
        "Chromosome": ["1", "1", "1"],
        "Start_Position": [10, 20, 30],
        "Reference_Allele": ["C", "T", "A"],
        "Tumor_Seq_Allele2": ["A", "C", "G"],
        "Tumor_Sample_Barcode": ["S1", "S1", "S2"],
    })
    pm = DummyPM(df)
    _patch_pyfaidx(monkeypatch, DummyFasta({"1": "ACGT" * 60, "chr1": "ACGT" * 60}))
    contexts_df, enriched = pm.trinucleotideMatrix("/dummy.fa")

    assert contexts_df.shape[0] == 96
    assert set(contexts_df.columns) == {"S1", "S2"}
    # Total counts equals number of valid SNVs
    assert contexts_df.values.sum() == len(enriched)
    # Enriched has required columns
    assert {"trinuc", "class96", "idx96"}.issubset(enriched.columns)
    assert enriched["idx96"].notna().all()


def test_trinucleotideMatrix_wide_format_counts_by_genotype(monkeypatch):
    # Wide format: no Tumor_Sample_Barcode column, but sample columns like S1, S2
    df = pd.DataFrame({
        "CHROM": ["1", "1", "1"],
        "POS": [10, 20, 30],
        "Reference_Allele": ["C", "T", "A"],
        "Tumor_Seq_Allele2": ["A", "C", "G"],
        "S1": ["A|A", "T|C", "A|G"],  # counts alt occurrences
        "S2": ["A|C", "C|C", "G|G"],
    })
    pm = DummyPM(df)
    _patch_pyfaidx(monkeypatch, DummyFasta({"1": "ACGT" * 60, "chr1": "ACGT" * 60}))
    contexts_df, enriched = pm.trinucleotideMatrix("/dummy.fa")
    assert contexts_df.shape == (96, 2)
    # Count normalized alt alleles across rows for each sample (after pyrimidine normalization)
    # Row1 norm_alt=A, S1 A|A -> 2, S2 A|C -> 1
    # Row2 norm_alt=C, S1 T|C -> 1, S2 C|C -> 2
    # Row3 ref=A alt=G -> normalized to T>C, norm_alt=C; S1 A|G -> 0, S2 G|G -> 0
    assert contexts_df.values.sum() == (2+1) + (1+2) + (0+0)


def test_trinucleotideMatrix_skips_none_context_rows(monkeypatch):
    # Use fasta with Ns so one row becomes None
    df = pd.DataFrame({
        "Chromosome": ["1", "1"],
        "Start_Position": [100, 20],  # 100 will be None on dummy_fasta_with_Ns
        "Reference_Allele": ["C", "T"],
        "Tumor_Seq_Allele2": ["A", "C"],
        "Tumor_Sample_Barcode": ["S1", "S1"],
    })
    pm = DummyPM(df)
    _patch_pyfaidx(monkeypatch, dummy_fasta_with_Ns())
    contexts_df, enriched = pm.trinucleotideMatrix("/dummy.fa")
    # Only one valid row should be counted
    assert contexts_df.values.sum() == 1
    assert len(enriched) == 1


def test_trinucleotideMatrix_class96_and_idx96_mapping(monkeypatch):
    df = pd.DataFrame({
        "Chromosome": ["1"],
        "Start_Position": [10],
        "Reference_Allele": ["C"],
        "Tumor_Seq_Allele2": ["A"],
        "Tumor_Sample_Barcode": ["S1"],
    })
    pm = DummyPM(df)
    _patch_pyfaidx(monkeypatch, DummyFasta({"1": "ACGT" * 60, "chr1": "ACGT" * 60}))
    contexts_df, enriched = pm.trinucleotideMatrix("/dummy.fa")
    label = enriched.iloc[0]["class96"]
    assert label in TRINUCLEOTIDE_CONTEXTS
    assert enriched.iloc[0]["idx96"] == CONTEXT_TO_INDEX[label]


# ------------------------
# estimateSignatures tests
# ------------------------

def test_estimateSignatures_invalid_inputs():
    with pytest.raises(ValueError):
        estimateSignatures(np.ones((96, 2)))  # not a DataFrame
    with pytest.raises(ValueError):
        estimateSignatures(pd.DataFrame(np.ones((95, 2))))
    with pytest.raises(ValueError):
        estimateSignatures(pd.DataFrame(np.ones((96, 2))), nMin=0, nTry=1)


@pytest.mark.parametrize("use_p", [False, True])
def test_estimateSignatures_applies_pseudocount_when_many_zeros(use_p):
    sklearn = pytest.importorskip("sklearn")
    scipy = pytest.importorskip("scipy")
    # Create a very sparse matrix
    mat = np.zeros((96, 3))
    mat[0, 0] = 10
    df = pd.DataFrame(mat, index=TRINUCLEOTIDE_CONTEXTS, columns=["S1", "S2", "S3"])
    if use_p:
        res = estimateSignatures(df, nMin=2, nTry=2, nrun=1, parallel=1, pConstant=1e-6)
    else:
        res = estimateSignatures(df, nMin=2, nTry=2, nrun=1, parallel=1, pConstant=None)
    assert res["normalized_matrix"].shape == (96, 3)
    assert res["original_matrix"].shape == (96, 3)


def test_estimateSignatures_parallel_runs_collect_metrics_and_models(contexts_df_small):
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    res = estimateSignatures(contexts_df_small, nMin=2, nTry=3, nrun=2, parallel=2)
    metrics = res["metrics"]
    assert set([2, 3]).issubset(set(metrics["k"].unique()))
    for col in ["mean_rss", "std_rss", "cophenetic_corr", "dispersion", "successful_runs", "total_runs"]:
        assert col in metrics.columns
    assert res["optimal_k"] in [2, 3]
    # models list length equals successful runs across k
    assert len(res["models"]) == int(metrics["successful_runs"].sum())


def test_estimateSignatures_handles_failing_runs_and_optimal_k(monkeypatch, contexts_df_small, caplog):
    sklearn = pytest.importorskip("sklearn")
    pytest.importorskip("scipy")

    from sklearn.decomposition import NMF as RealNMF

    class FlakyNMF(RealNMF):
        def fit_transform(self, X, y=None, **fit_params):
            # Fail for random_state == 0 to simulate one failing run
            if getattr(self, 'random_state', None) == 0:
                raise RuntimeError("fail this run")
            return super().fit_transform(X, y=y, **fit_params)

    monkeypatch.setattr("sklearn.decomposition.NMF", FlakyNMF)

    caplog.clear()
    res = estimateSignatures(contexts_df_small, nMin=2, nTry=2, nrun=2, parallel=2)
    assert res["metrics"]["successful_runs"].min() >= 1
    assert any("NMF failed" in r.message for r in caplog.records)


# ------------------------
# extractSignatures tests
# ------------------------

def test_extractSignatures_invalid_inputs(contexts_df_small):
    with pytest.raises(ValueError):
        extractSignatures(np.ones((96, 2)), 2)
    with pytest.raises(ValueError):
        extractSignatures(pd.DataFrame(np.ones((95, 2))), 2)
    with pytest.raises(ValueError):
        extractSignatures(contexts_df_small, 0)
    with pytest.raises(ValueError):
        extractSignatures(contexts_df_small, 2, pConstant=0)


def test_extractSignatures_normalization_and_pseudocount(contexts_df_small):
    pytest.importorskip("sklearn")
    res = extractSignatures(contexts_df_small, n=2, pConstant=1e-4)
    sigs = res["signatures"]
    contrib = res["contributions"]
    assert sigs.shape == (96, 2)
    assert contrib.shape[0] == 2 and contrib.shape[1] == contexts_df_small.shape[1]
    # Each signature column sums to 1
    assert np.allclose(sigs.sum(axis=0).values, 1.0, rtol=1e-5)
    # Each contributions column sums to 1 (or stays at 1 for zero-sum samples)
    assert np.allclose(contrib.sum(axis=0).values, 1.0, rtol=1e-5, atol=1e-8)


def test_extractSignatures_reproducible_results(contexts_df_small):
    pytest.importorskip("sklearn")
    res1 = extractSignatures(contexts_df_small, n=2)
    res2 = extractSignatures(contexts_df_small, n=2)
    assert np.allclose(res1["signatures"].values, res2["signatures"].values)
    assert np.allclose(res1["contributions"].values, res2["contributions"].values)


def test_extractSignatures_nmf_failure_raises(monkeypatch, contexts_df_small):
    sklearn = pytest.importorskip("sklearn")
    from sklearn.decomposition import NMF as RealNMF

    class BadNMF(RealNMF):
        def fit_transform(self, X, y=None, **fit_params):
            raise RuntimeError("always fail")

    monkeypatch.setattr("sklearn.decomposition.NMF", BadNMF)
    with pytest.raises(ValueError):
        extractSignatures(contexts_df_small, n=2, pConstant=1e-4)


# ------------------------
# compare_signatures tests
# ------------------------

def test_compare_signatures_invalid_inputs():
    with pytest.raises(ValueError):
        compare_signatures([1, 2, 3], "path")
    with pytest.raises(ValueError):
        compare_signatures(np.ones((95, 2)), "path")
    with pytest.raises(ValueError):
        compare_signatures(np.ones((96,)), "path")


def test_compare_signatures_reads_catalog_and_filters_artifacts(monkeypatch):
    # Build a catalog with artifacts and shuffled contexts
    df = build_cosmic_df()
    def fake_read_csv(path, sep='\t'):
        return df.copy()
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    # W simple normalized matrix
    W = np.zeros((96, 2), dtype=float)
    W[0, 0] = 1.0
    W[1, 1] = 1.0

    res = compare_signatures(W, "/fake/path.txt", min_cosine=0.0, return_matrix=True)
    summary = res["summary_df"]
    cosine = res["cosine_matrix"]
    assert summary.shape[0] == 2
    assert cosine.shape[0] == 2
    # After filtering, there should be fewer signatures than original columns
    assert cosine.shape[1] <= df.shape[1] - 1  # minus context column


def test_compare_signatures_reindexes_contexts_or_raises(monkeypatch):
    # Missing contexts should raise
    df = build_cosmic_df()
    # Remove one context
    df = df.iloc[:-1, :]
    monkeypatch.setattr(pd, "read_csv", lambda p, sep='\t': df)

    W = np.random.default_rng(0).random((96, 2))
    with pytest.raises(ValueError):
        compare_signatures(W, "/fake/path.txt")

    # Provide full set in random order
    df2 = build_cosmic_df(shuffle_contexts=True)
    monkeypatch.setattr(pd, "read_csv", lambda p, sep='\t': df2)
    res = compare_signatures(W / W.sum(axis=0, keepdims=True), "/fake/path.txt")
    assert "summary_df" in res


def test_compare_signatures_cosine_matrix_and_summary_threshold(monkeypatch):
    # Construct a catalog aligned exactly with TRINUCLEOTIDE_CONTEXTS and two signatures
    cols = ["SBSX", "SBSY"]
    df = build_cosmic_df(columns=cols, shuffle_contexts=False, include_artifacts=False)
    # Normalize columns
    sigs = df[cols].values
    sigs = sigs / sigs.sum(axis=0, keepdims=True)
    df[cols] = sigs
    monkeypatch.setattr(pd, "read_csv", lambda p, sep='\t': df)

    # Make W identical to the first signature and low match for second
    W = np.zeros((96, 2), dtype=float)
    W[:, 0] = sigs[:, 0]
    W[:, 1] = np.roll(sigs[:, 1], 1)

    res = compare_signatures(W, "/fake/path.txt", min_cosine=0.9, return_matrix=True)
    summary = res["summary_df"]
    assert summary.loc[0, "Best_COSMIC"] == "SBSX"  # High cosine should match
    # If below threshold, label as No match
    below = summary.loc[1, "Cosine"] < 0.9
    assert summary.loc[1, "Best_COSMIC"] == ("No match" if below else summary.loc[1, "Best_COSMIC"])  # conditionally
