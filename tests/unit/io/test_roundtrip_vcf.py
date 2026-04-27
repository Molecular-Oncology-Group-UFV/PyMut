import pandas as pd
import pytest
from pathlib import Path

from src.pyMut.input import read_vcf
from src.pyMut.core import PyMutation


def _normalize_for_compare(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()

    def _norm_chrom_val(v: str) -> str:
        v = str(v)
        if v.lower().startswith("chr"):
            v = v[3:]
        v = v.upper()
        if v == "MT":
            v = "M"
        return v

    if "CHROM" in out.columns:
        out["CHROM"] = out["CHROM"].map(_norm_chrom_val)
    if "POS" in out.columns:
        out["POS"] = pd.to_numeric(out["POS"], errors="coerce").astype("Int64")
    if "REF" in out.columns:
        out["REF"] = out["REF"].astype(str)
    if "ALT" in out.columns:
        out["ALT"] = out["ALT"].astype(str)
    return out.sort_values(cols).reset_index(drop=True)


def _align_on_variant_key(df: pd.DataFrame) -> pd.DataFrame:
    # Create a stable key for aligning rows across round-trips
    key = (
        df["CHROM"].astype(str).str.upper().str.replace("^CHR", "", regex=True).str.replace("^MT$", "M", regex=True)
        + ":" + df["POS"].astype("Int64").astype(str)
        + ":" + df["REF"].astype(str)
        + ":" + df["ALT"].astype(str)
    )
    out = df.copy()
    out["__key__"] = key
    return out.set_index("__key__", drop=True)


@pytest.mark.parametrize("assembly", ["38"])  # assembly fixed to GRCh38
def test_roundtrip_vcf_basic(assembly, tmp_path):
    # Arrange
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "data"
        / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
    )

    pm1 = read_vcf(fixture_path, assembly=assembly)
    assert isinstance(pm1, PyMutation)
    assert pm1.metadata.source_format == "VCF"

    # Act: export to VCF and re-read
    out_path = tmp_path / "roundtrip_out.vcf"
    pm1.to_vcf(out_path)
    pm2 = read_vcf(out_path, assembly=assembly)

    # Assert: types and metadata
    assert isinstance(pm2, PyMutation)
    assert pm2.metadata.source_format == "VCF"
    assert pm2.metadata.assembly == assembly

    # Ensure required VCF-like columns exist
    for col in ["CHROM", "POS", "REF", "ALT"]:
        assert col in pm1.data.columns, f"Missing column in first read: {col}"
        assert col in pm2.data.columns, f"Missing column in second read: {col}"

    # Variant identity (order-insensitive)
    core = ["CHROM", "POS", "REF", "ALT"]
    v1 = _normalize_for_compare(pm1.data, core)
    v2 = _normalize_for_compare(pm2.data, core)
    pd.testing.assert_frame_equal(v1, v2, check_like=True)

    # Sample set equality
    assert set(pm1.samples) == set(pm2.samples)

    # Align both DataFrames by variant key for per-sample genotype comparisons
    a1 = _align_on_variant_key(pm1.data)
    a2 = _align_on_variant_key(pm2.data)

    # Ensure same set of variants/keys
    assert set(a1.index) == set(a2.index)

    # Sort by key for deterministic comparison
    a1 = a1.sort_index()
    a2 = a2.sort_index()

    # Per-sample genotype equality and phasing preservation
    # Compare only for samples present in both (should be identical sets)
    common_samples = [s for s in pm1.samples if s in pm2.samples]
    assert common_samples, "No sample columns detected for VCF genotypes"

    # Compare genotype strings exactly (representation is allele-based in this library)
    for s in common_samples:
        assert s in a1.columns and s in a2.columns, f"Sample column missing after round-trip: {s}"
        # Normalize dtype to string for stable comparison
        s1 = a1[s].astype(str)
        s2 = a2[s].astype(str)
        pd.testing.assert_series_equal(s1, s2, check_names=False)

    # Phasing check: if any phased genotypes '|' exist originally, they should still exist
    phased_count_1 = sum(a1[s].astype(str).str.contains("|", regex=False).sum() for s in common_samples)
    phased_count_2 = sum(a2[s].astype(str).str.contains("|", regex=False).sum() for s in common_samples)
    if phased_count_1 > 0:
        assert phased_count_2 > 0, "Phased genotypes lost after round-trip"
        assert phased_count_1 == phased_count_2, "Number of phased genotype entries changed after round-trip"

    # Global invariants
    assert len(pm1.data) == len(pm2.data)
    assert len(pm1.samples) == len(pm2.samples)

    # Total number of non-missing genotypes preserved ('.' = missing)
    def _non_missing_count(df: pd.DataFrame, samples: list[str]) -> int:
        cnt = 0
        for s in samples:
            if s in df.columns:
                col = df[s].astype(str)
                cnt += (col != ".").sum()
        return int(cnt)

    assert _non_missing_count(pm1.data, common_samples) == _non_missing_count(pm2.data, common_samples)
