import pandas as pd
import pytest
from pathlib import Path

from src.pyMut.input import read_maf
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


@pytest.mark.parametrize("assembly", ["37"])  # assembly fixed to 37
def test_roundtrip_maf_variants_and_samples(assembly, tmp_path):
    # Arrange
    fixture_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"

    pm1 = read_maf(fixture_path, assembly=assembly)
    assert isinstance(pm1, PyMutation)
    assert pm1.metadata.source_format == "MAF"

    # Act: export to MAF and re-read
    out_path = tmp_path / "roundtrip_out.maf"
    pm1.to_maf(out_path)
    pm2 = read_maf(out_path, assembly=assembly)

    # Assert: types and metadata
    assert isinstance(pm2, PyMutation)
    assert pm2.metadata.source_format == "MAF"
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

    # Per-sample distribution (representation-agnostic):
    # Prefer explicit sample column if available (e.g., Tumor_Sample_Barcode or 'SAMPLE')
    sample_cols = [c for c in ["SAMPLE", "Tumor_Sample_Barcode"] if c in pm1.data.columns and c in pm2.data.columns]
    if sample_cols:
        sc = sample_cols[0]
        c1 = pm1.data.groupby(sc).size().sort_index()
        c2 = pm2.data.groupby(sc).size().sort_index()
        # Normalize index types (e.g., string vs string[pyarrow]) and names for stable comparison
        c1.index = c1.index.astype(str)
        c2.index = c2.index.astype(str)
        pd.testing.assert_series_equal(c1.sort_index(), c2.sort_index(), check_dtype=False, check_names=False)
    else:
        # Minimal fallback: ensure sample count and row count are unchanged
        assert len(pm1.samples) == len(pm2.samples)
        assert len(pm1.data) == len(pm2.data)
