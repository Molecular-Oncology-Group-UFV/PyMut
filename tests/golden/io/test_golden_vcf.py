import os
import pandas as pd
from pathlib import Path

import pytest

from src.pyMut.input import read_vcf
from src.pyMut.core import PyMutation


GOLDEN_DIR = Path(__file__).parent
EXPECTED_VCF = GOLDEN_DIR / "expected_vcf.parquet"


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
    # All other columns (samples) to string for deterministic compare
    for c in out.columns:
        if c not in ("CHROM", "POS", "REF", "ALT"):
            out[c] = out[c].astype(str)
    return out.sort_values(cols).reset_index(drop=True)


@pytest.mark.unit
@pytest.mark.parametrize("assembly", ["38"])  # Golden requires GRCh38 for this fixture
def test_golden_vcf_against_parquet(assembly):
    # Input VCF (gz) fixture required by the user
    base_dir = Path(__file__).parents[2] / "unit" / "io" / "fixtures" / "data"
    gz_path = base_dir / (
        "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf.gz"
    )
    assert gz_path.exists(), f"Fixture missing: {gz_path}"

    pm = read_vcf(gz_path, assembly=assembly)

    # Basic invariants
    assert isinstance(pm, PyMutation)
    assert pm.metadata.source_format == "VCF"
    assert pm.metadata.assembly == assembly

    # Build normalized view for golden
    core = ["CHROM", "POS", "REF", "ALT"]
    for col in core:
        assert col in pm.data.columns, f"Missing required column: {col}"
    # include sample columns if present
    sample_cols = [c for c in pm.samples if c in pm.data.columns]
    cols = core + sorted(sample_cols)

    actual = _normalize_for_compare(pm.data, cols)

    # Load expected parquet (must exist; no auto-generation)
    assert EXPECTED_VCF.exists(), f"Expected parquet missing: {EXPECTED_VCF}"
    expected = pd.read_parquet(EXPECTED_VCF)
    # Re-normalize expected in case of backend dtype differences
    expected = _normalize_for_compare(expected, cols)

    # Schema/columns check
    assert list(expected.columns) == cols, (
        "Expected parquet columns differ from actual output columns. Update the snapshot explicitly if the change is intentional."
    )

    # Equality check (order-insensitive thanks to normalization/sort)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_like=False)
