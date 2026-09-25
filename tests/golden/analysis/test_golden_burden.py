import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pyMut.input import read_vcf, read_maf

# The MutationBurden functionality is provided as a mixin on the returned
# PyMutation object via the method `calculate_tmb_analysis`.
#
# NOTE (API update): the analysis dataframe now uses the maftools-style
# schema:
#   Tumor_Sample_Barcode, total, total_perMB, total_perMB_log
# where `total` counts NON-SYNONYMOUS mutations (old snapshots with the
# Sample/Total_Mutations/... schema must be regenerated: delete the folder's
# analysis.parquet + meta.json and rerun this test).

GOLDEN_ROOT = Path(__file__).parent
VCF_DIR = GOLDEN_ROOT / "vcf_burden"
MAF_DIR = GOLDEN_ROOT / "maf_burden"

SAMPLE_COL = "Tumor_Sample_Barcode"


def _resolve_input(assembly: str) -> str:
    data_path = os.environ.get(
        "PYMUT_VCF_PATH" if assembly == "38" else "PYMUT_MAF_PATH",
        None,
    )
    if data_path is None:
        if assembly == "38":
            data_path = "tests/unit/io/fixtures/data/test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        else:
            data_path = "tests/unit/io/fixtures/data/test_tcga_laml_100variants.maf"
    assert Path(data_path).exists(), f"Missing input file: {data_path}"
    return data_path


def _run_tmb(loader, data_path: str, assembly: str, genome_size_bp: int, variant_classification_column):
    pm = loader(data_path, assembly=assembly)
    return pm.calculate_tmb_analysis(
        variant_classification_column=variant_classification_column,
        genome_size_bp=genome_size_bp,
        save_files=False,
    )["analysis"]


def _load_expected(expected_dir: Path):
    """Load expected analysis parquet and optional meta.json. Skip if missing."""
    analysis_path = expected_dir / "analysis.parquet"
    meta_path = expected_dir / "meta.json"

    if not analysis_path.exists():
        pytest.skip(
            f"Golden snapshots not present in {expected_dir}. "
            "Delete stale snapshots to regenerate them with the current implementation, "
            "inspect the result, and commit analysis.parquet (and meta.json)."
        )

    analysis_exp = pd.read_parquet(analysis_path)

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

    return analysis_exp, meta


def _assert_df_equal_numeric(actual: pd.DataFrame, expected: pd.DataFrame, rtol=1e-12, atol=1e-12):
    assert list(actual.index) == list(expected.index), "Row index mismatch"
    assert list(actual.columns) == list(expected.columns), "Columns mismatch"
    np.testing.assert_allclose(actual.values.astype(float), expected.values.astype(float), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "loader, expected_dir, assembly",
    [
        (read_vcf, VCF_DIR, "38"),
        (read_maf, MAF_DIR, "37"),
    ],
)
def test_golden_mutation_burden(loader, expected_dir, assembly):
    expected_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = expected_dir / "analysis.parquet"
    meta_path = expected_dir / "meta.json"

    data_path = _resolve_input(assembly)

    # Frozen parameters
    genome_size_bp_default = 60456963  # WES default

    if not analysis_path.exists():
        # REGENERATION MODE: produce the snapshot with the current
        # implementation. calculate_tmb_analysis is deterministic for a
        # frozen input + parameters.
        res_df = _run_tmb(loader, data_path, assembly, genome_size_bp_default, None)
        res_df.to_parquet(analysis_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({
                "genome_size_bp": genome_size_bp_default,
                "variant_classification_column": None,
                "schema": [SAMPLE_COL, "total", "total_perMB", "total_perMB_log"],
            }, fh, indent=2)
        # Minimal sanity: non-negative counts and TMB
        assert (res_df["total"].values >= 0).all()
        assert (res_df["total_perMB"].values >= 0).all()
        return

    # COMPARISON MODE
    analysis_exp, meta = _load_expected(expected_dir)

    genome_size_bp = int(meta.get("genome_size_bp", genome_size_bp_default))
    variant_classification_column = meta.get("variant_classification_column")  # None -> autodetect

    analysis_act = _run_tmb(loader, data_path, assembly, genome_size_bp, variant_classification_column).copy()

    # Index by sample column (new schema: Tumor_Sample_Barcode; tolerate the
    # old 'Sample' name in stale snapshots so the error below is meaningful)
    sample_col_act = SAMPLE_COL if SAMPLE_COL in analysis_act.columns else "Sample"
    analysis_act = analysis_act.set_index(sample_col_act)
    if "Sample" in analysis_exp.columns:
        analysis_exp = analysis_exp.set_index("Sample")
    elif SAMPLE_COL in analysis_exp.columns:
        analysis_exp = analysis_exp.set_index(SAMPLE_COL)

    # Reorder rows to samples_order from meta for a 1:1 comparison
    samples_order = meta.get("samples_order")
    if samples_order is not None:
        missing = [s for s in samples_order if s not in analysis_act.index]
        assert not missing, f"Missing samples in actual analysis: {missing}"
        analysis_act = analysis_act.loc[samples_order]

    # Align columns to the expected schema
    analysis_act = analysis_act[analysis_exp.columns]

    assert analysis_act.shape == analysis_exp.shape, "Analysis shape changed; regenerate snapshots if intentional"

    # New schema: 'total' is an integer count; 'total_perMB'/'total_perMB_log' are floats
    int_cols = [c for c in analysis_exp.columns if c == "total"]
    float_cols = [c for c in analysis_exp.columns if c in ("total_perMB", "total_perMB_log")]

    if int_cols:
        _assert_df_equal_numeric(analysis_act[int_cols], analysis_exp[int_cols], rtol=0, atol=0)
    if float_cols:
        _assert_df_equal_numeric(analysis_act[float_cols], analysis_exp[float_cols], rtol=1e-9, atol=1e-12)

    # Invariants
    assert (analysis_act[int_cols].values >= 0).all(), "Counts must be non-negative"
    if "total_perMB" in analysis_act.columns:
        # Linear-scale TMB is non-negative
        assert (analysis_act["total_perMB"].values >= 0).all(), "TMB must be non-negative"
        # Log-scale TMB: log10(0) == -inf for zero-mutation samples (correct),
        # otherwise it must equal log10(total_perMB)
        permb = analysis_act["total_perMB"].values.astype(float)
        if "total_perMB_log" in analysis_act.columns:
            log_vals = analysis_act["total_perMB_log"].values.astype(float)
            zero_mask = permb == 0
            assert (log_vals[zero_mask] == -np.inf).all(), (
                "log10 TMB of zero-mutation samples must be -inf"
            )
            if (~zero_mask).any():
                np.testing.assert_allclose(
                    log_vals[~zero_mask],
                    np.log10(permb[~zero_mask]),
                    rtol=1e-9, atol=1e-12,
                )
        # total_perMB == total / genome_size_bp * 1e6
        if "total" in analysis_act.columns and genome_size_bp > 0:
            expected_tmb = analysis_act["total"].astype(float) / float(genome_size_bp) * 1_000_000
            np.testing.assert_allclose(
                analysis_act["total_perMB"].values,
                expected_tmb.values,
                rtol=1e-9,
                atol=1e-12,
            )
