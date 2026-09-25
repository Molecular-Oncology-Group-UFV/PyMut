import json
import os
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.pyMut.input import read_vcf, read_maf

# NOTE (design): these golden tests must be HERMETIC -- no network, no
# first-use database builds. annotate_pfam(strategy='extended') resolves
# UniProt identifiers against the annotation database, whose initial
# download/build is slow and environment-dependent, so it is NOT a stable
# anchor for a golden snapshot (extended-strategy behaviour is covered by
# the unit tests with an injected in-memory database instead).
#
# strategy='basic' (the maftools-parity mapping: gene symbol + HGVS protein
# change against the bundled gene->Pfam table) only needs:
#   - the bundled data/pfam_domains.csv (shipped with the package), and
#   - a DuckDB connection used purely to register temporary tables for the
#     join -- an in-memory database is injected here for that, so the test
#     never touches connect_db() and runs offline in seconds.
#
# Old snapshots (annotated with the removed prefer_database=False pipeline,
# or with strategy='extended') must be regenerated: delete the folder's
# parquets + meta.json and rerun this test.

GOLDEN_ROOT = Path(__file__).parent
VCF_DIR = GOLDEN_ROOT / "vcf_pfam"
MAF_DIR = GOLDEN_ROOT / "maf_pfam"


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


# VEP->MAF column mapping used to give VEP-annotated fixtures the HGVS
# protein-change column that annotate_pfam(strategy='basic') expects. This is
# deterministic fixture normalization (the golden tests the pfam_domains
# summarization, not the VEP->MAF column mapping -- that is covered by the
# read_vcf unit tests).
_VEP_TO_HGVS = {
    "VEP_HGVSp": "HGVSp_Short",
    "VEP_HGVSp_Short": "HGVSp_Short",
    "VEP_HGVS": "HGVSp",
    "VEP_Protein_Change": "Protein_Change",
}


def _run_pfam(loader, data_path: str, assembly: str, top_n: int, include_synonymous: bool):
    pm = loader(data_path, assembly=assembly)

    # Pre-flight: strategy='basic' needs Hugo_Symbol + one HGVS protein-change
    # column (HGVSp_Short / Protein_Change / AAChange / HGVSp).
    aa_candidates = [c for c in ("HGVSp_Short", "Protein_Change", "AAChange", "HGVSp")
                     if c in pm.data.columns]
    assert "Hugo_Symbol" in pm.data.columns, (
        f"Fixture {data_path} lacks Hugo_Symbol; annotate_pfam(strategy='basic') "
        "cannot run on it"
    )
    if not aa_candidates:
        # Try to derive the HGVS column from VEP annotations
        for vep_col, target in _VEP_TO_HGVS.items():
            if vep_col in pm.data.columns and target not in pm.data.columns:
                pm.data = pm.data.copy()
                pm.data[target] = pm.data[vep_col]
                aa_candidates = [target]
                break
    if not aa_candidates:
        pytest.skip(
            f"Fixture {data_path} has no HGVS protein-change column "
            "(HGVSp_Short / Protein_Change / AAChange / HGVSp, nor a usable "
            "VEP_HGVSp* column); annotate_pfam(strategy='basic') cannot run on it. "
            "VCF-based PFAM coverage lives in the unit tests and in the "
            "pfam_domain_correlation validation with real VEP-annotated data."
        )

    # Inject an in-memory DuckDB handle: annotate_pfam uses the connection
    # only to register temporary tables for the SQL join. This keeps the
    # golden fully offline and independent of connect_db() and its resources.
    conn = duckdb.connect(":memory:")
    try:
        pm_ann = pm.annotate_pfam(db_conn=conn, strategy="basic")
    finally:
        conn.close()

    dom = pm_ann.pfam_domains(
        summarize_by="PfamDomain", top_n=top_n, include_synonymous=include_synonymous
    )
    pos = pm_ann.pfam_domains(
        summarize_by="AAPos", top_n=top_n, include_synonymous=include_synonymous
    )
    # pfam_domains stores implementation metadata (DataFrames) in df.attrs;
    # pandas serializes attrs to JSON when writing parquet, which fails.
    # The snapshots must contain the table only, so attrs are stripped here
    # (and on load they are absent anyway, keeping comparisons symmetric).
    for df in (dom, pos):
        df.attrs = {}
    return dom, pos


def _load_expected(dir_: Path):
    """Load expected PFAM domain summaries and optional meta.json. Skip if missing."""
    dom_path = dir_ / "pfam_by_domain.parquet"
    pos_path = dir_ / "pfam_by_aapos.parquet"
    meta_path = dir_ / "meta.json"

    if not dom_path.exists() or not pos_path.exists():
        pytest.skip(
            f"Golden snapshots not present in {dir_}. "
            "Delete stale snapshots to regenerate them with the current implementation, "
            "inspect the result, and commit pfam_by_domain.parquet, pfam_by_aapos.parquet (and meta.json)."
        )

    dom_exp = pd.read_parquet(dom_path)
    pos_exp = pd.read_parquet(pos_path)

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

    return dom_exp, pos_exp, meta


def _assert_schema_and_values(actual: pd.DataFrame, expected: pd.DataFrame):
    assert list(actual.columns) == list(expected.columns), "Columns mismatch"
    assert actual.shape == expected.shape, "Shape mismatch"
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
    )


@pytest.mark.parametrize(
    "loader, expected_dir, assembly",
    [
        (read_vcf, VCF_DIR, "38"),
        (read_maf, MAF_DIR, "37"),
    ],
)
def test_golden_pfam_domains(loader, expected_dir, assembly):
    expected_dir.mkdir(parents=True, exist_ok=True)
    dom_path = expected_dir / "pfam_by_domain.parquet"
    pos_path = expected_dir / "pfam_by_aapos.parquet"
    meta_path = expected_dir / "meta.json"

    data_path = _resolve_input(assembly)

    # Frozen parameters
    top_n_default = 10
    include_synonymous_default = False

    if not dom_path.exists() or not pos_path.exists():
        # REGENERATION MODE: produce snapshots with the current implementation.
        # strategy='basic' on a frozen input is deterministic (bundled
        # gene->Pfam table + fixed parsing rules + no external DB).
        dom_act, pos_act = _run_pfam(
            loader, data_path, assembly, top_n_default, include_synonymous_default
        )
        dom_act.to_parquet(dom_path)
        pos_act.to_parquet(pos_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({
                "strategy": "basic",
                "db": "in-memory duckdb (no external annotation DB)",
                "top_n": top_n_default,
                "include_synonymous": include_synonymous_default,
            }, fh, indent=2)
        # Minimal sanity
        assert (dom_act["n_genes"].values >= 0).all() and (dom_act["n_variants"].values >= 0).all()
        assert (pos_act["n_genes"].values >= 0).all() and (pos_act["n_variants"].values >= 0).all()
        assert len(dom_act) <= top_n_default and len(pos_act) <= top_n_default
        return

    # COMPARISON MODE
    dom_exp, pos_exp, meta = _load_expected(expected_dir)

    top_n = int(meta.get("top_n", top_n_default))
    include_synonymous = bool(meta.get("include_synonymous", include_synonymous_default))

    dom_act, pos_act = _run_pfam(loader, data_path, assembly, top_n, include_synonymous)

    # Align columns to expected (the summary schema may evolve; stale
    # snapshots then fail here with a clear message and must be regenerated)
    dom_act = dom_act[dom_exp.columns]
    pos_act = pos_act[pos_exp.columns]

    _assert_schema_and_values(dom_act, dom_exp)
    _assert_schema_and_values(pos_act, pos_exp)

    # Useful invariants
    assert len(dom_act) <= top_n and len(pos_act) <= top_n
