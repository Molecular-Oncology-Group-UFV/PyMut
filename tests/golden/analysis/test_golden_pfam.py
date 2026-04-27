import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pyMut.input import read_vcf, read_maf

GOLDEN_ROOT = Path(__file__).parent
VCF_DIR = GOLDEN_ROOT / "vcf_pfam"
MAF_DIR = GOLDEN_ROOT / "maf_pfam"


def _load_expected(dir_: Path):
    """Load expected PFAM domain summaries and optional meta.json. Skip if missing."""
    dom_path = dir_ / "pfam_by_domain.parquet"
    pos_path = dir_ / "pfam_by_aapos.parquet"
    meta_path = dir_ / "meta.json"

    if not dom_path.exists() or not pos_path.exists():
        pytest.skip(
            f"Golden snapshots not present in {dir_}. "
            f"Generate them offline as documented and commit pfam_by_domain.parquet, pfam_by_aapos.parquet (and meta.json)."
        )

    dom_exp = pd.read_parquet(dom_path)
    pos_exp = pd.read_parquet(pos_path)

    meta = {}
    if meta_path.exists():
        # Always use std json to preserve list types precisely
        import json
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

    return dom_exp, pos_exp, meta


def _assert_schema_and_values(actual: pd.DataFrame, expected: pd.DataFrame, allow_float: bool = False):
    # Column order must match
    assert list(actual.columns) == list(expected.columns), "Columns mismatch"
    # Exact shape
    assert actual.shape == expected.shape, "Shape mismatch"
    # Values: ints expected, but allow float comparison if requested
    if allow_float:
        np.testing.assert_allclose(actual.values.astype(float), expected.values.astype(float), rtol=0, atol=0)
    else:
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
    # Load expected
    dom_exp, pos_exp, meta = _load_expected(expected_dir)

    # Resolve input path (allow override via env)
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

    # Read dataset
    pm = loader(data_path, assembly=assembly)

    # Annotate PFAM without DB for determinism
    pm_ann = pm.annotate_pfam(prefer_database=False)

    # Frozen parameters
    aa_column = meta.get("aa_column", "aa_pos")
    top_n = int(meta.get("top_n", 10))
    include_synonymous = bool(meta.get("include_synonymous", False))

    # Run pfam_domains in both modes
    dom_act = pm_ann.pfam_domains(
        summarize_by="PfamDomain", top_n=top_n, aa_column=aa_column, include_synonymous=include_synonymous
    )
    pos_act = pm_ann.pfam_domains(
        summarize_by="AAPos", top_n=top_n, aa_column=aa_column, include_synonymous=include_synonymous
    )

    # Align columns to expected (in case the uniprot alias varies)
    dom_act = dom_act[dom_exp.columns]
    pos_act = pos_act[pos_exp.columns]

    # Assertions
    _assert_schema_and_values(dom_act, dom_exp)
    _assert_schema_and_values(pos_act, pos_exp)

    # Useful invariants
    assert (dom_act['n_genes'] >= 0).all() and (dom_act['n_variants'] >= 0).all()
    assert (pos_act['n_genes'] >= 0).all() and (pos_act['n_variants'] >= 0).all()
    assert len(dom_act) <= top_n and len(pos_act) <= top_n
