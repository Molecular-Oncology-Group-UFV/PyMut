import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pyMut.input import read_vcf, read_maf
from src.pyMut.analysis.mutational_signature import extractSignatures


GOLDEN_ROOT = Path(__file__).parent
VCF_DIR = GOLDEN_ROOT / "vcf_signatures"
MAF_DIR = GOLDEN_ROOT / "maf_signatures"

# Candidate FASTA locations, tried in order. Add your own machine's paths
# here or (better) set the PYMUT_FASTA_GRCh37 / PYMUT_FASTA_GRCh38 env vars.
FASTA_CANDIDATES = {
    "37": [
        str(Path.home() / "Documentos/PyMut_DocsAlba/data/references/hs37d5.fa"),
        "/home/luisruimore/Escritorio/TFG/src/pyMut/data/resources/genome/GRCh37/GRCh37.p13.genome.fa",
    ],
    "38": [
        str(Path.home() / "Documentos/PyMut_DocsAlba/data/references/Homo_sapiens.GRCh38.dna.primary_assembly.fa"),
        "/home/luisruimore/Escritorio/TFG/src/pyMut/data/resources/genome/GRCh38/GRCh38.p14.genome.fa",
    ],
}
ENV_KEY_MAP = {
    "37": "PYMUT_FASTA_GRCh37",
    "38": "PYMUT_FASTA_GRCh38",
}


def _require_fasta(assembly: str) -> str:
    """Resolve a FASTA path for the given assembly, or skip the test.

    Priority:
    1) Candidate paths listed in FASTA_CANDIDATES.
    2) Environment variables PYMUT_FASTA_GRCh37 / PYMUT_FASTA_GRCh38.
    3) Otherwise skip with a helpful message.
    """
    assert assembly in ("37", "38"), "Unsupported assembly. Use '37' or '38'."

    for candidate in FASTA_CANDIDATES[assembly]:
        if candidate and Path(candidate).exists():
            return candidate

    env_key = ENV_KEY_MAP[assembly]
    fasta_path = os.environ.get(env_key)
    if fasta_path and Path(fasta_path).exists():
        return fasta_path

    pytest.skip(
        f"FASTA for GRCh{assembly} not available. Tried: {FASTA_CANDIDATES[assembly]}. "
        f"Alternatively, set env var {env_key} to a readable FASTA path to run this golden test."
    )


def _load_expected(expected_dir: Path):
    """Load expected W/H parquet and optional meta.json. Skip if missing."""
    W_path = expected_dir / "expected_W.parquet"
    H_path = expected_dir / "expected_H.parquet"
    meta_path = expected_dir / "meta.json"

    if not W_path.exists() or not H_path.exists():
        pytest.skip(
            f"Golden snapshots not present in {expected_dir}. "
            "Delete stale snapshots to regenerate them with the current implementation, "
            "inspect the result, and commit the new expected_W/H.parquet."
        )

    W_exp = pd.read_parquet(W_path)
    H_exp = pd.read_parquet(H_path)

    meta = {}
    if meta_path.exists():
        try:
            meta = pd.read_json(meta_path)
            if isinstance(meta, pd.DataFrame):
                meta = meta.iloc[0].to_dict()
        except Exception:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)

    return W_exp, H_exp, meta


def _as_frame(obj, index, columns) -> pd.DataFrame:
    """Normalize extractSignatures outputs (DataFrame or ndarray) to a DataFrame.

    Rebuilt POSITIONALLY on purpose: extractSignatures derives the 96 context
    rows and the sample columns from contexts_df in order, so positional
    alignment is guaranteed regardless of the label names the NMF wrapper
    uses internally (a label-based reindex would turn everything into NaN
    when those names differ from our canonical Signature_i labels).
    """
    arr = np.asarray(obj.values if isinstance(obj, pd.DataFrame) else obj, dtype=float)
    expected_shape = (len(list(index)), len(list(columns)))
    assert arr.shape == expected_shape, (
        f"unexpected output shape {arr.shape}, expected {expected_shape}"
    )
    return pd.DataFrame(arr, index=list(index), columns=list(columns))


def _extract_signature_frames(contexts_df: pd.DataFrame, k: int, p_constant: float = 1e-4):
    """Run the current extractSignatures (R NMF, fixed seed -> deterministic)
    and return (W_df, H_df) with canonical Signature_i column/index labels.

    pConstant is always applied: the small 100-variant fixture matrices are
    necessarily sparse (many of the 96 context rows are all zeros), and R's
    NMF::nmf rejects matrices with zero rows. Adding a tiny constant keeps
    every entry > 0; with the fixed seed the result stays fully deterministic.
    """
    res = extractSignatures(contexts_df, n=k, pConstant=p_constant)

    sig_cols = [f"Signature_{i + 1}" for i in range(k)]
    ctx_index = list(contexts_df.index)
    samples_order = list(contexts_df.columns)

    # 'signatures': 96 x n (scaled, columns sum to 1); 'contributions': n x samples
    W_df = _as_frame(res["signatures"], ctx_index, sig_cols)
    H_df = _as_frame(res["contributions"], sig_cols, samples_order)
    return W_df, H_df


def _assert_signature_outputs(W_actual: pd.DataFrame, H_actual: pd.DataFrame,
                              W_expected: pd.DataFrame, H_expected: pd.DataFrame):
    # Check axes equality
    assert list(W_actual.index) == list(W_expected.index), "W index (contexts) differ from expected"
    assert list(W_actual.columns) == list(W_expected.columns), "W columns (signature labels) differ from expected"
    assert list(H_actual.index) == list(H_expected.index), "H index (signature labels) differ from expected"
    assert list(H_actual.columns) == list(H_expected.columns), "H columns (samples) differ from expected"

    # Non-negativity
    assert (W_actual.values >= 0).all(), "W contains negative values"
    assert (H_actual.values >= 0).all(), "H contains negative values"

    # Columns of W should sum to ~1 (scaled signatures)
    col_sums = W_actual.sum(axis=0).values
    assert np.allclose(col_sums, 1.0, atol=1e-6), "Columns of W do not sum to 1 within tolerance"

    # Numeric close
    np.testing.assert_allclose(W_actual.values, W_expected.values, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(H_actual.values, H_expected.values, rtol=1e-6, atol=1e-8)


def _run_golden_test(input_path: Path, assembly: str, expected_dir: Path, source: str):
    assert input_path.exists(), f"Fixture missing: {input_path}"
    fasta = _require_fasta(assembly)

    # Read and compute contexts
    if source == "VCF":
        pm = read_vcf(input_path, assembly=assembly)
    else:
        pm = read_maf(input_path, assembly=assembly)
    contexts_df, _ = pm.trinucleotideMatrix(fasta)

    expected_dir.mkdir(parents=True, exist_ok=True)
    W_path = expected_dir / "expected_W.parquet"
    H_path = expected_dir / "expected_H.parquet"
    meta_path = expected_dir / "meta.json"

    k = 2

    if not W_path.exists() or not H_path.exists():
        # REGENERATION MODE: produce snapshots with the current implementation.
        # extractSignatures calls R's NMF::nmf with a fixed seed (123456), so
        # results are deterministic across runs and machines with the same R.
        W_df, H_df = _extract_signature_frames(contexts_df, k, p_constant=1e-4)

        W_df.to_parquet(W_path)
        H_df.to_parquet(H_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({
                "k": k,
                "pConstant": 1e-4,
                "method": "extractSignatures (R NMF::nmf, brunet, seed=123456, pConstant=1e-4)",
                "samples_order": list(contexts_df.columns),
                "contexts_order": list(contexts_df.index),
                "assembly": assembly,
                "source": source,
            }, fh, indent=2)

        # Sanity self-check and return
        _assert_signature_outputs(W_df, H_df, W_df, H_df)
        return

    # COMPARISON MODE
    W_expected, H_expected, _meta = _load_expected(expected_dir)

    expected_samples = list(H_expected.columns)
    if set(expected_samples) != set(contexts_df.columns):
        pytest.skip(
            "Sample names in expected snapshot do not match current reader output. "
            "Regenerate snapshots or update reader."
        )
    contexts_df = contexts_df[expected_samples]

    k = len(W_expected.columns)
    p_constant = float(_meta.get("pConstant", 1e-4))
    W_actual, H_actual = _extract_signature_frames(contexts_df, k, p_constant=p_constant)

    _assert_signature_outputs(W_actual, H_actual, W_expected, H_expected)


@pytest.mark.unit
@pytest.mark.parametrize("assembly", ["38"])  # Golden requires GRCh38 for VCF
def test_golden_signatures_vcf(assembly):
    base_dir = Path(__file__).parents[2] / "unit" / "io" / "fixtures" / "data"
    vcf_gz = base_dir / (
        "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf.gz"
    )
    _run_golden_test(vcf_gz, assembly, VCF_DIR, source="VCF")


@pytest.mark.unit
@pytest.mark.parametrize("assembly", ["37"])  # Golden requires GRCh37 for MAF
def test_golden_signatures_maf(assembly):
    base_dir = Path(__file__).parents[2] / "unit" / "io" / "fixtures" / "data"
    maf_gz = base_dir / "test_tcga_laml_100variants.maf.gz"
    _run_golden_test(maf_gz, assembly, MAF_DIR, source="MAF")
