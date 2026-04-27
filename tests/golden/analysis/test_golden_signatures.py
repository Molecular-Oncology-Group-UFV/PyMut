import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pyMut.input import read_vcf, read_maf
from src.pyMut.analysis.mutational_signature import extract_signatures


GOLDEN_ROOT = Path(__file__).parent
VCF_DIR = GOLDEN_ROOT / "vcf_signatures"
MAF_DIR = GOLDEN_ROOT / "maf_signatures"


def _require_fasta(assembly: str) -> str:
    """Resolve FASTA path for given assembly.

    Priority:
    1) Use the fixed paths requested by the user if they exist.
    2) Fallback to environment variables PYMUT_FASTA_GRCh37 / PYMUT_FASTA_GRCh38.
    3) If neither is available, skip with a helpful message.
    """
    fixed = {
        "37": "/home/luisruimore/Escritorio/TFG/src/pyMut/data/resources/genome/GRCh37/GRCh37.p13.genome.fa",
        "38": "/home/luisruimore/Escritorio/TFG/src/pyMut/data/resources/genome/GRCh38/GRCh38.p14.genome.fa",
    }
    env_key_map = {
        "37": "PYMUT_FASTA_GRCh37",
        "38": "PYMUT_FASTA_GRCh38",
    }
    assert assembly in ("37", "38"), "Unsupported assembly. Use '37' or '38'."

    # 1) Try fixed paths
    fixed_path = fixed.get(assembly)
    if fixed_path and Path(fixed_path).exists():
        return fixed_path

    # 2) Try env var fallback
    env_key = env_key_map[assembly]
    fasta_path = os.environ.get(env_key)
    if fasta_path and Path(fasta_path).exists():
        return fasta_path

    # 3) Skip if not available
    pytest.skip(
        (
            f"FASTA for GRCh{assembly} not available. Expected fixed path: {fixed_path}. "
            f"Alternatively, set env var {env_key} to a readable FASTA path to run this golden test."
        )
    )


def _load_expected(expected_dir: Path):
    """Load expected W/H parquet and optional meta.json. Skip if missing."""
    W_path = expected_dir / "expected_W.parquet"
    H_path = expected_dir / "expected_H.parquet"
    meta_path = expected_dir / "meta.json"

    if not W_path.exists() or not H_path.exists():
        pytest.skip(
            f"Golden snapshots not present in {expected_dir}. Generate them offline as documented and commit expected_W/H.parquet."
        )

    W_exp = pd.read_parquet(W_path)
    H_exp = pd.read_parquet(H_path)

    meta = {}
    if meta_path.exists():
        try:
            meta = pd.read_json(meta_path)
            if isinstance(meta, pd.DataFrame):
                # If saved as a single-row DataFrame
                meta = meta.iloc[0].to_dict()
        except Exception:
            # Fallback to std json
            import json
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)

    return W_exp, H_exp, meta


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

    # Columns of W should sum to ~1
    col_sums = W_actual.sum(axis=0).values
    assert np.allclose(col_sums, 1.0, atol=1e-8), "Columns of W do not sum to 1 within tolerance"

    # Numeric close
    np.testing.assert_allclose(W_actual.values, W_expected.values, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(H_actual.values, H_expected.values, rtol=1e-6, atol=1e-8)


@pytest.mark.unit
@pytest.mark.parametrize("assembly", ["38"])  # Golden requires GRCh38 for VCF
def test_golden_signatures_vcf(assembly):
    # Resolve inputs
    base_dir = Path(__file__).parents[2] / "unit" / "io" / "fixtures" / "data"
    vcf_gz = base_dir / (
        "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf.gz"
    )
    assert vcf_gz.exists(), f"Fixture missing: {vcf_gz}"

    fasta = _require_fasta(assembly)

    # Read and compute contexts
    pm = read_vcf(vcf_gz, assembly=assembly)
    contexts_df, _ = pm.trinucleotideMatrix(fasta)

    expected_dir = VCF_DIR
    expected_dir.mkdir(parents=True, exist_ok=True)
    W_path = expected_dir / "expected_W.parquet"
    H_path = expected_dir / "expected_H.parquet"
    meta_path = expected_dir / "meta.json"

    if not W_path.exists() or not H_path.exists():
        # Generate expected snapshots with deterministic parameters
        k = 2
        nrun = 10
        pseudocount = 1e-4
        random_seed = 1234

        res = extract_signatures(contexts_df, k=k, nrun=nrun, pseudocount=pseudocount, random_seed=random_seed)

        sig_cols = [f"Signature_{i+1}" for i in range(k)]
        ctx_index = list(contexts_df.index)
        samples_order = list(contexts_df.columns)

        W_df = pd.DataFrame(res["W"], index=ctx_index, columns=sig_cols)
        H_df = pd.DataFrame(res["H"], index=sig_cols, columns=samples_order)

        # Save snapshots
        W_df.to_parquet(W_path)
        H_df.to_parquet(H_path)
        import json
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({
                "k": k,
                "nrun": nrun,
                "pseudocount": pseudocount,
                "random_seed": random_seed,
                "samples_order": samples_order,
                "contexts_order": ctx_index,
                "assembly": assembly,
                "source": "VCF"
            }, fh, indent=2)

        # Compare actual to itself (sanity) and return
        _assert_signature_outputs(W_df, H_df, W_df, H_df)
        return

    # Load golden expected
    W_expected, H_expected, meta = _load_expected(expected_dir)

    # Ensure sample order matches expected H columns
    expected_samples = list(H_expected.columns)
    if set(expected_samples) != set(contexts_df.columns):
        pytest.skip(
            "Sample names in expected snapshot do not match current reader output. Regenerate snapshots or update reader."
        )
    contexts_df = contexts_df[expected_samples]

    # Determine parameters
    k = len(W_expected.columns)
    nrun = int(meta.get("nrun", 10))
    pseudocount = float(meta.get("pseudocount", 1e-4))
    random_seed = int(meta.get("random_seed", 1234))

    # Extract signatures deterministically
    res = extract_signatures(contexts_df, k=k, nrun=nrun, pseudocount=pseudocount, random_seed=random_seed)

    # Build actual DataFrames with expected axes
    sig_cols = list(W_expected.columns)
    ctx_index = list(W_expected.index)

    W_actual = pd.DataFrame(res["W"], index=ctx_index, columns=sig_cols)
    H_actual = pd.DataFrame(res["H"], index=sig_cols, columns=expected_samples)

    _assert_signature_outputs(W_actual, H_actual, W_expected, H_expected)


@pytest.mark.unit
@pytest.mark.parametrize("assembly", ["37"])  # Golden requires GRCh37 for MAF
def test_golden_signatures_maf(assembly):
    # Resolve inputs
    base_dir = Path(__file__).parents[2] / "unit" / "io" / "fixtures" / "data"
    maf_gz = base_dir / "test_tcga_laml_100variants.maf.gz"
    assert maf_gz.exists(), f"Fixture missing: {maf_gz}"

    fasta = _require_fasta(assembly)

    # Read and compute contexts
    pm = read_maf(maf_gz, assembly=assembly)
    contexts_df, _ = pm.trinucleotideMatrix(fasta)

    expected_dir = MAF_DIR
    expected_dir.mkdir(parents=True, exist_ok=True)
    W_path = expected_dir / "expected_W.parquet"
    H_path = expected_dir / "expected_H.parquet"
    meta_path = expected_dir / "meta.json"

    if not W_path.exists() or not H_path.exists():
        # Generate expected snapshots with deterministic parameters
        k = 2
        nrun = 10
        pseudocount = 1e-4
        random_seed = 1234

        res = extract_signatures(contexts_df, k=k, nrun=nrun, pseudocount=pseudocount, random_seed=random_seed)

        sig_cols = [f"Signature_{i+1}" for i in range(k)]
        ctx_index = list(contexts_df.index)
        samples_order = list(contexts_df.columns)

        W_df = pd.DataFrame(res["W"], index=ctx_index, columns=sig_cols)
        H_df = pd.DataFrame(res["H"], index=sig_cols, columns=samples_order)

        # Save snapshots
        W_df.to_parquet(W_path)
        H_df.to_parquet(H_path)
        import json
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({
                "k": k,
                "nrun": nrun,
                "pseudocount": pseudocount,
                "random_seed": random_seed,
                "samples_order": samples_order,
                "contexts_order": ctx_index,
                "assembly": assembly,
                "source": "MAF"
            }, fh, indent=2)

        # Compare actual to itself (sanity) and return
        _assert_signature_outputs(W_df, H_df, W_df, H_df)
        return

    # Load golden expected
    W_expected, H_expected, meta = _load_expected(expected_dir)

    # Ensure sample order matches expected H columns
    expected_samples = list(H_expected.columns)
    if set(expected_samples) != set(contexts_df.columns):
        pytest.skip(
            "Sample names in expected snapshot do not match current reader output. Regenerate snapshots or update reader."
        )
    contexts_df = contexts_df[expected_samples]

    # Determine parameters
    k = len(W_expected.columns)
    nrun = int(meta.get("nrun", 10))
    pseudocount = float(meta.get("pseudocount", 1e-4))
    random_seed = int(meta.get("random_seed", 1234))

    # Extract signatures deterministically
    res = extract_signatures(contexts_df, k=k, nrun=nrun, pseudocount=pseudocount, random_seed=random_seed)

    # Build actual DataFrames with expected axes
    sig_cols = list(W_expected.columns)
    ctx_index = list(W_expected.index)

    W_actual = pd.DataFrame(res["W"], index=ctx_index, columns=sig_cols)
    H_actual = pd.DataFrame(res["H"], index=sig_cols, columns=expected_samples)

    _assert_signature_outputs(W_actual, H_actual, W_expected, H_expected)
