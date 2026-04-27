#!/usr/bin/env python3
"""
CLI utility to compare extracted mutational signatures against COSMIC catalog.

Usage examples:

./compare_signatures.py
# or explicitly:
python compare_signatures.py \
  --signatures ../extract_signatures/R_signatures_W_96x3.csv \
  --exposures  ../extract_signatures/R_exposures_3x187.csv \
  --output     .

This script reads the 96xK signatures matrix (rows=trinucleotide contexts, columns=signatures),
optionally ensures the row order matches the standard 96-context ordering, and calls
src.pyMut.analysis.mutational_signature.compare_signatures to compute cosine similarity
against the COSMIC v3.4 SBS catalog (GRCh37)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project's 'src' directory is on sys.path for src-layout imports
_THIS_FILE = Path(__file__).resolve()
for parent in [_THIS_FILE.parent, *_THIS_FILE.parents]:
    candidate = parent / "src"
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        break

# Import compare_signatures and the standard context order from package 'pyMut'
try:
    from pyMut.analysis.mutational_signature import (
        compare_signatures,
        TRINUCLEOTIDE_CONTEXTS,
    )
except Exception as e:
    print(f"[ERROR] Unable to import compare_signatures from pyMut: {e}", file=sys.stderr)
    sys.exit(1)

# Default COSMIC catalog relative path from this script directory
DEFAULT_COSMIC_PATH = str((_THIS_FILE.parent / "../../../src/pyMut/data/examples/COSMIC_catalogue-signatures_SBS96_v3.4/COSMIC_v3.4_SBS_GRCh37.txt").resolve())
# Default inputs: point to R outputs in ../extract_signatures relative to this script
DEFAULT_SIGNATURES_PATH = str((_THIS_FILE.parent.parent / "extract_signatures" / "R_signatures_W_96x3.csv").resolve())
DEFAULT_EXPOSURES_PATH = str((_THIS_FILE.parent.parent / "extract_signatures" / "R_exposures_3x187.csv").resolve())


def load_signatures(signatures_csv: str) -> tuple[np.ndarray, list[str]]:
    """Load 96xK signatures matrix W from CSV and return (W, signature_names).

    The CSV is expected to have:
    - First column: trinucleotide context labels (e.g., A[C>A]A, ..., T[T>G]T)
    - Remaining columns: one column per signature (e.g., Sig1, Sig2, ...)
    """
    df = pd.read_csv(signatures_csv, index_col=0)

    # Basic validation
    if df.shape[0] != 96:
        print(f"[WARNING] Signatures CSV has {df.shape[0]} rows; expected 96.")

    # Reorder rows to the standard 96-context order if possible
    try:
        missing = set(TRINUCLEOTIDE_CONTEXTS) - set(df.index)
        if missing:
            print(f"[WARNING] Missing {len(missing)} contexts in signatures CSV. Proceeding with available order.")
        else:
            df = df.reindex(TRINUCLEOTIDE_CONTEXTS)
    except Exception as e:
        print(f"[WARNING] Could not reindex signatures to standard contexts: {e}")

    # Ensure numeric dtype
    df = df.apply(pd.to_numeric, errors='coerce')

    # Build W and names
    W = df.values.astype(float)
    signature_names = list(df.columns)

    return W, signature_names


def load_exposures(exposures_csv: str) -> pd.DataFrame:
    """Load exposures H (KxN) from CSV. Not used for comparison, but accepted as input."""
    try:
        # Many tools export exposures with signatures as first column (rows), samples as columns
        df = pd.read_csv(exposures_csv, index_col=0)
        # Coerce to numeric where possible
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read exposures CSV '{exposures_csv}': {e}", file=sys.stderr)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare mutational signatures against COSMIC catalog")
    parser.add_argument("--signatures", "-s", default=DEFAULT_SIGNATURES_PATH, help="Path to signatures CSV (96xK). Default: R_signatures_W_96x3.csv in ../extract_signatures")
    parser.add_argument("--exposures", "-e", default=DEFAULT_EXPOSURES_PATH, help="Path to exposures CSV (KxN). Default: R_exposures_3x187.csv in ../extract_signatures")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: signatures CSV folder)")
    parser.add_argument("--min-cosine", type=float, default=0.6, help="Minimum cosine similarity to report a match")
    parser.add_argument(
        "--cosmic-path",
        default=DEFAULT_COSMIC_PATH,
        help="Path to COSMIC SBS96 catalog (default is the repository path requested)",
    )

    args = parser.parse_args(argv)

    signatures_csv = args.signatures
    exposures_csv = args.exposures
    out_dir = args.output or str(Path(signatures_csv).resolve().parent)
    min_cosine = float(args.min_cosine)
    cosmic_path = args.cosmic_path

    # Validate input files exist
    if not os.path.exists(signatures_csv):
        print(f"[ERROR] Signatures CSV not found: {signatures_csv}", file=sys.stderr)
        return 2
    if not os.path.exists(exposures_csv):
        print(f"[ERROR] Exposures CSV not found: {exposures_csv}", file=sys.stderr)
        return 2
    if not os.path.exists(cosmic_path):
        print(f"[ERROR] COSMIC catalog not found: {cosmic_path}", file=sys.stderr)
        return 2

    os.makedirs(out_dir, exist_ok=True)

    # Load inputs
    W, sig_names = load_signatures(signatures_csv)
    H_df = load_exposures(exposures_csv)  # Not used below, but validate availability

    # Run comparison
    result = compare_signatures(W=W, cosmic_path=cosmic_path, min_cosine=min_cosine, return_matrix=True)

    summary_df = result["summary_df"].copy()

    # Replace generic Signature_1..k with original column names if lengths match
    if len(sig_names) == summary_df.shape[0]:
        summary_df["Signature_W"] = sig_names

    # Save outputs
    summary_path = os.path.join(out_dir, "py_summary_compare_signatures.csv")
    matrix_path = os.path.join(out_dir, "py_cosine_matrix.csv")
    summary_df.to_csv(summary_path, index=False)

    cosine_matrix = result.get("cosine_matrix")
    if cosine_matrix is not None:
        # Create a DataFrame for better readability
        cosine_df = pd.DataFrame(cosine_matrix, index=sig_names)
        cosine_df.to_csv(matrix_path)

    # Console output
    print("Comparison completed.")
    print(f"- Signatures: {signatures_csv}")
    print(f"- Exposures:  {exposures_csv}")
    print(f"- COSMIC:     {cosmic_path}")
    print(f"- Output dir: {out_dir}")
    print("\nBest matches per signature:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(summary_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
