#!/usr/bin/env python3
"""
Example script to compute the 96xN trinucleotide context matrix from a MAF file
using PyMutation.trinucleotideMatrix, and write the resulting contexts dataframe
to a CSV file.

Defaults are set to relative paths for quick testing:
- MAF:   ../../../src/pyMut/data/examples/MAF/tcga_laml.maf.gz
- FASTA: ../../../src/pyMut/data/resources/hs37d5.fa

Notes
-----
- Requires pyfaidx to read the FASTA index (.fai should exist alongside the FASTA).
- The script also prints a quick sanity check comparing the column sums of the
  contexts matrix vs. per-sample counts in the enriched table returned by
  trinucleotideMatrix.
"""

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from src.pyMut.core import PyMutation
except Exception:
    # Fallback if running as installed package (pyMut on PYTHONPATH)
    try:
        from pyMut.core import PyMutation  # type: ignore
    except Exception as e:
        print(f"[ERROR] Could not import PyMutation: {e}")
        sys.exit(1)


# Default configuration for direct execution (no CLI)
DEFAULTS = {
    "maf": "../../../src/pyMut/data/examples/MAF/tcga_laml.maf.gz",
    "fasta": "../../../src/pyMut/data/resources/hs37d5.fa",
    "out": "py_trinuc_matrix.csv",
    "out_enriched": None,
    "prefix": "chr",
    "add_prefix": True,
    "ignore_chr": ["chrM"],
    "include_syn": True,
}


def compute_and_save(
    maf: str = DEFAULTS["maf"],
    fasta: str = DEFAULTS["fasta"],
    out: str = DEFAULTS["out"],
    out_enriched: str | None = DEFAULTS["out_enriched"],
    prefix: str = DEFAULTS["prefix"],
    add_prefix: bool = DEFAULTS["add_prefix"],
    ignore_chr: list[str] = DEFAULTS["ignore_chr"],
    include_syn: bool = DEFAULTS["include_syn"],
):
    """
    Compute the 96xN trinucleotide context matrix from a MAF file and write outputs.

    This module is intentionally not a CLI. Execute this file directly to run with
    the DEFAULTS, or import and call compute_and_save(...) programmatically.
    """
    # Read MAF (supports gz via pandas compression inference)
    print(f"[INFO] Reading MAF: {maf}")
    try:
        maf_df = pd.read_csv(
            maf,
            sep="\t",
            comment="#",
            low_memory=False,
        )
    except Exception as e:
        print(f"[ERROR] Could not read MAF: {e}")
        sys.exit(1)

    # Build PyMutation and compute matrix
    pm = PyMutation(maf_df)
    print(f"[INFO] Computing trinucleotide contexts using FASTA: {fasta}")
    try:
        contexts_df, enriched_df = pm.trinucleotideMatrix(
            ref_genome=fasta,
            prefix=prefix,
            add=add_prefix,
            ignoreChr=ignore_chr,
            useSyn=include_syn,
            fn=None,
        )
    except Exception as e:
        print(f"[ERROR] trinucleotideMatrix failed: {e}")
        sys.exit(1)

    # Write outputs
    try:
        contexts_df.to_csv(out, index=True)
        print(f"[OK] Wrote 96xN contexts matrix: {out} (shape={contexts_df.shape})")
    except Exception as e:
        print(f"[ERROR] Could not write contexts CSV: {e}")
        sys.exit(1)

    if out_enriched:
        try:
            enriched_df.to_csv(out_enriched, index=False)
            print(f"[OK] Wrote enriched SNV table: {out_enriched} (rows={len(enriched_df)})")
        except Exception as e:
            print(f"[WARN] Could not write enriched table: {e}")

    # Sanity check: per-sample counts
    if "Tumor_Sample_Barcode" in enriched_df.columns:
        try:
            sums = contexts_df.sum(axis=0)
            per_sample = (
                enriched_df.groupby("Tumor_Sample_Barcode")["idx96"].count()
                .reindex(contexts_df.columns, fill_value=0)
            )
            ok = (sums.values == per_sample.values).all()
            print(f"[CHECK] Sum per sample equals enriched counts: {ok}")
            if not ok:
                diff = (sums - per_sample).abs()
                print("[DETAIL] Differences per sample (contexts_sum - enriched_count):")
                print(diff[diff != 0])
        except Exception as e:
            print(f"[WARN] Could not perform sanity check: {e}")

    return contexts_df, enriched_df


def main():
    # Run with defaults (non-CLI execution)
    compute_and_save(
        maf=DEFAULTS["maf"],
        fasta=DEFAULTS["fasta"],
        out=DEFAULTS["out"],
        out_enriched=DEFAULTS["out_enriched"],
        prefix=DEFAULTS["prefix"],
        add_prefix=DEFAULTS["add_prefix"],
        ignore_chr=DEFAULTS["ignore_chr"],
        include_syn=DEFAULTS["include_syn"],
    )


if __name__ == "__main__":
    main()
