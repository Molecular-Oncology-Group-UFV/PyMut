"""Fixtures and helpers for testing combine_pymutations in src/pyMut/combination.py"""

import pandas as pd
from src.pyMut.core import PyMutation, MutationMetadata


def make_meta(path: str = "a.vcf", fmt: str = "VCF", assembly: str = "38") -> MutationMetadata:
    """Create a minimal MutationMetadata instance for tests."""
    return MutationMetadata(source_format=fmt, file_path=path, filters=["PASS"], assembly=assembly, notes=None)


def make_pm(data: dict, samples: list, meta: MutationMetadata) -> PyMutation:
    """Convenience constructor for PyMutation from a dict of columns."""
    df = pd.DataFrame(data)
    return PyMutation(df, meta, samples)


def sort_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by variant identity columns if present, otherwise reset index."""
    required = {"CHROM", "POS", "REF", "ALT"}
    if required.issubset(df.columns):
        return df.sort_values(["CHROM", "POS", "REF", "ALT"]).reset_index(drop=True)
    return df.reset_index(drop=True)
