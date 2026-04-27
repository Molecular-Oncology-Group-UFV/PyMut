"""
Fixtures and utilities for testing cosmic_cancer_annotate.py
"""
import io
import gzip
from pathlib import Path

import pandas as pd
import pytest

from src.pyMut.annotate.cosmic_cancer_annotate import (
    _read_file_auto,
    _write_file_auto,
)


@pytest.fixture
def maf_df_min():
    """Minimal MAF-like dataframe with a joinable gene column.

    Use an alias recognized by find_alias for canonical "Hugo_Symbol".
    """
    return pd.DataFrame(
        {
            # alias from FIELDS["Hugo_Symbol"]
            "Hugo_Symbol": ["TP53", "EGFR", "ALK", None, "BRAF"],
            "OtherCol": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def cosmic_df():
    """Small COSMIC-like dataframe."""
    return pd.DataFrame(
        {
            "GENE_SYMBOL": ["TP53", "EGFR", "BRAF"],
            "SYNONYMS": ["P53,TRP53", None, "B-RAF"],
            "ROLE_IN_CANCER": ["Tumor suppressor", "Oncogene", None],
            "TIER": ["1", None, "2"],
            "Numeric_Feature": [1, None, 3],
        }
    )


@pytest.fixture
def oncokb_df():
    """Small OncoKB-like dataframe."""
    return pd.DataFrame(
        {
            "Hugo Symbol": ["TP53", "EGFR", "BRAF"],
            "Gene Aliases": ["TRP53,P53", "ERBB1", None],
            "Is Oncogene": ["False", "True", "yes"],
            "Is Tumor Suppressor Gene": ["True", "False", "False"],
            "OncoKB Annotated": ["Yes", None, "No"],
            "MSK-IMPACT": [1, None, 0],
        }
    )


@pytest.fixture
def tmp_tsv_files(tmp_path, cosmic_df, oncokb_df):
    """Write COSMIC and OncoKB to disk (tsv and gz)."""
    cosmic_tsv = tmp_path / "cosmic.tsv"
    cosmic_gz = tmp_path / "cosmic.tsv.gz"
    oncokb_tsv = tmp_path / "oncokb.tsv"
    oncokb_gz = tmp_path / "oncokb.tsv.gz"

    cosmic_df.to_csv(cosmic_tsv, sep="\t", index=False)
    with gzip.open(cosmic_gz, "wt") as f:
        cosmic_df.to_csv(f, sep="\t", index=False)

    oncokb_df.to_csv(oncokb_tsv, sep="\t", index=False)
    with gzip.open(oncokb_gz, "wt") as f:
        oncokb_df.to_csv(f, sep="\t", index=False)

    return {
        "cosmic_tsv": cosmic_tsv,
        "cosmic_gz": cosmic_gz,
        "oncokb_tsv": oncokb_tsv,
        "oncokb_gz": oncokb_gz,
    }


@pytest.fixture
def make_tsv(tmp_path):
    """Helper to write a generic TSV or TSV.GZ from a dataframe."""

    def _writer(df: pd.DataFrame, name: str, gz: bool = False) -> Path:
        path = tmp_path / name
        if gz and not str(path).endswith(".gz"):
            path = path.with_suffix(path.suffix + ".gz")
        if gz:
            with gzip.open(path, "wt") as f:
                df.to_csv(f, sep="\t", index=False)
        else:
            df.to_csv(path, sep="\t", index=False)
        return path

    return _writer


class DummyMeta:
    def __init__(self):
        self.notes = ""
        self.assembly = 37


class DummyPM:
    """Minimal object with .data and .metadata to use CancerAnnotateMixin."""

    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.metadata = DummyMeta()


# Convenience wrappers to exercise read/write helpers directly
@pytest.fixture
def read_file_auto():
    return _read_file_auto


@pytest.fixture
def write_file_auto():
    return _write_file_auto
