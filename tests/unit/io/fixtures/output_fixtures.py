import pytest
import pandas as pd
from unittest.mock import Mock

from src.pyMut.output import OutputMixin


class DummyMeta:
    def __init__(self, assembly: str = "GRCh38", notes: str | None = None):
        self.assembly = assembly
        self.notes = notes


class DummyPyMutation(OutputMixin):
    def __init__(self, data: pd.DataFrame, samples: list[str], metadata: DummyMeta):
        self.data = data
        self.samples = samples
        self.metadata = metadata


@pytest.fixture
def dummy_meta() -> DummyMeta:
    return DummyMeta(assembly="GRCh38", notes=None)


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    return pd.DataFrame({
        "CHROM": ["chr1"],
        "POS": [123],
        "REF": ["A"],
        "ALT": ["T"],
        "ID": ["rs1"],
        "S1": ["A|T"],
    })


@pytest.fixture
def pymut(minimal_df, dummy_meta):
    return DummyPyMutation(minimal_df.copy(), ["S1"], dummy_meta)


@pytest.fixture
def caplog_info(caplog):
    caplog.set_level("INFO")
    return caplog
