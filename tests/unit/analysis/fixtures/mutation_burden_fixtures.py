import pandas as pd
import pytest

from src.pyMut.analysis.mutation_burden import MutationBurdenMixin


class DummyPyMutation(MutationBurdenMixin):
    def __init__(self, data: pd.DataFrame, samples: list[str]):
        self.data = data
        self.samples = samples


@pytest.fixture
def tiny_variants_df():
    """Small dataframe exercising REF/ALT and a couple of sample genotype formats.

    Columns:
      - REF/ALT per row
      - S1, S2 sample genotypes with mixed separators and missing values
      - variant_classification mixed-case and with None
    """
    return pd.DataFrame({
        'REF': ['A', 'C', 'G', 'T'],
        'ALT': ['G', 'T', 'A', 'C'],
        'S1':  ['0|1', '0/0', '1/1', '.'],
        'S2':  ['A|G', 'C/T', 'G/A', 'T/T'],
        'variant_classification': ['missense_mutation', 'silent', None, 'NONSENSE_MUTATION']
    })
