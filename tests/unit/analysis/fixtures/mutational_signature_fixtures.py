import numpy as np
import pandas as pd
import pytest
from typing import Dict, Optional, Iterable

from src.pyMut.analysis.mutational_signature import TRINUCLEOTIDE_CONTEXTS


class _Slice:
    def __init__(self, seq: str):
        self.seq = seq


class _Chrom:
    def __init__(self, seq: str):
        self._seq = seq

    def __getitem__(self, s: slice) -> _Slice:
        # Python slicing semantics; indices provided in mutational_signature are 0-based like here
        return _Slice(self._seq[s.start:s.stop])


class DummyFasta:
    """A minimal stub that mimics pyfaidx.Fasta for testing.

    - keys(): returns available chromosome names.
    - __getitem__(chrom)[start:stop].seq returns substring in upper-case.
    """

    def __init__(self, sequences: Dict[str, str]):
        # Store sequences in uppercase as pyfaidx would
        self._seqs = {k: v.upper() for k, v in sequences.items()}

    def keys(self) -> Iterable[str]:
        return self._seqs.keys()

    def __getitem__(self, chrom: str) -> _Chrom:
        if chrom not in self._seqs:
            raise KeyError(chrom)
        return _Chrom(self._seqs[chrom])


@pytest.fixture
def dummy_fasta() -> DummyFasta:
    # Provide both naming schemes for chromosome 1
    # Construct an easy sequence where positions are predictable.
    # e.g., repeating pattern A C G T ... across 200 bases
    bases = ("ACGT" * 60)[:240]
    return DummyFasta({
        "1": bases,
        "chr1": bases,
    })


def dummy_fasta_with_Ns() -> DummyFasta:
    # Same as above but inject Ns around certain positions to simulate invalid context windows
    bases = list(("ACGT" * 60)[:240])
    # Put Ns around position 100 (1-based) so window contains N
    for i in [98, 99, 100]:  # 0-based indices ~ positions 99,100,101 1-based
        bases[i] = 'N'
    seq = ''.join(bases)
    return DummyFasta({"1": seq, "chr1": seq})


# Register a pytest fixture with the same public name that delegates to the factory above
@pytest.fixture(name="dummy_fasta_with_Ns")
def _dummy_fasta_with_Ns_fixture() -> DummyFasta:
    return dummy_fasta_with_Ns()


@pytest.fixture
def contexts_df_small() -> pd.DataFrame:
    # Build a 96 x 2 matrix with small POSITIVE integer counts in every
    # context row. extractSignatures() runs R's NMF::nmf under the hood, and
    # NMF::nmf errors out ("Input matrix x contains at least one null or
    # NA-filled row") when any of the 96 context rows is all zeros — a
    # mostly-zero fixture is therefore not usable for extractSignatures
    # tests. The deterministic patterns below keep every row > 0.
    idx = TRINUCLEOTIDE_CONTEXTS
    col1 = (np.arange(96) % 9) + 1
    col2 = ((np.arange(96) * 5) % 7) + 1
    df = pd.DataFrame({"S1": col1, "S2": col2}, index=idx)
    return df


def build_cosmic_df(columns: Optional[list] = None, shuffle_contexts: bool = True,
                    include_artifacts: bool = True) -> pd.DataFrame:
    """Helper to build a synthetic COSMIC-like DataFrame.

    The first column is the context label; subsequent columns are signatures.
    """
    rng = np.random.default_rng(42)
    contexts = list(TRINUCLEOTIDE_CONTEXTS)
    if shuffle_contexts:
        rng.shuffle(contexts)
    # Default signature column names
    if columns is None:
        columns = ["SBS1", "SBS5", "SBS17a"]
        if include_artifacts:
            columns += ["SBS45", "Artefact_X", "SBS7c"]  # will be filtered
    data = rng.random((96, len(columns)))
    df = pd.DataFrame(data, columns=columns)
    df.insert(0, "Type", contexts)
    return df
