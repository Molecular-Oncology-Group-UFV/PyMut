import types
import pytest
import duckdb
import pandas as pd

from src.pyMut.analysis.pfam_annotation import PfamAnnotationMixin


class DummyMeta:
    def __init__(self):
        self.pfam_resolution_stats = None


class DummyPM(PfamAnnotationMixin):
    def __init__(self, df: pd.DataFrame, metadata=None, samples=None):
        self.data = df
        self.metadata = metadata
        self.samples = samples


@pytest.fixture
def duckdb_conn():
    """In-memory DuckDB with minimal schema and example data.

    Tables:
      - xref(uniprot TEXT, short_name TEXT, prot_id TEXT)
      - pfam(uniprot TEXT, pfam_id TEXT, pfam_name TEXT, seq_start INTEGER, seq_end INTEGER)
    """
    con = duckdb.connect(database=':memory:')
    con.execute("CREATE TABLE xref(uniprot TEXT, short_name TEXT, prot_id TEXT)")
    con.execute(
        "CREATE TABLE pfam(uniprot TEXT, pfam_id TEXT, pfam_name TEXT, seq_start INTEGER, seq_end INTEGER)"
    )
    # Example rows suggested in the spec
    con.execute("INSERT INTO xref VALUES ('Q99613','1433B_HUMAN',NULL)")
    con.execute("INSERT INTO xref VALUES ('P31749',NULL,'ENSP00000354587')")
    con.execute("INSERT INTO pfam VALUES ('Q99613','PF00244','14-3-3 domain',100,120)")
    con.execute("INSERT INTO pfam VALUES ('P31946','PF00001','Example domain',10,50)")
    try:
        yield con
    finally:
        con.close()


@pytest.fixture
def df_variants_base():
    df = pd.DataFrame({
        'Hugo_Symbol': ['TP53', 'BRAF'],
        'UNIPROT': ['P04637.1', 'P15056'],
        'Protein_Change': ['p.Arg175His', 'p.Val600Glu'],
        'Domains': ['Pfam:PF08563', ''],
        'aa_pos': [175, 600],
    })
    return df
