import pytest
import duckdb
import pandas as pd

from src.pyMut.analysis.pfam_annotation import PfamAnnotationMixin


class DummyMeta:
    def __init__(self):
        # pfam_columns is set in-place by annotate_pfam() on the metadata it
        # receives (the same object is passed through to the new PyMutation).
        self.pfam_columns = None
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
    con.execute("INSERT INTO xref VALUES ('Q99613','1433B_HUMAN',NULL)")
    con.execute("INSERT INTO xref VALUES ('P31749',NULL,'ENSP00000354587')")
    con.execute("INSERT INTO pfam VALUES ('Q99613','PF00244','14-3-3 domain',100,120)")
    con.execute("INSERT INTO pfam VALUES ('P31946','PF00001','Example domain',10,50)")
    try:
        yield con
    finally:
        con.close()


@pytest.fixture
def gene_domain_table():
    """Small stand-in for the bundled gene -> Pfam domain table
    (data/pfam_domains.csv), used via monkeypatched _load_gene_domain_table()
    so the basic-strategy tests stay hermetic."""
    return pd.DataFrame({
        'HGNC': ['BRAF', 'BRAF', 'TP53'],
        'Start': [100, 300, 10],
        'End': [200, 400, 50],
        'Label': ['Prot_kinase', 'Linker', 'P53'],
        'pfam': ['PF00001', 'PF00002', 'PF00003'],
        'Description': ['Protein kinase domain', 'Linker region', 'P53 domain'],
    })


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
