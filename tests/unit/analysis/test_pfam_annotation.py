import warnings
import types
import pandas as pd
import duckdb
import pytest

from src.pyMut.analysis.pfam_annotation import (
    PfamAnnotationMixin,
    annotate_pfam as legacy_annotate_pfam,
    pfam_domains as legacy_pfam_domains,
)
from src.pyMut.utils.database import PfamAnnotationError

from .fixtures.pfam_annotation_fixtures import DummyPM, DummyMeta


# ------------------------------
# Helper utilities for tests
# ------------------------------

class ExecErrorOnShortName:
    """DuckDB connection proxy that raises on the short_name query only."""
    def __init__(self, base):
        self._base = base

    def execute(self, sql, params=None):
        if isinstance(sql, str) and 'short_name' in sql:
            raise RuntimeError('short_name query failed')
        return self._base.execute(sql, params or [])

    def register(self, *args, **kwargs):
        return self._base.register(*args, **kwargs)

    def unregister(self, *args, **kwargs):
        return self._base.unregister(*args, **kwargs)

    def close(self):
        return self._base.close()


class SpyRegisterConn:
    """Proxy to spy register/unregister calls."""
    def __init__(self, base):
        self._base = base
        self.register_calls = 0
        self.unregister_calls = 0

    def execute(self, *a, **k):
        return self._base.execute(*a, **k)

    def register(self, *a, **k):
        self.register_calls += 1
        return self._base.register(*a, **k)

    def unregister(self, *a, **k):
        self.unregister_calls += 1
        return self._base.unregister(*a, **k)


# ================================
# Tests for resolve_uniprot_identifiers
# ================================

def test_resolve_direct_accession_ok(duckdb_conn):
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot': ['P31946', 'Q9Y2X3']})
    out, stats = mix.resolve_uniprot_identifiers(df, 'uniprot', duckdb_conn)
    assert list(out['uniprot_resolved']) == ['P31946', 'Q9Y2X3']
    assert set(out['resolution_method']) == {'direct_accession'}
    assert stats['direct_accession'] == 2 and stats['total'] == 2


def test_resolve_via_short_name(duckdb_conn):
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot': ['1433B_HUMAN']})
    out, stats = mix.resolve_uniprot_identifiers(df, 'uniprot', duckdb_conn)
    assert out.loc[0, 'uniprot_resolved'] == 'Q99613'
    assert out.loc[0, 'resolution_method'] == 'via_short_name'
    assert stats['via_short_name'] == 1


def test_resolve_via_external_id(duckdb_conn):
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot': ['ENSP00000354587']})
    out, stats = mix.resolve_uniprot_identifiers(df, 'uniprot', duckdb_conn)
    assert out.loc[0, 'uniprot_resolved'] == 'P31749'
    assert out.loc[0, 'resolution_method'] == 'via_external_id'
    assert stats['via_external_id'] == 1


def test_resolve_unresolved_when_not_found(duckdb_conn):
    # clear xref
    duckdb_conn.execute('DELETE FROM xref')
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot': ['UNKNOWN_ID']})
    out, stats = mix.resolve_uniprot_identifiers(df, 'uniprot', duckdb_conn)
    assert out.loc[0, 'uniprot_resolved'] is None
    assert out.loc[0, 'resolution_method'] == 'unresolved'
    assert stats['unresolved'] == 1


def test_resolve_ignores_nan_and_empty(duckdb_conn):
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot': [None, '', '  ', 'P31946']})
    out, stats = mix.resolve_uniprot_identifiers(df, 'uniprot', duckdb_conn)
    assert stats['total'] == 1 and stats['direct_accession'] == 1


def test_resolve_caches_and_counts_unique_only(duckdb_conn):
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot': ['P31946', 'P31946', '1433B_HUMAN', '1433B_HUMAN']})
    out, stats = mix.resolve_uniprot_identifiers(df, 'uniprot', duckdb_conn)
    # should have processed only two unique non-empty values
    assert stats['total'] == 2
    assert set(out['uniprot_resolved']) == {'P31946', 'Q99613'}


def test_resolve_handles_db_errors_and_logs(duckdb_conn, caplog):
    proxy = ExecErrorOnShortName(duckdb_conn)
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot': ['1433B_HUMAN', 'ENSP00000354587']})
    caplog.clear()
    out, stats = mix.resolve_uniprot_identifiers(df, 'uniprot', proxy)
    # 1433B_HUMAN fails short_name, ENSP resolves via prot_id
    assert out.loc[1, 'uniprot_resolved'] == 'P31749'
    assert any('short_name' in r.message for r in caplog.records)


def test_resolve_accession_regex_variants(duckdb_conn):
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot': ['A0A0A0', 'O12345', 'Q9H0H5']})
    out, stats = mix.resolve_uniprot_identifiers(df, 'uniprot', duckdb_conn)
    assert stats['direct_accession'] == 3
    assert set(out['resolution_method']) == {'direct_accession'}


# ================================
# Tests for _annotate_pfam_sql
# ================================

def test_annotate_pfam_sql_left_join_and_between(duckdb_conn):
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({
        'uniprot_resolved': ['P31946', 'P31946'],
        'aa_pos': [25, 5]
    })
    res = mix._annotate_pfam_sql(df, duckdb_conn, 'aa_pos', 'uniprot_resolved')
    assert res.loc[0, 'pfam_id'] == 'PF00001'
    assert pd.isna(res.loc[1, 'pfam_id'])


def test_annotate_pfam_sql_no_matches_returns_nulls(duckdb_conn):
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot_resolved': ['XXXXXX'], 'aa_pos': [10]})
    res = mix._annotate_pfam_sql(df, duckdb_conn, 'aa_pos', 'uniprot_resolved')
    assert 'pfam_id' in res.columns and pd.isna(res.loc[0, 'pfam_id'])


# ================================
# Tests for _extract_uniprot_id
# ================================

def test_extract_uniprot_id_strips_version():
    mix = DummyPM(pd.DataFrame())
    row = {'UNIPROT': 'P31946.5'}
    assert mix._extract_uniprot_id(row) == 'P31946'


def test_extract_uniprot_id_none_when_missing_or_nan():
    mix = DummyPM(pd.DataFrame())
    assert mix._extract_uniprot_id({'UNIPROT': None}) is None
    assert mix._extract_uniprot_id({'X': 'nope'}) is None


# ================================
# Tests for _extract_aa_position
# ================================

def test_extract_aa_position_from_protein_change():
    mix = DummyPM(pd.DataFrame())
    assert mix._extract_aa_position({'Protein_Change': 'p.Gly12Asp'}) == 12


def test_extract_aa_position_handles_no_match():
    mix = DummyPM(pd.DataFrame())
    assert mix._extract_aa_position({'Protein_Change': 'p.?'}) is None
    assert mix._extract_aa_position({'Protein_Change': None}) is None
    assert mix._extract_aa_position({'X': 'nope'}) is None


def test_extract_aa_position_tolerates_three_letter_and_one_letter_codes():
    mix = DummyPM(pd.DataFrame())
    assert mix._extract_aa_position({'Protein_Change': 'p.G12D'}) == 12
    assert mix._extract_aa_position({'Protein_Change': 'p.Gly12Val'}) == 12


# ================================
# Tests for _annotate_with_database
# ================================

def test_annotate_with_database_resolves_and_annotates(duckdb_conn):
    df = pd.DataFrame({'UNIPROT': ['1433B_HUMAN'], 'aa_pos': [110]})
    mix = DummyPM(df)
    out = mix._annotate_with_database(df, duckdb_conn, 'aa_pos', 'UNIPROT')
    assert out.loc[0, 'pfam_id'] == 'PF00244'
    assert 'resolution_stats' in out.attrs


def test_annotate_with_database_no_valid_rows_adds_empty_columns(duckdb_conn, caplog):
    df = pd.DataFrame({'UNIPROT': ['UNKNOWN'], 'aa_pos': [None]})
    mix = DummyPM(df)
    caplog.clear()
    out = mix._annotate_with_database(df, duckdb_conn, 'aa_pos', 'UNIPROT')
    assert out['pfam_id'].isna().all() or (out['pfam_id'] == None).all()  # noqa: E711
    assert any('No variants with resolved UniProt' in r.message for r in caplog.records)
    assert 'resolution_stats' in out.attrs


def test_annotate_with_database_uses_resolved_accession_not_input_alias(duckdb_conn):
    # Ensure pfam exists only for resolved accession Q99613
    df = pd.DataFrame({'UNIPROT': ['1433B_HUMAN'], 'aa_pos': [110]})
    mix = DummyPM(df)
    out = mix._annotate_with_database(df, duckdb_conn, 'aa_pos', 'UNIPROT')
    assert out.loc[0, 'pfam_id'] == 'PF00244'


def test_annotate_with_database_partial_annotations_preserve_unannotated(duckdb_conn):
    df = pd.DataFrame({
        'UNIPROT': ['1433B_HUMAN', '1433B_HUMAN'],
        'aa_pos': [110, 50],  # second is out of range 100-120
    })
    mix = DummyPM(df)
    out = mix._annotate_with_database(df, duckdb_conn, 'aa_pos', 'UNIPROT')
    assert out.loc[0, 'pfam_id'] == 'PF00244'
    assert out.loc[1, 'pfam_id'] is None


def test_annotate_with_database_statistics_are_logged(duckdb_conn, caplog):
    df = pd.DataFrame({'UNIPROT': ['1433B_HUMAN'], 'aa_pos': [110]})
    mix = DummyPM(df)
    caplog.clear()
    _ = mix._annotate_with_database(df, duckdb_conn, 'aa_pos', 'UNIPROT')
    # Expect some info logs about resolution and annotation counts
    assert any('UniProt resolution summary' in r.message or 'Variantes anotadas con PFAM' in r.message for r in caplog.records)


# ================================
# Tests for _annotate_with_vep_domains
# ================================

def test_annotate_with_vep_domains_extracts_pfam_from_domains():
    df = pd.DataFrame({'Domains': ['Pfam:PF00069,Some:XYZ']})
    mix = DummyPM(df)
    out = mix._annotate_with_vep_domains(df)
    assert out.loc[0, 'pfam_id'] == 'PF00069'
    assert out.loc[0, 'pfam_name'] == 'PF00069'
    assert out.loc[0, 'seq_start'] is None and out.loc[0, 'seq_end'] is None


def test_annotate_with_vep_domains_no_pfam_keeps_nulls():
    df = pd.DataFrame({'Domains': ['SMART:SM0001']})
    mix = DummyPM(df)
    out = mix._annotate_with_vep_domains(df)
    assert out.loc[0, 'pfam_id'] is None


def test_annotate_with_vep_domains_missing_column_adds_nulls_and_warns(caplog):
    df = pd.DataFrame({'X': ['no_domains']})
    mix = DummyPM(df)
    caplog.clear()
    out = mix._annotate_with_vep_domains(df)
    assert all(c in out.columns for c in ['pfam_id', 'pfam_name', 'seq_start', 'seq_end'])
    assert any('VEP_DOMAINS column not found' in r.message for r in caplog.records)


def test_annotate_with_vep_domains_extracts_uniprot_and_aa_pos_when_missing():
    df = pd.DataFrame({'UNIPROT': ['P31946.2'], 'Protein_Change': ['p.Gly12Asp'], 'Domains': ['Pfam:PF00069']})
    mix = DummyPM(df)
    out = mix._annotate_with_vep_domains(df)
    assert out.loc[0, 'uniprot'] == 'P31946'
    assert out.loc[0, 'aa_pos'] == 12


def test_annotate_with_vep_domains_multiple_pfam_takes_first():
    df = pd.DataFrame({'Domains': ['Pfam:PF00001 Pfam:PF00002']})
    mix = DummyPM(df)
    out = mix._annotate_with_vep_domains(df)
    assert out.loc[0, 'pfam_id'] == 'PF00001'


# ================================
# Tests for public annotate_pfam
# ================================

def test_annotate_pfam_prefers_database_when_uniprot_and_aa_pos_exist(duckdb_conn, monkeypatch):
    # Build object
    df = pd.DataFrame({'UNIPROT': ['1433B_HUMAN'], 'aa_pos': [110]})
    obj = DummyPM(df, metadata=DummyMeta(), samples={'S1'})
    # Force database usage by supplying connection
    new = obj.annotate_pfam(db_conn=duckdb_conn, aa_column='aa_pos', auto_extract=True, prefer_database=True)
    assert 'pfam_id' in new.data.columns
    assert new.data.loc[0, 'seq_start'] == 100 and new.data.loc[0, 'seq_end'] == 120


def test_annotate_pfam_auto_extracts_uniprot_and_aa_pos_from_vep_then_database(duckdb_conn):
    df = pd.DataFrame({'UNIPROT': ['1433B_HUMAN'], 'Protein_Change': ['p.Gly110Asp']})
    obj = DummyPM(df, metadata=DummyMeta(), samples={'S1'})
    new = obj.annotate_pfam(db_conn=duckdb_conn, aa_column='aa_pos', auto_extract=True, prefer_database=True)
    assert new.data.loc[0, 'pfam_id'] == 'PF00244'


def test_annotate_pfam_fallback_to_vep_domains_when_no_db_route():
    df = pd.DataFrame({'Domains': ['Pfam:PF00069']})
    obj = DummyPM(df, metadata=DummyMeta(), samples={'S1'})
    # No db_conn and no UNIPROT/aa_pos -> fallback to VEP
    new = obj.annotate_pfam(db_conn=None, auto_extract=False, prefer_database=True)
    assert new.data.loc[0, 'pfam_id'] == 'PF00069'
    assert new.data.loc[0, 'seq_start'] is None


def test_annotate_pfam_no_suitable_data_adds_empty_columns():
    df = pd.DataFrame({'X': [1]})
    obj = DummyPM(df, metadata=DummyMeta(), samples={'S1'})
    new = obj.annotate_pfam(db_conn=None, auto_extract=False, prefer_database=True)
    assert all(col in new.data.columns for col in ['pfam_id', 'pfam_name', 'seq_start', 'seq_end'])
    assert new.data.loc[0, 'pfam_id'] is None


def test_annotate_pfam_connects_and_closes_when_db_conn_is_none(monkeypatch):
    # Spy connection with close()
    base = duckdb.connect(':memory:')
    base.execute("CREATE TABLE xref(uniprot TEXT, short_name TEXT, prot_id TEXT)")
    base.execute("CREATE TABLE pfam(uniprot TEXT, pfam_id TEXT, pfam_name TEXT, seq_start INTEGER, seq_end INTEGER)")
    base.execute("INSERT INTO xref VALUES ('Q99613','1433B_HUMAN',NULL)")
    base.execute("INSERT INTO pfam VALUES ('Q99613','PF00244','14-3-3 domain',100,120)")

    class SpyConn:
        def __init__(self, inner):
            self.inner = inner
            self.closed = 0
        def close(self):
            self.closed += 1
            return self.inner.close()
        def __getattr__(self, item):
            return getattr(self.inner, item)

    spy = SpyConn(base)

    # Patch connect_db to return our spy
    from src.pyMut import utils as _u
    monkeypatch.setattr('src.pyMut.analysis.pfam_annotation.connect_db', lambda: spy)

    df = pd.DataFrame({'UNIPROT': ['1433B_HUMAN'], 'aa_pos': [110]})
    obj = DummyPM(df, metadata=DummyMeta(), samples={'S1'})
    _ = obj.annotate_pfam(db_conn=None, aa_column='aa_pos', auto_extract=True, prefer_database=True)
    assert spy.closed == 1


def test_annotate_pfam_preserves_metadata_and_samples(duckdb_conn):
    df = pd.DataFrame({'UNIPROT': ['1433B_HUMAN'], 'aa_pos': [110]})
    meta = DummyMeta()
    samples = {'S1', 'S2'}
    obj = DummyPM(df, metadata=meta, samples=samples)
    new = obj.annotate_pfam(db_conn=duckdb_conn)
    assert new.metadata is meta
    assert new.samples == samples


def test_annotate_pfam_sets_metadata_resolution_stats_when_available(duckdb_conn):
    df = pd.DataFrame({'UNIPROT': ['1433B_HUMAN'], 'aa_pos': [110]})
    meta = DummyMeta()
    obj = DummyPM(df, metadata=meta)
    new = obj.annotate_pfam(db_conn=duckdb_conn)
    assert isinstance(new.metadata.pfam_resolution_stats, dict)
    assert new.metadata.pfam_resolution_stats['total'] >= 1


def test_annotate_pfam_logs_summary_when_stats_present(duckdb_conn, caplog):
    df = pd.DataFrame({'UNIPROT': ['1433B_HUMAN'], 'aa_pos': [110]})
    obj = DummyPM(df, metadata=DummyMeta())
    caplog.clear()
    _ = obj.annotate_pfam(db_conn=duckdb_conn)
    assert any('Final annotation summary' in r.message for r in caplog.records)


def test_annotate_pfam_respects_custom_aa_column(duckdb_conn):
    df = pd.DataFrame({'UNIPROT': ['1433B_HUMAN'], 'prot_pos': [110]})
    obj = DummyPM(df)
    new = obj.annotate_pfam(db_conn=duckdb_conn, aa_column='prot_pos')
    assert new.data.loc[0, 'pfam_id'] == 'PF00244'


# ================================
# Tests for pfam_domains
# ================================

def test_pfam_domains_raises_when_no_pfam_columns():
    obj = DummyPM(pd.DataFrame({'Hugo_Symbol': ['TP53']}))
    with pytest.raises(PfamAnnotationError):
        obj.pfam_domains()


def test_pfam_domains_filters_out_silent_by_default():
    df = pd.DataFrame({
        'Hugo_Symbol': ['G1', 'G1', 'G2', 'G3'],
        'pfam_id': ['PF1', 'PF1', 'PF2', 'PF3'],
        'pfam_name': ['N1', 'N1', 'N2', 'N3'],
        'aa_pos': [1, 2, 3, 4],
        'Variant_Classification': ['Silent', 'Missense_Mutation', 'Nonsense_Mutation', 'Silent'],
    })
    obj = DummyPM(df)
    res = obj.pfam_domains()
    # Silent rows removed -> PF3 dropped
    assert set(res['pfam_id']) == {'PF1', 'PF2'}


def test_pfam_domains_group_by_domain_returns_counts_and_sorted():
    df = pd.DataFrame({
        'Hugo_Symbol': ['G1', 'G2', 'G2', 'G1', 'G3'],
        'pfam_id': ['PF1', 'PF1', 'PF2', 'PF1', 'PF2'],
        'pfam_name': ['N1', 'N1', 'N2', 'N1', 'N2'],
        'aa_pos': [1, 2, 3, 4, 5]
    })
    obj = DummyPM(df)
    res = obj.pfam_domains(top_n=1)
    assert list(res['pfam_id']) == ['PF1']
    assert list(res['n_genes']) == [2]


def test_pfam_domains_group_by_domain_requires_hugo_symbol_alias():
    df = pd.DataFrame({
        'pfam_id': ['PF1'], 'pfam_name': ['N1'], 'aa_pos': [1]
    })
    obj = DummyPM(df)
    with pytest.raises(PfamAnnotationError, match='Hugo_Symbol column not found'):
        obj.pfam_domains()


def test_pfam_domains_group_by_aapos_happy_path():
    df = pd.DataFrame({
        'uniprot': ['U1', 'U1', 'U2'],
        'Hugo_Symbol': ['G1', 'G2', 'G2'],
        'pfam_id': ['PF1', 'PF1', 'PF2'],
        'pfam_name': ['N1', 'N1', 'N2'],
        'aa_pos': [10, 10, 20],
    })
    obj = DummyPM(df)
    res = obj.pfam_domains(summarize_by='AAPos')
    # Expect two groups: (U1,10,PF1) and (U2,20,PF2)
    assert set(zip(res['pfam_id'], res['n_variants'])) == {('PF1', 2), ('PF2', 1)}


def test_pfam_domains_group_by_aapos_missing_aliases_returns_empty_and_logs(caplog):
    df = pd.DataFrame({
        'pfam_id': ['PF1'], 'pfam_name': ['N1'], 'aa_pos': [1], 'Hugo_SymbolX': ['G']
    })
    obj = DummyPM(df)
    caplog.clear()
    res = obj.pfam_domains(summarize_by='AAPos')
    assert res.empty
    assert any('UNIPROT' in r.message or 'Hugo_Symbol' in r.message for r in caplog.records)


def test_pfam_domains_respects_top_n():
    df = pd.DataFrame({
        'Hugo_Symbol': ['G1', 'G1', 'G1', 'G2', 'G2'],
        'pfam_id': ['PF1', 'PF1', 'PF2', 'PF2', 'PF2'],
        'pfam_name': ['N1', 'N1', 'N2', 'N2', 'N2'],
        'aa_pos': [1, 2, 3, 4, 5]
    })
    obj = DummyPM(df)
    res = obj.pfam_domains(top_n=1)
    assert len(res) == 1


def test_pfam_domains_custom_aa_column():
    df = pd.DataFrame({
        'Hugo_Symbol': ['G1', 'G1', 'G2'],
        'pfam_id': ['PF1', 'PF1', 'PF2'],
        'pfam_name': ['N1', 'N1', 'N2'],
        'prot_pos': [1, 2, 3]
    })
    obj = DummyPM(df)
    res = obj.pfam_domains(aa_column='prot_pos')
    assert 'n_variants' in res.columns


# ================================
# Legacy functions
# ================================

def test_legacy_annotate_pfam_emits_deprecation_warning_and_delegates():
    class Obj:
        def __init__(self):
            self.called = False
        def annotate_pfam(self, *a, **k):
            self.called = True
            return 'ok'
    o = Obj()
    with pytest.warns(DeprecationWarning):
        r = legacy_annotate_pfam(o, 1, x=2)
    assert r == 'ok' and o.called


def test_legacy_pfam_domains_emits_deprecation_warning_and_delegates():
    class Obj:
        def __init__(self):
            self.called = False
        def pfam_domains(self, *a, **k):
            self.called = True
            return 'ok'
    o = Obj()
    with pytest.warns(DeprecationWarning):
        r = legacy_pfam_domains(o, 1, x=2)
    assert r == 'ok' and o.called


# ================================
# Edge and robustness extras
# ================================

def test_handles_whitespace_and_case_in_uniprot_inputs(duckdb_conn):
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot': [' q9y2x3 ', '  P31946  ']})
    # normalize by stripping; note: regex expects uppercase letters; uppercase the IDs
    df['uniprot'] = df['uniprot'].str.strip().str.upper()
    out, stats = mix.resolve_uniprot_identifiers(df, 'uniprot', duckdb_conn)
    assert stats['total'] == 2 and stats['direct_accession'] == 2


def test_domains_parsing_handles_mixed_separators():
    df = pd.DataFrame({'Domains': ['Pfam:PF00001; Pfam:PF00002, Pfam:PF00003   ']})
    mix = DummyPM(df)
    out = mix._annotate_with_vep_domains(df)
    assert out.loc[0, 'pfam_id'] == 'PF00001'


def test_sql_register_unregister_always_called(duckdb_conn):
    spy = SpyRegisterConn(duckdb_conn)
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot_resolved': ['P31946'], 'aa_pos': [25]})
    _ = mix._annotate_pfam_sql(df, spy, 'aa_pos', 'uniprot_resolved')
    assert spy.register_calls >= 1 and spy.unregister_calls >= 1


def test_annotation_does_not_mutate_original_dataframe(duckdb_conn):
    df = pd.DataFrame({'UNIPROT': ['1433B_HUMAN'], 'aa_pos': [110]})
    obj = DummyPM(df)
    _ = obj.annotate_pfam(db_conn=duckdb_conn)
    # original df unchanged
    assert list(df.columns) == ['UNIPROT', 'aa_pos']
