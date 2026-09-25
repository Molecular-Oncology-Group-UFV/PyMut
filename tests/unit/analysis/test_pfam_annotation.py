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

from .fixtures.pfam_annotation_fixtures import DummyPM, DummyMeta, gene_domain_table


# NOTE (API update): annotate_pfam() now dispatches on a keyword-only
# ``strategy`` argument:
#   - strategy='basic' (default): maftools-equivalent mapping via gene symbol
#     (Hugo_Symbol) + HGVS protein change (HGVSp_Short / Protein_Change /
#     AAChange / HGVSp) against the bundled gene->Pfam table. No fallbacks;
#     raises ValueError if the required columns or the table are missing.
#   - strategy='extended': layered gene-symbol + VEP-UniProt mapping plus a
#     VEP-domains / text-parse cascade. Requires VEP_SWISSPROT/TREMBL/UNIPROT
#     and VEP_Protein_position/Protein_position columns.
# The old keyword arguments (aa_column, auto_extract, prefer_database) no
# longer exist. aa_column resolution only lives in pfam_domains() now.

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
    assert stats['total'] == 2
    assert set(out['uniprot_resolved']) == {'P31946', 'Q99613'}


def test_resolve_handles_db_errors_and_logs(duckdb_conn, caplog):
    proxy = ExecErrorOnShortName(duckdb_conn)
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot': ['1433B_HUMAN', 'ENSP00000354587']})
    caplog.clear()
    out, stats = mix.resolve_uniprot_identifiers(df, 'uniprot', proxy)
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


def test_sql_register_unregister_always_called(duckdb_conn):
    spy = SpyRegisterConn(duckdb_conn)
    mix = DummyPM(pd.DataFrame())
    df = pd.DataFrame({'uniprot_resolved': ['P31946'], 'aa_pos': [25]})
    _ = mix._annotate_pfam_sql(df, spy, 'aa_pos', 'uniprot_resolved')
    assert spy.register_calls >= 1 and spy.unregister_calls >= 1


# ================================
# Tests for _parse_hgvs_aa_position (replaces the old
# _extract_uniprot_id / _extract_aa_position helpers, which were removed
# when annotate_pfam() switched to strategy-based dispatch)
# ================================

def test_parse_hgvs_basic_one_and_three_letter():
    mix = DummyPM(pd.DataFrame())
    assert mix._parse_hgvs_aa_position('p.Gly12Asp') == 12
    assert mix._parse_hgvs_aa_position('p.G12D') == 12
    assert mix._parse_hgvs_aa_position('p.Gly12Val') == 12


def test_parse_hgvs_strips_transcript_prefix_and_ranges():
    mix = DummyPM(pd.DataFrame())
    assert mix._parse_hgvs_aa_position('ENST00000369535.4:p.Arg882His') == 882
    assert mix._parse_hgvs_aa_position('p.700_704del') == 700


def test_parse_hgvs_stop_codon_and_negative_position():
    mix = DummyPM(pd.DataFrame())
    assert mix._parse_hgvs_aa_position('p.R882*') == 882
    # mirrors maftools: a malformed negative position parses as a negative
    # number, which can never fall inside a domain range (Start/End >= 1)
    assert mix._parse_hgvs_aa_position('p.-169fs') == -169


def test_parse_hgvs_returns_none_when_no_position():
    mix = DummyPM(pd.DataFrame())
    assert mix._parse_hgvs_aa_position('p.?') is None
    assert mix._parse_hgvs_aa_position(None) is None
    assert mix._parse_hgvs_aa_position(float('nan')) is None
    assert mix._parse_hgvs_aa_position('') is None


# ================================
# Tests for _annotate_with_database (used by strategy='extended')
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
    assert out['pfam_id'].isna().all()
    assert any('No variants with resolved UniProt' in r.message for r in caplog.records)
    assert 'resolution_stats' in out.attrs


def test_annotate_with_database_uses_resolved_accession_not_input_alias(duckdb_conn):
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
    assert any('UniProt resolution summary' in r.message for r in caplog.records)


# ================================
# Tests for _annotate_with_vep_domains (cascade step 2 of 'extended')
# ================================

def test_annotate_with_vep_domains_extracts_pfam_from_domains():
    df = pd.DataFrame({'Domains': ['Pfam:PF00069,Some:XYZ']})
    mix = DummyPM(df)
    out = mix._annotate_with_vep_domains(df)
    assert out.loc[0, 'pfam_id'] == 'PF00069'
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


def test_domains_parsing_handles_mixed_separators():
    df = pd.DataFrame({'Domains': ['Pfam:PF00001; Pfam:PF00002, Pfam:PF00003   ']})
    mix = DummyPM(df)
    out = mix._annotate_with_vep_domains(df)
    assert out.loc[0, 'pfam_id'] == 'PF00001'


# ================================
# Tests for strategy='basic' (default; maftools-equivalent)
# ================================

def _patch_gene_table(monkeypatch, table):
    monkeypatch.setattr(
        PfamAnnotationMixin, '_load_gene_domain_table',
        lambda self: table, raising=True,
    )


def test_basic_strategy_maps_via_gene_symbol_and_protein_change(duckdb_conn, gene_domain_table, monkeypatch):
    _patch_gene_table(monkeypatch, gene_domain_table)
    df = pd.DataFrame({
        'Hugo_Symbol': ['BRAF', 'BRAF', 'TP53'],
        'Protein_Change': ['p.Val150Glu', 'p.Val600Glu', 'p.Pro30Leu'],
    })
    obj = DummyPM(df, metadata=DummyMeta(), samples={'S1'})
    new = obj.annotate_pfam(db_conn=duckdb_conn, strategy='basic')

    # The DuckDB join does not guarantee input row order -> assert by content,
    # not by position, and check that no row got duplicated by the join.
    assert len(new.data) == 3

    def _row(change):
        m = new.data[new.data['Protein_Change'] == change]
        assert len(m) == 1, f"expected exactly one row for {change}"
        return m.iloc[0]

    # p.Val150Glu -> aa 150 -> BRAF Prot_kinase (100-200)
    r = _row('p.Val150Glu')
    assert r['pfam_id'] == 'PF00001'
    assert r['pfam_name'] == 'Prot_kinase'
    assert r['seq_start'] == 100 and r['seq_end'] == 200
    # p.Val600Glu -> aa 600 -> out of every BRAF domain range -> unmapped
    assert pd.isna(_row('p.Val600Glu')['pfam_id'])
    # p.Pro30Leu -> aa 30 -> TP53 P53 (10-50)
    assert _row('p.Pro30Leu')['pfam_id'] == 'PF00003'


def test_basic_strategy_prefers_hgvsp_short_over_protein_change(duckdb_conn, gene_domain_table, monkeypatch):
    _patch_gene_table(monkeypatch, gene_domain_table)
    # HGVSp_Short wins over Protein_Change in the candidate order, exactly
    # like maftools' AACol default (c("HGVSp_Short", "Protein_Change", ...)).
    df = pd.DataFrame({
        'Hugo_Symbol': ['BRAF'],
        'HGVSp_Short': ['p.Val150Asp'],
        'Protein_Change': ['p.Val600Glu'],
    })
    obj = DummyPM(df)
    new = obj.annotate_pfam(db_conn=duckdb_conn, strategy='basic')
    assert new.data.loc[0, 'pfam_id'] == 'PF00001'  # position 150 was used, not 600


def test_basic_strategy_requires_gene_and_protein_change_columns(duckdb_conn):
    df = pd.DataFrame({'X': [1]})
    obj = DummyPM(df)
    with pytest.raises(ValueError, match="strategy='basic' requires"):
        obj.annotate_pfam(db_conn=duckdb_conn, strategy='basic')


def test_basic_strategy_requires_bundled_domain_table(duckdb_conn, monkeypatch):
    monkeypatch.setattr(
        PfamAnnotationMixin, '_load_gene_domain_table', lambda self: None, raising=True,
    )
    df = pd.DataFrame({'Hugo_Symbol': ['BRAF'], 'Protein_Change': ['p.Val150Glu']})
    obj = DummyPM(df)
    with pytest.raises(ValueError, match='requires the bundled gene->Pfam domain table'):
        obj.annotate_pfam(db_conn=duckdb_conn, strategy='basic')


def test_basic_strategy_records_columns_in_metadata(duckdb_conn, gene_domain_table, monkeypatch):
    _patch_gene_table(monkeypatch, gene_domain_table)
    df = pd.DataFrame({'Hugo_Symbol': ['BRAF'], 'Protein_Change': ['p.Val150Glu']})
    meta = DummyMeta()
    obj = DummyPM(df, metadata=meta)
    new = obj.annotate_pfam(db_conn=duckdb_conn, strategy='basic')
    assert new.metadata is meta
    assert new.metadata.pfam_columns['strategy'] == 'basic'
    assert new.metadata.pfam_columns['hugo_column'] == 'Hugo_Symbol'
    assert new.metadata.pfam_columns['aachange_column'] == 'Protein_Change'


def test_basic_strategy_logs_mapping_summary(duckdb_conn, gene_domain_table, monkeypatch, caplog):
    _patch_gene_table(monkeypatch, gene_domain_table)
    df = pd.DataFrame({'Hugo_Symbol': ['BRAF'], 'Protein_Change': ['p.Val150Glu']})
    obj = DummyPM(df)
    caplog.clear()
    with caplog.at_level('INFO'):
        obj.annotate_pfam(db_conn=duckdb_conn, strategy='basic')
    assert any("strategy='basic'" in r.message and 'mapped to a Pfam domain' in r.message
               for r in caplog.records)


def test_basic_strategy_connects_and_closes_db_when_conn_is_none(gene_domain_table, monkeypatch):
    # duckdb connections are C-extension objects: attributes cannot be
    # reassigned, so spy on close() through a delegating proxy instead.
    class CountingConn:
        def __init__(self, base):
            self._base = base
            self.closed = 0

        def close(self):
            self.closed += 1
            return self._base.close()

        def __getattr__(self, name):
            return getattr(self._base, name)

    created = []

    def _factory():
        conn = CountingConn(duckdb.connect(':memory:'))
        created.append(conn)
        return conn

    monkeypatch.setattr(
        'src.pyMut.analysis.pfam_annotation.connect_db', _factory,
    )
    _patch_gene_table(monkeypatch, gene_domain_table)

    df = pd.DataFrame({'Hugo_Symbol': ['BRAF'], 'Protein_Change': ['p.Val150Glu']})
    obj = DummyPM(df, metadata=DummyMeta(), samples={'S1'})
    _ = obj.annotate_pfam(db_conn=None, strategy='basic')
    assert len(created) == 1 and created[0].closed == 1


# ================================
# Tests for strategy='extended'
# ================================

def test_extended_strategy_maps_gene_and_uniprot_with_conflict_policy(
    duckdb_conn, gene_domain_table, monkeypatch
):
    _patch_gene_table(monkeypatch, gene_domain_table)
    # VEP-derived UniProt path (Q99613, 110 -> PF00244 via the duckdb fixture)
    # and gene-symbol path (YWHAB not in the fake gene table -> unmapped there)
    df = pd.DataFrame({
        'Hugo_Symbol': ['YWHAB'],
        'Protein_Change': ['p.Gly110Ala'],
        'VEP_SWISSPROT': ['1433B_HUMAN'],
        'VEP_Protein_position': ['110'],
    })
    meta = DummyMeta()
    obj = DummyPM(df, metadata=meta)
    new = obj.annotate_pfam(db_conn=duckdb_conn, strategy='extended')

    assert new.data.loc[0, 'pfam_id'] == 'PF00244'
    assert new.data.loc[0, 'pfam_mapping_source'] == 'uniprot_db'
    assert new.metadata.pfam_columns['strategy'] == 'extended'
    assert 'category_counts' in new.metadata.pfam_columns
    assert new.metadata.pfam_columns['category_counts'].get('uniprot_db', 0) >= 1


def test_extended_strategy_requires_vep_columns(duckdb_conn):
    df = pd.DataFrame({'Hugo_Symbol': ['BRAF'], 'Protein_Change': ['p.Val150Glu']})
    obj = DummyPM(df)
    with pytest.raises(ValueError, match="strategy='extended' requires VEP-annotated data"):
        obj.annotate_pfam(db_conn=duckdb_conn, strategy='extended')


def test_annotate_pfam_rejects_unknown_strategy(duckdb_conn):
    df = pd.DataFrame({'Hugo_Symbol': ['BRAF'], 'Protein_Change': ['p.Val150Glu']})
    obj = DummyPM(df)
    with pytest.raises(ValueError, match="strategy must be 'basic' or 'extended'"):
        obj.annotate_pfam(db_conn=duckdb_conn, strategy='nope')


def test_annotation_does_not_mutate_original_dataframe(duckdb_conn, gene_domain_table, monkeypatch):
    _patch_gene_table(monkeypatch, gene_domain_table)
    df = pd.DataFrame({'Hugo_Symbol': ['BRAF'], 'Protein_Change': ['p.Val150Glu']})
    obj = DummyPM(df)
    _ = obj.annotate_pfam(db_conn=duckdb_conn, strategy='basic')
    assert list(df.columns) == ['Hugo_Symbol', 'Protein_Change']


# ================================
# Tests for pfam_domains()
# ================================

def test_pfam_domains_raises_when_no_pfam_columns():
    # aa_pos is provided so column resolution succeeds and the error is
    # specifically about the missing annotation columns.
    obj = DummyPM(pd.DataFrame({'Hugo_Symbol': ['TP53'], 'aa_pos': [1]}))
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


def test_pfam_domains_group_by_domain_returns_counts_sorted_and_summary_columns():
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
    # Summary carries the newer descriptive columns as well
    for col in ['n_variants', 'pct_of_mapped', 'pct_of_considered', 'display_name', 'protein_breakdown']:
        assert col in res.columns


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


def test_pfam_domains_uses_aa_column_recorded_by_annotate_pfam(
    duckdb_conn, gene_domain_table, monkeypatch
):
    _patch_gene_table(monkeypatch, gene_domain_table)
    df = pd.DataFrame({'Hugo_Symbol': ['BRAF'], 'Protein_Change': ['p.Val150Glu']})
    meta = DummyMeta()
    obj = DummyPM(df, metadata=meta)
    annotated = obj.annotate_pfam(db_conn=duckdb_conn, strategy='basic')
    # pfam_domains() should pick up 'aa_pos' from metadata.pfam_columns
    res = annotated.pfam_domains()
    assert len(res) == 1
    assert res.loc[0, 'pfam_id'] == 'PF00001'


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
    df['uniprot'] = df['uniprot'].str.strip().str.upper()
    out, stats = mix.resolve_uniprot_identifiers(df, 'uniprot', duckdb_conn)
    assert stats['total'] == 2 and stats['direct_accession'] == 2


def test_resolve_hugo_column_candidates():
    mix = DummyPM(pd.DataFrame())
    assert mix._resolve_hugo_column(pd.DataFrame(columns=['Hugo_Symbol'])) == 'Hugo_Symbol'
    assert mix._resolve_hugo_column(pd.DataFrame(columns=['gene_symbol'])) == 'gene_symbol'
    assert mix._resolve_hugo_column(pd.DataFrame(columns=['X'])) is None


def test_resolve_aachange_column_priority_order():
    mix = DummyPM(pd.DataFrame())
    assert mix._resolve_aachange_column(pd.DataFrame(columns=['HGVSp_Short', 'Protein_Change'])) == 'HGVSp_Short'
    assert mix._resolve_aachange_column(pd.DataFrame(columns=['Protein_Change', 'AAChange'])) == 'Protein_Change'
    assert mix._resolve_aachange_column(pd.DataFrame(columns=['AAChange'])) == 'AAChange'
    assert mix._resolve_aachange_column(pd.DataFrame(columns=['HGVSp'])) == 'HGVSp'
    assert mix._resolve_aachange_column(pd.DataFrame(columns=['X'])) is None
