import os
import re
import math
import pandas as pd
import numpy as np
import pytest

from src.pyMut.analysis.mutation_burden import MutationBurdenMixin, log_tmb_summary
from .fixtures.mutation_burden_fixtures import DummyPyMutation, tiny_variants_df

# NOTE (API update): calculate_tmb_analysis() now returns the maftools-style
# schema:
#   analysis   : Tumor_Sample_Barcode, total, total_perMB, total_perMB_log
#   statistics : Metric, Count, Median, Mean, Min, Max, Q1.25., Q3.75., Std
# `total` counts NON-SYNONYMOUS mutations only (like maftools::tmb()); the
# old Total_Mutations / Non_Synonymous_Mutations split no longer exists.
# Rows without a variant-classification column, or with synonymous/None
# classifications, do not contribute to `total`.

SAMPLE_COL = 'Tumor_Sample_Barcode'


# -------------------------------
# A. Input validation
# -------------------------------

def test_fails_when_missing_data_or_samples_attribute():
    class Obj(MutationBurdenMixin):
        def __init__(self):
            pass
    with pytest.raises(ValueError, match="missing 'data' or 'samples'"):
        Obj().calculate_tmb_analysis()


def test_fails_when_data_is_empty():
    df = pd.DataFrame()
    dummy = DummyPyMutation(df, samples=['S1'])
    with pytest.raises(ValueError, match='data is empty'):
        dummy.calculate_tmb_analysis()


def test_fails_when_samples_is_empty():
    df = pd.DataFrame({'REF': ['A'], 'ALT': ['G'], 'S1': ['0|1']})
    dummy = DummyPyMutation(df, samples=[])
    with pytest.raises(ValueError, match='No samples found'):
        dummy.calculate_tmb_analysis()


def test_fails_when_missing_required_columns():
    df1 = pd.DataFrame({'ALT': ['G'], 'S1': ['0|1']})
    df2 = pd.DataFrame({'REF': ['A'], 'S1': ['0|1']})
    df3 = pd.DataFrame({'S1': ['0|1']})
    for df in (df1, df2, df3):
        dummy = DummyPyMutation(df, samples=['S1'])
        with pytest.raises(ValueError, match='Missing required columns'):
            dummy.calculate_tmb_analysis()


def test_fails_when_provided_variant_classification_column_missing(tiny_variants_df):
    dummy = DummyPyMutation(tiny_variants_df.drop(columns=['variant_classification']), samples=['S1'])
    with pytest.raises(ValueError, match="Column provide 'foo' not found"):
        dummy.calculate_tmb_analysis(variant_classification_column='foo')


# -----------------------------------------------------
# B. Auto-detect variant classification column
# -----------------------------------------------------

def test_autodetects_basic_variant_classification_column(tiny_variants_df, caplog):
    dummy = DummyPyMutation(tiny_variants_df, samples=['S1'])
    caplog.clear()
    with caplog.at_level('INFO'):
        out = dummy.calculate_tmb_analysis(save_files=False)
    assert any('Auto-detected variant classification column' in r.message for r in caplog.records)
    assert 'analysis' in out and 'statistics' in out


def test_autodetects_gencode_prefix(caplog):
    df = pd.DataFrame({
        'REF': ['A'], 'ALT': ['G'], 'S1': ['0|1'], 'gencode_28_variant_classification': ['missense_mutation']
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    caplog.clear()
    with caplog.at_level('INFO'):
        dummy.calculate_tmb_analysis(save_files=False)
    assert any('Auto-detected' in r.message and 'gencode_28_variant_classification' in r.message for r in caplog.records)


def test_autodetects_case_and_underscore_variants(caplog):
    for col in ['variantclassification', 'VaRiAnT_ClAsSiFiCaTiOn']:
        df = pd.DataFrame({'REF': ['A'], 'ALT': ['G'], 'S1': ['0|1'], col: ['missense_mutation']})
        dummy = DummyPyMutation(df, samples=['S1'])
        caplog.clear()
        with caplog.at_level('INFO'):
            dummy.calculate_tmb_analysis(save_files=False)
        assert any('Auto-detected' in r.message for r in caplog.records)


def test_multiple_matching_columns_uses_first_and_logs(caplog):
    # Order matters: the first column should be chosen
    df = pd.DataFrame({
        'REF': ['A'], 'ALT': ['G'], 'S1': ['0|1'],
        'variant_classification': ['missense_mutation'],
        'gencode_29_variant_classification': ['silent']
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    caplog.clear()
    with caplog.at_level('INFO'):
        out = dummy.calculate_tmb_analysis(save_files=False)
    assert any('Auto-detected' in r.message and 'variant_classification' in r.message for r in caplog.records)
    # First column marks the row as non-synonymous, so total (non-syn) == 1
    assert int(out['analysis'].loc[0, 'total']) == 1


def test_no_variant_column_sets_non_synonymous_to_zero_and_warns(caplog):
    df = pd.DataFrame({'REF': ['A'], 'ALT': ['G'], 'S1': ['0|1']})
    dummy = DummyPyMutation(df, samples=['S1'])
    caplog.clear()
    with caplog.at_level('WARNING'):
        out = dummy.calculate_tmb_analysis(save_files=False)
    assert any('No variant classification column found' in r.message for r in caplog.records)
    assert int(out['analysis'].loc[0, 'total']) == 0


# -----------------------------------------------------
# C. Genotype parsing (counted via non-synonymous rows)
# -----------------------------------------------------

def test_ignores_empty_or_missing_genotypes():
    df = pd.DataFrame({
        'REF': ['A', 'A', 'A'],
        'ALT': ['G', 'G', 'G'],
        'S1':  ['', '.', np.nan],
        'variant_classification': ['missense_mutation'] * 3,
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False)
    assert int(out['analysis'].loc[0, 'total']) == 0


def test_supports_separators_and_single_allele_and_homozygous():
    df = pd.DataFrame({
        'REF': ['A', 'A', 'A', 'A', 'A'],
        'ALT': ['G', 'G', 'G', 'G', 'G'],
        'S1':  ['0|1', '1/1', '1', '0/0', '0|0'],
        'variant_classification': ['missense_mutation'] * 5,
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False)
    # Observed rule: any non-empty genotype counts as a mutation when its row
    # is non-synonymous -- alleles are compared to REF as literal strings and
    # VCF-numeric genotypes are NOT decoded (so '0/0' != 'A' counts).
    # NOTE: worth confirming with the pyMut authors whether 0/0 should really
    # count as a mutation; with allele-string genotypes (MAF-style 'G|A')
    # this case never arises.
    assert int(out['analysis'].loc[0, 'total']) == 5


def test_marks_mutation_when_allele_is_not_ref_even_if_not_alt():
    # allele T vs REF A, ALT G => counts as mutation
    df = pd.DataFrame({
        'REF': ['A'], 'ALT': ['G'], 'S1': ['T|A'],
        'variant_classification': ['missense_mutation'],
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False)
    assert int(out['analysis'].loc[0, 'total']) == 1


def test_handles_malformed_genotypes_as_single_allele():
    df = pd.DataFrame({
        'REF': ['A', 'A'], 'ALT': ['G', 'G'], 'S1': ['0 1', 'foo'],
        'variant_classification': ['missense_mutation'] * 2,
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False)
    # Both malformed genotypes are treated as a single non-REF allele
    assert int(out['analysis'].loc[0, 'total']) == 2


def test_warns_and_skips_missing_sample_and_raises_if_none_valid(caplog):
    df = pd.DataFrame({'REF': ['A'], 'ALT': ['G'], 'X': ['0|1']})
    dummy = DummyPyMutation(df, samples=['NOPE'])
    caplog.clear()
    with caplog.at_level('WARNING'):
        with pytest.raises(ValueError, match='No valid samples found'):
            dummy.calculate_tmb_analysis(save_files=False)
    assert any("Sample 'NOPE' not found" in r.message for r in caplog.records)


# -----------------------------------------------------
# D. Metrics and normalization
# -----------------------------------------------------

def test_counts_and_non_synonymous_and_normalization(tiny_variants_df):
    dummy = DummyPyMutation(tiny_variants_df, samples=['S1', 'S2'])
    out = dummy.calculate_tmb_analysis(save_files=False, genome_size_bp=60456963)
    analysis = out['analysis'].set_index(SAMPLE_COL)

    # `total` counts non-synonymous mutated rows:
    # S1: row0 (0|1, missense) -> 1; row1 0/0 not mutated; row2 1/1 but
    #     classification None; row3 '.' ignored              => 1
    # S2: row0 (A|G, missense) -> 1; row1 C/T silent; row2 None;
    #     row3 T/T not mutated                               => 1
    assert int(analysis.loc['S1', 'total']) == 1
    assert int(analysis.loc['S2', 'total']) == 1

    # Normalization
    s1_total = analysis.loc['S1', 'total']
    s1_tmb = analysis.loc['S1', 'total_perMB']
    assert math.isclose(s1_tmb, (s1_total / 60456963) * 1_000_000, rel_tol=1e-9)


def test_normalization_with_zero_genome_size(tiny_variants_df):
    dummy = DummyPyMutation(tiny_variants_df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False, genome_size_bp=0)
    row = out['analysis'].iloc[0]
    assert row['total_perMB'] == 0


def test_analysis_dataframe_structure(tiny_variants_df):
    dummy = DummyPyMutation(tiny_variants_df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False)
    cols = [SAMPLE_COL, 'total', 'total_perMB', 'total_perMB_log']
    assert list(out['analysis'].columns) == cols
    assert len(out['analysis']) == 1


def test_statistics_dataframe_values_small_dataset():
    df = pd.DataFrame({
        'REF': ['A', 'C', 'G'],
        'ALT': ['G', 'T', 'A'],
        'S1':  ['0|1', '0/0', '1/1'],
        'S2':  ['A|G', 'C/T', 'G/G'],
        'variant_classification': ['missense_mutation', 'silent', 'nonsense_mutation']
    })
    dummy = DummyPyMutation(df, samples=['S1', 'S2'])
    out = dummy.calculate_tmb_analysis(save_files=False, genome_size_bp=1000)
    analysis = out['analysis'].set_index(SAMPLE_COL)
    stats = out['statistics'].set_index('Metric')

    # Non-synonymous mutated rows: S1 = rows 0 and 2 (0/0 not mutated) -> 2
    #                              S2 = row 0 only (C/T silent, G/G ref) -> 1
    assert int(analysis.loc['S1', 'total']) == 2
    assert int(analysis.loc['S2', 'total']) == 1

    # Statistics for `total` over the two samples
    vals = np.array([1, 2])
    s = stats.loc['total']
    assert s['Count'] == 2
    assert s['Mean'] == vals.mean()
    assert s['Median'] == np.median(vals)
    assert s['Min'] == vals.min()
    assert s['Max'] == vals.max()
    assert s['Q1'] == np.quantile(vals, 0.25)
    assert s['Q3'] == np.quantile(vals, 0.75)
    assert np.isclose(s['Std'], vals.std(ddof=1))


# -----------------------------------------------------
# E. Guardado de ficheros
# -----------------------------------------------------

def test_saves_files_and_creates_dir(tmp_path, tiny_variants_df, caplog):
    out_dir = tmp_path / 'tmb_out'
    dummy = DummyPyMutation(tiny_variants_df, samples=['S1', 'S2'])
    caplog.clear()
    with caplog.at_level('INFO'):
        out = dummy.calculate_tmb_analysis(save_files=True, output_dir=str(out_dir))
    assert out_dir.exists()
    a_path = out_dir / 'TMB_analysis.tsv'
    s_path = out_dir / 'TMB_statistics.tsv'
    assert a_path.exists() and s_path.exists()
    # Check headers and number of rows
    df_a = pd.read_csv(a_path, sep='\t')
    df_s = pd.read_csv(s_path, sep='\t')
    assert list(df_a.columns) == [SAMPLE_COL, 'total', 'total_perMB', 'total_perMB_log']
    assert len(df_a) == 2
    assert not df_s.empty
    # Float format with 6 decimals in TMB columns
    with open(a_path, 'r') as f:
        content = f.read()
        assert re.search(r"\d+\.\d{6}", content) is not None


def test_does_not_create_dir_when_save_files_false(tmp_path, tiny_variants_df):
    out_dir = tmp_path / 'tmb_out2'
    dummy = DummyPyMutation(tiny_variants_df, samples=['S1'])
    dummy.calculate_tmb_analysis(save_files=False, output_dir=str(out_dir))
    assert not out_dir.exists()


# -----------------------------------------------------
# F. Logging y resumen
# -----------------------------------------------------

def test_log_tmb_summary_with_empty_df(caplog):
    caplog.clear()
    with caplog.at_level('WARNING'):
        log_tmb_summary(pd.DataFrame())
    assert any('No analysis data provided for summary' in r.message for r in caplog.records)
    assert not any(r.levelname == 'INFO' for r in caplog.records)


def test_log_tmb_summary_with_non_empty_df(caplog):
    df = pd.DataFrame({
        SAMPLE_COL: ['S1', 'S2'],
        'total': [2, 3],
        'total_perMB': [0.123456, 0.654321],
        'total_perMB_log': [-0.908485, -0.184198],
    })
    caplog.clear()
    with caplog.at_level('INFO'):
        log_tmb_summary(df)
    messages = '\n'.join(r.message for r in caplog.records)
    assert 'TMB ANALYSIS SUMMARY' in messages
    assert 'Average non-synonymous mutations per sample' in messages
    assert 'Sample with highest TMB' in messages and 'Sample with lowest TMB' in messages
    assert re.search(r"\b0\.123456\b", messages) and re.search(r"\b0\.654321\b", messages)


def test_calculate_tmb_analysis_calls_log_summary(tiny_variants_df, caplog):
    dummy = DummyPyMutation(tiny_variants_df, samples=['S1'])
    caplog.clear()
    with caplog.at_level('INFO'):
        dummy.calculate_tmb_analysis(save_files=False)
    assert any('TMB ANALYSIS SUMMARY' in r.message for r in caplog.records)


# -----------------------------------------------------
# G. Casos adicionales y bordes
# -----------------------------------------------------

def test_mixed_valid_and_invalid_samples_logs_and_computes(tiny_variants_df, caplog):
    dummy = DummyPyMutation(tiny_variants_df, samples=['S1', 'NOPE', 'S2'])
    caplog.clear()
    with caplog.at_level('WARNING'):
        out = dummy.calculate_tmb_analysis(save_files=False)
    assert any("Sample 'NOPE' not found" in r.message for r in caplog.records)
    assert set(out['analysis'][SAMPLE_COL]) == {'S1', 'S2'}


def test_multiple_variant_columns_order_changes_non_synonymous_counts():
    df1 = pd.DataFrame({
        'REF': ['A', 'C', 'G'], 'ALT': ['G', 'T', 'A'], 'S1': ['0|1', '1/1', '1/1'],
        'variant_classification': ['missense_mutation', 'silent', 'nonsense_mutation'],
        'gencode_29_variant_classification': ['silent', 'silent', 'silent']
    })
    df2 = df1[['REF', 'ALT', 'S1', 'gencode_29_variant_classification', 'variant_classification']].copy()

    a = DummyPyMutation(df1, samples=['S1']).calculate_tmb_analysis(save_files=False)['analysis']
    b = DummyPyMutation(df2, samples=['S1']).calculate_tmb_analysis(save_files=False)['analysis']
    # df1: missense + nonsense rows mutated -> 2 ; df2: gencode (all silent) first -> 0
    assert int(a.loc[0, 'total']) == 2
    assert int(b.loc[0, 'total']) == 0


def test_per_row_alt_and_mixed_ref_alt_types_and_whitespace_genotypes():
    df = pd.DataFrame({
        'REF': ['A', 1, 'G'],
        'ALT': ['G', 2, 'A'],
        'S1':  [' A / G ', ' 2 | 2 ', ' 0 / 0 '],
        'variant_classification': ['missense_mutation'] * 3,
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False)
    # Under literal-allele rules all three rows carry a non-REF allele.
    assert int(out['analysis'].loc[0, 'total']) == 3


def test_zero_mutations_in_sample_produces_zero_tmb_and_stats():
    # Use explicit REF alleles so no mutation is detected
    df = pd.DataFrame({
        'REF': ['A', 'A'], 'ALT': ['G', 'G'], 'S1': ['A/A', 'A|A'],
        'variant_classification': ['missense_mutation'] * 2,
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False, genome_size_bp=1000)
    row = out['analysis'].iloc[0]
    assert row['total'] == 0
    assert row['total_perMB'] == 0
    stats = out['statistics'].set_index('Metric')
    assert stats.loc['total', 'Min'] == 0


def test_silent_mutation_produces_zero_non_synonymous_total():
    # A mutated row with a synonymous classification does not count towards
    # `total` (which is non-synonymous only, like maftools::tmb()).
    df = pd.DataFrame({
        'REF': ['A'], 'ALT': ['G'], 'S1': ['0|1'], 'variant_classification': ['silent']
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False, genome_size_bp=1000)
    row = out['analysis'].iloc[0]
    assert row['total'] == 0
    assert row['total_perMB'] == 0
    stats = out['statistics'].set_index('Metric')
    assert stats.loc['total', 'Max'] == 0
