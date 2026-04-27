import os
import io
import re
import math
import pandas as pd
import numpy as np
import pytest

from src.pyMut.analysis.mutation_burden import MutationBurdenMixin, log_tmb_summary
from .fixtures.mutation_burden_fixtures import DummyPyMutation, tiny_variants_df


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
    # Order matters: the first should be chosen
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
    # Since first column marks as non-syn, Non_Synonymous_Mutations should be 1
    assert int(out['analysis'].loc[0, 'Non_Synonymous_Mutations']) == 1


def test_no_variant_column_sets_non_synonymous_to_zero_and_warns(caplog):
    df = pd.DataFrame({'REF': ['A'], 'ALT': ['G'], 'S1': ['0|1']})
    dummy = DummyPyMutation(df, samples=['S1'])
    caplog.clear()
    with caplog.at_level('WARNING'):
        out = dummy.calculate_tmb_analysis(save_files=False)
    assert any('No variant classification column found' in r.message for r in caplog.records)
    assert int(out['analysis'].loc[0, 'Non_Synonymous_Mutations']) == 0


# -----------------------------------------------------
# C. Genotype parsing
# -----------------------------------------------------

def test_ignores_empty_or_missing_genotypes():
    df = pd.DataFrame({
        'REF': ['A', 'A', 'A'],
        'ALT': ['G', 'G', 'G'],
        'S1':  ['', '.', np.nan]
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False)
    assert int(out['analysis'].loc[0, 'Total_Mutations']) == 0


def test_supports_separators_and_single_allele_and_homozygous():
    df = pd.DataFrame({
        'REF': ['A', 'A', 'A', 'A', 'A'],
        'ALT': ['G', 'G', 'G', 'G', 'G'],
        'S1':  ['0|1', '1/1', '1', '0/0', '0|0']
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False)
    # According to spec: any allele != REF (and not '.'/''), or equal to ALT counts as mutation.
    # Here, all 5 rows should be counted as mutations.
    assert int(out['analysis'].loc[0, 'Total_Mutations']) == 5


def test_marks_mutation_when_allele_is_not_ref_even_if_not_alt():
    # allele T vs REF A, ALT G => counts as mutation
    df = pd.DataFrame({'REF': ['A'], 'ALT': ['G'], 'S1': ['T|A']})
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False)
    assert int(out['analysis'].loc[0, 'Total_Mutations']) == 1


def test_handles_malformed_genotypes_as_single_allele():
    df = pd.DataFrame({'REF': ['A', 'A'], 'ALT': ['G', 'G'], 'S1': ['0 1', 'foo']})
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False)
    # Both alleles are not equal to REF, thus both count as mutations
    assert int(out['analysis'].loc[0, 'Total_Mutations']) == 2


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
    analysis = out['analysis'].set_index('Sample')

    # Total mutations per sample
    # S1: rows with genotypes 0|1, 0/0, 1/1 => 3; '.' ignored
    # S2: A|G, C/T, G/A mutated; T/T is REF only -> 3
    assert int(analysis.loc['S1', 'Total_Mutations']) == 3
    assert int(analysis.loc['S2', 'Total_Mutations']) == 3

    # Non-synonymous: rows 0 and 3 are non-syn; S1 mutates row0 only; S2 mutates row0 only
    assert int(analysis.loc['S1', 'Non_Synonymous_Mutations']) == 1
    assert int(analysis.loc['S2', 'Non_Synonymous_Mutations']) == 1

    # Normalization
    s1_total = analysis.loc['S1', 'Total_Mutations']
    s1_tmb = analysis.loc['S1', 'TMB_Total_Normalized']
    assert math.isclose(s1_tmb, (s1_total/60456963)*1_000_000, rel_tol=1e-9)


def test_normalization_with_zero_genome_size(tiny_variants_df):
    dummy = DummyPyMutation(tiny_variants_df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False, genome_size_bp=0)
    row = out['analysis'].iloc[0]
    assert row['TMB_Total_Normalized'] == 0
    assert row['TMB_Non_Synonymous_Normalized'] == 0


def test_analysis_dataframe_structure(tiny_variants_df):
    dummy = DummyPyMutation(tiny_variants_df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False)
    cols = ['Sample', 'Total_Mutations', 'Non_Synonymous_Mutations', 'TMB_Total_Normalized', 'TMB_Non_Synonymous_Normalized']
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
    analysis = out['analysis'].set_index('Sample')
    stats = out['statistics'].set_index('Metric')

    # Expected totals under the literal allele logic: S1=3 (all valid rows), S2=2 (rows 0 and 1)
    assert int(analysis.loc['S1', 'Total_Mutations']) == 3
    assert int(analysis.loc['S2', 'Total_Mutations']) == 2

    # Statistics for Total_Mutations over two samples
    vals = np.array([2, 3])
    s = stats.loc['Total_Mutations']
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
    assert list(df_a.columns) == ['Sample', 'Total_Mutations', 'Non_Synonymous_Mutations', 'TMB_Total_Normalized', 'TMB_Non_Synonymous_Normalized']
    assert len(df_a) == 2
    assert not df_s.empty
    # Float format with 6 decimals in TMB columns
    with open(a_path, 'r') as f:
        content = f.read()
        # Find numbers with 6 decimals like 0.123456
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
    # No INFO lines should be present
    assert not any(r.levelname == 'INFO' for r in caplog.records)


def test_log_tmb_summary_with_non_empty_df(caplog):
    df = pd.DataFrame({
        'Sample': ['S1', 'S2'],
        'Total_Mutations': [2, 3],
        'Non_Synonymous_Mutations': [1, 2],
        'TMB_Total_Normalized': [0.123456, 0.654321],
        'TMB_Non_Synonymous_Normalized': [0.111111, 0.222222]
    })
    caplog.clear()
    with caplog.at_level('INFO'):
        log_tmb_summary(df)
    messages = '\n'.join(r.message for r in caplog.records)
    assert 'TMB ANALYSIS SUMMARY' in messages
    assert 'Average total mutations per sample' in messages
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
    assert set(out['analysis']['Sample']) == {'S1', 'S2'}


def test_multiple_variant_columns_order_changes_non_synonymous_counts():
    df1 = pd.DataFrame({
        'REF': ['A', 'C', 'G'], 'ALT': ['G', 'T', 'A'], 'S1': ['0|1', '1/1', '1/1'],
        'variant_classification': ['missense_mutation', 'silent', 'nonsense_mutation'],
        'gencode_29_variant_classification': ['silent', 'silent', 'silent']
    })
    df2 = df1[['REF', 'ALT', 'S1', 'gencode_29_variant_classification', 'variant_classification']].copy()

    a = DummyPyMutation(df1, samples=['S1']).calculate_tmb_analysis(save_files=False)['analysis']
    b = DummyPyMutation(df2, samples=['S1']).calculate_tmb_analysis(save_files=False)['analysis']
    assert int(a.loc[0, 'Non_Synonymous_Mutations']) != int(b.loc[0, 'Non_Synonymous_Mutations'])


def test_per_row_alt_and_mixed_ref_alt_types_and_whitespace_genotypes():
    df = pd.DataFrame({
        'REF': ['A', 1, 'G'],
        'ALT': ['G', 2, 'A'],
        'S1':  [' A / G ', ' 2 | 2 ', ' 0 / 0 ']
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False)
    # Under literal-allele rules, all three rows are mutations (third row '0/0' != REF 'G').
    assert int(out['analysis'].loc[0, 'Total_Mutations']) == 3


def test_zero_mutations_in_sample_produces_zero_tmb_and_stats():
    # Use explicit REF alleles so no mutation is detected
    df = pd.DataFrame({'REF': ['A', 'A'], 'ALT': ['G', 'G'], 'S1': ['A/A', 'A|A']})
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False, genome_size_bp=1000)
    row = out['analysis'].iloc[0]
    assert row['Total_Mutations'] == 0
    assert row['TMB_Total_Normalized'] == 0
    stats = out['statistics'].set_index('Metric')
    assert stats.loc['Total_Mutations', 'Min'] == 0


def test_non_synonymous_zero_while_total_positive_reflected_in_stats():
    df = pd.DataFrame({
        'REF': ['A'], 'ALT': ['G'], 'S1': ['0|1'], 'variant_classification': ['silent']
    })
    dummy = DummyPyMutation(df, samples=['S1'])
    out = dummy.calculate_tmb_analysis(save_files=False, genome_size_bp=1000)
    row = out['analysis'].iloc[0]
    assert row['Total_Mutations'] == 1
    assert row['Non_Synonymous_Mutations'] == 0
    stats = out['statistics'].set_index('Metric')
    assert stats.loc['Non_Synonymous_Mutations', 'Max'] == 0
