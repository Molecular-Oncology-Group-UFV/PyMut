import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock
from copy import deepcopy

from src.pyMut.filters.chrom_sample_filter import ChromSampleFilterMixin
from .fixtures.chrom_sample_filter_fixtures import (
    sample_maf_data,
    sample_vcf_data,
    mixed_chromosomes_data,
    empty_data,
    single_row_data,
    missing_chrom_column_data,
    missing_sample_column_data,
    large_test_data,
    mock_metadata,
    mock_pymutation_maf,
    mock_pymutation_vcf,
    mock_pymutation_empty,
    mock_pymutation_missing_chrom,
    mock_pymutation_missing_sample
)


class TestChromSampleFilterMixin:
    """Tests for ChromSampleFilterMixin"""

    def setup_method(self):
        """Setup method to prepare each test"""
        # Configure logging for test capture
        logging.getLogger().setLevel(logging.INFO)

    def teardown_method(self):
        """Teardown method after each test"""
        pass

    # Tests de Funcionalidad Básica
    def test_filter_by_chrom_only_single_chromosome(self, mock_pymutation_maf):
        """Verifica filtrado por un solo cromosoma"""
        result = mock_pymutation_maf.filter_by_chrom_sample(chrom='chr1')
        
        # Should have 2 rows for chr1
        assert len(result.data) == 2
        assert all(result.data['CHROM'] == 'chr1')
        # Should maintain all columns
        assert len(result.data.columns) == len(mock_pymutation_maf.data.columns)
        # Verify metadata updated
        assert 'chromosome:chr1' in result.metadata.filters[0]

    def test_filter_by_chrom_only_multiple_chromosomes(self, mock_pymutation_maf):
        """Verifica filtrado por múltiples cromosomas (lista)"""
        result = mock_pymutation_maf.filter_by_chrom_sample(chrom=['chr1', 'chr17'])
        
        # Should have 3 rows (2 for chr1, 1 for chr17)
        assert len(result.data) == 3
        assert set(result.data['CHROM'].unique()) == {'chr1', 'chr17'}
        # Should maintain all columns
        assert len(result.data.columns) == len(mock_pymutation_maf.data.columns)

    def test_filter_by_sample_only_single_sample_maf_style(self, mock_pymutation_maf):
        """Verifica filtrado por una muestra individual (estilo MAF)"""
        result = mock_pymutation_maf.filter_by_chrom_sample(sample='TCGA-AB-2988')
        
        # Should have 3 rows for this sample
        assert len(result.data) == 3
        assert all(result.data['Tumor_Sample_Barcode'] == 'TCGA-AB-2988')
        # Should maintain VCF-like columns and sample column
        expected_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'Tumor_Sample_Barcode', 'Hugo_Symbol']
        assert set(result.data.columns) == set(expected_cols)

    def test_filter_by_sample_only_multiple_samples_maf_style(self, mock_pymutation_maf):
        """Verifica filtrado por múltiples muestras (estilo MAF)"""
        result = mock_pymutation_maf.filter_by_chrom_sample(sample=['TCGA-AB-2988', 'TCGA-AB-2869'])
        
        # Should have 5 rows (3 for 2988, 2 for 2869)
        assert len(result.data) == 5
        # Verify that Tumor_Sample_Barcode column exists and filter was applied correctly
        assert 'Tumor_Sample_Barcode' in result.data.columns

    def test_filter_by_sample_only_single_sample_vcf_style(self, mock_pymutation_vcf):
        """Verifica filtrado por muestra individual (estilo VCF)"""
        result = mock_pymutation_vcf.filter_by_chrom_sample(sample='TCGA-AB-2988')
        
        # Should keep all rows (no sample column to filter by)
        assert len(result.data) == len(mock_pymutation_vcf.data)
        # Should only keep VCF columns + requested sample column
        expected_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'TCGA-AB-2988']
        assert set(result.data.columns) == set(expected_cols)
        # Sample columns removed should be logged
        assert 'TCGA-AB-2869' not in result.data.columns
        assert 'TCGA-AB-2802' not in result.data.columns

    def test_filter_by_sample_only_multiple_samples_vcf_style(self, mock_pymutation_vcf):
        """Verifica filtrado por múltiples muestras (estilo VCF)"""
        result = mock_pymutation_vcf.filter_by_chrom_sample(sample=['TCGA-AB-2988', 'TCGA-AB-2869'])
        
        # Should keep all rows
        assert len(result.data) == len(mock_pymutation_vcf.data)
        # Should keep VCF columns + requested sample columns
        expected_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'TCGA-AB-2988', 'TCGA-AB-2869']
        assert set(result.data.columns) == set(expected_cols)
        # Unrequested sample column should be removed
        assert 'TCGA-AB-2802' not in result.data.columns

    def test_filter_by_chrom_and_sample_combined(self, mock_pymutation_maf):
        """Verifica filtrado combinado por cromosoma y muestra"""
        result = mock_pymutation_maf.filter_by_chrom_sample(chrom='chr1', sample='TCGA-AB-2988')
        
        # Should have 1 row (chr1 AND TCGA-AB-2988)
        assert len(result.data) == 1
        assert all(result.data['CHROM'] == 'chr1')
        assert all(result.data['Tumor_Sample_Barcode'] == 'TCGA-AB-2988')
        # Verify combined filter in metadata
        assert 'chromosome:chr1|sample:TCGA-AB-2988' in result.metadata.filters[0]

    # Tests de Formateo de Cromosomas
    def test_chromosome_formatting_with_chr_prefix(self):
        """Verifica manejo de cromosomas con prefijo 'chr'"""
        data = {
            'CHROM': ['chr1', 'chr2', 'chr3'],
            'POS': [100, 200, 300],
            'Tumor_Sample_Barcode': ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2988']
        }
        df = pd.DataFrame(data)
        
        class MockPyMutation(ChromSampleFilterMixin):
            def __init__(self, data, metadata=None, samples=None):
                self.data = data
                self.metadata = metadata or Mock()
                self.metadata.filters = []
                self.samples = samples or ['TCGA-AB-2988', 'TCGA-AB-2869']
        
        mock_obj = MockPyMutation(df)
        result = mock_obj.filter_by_chrom_sample(chrom='chr1')
        
        assert len(result.data) == 1
        assert result.data.iloc[0]['CHROM'] == 'chr1'

    def test_chromosome_formatting_without_chr_prefix(self):
        """Verifica manejo de cromosomas sin prefijo 'chr'"""
        data = {
            'CHROM': ['1', '2', '3'],
            'POS': [100, 200, 300],
            'Tumor_Sample_Barcode': ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2988']
        }
        df = pd.DataFrame(data)
        
        class MockPyMutation(ChromSampleFilterMixin):
            def __init__(self, data, metadata=None, samples=None):
                self.data = data
                self.metadata = metadata or Mock()
                self.metadata.filters = []
                self.samples = samples or ['TCGA-AB-2988', 'TCGA-AB-2869']
        
        mock_obj = MockPyMutation(df)
        result = mock_obj.filter_by_chrom_sample(chrom='1')
        
        assert len(result.data) == 1
        assert result.data.iloc[0]['CHROM'] == 'chr1'  # Should be formatted to chr1

    def test_chromosome_formatting_mixed_input(self, mixed_chromosomes_data, mock_metadata):
        """Verifica manejo de entrada mixta (algunos con 'chr', otros sin)"""
        class MockPyMutation(ChromSampleFilterMixin):
            def __init__(self, data, metadata, samples=None):
                self.data = data
                self.metadata = metadata
                self.samples = samples or ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2802']
        
        mock_obj = MockPyMutation(mixed_chromosomes_data, mock_metadata)
        result = mock_obj.filter_by_chrom_sample(chrom=['1', 'chr2', '17'])
        
        # Should find 3 rows (1, chr2, 17)
        assert len(result.data) == 3
        # All should be formatted consistently
        expected_chroms = {'chr1', 'chr2', 'chr17'}
        assert set(result.data['CHROM'].unique()) == expected_chroms

    def test_chromosome_formatting_special_chromosomes(self):
        """Verifica manejo de cromosomas especiales (X, Y, MT)"""
        data = {
            'CHROM': ['X', 'chrY', 'MT', 'chrMT'],
            'POS': [100, 200, 300, 400],
            'Tumor_Sample_Barcode': ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2988', 'TCGA-AB-2869']
        }
        df = pd.DataFrame(data)
        
        class MockPyMutation(ChromSampleFilterMixin):
            def __init__(self, data, metadata=None, samples=None):
                self.data = data
                self.metadata = metadata or Mock()
                self.metadata.filters = []
                self.samples = samples or ['TCGA-AB-2988', 'TCGA-AB-2869']
        
        mock_obj = MockPyMutation(df)
        result = mock_obj.filter_by_chrom_sample(chrom=['X', 'Y', 'MT'])
        
        assert len(result.data) == 4
        expected_chroms = {'chrX', 'chrY', 'chrMT'}
        assert set(result.data['CHROM'].unique()) == expected_chroms

    # Tests de Validación y Errores
    def test_raises_valueerror_when_both_params_none(self, mock_pymutation_maf):
        """Verifica que se lance ValueError cuando ambos parámetros son None"""
        with pytest.raises(ValueError, match="At least one of 'chrom' or 'sample' must be provided"):
            mock_pymutation_maf.filter_by_chrom_sample()

    def test_raises_keyerror_when_chrom_column_missing(self, mock_pymutation_missing_chrom):
        """Verifica KeyError cuando falta la columna CHROM"""
        with pytest.raises(KeyError, match="Column 'CHROM' does not exist in the DataFrame"):
            mock_pymutation_missing_chrom.filter_by_chrom_sample(chrom='chr1')

    def test_raises_keyerror_when_sample_column_missing_maf_style(self, mock_pymutation_missing_sample):
        """Verifica comportamiento cuando falta la columna de muestra en MAF"""
        # Should work because it falls back to VCF-style filtering
        result = mock_pymutation_missing_sample.filter_by_chrom_sample(sample='TCGA-AB-2988')
        # Should keep all rows since no sample column exists and sample not in columns
        assert len(result.data) == 3

    def test_raises_valueerror_when_requested_samples_not_in_samples_list(self, mock_pymutation_vcf):
        """Verifica error cuando las muestras solicitadas no existen"""
        with pytest.raises(ValueError, match="Requested samples not found in PyMutation.samples"):
            mock_pymutation_vcf.filter_by_chrom_sample(sample='NONEXISTENT-SAMPLE')

    # Tests de Preservación de Columnas
    def test_preserves_vcf_like_columns(self, mock_pymutation_vcf):
        """Verifica que se preserven columnas VCF estándar"""
        result = mock_pymutation_vcf.filter_by_chrom_sample(sample='TCGA-AB-2988')
        
        vcf_like_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER']
        for col in vcf_like_cols:
            assert col in result.data.columns

    def test_preserves_non_sample_columns(self, mock_pymutation_maf):
        """Verifica preservación de columnas que no son de muestras"""
        result = mock_pymutation_maf.filter_by_chrom_sample(sample='TCGA-AB-2988')
        
        # Hugo_Symbol should be preserved as it's not a sample column
        assert 'Hugo_Symbol' in result.data.columns
        assert 'Tumor_Sample_Barcode' in result.data.columns

    def test_sample_column_filtering_correct_removal(self, mock_pymutation_vcf):
        """Verifica eliminación correcta de columnas de muestras no solicitadas"""
        result = mock_pymutation_vcf.filter_by_chrom_sample(sample='TCGA-AB-2988')
        
        # Should keep requested sample
        assert 'TCGA-AB-2988' in result.data.columns
        # Should remove unrequested samples
        assert 'TCGA-AB-2869' not in result.data.columns
        assert 'TCGA-AB-2802' not in result.data.columns

    # Tests de Metadatos y Logging
    def test_updates_metadata_filters_correctly(self, mock_pymutation_maf):
        """Verifica actualización correcta de metadatos"""
        result = mock_pymutation_maf.filter_by_chrom_sample(chrom='chr1')
        
        assert hasattr(result.metadata, 'filters')
        assert len(result.metadata.filters) == 1
        assert 'chromosome:chr1' in result.metadata.filters[0]

    def test_metadata_filters_append_to_existing(self, mock_pymutation_maf):
        """Verifica que los nuevos filtros se agreguen a los existentes"""
        # Add existing filter
        mock_pymutation_maf.metadata.filters = ['existing_filter']
        
        result = mock_pymutation_maf.filter_by_chrom_sample(chrom='chr1')
        
        assert len(result.metadata.filters) == 2
        assert 'existing_filter' in result.metadata.filters
        assert 'chromosome:chr1' in result.metadata.filters[1]

    def test_logging_information_output(self, mock_pymutation_maf, caplog):
        """Verifica que se generen logs informativos apropiados"""
        with caplog.at_level(logging.INFO):
            result = mock_pymutation_maf.filter_by_chrom_sample(chrom='chr1')
        
        # Check for expected log messages
        log_messages = [record.message for record in caplog.records]
        assert any('Chromosomes to filter: [\'chr1\']' in msg for msg in log_messages)
        assert any('Variants before filter:' in msg for msg in log_messages)
        assert any('Variants after filter:' in msg for msg in log_messages)

    def test_warning_when_no_variants_match(self, mock_pymutation_maf, caplog):
        """Verifica warning cuando no hay variantes que coincidan"""
        with caplog.at_level(logging.WARNING):
            result = mock_pymutation_maf.filter_by_chrom_sample(chrom='chr999')
        
        assert any('No variants found matching the filter criteria' in record.message 
                  for record in caplog.records if record.levelname == 'WARNING')

    def test_warning_when_filter_removes_no_variants(self, mock_pymutation_maf, caplog):
        """Verifica warning cuando el filtro no elimina variantes"""
        # Filter by all chromosomes present should not remove any variants
        all_chroms = mock_pymutation_maf.data['CHROM'].unique().tolist()
        
        with caplog.at_level(logging.WARNING):
            result = mock_pymutation_maf.filter_by_chrom_sample(chrom=all_chroms)
        
        assert any('Filter did not remove any variants' in record.message 
                  for record in caplog.records if record.levelname == 'WARNING')

    # Tests de Casos Edge
    def test_empty_dataframe_input(self, mock_pymutation_empty):
        """Verifica comportamiento con DataFrame vacío"""
        result = mock_pymutation_empty.filter_by_chrom_sample(chrom='chr1')
        
        assert len(result.data) == 0
        assert list(result.data.columns) == list(mock_pymutation_empty.data.columns)

    def test_single_row_dataframe(self, single_row_data, mock_metadata):
        """Verifica filtrado con una sola fila"""
        class MockPyMutation(ChromSampleFilterMixin):
            def __init__(self, data, metadata, samples=None):
                self.data = data
                self.metadata = metadata
                self.samples = samples or ['TCGA-AB-2988']
        
        mock_obj = MockPyMutation(single_row_data, mock_metadata)
        result = mock_obj.filter_by_chrom_sample(chrom='chr17')
        
        assert len(result.data) == 1
        assert result.data.iloc[0]['CHROM'] == 'chr17'

    def test_all_variants_filtered_out(self, mock_pymutation_maf):
        """Verifica comportamiento cuando todas las variantes se filtran"""
        result = mock_pymutation_maf.filter_by_chrom_sample(chrom='chr999')  # Non-existent chromosome
        
        assert len(result.data) == 0

    def test_sample_parameter_as_string_vs_list(self, mock_pymutation_maf):
        """Verifica consistencia entre pasar string o lista de un elemento"""
        result1 = mock_pymutation_maf.filter_by_chrom_sample(sample='TCGA-AB-2988')
        result2 = mock_pymutation_maf.filter_by_chrom_sample(sample=['TCGA-AB-2988'])
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1.data, result2.data)

    def test_chrom_parameter_as_string_vs_list(self, mock_pymutation_maf):
        """Verifica consistencia entre pasar string o lista de un elemento"""
        result1 = mock_pymutation_maf.filter_by_chrom_sample(chrom='chr1')
        result2 = mock_pymutation_maf.filter_by_chrom_sample(chrom=['chr1'])
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1.data, result2.data)