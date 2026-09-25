import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock
import time

from src.pyMut.filters.pass_filter import PassFilterMixin
from .fixtures.pass_filter_fixtures import (
    sample_mutation_data,
    empty_mutation_data,
    mutation_data_with_duplicates,
    mutation_data_mixed_chromosomes,
    mutation_data_with_nulls,
    mutation_data_missing_columns,
    mutation_data_different_types,
    large_mutation_data,
    mock_pymutation,
    mock_pymutation_empty,
    mock_pymutation_missing_columns
)


class TestPassFilterMixin:
    """Tests for PassFilterMixin class"""
    
    def setup_method(self):
        """Setup for each test"""
        # Configure logging to capture logs during tests
        logging.getLogger('src.pyMut.filters.pass_filter').setLevel(logging.DEBUG)
    
    def teardown_method(self):
        """Teardown after each test"""
        pass
    
    # ==================== Basic functionality tests ====================
    
    def test_pass_filter_returns_true_when_record_exists_and_filter_is_pass(self, mock_pymutation):
        """Test que retorna True cuando existe el registro y FILTER == 'PASS'"""
        result = mock_pymutation.pass_filter('chr1', 100, 'A', 'T')
        assert result is True
    
    def test_pass_filter_returns_false_when_record_exists_but_filter_not_pass(self, mock_pymutation):
        """Test que retorna False cuando existe el registro pero FILTER != 'PASS'"""
        result = mock_pymutation.pass_filter('chr3', 300, 'G', 'A')
        assert result is False
    
    def test_pass_filter_returns_false_when_record_not_found(self, mock_pymutation):
        """Test que retorna False cuando no existe el registro"""
        result = mock_pymutation.pass_filter('chr99', 999, 'Z', 'X')
        assert result is False
    
    def test_pass_filter_handles_chromosome_formatting_with_chr_prefix(self, mock_pymutation):
        """Test que maneja cromosomas con prefijo 'chr'"""
        result = mock_pymutation.pass_filter('chr1', 100, 'A', 'T')
        assert result is True
    
    def test_pass_filter_handles_chromosome_formatting_without_chr_prefix(self):
        """Test que maneja cromosomas sin prefijo"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        # Data with chromosome without 'chr' prefix
        data = pd.DataFrame({
            'CHROM': ['1', '2', 'X'],
            'POS': [100, 200, 300],
            'REF': ['A', 'T', 'G'],
            'ALT': ['T', 'C', 'A'],
            'FILTER': ['PASS', 'PASS', 'FAIL']
        })
        
        mock_obj = MockPyMutation(data)
        
        # Should work with both formats
        result1 = mock_obj.pass_filter('1', 100, 'A', 'T')
        result2 = mock_obj.pass_filter('chr1', 100, 'A', 'T')
        
        assert result1 is True
        assert result2 is True
    
    def test_pass_filter_handles_chromosome_formatting_consistency(self, mutation_data_mixed_chromosomes):
        """Test que funciona cuando el input y datos tienen diferentes formatos"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        mock_obj = MockPyMutation(mutation_data_mixed_chromosomes)
        
        # Test different chromosome format combinations
        result1 = mock_obj.pass_filter('1', 100, 'A', 'T')  # Data has '1', input '1'
        result2 = mock_obj.pass_filter('chr1', 100, 'A', 'T')  # Data has '1', input 'chr1'
        result3 = mock_obj.pass_filter('2', 200, 'T', 'C')  # Data has 'chr2', input '2'
        result4 = mock_obj.pass_filter('chr2', 200, 'T', 'C')  # Data has 'chr2', input 'chr2'
        
        assert result1 is True
        assert result2 is True
        assert result3 is True
        assert result4 is True
    
    # ==================== Data validation tests ====================
    
    def test_pass_filter_raises_keyerror_when_chrom_column_missing(self):
        """Test que lanza KeyError cuando falta columna CHROM"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        data = pd.DataFrame({
            'POS': [100],
            'REF': ['A'],
            'ALT': ['T'],
            'FILTER': ['PASS']
        })
        
        mock_obj = MockPyMutation(data)
        
        with pytest.raises(KeyError, match="Missing required columns.*CHROM"):
            mock_obj.pass_filter('chr1', 100, 'A', 'T')
    
    def test_pass_filter_raises_keyerror_when_pos_column_missing(self):
        """Test que lanza KeyError cuando falta columna POS"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        data = pd.DataFrame({
            'CHROM': ['chr1'],
            'REF': ['A'],
            'ALT': ['T'],
            'FILTER': ['PASS']
        })
        
        mock_obj = MockPyMutation(data)
        
        with pytest.raises(KeyError, match="Missing required columns.*POS"):
            mock_obj.pass_filter('chr1', 100, 'A', 'T')
    
    def test_pass_filter_raises_keyerror_when_ref_column_missing(self):
        """Test que lanza KeyError cuando falta columna REF"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        data = pd.DataFrame({
            'CHROM': ['chr1'],
            'POS': [100],
            'ALT': ['T'],
            'FILTER': ['PASS']
        })
        
        mock_obj = MockPyMutation(data)
        
        with pytest.raises(KeyError, match="Missing required columns.*REF"):
            mock_obj.pass_filter('chr1', 100, 'A', 'T')
    
    def test_pass_filter_raises_keyerror_when_alt_column_missing(self):
        """Test que lanza KeyError cuando falta columna ALT"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        data = pd.DataFrame({
            'CHROM': ['chr1'],
            'POS': [100],
            'REF': ['A'],
            'FILTER': ['PASS']
        })
        
        mock_obj = MockPyMutation(data)
        
        with pytest.raises(KeyError, match="Missing required columns.*ALT"):
            mock_obj.pass_filter('chr1', 100, 'A', 'T')
    
    def test_pass_filter_raises_keyerror_when_filter_column_missing(self):
        """Test que lanza KeyError cuando falta columna FILTER"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        data = pd.DataFrame({
            'CHROM': ['chr1'],
            'POS': [100],
            'REF': ['A'],
            'ALT': ['T']
        })
        
        mock_obj = MockPyMutation(data)
        
        with pytest.raises(KeyError, match="Missing required columns.*FILTER"):
            mock_obj.pass_filter('chr1', 100, 'A', 'T')
    
    def test_pass_filter_raises_keyerror_when_multiple_columns_missing(self, mock_pymutation_missing_columns):
        """Test comportamiento cuando faltan múltiples columnas"""
        with pytest.raises(KeyError, match="Missing required columns.*ALT.*FILTER"):
            mock_pymutation_missing_columns.pass_filter('chr1', 100, 'A', 'T')
    
    def test_pass_filter_handles_different_chromosome_types(self, mutation_data_different_types):
        """Test que maneja cromosomas como str, int, etc."""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        mock_obj = MockPyMutation(mutation_data_different_types)
        
        # Test with different input types - all should work due to format_chr
        result1 = mock_obj.pass_filter(1, 100, 'A', 'T')  # int chromosome
        result2 = mock_obj.pass_filter('1', 100, 'A', 'T')  # str chromosome
        result3 = mock_obj.pass_filter('chr1', 100, 'A', 'T')  # chr prefixed
        
        assert result1 == True
        assert result2 == True
        assert result3 == True
    
    def test_pass_filter_handles_different_position_types(self, mutation_data_different_types):
        """Test que maneja posiciones como int, float, str"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        mock_obj = MockPyMutation(mutation_data_different_types)
        
        # Simple test - just verify the method doesn't crash with different data types
        try:
            result1 = mock_obj.pass_filter(1, 100, 'A', 'T')
            result2 = mock_obj.pass_filter('chr3', '300', 'G', 'A')
            result3 = mock_obj.pass_filter('nonexistent', 999, 'Z', 'Y')
            
            # If we get here without exceptions, the test passes
            # Just verify we get some kind of result
            assert result1 is not None
            assert result2 is not None
            assert result3 is not None
            
        except Exception as e:
            pytest.fail(f"Method crashed with different position types: {e}")
    
    # ==================== PyArrow optimization tests ====================
    
    @patch('pandas.DataFrame.astype')
    def test_pass_filter_uses_pyarrow_optimization_when_available(self, mock_astype, mock_pymutation, caplog):
        """Test que usa la ruta optimizada PyArrow cuando está disponible"""
        with caplog.at_level(logging.INFO, logger='src.pyMut.filters.pass_filter'):
            result = mock_pymutation.pass_filter('chr1', 100, 'A', 'T')
        
        assert result is True
        assert "Attempting to use PyArrow optimization" in caplog.text
    
    def test_pass_filter_converts_columns_to_pyarrow_types(self, mock_pymutation):
        """Test que convierte columnas a tipos PyArrow correctos"""
        # This test verifies the conversion process by checking the result is correct
        result = mock_pymutation.pass_filter('chr1', 100, 'A', 'T')
        assert result is True
        
        # The conversion should happen internally and still produce correct results
        result2 = mock_pymutation.pass_filter('chr2', 200, 'T', 'C')
        assert result2 is True
    
    def test_pass_filter_falls_back_when_pyarrow_not_available(self, mock_pymutation):
        """Test que hace fallback cuando PyArrow no está disponible"""
        # Mock the astype method to raise ImportError for pyarrow types
        original_astype = mock_pymutation.data.astype
        def mock_astype_func(dtype):
            if 'pyarrow' in str(dtype):
                raise ImportError("PyArrow not available")
            return original_astype(dtype)
        
        with patch.object(mock_pymutation.data, 'astype', side_effect=mock_astype_func):
            # The important thing is that it still works and returns the correct result
            result = mock_pymutation.pass_filter('chr1', 100, 'A', 'T')
        
        # Should still work correctly despite PyArrow issues
        assert result is True
    
    def test_pass_filter_falls_back_when_pyarrow_fails(self, mock_pymutation):
        """Test que hace fallback cuando la optimización PyArrow falla"""
        # Mock the astype method to raise a general Exception for pyarrow types
        original_astype = mock_pymutation.data.astype
        def mock_astype_func(dtype):
            if 'pyarrow' in str(dtype):
                raise Exception("PyArrow conversion failed")
            return original_astype(dtype)
        
        with patch.object(mock_pymutation.data, 'astype', side_effect=mock_astype_func):
            # The important thing is that it still works and returns the correct result
            result = mock_pymutation.pass_filter('chr1', 100, 'A', 'T')
        
        # Should still work correctly despite PyArrow issues
        assert result is True
    
    # ==================== Edge case tests ====================
    
    def test_pass_filter_warns_when_multiple_records_found_pyarrow(self, mutation_data_with_duplicates, caplog):
        """Test que advierte cuando encuentra múltiples registros (ruta PyArrow)"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        mock_obj = MockPyMutation(mutation_data_with_duplicates)
        
        with caplog.at_level(logging.WARNING, logger='src.pyMut.filters.pass_filter'):
            result = mock_obj.pass_filter('chr1', 100, 'A', 'T')
        
        assert result is True  # Should return True because one of the duplicates is PASS
        assert any("Multiple records found" in record.message 
                  for record in caplog.records if record.levelname == "WARNING")
    
    @patch('pandas.DataFrame.astype', side_effect=ImportError("PyArrow not available"))
    def test_pass_filter_warns_when_multiple_records_found_standard(self, mock_astype, mutation_data_with_duplicates, caplog):
        """Test que advierte cuando encuentra múltiples registros (ruta estándar)"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        mock_obj = MockPyMutation(mutation_data_with_duplicates)
        
        with caplog.at_level(logging.WARNING, logger='src.pyMut.filters.pass_filter'):
            result = mock_obj.pass_filter('chr1', 100, 'A', 'T')
        
        assert result is True  # Should return True because one of the duplicates is PASS
        assert any("Multiple records found" in record.message 
                  for record in caplog.records if record.levelname == "WARNING")
    
    def test_pass_filter_returns_true_when_any_duplicate_record_passes(self, mutation_data_with_duplicates):
        """Test que retorna True si algún registro duplicado es PASS"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        mock_obj = MockPyMutation(mutation_data_with_duplicates)
        
        result = mock_obj.pass_filter('chr1', 100, 'A', 'T')
        assert result is True
        
        # Test case where all duplicates are PASS
        result2 = mock_obj.pass_filter('chr2', 200, 'T', 'C')
        assert result2 is True
    
    def test_pass_filter_handles_empty_dataframe(self, mock_pymutation_empty):
        """Test comportamiento con DataFrame vacío"""
        result = mock_pymutation_empty.pass_filter('chr1', 100, 'A', 'T')
        assert result is False
    
    def test_pass_filter_handles_null_values_in_filter_column(self, mutation_data_with_nulls):
        """Test comportamiento con valores NULL/NaN en FILTER"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        mock_obj = MockPyMutation(mutation_data_with_nulls)
        
        # Record with PASS should work
        result1 = mock_obj.pass_filter('chr1', 100, 'A', 'T')
        assert result1 is True
        
        # Records with NaN/None should return False
        result2 = mock_obj.pass_filter('chr2', 200, 'T', 'C')
        assert result2 is False
        
        result3 = mock_obj.pass_filter('chr3', 300, 'G', 'A')
        assert result3 is False
    
    def test_pass_filter_handles_special_chromosomes(self):
        """Test con cromosomas especiales: X, Y, MT, etc."""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        data = pd.DataFrame({
            'CHROM': ['X', 'Y', 'MT', 'chrM', '23', '24'],
            'POS': [100, 200, 300, 400, 500, 600],
            'REF': ['A', 'T', 'G', 'C', 'A', 'T'],
            'ALT': ['T', 'C', 'A', 'G', 'T', 'C'],
            'FILTER': ['PASS', 'PASS', 'PASS', 'PASS', 'FAIL', 'PASS']
        })
        
        mock_obj = MockPyMutation(data)
        
        # Test special chromosomes
        assert mock_obj.pass_filter('X', 100, 'A', 'T') is True
        assert mock_obj.pass_filter('Y', 200, 'T', 'C') is True
        assert mock_obj.pass_filter('MT', 300, 'G', 'A') is True
        assert mock_obj.pass_filter('chrM', 400, 'C', 'G') is True
        assert mock_obj.pass_filter('chrX', 100, 'A', 'T') is True  # Should match X
        assert mock_obj.pass_filter('chrY', 200, 'T', 'C') is True  # Should match Y
    
    # ==================== Logging tests ====================
    
    def test_pass_filter_logs_search_parameters(self, mock_pymutation, caplog):
        """Test que logea los parámetros de búsqueda"""
        with caplog.at_level(logging.INFO, logger='src.pyMut.filters.pass_filter'):
            mock_pymutation.pass_filter('chr1', 100, 'A', 'T')
        
        assert any("Checking PASS filter for: chr1:100 A>T" in record.message 
                  for record in caplog.records if record.levelname == "INFO")
    
    def test_pass_filter_logs_when_record_not_found(self, mock_pymutation, caplog):
        """Test que logea cuando no encuentra registro"""
        with caplog.at_level(logging.INFO, logger='src.pyMut.filters.pass_filter'):
            mock_pymutation.pass_filter('chr99', 999, 'Z', 'X')
        
        assert any("Record not found: chr99:999 Z>X" in record.message 
                  for record in caplog.records if record.levelname == "INFO")
    
    def test_pass_filter_logs_final_result(self, mock_pymutation, caplog):
        """Test que logea el resultado final"""
        with caplog.at_level(logging.INFO, logger='src.pyMut.filters.pass_filter'):
            result = mock_pymutation.pass_filter('chr1', 100, 'A', 'T')
        
        assert result is True
        assert any("PASS filter result: True" in record.message 
                  for record in caplog.records if record.levelname == "INFO")
    
    def test_pass_filter_logs_pyarrow_usage(self, mock_pymutation, caplog):
        """Test que logea el uso de PyArrow"""
        with caplog.at_level(logging.INFO, logger='src.pyMut.filters.pass_filter'):
            mock_pymutation.pass_filter('chr1', 100, 'A', 'T')
        
        assert any("Attempting to use PyArrow optimization" in record.message 
                  for record in caplog.records if record.levelname == "INFO")
    
    def test_pass_filter_logs_warnings_appropriately(self, mutation_data_with_duplicates, caplog):
        """Test que logea warnings apropiadamente"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        mock_obj = MockPyMutation(mutation_data_with_duplicates)
        
        with caplog.at_level(logging.WARNING, logger='src.pyMut.filters.pass_filter'):
            mock_obj.pass_filter('chr1', 100, 'A', 'T')
        
        assert any("Multiple records found" in record.message 
                  for record in caplog.records if record.levelname == "WARNING")
    
    # ==================== Integration tests ====================
    
    def test_pass_filter_with_real_vcf_data(self):
        """Test con datos VCF reales para verificar integración completa"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        # Simulate real VCF data structure
        vcf_data = pd.DataFrame({
            'CHROM': ['chr1', 'chr1', 'chr2', 'chrX', 'chrY'],
            'POS': [12345, 67890, 11111, 22222, 33333],
            'ID': ['.', 'rs123', '.', 'rs456', '.'],
            'REF': ['A', 'TG', 'C', 'G', 'T'],
            'ALT': ['G', 'T', 'A', 'C', 'A'],
            'QUAL': [30.0, 45.2, 25.1, 40.0, 35.5],
            'FILTER': ['PASS', 'PASS', 'LowQual', 'PASS', '.'],
            'INFO': ['AF=0.3', 'AF=0.1;DP=100', 'AF=0.05', 'AF=0.2', 'AF=0.4']
        })
        
        mock_obj = MockPyMutation(vcf_data)
        
        # Test various scenarios
        assert mock_obj.pass_filter('chr1', 12345, 'A', 'G') is True
        assert mock_obj.pass_filter('chr1', 67890, 'TG', 'T') is True
        assert mock_obj.pass_filter('chr2', 11111, 'C', 'A') is False  # LowQual
        assert mock_obj.pass_filter('chrX', 22222, 'G', 'C') is True
        assert mock_obj.pass_filter('chrY', 33333, 'T', 'A') is False  # Filter = '.'
        assert mock_obj.pass_filter('chr99', 99999, 'N', 'N') is False  # Not found
    
    def test_pass_filter_with_different_data_sources(self):
        """Test que funciona con diferentes fuentes de datos"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        # MAF-like data
        maf_data = pd.DataFrame({
            'CHROM': ['1', '2', '3'],
            'POS': [100, 200, 300],
            'REF': ['A', 'T', 'G'],
            'ALT': ['T', 'C', 'A'],
            'FILTER': ['PASS', 'PASS', 'FAIL']
        })
        
        mock_obj = MockPyMutation(maf_data)
        
        assert mock_obj.pass_filter('1', 100, 'A', 'T') is True
        assert mock_obj.pass_filter('chr2', 200, 'T', 'C') is True  # Format consistency
        assert mock_obj.pass_filter('3', 300, 'G', 'A') is False
    
    # ==================== Performance tests ====================
    
    def test_pass_filter_performance_with_large_dataset(self, large_mutation_data):
        """Test performance con datasets grandes"""
        from src.pyMut.filters.pass_filter import PassFilterMixin
        
        class MockPyMutation(PassFilterMixin):
            def __init__(self, data):
                self.data = data
        
        # Add a known record to the large dataset
        large_mutation_data.loc[0] = ['chr1', 100, 'A', 'T', 'PASS']
        mock_obj = MockPyMutation(large_mutation_data)
        
        # Measure time for multiple operations
        start_time = time.time()
        
        # Perform multiple searches
        for _ in range(10):
            result = mock_obj.pass_filter('chr1', 100, 'A', 'T')
            assert result is True
        
        # Search for non-existent records
        for i in range(5):
            result = mock_obj.pass_filter('chr99', 999999 + i, 'Z', 'X')
            assert result is False
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance should be reasonable (less than 5 seconds for 15 operations on 10k records)
        assert execution_time < 5.0, f"Performance test failed: {execution_time:.2f} seconds"
    
    # ==================== Parameter validation tests ====================
    
    def test_pass_filter_handles_invalid_position_types(self, mock_pymutation):
        """Test comportamiento con posiciones no numéricas"""
        # String positions should still work if they can be converted
        result = mock_pymutation.pass_filter('chr1', '100', 'A', 'T')
        # This might work depending on pandas comparison behavior
        # The test verifies the method doesn't crash
        assert isinstance(result, bool)
    
    def test_pass_filter_handles_empty_string_parameters(self, mock_pymutation):
        """Test comportamiento con strings vacíos"""
        result = mock_pymutation.pass_filter('', 100, 'A', 'T')
        assert result is False
        
        result = mock_pymutation.pass_filter('chr1', 100, '', 'T')
        assert result is False
        
        result = mock_pymutation.pass_filter('chr1', 100, 'A', '')
        assert result is False
    
    def test_pass_filter_handles_none_parameters(self, mock_pymutation):
        """Test comportamiento con parámetros None"""
        # These should not crash but return False
        result = mock_pymutation.pass_filter(None, 100, 'A', 'T')
        assert result is False
        
        result = mock_pymutation.pass_filter('chr1', None, 'A', 'T')
        assert result is False
        
        result = mock_pymutation.pass_filter('chr1', 100, None, 'T')
        assert result is False
        
        result = mock_pymutation.pass_filter('chr1', 100, 'A', None)
        assert result is False