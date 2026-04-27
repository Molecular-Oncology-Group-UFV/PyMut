import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock


@pytest.fixture
def sample_mutation_data():
    """DataFrame de prueba con datos de mutaciones normales"""
    data = {
        'CHROM': ['chr1', 'chr2', 'chr3', 'X', 'Y'],
        'POS': [100, 200, 300, 400, 500],
        'REF': ['A', 'T', 'G', 'C', 'A'],
        'ALT': ['T', 'C', 'A', 'G', 'T'],
        'FILTER': ['PASS', 'PASS', 'FAIL', 'PASS', '.']
    }
    return pd.DataFrame(data)


@pytest.fixture
def empty_mutation_data():
    """DataFrame vacío para tests"""
    return pd.DataFrame(columns=['CHROM', 'POS', 'REF', 'ALT', 'FILTER'])


@pytest.fixture
def mutation_data_with_duplicates():
    """DataFrame con registros duplicados"""
    data = {
        'CHROM': ['chr1', 'chr1', 'chr2', 'chr2'],
        'POS': [100, 100, 200, 200],
        'REF': ['A', 'A', 'T', 'T'],
        'ALT': ['T', 'T', 'C', 'C'],
        'FILTER': ['PASS', 'FAIL', 'PASS', 'PASS']
    }
    return pd.DataFrame(data)


@pytest.fixture
def mutation_data_mixed_chromosomes():
    """DataFrame con diferentes formatos de cromosomas"""
    data = {
        'CHROM': ['1', 'chr2', '3', 'chrX', 'Y', 'chrMT'],
        'POS': [100, 200, 300, 400, 500, 600],
        'REF': ['A', 'T', 'G', 'C', 'A', 'T'],
        'ALT': ['T', 'C', 'A', 'G', 'T', 'C'],
        'FILTER': ['PASS', 'PASS', 'PASS', 'FAIL', 'PASS', 'PASS']
    }
    return pd.DataFrame(data)


@pytest.fixture
def mutation_data_with_nulls():
    """DataFrame con valores nulos en FILTER"""
    data = {
        'CHROM': ['chr1', 'chr2', 'chr3'],
        'POS': [100, 200, 300],
        'REF': ['A', 'T', 'G'],
        'ALT': ['T', 'C', 'A'],
        'FILTER': ['PASS', np.nan, None]
    }
    return pd.DataFrame(data)


@pytest.fixture
def mutation_data_missing_columns():
    """DataFrame sin todas las columnas requeridas"""
    data = {
        'CHROM': ['chr1', 'chr2'],
        'POS': [100, 200],
        'REF': ['A', 'T']
        # Faltan ALT y FILTER
    }
    return pd.DataFrame(data)


@pytest.fixture
def mutation_data_different_types():
    """DataFrame con diferentes tipos de datos"""
    data = {
        'CHROM': [1, 2.0, 'chr3', 'X'],  # int, float, str
        'POS': [100, 200.5, '300', 400],  # int, float, str
        'REF': ['A', 'T', 'G', 'C'],
        'ALT': ['T', 'C', 'A', 'G'],
        'FILTER': ['PASS', 'PASS', 'PASS', 'FAIL']
    }
    return pd.DataFrame(data)


@pytest.fixture
def large_mutation_data():
    """DataFrame grande para tests de performance"""
    n_records = 10000
    chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
    
    data = {
        'CHROM': np.random.choice(chromosomes, n_records),
        'POS': np.random.randint(1, 1000000, n_records),
        'REF': np.random.choice(['A', 'T', 'G', 'C'], n_records),
        'ALT': np.random.choice(['A', 'T', 'G', 'C'], n_records),
        'FILTER': np.random.choice(['PASS', 'FAIL', '.', 'LowQual'], n_records)
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_pymutation(sample_mutation_data):
    """Mock de PyMutation con PassFilterMixin"""
    from src.pyMut.filters.pass_filter import PassFilterMixin
    
    class MockPyMutation(PassFilterMixin):
        def __init__(self, data):
            self.data = data
    
    return MockPyMutation(sample_mutation_data)


@pytest.fixture
def mock_pymutation_empty(empty_mutation_data):
    """Mock de PyMutation con DataFrame vacío"""
    from src.pyMut.filters.pass_filter import PassFilterMixin
    
    class MockPyMutation(PassFilterMixin):
        def __init__(self, data):
            self.data = data
    
    return MockPyMutation(empty_mutation_data)


@pytest.fixture
def mock_pymutation_missing_columns(mutation_data_missing_columns):
    """Mock de PyMutation con columnas faltantes"""
    from src.pyMut.filters.pass_filter import PassFilterMixin
    
    class MockPyMutation(PassFilterMixin):
        def __init__(self, data):
            self.data = data
    
    return MockPyMutation(mutation_data_missing_columns)