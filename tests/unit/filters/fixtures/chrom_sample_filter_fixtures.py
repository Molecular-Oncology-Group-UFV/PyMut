import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock


@pytest.fixture
def sample_maf_data():
    """Sample MAF-style DataFrame with Tumor_Sample_Barcode column"""
    data = {
        'CHROM': ['chr1', 'chr2', 'chr3', 'chrX', 'chrY', 'chr1', 'chr17'],
        'POS': [100, 200, 300, 400, 500, 600, 700],
        'ID': ['rs1', 'rs2', 'rs3', 'rs4', 'rs5', 'rs6', 'rs7'],
        'REF': ['A', 'T', 'G', 'C', 'A', 'T', 'G'],
        'ALT': ['T', 'C', 'A', 'G', 'T', 'C', 'A'],
        'QUAL': [60, 70, 80, 90, 95, 85, 75],
        'FILTER': ['PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS'],
        'Tumor_Sample_Barcode': ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2988', 
                                'TCGA-AB-2802', 'TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2802'],
        'Hugo_Symbol': ['TP53', 'KRAS', 'EGFR', 'BRCA1', 'PIK3CA', 'BRAF', 'NF1']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_vcf_data():
    """Sample VCF-style DataFrame with individual sample columns"""
    data = {
        'CHROM': ['chr1', 'chr2', 'chr3', 'chrX', 'chrY', 'chr1'],
        'POS': [100, 200, 300, 400, 500, 600],
        'ID': ['rs1', 'rs2', 'rs3', 'rs4', 'rs5', 'rs6'],
        'REF': ['A', 'T', 'G', 'C', 'A', 'T'],
        'ALT': ['T', 'C', 'A', 'G', 'T', 'C'],
        'QUAL': [60, 70, 80, 90, 95, 85],
        'FILTER': ['PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS'],
        'TCGA-AB-2988': ['0/1', '1/1', '0/0', '0/1', '1/1', '0/0'],
        'TCGA-AB-2869': ['1/1', '0/1', '1/0', '0/0', '0/1', '1/1'],
        'TCGA-AB-2802': ['0/0', '0/1', '1/1', '0/1', '0/0', '0/1']
    }
    return pd.DataFrame(data)


@pytest.fixture
def mixed_chromosomes_data():
    """DataFrame with mixed chromosome formatting"""
    data = {
        'CHROM': ['1', 'chr2', '3', 'chrX', 'Y', 'chrMT', '17'],
        'POS': [100, 200, 300, 400, 500, 600, 700],
        'ID': ['rs1', 'rs2', 'rs3', 'rs4', 'rs5', 'rs6', 'rs7'],
        'REF': ['A', 'T', 'G', 'C', 'A', 'T', 'G'],
        'ALT': ['T', 'C', 'A', 'G', 'T', 'C', 'A'],
        'QUAL': [60, 70, 80, 90, 95, 85, 75],
        'FILTER': ['PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS'],
        'Tumor_Sample_Barcode': ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2988', 
                                'TCGA-AB-2802', 'TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2802']
    }
    return pd.DataFrame(data)


@pytest.fixture
def empty_data():
    """Empty DataFrame with proper columns"""
    columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'Tumor_Sample_Barcode']
    return pd.DataFrame(columns=columns)


@pytest.fixture
def single_row_data():
    """DataFrame with single row"""
    data = {
        'CHROM': ['chr17'],
        'POS': [43094077],
        'ID': ['rs28897677'],
        'REF': ['G'],
        'ALT': ['A'],
        'QUAL': [99],
        'FILTER': ['PASS'],
        'Tumor_Sample_Barcode': ['TCGA-AB-2988'],
        'Hugo_Symbol': ['TP53']
    }
    return pd.DataFrame(data)


@pytest.fixture
def missing_chrom_column_data():
    """DataFrame missing CHROM column"""
    data = {
        'POS': [100, 200, 300],
        'ID': ['rs1', 'rs2', 'rs3'],
        'REF': ['A', 'T', 'G'],
        'ALT': ['T', 'C', 'A'],
        'Tumor_Sample_Barcode': ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2988']
    }
    return pd.DataFrame(data)


@pytest.fixture
def missing_sample_column_data():
    """DataFrame missing sample column"""
    data = {
        'CHROM': ['chr1', 'chr2', 'chr3'],
        'POS': [100, 200, 300],
        'ID': ['rs1', 'rs2', 'rs3'],
        'REF': ['A', 'T', 'G'],
        'ALT': ['T', 'C', 'A'],
        'QUAL': [60, 70, 80],
        'FILTER': ['PASS', 'PASS', 'PASS']
    }
    return pd.DataFrame(data)


@pytest.fixture
def large_test_data():
    """Large DataFrame for performance testing"""
    n_records = 1000
    chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
    samples = ['TCGA-AB-' + str(i).zfill(4) for i in range(2800, 3000)]
    
    data = {
        'CHROM': np.random.choice(chromosomes, n_records),
        'POS': np.random.randint(1, 250000000, n_records),
        'ID': ['rs' + str(i) for i in range(n_records)],
        'REF': np.random.choice(['A', 'T', 'G', 'C'], n_records),
        'ALT': np.random.choice(['A', 'T', 'G', 'C'], n_records),
        'QUAL': np.random.randint(20, 100, n_records),
        'FILTER': ['PASS'] * n_records,
        'Tumor_Sample_Barcode': np.random.choice(samples, n_records)
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_metadata():
    """Mock metadata object"""
    metadata = Mock()
    metadata.filters = []
    return metadata


@pytest.fixture
def mock_pymutation_maf(sample_maf_data, mock_metadata):
    """Mock PyMutation object with MAF-style data"""
    from src.pyMut.filters.chrom_sample_filter import ChromSampleFilterMixin
    
    class MockPyMutation(ChromSampleFilterMixin):
        def __init__(self, data, metadata=None, samples=None):
            self.data = data
            self.metadata = metadata
            self.samples = samples or ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2802']
    
    return MockPyMutation(sample_maf_data, mock_metadata)


@pytest.fixture
def mock_pymutation_vcf(sample_vcf_data, mock_metadata):
    """Mock PyMutation object with VCF-style data"""
    from src.pyMut.filters.chrom_sample_filter import ChromSampleFilterMixin
    
    class MockPyMutation(ChromSampleFilterMixin):
        def __init__(self, data, metadata=None, samples=None):
            self.data = data
            self.metadata = metadata
            self.samples = samples or ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2802']
    
    return MockPyMutation(sample_vcf_data, mock_metadata)


@pytest.fixture
def mock_pymutation_empty(empty_data, mock_metadata):
    """Mock PyMutation object with empty data"""
    from src.pyMut.filters.chrom_sample_filter import ChromSampleFilterMixin
    
    class MockPyMutation(ChromSampleFilterMixin):
        def __init__(self, data, metadata=None, samples=None):
            self.data = data
            self.metadata = metadata
            self.samples = samples or []
    
    return MockPyMutation(empty_data, mock_metadata)


@pytest.fixture
def mock_pymutation_missing_chrom(missing_chrom_column_data, mock_metadata):
    """Mock PyMutation object with missing CHROM column"""
    from src.pyMut.filters.chrom_sample_filter import ChromSampleFilterMixin
    
    class MockPyMutation(ChromSampleFilterMixin):
        def __init__(self, data, metadata=None, samples=None):
            self.data = data
            self.metadata = metadata
            self.samples = samples or ['TCGA-AB-2988', 'TCGA-AB-2869']
    
    return MockPyMutation(missing_chrom_column_data, mock_metadata)


@pytest.fixture
def mock_pymutation_missing_sample(missing_sample_column_data, mock_metadata):
    """Mock PyMutation object with missing sample column"""
    from src.pyMut.filters.chrom_sample_filter import ChromSampleFilterMixin
    
    class MockPyMutation(ChromSampleFilterMixin):
        def __init__(self, data, metadata=None, samples=None):
            self.data = data
            self.metadata = metadata
            self.samples = samples or ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2802']
    
    return MockPyMutation(missing_sample_column_data, mock_metadata)