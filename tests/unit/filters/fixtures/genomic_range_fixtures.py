import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock


@pytest.fixture
def genomic_range_data():
    """Sample data for genomic range testing"""
    data = {
        'CHROM': ['chr1', 'chr1', 'chr1', 'chr2', 'chr2', 'chrX', 'chrY', 'chr17'],
        'POS': [100, 150, 300, 200, 250, 400, 500, 43094077],
        'ID': ['rs1', 'rs2', 'rs3', 'rs4', 'rs5', 'rs6', 'rs7', 'rs8'],
        'REF': ['A', 'T', 'G', 'C', 'A', 'T', 'G', 'C'],
        'ALT': ['T', 'C', 'A', 'G', 'T', 'C', 'A', 'G'],
        'QUAL': [60, 70, 80, 90, 95, 85, 75, 99],
        'FILTER': ['PASS'] * 8,
        'Hugo_Symbol': ['GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5', 'GENEX', 'GENEY', 'TP53']
    }
    return pd.DataFrame(data)


@pytest.fixture
def maf_format_data():
    """MAF format data for gene region testing"""
    data = {
        'CHROM': ['chr17', 'chr17', 'chr12', 'chr7', 'chr3', 'chr17', 'chr1'],
        'POS': [43094077, 43094078, 25398285, 55259515, 178917000, 43094079, 115252000],
        'ID': ['rs1', 'rs2', 'rs3', 'rs4', 'rs5', 'rs6', 'rs7'],
        'REF': ['G', 'C', 'G', 'A', 'C', 'A', 'T'],
        'ALT': ['A', 'T', 'T', 'G', 'T', 'G', 'C'],
        'QUAL': [99, 95, 88, 92, 85, 97, 90],
        'FILTER': ['PASS'] * 7,
        'Hugo_Symbol': ['TP53', 'TP53', 'KRAS', 'EGFR', 'PIK3CA', 'TP53', 'NRAS'],
        'Tumor_Sample_Barcode': ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2988', 
                                'TCGA-AB-2802', 'TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2802']
    }
    return pd.DataFrame(data)


@pytest.fixture
def mixed_chromosomes_genomic_data():
    """Data with mixed chromosome formatting for genomic range testing"""
    data = {
        'CHROM': ['1', 'chr1', '2', 'chrX', 'Y', 'chrMT', '17'],
        'POS': [100, 200, 300, 400, 500, 16569, 43094077],
        'ID': ['rs1', 'rs2', 'rs3', 'rs4', 'rs5', 'rs6', 'rs7'],
        'REF': ['A', 'T', 'G', 'C', 'A', 'T', 'G'],
        'ALT': ['T', 'C', 'A', 'G', 'T', 'C', 'A'],
        'QUAL': [60, 70, 80, 90, 95, 85, 75],
        'FILTER': ['PASS'] * 7
    }
    return pd.DataFrame(data)


@pytest.fixture
def large_genomic_data():
    """Large dataset for performance testing"""
    n_records = 5000
    chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
    
    # Generate positions with some clustering for realistic testing
    positions = []
    for _ in range(n_records):
        # Generate clustered positions around certain regions
        base_pos = np.random.choice([100000, 500000, 1000000, 5000000, 10000000])
        offset = np.random.randint(-50000, 50000)
        positions.append(max(1, base_pos + offset))
    
    data = {
        'CHROM': np.random.choice(chromosomes, n_records),
        'POS': positions,
        'ID': ['rs' + str(i) for i in range(n_records)],
        'REF': np.random.choice(['A', 'T', 'G', 'C'], n_records),
        'ALT': np.random.choice(['A', 'T', 'G', 'C'], n_records),
        'QUAL': np.random.randint(20, 100, n_records),
        'FILTER': ['PASS'] * n_records
    }
    return pd.DataFrame(data)


@pytest.fixture
def empty_genomic_data():
    """Empty DataFrame for genomic range testing"""
    columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER']
    return pd.DataFrame(columns=columns)


@pytest.fixture
def missing_chrom_genomic_data():
    """Data missing CHROM column"""
    data = {
        'POS': [100, 200, 300],
        'ID': ['rs1', 'rs2', 'rs3'],
        'REF': ['A', 'T', 'G'],
        'ALT': ['T', 'C', 'A'],
        'QUAL': [60, 70, 80],
        'FILTER': ['PASS', 'PASS', 'PASS']
    }
    return pd.DataFrame(data)


@pytest.fixture
def missing_pos_genomic_data():
    """Data missing POS column"""
    data = {
        'CHROM': ['chr1', 'chr2', 'chr3'],
        'ID': ['rs1', 'rs2', 'rs3'],
        'REF': ['A', 'T', 'G'],
        'ALT': ['T', 'C', 'A'],
        'QUAL': [60, 70, 80],
        'FILTER': ['PASS', 'PASS', 'PASS']
    }
    return pd.DataFrame(data)


@pytest.fixture
def single_variant_data():
    """Single variant data for edge case testing"""
    data = {
        'CHROM': ['chr1'],
        'POS': [150],
        'ID': ['rs1'],
        'REF': ['A'],
        'ALT': ['T'],
        'QUAL': [90],
        'FILTER': ['PASS']
    }
    return pd.DataFrame(data)


@pytest.fixture
def negative_coordinates_data():
    """Data with negative coordinates (edge case)"""
    data = {
        'CHROM': ['chr1', 'chr1', 'chr2'],
        'POS': [-10, 100, 200],
        'ID': ['rs1', 'rs2', 'rs3'],
        'REF': ['A', 'T', 'G'],
        'ALT': ['T', 'C', 'A'],
        'QUAL': [60, 70, 80],
        'FILTER': ['PASS', 'PASS', 'PASS']
    }
    return pd.DataFrame(data)


@pytest.fixture
def different_coordinate_types_data():
    """Data with different coordinate data types"""
    data = {
        'CHROM': ['chr1', 'chr1', 'chr2', 'chr2'],
        'POS': [100, 200, 300, 400],  # All converted to int for consistency
        'ID': ['rs1', 'rs2', 'rs3', 'rs4'],
        'REF': ['A', 'T', 'G', 'C'],
        'ALT': ['T', 'C', 'A', 'G'],
        'QUAL': [60, 70, 80, 90],
        'FILTER': ['PASS', 'PASS', 'PASS', 'PASS']
    }
    return pd.DataFrame(data)


@pytest.fixture
def maf_missing_hugo_symbol():
    """MAF data missing Hugo_Symbol column"""
    data = {
        'CHROM': ['chr17', 'chr12', 'chr7'],
        'POS': [43094077, 25398285, 55259515],
        'ID': ['rs1', 'rs2', 'rs3'],
        'REF': ['G', 'G', 'A'],
        'ALT': ['A', 'T', 'G'],
        'QUAL': [99, 88, 92],
        'FILTER': ['PASS', 'PASS', 'PASS'],
        'Tumor_Sample_Barcode': ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2802']
    }
    return pd.DataFrame(data)


@pytest.fixture
def maf_case_insensitive_hugo():
    """MAF data with case variations in Hugo_Symbol column name"""
    data = {
        'CHROM': ['chr17', 'chr12', 'chr7'],
        'POS': [43094077, 25398285, 55259515],
        'ID': ['rs1', 'rs2', 'rs3'],
        'REF': ['G', 'G', 'A'],
        'ALT': ['A', 'T', 'G'],
        'QUAL': [99, 88, 92],
        'FILTER': ['PASS', 'PASS', 'PASS'],
        'hugo_symbol': ['TP53', 'KRAS', 'EGFR'],  # lowercase column name
        'Tumor_Sample_Barcode': ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2802']
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_genomic_metadata():
    """Mock metadata object for genomic range testing"""
    metadata = Mock()
    metadata.filters = []
    metadata.source_format = "VCF"
    return metadata


@pytest.fixture
def mock_maf_metadata():
    """Mock metadata object for MAF testing"""
    metadata = Mock()
    metadata.filters = []
    metadata.source_format = "MAF"
    return metadata


@pytest.fixture
def mock_pymutation_genomic(genomic_range_data, mock_genomic_metadata):
    """Mock PyMutation object with genomic range data"""
    from src.pyMut.filters.genomic_range import GenomicRangeMixin
    
    class MockPyMutation(GenomicRangeMixin):
        def __init__(self, data, metadata, samples=None):
            self.data = data
            self.metadata = metadata
            self.samples = samples or []
    
    return MockPyMutation(genomic_range_data, mock_genomic_metadata)


@pytest.fixture
def mock_pymutation_maf_genomic(maf_format_data, mock_maf_metadata):
    """Mock PyMutation object with MAF data"""
    from src.pyMut.filters.genomic_range import GenomicRangeMixin
    
    class MockPyMutation(GenomicRangeMixin):
        def __init__(self, data, metadata, samples=None):
            self.data = data
            self.metadata = metadata
            self.samples = samples or ['TCGA-AB-2988', 'TCGA-AB-2869', 'TCGA-AB-2802']
    
    return MockPyMutation(maf_format_data, mock_maf_metadata)


@pytest.fixture
def mock_pymutation_empty_genomic(empty_genomic_data, mock_genomic_metadata):
    """Mock PyMutation object with empty data"""
    from src.pyMut.filters.genomic_range import GenomicRangeMixin
    
    class MockPyMutation(GenomicRangeMixin):
        def __init__(self, data, metadata, samples=None):
            self.data = data
            self.metadata = metadata
            self.samples = samples or []
    
    return MockPyMutation(empty_genomic_data, mock_genomic_metadata)


@pytest.fixture
def mock_pymutation_missing_chrom(missing_chrom_genomic_data, mock_genomic_metadata):
    """Mock PyMutation object with missing CHROM column"""
    from src.pyMut.filters.genomic_range import GenomicRangeMixin
    
    class MockPyMutation(GenomicRangeMixin):
        def __init__(self, data, metadata, samples=None):
            self.data = data
            self.metadata = metadata
            self.samples = samples or []
    
    return MockPyMutation(missing_chrom_genomic_data, mock_genomic_metadata)


@pytest.fixture
def mock_pymutation_missing_pos(missing_pos_genomic_data, mock_genomic_metadata):
    """Mock PyMutation object with missing POS column"""
    from src.pyMut.filters.genomic_range import GenomicRangeMixin
    
    class MockPyMutation(GenomicRangeMixin):
        def __init__(self, data, metadata, samples=None):
            self.data = data
            self.metadata = metadata
            self.samples = samples or []
    
    return MockPyMutation(missing_pos_genomic_data, mock_genomic_metadata)


@pytest.fixture
def mock_pymutation_large_genomic(large_genomic_data, mock_genomic_metadata):
    """Mock PyMutation object with large dataset"""
    from src.pyMut.filters.genomic_range import GenomicRangeMixin
    
    class MockPyMutation(GenomicRangeMixin):
        def __init__(self, data, metadata, samples=None):
            self.data = data
            self.metadata = metadata
            self.samples = samples or []
    
    return MockPyMutation(large_genomic_data, mock_genomic_metadata)