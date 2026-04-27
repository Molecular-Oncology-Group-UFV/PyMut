"""
Fixtures and test utilities for annotation tests (actionable_mutation, etc.).
"""
import pytest
import pandas as pd
from unittest.mock import Mock
import requests
from types import SimpleNamespace


@pytest.fixture
def valid_mutation_data():
    """DataFrame with valid mutation data for testing."""
    return pd.DataFrame({
        'CHROM': ['1', '2', 'chr3', '4', 'X'],
        'POS': [100, 200, 300, 400, 500],
        'REF': ['A', 'T', 'GC', 'C', 'G'],
        'ALT': ['G', 'C', 'A', 'T', 'A'],
        'SAMPLE': ['S1', 'S2', 'S3', 'S4', 'S5']
    })


@pytest.fixture
def invalid_mutation_data_missing_chrom():
    """DataFrame missing CHROM column."""
    return pd.DataFrame({
        'POS': [100, 200, 300],
        'REF': ['A', 'T', 'GC'],
        'ALT': ['G', 'C', 'A']
    })


@pytest.fixture
def invalid_mutation_data_missing_pos():
    """DataFrame missing POS column."""
    return pd.DataFrame({
        'CHROM': ['1', '2', '3'],
        'REF': ['A', 'T', 'GC'],
        'ALT': ['G', 'C', 'A']
    })


@pytest.fixture
def invalid_mutation_data_missing_ref():
    """DataFrame missing REF column."""
    return pd.DataFrame({
        'CHROM': ['1', '2', '3'],
        'POS': [100, 200, 300],
        'ALT': ['G', 'C', 'A']
    })


@pytest.fixture
def invalid_mutation_data_missing_alt():
    """DataFrame missing ALT column."""
    return pd.DataFrame({
        'CHROM': ['1', '2', '3'],
        'POS': [100, 200, 300],
        'REF': ['A', 'T', 'GC']
    })


@pytest.fixture
def mutation_data_with_nulls():
    """DataFrame with null values in various columns."""
    return pd.DataFrame({
        'CHROM': ['1', None, '3', '4', '5'],
        'POS': [100, 200, None, 400, 500],
        'REF': ['A', 'T', 'GC', None, 'G'],
        'ALT': ['G', 'C', 'A', 'T', None],
        'SAMPLE': ['S1', 'S2', 'S3', 'S4', 'S5']
    })


@pytest.fixture
def single_variant_data():
    """DataFrame with a single variant."""
    return pd.DataFrame({
        'CHROM': ['1'],
        'POS': [100],
        'REF': ['A'],
        'ALT': ['G']
    })


@pytest.fixture
def test_token():
    """Test OncoKB API token."""
    return "test_token_12345"


@pytest.fixture
def mock_metadata_grch37():
    """Mock metadata with GRCh37 assembly."""
    metadata = SimpleNamespace()
    metadata.assembly = 37
    return metadata


@pytest.fixture
def mock_metadata_grch38():
    """Mock metadata with GRCh38 assembly."""
    metadata = SimpleNamespace()
    metadata.assembly = 38
    return metadata


@pytest.fixture
def mock_metadata_invalid():
    """Mock metadata with invalid assembly."""
    metadata = SimpleNamespace()
    metadata.assembly = 36
    return metadata


@pytest.fixture
def mock_oncokb_response_success():
    """Mock successful OncoKB API response."""
    return [
        {
            "geneExist": True,
            "oncogenic": "Oncogenic",
            "highestSensitiveLevel": "LEVEL_1",
            "highestResistanceLevel": None,
            "highestDiagnosticImplicationLevel": None,
            "highestPrognosticImplicationLevel": None,
            "otherSignificantSensitiveLevels": [],
            "otherSignificantResistanceLevels": [],
            "hotspot": True,
            "geneSummary": "Gene summary text",
            "variantSummary": "Variant summary text",
            "tumorTypeSummary": "Tumor type summary",
            "prognosticSummary": "",
            "diagnosticSummary": "",
            "diagnosticImplications": [],
            "prognosticImplications": [],
            "treatments": [{"level": "LEVEL_1", "drugs": ["Drug A"]}],
            "dataVersion": "v3.14",
            "lastUpdate": "2023-01-01",
            "vus": False
        },
        {
            "geneExist": True,
            "oncogenic": "Unknown",
            "highestSensitiveLevel": None,
            "highestResistanceLevel": "LEVEL_R1",
            "highestDiagnosticImplicationLevel": None,
            "highestPrognosticImplicationLevel": None,
            "otherSignificantSensitiveLevels": [],
            "otherSignificantResistanceLevels": [],
            "hotspot": False,
            "geneSummary": "Another gene summary",
            "variantSummary": "Another variant summary",
            "tumorTypeSummary": "",
            "prognosticSummary": "",
            "diagnosticSummary": "",
            "diagnosticImplications": [],
            "prognosticImplications": [],
            "treatments": [],
            "dataVersion": "v3.14",
            "lastUpdate": "2023-01-01",
            "vus": True
        }
    ]


@pytest.fixture
def mock_oncokb_response_gene_not_exist():
    """Mock OncoKB API response with non-existent gene."""
    return [
        {
            "geneExist": False,
            "oncogenic": "Unknown",
            "highestSensitiveLevel": None,
            "highestResistanceLevel": None,
            "highestDiagnosticImplicationLevel": None,
            "highestPrognosticImplicationLevel": None,
            "otherSignificantSensitiveLevels": [],
            "otherSignificantResistanceLevels": [],
            "hotspot": False,
            "geneSummary": "",
            "variantSummary": "",
            "tumorTypeSummary": "",
            "prognosticSummary": "",
            "diagnosticSummary": "",
            "diagnosticImplications": [],
            "prognosticImplications": [],
            "treatments": [],
            "dataVersion": "v3.14",
            "lastUpdate": "2023-01-01",
            "vus": False
        }
    ]


@pytest.fixture
def mock_session():
    """Mock HTTP session for testing."""
    session = Mock(spec=requests.Session)
    response = Mock()
    response.status_code = 200
    response.json.return_value = []
    session.post.return_value = response
    return session


@pytest.fixture
def mock_pymutation_instance():
    """Mock PyMutation instance for testing."""
    instance = Mock()
    instance.data = pd.DataFrame({
        'CHROM': ['1', '2', '3'],
        'POS': [100, 200, 300],
        'REF': ['A', 'T', 'GC'],
        'ALT': ['G', 'C', 'A']
    })
    instance.metadata = SimpleNamespace()
    instance.metadata.assembly = 37
    return instance


@pytest.fixture
def large_mutation_data():
    """Large DataFrame for batch testing."""
    size = 12000  # More than default batch_size of 5000
    return pd.DataFrame({
        'CHROM': [str(i % 22 + 1) for i in range(size)],
        'POS': [100 + i for i in range(size)],
        'REF': [['A', 'T', 'G', 'C'][i % 4] for i in range(size)],
        'ALT': [['T', 'A', 'C', 'G'][i % 4] for i in range(size)]
    })


@pytest.fixture
def various_batch_sizes():
    """Various batch sizes for testing."""
    return [1, 10, 100, 1000, 5000, 10000]


@pytest.fixture
def mock_requests_exception():
    """Mock requests exception for testing."""
    return requests.exceptions.ConnectionError("Connection failed")


@pytest.fixture
def mock_timeout_exception():
    """Mock timeout exception for testing."""
    return requests.exceptions.Timeout("Request timed out")


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame(columns=['CHROM', 'POS', 'REF', 'ALT'])
