"""
Tests for constants in actionable_mutation.py
"""
import pytest
from src.pyMut.annotate.actionable_mutation import ONCOKB_ENDPOINT, VALID_REFERENCE_GENOMES


class TestConstants:
    """Test class for actionable_mutation constants."""

    def test_oncokb_endpoint_constant(self):
        """Verify that ONCOKB_ENDPOINT has the correct URL."""
        expected_url = "https://www.oncokb.org/api/v1/annotate/mutations/byGenomicChange"
        assert ONCOKB_ENDPOINT == expected_url
        assert isinstance(ONCOKB_ENDPOINT, str)
        assert ONCOKB_ENDPOINT.startswith("https://")
        assert "oncokb.org" in ONCOKB_ENDPOINT

    def test_valid_reference_genomes_constant(self):
        """Verify that VALID_REFERENCE_GENOMES contains the expected genomes."""
        expected_genomes = ["GRCh37", "GRCh38"]
        assert VALID_REFERENCE_GENOMES == expected_genomes
        assert isinstance(VALID_REFERENCE_GENOMES, list)
        assert len(VALID_REFERENCE_GENOMES) == 2
        assert "GRCh37" in VALID_REFERENCE_GENOMES
        assert "GRCh38" in VALID_REFERENCE_GENOMES

    def test_constants_immutability(self):
        """Verify that the constants are of the correct type and are well defined."""
        # ONCOKB_ENDPOINT should be a string
        assert isinstance(ONCOKB_ENDPOINT, str)
        assert len(ONCOKB_ENDPOINT) > 0
        
        # VALID_REFERENCE_GENOMES should be a list
        assert isinstance(VALID_REFERENCE_GENOMES, list)
        assert len(VALID_REFERENCE_GENOMES) > 0
        
        # All elements in VALID_REFERENCE_GENOMES should be strings
        for genome in VALID_REFERENCE_GENOMES:
            assert isinstance(genome, str)
            assert len(genome) > 0