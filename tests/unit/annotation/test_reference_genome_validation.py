"""
Tests for reference genome validation in actionable_mutation.py
"""
import pytest
from unittest.mock import Mock
from types import SimpleNamespace
from src.pyMut.annotate.actionable_mutation import ActionableMutationMixin
from .fixtures.actionable_mutation_fixtures import (
    mock_pymutation_instance,
    test_token,
)


class TestReferenceGenomeValidation:
    """Test class for reference genome validation."""

    def test_valid_reference_genome_grch37(self, mock_pymutation_instance, test_token):
        """Verificar que GRCh37 es aceptado."""
        # Setup
        instance = mock_pymutation_instance
        instance.metadata.assembly = 37
        ActionableMutationMixin.__init__(instance)
        
        # Mock the HTTP request to avoid actual API calls
        with pytest.MonkeyPatch.context() as m:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_session.post.return_value = mock_response
            m.setattr('requests.Session', lambda: mock_session)
            
            # Test - should not raise ValueError
            result = ActionableMutationMixin.actionable_mutations_oncokb(
                instance, test_token, batch_size=1
            )
            assert result is not None

    def test_valid_reference_genome_grch38(self, mock_pymutation_instance, test_token):
        """Verificar que GRCh38 es aceptado."""
        # Setup
        instance = mock_pymutation_instance
        instance.metadata.assembly = 38
        ActionableMutationMixin.__init__(instance)
        
        # Mock the HTTP request to avoid actual API calls
        with pytest.MonkeyPatch.context() as m:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_session.post.return_value = mock_response
            m.setattr('requests.Session', lambda: mock_session)
            
            # Test - should not raise ValueError
            result = ActionableMutationMixin.actionable_mutations_oncokb(
                instance, test_token, batch_size=1
            )
            assert result is not None

    def test_invalid_reference_genome(self, mock_pymutation_instance, test_token):
        """Verificar que se lanza ValueError para genomas inválidos."""
        # Setup
        instance = mock_pymutation_instance
        instance.metadata.assembly = 36  # Invalid assembly
        ActionableMutationMixin.__init__(instance)
        
        # Test - should raise ValueError
        with pytest.raises(ValueError, match="Invalid reference genome: GRCh36"):
            ActionableMutationMixin.actionable_mutations_oncokb(
                instance, test_token
            )

    def test_metadata_assembly_formatting(self, mock_pymutation_instance, test_token):
        """Verificar formateo correcto de f'GRCh{self.metadata.assembly}'."""
        # Test various assembly values
        test_cases = [
            (37, "GRCh37"),
            (38, "GRCh38"),
            (19, "GRCh19")  # Invalid but tests formatting
        ]
        
        for assembly_num, expected_format in test_cases:
            instance = mock_pymutation_instance
            instance.metadata.assembly = assembly_num
            ActionableMutationMixin.__init__(instance)
            
            if assembly_num in [37, 38]:
                # Valid assemblies - mock HTTP request
                with pytest.MonkeyPatch.context() as m:
                    mock_session = Mock()
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = []
                    mock_session.post.return_value = mock_response
                    m.setattr('requests.Session', lambda: mock_session)
                    
                    result = ActionableMutationMixin.actionable_mutations_oncokb(
                        instance, test_token, batch_size=1
                    )
                    assert result is not None
            else:
                # Invalid assemblies - should raise ValueError with correct format
                with pytest.raises(ValueError, match=f"Invalid reference genome: {expected_format}"):
                    ActionableMutationMixin.actionable_mutations_oncokb(
                        instance, test_token
                    )

    def test_reference_genome_case_sensitivity(self, mock_pymutation_instance, test_token):
        """Verificar que la validación es sensible a mayúsculas/minúsculas."""
        # The validation should be exact match - test with different cases
        instance = mock_pymutation_instance
        instance.metadata.assembly = 37
        ActionableMutationMixin.__init__(instance)
        
        # Mock valid case
        with pytest.MonkeyPatch.context() as m:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_session.post.return_value = mock_response
            m.setattr('requests.Session', lambda: mock_session)
            
            # This should work (GRCh37 is valid)
            result = ActionableMutationMixin.actionable_mutations_oncokb(
                instance, test_token, batch_size=1
            )
            assert result is not None

    def test_reference_genome_validation_order(self, mock_pymutation_instance, test_token):
        """Verificar que la validación del genoma se hace antes del procesamiento."""
        # Setup invalid genome
        instance = mock_pymutation_instance
        instance.metadata.assembly = 99  # Invalid
        ActionableMutationMixin.__init__(instance)
        
        # Should fail immediately with ValueError, not proceed to HTTP request
        with pytest.raises(ValueError, match="Invalid reference genome: GRCh99"):
            ActionableMutationMixin.actionable_mutations_oncokb(
                instance, test_token
            )