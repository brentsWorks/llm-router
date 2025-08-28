"""Tests to clarify validation design decisions and requirements."""

import pytest
from pydantic import ValidationError
from llm_router.utils import format_validation_error


class TestValidationDesignDecisions:
    """Test validation approach to ensure we pick the right design."""
    
    def test_should_provide_clear_error_messages_for_business_users(self):
        """Test that error messages are suitable for business users, not just developers."""
        from llm_router.models import PromptClassification
        
        with pytest.raises(ValidationError) as exc_info:
            PromptClassification(
                category="code",
                confidence=-0.1,  # Invalid
                embedding=[0.1, 0.2, 0.3]
            )
        
        # Use our utility for clear error messages
        error_message = format_validation_error(exc_info.value)
        # Business requirement: Error should be understandable by non-technical users
        # Our utility provides: "confidence must be greater than or equal to 0"
        
        # Verify our clear error message
        assert "confidence" in error_message
        assert "greater than or equal to" in error_message
    
    def test_should_validate_confidence_bounds_consistently_across_models(self):
        """Test that confidence validation is consistent across all models."""
        from llm_router.models import PromptClassification, RoutingDecision, ModelCandidate
        
        # All models should reject confidence > 1.0 with same message type
        test_cases = [
            # Test PromptClassification
            lambda: PromptClassification(category="code", confidence=1.5, embedding=[0.1]),
            # Test RoutingDecision (need valid nested objects)
            lambda: self._create_routing_decision_with_invalid_confidence()
        ]
        
        for create_invalid_model in test_cases:
            with pytest.raises(ValidationError) as exc_info:
                create_invalid_model()
            # Use our utility for clear error messages
            error_message = format_validation_error(exc_info.value)
            assert "confidence" in error_message
            assert "less than or equal to" in error_message
    
    def _create_routing_decision_with_invalid_confidence(self):
        """Helper to create RoutingDecision with invalid confidence."""
        from llm_router.models import RoutingDecision, ModelCandidate, PromptClassification
        
        classification = PromptClassification(
            category="code", confidence=0.95, embedding=[0.1, 0.2, 0.3]
        )
        selected_model = ModelCandidate(
            provider="test", model="test-model", score=0.9,
            estimated_cost=0.001, estimated_latency=100.0,
            quality_match=0.9, constraint_violations=[]
        )
        
        return RoutingDecision(
            selected_model=selected_model,
            classification=classification,
            alternatives=[],
            routing_time_ms=45.2,
            confidence=1.5  # Invalid: above 1.0
        )
    
    def test_should_handle_edge_case_validation_precisely(self):
        """Test that validation handles precise edge cases correctly."""
        from llm_router.models import PromptClassification
        
        # These should all be valid (boundary conditions)
        valid_edge_cases = [
            {"confidence": 0.0},  # Exactly 0
            {"confidence": 1.0},  # Exactly 1
            {"confidence": 0.0000001},  # Very small positive
            {"confidence": 0.9999999},  # Very close to 1
        ]
        
        for case in valid_edge_cases:
            # Should not raise exception
            classification = PromptClassification(
                category="code",
                confidence=case["confidence"],
                embedding=[0.1, 0.2, 0.3]
            )
            assert classification.confidence == case["confidence"]
    
    def test_should_reject_boundary_violations_precisely(self):
        """Test that validation rejects boundary violations correctly."""
        from llm_router.models import PromptClassification
        
        # These should all be invalid
        invalid_edge_cases = [
            -0.0000001,  # Just below 0
            1.0000001,   # Just above 1
            -1.0,        # Clearly negative
            2.0,         # Clearly above 1
        ]
        
        for invalid_confidence in invalid_edge_cases:
            with pytest.raises(ValidationError) as exc_info:
                PromptClassification(
                    category="code",
                    confidence=invalid_confidence,
                    embedding=[0.1, 0.2, 0.3]
                )
            # Use our utility for clear error messages
            error_message = format_validation_error(exc_info.value)
            assert "confidence" in error_message


class TestValidationPerformance:
    """Test that validation doesn't impact performance significantly."""
    
    def test_should_validate_quickly_for_batch_processing(self):
        """Test that validation is fast enough for batch processing."""
        import time
        from llm_router.models import PromptClassification
        
        start_time = time.time()
        
        # Create 1000 valid models to test performance
        for i in range(1000):
            PromptClassification(
                category="code",
                confidence=0.95,
                embedding=[0.1, 0.2, 0.3]
            )
        
        elapsed = time.time() - start_time
        
        # Should be very fast (less than 1 second for 1000 validations)
        assert elapsed < 1.0, f"Validation too slow: {elapsed:.3f}s for 1000 models"
