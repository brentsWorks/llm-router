"""Tests for core data models."""

import pytest
from llm_router.models import (
    PromptClassification, ModelCandidate, RoutingDecision, VALID_CATEGORIES
)
from llm_router.capabilities import TaskType


class TestPromptClassification:
    """Test PromptClassification data model."""

    def test_should_create_valid_prompt_classification_when_all_fields_provided(self):
        """Test that PromptClassification can be created with valid data."""
        from llm_router.models import PromptClassification

        classification = PromptClassification(
            category="code",
            subcategory="python",
            confidence=0.95,
            embedding=[0.1, 0.2, 0.3],
            reasoning="Contains programming keywords"
        )

        # With use_enum_values=True, category is stored as string
        assert classification.category == "code"
        assert classification.subcategory == "python"
        assert classification.confidence == 0.95
        assert classification.embedding == [0.1, 0.2, 0.3]
        assert classification.reasoning == "Contains programming keywords"

    def test_should_create_prompt_classification_when_optional_fields_missing(self):
        """Test that PromptClassification can be created with only required fields."""
        from llm_router.models import PromptClassification

        classification = PromptClassification(
            category="creative",
            confidence=0.85,
            embedding=[0.4, 0.5, 0.6]
        )

        # With use_enum_values=True, category is stored as string
        assert classification.category == "creative"
        assert classification.subcategory is None
        assert classification.confidence == 0.85
        assert classification.embedding == [0.4, 0.5, 0.6]
        assert classification.reasoning is None

    def test_should_reject_invalid_confidence_when_below_zero(self):
        """Test that confidence below 0.0 is rejected."""
        from llm_router.models import PromptClassification

        with pytest.raises(ValueError):
            PromptClassification(
                category="qa",
                confidence=-0.1,
                embedding=[0.1, 0.2, 0.3]
            )

    def test_should_reject_invalid_confidence_when_above_one(self):
        """Test that confidence above 1.0 is rejected."""
        from llm_router.models import PromptClassification

        with pytest.raises(ValueError):
            PromptClassification(
                category="qa",
                confidence=1.1,
                embedding=[0.1, 0.2, 0.3]
            )

    def test_should_reject_invalid_category_when_not_in_allowed_list(self):
        """Test that invalid categories are rejected."""
        from llm_router.models import PromptClassification

        with pytest.raises(ValueError):
            PromptClassification(
                category="invalid_category",
                confidence=0.8,
                embedding=[0.1, 0.2, 0.3]
            )

    def test_should_accept_all_valid_categories(self):
        """Test that all valid categories are accepted."""
        from llm_router.models import PromptClassification, VALID_CATEGORIES

        for category in VALID_CATEGORIES:
            classification = PromptClassification(
                category=category,
                confidence=0.8,
                embedding=[0.1, 0.2, 0.3]
            )
            # With use_enum_values=True, category is stored as string
            assert classification.category == category

    def test_should_serialize_to_dict_correctly(self):
        """Test that PromptClassification serializes to dict correctly."""
        from llm_router.models import PromptClassification

        classification = PromptClassification(
            category="qa",
            subcategory="factual",
            confidence=0.92,
            embedding=[0.1, 0.2],
            reasoning="Question format detected"
        )

        result = classification.model_dump()

        expected = {
            "category": "qa",
            "subcategory": "factual",
            "confidence": 0.92,
            "embedding": [0.1, 0.2],
            "reasoning": "Question format detected"
        }
        assert result == expected

    def test_should_deserialize_from_dict_correctly(self):
        """Test that PromptClassification can be created from dict."""
        from llm_router.models import PromptClassification

        data = {
            "category": "summarization",
            "confidence": 0.88,
            "embedding": [0.3, 0.4, 0.5]
        }

        classification = PromptClassification(**data)

        # With use_enum_values=True, category is stored as string
        assert classification.category == "summarization"
        assert classification.confidence == 0.88
        assert classification.embedding == [0.3, 0.4, 0.5]


class TestModelCandidate:
    """Test ModelCandidate data model."""

    def test_should_create_valid_model_candidate_when_all_fields_provided(self):
        """Test that ModelCandidate can be created with valid data."""
        from llm_router.models import ModelCandidate

        candidate = ModelCandidate(
            provider="openai",
            model="gpt-4",
            score=0.95,
            estimated_cost=0.03,
            estimated_latency=500.0,
            quality_match=0.92,
            constraint_violations=["context_length"]
        )

        assert candidate.provider == "openai"
        assert candidate.model == "gpt-4"
        assert candidate.score == 0.95
        assert candidate.estimated_cost == 0.03
        assert candidate.estimated_latency == 500.0
        assert candidate.quality_match == 0.92
        assert candidate.constraint_violations == ["context_length"]

    def test_should_reject_negative_score_when_below_zero(self):
        """Test that negative scores are rejected."""
        from llm_router.models import ModelCandidate

        with pytest.raises(ValueError):
            ModelCandidate(
                provider="openai",
                model="gpt-4",
                score=-0.1,
                estimated_cost=0.03,
                estimated_latency=500.0,
                quality_match=0.92
            )

    def test_should_reject_negative_cost_when_below_zero(self):
        """Test that negative costs are rejected."""
        from llm_router.models import ModelCandidate

        with pytest.raises(ValueError):
            ModelCandidate(
                provider="openai",
                model="gpt-4",
                score=0.95,
                estimated_cost=-0.01,
                estimated_latency=500.0,
                quality_match=0.92
            )

    def test_should_reject_negative_latency_when_below_zero(self):
        """Test that negative latency is rejected."""
        from llm_router.models import ModelCandidate

        with pytest.raises(ValueError):
            ModelCandidate(
                provider="openai",
                model="gpt-4",
                score=0.95,
                estimated_cost=0.03,
                estimated_latency=-100.0,
                quality_match=0.92
            )

    def test_should_support_comparison_by_score_for_sorting(self):
        """Test that ModelCandidate can be compared by score for sorting."""
        from llm_router.models import ModelCandidate

        candidate1 = ModelCandidate(
            provider="openai", model="gpt-4", score=0.95,
            estimated_cost=0.03, estimated_latency=500.0, quality_match=0.92
        )
        candidate2 = ModelCandidate(
            provider="anthropic", model="claude-3", score=0.87,
            estimated_cost=0.02, estimated_latency=400.0, quality_match=0.89
        )

        # Should be able to sort by score
        candidates = [candidate2, candidate1]
        candidates.sort(key=lambda c: c.score, reverse=True)
        
        assert candidates[0].score == 0.95
        assert candidates[1].score == 0.87

    def test_should_serialize_to_dict_correctly(self):
        """Test that ModelCandidate serializes to dict correctly."""
        from llm_router.models import ModelCandidate

        candidate = ModelCandidate(
            provider="openai",
            model="gpt-4",
            score=0.95,
            estimated_cost=0.03,
            estimated_latency=500.0,
            quality_match=0.92
        )

        result = candidate.model_dump()

        expected = {
            "provider": "openai",
            "model": "gpt-4",
            "score": 0.95,
            "estimated_cost": 0.03,
            "estimated_latency": 500.0,
            "quality_match": 0.92,
            "constraint_violations": []
        }
        assert result == expected


class TestRoutingDecision:
    """Test RoutingDecision data model."""

    def test_should_create_valid_routing_decision_when_all_fields_provided(self):
        """Test that RoutingDecision can be created with valid data."""
        from llm_router.models import RoutingDecision, ModelCandidate, PromptClassification

        classification = PromptClassification(
            category="creative", confidence=0.88, embedding=[0.4, 0.5]
        )
        selected_model = ModelCandidate(
            provider="anthropic", model="claude-3-sonnet", score=0.95,
            estimated_cost=0.015, estimated_latency=600.0,
            quality_match=0.9, constraint_violations=[]
        )
        alternative = ModelCandidate(
            provider="openai", model="gpt-4", score=0.87,
            estimated_cost=0.03, estimated_latency=500.0,
            quality_match=0.88, constraint_violations=[]
        )

        decision = RoutingDecision(
            selected_model=selected_model,
            classification=classification,
            alternatives=[alternative],
            routing_time_ms=32.1,
            confidence=0.89
        )

        assert decision.selected_model == selected_model
        assert decision.classification == classification
        assert len(decision.alternatives) == 1
        assert decision.alternatives[0] == alternative
        assert decision.routing_time_ms == 32.1
        assert decision.confidence == 0.89

    def test_should_reject_invalid_confidence_when_out_of_bounds(self):
        """Test that confidence outside 0-1 range is rejected."""
        from llm_router.models import RoutingDecision, ModelCandidate, PromptClassification

        classification = PromptClassification(
            category="qa", confidence=0.8, embedding=[0.1, 0.2]
        )
        selected_model = ModelCandidate(
            provider="openai", model="gpt-4", score=0.9,
            estimated_cost=0.03, estimated_latency=500.0, quality_match=0.9
        )

        # Test confidence below 0
        with pytest.raises(ValueError):
            RoutingDecision(
                selected_model=selected_model,
                classification=classification,
                routing_time_ms=25.0,
                confidence=-0.1
            )

        # Test confidence above 1
        with pytest.raises(ValueError):
            RoutingDecision(
                selected_model=selected_model,
                classification=classification,
                routing_time_ms=25.0,
                confidence=1.1
            )

    def test_should_reject_negative_routing_time_when_below_zero(self):
        """Test that negative routing time is rejected."""
        from llm_router.models import RoutingDecision, ModelCandidate, PromptClassification

        classification = PromptClassification(
            category="qa", confidence=0.8, embedding=[0.1, 0.2]
        )
        selected_model = ModelCandidate(
            provider="openai", model="gpt-4", score=0.9,
            estimated_cost=0.03, estimated_latency=500.0, quality_match=0.9
        )

        with pytest.raises(ValueError):
            RoutingDecision(
                selected_model=selected_model,
                classification=classification,
                routing_time_ms=-10.0,
                confidence=0.9
            )

    def test_should_serialize_to_dict_with_nested_objects(self):
        """Test that RoutingDecision serializes with all nested objects."""
        from llm_router.models import RoutingDecision, ModelCandidate, PromptClassification

        classification = PromptClassification(
            category="creative", confidence=0.88, embedding=[0.4, 0.5]
        )
        selected_model = ModelCandidate(
            provider="anthropic", model="claude-3-sonnet", score=0.95,
            estimated_cost=0.015, estimated_latency=600.0,
            quality_match=0.9, constraint_violations=[]
        )

        decision = RoutingDecision(
            selected_model=selected_model,
            classification=classification,
            alternatives=[],
            routing_time_ms=32.1,
            confidence=0.89
        )

        result = decision.model_dump()

        # Should contain nested serialized objects
        assert "selected_model" in result
        assert "classification" in result
        assert result["selected_model"]["provider"] == "anthropic"
        # With use_enum_values=True, category is serialized as string
        assert result["classification"]["category"] == "creative"
        assert result["routing_time_ms"] == 32.1
        assert result["confidence"] == 0.89
