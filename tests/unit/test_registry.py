"""Tests for provider registry data model and operations."""

import pytest
from typing import List
from pydantic import ValidationError


class TestProviderModel:
    """Test ProviderModel data structure and validation."""

    def test_should_create_valid_provider_model_with_all_fields(self):
        """Test that ProviderModel can be created with complete data."""
        from llm_router.registry import ProviderModel
        
        model = ProviderModel(
            provider="openai",
            model="gpt-3.5-turbo",
            capabilities=["code", "creative", "qa", "summarization"],
            pricing={
                "input_tokens_per_1k": 0.001,
                "output_tokens_per_1k": 0.002
            },
            limits={
                "context_length": 4096,
                "rate_limit": 3500,
                "safety_level": "moderate"
            },
            performance={
                "avg_latency_ms": 800,
                "quality_scores": {
                    "code": 0.85,
                    "creative": 0.9,
                    "reasoning": 0.88,
                    "summarization": 0.92
                }
            }
        )
        
        assert model.provider == "openai"
        assert model.model == "gpt-3.5-turbo"
        assert "code" in model.capabilities
        assert model.pricing["input_tokens_per_1k"] == 0.001
        assert model.limits["context_length"] == 4096
        assert model.performance["avg_latency_ms"] == 800

    def test_should_validate_pricing_structure(self):
        """Test that pricing must have required fields."""
        from llm_router.registry import ProviderModel
        
        # Valid pricing should work
        valid_model = ProviderModel(
            provider="test",
            model="test-model",
            capabilities=["qa"],
            pricing={
                "input_tokens_per_1k": 0.001,
                "output_tokens_per_1k": 0.002
            },
            limits={"context_length": 2048},
            performance={"avg_latency_ms": 500}
        )
        assert valid_model.pricing["input_tokens_per_1k"] == 0.001

        # Missing required pricing fields should raise error
        with pytest.raises(ValidationError) as exc_info:
            ProviderModel(
                provider="test",
                model="test-model", 
                capabilities=["qa"],
                pricing={"input_tokens_per_1k": 0.001},  # Missing output pricing
                limits={"context_length": 2048},
                performance={"avg_latency_ms": 500}
            )
        assert "output_tokens_per_1k" in str(exc_info.value).lower()

    def test_should_validate_capabilities_are_valid_categories(self):
        """Test that capabilities must be from valid category list."""
        from llm_router.registry import ProviderModel
        
        # Valid capabilities should work
        valid_model = ProviderModel(
            provider="test",
            model="test-model",
            capabilities=["code", "creative", "qa"],
            pricing={"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
            limits={"context_length": 2048},
            performance={"avg_latency_ms": 500}
        )
        assert "code" in valid_model.capabilities

        # Invalid capability should raise error
        with pytest.raises(ValidationError) as exc_info:
            ProviderModel(
                provider="test",
                model="test-model",
                capabilities=["invalid_capability"],
                pricing={"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
                limits={"context_length": 2048},
                performance={"avg_latency_ms": 500}
            )
        assert "capability" in str(exc_info.value).lower()

    def test_should_validate_non_negative_pricing(self):
        """Test that pricing values must be non-negative."""
        from llm_router.registry import ProviderModel
        
        # Negative pricing should raise error
        with pytest.raises(ValidationError):
            ProviderModel(
                provider="test",
                model="test-model",
                capabilities=["qa"],
                pricing={
                    "input_tokens_per_1k": -0.001,  # Invalid: negative
                    "output_tokens_per_1k": 0.002
                },
                limits={"context_length": 2048},
                performance={"avg_latency_ms": 500}
            )

    def test_should_validate_positive_context_length(self):
        """Test that context length must be positive."""
        from llm_router.registry import ProviderModel
        
        # Zero or negative context length should raise error
        with pytest.raises(ValidationError):
            ProviderModel(
                provider="test",
                model="test-model",
                capabilities=["qa"],
                pricing={"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
                limits={"context_length": 0},  # Invalid: zero context length
                performance={"avg_latency_ms": 500}
            )

    def test_should_validate_quality_scores_bounds(self):
        """Test that quality scores must be between 0 and 1."""
        from llm_router.registry import ProviderModel
        
        # Invalid quality score should raise error
        with pytest.raises(ValidationError):
            ProviderModel(
                provider="test",
                model="test-model",
                capabilities=["code"],
                pricing={"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
                limits={"context_length": 2048},
                performance={
                    "avg_latency_ms": 500,
                    "quality_scores": {
                        "code": 1.5  # Invalid: above 1.0
                    }
                }
            )

    def test_should_serialize_to_dict_correctly(self):
        """Test that ProviderModel serializes properly."""
        from llm_router.registry import ProviderModel
        
        model = ProviderModel(
            provider="anthropic",
            model="claude-3-haiku",
            capabilities=["creative", "reasoning"],
            pricing={"input_tokens_per_1k": 0.00025, "output_tokens_per_1k": 0.00125},
            limits={"context_length": 200000, "safety_level": "high"},
            performance={
                "avg_latency_ms": 400,
                "quality_scores": {"creative": 0.95, "reasoning": 0.88}
            }
        )
        
        result = model.model_dump()
        
        assert result["provider"] == "anthropic"
        assert result["model"] == "claude-3-haiku"
        assert "creative" in result["capabilities"]
        assert result["pricing"]["input_tokens_per_1k"] == 0.00025
        assert result["limits"]["context_length"] == 200000


class TestProviderRegistry:
    """Test ProviderRegistry operations and functionality."""

    def test_should_create_empty_registry(self):
        """Test that ProviderRegistry can be created empty."""
        from llm_router.registry import ProviderRegistry
        
        registry = ProviderRegistry()
        
        assert len(registry.get_all_models()) == 0
        assert registry.get_providers() == []

    def test_should_add_provider_model_to_registry(self):
        """Test that models can be added to the registry."""
        from llm_router.registry import ProviderRegistry, ProviderModel
        
        registry = ProviderRegistry()
        
        model = ProviderModel(
            provider="openai",
            model="gpt-3.5-turbo",
            capabilities=["code", "qa"],
            pricing={"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
            limits={"context_length": 4096},
            performance={"avg_latency_ms": 800}
        )
        
        registry.add_model(model)
        
        assert len(registry.get_all_models()) == 1
        assert "openai" in registry.get_providers()

    def test_should_prevent_duplicate_model_addition(self):
        """Test that adding the same model twice raises an error."""
        from llm_router.registry import ProviderRegistry, ProviderModel
        
        registry = ProviderRegistry()
        
        model = ProviderModel(
            provider="openai",
            model="gpt-3.5-turbo",
            capabilities=["code"],
            pricing={"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
            limits={"context_length": 4096},
            performance={"avg_latency_ms": 800}
        )
        
        registry.add_model(model)
        
        # Adding the same model again should raise error
        with pytest.raises(ValueError) as exc_info:
            registry.add_model(model)
        assert "already exists" in str(exc_info.value).lower()

    def test_should_get_models_by_capability(self):
        """Test that models can be filtered by capability."""
        from llm_router.registry import ProviderRegistry, ProviderModel
        
        registry = ProviderRegistry()
        
        # Add model with code capability
        code_model = ProviderModel(
            provider="openai", model="gpt-3.5-turbo",
            capabilities=["code", "qa"],
            pricing={"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
            limits={"context_length": 4096},
            performance={"avg_latency_ms": 800}
        )
        
        # Add model with only creative capability
        creative_model = ProviderModel(
            provider="anthropic", model="claude-3-haiku",
            capabilities=["creative"],
            pricing={"input_tokens_per_1k": 0.00025, "output_tokens_per_1k": 0.00125},
            limits={"context_length": 200000},
            performance={"avg_latency_ms": 400}
        )
        
        registry.add_model(code_model)
        registry.add_model(creative_model)
        
        # Filter by capability
        code_models = registry.get_models_by_capability("code")
        creative_models = registry.get_models_by_capability("creative")
        
        assert len(code_models) == 1
        assert code_models[0].model == "gpt-3.5-turbo"
        assert len(creative_models) == 1
        assert creative_models[0].model == "claude-3-haiku"

    def test_should_get_models_by_provider(self):
        """Test that models can be filtered by provider."""
        from llm_router.registry import ProviderRegistry, ProviderModel
        
        registry = ProviderRegistry()
        
        # Add OpenAI models
        openai_model1 = ProviderModel(
            provider="openai", model="gpt-3.5-turbo",
            capabilities=["code"], pricing={"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
            limits={"context_length": 4096}, performance={"avg_latency_ms": 800}
        )
        
        openai_model2 = ProviderModel(
            provider="openai", model="gpt-4",
            capabilities=["reasoning"], pricing={"input_tokens_per_1k": 0.03, "output_tokens_per_1k": 0.06},
            limits={"context_length": 8192}, performance={"avg_latency_ms": 2000}
        )
        
        # Add Anthropic model
        anthropic_model = ProviderModel(
            provider="anthropic", model="claude-3-haiku",
            capabilities=["creative"], pricing={"input_tokens_per_1k": 0.00025, "output_tokens_per_1k": 0.00125},
            limits={"context_length": 200000}, performance={"avg_latency_ms": 400}
        )
        
        registry.add_model(openai_model1)
        registry.add_model(openai_model2)
        registry.add_model(anthropic_model)
        
        # Filter by provider
        openai_models = registry.get_models_by_provider("openai")
        anthropic_models = registry.get_models_by_provider("anthropic")
        
        assert len(openai_models) == 2
        assert len(anthropic_models) == 1
        assert all(m.provider == "openai" for m in openai_models)

    def test_should_get_model_by_full_name(self):
        """Test that specific models can be retrieved by provider/model name."""
        from llm_router.registry import ProviderRegistry, ProviderModel
        
        registry = ProviderRegistry()
        
        model = ProviderModel(
            provider="openai", model="gpt-3.5-turbo",
            capabilities=["code"], pricing={"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
            limits={"context_length": 4096}, performance={"avg_latency_ms": 800}
        )
        
        registry.add_model(model)
        
        # Should find the model
        found_model = registry.get_model("openai", "gpt-3.5-turbo")
        assert found_model is not None
        assert found_model.provider == "openai"
        assert found_model.model == "gpt-3.5-turbo"
        
        # Should return None for non-existent model
        not_found = registry.get_model("openai", "gpt-5")
        assert not_found is None

    def test_should_filter_models_by_constraints(self):
        """Test that models can be filtered by constraints like context length."""
        from llm_router.registry import ProviderRegistry, ProviderModel
        
        registry = ProviderRegistry()
        
        # Small context model
        small_model = ProviderModel(
            provider="openai", model="gpt-3.5-turbo",
            capabilities=["code"], pricing={"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
            limits={"context_length": 4096}, performance={"avg_latency_ms": 800}
        )
        
        # Large context model
        large_model = ProviderModel(
            provider="anthropic", model="claude-3-haiku",
            capabilities=["code"], pricing={"input_tokens_per_1k": 0.00025, "output_tokens_per_1k": 0.00125},
            limits={"context_length": 200000}, performance={"avg_latency_ms": 400}
        )
        
        registry.add_model(small_model)
        registry.add_model(large_model)
        
        # Filter by minimum context length
        large_context_models = registry.filter_by_constraints(min_context_length=10000)
        small_context_models = registry.filter_by_constraints(min_context_length=2000)
        
        assert len(large_context_models) == 1
        assert large_context_models[0].model == "claude-3-haiku"
        assert len(small_context_models) == 2  # Both models meet 2000 requirement

    def test_should_serialize_entire_registry(self):
        """Test that entire registry can be serialized to dict/JSON."""
        from llm_router.registry import ProviderRegistry, ProviderModel
        
        registry = ProviderRegistry()
        
        model = ProviderModel(
            provider="openai", model="gpt-3.5-turbo",
            capabilities=["code"], pricing={"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
            limits={"context_length": 4096}, performance={"avg_latency_ms": 800}
        )
        
        registry.add_model(model)
        
        serialized = registry.to_dict()
        
        assert "models" in serialized
        assert len(serialized["models"]) == 1
        assert serialized["models"][0]["provider"] == "openai"


class TestProviderRegistryPersistence:
    """Test loading and saving provider registry data."""

    def test_should_load_registry_from_dict(self):
        """Test that registry can be created from dictionary data."""
        from llm_router.registry import ProviderRegistry
        
        data = {
            "models": [
                {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo", 
                    "capabilities": ["code", "qa"],
                    "pricing": {"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
                    "limits": {"context_length": 4096},
                    "performance": {"avg_latency_ms": 800}
                },
                {
                    "provider": "anthropic",
                    "model": "claude-3-haiku",
                    "capabilities": ["creative"],
                    "pricing": {"input_tokens_per_1k": 0.00025, "output_tokens_per_1k": 0.00125},
                    "limits": {"context_length": 200000},
                    "performance": {"avg_latency_ms": 400}
                }
            ]
        }
        
        registry = ProviderRegistry.from_dict(data)
        
        assert len(registry.get_all_models()) == 2
        assert "openai" in registry.get_providers()
        assert "anthropic" in registry.get_providers()

    def test_should_handle_invalid_registry_data(self):
        """Test that loading invalid data raises appropriate errors."""
        from llm_router.registry import ProviderRegistry
        
        # Missing required fields should raise error
        invalid_data = {
            "models": [
                {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo"
                    # Missing capabilities, pricing, etc.
                }
            ]
        }
        
        with pytest.raises(ValidationError):
            ProviderRegistry.from_dict(invalid_data)

    def test_should_roundtrip_serialize_deserialize(self):
        """Test that registry can be serialized and deserialized without loss."""
        from llm_router.registry import ProviderRegistry, ProviderModel
        
        # Create original registry
        original = ProviderRegistry()
        model = ProviderModel(
            provider="openai", model="gpt-3.5-turbo",
            capabilities=["code", "qa"], 
            pricing={"input_tokens_per_1k": 0.001, "output_tokens_per_1k": 0.002},
            limits={"context_length": 4096, "safety_level": "moderate"},
            performance={"avg_latency_ms": 800, "quality_scores": {"code": 0.85}}
        )
        original.add_model(model)
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = ProviderRegistry.from_dict(data)
        
        # Should be equivalent
        assert len(restored.get_all_models()) == len(original.get_all_models())
        restored_model = restored.get_model("openai", "gpt-3.5-turbo")
        assert restored_model is not None
        assert restored_model.capabilities == model.capabilities
        assert restored_model.pricing == model.pricing
