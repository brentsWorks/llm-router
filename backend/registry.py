"""Provider registry for managing LLM model capabilities,
    pricing, and performance data."""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, ValidationError
from .models import VALID_CATEGORIES
from .capabilities import (
    ModelCapabilities,
    CapabilityRequirement,
    CapabilityScore,
    TaskType,
)


class PricingInfo(BaseModel):
    """Pricing information for a model."""

    input_tokens_per_1k: float = Field(ge=0.0)
    output_tokens_per_1k: float = Field(ge=0.0)

    def __getitem__(self, key: str):
        """Allow dictionary-style access for backward compatibility."""
        return getattr(self, key)


class LimitsInfo(BaseModel):
    """Limits and constraints for a model."""

    context_length: int = Field(gt=0)
    rate_limit: Optional[int] = Field(default=None, gt=0)
    safety_level: Optional[str] = None

    def __getitem__(self, key: str):
        """Allow dictionary-style access for backward compatibility."""
        return getattr(self, key)


class PerformanceInfo(BaseModel):
    """Performance metrics for a model."""

    avg_latency_ms: float = Field(ge=0.0)
    quality_scores: Optional[Dict[str, float]] = None

    @field_validator("quality_scores")
    @classmethod
    def validate_quality_scores(
        cls, v: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        """Validate quality scores are between 0 and 1."""
        if v is None:
            return v

        for category, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Quality score for {category} must be between 0.0 and 1.0"
                )

        return v

    def __getitem__(self, key: str):
        """Allow dictionary-style access for backward compatibility."""
        return getattr(self, key)


class ProviderModel(BaseModel):
    """Complete model definition with capabilities, pricing, and performance data."""

    provider: str
    model: str
    capabilities: List[str]
    pricing: PricingInfo
    limits: LimitsInfo
    performance: PerformanceInfo
    detailed_capabilities: Optional[ModelCapabilities] = Field(
        default=None, description="Detailed capability model"
    )

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v: List[str]) -> List[str]:
        """Validate that all capabilities are valid categories."""
        for capability in v:
            if capability not in VALID_CATEGORIES:
                raise ValueError(
                    f"'{capability}' is not a valid capability. "
                    f"Valid capabilities: {VALID_CATEGORIES}"
                )
        return v

    def has_capability(self, capability: str) -> bool:
        """Check if model has a specific capability."""
        return capability in self.capabilities

    def get_quality_score(self, category: str) -> Optional[float]:
        """Get quality score for a specific category."""
        if self.performance.quality_scores is None:
            return None
        return self.performance.quality_scores.get(category)

    @field_validator("detailed_capabilities")
    @classmethod
    def validate_capability_sync(
        cls, v: Optional[ModelCapabilities], info
    ) -> Optional[ModelCapabilities]:
        """Validate that detailed capabilities are consistent with legacy
            capabilities."""
        if v is None:
            return v

        # Get legacy capabilities from info.data
        legacy_capabilities = info.data.get("capabilities", [])

        # Check that detailed capabilities don't contradict legacy ones
        for task_type in v.task_expertise:
            if v.task_expertise[task_type] > 0.0:  # Has expertise in this task
                task_name = task_type.value
                if task_name not in legacy_capabilities:
                    raise ValueError(
                        f"Capability mismatch: detailed capabilities show expertise in "
                        f"'{task_name}' but it's not in legacy capabilities list"
                    )

        return v

    def calculate_capability_score(
        self, requirement: CapabilityRequirement
    ) -> CapabilityScore:
        """Calculate capability score for this model against requirements."""
        from .capabilities import CapabilityMatcher

        matcher = CapabilityMatcher()

        if self.detailed_capabilities is None:
            # Fallback to legacy scoring if no detailed capabilities
            basic_capabilities = ModelCapabilities(
                task_expertise={TaskType(task): 0.8 for task in self.capabilities},
                context_length_max=self.limits.context_length,
                safety_level=self.limits.safety_level or "medium",
            )
            return matcher.calculate_score(basic_capabilities, requirement)

        return matcher.calculate_score(self.detailed_capabilities, requirement)


class ProviderRegistry:
    """Registry for managing provider models with filtering
        and querying capabilities."""

    def __init__(self):
        """Initialize empty registry."""
        self._models: Dict[str, ProviderModel] = {}

    def _get_model_key(self, provider: str, model: str) -> str:
        """Generate unique key for a model."""
        return f"{provider}/{model}"

    def add_model(self, model: ProviderModel) -> None:
        """Add a model to the registry.

        Args:
            model: ProviderModel instance to add

        Raises:
            ValueError: If model already exists in registry
        """
        key = self._get_model_key(model.provider, model.model)
        if key in self._models:
            raise ValueError(f"Model {key} already exists in registry")

        self._models[key] = model

    def get_model(self, provider: str, model: str) -> Optional[ProviderModel]:
        """Get a specific model by provider and model name.

        Args:
            provider: Provider name (e.g., "openai")
            model: Model name (e.g., "gpt-3.5-turbo")

        Returns:
            ProviderModel if found, None otherwise
        """
        key = self._get_model_key(provider, model)
        return self._models.get(key)

    def get_all_models(self) -> List[ProviderModel]:
        """Get all models in the registry.

        Returns:
            List of all ProviderModel instances
        """
        return list(self._models.values())

    def get_providers(self) -> List[str]:
        """Get list of all unique providers in the registry.

        Returns:
            List of unique provider names
        """
        providers = set()
        for model in self._models.values():
            providers.add(model.provider)
        return sorted(list(providers))

    def get_models_by_provider(self, provider: str) -> List[ProviderModel]:
        """Get all models from a specific provider.

        Args:
            provider: Provider name to filter by

        Returns:
            List of models from the specified provider
        """
        return [model for model in self._models.values() if model.provider == provider]

    def get_models_by_capability(self, capability: str) -> List[ProviderModel]:
        """Get all models that have a specific capability.

        Args:
            capability: Capability to filter by (e.g., "code", "creative")

        Returns:
            List of models with the specified capability
        """
        return [
            model for model in self._models.values() if model.has_capability(capability)
        ]

    def find_models_by_requirements(
        self, requirement, min_score: float = 0.0
    ) -> List[Tuple[ProviderModel, Any]]:
        """Find models that match capability requirements.

        Args:
            requirement: CapabilityRequirement instance
            min_score: Minimum overall score required

        Returns:
            List of tuples (model, score) sorted by score descending
        """
        from .capabilities import CapabilityMatcher

        matcher = CapabilityMatcher()
        matches = []

        for model in self._models.values():
            if model.detailed_capabilities is not None:
                score = matcher.calculate_score(
                    model.detailed_capabilities, requirement
                )
                if score.overall_score >= min_score:
                    matches.append((model, score))

        # Sort by overall score descending
        matches.sort(key=lambda x: x[1].overall_score, reverse=True)
        return matches

    def filter_by_constraints(
        self,
        min_context_length: Optional[int] = None,
        max_latency_ms: Optional[float] = None,
        safety_level: Optional[str] = None,
    ) -> List[ProviderModel]:
        """Filter models by various constraints.

        Args:
            min_context_length: Minimum context length required
            max_latency_ms: Maximum latency allowed
            safety_level: Required safety level

        Returns:
            List of models matching all specified constraints
        """
        filtered_models = []

        for model in self._models.values():
            # Check context length constraint
            if min_context_length is not None:
                if model.limits.context_length < min_context_length:
                    continue

            # Check latency constraint
            if max_latency_ms is not None:
                if model.performance.avg_latency_ms > max_latency_ms:
                    continue

            # Check safety level constraint
            if safety_level is not None:
                if model.limits.safety_level != safety_level:
                    continue

            filtered_models.append(model)

        return filtered_models

    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry to dictionary.

        Returns:
            Dictionary representation of the registry
        """
        return {"models": [model.model_dump() for model in self._models.values()]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderRegistry":
        """Create registry from dictionary data.

        Args:
            data: Dictionary containing models data

        Returns:
            ProviderRegistry instance populated with data

        Raises:
            ValidationError: If data is invalid
        """
        registry = cls()

        for model_data in data.get("models", []):
            try:
                # Convert nested dicts to proper model instances
                pricing_info = PricingInfo(**model_data["pricing"])
                limits_info = LimitsInfo(**model_data["limits"])
                performance_info = PerformanceInfo(**model_data["performance"])

                model = ProviderModel(
                    provider=model_data["provider"],
                    model=model_data["model"],
                    capabilities=model_data["capabilities"],
                    pricing=pricing_info,
                    limits=limits_info,
                    performance=performance_info,
                )

                registry.add_model(model)
            except KeyError as e:
                from pydantic_core import ValidationError as CoreValidationError

                raise CoreValidationError.from_exception_data(
                    "ValidationError",
                    [
                        {
                            "type": "missing",
                            "loc": ("models",),
                            "msg": f"Missing required field: {e}",
                            "input": model_data,
                        }
                    ],
                )
            except ValidationError:
                # Re-raise ValidationError as-is
                raise
            except Exception as e:
                from pydantic_core import ValidationError as CoreValidationError

                raise CoreValidationError.from_exception_data(
                    "ValidationError",
                    [
                        {
                            "type": "value_error",
                            "loc": ("models",),
                            "msg": f"Invalid data: {e}",
                            "input": model_data,
                        }
                    ],
                )

        return registry
