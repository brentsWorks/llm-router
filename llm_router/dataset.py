"""
Example dataset management for LLM Router.

This module provides functionality for loading, validating, and querying
example prompts with their ideal model classifications.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """Exception raised for dataset-related errors."""
    pass


class PromptCategory(Enum):
    """Categories of prompts for classification."""
    CREATIVE_WRITING = "creative_writing"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    REASONING = "reasoning"
    CONVERSATION = "conversation"


@dataclass
class ExamplePrompt:
    """Represents a single example prompt with classification metadata."""
    
    # Required fields
    text: str
    category: PromptCategory
    preferred_models: List[str]
    
    # Optional metadata
    description: Optional[str] = None
    difficulty: Optional[str] = None  # "easy", "medium", "hard"
    expected_length: Optional[str] = None  # "short", "medium", "long"
    domain: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate the example prompt after initialization."""
        if not self.text or not self.text.strip():
            raise DatasetError("Prompt text cannot be empty")
        
        if not self.preferred_models:
            raise DatasetError("At least one preferred model must be specified")
        
        # Convert string category to enum if needed
        if isinstance(self.category, str):
            try:
                self.category = PromptCategory(self.category)
            except ValueError:
                valid_categories = [cat.value for cat in PromptCategory]
                raise DatasetError(
                    f"Invalid category '{self.category}'. "
                    f"Valid categories: {valid_categories}"
                )
    
    def get_embedding_text(self) -> str:
        """Get the text that should be used for embedding generation."""
        return self.text.strip()
    
    def matches_category(self, category: PromptCategory) -> bool:
        """Check if this example matches the given category."""
        return self.category == category
    
    def matches_model(self, model_name: str) -> bool:
        """Check if this example prefers the given model."""
        return model_name in self.preferred_models
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "category": self.category.value,
            "preferred_models": self.preferred_models,
            "description": self.description,
            "difficulty": self.difficulty,
            "expected_length": self.expected_length,
            "domain": self.domain,
            "tags": self.tags
        }


class ExampleDataset:
    """Manages a collection of example prompts for classification."""
    
    def __init__(self, examples: Optional[List[ExamplePrompt]] = None):
        """Initialize dataset with optional examples.
        
        Args:
            examples: List of example prompts
        """
        self._examples: List[ExamplePrompt] = examples or []
        self._validate_dataset()
    
    def _validate_dataset(self) -> None:
        """Validate the dataset structure and content."""
        if not isinstance(self._examples, list):
            raise DatasetError("Examples must be a list")
        
        # Check for duplicate texts (potential data quality issue)
        texts = [ex.text for ex in self._examples]
        if len(texts) != len(set(texts)):
            logger.warning("Dataset contains duplicate prompt texts")
        
        logger.info(f"Dataset validated: {len(self._examples)} examples")
    
    @classmethod
    def from_json_file(cls, file_path: Path) -> 'ExampleDataset':
        """Load dataset from JSON file.
        
        Args:
            file_path: Path to JSON file containing examples
            
        Returns:
            ExampleDataset instance
            
        Raises:
            DatasetError: If file cannot be loaded or is invalid
        """
        try:
            if not file_path.exists():
                raise DatasetError(f"Dataset file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return cls.from_dict(data)
            
        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON in dataset file: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {e}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExampleDataset':
        """Create dataset from dictionary data.
        
        Args:
            data: Dictionary containing dataset structure
            
        Returns:
            ExampleDataset instance
        """
        if not isinstance(data, dict):
            raise DatasetError("Dataset data must be a dictionary")
        
        if "examples" not in data:
            raise DatasetError("Dataset must contain 'examples' key")
        
        examples_data = data["examples"]
        if not isinstance(examples_data, list):
            raise DatasetError("Examples must be a list")
        
        examples = []
        for i, example_data in enumerate(examples_data):
            try:
                example = ExamplePrompt(**example_data)
                examples.append(example)
            except (TypeError, ValueError) as e:
                raise DatasetError(f"Invalid example at index {i}: {e}")
        
        return cls(examples)
    
    def add_example(self, example: ExamplePrompt) -> None:
        """Add a new example to the dataset.
        
        Args:
            example: Example prompt to add
        """
        if not isinstance(example, ExamplePrompt):
            raise DatasetError("Example must be an ExamplePrompt instance")
        
        self._examples.append(example)
        logger.debug(f"Added example: {example.text[:50]}...")
    
    def get_all_examples(self) -> List[ExamplePrompt]:
        """Get all examples in the dataset.
        
        Returns:
            List of all example prompts
        """
        return self._examples.copy()
    
    def query_by_category(self, category: PromptCategory) -> List[ExamplePrompt]:
        """Query examples by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of examples matching the category
        """
        return [ex for ex in self._examples if ex.matches_category(category)]
    
    def query_by_model(self, model_name: str) -> List[ExamplePrompt]:
        """Query examples that prefer a specific model.
        
        Args:
            model_name: Name of the model to filter by
            
        Returns:
            List of examples that prefer this model
        """
        return [ex for ex in self._examples if ex.matches_model(model_name)]
    
    def get_categories(self) -> Set[PromptCategory]:
        """Get all categories present in the dataset.
        
        Returns:
            Set of categories found in examples
        """
        return {ex.category for ex in self._examples}
    
    def get_preferred_models(self) -> Set[str]:
        """Get all preferred models mentioned in the dataset.
        
        Returns:
            Set of model names found in examples
        """
        models = set()
        for ex in self._examples:
            models.update(ex.preferred_models)
        return models
    
    def get_embedding_texts(self) -> List[str]:
        """Get all example texts for embedding generation.
        
        Returns:
            List of texts suitable for embedding
        """
        return [ex.get_embedding_text() for ex in self._examples]
    
    def __len__(self) -> int:
        """Get number of examples in dataset."""
        return len(self._examples)
    
    def __iter__(self):
        """Iterate over examples."""
        return iter(self._examples)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary for serialization.
        
        Returns:
            Dictionary representation of the dataset
        """
        return {
            "examples": [ex.to_dict() for ex in self._examples],
            "metadata": {
                "total_examples": len(self._examples),
                "categories": [cat.value for cat in self.get_categories()],
                "preferred_models": list(self.get_preferred_models())
            }
        }
