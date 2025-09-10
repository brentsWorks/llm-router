"""
Unit tests for example dataset functionality.

This module tests dataset loading, validation, and querying for the
LLM Router's example-based classification system.
"""

import pytest
import json
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock, mock_open

from llm_router.dataset import ExampleDataset, DatasetError, ExamplePrompt, PromptCategory


def mock_open_json(data):
    """Helper to mock opening JSON files."""
    return mock_open(read_data=json.dumps(data))


class TestExampleDataset:
    """Test example dataset loading and validation."""
    
    def test_should_create_dataset_with_valid_data(self):
        """Test creating dataset with valid example data."""
        example = ExamplePrompt(
            text="Write a creative story about a robot",
            category=PromptCategory.CREATIVE,
            preferred_models=["gpt-4", "claude-3"]
        )
        
        dataset = ExampleDataset([example])
        
        assert len(dataset) == 1
        assert list(dataset)[0] == example
    
    def test_should_load_dataset_from_json_file(self):
        """Test loading dataset from JSON file."""
        # Create temporary JSON data
        json_data = {
            "examples": [
                {
                    "text": "Explain quantum physics",
                    "category": "qa",
                    "preferred_models": ["gpt-4"]
                }
            ]
        }
        
        # Mock file operations
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        
        with patch('builtins.open', mock_open_json(json_data)):
            dataset = ExampleDataset.from_json_file(mock_path)
            
        assert len(dataset) == 1
        example = list(dataset)[0]
        assert example.text == "Explain quantum physics"
        assert example.category == PromptCategory.QA
    
    def test_should_validate_dataset_structure(self):
        """Test dataset validation for required fields."""
        valid_example = ExamplePrompt(
            text="Generate Python code",
            category=PromptCategory.CODE,
            preferred_models=["codex", "gpt-4"]
        )
        
        # Should not raise any errors
        dataset = ExampleDataset([valid_example])
        assert len(dataset) == 1
    
    def test_should_reject_invalid_dataset_structure(self):
        """Test rejection of malformed dataset."""
        # Test with non-list examples
        with pytest.raises(DatasetError) as exc_info:
            ExampleDataset("not a list")
        assert "must be a list" in str(exc_info.value)
        
        # Test invalid dictionary structure
        invalid_data = {"invalid": "structure"}
        with pytest.raises(DatasetError) as exc_info:
            ExampleDataset.from_dict(invalid_data)
        assert "examples" in str(exc_info.value)
    
    def test_should_handle_missing_dataset_file(self):
        """Test handling of missing dataset file."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        
        with pytest.raises(DatasetError) as exc_info:
            ExampleDataset.from_json_file(mock_path)
        assert "not found" in str(exc_info.value)


class TestExamplePrompt:
    """Test individual example prompt structure."""
    
    def test_should_create_valid_example_prompt(self):
        """Test creating a valid example prompt."""
        prompt = ExamplePrompt(
            text="Translate this text to French",
            category=PromptCategory.TRANSLATION,
            preferred_models=["gpt-4", "claude-3"]
        )
        
        assert prompt.text == "Translate this text to French"
        assert prompt.category == PromptCategory.TRANSLATION
        assert prompt.preferred_models == ["gpt-4", "claude-3"]
        assert prompt.description is None
    
    def test_should_validate_required_fields(self):
        """Test validation of required prompt fields."""
        # Test empty text
        with pytest.raises(DatasetError) as exc_info:
            ExamplePrompt(
                text="",
                category=PromptCategory.CREATIVE,
                preferred_models=["gpt-4"]
            )
        assert "empty" in str(exc_info.value)
        
        # Test empty preferred models
        with pytest.raises(DatasetError) as exc_info:
            ExamplePrompt(
                text="Valid text",
                category=PromptCategory.CREATIVE,
                preferred_models=[]
            )
        assert "preferred model" in str(exc_info.value)
    
    def test_should_handle_optional_metadata(self):
        """Test handling of optional metadata fields."""
        prompt = ExamplePrompt(
            text="Analyze this data",
            category=PromptCategory.ANALYSIS,
            preferred_models=["gpt-4"],
            description="Data analysis example",
            difficulty="medium",
            expected_length="long",
            domain="finance",
            tags=["analysis", "charts"]
        )
        
        assert prompt.description == "Data analysis example"
        assert prompt.difficulty == "medium"
        assert prompt.expected_length == "long"
        assert prompt.domain == "finance"
        assert prompt.tags == ["analysis", "charts"]


class TestDatasetQuerying:
    """Test dataset querying functionality."""
    
    def test_should_query_by_category(self):
        """Test querying examples by category."""
        examples = [
            ExamplePrompt("Write a story", PromptCategory.CREATIVE, ["gpt-4"]),
            ExamplePrompt("Generate code", PromptCategory.CODE, ["codex"]),
            ExamplePrompt("Write a poem", PromptCategory.CREATIVE, ["claude-3"])
        ]
        dataset = ExampleDataset(examples)
        
        creative_examples = dataset.query_by_category(PromptCategory.CREATIVE)
        
        assert len(creative_examples) == 2
        assert all(ex.category == PromptCategory.CREATIVE for ex in creative_examples)
    
    def test_should_query_by_model_preference(self):
        """Test querying examples by preferred model."""
        examples = [
            ExamplePrompt("Write a story", PromptCategory.CREATIVE, ["gpt-4", "claude-3"]),
            ExamplePrompt("Generate code", PromptCategory.CODE, ["codex"]),
            ExamplePrompt("Analyze data", PromptCategory.ANALYSIS, ["gpt-4"])
        ]
        dataset = ExampleDataset(examples)
        
        gpt4_examples = dataset.query_by_model("gpt-4")
        
        assert len(gpt4_examples) == 2
        assert all("gpt-4" in ex.preferred_models for ex in gpt4_examples)
    
    def test_should_get_all_examples(self):
        """Test retrieving all examples."""
        examples = [
            ExamplePrompt("Example 1", PromptCategory.CREATIVE, ["gpt-4"]),
            ExamplePrompt("Example 2", PromptCategory.CODE, ["codex"])
        ]
        dataset = ExampleDataset(examples)
        
        all_examples = dataset.get_all_examples()
        
        assert len(all_examples) == 2
        assert all_examples == examples
    
    def test_should_handle_empty_query_results(self):
        """Test handling when query returns no results."""
        examples = [
            ExamplePrompt("Example 1", PromptCategory.CREATIVE, ["gpt-4"])
        ]
        dataset = ExampleDataset(examples)
        
        # Query for non-existent category
        code_examples = dataset.query_by_category(PromptCategory.CODE)
        assert len(code_examples) == 0
        
        # Query for non-existent model
        model_examples = dataset.query_by_model("non-existent-model")
        assert len(model_examples) == 0
    
    def test_should_support_text_embedding_access(self):
        """Test that examples can provide text for embedding."""
        examples = [
            ExamplePrompt("  Text with spaces  ", PromptCategory.CREATIVE, ["gpt-4"]),
            ExamplePrompt("Another example", PromptCategory.CODE, ["codex"])
        ]
        dataset = ExampleDataset(examples)
        
        embedding_texts = dataset.get_embedding_texts()
        
        assert len(embedding_texts) == 2
        assert embedding_texts[0] == "Text with spaces"  # Stripped
        assert embedding_texts[1] == "Another example"


