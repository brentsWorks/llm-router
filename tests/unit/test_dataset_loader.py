"""
Unit tests for dataset loader functionality.

This module tests dataset loading utilities and convenience functions.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from llm_router.dataset_loader import (
    load_default_dataset, 
    load_dataset, 
    get_dataset_info,
    DEFAULT_DATASET_PATH
)
from llm_router.dataset import ExampleDataset, ExamplePrompt, PromptCategory


class TestDatasetLoader:
    """Test dataset loader functions."""
    
    def test_should_load_default_dataset(self):
        """Test loading the default dataset."""
        # Mock the dataset loading to avoid file system dependency
        mock_dataset = ExampleDataset([
            ExamplePrompt("Test prompt", PromptCategory.CREATIVE, ["gpt-4"])
        ])
        
        with patch.object(ExampleDataset, 'from_json_file', return_value=mock_dataset) as mock_load:
            dataset = load_default_dataset()
            
            assert dataset == mock_dataset
            mock_load.assert_called_once_with(DEFAULT_DATASET_PATH)
    
    def test_should_load_dataset_from_custom_path(self):
        """Test loading dataset from custom path."""
        custom_path = Path("/custom/dataset.json")
        mock_dataset = ExampleDataset([
            ExamplePrompt("Custom prompt", PromptCategory.CODE, ["codex"])
        ])
        
        with patch.object(ExampleDataset, 'from_json_file', return_value=mock_dataset) as mock_load:
            dataset = load_dataset(custom_path)
            
            assert dataset == mock_dataset
            mock_load.assert_called_once_with(custom_path)
    
    def test_should_use_default_path_when_none_provided(self):
        """Test that None path defaults to default dataset."""
        mock_dataset = ExampleDataset([])
        
        with patch.object(ExampleDataset, 'from_json_file', return_value=mock_dataset) as mock_load:
            dataset = load_dataset(None)
            
            mock_load.assert_called_once_with(DEFAULT_DATASET_PATH)
    
    def test_should_get_dataset_info(self):
        """Test getting dataset information and statistics."""
        examples = [
            ExamplePrompt("Creative prompt", PromptCategory.CREATIVE, ["gpt-4", "claude-3"]),
            ExamplePrompt("Code prompt", PromptCategory.CODE, ["codex"]),
            ExamplePrompt("Another creative", PromptCategory.CREATIVE, ["gpt-4"])
        ]
        dataset = ExampleDataset(examples)
        
        info = get_dataset_info(dataset)
        
        assert info["total_examples"] == 3
        assert info["categories"]["count"] == 2
        assert "creative" in info["categories"]["types"]
        assert "code" in info["categories"]["types"]
        assert info["categories"]["distribution"]["creative"] == 2
        assert info["categories"]["distribution"]["code"] == 1
        
        assert info["models"]["count"] == 3
        assert "gpt-4" in info["models"]["types"]
        assert "claude-3" in info["models"]["types"]
        assert "codex" in info["models"]["types"]
        assert info["models"]["distribution"]["gpt-4"] == 2
        assert info["models"]["distribution"]["claude-3"] == 1
        assert info["models"]["distribution"]["codex"] == 1
    
    def test_should_handle_empty_dataset_info(self):
        """Test getting info from empty dataset."""
        dataset = ExampleDataset([])
        
        info = get_dataset_info(dataset)
        
        assert info["total_examples"] == 0
        assert info["categories"]["count"] == 0
        assert info["categories"]["types"] == []
        assert info["categories"]["distribution"] == {}
        assert info["models"]["count"] == 0
        assert info["models"]["types"] == []
        assert info["models"]["distribution"] == {}


class TestDatasetLoaderIntegration:
    """Integration tests for dataset loader with actual file system."""
    
    def test_default_dataset_path_exists(self):
        """Test that the default dataset file exists."""
        assert DEFAULT_DATASET_PATH.exists(), f"Default dataset not found at {DEFAULT_DATASET_PATH}"
    
    def test_can_load_actual_default_dataset(self):
        """Test loading the actual default dataset file."""
        dataset = load_default_dataset()
        
        assert len(dataset) > 0, "Default dataset should not be empty"
        
        # Verify dataset has expected categories
        categories = dataset.get_categories()
        assert len(categories) > 0, "Dataset should have categories"
        
        # Verify dataset has preferred models
        models = dataset.get_preferred_models()
        assert len(models) > 0, "Dataset should have preferred models"
    
    def test_dataset_info_with_actual_data(self):
        """Test getting info from actual default dataset."""
        dataset = load_default_dataset()
        info = get_dataset_info(dataset)
        
        assert info["total_examples"] > 0
        assert info["categories"]["count"] > 0
        assert info["models"]["count"] > 0
        assert len(info["categories"]["distribution"]) > 0
        assert len(info["models"]["distribution"]) > 0
