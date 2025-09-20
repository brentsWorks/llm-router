"""
Dataset loader utilities for LLM Router.

This module provides convenience functions for loading and managing
example datasets.
"""

import logging
from pathlib import Path
from typing import Optional

from .dataset import ExampleDataset

logger = logging.getLogger(__name__)

# Default dataset path
DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "examples.json"


def load_default_dataset() -> ExampleDataset:
    """Load the default example dataset.
    
    Returns:
        ExampleDataset instance with default examples
        
    Raises:
        DatasetError: If default dataset cannot be loaded
    """
    return ExampleDataset.from_json_file(DEFAULT_DATASET_PATH)


def load_dataset(dataset_path: Optional[Path] = None) -> ExampleDataset:
    """Load dataset from specified path or use default.
    
    Args:
        dataset_path: Path to dataset JSON file. If None, uses default.
        
    Returns:
        ExampleDataset instance
        
    Raises:
        DatasetError: If dataset cannot be loaded
    """
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_PATH
    
    logger.info(f"Loading dataset from: {dataset_path}")
    return ExampleDataset.from_json_file(dataset_path)


def get_dataset_info(dataset: ExampleDataset) -> dict:
    """Get summary information about a dataset.
    
    Args:
        dataset: ExampleDataset to analyze
        
    Returns:
        Dictionary with dataset statistics
    """
    categories = dataset.get_categories()
    models = dataset.get_preferred_models()
    
    # Count examples by category
    category_counts = {}
    for category in categories:
        category_counts[category.value] = len(dataset.query_by_category(category))
    
    # Count examples by model
    model_counts = {}
    for model in models:
        model_counts[model] = len(dataset.query_by_model(model))
    
    return {
        "total_examples": len(dataset),
        "categories": {
            "count": len(categories),
            "types": [cat.value for cat in categories],
            "distribution": category_counts
        },
        "models": {
            "count": len(models),
            "types": list(models),
            "distribution": model_counts
        }
    }
