"""
Model Configuration Loader
==========================

This module handles loading LLM model configurations from JSON files and creating
configured ProviderRegistry instances. Keeps model configuration separate from
application logic.
"""

import json
from pathlib import Path
from typing import List, Optional
from ..registry import ProviderRegistry, ProviderModel, PricingInfo, LimitsInfo, PerformanceInfo


def load_models_from_config(config_path: Optional[str] = None) -> List[ProviderModel]:
    """
    Load model configurations from JSON file.
    
    Args:
        config_path: Path to the models configuration file. 
                    Defaults to models.json in the same directory.
    
    Returns:
        List of ProviderModel instances loaded from configuration.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config file has invalid format.
    """
    if config_path is None:
        # Default to models.json in the same directory as this file
        config_path = Path(__file__).parent / "models.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    
    if 'models' not in config:
        raise ValueError("Config file must contain a 'models' key")
    
    models = []
    for i, model_config in enumerate(config['models']):
        try:
            # Create Pydantic models from config data
            pricing = PricingInfo(**model_config['pricing'])
            limits = LimitsInfo(**model_config['limits'])  
            performance = PerformanceInfo(**model_config['performance'])
            
            model = ProviderModel(
                provider=model_config['provider'],
                model=model_config['model'],
                capabilities=model_config['capabilities'],
                pricing=pricing,
                limits=limits,
                performance=performance
            )
            models.append(model)
            
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid model configuration at index {i}: {e}")
    
    return models


def create_configured_registry(config_path: Optional[str] = None) -> ProviderRegistry:
    """
    Create a ProviderRegistry pre-populated with models from configuration.
    
    Args:
        config_path: Path to the models configuration file.
                    Defaults to models.json in the same directory.
    
    Returns:
        ProviderRegistry instance with models loaded from config.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config file has invalid format.
    """
    registry = ProviderRegistry()
    models = load_models_from_config(config_path)
    
    # Add all models to the registry
    for model in models:
        registry.add_model(model)
    
    return registry


def get_default_models() -> List[ProviderModel]:
    """
    Get the default model configuration (for testing/fallback).
    
    Returns:
        List of default ProviderModel instances.
    """
    return load_models_from_config()
