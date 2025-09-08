"""
Configuration module for LLM Router.

This module handles loading and managing configuration for models, services, and other
application settings.
"""

from .model_loader import load_models_from_config, create_configured_registry

__all__ = ["load_models_from_config", "create_configured_registry"]
