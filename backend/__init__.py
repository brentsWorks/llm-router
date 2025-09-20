"""LLM Router - Hybrid LLM routing with semantic analysis and fallback."""

__version__ = "0.1.0"

from .core import Router
from .config import create_configured_registry

__all__ = ["Router", "create_configured_registry", "__version__"]
