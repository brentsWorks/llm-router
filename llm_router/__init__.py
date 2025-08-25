"""LLM Router - Hybrid LLM routing with semantic analysis and fallback."""

__version__ = "0.1.0"

from llm_router.core import Router
from llm_router.config import get_config

__all__ = ["Router", "get_config", "__version__"]
