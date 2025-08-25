"""Shared test configuration and fixtures."""

import pytest
from llm_router import Router
from llm_router.config import _reset_config, get_config


@pytest.fixture
def router():
    """Create a Router instance for testing."""
    return Router()


@pytest.fixture
def config():
    """Create a fresh config instance for testing."""
    _reset_config()  # Clear any existing config
    return get_config()


@pytest.fixture(autouse=True)
def reset_config_after_test():
    """Automatically reset config after each test to avoid test pollution."""
    yield
    _reset_config()
