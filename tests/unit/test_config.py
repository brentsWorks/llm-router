"""Tests for application configuration system."""

import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError
from llm_router.utils import format_validation_error


class TestAppConfig:
    """Test application configuration loading and validation."""

    def test_should_create_config_with_default_values(self):
        """Test that config can be created with sensible defaults."""
        from llm_router.app_config import AppConfig
        
        config = AppConfig()
        
        # Should have sensible defaults
        assert config.app_name == "llm-router"
        assert config.version == "0.1.0"
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.max_routing_time_ms == 5000
        assert config.confidence_threshold == 0.8

    def test_should_load_config_from_environment_variables(self):
        """Test that config loads values from environment variables."""
        from llm_router.app_config import AppConfig
        
        env_vars = {
            "LLM_ROUTER_DEBUG": "true",
            "LLM_ROUTER_LOG_LEVEL": "DEBUG", 
            "LLM_ROUTER_MAX_ROUTING_TIME_MS": "3000",
            "LLM_ROUTER_CONFIDENCE_THRESHOLD": "0.9"
        }
        
        with patch.dict(os.environ, env_vars):
            config = AppConfig()
            
            assert config.debug is True
            assert config.log_level == "DEBUG"
            assert config.max_routing_time_ms == 3000
            assert config.confidence_threshold == 0.9

    def test_should_validate_log_level_values(self):
        """Test that only valid log levels are accepted."""
        from llm_router.app_config import AppConfig
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        # Valid levels should work
        for level in valid_levels:
            with patch.dict(os.environ, {"LLM_ROUTER_LOG_LEVEL": level}):
                config = AppConfig()
                assert config.log_level == level
        
        # Invalid level should raise error
        with patch.dict(os.environ, {"LLM_ROUTER_LOG_LEVEL": "INVALID"}):
            with pytest.raises(ValidationError) as exc_info:
                AppConfig()
            # Use our utility for clear error messages
            error_message = format_validation_error(exc_info.value)
            assert "log_level" in error_message.lower()

    def test_should_validate_confidence_threshold_bounds(self):
        """Test that confidence threshold is validated to be between 0 and 1."""
        from llm_router.app_config import AppConfig
        
        # Valid threshold should work
        with patch.dict(os.environ, {"LLM_ROUTER_CONFIDENCE_THRESHOLD": "0.75"}):
            config = AppConfig()
            assert config.confidence_threshold == 0.75
        
        # Invalid thresholds should raise error
        invalid_thresholds = ["-0.1", "1.5", "2.0"]
        
        for threshold in invalid_thresholds:
            with patch.dict(os.environ, {"LLM_ROUTER_CONFIDENCE_THRESHOLD": threshold}):
                with pytest.raises(ValidationError) as exc_info:
                    AppConfig()
                # Use our utility for clear error messages
                error_message = format_validation_error(exc_info.value)
                assert "confidence_threshold" in error_message.lower()

    def test_should_validate_max_routing_time_positive(self):
        """Test that max routing time must be positive."""
        from llm_router.app_config import AppConfig
        
        # Valid time should work
        with patch.dict(os.environ, {"LLM_ROUTER_MAX_ROUTING_TIME_MS": "2000"}):
            config = AppConfig()
            assert config.max_routing_time_ms == 2000
        
        # Invalid times should raise error
        invalid_times = ["-1", "0", "-1000"]
        
        for time_ms in invalid_times:
            with patch.dict(os.environ, {"LLM_ROUTER_MAX_ROUTING_TIME_MS": time_ms}):
                with pytest.raises(ValidationError) as exc_info:
                    AppConfig()
                # Use our utility for clear error messages
                error_message = format_validation_error(exc_info.value)
                assert "max_routing_time_ms" in error_message.lower()

    def test_should_serialize_config_to_dict(self):
        """Test that configuration can be serialized for logging/debugging."""
        from llm_router.app_config import AppConfig
        
        config = AppConfig()
        config_dict = config.model_dump()
        
        # Should contain all expected fields
        expected_fields = [
            "app_name", "version", "debug", "log_level", 
            "max_routing_time_ms", "confidence_threshold"
        ]
        
        for field in expected_fields:
            assert field in config_dict
        
        # Values should match
        assert config_dict["app_name"] == config.app_name
        assert config_dict["debug"] == config.debug
        assert config_dict["log_level"] == config.log_level


class TestMLConfig:
    """Test ML-specific configuration settings."""

    def test_should_create_ml_config_with_defaults(self):
        """Test that ML config has sensible defaults."""
        from llm_router.app_config import MLConfig
        
        config = MLConfig()
        
        # Should have ML-specific defaults
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.vector_store_type == "chromadb"
        assert config.max_embedding_batch_size == 32
        assert config.similarity_threshold == 0.7

    def test_should_load_ml_config_from_environment(self):
        """Test that ML config loads from environment variables."""
        from llm_router.app_config import MLConfig
        
        env_vars = {
            "LLM_ROUTER_EMBEDDING_MODEL": "custom-embedding-model",
            "LLM_ROUTER_VECTOR_STORE_TYPE": "pinecone",
            "LLM_ROUTER_MAX_EMBEDDING_BATCH_SIZE": "64",
            "LLM_ROUTER_SIMILARITY_THRESHOLD": "0.8"
        }
        
        with patch.dict(os.environ, env_vars):
            config = MLConfig()
            
            assert config.embedding_model == "custom-embedding-model"
            assert config.vector_store_type == "pinecone"
            assert config.max_embedding_batch_size == 64
            assert config.similarity_threshold == 0.8

    def test_should_validate_vector_store_type(self):
        """Test that only supported vector store types are allowed."""
        from llm_router.app_config import MLConfig
        
        valid_stores = ["chromadb", "pinecone", "memory"]
        
        # Valid stores should work
        for store in valid_stores:
            with patch.dict(os.environ, {"LLM_ROUTER_VECTOR_STORE_TYPE": store}):
                config = MLConfig()
                assert config.vector_store_type == store
        
        # Invalid store should raise error
        with patch.dict(os.environ, {"LLM_ROUTER_VECTOR_STORE_TYPE": "invalid_store"}):
            with pytest.raises(ValidationError) as exc_info:
                MLConfig()
            # Use our utility for clear error messages
            error_message = format_validation_error(exc_info.value)
            assert "vector_store_type" in error_message.lower()

    def test_should_validate_similarity_threshold_bounds(self):
        """Test that similarity threshold is between 0 and 1."""
        from llm_router.app_config import MLConfig
        
        # Valid threshold should work
        with patch.dict(os.environ, {"LLM_ROUTER_SIMILARITY_THRESHOLD": "0.6"}):
            config = MLConfig()
            assert config.similarity_threshold == 0.6
        
        # Invalid thresholds should raise error
        invalid_thresholds = ["-0.1", "1.1", "2.0"]
        
        for threshold in invalid_thresholds:
            with patch.dict(os.environ, {"LLM_ROUTER_SIMILARITY_THRESHOLD": threshold}):
                with pytest.raises(ValidationError) as exc_info:
                    MLConfig()
                # Use our utility for clear error messages
                error_message = format_validation_error(exc_info.value)
                assert "similarity_threshold" in error_message.lower()


class TestConfigFactory:
    """Test configuration factory and global config access."""

    def test_should_create_full_config_with_all_sections(self):
        """Test that factory creates complete configuration."""
        from llm_router.app_config import create_config
        
        config = create_config()
        
        # Should have app config
        assert hasattr(config, 'app')
        assert config.app.app_name == "llm-router"
        
        # Should have ML config
        assert hasattr(config, 'ml')
        assert config.ml.embedding_model is not None

    def test_should_provide_singleton_config_access(self):
        """Test that get_config returns same instance."""
        from llm_router.app_config import get_config
        
        config1 = get_config()
        config2 = get_config()
        
        # Should be the same instance (singleton pattern)
        assert config1 is config2

    def test_should_allow_config_override_for_testing(self):
        """Test that config can be overridden for testing purposes."""
        from llm_router.app_config import get_config, _reset_config
        
        # Get initial config
        config1 = get_config()
        original_debug = config1.app.debug
        
        # Reset and create new config with different env
        _reset_config()
        
        with patch.dict(os.environ, {"LLM_ROUTER_DEBUG": "true"}):
            config2 = get_config()
            
            # Should have different debug setting
            assert config2.app.debug != original_debug
            assert config2.app.debug is True

    def test_should_serialize_full_config_for_debugging(self):
        """Test that entire config can be serialized safely."""
        from llm_router.app_config import get_config
        
        config = get_config()
        config_dict = config.model_dump()
        
        # Should contain all sections
        assert "app" in config_dict
        assert "ml" in config_dict
        
        # Should be JSON-serializable (no complex objects)
        import json
        json_str = json.dumps(config_dict)  # Should not raise exception
        assert len(json_str) > 0


class TestConfigValidation:
    """Test comprehensive config validation scenarios."""

    def test_should_handle_missing_optional_env_vars(self):
        """Test that config works when optional env vars are missing."""
        from llm_router.app_config import AppConfig
        
        # Clear all LLM_ROUTER_ env vars
        env_backup = {}
        for key in os.environ:
            if key.startswith("LLM_ROUTER_"):
                env_backup[key] = os.environ[key]
        
        # Remove LLM_ROUTER_ env vars
        for key in env_backup:
            del os.environ[key]
        
        try:
            # Should still work with defaults
            config = AppConfig()
            assert config.app_name == "llm-router"
            assert config.debug is False
        finally:
            # Restore env vars
            os.environ.update(env_backup)

    def test_should_validate_type_conversions(self):
        """Test that string env vars are properly converted to correct types."""
        from llm_router.app_config import AppConfig
        
        env_vars = {
            "LLM_ROUTER_DEBUG": "false",  # String to bool
            "LLM_ROUTER_MAX_ROUTING_TIME_MS": "1500",  # String to int
            "LLM_ROUTER_CONFIDENCE_THRESHOLD": "0.85"  # String to float
        }
        
        with patch.dict(os.environ, env_vars):
            config = AppConfig()
            
            assert isinstance(config.debug, bool)
            assert config.debug is False
            assert isinstance(config.max_routing_time_ms, int)
            assert config.max_routing_time_ms == 1500
            assert isinstance(config.confidence_threshold, float)
            assert config.confidence_threshold == 0.85
