"""Configuration system for LLM Router application."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Valid log levels for the application."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    CHROMADB = "chromadb"
    PINECONE = "pinecone" 
    MEMORY = "memory"


class AppConfig(BaseSettings):
    """Application configuration with environment variable loading."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_ROUTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application settings
    app_name: str = "llm-router"
    version: str = "0.1.0"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO

    # Performance settings
    max_routing_time_ms: int = Field(default=5000, gt=0)
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class MLConfig(BaseSettings):
    """ML-specific configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_ROUTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_embedding_batch_size: int = Field(default=32, gt=0)

    # Vector store settings 
    vector_store_type: VectorStoreType = VectorStoreType.PINECONE
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Pinecone settings
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "llm-router"


class Config(BaseModel):
    """Complete application configuration combining all settings."""

    app: AppConfig
    ml: MLConfig


# Global configuration instance
_config: Optional[Config] = None


def create_config() -> Config:
    """Create a new configuration instance."""
    return Config(
        app=AppConfig(),
        ml=MLConfig()
    )


def get_config() -> Config:
    """Get the global configuration instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = create_config()
    return _config


def _reset_config() -> None:
    """Reset the global configuration instance (for testing)."""
    global _config
    _config = None
