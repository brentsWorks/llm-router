"""
Classifier factory for creating different types of classifiers.

This module provides factory functions for creating classifiers that can be used
by the RouterService, including hybrid classifiers with RAG integration.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

from .classification import KeywordClassifier
from .hybrid_classification import HybridClassifier, create_hybrid_classifier
from .rag_classification import RAGClassifier, create_rag_classifier
from .vector_service import create_vector_service

logger = logging.getLogger(__name__)


def load_env():
    """Load environment variables from .env file if it exists."""
    # Look for .env file in the project root (two levels up from this file)
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    # Only set if not already in environment (don't override existing)
                    if key.strip() not in os.environ:
                        os.environ[key.strip()] = value
        logger.info("Loaded environment variables from .env file")
    else:
        logger.debug(f".env file not found at {env_file}")


# Load environment variables when module is imported
load_env()


class ClassifierType:
    """Supported classifier types."""
    KEYWORD = "keyword"
    RAG = "rag" 
    HYBRID = "hybrid"


def create_keyword_classifier() -> KeywordClassifier:
    """Create a keyword-based classifier.
    
    Returns:
        Configured KeywordClassifier instance
    """
    return KeywordClassifier()


def create_rag_classifier_from_env(
    confidence_threshold: float = 0.7,
    max_similar_examples: int = 3
) -> Optional[RAGClassifier]:
    """Create a RAG classifier using environment variables.
    
    Args:
        confidence_threshold: Minimum confidence for RAG classification
        max_similar_examples: Maximum number of similar examples to use
        
    Returns:
        Configured RAGClassifier instance or None if environment not set up
    """
    # Check for required environment variables
    pinecone_key = os.getenv("PINECONE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    
    if not pinecone_key:
        logger.warning("PINECONE_API_KEY not found, cannot create RAG classifier")
        return None
    
    if not gemini_key:
        logger.warning("GEMINI_API_KEY not found, cannot create RAG classifier")
        return None
    
    try:
        # Create vector service
        vector_service = create_vector_service(
            api_key=pinecone_key,
            environment=pinecone_env,
            index_name="llm-router"
        )
        
        # Create RAG classifier
        rag_classifier = create_rag_classifier(
            vector_service=vector_service,
            api_key=gemini_key,
            confidence_threshold=confidence_threshold,
            max_similar_examples=max_similar_examples
        )
        
        logger.info("RAG classifier created successfully")
        return rag_classifier
        
    except Exception as e:
        logger.error(f"Failed to create RAG classifier: {e}")
        return None


def create_hybrid_classifier_from_env(
    rag_threshold: float = 0.7,
    rule_threshold: float = 0.5
) -> Union[HybridClassifier, KeywordClassifier]:
    """Create a hybrid classifier using environment variables.
    
    Falls back to keyword classifier if RAG components are not available.
    
    Args:
        rag_threshold: Minimum confidence for using RAG classification
        rule_threshold: Minimum confidence for using rule-based classification
        
    Returns:
        HybridClassifier if RAG is available, otherwise KeywordClassifier
    """
    # Try to create RAG classifier
    rag_classifier = create_rag_classifier_from_env()
    
    if rag_classifier is None:
        logger.warning("RAG classifier unavailable, falling back to keyword classifier")
        return create_keyword_classifier()
    
    # Create rule-based classifier
    rule_classifier = create_keyword_classifier()
    
    # Create hybrid classifier
    try:
        hybrid_classifier = create_hybrid_classifier(
            rag_classifier=rag_classifier,
            rule_classifier=rule_classifier,
            rag_threshold=rag_threshold,
            rule_threshold=rule_threshold
        )
        
        logger.info("Hybrid classifier created successfully")
        return hybrid_classifier
        
    except Exception as e:
        logger.error(f"Failed to create hybrid classifier: {e}")
        logger.warning("Falling back to keyword classifier")
        return create_keyword_classifier()


def create_classifier(
    classifier_type: str = "hybrid",
    **kwargs
) -> Union[KeywordClassifier, RAGClassifier, HybridClassifier]:
    """Create a classifier based on the specified type.
    
    Args:
        classifier_type: Type of classifier to create ("keyword", "rag", "hybrid")
        **kwargs: Additional configuration options
        
    Returns:
        Configured classifier instance
        
    Raises:
        ValueError: If classifier_type is not supported
    """
    if classifier_type == ClassifierType.KEYWORD:
        return create_keyword_classifier()
    
    elif classifier_type == ClassifierType.RAG:
        rag_classifier = create_rag_classifier_from_env(**kwargs)
        if rag_classifier is None:
            logger.warning("RAG classifier unavailable, falling back to keyword classifier")
            return create_keyword_classifier()
        return rag_classifier
    
    elif classifier_type == ClassifierType.HYBRID:
        return create_hybrid_classifier_from_env(**kwargs)
    
    else:
        raise ValueError(
            f"Unsupported classifier type: {classifier_type}. "
            f"Supported types: {ClassifierType.KEYWORD}, {ClassifierType.RAG}, {ClassifierType.HYBRID}"
        )


def get_classifier_info(classifier) -> dict:
    """Get information about a classifier instance.
    
    Args:
        classifier: Classifier instance
        
    Returns:
        Dictionary with classifier information
    """
    classifier_type = type(classifier).__name__
    info = {
        "type": classifier_type,
        "class": classifier_type
    }
    
    # Add specific information based on classifier type
    if isinstance(classifier, HybridClassifier):
        stats = classifier.get_stats()
        info.update({
            "rag_threshold": stats.get("rag_threshold"),
            "rule_threshold": stats.get("rule_threshold"),
            "last_method": stats.get("last_method"),
            "capabilities": ["semantic_search", "rule_based", "fallback"]
        })
    elif isinstance(classifier, RAGClassifier):
        info.update({
            "confidence_threshold": classifier.get_confidence_threshold(),
            "capabilities": ["semantic_search", "llm_assisted"]
        })
    elif isinstance(classifier, KeywordClassifier):
        info.update({
            "capabilities": ["rule_based", "keyword_matching"]
        })
    
    return info
