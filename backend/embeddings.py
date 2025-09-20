"""
Embedding Service for LLM Router

This module provides text embedding generation using sentence-transformers,
with caching support for performance optimization.
"""

import numpy as np
from typing import List, Union, Optional, Dict, Any
from functools import lru_cache
import logging
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Exception raised for embedding-related errors."""
    pass


class EmbeddingCache:
    """LRU cache for text embeddings."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        if max_size <= 0:
            raise EmbeddingError("Cache size must be positive")
            
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache.
        
        Args:
            text: Text to look up
            
        Returns:
            Cached embedding or None if not found
        """
        if text in self._cache:
            # Move to end (most recently used)
            embedding = self._cache.pop(text)
            self._cache[text] = embedding
            return embedding.copy()  # Return copy to prevent modification
        return None
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache.
        
        Args:
            text: Text key
            embedding: Embedding to cache
        """
        if text in self._cache:
            # Update existing entry
            self._cache.pop(text)
        elif len(self._cache) >= self.max_size:
            # Remove oldest entry
            self._cache.popitem(last=False)
        
        self._cache[text] = embedding.copy()
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
    
    def __len__(self) -> int:
        """Get number of cached embeddings."""
        return len(self._cache)


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        use_cache: bool = True,
        cache_size: int = 1000,
        batch_size: int = 32
    ):
        """Initialize embedding service.
        
        Args:
            model_name: Name of sentence-transformer model to use
            normalize: Whether to normalize embeddings (L2 norm = 1)
            use_cache: Whether to cache embeddings
            cache_size: Maximum number of embeddings to cache
            batch_size: Batch size for embedding generation
        """
        if not model_name or not model_name.strip():
            raise EmbeddingError("Model name cannot be empty")
        
        if batch_size <= 0:
            raise EmbeddingError("Batch size must be positive")
        
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        
        # Initialize cache if enabled
        self.use_cache = use_cache
        self._cache = EmbeddingCache(cache_size) if use_cache else None
        
        # Load model
        try:
            self._load_model()
        except Exception as e:
            raise EmbeddingError(f"Model loading failed: {str(e)}") from e
    
    def _load_model(self):
        """Load the sentence-transformer model."""
        try:
            # Import here to avoid dependency issues during testing
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            # Use a dummy text to get the dimension
            dummy_embedding = self.model.encode(["dummy"], convert_to_numpy=True)
            self.embedding_dim = dummy_embedding.shape[1]
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            EmbeddingError: If text is invalid or embedding generation fails
        """
        if text is None:
            raise EmbeddingError("Text cannot be None")
        
        if not text or not text.strip():
            raise EmbeddingError("Text cannot be empty")
        
        # Check cache first
        if self.use_cache and self._cache is not None:
            cached_embedding = self._cache.get(text)
            if cached_embedding is not None:
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return cached_embedding
        
        try:
            # Generate embedding
            embeddings = self._generate_embeddings([text])
            embedding = embeddings[0]  # Get first (and only) embedding
            
            # Cache if enabled
            if self.use_cache and self._cache is not None:
                self._cache.put(text, embedding)
                logger.debug(f"Cached embedding for text: {text[:50]}...")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Exception in embed method: {str(e)}")
            raise EmbeddingError(f"Embedding generation failed: {str(e)}") from e
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embeddings as numpy array with shape (len(texts), embedding_dim)
            
        Raises:
            EmbeddingError: If texts are invalid or embedding generation fails
        """
        if not texts:
            raise EmbeddingError("Text list cannot be empty")
        
        # Validate all texts
        for i, text in enumerate(texts):
            if text is None:
                raise EmbeddingError(f"Text at index {i} cannot be None")
            if not text or not text.strip():
                raise EmbeddingError(f"Text at index {i} cannot be empty")
        
        # Check cache for all texts
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        if self.use_cache and self._cache is not None:
            for i, text in enumerate(texts):
                cached_embedding = self._cache.get(text)
                if cached_embedding is not None:
                    cached_embeddings[i] = cached_embedding
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                new_embeddings = self._generate_embeddings(uncached_texts)
                
                # Cache new embeddings
                if self.use_cache and self._cache is not None:
                    for text, embedding in zip(uncached_texts, new_embeddings):
                        self._cache.put(text, embedding)
                
            except Exception as e:
                raise EmbeddingError(f"Batch embedding generation failed: {str(e)}") from e
        else:
            new_embeddings = np.array([]).reshape(0, self.get_embedding_dimension())
        
        # Combine cached and new embeddings in original order
        result = np.zeros((len(texts), self.get_embedding_dimension()))
        
        # Fill in cached embeddings
        for i, embedding in cached_embeddings.items():
            result[i] = embedding
        
        # Fill in new embeddings
        for i, embedding in zip(uncached_indices, new_embeddings):
            result[i] = embedding
        
        return result
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using the sentence-transformer model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embeddings as numpy array
        """
        start_time = time.time()
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            
            duration = time.time() - start_time
            logger.debug(
                f"Generated {len(texts)} embeddings in {duration:.3f}s "
                f"({len(texts)/duration:.1f} texts/sec)"
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self._cache:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self._cache),
            "max_size": self._cache.max_size,
            "usage_percent": (len(self._cache) / self._cache.max_size) * 100
        }


# Factory function for easy instantiation
def create_embedding_service(
    model_name: Optional[str] = None,
    **kwargs
) -> EmbeddingService:
    """Create an embedding service with sensible defaults.
    
    Args:
        model_name: Optional model name override
        **kwargs: Additional arguments passed to EmbeddingService
        
    Returns:
        Configured EmbeddingService instance
    """
    if model_name is None:
        # Use lightweight model by default
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    return EmbeddingService(model_name=model_name, **kwargs)
