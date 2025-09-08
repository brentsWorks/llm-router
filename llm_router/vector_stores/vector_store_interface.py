"""
Vector store interface for pluggable backends.

This module defines the interface that allows easy switching between
in-memory vector storage and cloud providers like Pinecone.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum

class VectorStoreError(Exception):
    """Exception raised for vector store related errors."""
    pass


class SimilarityMetric(Enum):
    """Supported similarity metrics."""
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


@dataclass
class SearchResult:
    """Represents a search result with similarity and metadata."""
    vector: np.ndarray
    similarity: float
    index: int
    metadata: Optional[Dict[str, Any]] = None


class VectorStoreBackend(Enum):
    """Supported vector store backends."""
    PINECONE = "pinecone"
    CHROMA = "chroma"
    WEAVIATE = "weaviate"


@dataclass
class VectorRecord:
    """Represents a vector record with ID and metadata."""
    id: str
    vector: np.ndarray
    metadata: Optional[Dict[str, Any]] = None


class VectorStoreInterface(ABC):
    """Abstract interface for vector storage and similarity search."""
    
    @abstractmethod
    def add_vector(self, vector_id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a vector to the store.
        
        Args:
            vector_id: Unique identifier for the vector
            vector: Vector to add
            metadata: Optional metadata associated with the vector
            
        Raises:
            VectorStoreError: If vector is invalid or ID already exists
        """
        pass
    
    @abstractmethod
    def add_vectors_batch(self, records: List[VectorRecord]) -> None:
        """Add multiple vectors in a batch operation.
        
        Args:
            records: List of vector records to add
            
        Raises:
            VectorStoreError: If any vector is invalid
        """
        pass
    
    @abstractmethod
    def search(self, 
               query_vector: np.ndarray, 
               k: int = 5, 
               filter_metadata: Optional[Dict[str, Any]] = None,
               exclude_ids: Optional[List[str]] = None) -> List[SearchResult]:
        """Search for k most similar vectors.
        
        Args:
            query_vector: Vector to search for
            k: Number of nearest neighbors to return
            filter_metadata: Optional metadata filters
            exclude_ids: Optional vector IDs to exclude from results
            
        Returns:
            List of SearchResult objects sorted by similarity (highest first)
            
        Raises:
            VectorStoreError: If query vector is invalid or store is empty
        """
        pass
    
    @abstractmethod
    def get_vector(self, vector_id: str) -> Optional[VectorRecord]:
        """Get a vector by ID.
        
        Args:
            vector_id: ID of the vector to retrieve
            
        Returns:
            VectorRecord if found, None otherwise
        """
        pass
    
    @abstractmethod
    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID.
        
        Args:
            vector_id: ID of the vector to delete
            
        Returns:
            True if vector was deleted, False if not found
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the store."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get the number of vectors in the store."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> Optional[int]:
        """Get the vector dimension."""
        pass
    
    def calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """Calculate confidence score based on search results.
        
        This method provides a default implementation that can be overridden.
        
        Args:
            search_results: List of search results
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not search_results:
            return 0.0
        
        if len(search_results) == 1:
            return max(0.0, min(1.0, search_results[0].similarity))
        
        # Multiple results: consider score distribution
        scores = [result.similarity for result in search_results]
        max_score = max(scores)
        
        if len(scores) >= 2:
            sorted_scores = sorted(scores, reverse=True)
            score_gap = sorted_scores[0] - sorted_scores[1]
            confidence = max_score * 0.8 + score_gap * 0.2
        else:
            confidence = max_score
        
        return max(0.0, min(1.0, confidence))


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create_store(backend: VectorStoreBackend, 
                    dimension: Optional[int] = None,
                    **kwargs) -> VectorStoreInterface:
        """Create a vector store instance.
        
        Args:
            backend: Type of vector store backend
            dimension: Expected vector dimension
            **kwargs: Backend-specific configuration
            
        Returns:
            VectorStoreInterface implementation
            
        Raises:
            VectorStoreError: If backend is not supported
        """
        if backend == VectorStoreBackend.PINECONE:
            from .vector_store_pinecone import PineconeVectorStore
            return PineconeVectorStore(dimension=dimension, **kwargs)
        
        elif backend == VectorStoreBackend.CHROMA:
            from .vector_store_chroma import ChromaVectorStore
            return ChromaVectorStore(dimension=dimension, **kwargs)
        
        else:
            raise VectorStoreError(f"Unsupported backend: {backend}")


# Convenience functions for backward compatibility
def create_vector_store(backend: str = "pinecone", **kwargs) -> VectorStoreInterface:
    """Create a vector store with string backend name."""
    backend_enum = VectorStoreBackend(backend)
    return VectorStoreFactory.create_store(backend_enum, **kwargs)


def create_pinecone_store(api_key: str, 
                         environment: str,
                         index_name: str,
                         dimension: int = 384,
                         **kwargs) -> VectorStoreInterface:
    """Create a Pinecone vector store."""
    return VectorStoreFactory.create_store(
        VectorStoreBackend.PINECONE,
        api_key=api_key,
        environment=environment,
        index_name=index_name,
        dimension=dimension,
        **kwargs
    )
