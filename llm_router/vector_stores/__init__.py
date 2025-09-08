"""
Vector store implementations for the LLM Router.

This package contains various vector store backends for storing and searching
text embeddings used in semantic similarity search.
"""

from .vector_store_interface import (
    VectorStoreInterface,
    VectorStoreError,
    SimilarityMetric,
    SearchResult,
    VectorStoreBackend,
    VectorRecord,
    VectorStoreFactory,
    create_vector_store,
    create_pinecone_store,
)
from .vector_store_pinecone import PineconeVectorStore

__all__ = [
    "VectorStoreInterface",
    "VectorStoreError",
    "SimilarityMetric", 
    "SearchResult",
    "VectorStoreBackend",
    "VectorRecord",
    "VectorStoreFactory",
    "create_vector_store",
    "create_pinecone_store",
    "PineconeVectorStore",
]
