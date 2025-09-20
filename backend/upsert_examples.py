#!/usr/bin/env python3
"""
Script to upsert examples into the vector index for RAG classification.
Run this from the root directory.
"""

import os
import logging
from pathlib import Path

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value

# Load environment variables
load_env()

from backend.vector_service import VectorService
from backend.dataset import ExampleDataset
from backend.embeddings import EmbeddingService
from backend.vector_stores.vector_store_interface import VectorRecord

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upsert_examples():
    """Load examples from JSON and upsert them into the vector index."""
    
    try:
        # Load examples from JSON file
        examples_path = Path(__file__).parent / "data" / "examples.json"
        logger.info(f"Loading examples from: {examples_path}")
        
        dataset = ExampleDataset.from_json_file(examples_path)
        examples = dataset.get_all_examples()
        logger.info(f"Loaded {len(examples)} examples")
        
        # Initialize vector service
        logger.info("Initializing vector service...")
        vector_service = VectorService()
        
        # Initialize embedding service
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService()
        
        # Upsert examples
        logger.info("Upserting examples into vector index...")
        
        # Process examples in batches
        batch_size = 50
        total_examples = len(examples)
        
        for i in range(0, total_examples, batch_size):
            batch = examples[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_examples + batch_size - 1)//batch_size} ({len(batch)} examples)")
            
            # Convert examples to vectors and upsert
            for example in batch:
                # Generate embedding for the example text
                embedding = embedding_service.embed(example.text)
                
                # Create metadata for the vector
                metadata = {
                    "text": example.text,
                    "category": example.category.value,
                    "preferred_models": example.preferred_models,
                    "description": example.description or "",
                    "difficulty": example.difficulty or "",
                    "expected_length": example.expected_length or "",
                    "domain": example.domain or "",
                    "tags": example.tags
                }
                
                # Create unique ID for the example
                example_id = f"example_{i + examples.index(example)}"
                
                # Add to vector store
                vector_service.vector_store.add_vector(
                    vector_id=example_id,
                    vector=embedding,
                    metadata=metadata
                )
        
        logger.info(f"Successfully upserted {total_examples} examples into vector index")
        
        # Verify the upsert by doing a test search
        logger.info("Verifying upsert with test search...")
        test_query = "Write a Python function"
        results = vector_service.find_similar_examples(test_query, top_k=3)
        
        logger.info(f"Test search returned {len(results)} results:")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result.text[:50]}... (category: {result.metadata.get('category', 'unknown')})")
        
        logger.info("✅ Examples successfully upserted and verified!")
        
    except Exception as e:
        logger.error(f"❌ Error upserting examples: {e}")
        raise

if __name__ == "__main__":
    upsert_examples()
