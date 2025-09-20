"""
RAG-enhanced classification system for LLM Router.

This module implements a RAG (Retrieval-Augmented Generation) classifier that
combines semantic similarity search with LLM-based classification using Gemini Pro.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai

from .models import PromptClassification
from .vector_service import VectorService
from .vector_stores import SearchResult

logger = logging.getLogger(__name__)


class RAGClassificationError(Exception):
    """Exception raised for RAG classification errors."""
    pass


class RAGClassifier:
    """RAG-enhanced classifier using semantic similarity and Gemini Pro."""

    def __init__(
        self,
        vector_service: VectorService,
        gemini_client: Optional[Any] = None,
        api_key: str = "",
        confidence_threshold: float = 0.5,
        max_similar_examples: int = 3,
        model_name: str = "gemini-2.5-flash-lite"
    ):
        """Initialize RAG classifier.
        
        Args:
            vector_service: Vector service for similarity search
            gemini_client: Pre-configured Gemini client (optional)
            api_key: Gemini API key
            confidence_threshold: Minimum confidence for RAG classification
            max_similar_examples: Maximum number of similar examples to use
            model_name: Gemini model name to use
            
        Raises:
            RAGClassificationError: If configuration is invalid
        """
        if not api_key or not api_key.strip():
            raise RAGClassificationError("API key is required for Gemini integration")
        
        self.vector_service = vector_service
        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        self.max_similar_examples = max_similar_examples
        self.model_name = model_name
        
        # Initialize Gemini client
        if gemini_client is not None:
            self.gemini_client = gemini_client
        else:
            genai.configure(api_key=api_key)
            self.gemini_client = genai.GenerativeModel(model_name)
        
        logger.info(f"RAG classifier initialized with model: {model_name}")

    def classify(self, prompt: str) -> PromptClassification:
        """Classify a prompt using RAG with Gemini Pro.
        
        Args:
            prompt: The input prompt to classify
            
        Returns:
            PromptClassification with category, confidence, and reasoning
            
        Raises:
            RAGClassificationError: If classification fails
        """
        # Validate input
        if prompt is None:
            raise RAGClassificationError("Prompt cannot be None")
        
        if not prompt or not prompt.strip():
            raise RAGClassificationError("Prompt cannot be empty")
        
        prompt = prompt.strip()
        
        try:
            # Find similar examples using vector search
            similar_examples = self._get_similar_examples(prompt)
            
            # Calculate confidence based on similarity scores
            rag_confidence = self._calculate_similarity_confidence(similar_examples)
            
            # Only use Gemini if we have good similarity matches
            if rag_confidence >= self.confidence_threshold:
                # Generate classification using Gemini Pro
                gemini_response = self._query_gemini(prompt, similar_examples)
                
                # Parse and validate response
                classification_data = self._parse_gemini_response(gemini_response)
                
                # Create PromptClassification object with similarity-based confidence
                return self._create_classification(classification_data, rag_confidence)
            else:
                # Low similarity - RAG is not confident enough
                raise RAGClassificationError(f"RAG confidence too low: {rag_confidence:.3f} < {self.confidence_threshold}")
            
        except Exception as e:
            if isinstance(e, RAGClassificationError):
                raise
            logger.error(f"Unexpected error in RAG classification: {e}")
            raise RAGClassificationError(f"Classification failed: {str(e)}") from e

    def _get_similar_examples(self, prompt: str) -> List[SearchResult]:
        """Get similar examples from vector service.
        
        Args:
            prompt: Input prompt
            
        Returns:
            List of similar examples
            
        Raises:
            RAGClassificationError: If vector search fails
        """
        try:
            similar_examples = self.vector_service.find_similar_examples(
                prompt,
                k=self.max_similar_examples
            )
            
            logger.debug(f"Found {len(similar_examples)} similar examples for prompt")
            return similar_examples
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RAGClassificationError(f"Vector search error: {str(e)}") from e

    def _calculate_similarity_confidence(self, similar_examples: List[SearchResult]) -> float:
        """Calculate confidence based on similarity scores.
        
        Args:
            similar_examples: List of similar examples with similarity scores
            
        Returns:
            Confidence score based on similarity (0.0-1.0)
        """
        if not similar_examples:
            return 0.0
        
        # Use the highest similarity score as the base confidence
        max_similarity = max(example.similarity for example in similar_examples)
        
        # Apply a scaling factor to convert similarity to confidence
        # Similarity scores are typically 0.0-1.0, but we want to be more conservative
        # Scale by 2.0 to make the threshold more meaningful for our similarity range
        confidence = min(max_similarity * 2.0, 1.0)
        
        logger.debug(f"Similarity confidence: {confidence:.3f} (max similarity: {max_similarity:.3f})")
        return confidence

    def _query_gemini(self, prompt: str, similar_examples: List[SearchResult]) -> str:
        """Query Gemini Pro for classification.
        
        Args:
            prompt: Input prompt
            similar_examples: List of similar examples
            
        Returns:
            Gemini response text
            
        Raises:
            RAGClassificationError: If Gemini API fails
        """
        try:
            # Build the classification prompt
            classification_prompt = self._build_classification_prompt(prompt, similar_examples)
            
            # Query Gemini
            response = self.gemini_client.generate_content(classification_prompt)
            
            if not response or not hasattr(response, 'text') or not response.text:
                raise RAGClassificationError("Empty response from Gemini API")
            
            logger.debug("Successfully received response from Gemini Pro")
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise RAGClassificationError(f"Gemini API error: {str(e)}") from e

    def _build_classification_prompt(self, prompt: str, similar_examples: List[SearchResult]) -> str:
        """Build the classification prompt for Gemini Pro.
        
        Args:
            prompt: Input prompt
            similar_examples: List of similar examples
            
        Returns:
            Formatted prompt for Gemini
        """
        # Format similar examples
        examples_text = ""
        if similar_examples:
            examples_text = "SIMILAR EXAMPLES:\n"
            for i, example in enumerate(similar_examples, 1):
                metadata = example.metadata
                examples_text += f"""
Example {i} (similarity: {example.similarity:.2f}):
- Text: "{metadata.get('text', 'N/A')}"
- Category: {metadata.get('category', 'N/A')}
- Preferred Models: {metadata.get('preferred_models', [])}
- Difficulty: {metadata.get('difficulty', 'N/A')}
- Domain: {metadata.get('domain', 'N/A')}
"""
        else:
            examples_text = "SIMILAR EXAMPLES: None found\n"
        
        # Build the full prompt
        classification_prompt = f"""You are an expert AI prompt classifier. Analyze the user prompt and classify it based on the similar examples provided.

{examples_text}

USER PROMPT: "{prompt}"

Based on the similar examples and prompt content, provide a JSON response with the following structure:
{{
    "category": "one of: code, creative, qa, summarization, reasoning, tool_use, translation, analysis",
    "confidence": 0.85,
    "recommended_models": ["model1", "model2"],
    "reasoning": "Detailed explanation of why this classification was chosen"
}}

Category mapping:
- code: Programming, coding, software development tasks
- creative: Creative writing, storytelling, poetry, artistic content
- qa: Question answering, factual queries, explanations
- summarization: Summarizing documents, text, or information
- reasoning: Logic puzzles, problem solving, critical thinking
- tool_use: Function calling, API usage, tool integration
- translation: Language translation tasks
- analysis: Data analysis, research, pattern recognition

Guidelines:
- Use the similar examples to inform your classification
- Consider the prompt's intent, complexity, and domain
- Confidence should be 0.0-1.0 (higher when similar examples are very relevant)
- Recommended models should be based on similar examples and task requirements
- Provide clear reasoning for your classification decision

Respond with ONLY the JSON object, no additional text."""

        return classification_prompt

    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate Gemini response.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Parsed classification data
            
        Raises:
            RAGClassificationError: If response is invalid
        """
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned_response = response_text.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            # Try to parse JSON
            data = json.loads(cleaned_response)
            
            # Validate required fields
            required_fields = ["category", "confidence", "reasoning"]
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                raise RAGClassificationError(
                    f"Missing required fields in Gemini response: {missing_fields}"
                )
            
            # Validate confidence bounds
            confidence = data["confidence"]
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                raise RAGClassificationError(
                    f"Invalid confidence value: {confidence}. Must be between 0.0 and 1.0"
                )
            
            # Validate category
            valid_categories = [
                "code", "creative", "qa", "summarization", 
                "reasoning", "tool_use", "translation", "analysis"
            ]
            
            if data["category"] not in valid_categories:
                logger.warning(f"Unexpected category from Gemini: {data['category']}")
                # Don't fail, but log the warning
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Gemini: {response_text}")
            raise RAGClassificationError(f"Invalid response format from Gemini: {str(e)}") from e

    def _create_classification(self, data: Dict[str, Any], similarity_confidence: float) -> PromptClassification:
        """Create PromptClassification object from parsed data.
        
        Args:
            data: Parsed classification data from Gemini
            similarity_confidence: Confidence based on similarity scores
            
        Returns:
            PromptClassification object
        """
        # Extract recommended models if present
        recommended_models = data.get("recommended_models", [])
        
        # Include recommended models in reasoning
        reasoning = data["reasoning"]
        if recommended_models:
            reasoning += f" (Recommended models: {', '.join(recommended_models)})"
        
        # Use similarity-based confidence instead of Gemini's confidence
        classification = PromptClassification(
            category=data["category"],
            confidence=similarity_confidence,  # Use similarity-based confidence
            embedding=[],  # Empty for RAG classification
            reasoning=reasoning
        )
        
        # Store recommended models as a custom attribute for retrieval
        # This is a workaround since PromptClassification doesn't have metadata field
        setattr(classification, '_recommended_models', recommended_models)
        
        return classification

    def get_confidence_threshold(self) -> float:
        """Get the confidence threshold for this classifier.
        
        Returns:
            Confidence threshold value
        """
        return self.confidence_threshold

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for this classifier.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
            
        Raises:
            RAGClassificationError: If threshold is invalid
        """
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            raise RAGClassificationError(
                f"Invalid confidence threshold: {threshold}. Must be between 0.0 and 1.0"
            )
        
        self.confidence_threshold = threshold
        logger.info(f"Updated confidence threshold to: {threshold}")


# Factory function for easy instantiation
def create_rag_classifier(
    vector_service: VectorService,
    api_key: str,
    **kwargs
) -> RAGClassifier:
    """Create a RAG classifier instance.
    
    Args:
        vector_service: Vector service for similarity search
        api_key: Gemini API key
        **kwargs: Additional configuration options
        
    Returns:
        Configured RAGClassifier instance
    """
    return RAGClassifier(
        vector_service=vector_service,
        api_key=api_key,
        **kwargs
    )
