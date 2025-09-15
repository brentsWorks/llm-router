"""
LLM Fallback Classification System.

This module provides LLM-based classification for edge cases and novel prompt types
when RAG + rule-based classification fails or has low confidence.
"""

import json
from typing import Dict, Any, Optional
from llm_router.models import PromptClassification
from llm_router.capabilities import TaskType


class LLMFallbackError(Exception):
    """Exception raised for LLM fallback classification errors."""
    pass


class LLMFallbackClassifier:
    """LLM-based classifier for edge cases and novel prompt types."""
    
    def __init__(
        self,
        llm_client: Any,
        api_key: str,
        model_name: str = "gpt-3.5-turbo"
    ):
        """Initialize the LLM fallback classifier.
        
        Args:
            llm_client: LLM client for making API calls
            api_key: API key for the LLM service
            model_name: Name of the LLM model to use
            
        Raises:
            LLMFallbackError: If API key is not provided
        """
        if not api_key:
            raise LLMFallbackError("API key is required")
        
        self.llm_client = llm_client
        self.api_key = api_key
        self.model_name = model_name
    
    def classify(self, prompt: str) -> PromptClassification:
        """Classify a prompt using LLM-based analysis.
        
        Args:
            prompt: The prompt to classify
            
        Returns:
            PromptClassification with category, confidence, and reasoning
            
        Raises:
            LLMFallbackError: If classification fails
        """
        try:
            # Generate classification prompt
            classification_prompt = self._generate_classification_prompt(prompt)
            
            # Call LLM
            response = self.llm_client.generate_content(classification_prompt)
            
            # Check for valid response
            if not response or not hasattr(response, 'text') or not response.text:
                raise LLMFallbackError("Empty response from LLM API")
            
            # Parse response
            result = self._parse_llm_response(response.text)
            
            # Validate result
            self._validate_classification_result(result)
            
            return PromptClassification(
                category=result["category"],
                confidence=result["confidence"],
                embedding=[],  # Empty for LLM fallback
                reasoning=result["reasoning"]
            )
            
        except Exception as e:
            if "API" in str(e) or "rate limit" in str(e).lower():
                raise LLMFallbackError(f"LLM API error: {e}")
            else:
                raise LLMFallbackError(f"Classification failed: {e}")
    
    def _generate_classification_prompt(self, prompt: str) -> str:
        """Generate the classification prompt for the LLM.
        
        Args:
            prompt: The original prompt to classify
            
        Returns:
            Formatted classification prompt
        """
        valid_categories = [cat.value for cat in TaskType]
        categories_str = ", ".join(valid_categories)
        
        return f"""
        Analyze the following prompt and classify it into one of these categories:
        - {categories_str}

        Prompt: "{prompt}"
        
        Respond with JSON format:
        {{
            "category": "category_name",
            "confidence": 0.0-1.0,
            "reasoning": "explanation of classification decision"
        }}
        """
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured format.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed classification result
            
        Raises:
            LLMFallbackError: If response cannot be parsed
        """
        try:
            # Clean up response text (remove markdown if present)
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            result = json.loads(cleaned_text)
            
            # Validate required fields
            required_fields = ["category", "confidence", "reasoning"]
            for field in required_fields:
                if field not in result:
                    raise LLMFallbackError(f"Missing required field: {field}")
            
            return result
            
        except json.JSONDecodeError as e:
            raise LLMFallbackError(f"Failed to parse LLM response: {e}")
        except Exception as e:
            raise LLMFallbackError(f"Failed to parse LLM response: {e}")
    
    def _validate_classification_result(self, result: Dict[str, Any]) -> None:
        """Validate the classification result.
        
        Args:
            result: Parsed classification result
            
        Raises:
            LLMFallbackError: If result is invalid
        """
        # Validate category
        valid_categories = [cat.value for cat in TaskType]
        if result["category"] not in valid_categories:
            raise LLMFallbackError(f"Invalid category '{result['category']}'. Valid categories: {valid_categories}")
        
        # Validate confidence
        confidence = result["confidence"]
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise LLMFallbackError(f"Invalid confidence value: {confidence}. Must be between 0.0 and 1.0")
        
        # Validate reasoning
        if not isinstance(result["reasoning"], str) or not result["reasoning"].strip():
            raise LLMFallbackError("Reasoning must be a non-empty string")
