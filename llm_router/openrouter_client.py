"""
OpenRouter API client for unified LLM access.

This module provides a client for accessing multiple LLM providers through
OpenRouter's unified API interface.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OpenRouterMessage(BaseModel):
    """Message format for OpenRouter API."""
    role: str
    content: str


class OpenRouterResponse(BaseModel):
    """Response from OpenRouter API."""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class OpenRouterError(Exception):
    """Exception raised for OpenRouter API errors."""
    pass


class OpenRouterClient:
    """Client for OpenRouter API using OpenAI SDK format."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        """Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            site_url: Optional site URL for rankings
            site_name: Optional site name for rankings
            
        Raises:
            OpenRouterError: If API key is not provided
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError("OpenRouter API key is required")
        
        self.site_url = site_url or os.getenv("SITE_URL", "https://github.com/your-repo")
        self.site_name = site_name or os.getenv("SITE_NAME", "LLM Router")
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        logger.info("OpenRouter client initialized")
    
    def chat_completion(
        self,
        model: str,
        messages: List[OpenRouterMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> OpenRouterResponse:
        """Generate chat completion using OpenRouter.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4", "anthropic/claude-3-haiku")
            messages: List of messages for the conversation
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for the API
            
        Returns:
            OpenRouterResponse with the generated content
            
        Raises:
            OpenRouterError: If the API call fails
        """
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Make API call
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract response
            choice = completion.choices[0]
            usage = completion.usage.dict() if completion.usage else None
            
            response = OpenRouterResponse(
                content=choice.message.content,
                model=completion.model,
                usage=usage,
                finish_reason=choice.finish_reason
            )
            
            logger.debug(f"OpenRouter completion successful: {model}")
            return response
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise OpenRouterError(f"OpenRouter API call failed: {str(e)}") from e
    
    def simple_completion(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Simple completion for a single prompt.
        
        Args:
            model: Model identifier
            prompt: The prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text content
            
        Raises:
            OpenRouterError: If the API call fails
        """
        messages = [OpenRouterMessage(role="user", content=prompt)]
        response = self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content


def create_openrouter_client() -> OpenRouterClient:
    """Create OpenRouter client from environment variables.
    
    Returns:
        Configured OpenRouter client
        
    Raises:
        OpenRouterError: If configuration is invalid
    """
    try:
        return OpenRouterClient()
    except Exception as e:
        logger.error(f"Failed to create OpenRouter client: {e}")
        raise OpenRouterError(f"Failed to create OpenRouter client: {str(e)}") from e
