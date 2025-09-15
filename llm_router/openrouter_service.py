"""
OpenRouter service for executing LLM prompts with selected models.

This service integrates OpenRouter API with our routing decisions to provide
end-to-end LLM execution.
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

from .openrouter_client import OpenRouterClient, OpenRouterMessage, OpenRouterError
from .models import RoutingDecision, ModelCandidate

logger = logging.getLogger(__name__)


class LLMExecutionRequest(BaseModel):
    """Request for LLM execution via OpenRouter."""
    prompt: str
    routing_decision: RoutingDecision
    temperature: float = 0.7
    max_tokens: Optional[int] = None


class LLMExecutionResponse(BaseModel):
    """Response from LLM execution."""
    content: str
    model_used: str
    routing_decision: RoutingDecision
    usage: Optional[Dict[str, Any]] = None
    execution_time_ms: float
    finish_reason: Optional[str] = None


class OpenRouterExecutionError(Exception):
    """Exception raised for OpenRouter execution errors."""
    pass


class OpenRouterService:
    """Service for executing LLM prompts via OpenRouter."""
    
    def __init__(self, openrouter_client: OpenRouterClient):
        """Initialize OpenRouter service.
        
        Args:
            openrouter_client: Configured OpenRouter client
        """
        self.client = openrouter_client
        logger.info("OpenRouter service initialized")
    
    def execute_prompt(self, request: LLMExecutionRequest) -> LLMExecutionResponse:
        """Execute a prompt using the selected model via OpenRouter.
        
        Args:
            request: LLM execution request with prompt and routing decision
            
        Returns:
            LLM execution response with generated content
            
        Raises:
            OpenRouterExecutionError: If execution fails
        """
        import time
        start_time = time.time()
        
        try:
            # Extract model information from routing decision
            selected_model = request.routing_decision.selected_model
            provider = selected_model.provider
            model_name = selected_model.model
            
            # Format model identifier for OpenRouter (provider/model)
            openrouter_model = f"{provider}/{model_name}"
            
            logger.info(f"Executing prompt with {openrouter_model}")
            logger.debug(f"Prompt: {request.prompt[:100]}...")
            
            # Create message for OpenRouter
            messages = [OpenRouterMessage(role="user", content=request.prompt)]
            
            # Execute via OpenRouter
            response = self.client.chat_completion(
                model=openrouter_model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Create execution response
            execution_response = LLMExecutionResponse(
                content=response.content,
                model_used=openrouter_model,
                routing_decision=request.routing_decision,
                usage=response.usage,
                execution_time_ms=execution_time,
                finish_reason=response.finish_reason
            )
            
            logger.info(f"Execution completed in {execution_time:.1f}ms")
            return execution_response
            
        except OpenRouterError as e:
            logger.error(f"OpenRouter execution failed: {e}")
            raise OpenRouterExecutionError(f"Failed to execute prompt: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during execution: {e}")
            raise OpenRouterExecutionError(f"Execution error: {str(e)}") from e
    
    def execute_simple(
        self,
        prompt: str,
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Simple execution without routing decision.
        
        Args:
            prompt: The prompt to execute
            provider: Model provider (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4", "claude-3-haiku")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated content
            
        Raises:
            OpenRouterExecutionError: If execution fails
        """
        try:
            openrouter_model = f"{provider}/{model}"
            content = self.client.simple_completion(
                model=openrouter_model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logger.info(f"Simple execution completed with {openrouter_model}")
            return content
            
        except OpenRouterError as e:
            logger.error(f"Simple execution failed: {e}")
            raise OpenRouterExecutionError(f"Failed to execute prompt: {str(e)}") from e


def create_openrouter_service() -> OpenRouterService:
    """Create OpenRouter service with default client.
    
    Returns:
        Configured OpenRouter service
        
    Raises:
        OpenRouterExecutionError: If service creation fails
    """
    try:
        from .openrouter_client import create_openrouter_client
        client = create_openrouter_client()
        return OpenRouterService(client)
    except Exception as e:
        logger.error(f"Failed to create OpenRouter service: {e}")
        raise OpenRouterExecutionError(f"Service creation failed: {str(e)}") from e
