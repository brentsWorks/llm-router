#!/usr/bin/env python3
"""
Direct OpenRouter Integration Test
=================================

Test OpenRouter integration directly without the API server.
This verifies that the OpenRouter service works correctly.
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from llm_router.openrouter_client import OpenRouterClient
from llm_router.openrouter_service import OpenRouterService, LLMExecutionRequest
from llm_router.models import RoutingDecision, ModelCandidate, PromptClassification
from llm_router.capabilities import TaskType

def test_openrouter_execution():
    """Test OpenRouter execution directly."""
    print("🧪 Direct OpenRouter Integration Test")
    print("=" * 50)
    
    try:
        # Create OpenRouter client and service
        print("1. Creating OpenRouter client...")
        client = OpenRouterClient()
        print("   ✅ OpenRouter client created")
        
        print("2. Creating OpenRouter service...")
        service = OpenRouterService(client)
        print("   ✅ OpenRouter service created")
        
        # Create a mock routing decision
        print("3. Creating mock routing decision...")
        selected_model = ModelCandidate(
            provider='openai',
            model='gpt-3.5-turbo',
            score=0.9,
            estimated_cost=0.001,
            estimated_latency=100.0,
            quality_match=0.9,
            constraint_violations=[]
        )
        
        classification = PromptClassification(
            category=TaskType.CODE,
            confidence=0.8,
            embedding=[0.1] * 384,  # Mock embedding vector
            reasoning="Test classification"
        )
        
        routing_decision = RoutingDecision(
            selected_model=selected_model,
            classification=classification,
            alternatives=[],
            routing_time_ms=10.5,
            confidence=0.8,
            reasoning="Test routing decision"
        )
        print("   ✅ Routing decision created")
        
        # Create execution request
        print("4. Creating execution request...")
        execution_request = LLMExecutionRequest(
            prompt="Write a simple Python function to add two numbers",
            routing_decision=routing_decision,
            temperature=0.7,
            max_tokens=200
        )
        print("   ✅ Execution request created")
        
        # Execute the request
        print("5. Executing via OpenRouter...")
        print(f"   Model: {selected_model.provider}/{selected_model.model}")
        print(f"   Prompt: {execution_request.prompt}")
        
        result = service.execute_prompt(execution_request)
        
        print("   ✅ Execution successful!")
        print(f"   Response: {result.content[:200]}...")
        print(f"   Model used: {result.model_used}")
        print(f"   Execution time: {result.execution_time_ms:.1f}ms")
        if result.usage:
            print(f"   Usage: {result.usage}")
        
        print("\n🎯 OpenRouter Integration: SUCCESS!")
        print("   - Client creation: ✅")
        print("   - Service creation: ✅") 
        print("   - Request creation: ✅")
        print("   - API execution: ✅")
        print("   - Response parsing: ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ OpenRouter Integration: FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the direct OpenRouter test."""
    # Check if API key is available
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ OPENROUTER_API_KEY not found in environment")
        print("   Please ensure your .env file contains the API key")
        return
    
    print(f"✅ OPENROUTER_API_KEY found: {os.getenv('OPENROUTER_API_KEY')[:10]}...")
    
    # Run the test
    success = test_openrouter_execution()
    
    if success:
        print("\n🚀 Ready for API integration!")
        print("   The OpenRouter service is working correctly.")
        print("   Restart the API server to enable the /execute endpoint.")
    else:
        print("\n🔧 Fix the issues above before proceeding.")

if __name__ == "__main__":
    main()
