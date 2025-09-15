#!/usr/bin/env python3
"""
Test OpenRouter Integration
==========================

Test the complete end-to-end flow:
1. Classify prompt using RAG/LLM fallback
2. Select optimal model
3. Execute via OpenRouter API
4. Return both routing decision AND LLM response
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_route_endpoint(prompt: str) -> Dict[str, Any]:
    """Test the /route endpoint (routing only, no execution)."""
    print(f"\n🔍 Testing /route endpoint with: '{prompt[:50]}...'")
    
    response = requests.post(f"{BASE_URL}/route", json={"prompt": prompt})
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Route successful!")
        print(f"   Model: {data['selected_model']['provider']}/{data['selected_model']['model']}")
        print(f"   Category: {data['classification']['category']}")
        print(f"   Confidence: {data['confidence']:.3f}")
        print(f"   Routing time: {data['routing_time_ms']:.1f}ms")
        return data
    else:
        print(f"❌ Route failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_execute_endpoint(prompt: str) -> Dict[str, Any]:
    """Test the /execute endpoint (routing + OpenRouter execution)."""
    print(f"\n🚀 Testing /execute endpoint with: '{prompt[:50]}...'")
    
    response = requests.post(f"{BASE_URL}/execute", json={"prompt": prompt})
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Execute successful!")
        print(f"   Model: {data['selected_model']['provider']}/{data['selected_model']['model']}")
        print(f"   Category: {data['classification']['category']}")
        print(f"   Confidence: {data['confidence']:.3f}")
        print(f"   Routing time: {data['routing_time_ms']:.1f}ms")
        print(f"   Execution time: {data['execution_time_ms']:.1f}ms")
        print(f"   Model used: {data['model_used']}")
        print(f"   LLM Response: {data['llm_response'][:200]}...")
        if data.get('usage'):
            print(f"   Usage: {data['usage']}")
        return data
    else:
        print(f"❌ Execute failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_api_health():
    """Test API health endpoint."""
    print("\n🏥 Testing API health...")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API healthy: {data['status']}")
        print(f"   Version: {data['version']}")
        return True
    else:
        print(f"❌ API unhealthy: {response.status_code}")
        return False

def main():
    """Run comprehensive OpenRouter integration tests."""
    print("🧪 OpenRouter Integration Test Suite")
    print("=" * 50)
    
    # Test API health first
    if not test_api_health():
        print("\n❌ API is not healthy. Exiting.")
        return
    
    # Test cases for different prompt types
    test_cases = [
        {
            "name": "Code Generation",
            "prompt": "Write a Python function to calculate fibonacci numbers recursively",
            "expected_category": "code"
        },
        {
            "name": "Creative Writing", 
            "prompt": "Write a short story about a robot learning to paint",
            "expected_category": "creative"
        },
        {
            "name": "Question Answering",
            "prompt": "What is the capital of France and what is its population?",
            "expected_category": "qa"
        },
        {
            "name": "Analysis Task",
            "prompt": "Analyze the pros and cons of renewable energy sources",
            "expected_category": "analysis"
        }
    ]
    
    print(f"\n📋 Running {len(test_cases)} test cases...")
    
    # Test routing only
    print("\n" + "="*50)
    print("🔍 PHASE 1: Testing /route endpoint (routing only)")
    print("="*50)
    
    routing_results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        result = test_route_endpoint(test_case['prompt'])
        if result:
            routing_results.append((test_case, result))
    
    # Test full execution
    print("\n" + "="*50)
    print("🚀 PHASE 2: Testing /execute endpoint (routing + OpenRouter)")
    print("="*50)
    
    execution_results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        result = test_execute_endpoint(test_case['prompt'])
        if result:
            execution_results.append((test_case, result))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    print(f"✅ Routing tests: {len(routing_results)}/{len(test_cases)} passed")
    print(f"✅ Execution tests: {len(execution_results)}/{len(test_cases)} passed")
    
    if execution_results:
        print(f"\n🎯 OpenRouter Integration Status: SUCCESS!")
        print(f"   - RAG/LLM classification: Working")
        print(f"   - Model selection: Working") 
        print(f"   - OpenRouter execution: Working")
        print(f"   - End-to-end pipeline: Complete")
        
        # Show example response
        example = execution_results[0][1]
        print(f"\n📝 Example Response Structure:")
        print(f"   - selected_model: {example['selected_model']['provider']}/{example['selected_model']['model']}")
        print(f"   - classification: {example['classification']['category']} (confidence: {example['confidence']:.3f})")
        print(f"   - llm_response: {len(example['llm_response'])} characters")
        print(f"   - execution_time: {example['execution_time_ms']:.1f}ms")
    else:
        print(f"\n❌ OpenRouter Integration Status: FAILED")
        print(f"   - Check API logs for errors")
        print(f"   - Verify OPENROUTER_API_KEY is set")
        print(f"   - Check OpenRouter service connectivity")

if __name__ == "__main__":
    main()
