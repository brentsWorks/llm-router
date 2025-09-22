#!/usr/bin/env python3
"""Test scoring algorithm with various prompt types and weight configurations"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.api.main import router_service
from backend.scoring import ScoringWeights

def test_prompt_scenario(prompt, description, weights_config):
    """Test a specific prompt with given weights"""
    print(f"\n🔍 {description}")
    print(f"Prompt: '{prompt}'")
    print(f"Weights: {weights_config}")
    
    try:
        result = router_service.route(prompt, preferences=weights_config)
        if result:
            print(f"✅ Selected: {result.selected_model.provider}/{result.selected_model.model}")
            print(f"   Category: {result.classification.category} (confidence: {result.classification.confidence:.2f})")
        else:
            print("❌ No model selected")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("=== TESTING SCORING ALGORITHM EDGE CASES ===\n")
    
    # Test scenarios
    test_cases = [
        {
            "prompt": "Write a Python function to calculate fibonacci",
            "description": "Code Generation (High Quality Needed)",
            "category": "code"
        },
        {
            "prompt": "Write a romantic poem about autumn leaves",
            "description": "Creative Writing (Quality vs Speed)",
            "category": "creative"
        },
        {
            "prompt": "What is the capital of France?",
            "description": "Simple Q&A (Speed Should Win)",
            "category": "qa"
        },
        {
            "prompt": "Analyze the economic impact of inflation on small businesses in developing countries",
            "description": "Complex Analysis (Quality Critical)",
            "category": "analysis"
        },
        {
            "prompt": "Summarize this text: [long document]",
            "description": "Summarization (Balance Speed + Quality)",
            "category": "summarization"
        },
        {
            "prompt": "Solve this complex calculus problem: ∫x²sin(x)dx",
            "description": "Math Problem (Quality Critical)",
            "category": "math"
        },
        {
            "prompt": "Help me debug this JavaScript error: Cannot read property of undefined",
            "description": "Technical Support (Need Accuracy)",
            "category": "code"
        }
    ]
    
    # Weight configurations to test
    weight_configs = [
        {"name": "Balanced", "weights": ScoringWeights(cost_weight=0.33, latency_weight=0.33, quality_weight=0.34)},
        {"name": "Cost-First", "weights": ScoringWeights(cost_weight=0.7, latency_weight=0.15, quality_weight=0.15)},
        {"name": "Speed-First", "weights": ScoringWeights(cost_weight=0.15, latency_weight=0.7, quality_weight=0.15)},
        {"name": "Quality-First", "weights": ScoringWeights(cost_weight=0.15, latency_weight=0.15, quality_weight=0.7)},
        {"name": "Production-Like", "weights": ScoringWeights(cost_weight=0.4, latency_weight=0.4, quality_weight=0.2)},
    ]
    
    # Test each case with different weight configurations
    for case in test_cases:
        print(f"\n{'='*80}")
        print(f"📝 SCENARIO: {case['description']}")
        print(f"Expected Category: {case['category']}")
        print('='*80)
        
        for config in weight_configs:
            test_prompt_scenario(
                case["prompt"], 
                f"{config['name']} ({config['weights'].cost_weight:.1f}/{config['weights'].latency_weight:.1f}/{config['weights'].quality_weight:.1f})",
                config["weights"]
            )
    
    print(f"\n{'='*80}")
    print("🧪 EDGE CASE TESTING")
    print('='*80)
    
    # Test extreme cases
    edge_cases = [
        {
            "prompt": "Hi",
            "description": "Very Short Prompt",
        },
        {
            "prompt": "a" * 1000,
            "description": "Very Long Prompt",
        },
        {
            "prompt": "🚀🎨💡🔥✨",
            "description": "Emoji-Only Prompt",
        },
        {
            "prompt": "def func():\n    return None\n\nclass MyClass:\n    pass",
            "description": "Code Block Prompt",
        }
    ]
    
    balanced_weights = ScoringWeights(cost_weight=0.33, latency_weight=0.33, quality_weight=0.34)
    
    for case in edge_cases:
        test_prompt_scenario(case["prompt"], case["description"], balanced_weights)

if __name__ == "__main__":
    main()