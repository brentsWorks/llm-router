#!/usr/bin/env python3
"""Debug script to test the exact scoring behavior"""

import json
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.scoring import ScoringEngine, ScoringWeights
from backend.registry import ProviderModel
from backend.registry import PricingInfo, PerformanceInfo

def load_models_from_json():
    """Load models from the JSON config"""
    with open('/Users/brent/Desktop/Software/projects/llm-router/backend/config/models.json', 'r') as f:
        data = json.load(f)
    
    models = []
    for model_data in data['models']:
        # Create ProviderModel instances
        pricing = PricingInfo(
            input_tokens_per_1k=model_data['pricing']['input_tokens_per_1k'],
            output_tokens_per_1k=model_data['pricing']['output_tokens_per_1k']
        )
        
        performance = PerformanceInfo(
            avg_latency_ms=model_data['performance']['avg_latency_ms'],
            quality_scores=model_data['performance']['quality_scores']
        )
        
        model = ProviderModel(
            provider=model_data['provider'],
            model=model_data['model'],
            capabilities=model_data['capabilities'],
            pricing=pricing,
            performance=performance,
            limits=model_data['limits']
        )
        models.append(model)
    
    return models

def test_scoring():
    """Test the actual scoring behavior"""
    print("=== DEBUGGING SCORING BEHAVIOR ===\n")
    
    # Load models
    models = load_models_from_json()
    engine = ScoringEngine()
    
    # Test with different weight configurations
    test_configs = [
        {"name": "Balanced (33/33/34)", "weights": ScoringWeights(cost_weight=0.33, latency_weight=0.33, quality_weight=0.34)},
        {"name": "Default Frontend (30/30/40)", "weights": ScoringWeights(cost_weight=0.3, latency_weight=0.3, quality_weight=0.4)},
        {"name": "Quality Focus (15/15/70)", "weights": ScoringWeights(cost_weight=0.15, latency_weight=0.15, quality_weight=0.7)},
        {"name": "Default None", "weights": None}
    ]
    
    category = "code"
    
    for config in test_configs:
        print(f"=== {config['name']} ===")
        print(f"Weights: {config['weights']}")
        
        scored_models = []
        for model in models:
            if category in model.capabilities:
                score_result = engine.calculate_score(model, category, config['weights'])
                scored_models.append((model, score_result))
        
        # Sort by overall score
        scored_models.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        print("\nRanking:")
        for i, (model, score) in enumerate(scored_models[:5]):  # Top 5
            print(f"{i+1}. {model.provider}/{model.model}")
            print(f"   Overall: {score.overall_score:.3f} | Quality: {score.quality_score:.3f} | Cost: {score.cost_score:.3f} | Latency: {score.latency_score:.3f}")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_scoring()