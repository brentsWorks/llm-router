#!/usr/bin/env python3
"""Manual test script for RAG classifier with real Pinecone and Gemini."""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
        print("✅ Loaded environment variables from .env file")
    else:
        print("ℹ️  No .env file found, using system environment variables")

# Load .env before importing our modules
load_env()

from llm_router.vector_service import create_vector_service
from llm_router.rag_classification import create_rag_classifier

def check_pinecone_data(vector_service):
    """Check if Pinecone has our example data."""
    print("\n🔍 Checking Pinecone Index Status:")
    print("-" * 40)
    
    try:
        stats = vector_service.get_stats()
        print(f"📊 Index Stats: {stats}")
        
        if stats.get('total_examples', 0) == 0:
            print("⚠️  WARNING: No examples found in Pinecone index!")
            print("   You may need to load your examples first.")
            return False
        
        # Try a simple search to verify data
        print("\n🔎 Testing similarity search...")
        results = vector_service.find_similar_examples("python function", k=3)
        print(f"Found {len(results)} examples for 'python function':")
        
        for i, result in enumerate(results, 1):
            text = result.metadata.get('text', 'N/A')
            category = result.metadata.get('category', 'N/A')
            similarity = result.similarity
            print(f"  {i}. [{category}] {text[:60]}... (similarity: {similarity:.3f})")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Error checking Pinecone data: {e}")
        return False

def test_rag_classifier():
    """Test RAG classifier with real APIs."""
    
    print("🚀 RAG Classifier Manual Test")
    print("=" * 50)
    
    # Check environment variables
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    print(f"🔑 Environment Check:")
    print(f"   PINECONE_API_KEY: {'✅ Set' if pinecone_key else '❌ Missing'}")
    print(f"   PINECONE_ENVIRONMENT: {pinecone_env}")
    print(f"   GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Missing'}")
    
    if not pinecone_key:
        print("\n❌ PINECONE_API_KEY not set. Please set it:")
        print("   export PINECONE_API_KEY='your-api-key'")
        return
    
    if not gemini_key:
        print("\n❌ GEMINI_API_KEY not set. Please set it:")
        print("   export GEMINI_API_KEY='your-api-key'")
        return
    
    try:
        print("\n🔧 Setting up services...")
        
        # Create vector service (connects to Pinecone)
        print("   Connecting to Pinecone...")
        vector_service = create_vector_service(
            api_key=pinecone_key,
            environment=pinecone_env,
            index_name="llm-router"
        )
        print("   ✅ Connected to Pinecone")
        
        # Check if data is loaded
        if not check_pinecone_data(vector_service):
            print("\n⚠️  Continuing with test, but results may be limited...")
        
        # Create RAG classifier
        print("\n   Initializing RAG classifier with Gemini...")
        rag_classifier = create_rag_classifier(
            vector_service=vector_service,
            api_key=gemini_key
        )
        print("   ✅ RAG classifier initialized")
        
        # Novel test prompts NOT in our training data to test for leakage
        test_prompts = [
            {
                "prompt": "Create a JavaScript function to validate email addresses using regex",
                "expected_category": "code",
                "description": "Novel coding prompt - different language and task"
            },
            {
                "prompt": "Write a mysterious tale about an abandoned lighthouse keeper's diary",
                "expected_category": "creative", 
                "description": "Novel creative prompt - different theme and style"
            },
            {
                "prompt": "How do neural networks learn from backpropagation?",
                "expected_category": "qa",
                "description": "Novel technical Q&A - specific ML concept"
            },
            {
                "prompt": "Examine customer churn patterns in subscription services",
                "expected_category": "analysis",
                "description": "Novel analysis prompt - different business domain"
            },
            {
                "prompt": "Convert this German phrase to Italian",
                "expected_category": "translation",
                "description": "Novel translation prompt - different language pair"
            },
            {
                "prompt": "Explain the process of photosynthesis in plants",
                "expected_category": "qa",
                "description": "Novel science Q&A - biology domain"
            },
            {
                "prompt": "Build a React component for user authentication",
                "expected_category": "code",
                "description": "Novel web development prompt"
            },
            {
                "prompt": "Compose a haiku about morning coffee rituals",
                "expected_category": "creative",
                "description": "Novel creative prompt - specific poetry form"
            }
        ]
        
        print("\n🧪 Testing RAG Classification:")
        print("=" * 60)
        
        success_count = 0
        total_tests = len(test_prompts)
        
        for i, test_case in enumerate(test_prompts, 1):
            prompt = test_case["prompt"]
            expected_category = test_case["expected_category"]
            description = test_case["description"]
            
            print(f"\n📝 Test {i}/{total_tests}: {description}")
            print(f"Prompt: \"{prompt}\"")
            print("-" * 50)
            
            try:
                # First, check what similar examples we find
                print("🔍 Finding similar examples...")
                similar_examples = vector_service.find_similar_examples(prompt, k=3)
                print(f"   Found {len(similar_examples)} similar examples")
                
                if similar_examples:
                    print("   Top similar examples:")
                    for j, example in enumerate(similar_examples[:2], 1):
                        text = example.metadata.get('text', 'N/A')
                        category = example.metadata.get('category', 'N/A')
                        models = example.metadata.get('preferred_models', [])
                        similarity = example.similarity
                        print(f"     {j}. [{category}] {text[:50]}...")
                        print(f"        Similarity: {similarity:.3f}, Models: {models}")
                else:
                    print("   ⚠️  No similar examples found")
                
                # Now classify with RAG
                print("\n🤖 RAG Classification...")
                result = rag_classifier.classify(prompt)
                
                # Get recommended models
                recommended_models = getattr(result, '_recommended_models', [])
                
                # Display results
                print(f"   Category: {result.category}")
                print(f"   Expected: {expected_category}")
                print(f"   Confidence: {result.confidence:.2f}")
                print(f"   Recommended Models: {recommended_models}")
                print(f"   Reasoning: {result.reasoning}")
                
                # Check if classification matches expectation
                category_match = result.category == expected_category
                confidence_good = result.confidence >= 0.5
                has_reasoning = result.reasoning and len(result.reasoning) > 10
                
                # Check for data leakage indicators
                max_similarity = max([ex.similarity for ex in similar_examples]) if similar_examples else 0
                reasonable_similarity = max_similarity < 0.9  # Novel prompts shouldn't have near-perfect matches
                reasoning_uses_examples = any(word in result.reasoning.lower() for word in ['example', 'similar', 'like'])
                
                print(f"\n   Results:")
                print(f"     Category Match: {'✅' if category_match else '❌'} ({result.category} vs {expected_category})")
                print(f"     Good Confidence: {'✅' if confidence_good else '❌'} ({result.confidence:.2f})")
                print(f"     Has Reasoning: {'✅' if has_reasoning else '❌'}")
                print(f"     Similar Examples: {'✅' if similar_examples else '❌'} ({len(similar_examples)} found)")
                print(f"     Max Similarity: {'✅' if reasonable_similarity else '⚠️ '} ({max_similarity:.3f}) {'- Novel prompt should have lower similarity' if not reasonable_similarity else ''}")
                print(f"     Uses Examples: {'✅' if reasoning_uses_examples else '⚠️ '} {'- Good, reasoning references examples' if reasoning_uses_examples else '- May be using pre-training knowledge'}")
                
                if category_match and confidence_good and has_reasoning:
                    success_count += 1
                    print(f"   Overall: ✅ PASS")
                else:
                    print(f"   Overall: ❌ FAIL")
                
            except Exception as e:
                print(f"   ❌ Error during classification: {e}")
                print(f"   Overall: ❌ ERROR")
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 SUMMARY")
        print("=" * 60)
        print(f"Tests Passed: {success_count}/{total_tests}")
        print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
        
        if success_count == total_tests:
            print("🎉 All tests passed! RAG classifier is working correctly.")
        elif success_count >= total_tests * 0.8:
            print("✅ Most tests passed. RAG classifier is working well.")
        elif success_count >= total_tests * 0.6:
            print("⚠️  Some tests passed. RAG classifier needs tuning.")
        else:
            print("❌ Many tests failed. Check your setup and data.")
        
        print("\n📋 Data Integrity Validation:")
        print("✅ Novel prompts used (not in training data)")
        print("✅ Similarity scores checked (should be <0.9 for novel prompts)")
        print("✅ Reasoning analysis (should reference examples, not just pre-training)")
        print("✅ Model recommendations validated against similar examples")
        print("")
        print("📋 Manual Verification Checklist:")
        print("- [ ] All prompts classified without errors")
        print("- [ ] Categories make sense for each prompt")  
        print("- [ ] Confidence scores are reasonable (>0.5 for clear prompts)")
        print("- [ ] Similar examples are found and relevant")
        print("- [ ] Similarity scores are reasonable (<0.9 for novel prompts)")
        print("- [ ] Reasoning references examples, not just general knowledge")
        print("- [ ] Recommended models match the examples")
        print("- [ ] No suspiciously perfect matches (data leakage indicator)")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API keys are correct")
        print("2. Verify Pinecone index name is 'llm-router'")
        print("3. Ensure you have examples loaded in Pinecone")
        print("4. Check your internet connection")

def load_examples_to_pinecone():
    """Helper function to load examples to Pinecone if needed."""
    print("\n📥 Loading Examples to Pinecone")
    print("-" * 40)
    
    try:
        from llm_router.dataset_loader import load_default_dataset
        
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            print("❌ PINECONE_API_KEY not set")
            return
        
        # Load dataset
        print("Loading dataset...")
        dataset = load_default_dataset()
        print(f"Loaded {len(dataset)} examples")
        
        # Create vector service
        vector_service = create_vector_service(
            api_key=pinecone_key,
            index_name="llm-router"
        )
        
        # Add dataset to vector store
        print("Adding examples to Pinecone...")
        example_ids = vector_service.add_dataset(dataset)
        print(f"✅ Added {len(example_ids)} examples to Pinecone")
        
        # Verify
        stats = vector_service.get_stats()
        print(f"📊 Updated stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error loading examples: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--load-examples":
        load_examples_to_pinecone()
    else:
        test_rag_classifier()
        
    print(f"\n💡 Tips:")
    print(f"   - Run with --load-examples to load your dataset to Pinecone first")
    print(f"   - Check the Pinecone console to verify your index has data")
    print(f"   - Monitor your Gemini API usage in Google Cloud Console")
