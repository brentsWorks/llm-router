# 🚀 LLM Router API - Quick Start Guide

Get started with the LLM Router API in minutes!

## 📋 Prerequisites

- Python 3.11+
- Virtual environment (recommended)

## ⚡ Quick Setup

1. **Clone and Setup**
   ```bash
   git clone https://github.com/yourusername/llm-router.git
   cd llm-router
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[api]"
   ```

2. **Start the API Server**
   ```bash
   python run_api.py
   ```
   
   The API will be available at: http://localhost:8000
   - **Swagger UI**: http://localhost:8000/docs
   - **ReDoc**: http://localhost:8000/redoc

## 🎯 Basic Usage

### 1. Check Available Models
```bash
curl http://localhost:8000/models
```

### 2. Basic Routing
```bash
curl -X POST "http://localhost:8000/route" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to calculate fibonacci numbers"
  }'
```

### 3. Cost-Optimized Routing
```bash
curl -X POST "http://localhost:8000/route" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize the benefits of renewable energy",
    "preferences": {
      "cost_weight": 0.8,
      "latency_weight": 0.1,
      "quality_weight": 0.1
    }
  }'
```

### 4. Constrained Routing
```bash
curl -X POST "http://localhost:8000/route" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a detailed business plan",
    "preferences": {
      "cost_weight": 0.4,
      "latency_weight": 0.3,
      "quality_weight": 0.3
    },
    "constraints": {
      "max_cost_per_1k_tokens": 0.01,
      "max_latency_ms": 2000,
      "excluded_providers": ["expensive_provider"]
    }
  }'
```

## 🔧 Configuration Options

### Preferences (Scoring Weights)
All weights must be between 0.0 and 1.0 and sum to 1.0:

- **`cost_weight`**: Prioritize cheaper models
- **`latency_weight`**: Prioritize faster models  
- **`quality_weight`**: Prioritize more capable models

### Constraints (Filtering)
Optional filters to limit model selection:

- **`max_cost_per_1k_tokens`**: Maximum cost per 1K tokens
- **`max_latency_ms`**: Maximum latency in milliseconds
- **`max_context_length`**: Maximum context window needed
- **`min_safety_level`**: Minimum safety level ("low", "moderate", "high")
- **`excluded_providers`**: List of providers to exclude (e.g., ["openai"])
- **`excluded_models`**: List of models to exclude (e.g., ["gpt-4"])

## 📊 Understanding Responses

### Successful Response
```json
{
  "selected_model": {
    "provider": "anthropic",
    "model": "claude-3-haiku"
  },
  "classification": {
    "category": "code",
    "confidence": 0.85
  },
  "confidence": 0.85,
  "routing_time_ms": 12.5,
  "reasoning": "Selected for optimal cost-performance ratio"
}
```

### Error Response
```json
{
  "detail": "Weights must sum to 1.0"
}
```

## 🔍 Monitoring & Debugging

### Health Check
```bash
curl http://localhost:8000/health
```

### Performance Metrics
```bash
curl http://localhost:8000/metrics
```

### Classification Testing
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a sorting algorithm"}'
```

## 📚 More Examples

Run the comprehensive examples:
```bash
python examples/api_usage_examples.py
```

This includes examples for:
- Basic routing
- Cost/quality/latency optimization
- Provider exclusions
- Error handling
- And more!

## 🆘 Troubleshooting

### Common Issues

1. **API Not Starting**
   - Check if port 8000 is available
   - Ensure virtual environment is activated
   - Install dependencies: `pip install -e ".[api]"`

2. **Validation Errors**
   - Ensure preference weights sum to 1.0
   - Check constraint values are positive
   - Verify JSON format is correct

3. **No Models Available**
   - Check `/models` endpoint to see available models
   - Adjust constraints to be less restrictive
   - Verify model configuration in `llm_router/config/models.json`

### Getting Help

- **Documentation**: Check `/docs` endpoint for interactive API docs
- **Examples**: See `examples/api_usage_examples.py`
- **Issues**: Report bugs on GitHub

## 🎉 Next Steps

- Explore the full API documentation at `/docs`
- Try different preference combinations
- Experiment with constraints
- Check out the comprehensive examples
- Integrate with your application!

Happy routing! 🚀
