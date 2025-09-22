# LLM Router Demo Script

## Overview
A comprehensive demo script to showcase the full functionality of the LLM Router application - intelligent model selection and execution for optimal LLM routing.

---

## 1. **Introduction & App Overview** (30 seconds)
- Open the deployed app at your Railway frontend URL
- Show the clean, professional interface with three main tabs
- Explain the core value proposition: "Automatically selects the best LLM for your specific task based on your preferences"

**Script**: *"This is the LLM Router - an intelligent system that analyzes your prompt, understands what you're trying to accomplish, and automatically selects the optimal language model based on your priorities for cost, speed, and quality."*

---

## 2. **Models Tab - Understanding Available Options** (45 seconds)

### Show Model Catalog
- Click on **Models** tab
- Scroll through the available models (GPT-4, Claude, Gemini, etc.)
- Point out the **8+ models** from major providers

### Demonstrate Filtering
- Click on **"Code Generation"** filter
- Show how models are filtered and ranked for coding tasks
- Try **"Creative Writing"** filter
- Show the **latency** and **cost** sorting options

**Script**: *"We have 8+ models from major providers. You can filter by capability - see how the rankings change when we filter for code generation versus creative writing. Each model has real performance data for different task types."*

---

## 3. **Router Tab - Core Functionality Demo** (2 minutes)

### Basic Routing Demo
- Go to **Router** tab
- Type a **coding prompt**: *"Write a Python function to calculate the Fibonacci sequence using dynamic programming"*
- **Before submitting**: Show the preferences panel on the right
- Click **Route Prompt**

### Show Status Indicator
- Point out the **status indicator** changing from "Ready to route" → "Finding optimal model..." → "[Provider]/[Model] selected"

### Explain Results
- Point out the **3 key metrics**:
  - **Routing Confidence**: "How certain we are about this choice"
  - **Overall Score**: "Weighted score based on your preferences" 
  - **Model Quality**: "Expected output quality for this specific task"
- Show the **selected model** with performance metrics
- Point out **category detection**: "code" with confidence percentage

**Script**: *"Watch the status indicator - it's analyzing the prompt... It detected this as a 'code' task with high confidence, and selected GPT-4 because it excels at code generation. The overall score of 87% reflects how well this model matches our current preferences."*

---

## 4. **Preferences Customization** (1 minute)

### Demonstrate Weight Adjustment
- Adjust the **Cost Optimization** slider to 80% (others will adjust automatically)
- Re-run the same prompt
- Show how a **different model** gets selected (likely a cheaper option)
- Point out the **new overall score** and reasoning

### Show Extreme Preferences
- Set **Quality** to 100% (others to 0%)
- Re-run the prompt
- Show premium model selection

**Script**: *"Let's change our priorities - if I prioritize cost at 80%, watch how it selects a more economical model. Now if I only care about quality at 100%, it chooses the premium option regardless of cost."*

---

## 5. **Full Execution Demo** (1.5 minutes)

### Execute with Selected Model
- Use a creative prompt: *"Write a short story about a robot discovering emotions for the first time"*
- Click **Route Prompt**
- Show the routing results
- Click **"Execute with this Model"**

### Show Live Execution
- Point out status changing to "Generating response..."
- Show the **real LLM response** appearing
- Point out **execution metrics**: model used, execution time, token usage

### Test Response Actions
- Click **"Copy Response"**
- Try **"Regenerate"** to show variability

**Script**: *"Now let's see the full pipeline in action. It routes to Claude for creative writing... and here's the actual response from the LLM. We can copy it, regenerate for variations, or try different models."*

---

## 6. **Advanced Constraints** (45 seconds)

### Show Constraint Options
- Expand the **"Advanced Settings"** in preferences
- Set **Max Cost**: $0.01 per 1K tokens
- Set **Max Latency**: 2000ms
- Try routing a complex prompt

### Demonstrate Filtering
- Show how models get **filtered out** based on constraints
- Explain **provider exclusions** functionality

**Script**: *"For production use, you can set hard constraints - maximum cost per token, latency limits, or exclude specific providers. This ensures the router only considers models that meet your requirements."*

---

## 7. **Error Handling & Edge Cases** (30 seconds)

### Show Input Validation
- Try submitting an **empty prompt**
- Try a **very short prompt** (under 10 characters)
- Show the **validation messages**

### Demonstrate Fallbacks
- Try an **unusual prompt** that might have lower confidence
- Show how the system still provides recommendations

**Script**: *"The system includes robust validation and handles edge cases gracefully, always providing intelligent fallbacks even for unusual requests."*

---

## 8. **About Tab - Technical Deep Dive** (45 seconds)

### Show Architecture Overview
- Click **About** tab
- Scroll through the **hybrid classification system**
- Point out **RAG + LLM fallback** architecture
- Show **real performance metrics**

**Script**: *"Under the hood, we use a sophisticated hybrid classification system - vector search through curated examples, with LLM fallback for novel requests. This ensures accuracy across diverse prompt types."*

---

## 9. **Mobile Responsiveness** (30 seconds)

### Show Mobile View
- Resize browser to mobile width (or show on phone)
- Navigate through all tabs
- Show **responsive layout** and **touch-friendly interface**
- Test routing on mobile

**Script**: *"The entire interface is fully responsive - all functionality works seamlessly on mobile devices with optimized layouts for smaller screens."*

---

## 10. **Real-World Value Proposition** (30 seconds)

### Summarize Key Benefits
- **Time Saving**: No manual model selection
- **Cost Optimization**: Automatic cost-aware routing
- **Performance**: Choose best model for each task
- **Simplicity**: One API for multiple providers

### Use Cases
- **Development Teams**: Integrate into applications
- **Content Creation**: Optimize for creative vs. analytical tasks
- **Production Systems**: Cost and latency constraints

**Script**: *"This solves the real problem of LLM proliferation - instead of manually choosing between dozens of models, developers can focus on their application while the router automatically optimizes for their specific needs and constraints."*

---

## Demo Tips

### Preparation
- Have 3-4 different prompt types ready (code, creative, Q&A, analysis)
- Test the full flow beforehand
- Ensure stable internet connection
- Have both desktop and mobile views ready

### Timing
- **Total Demo**: ~8-9 minutes for full coverage
- **Quick Demo**: Focus on sections 1, 3, 5 (~4 minutes)
- **Technical Demo**: Emphasize sections 2, 6, 8 for developer audience

### Key Selling Points to Emphasize
1. **Intelligence**: Automatic prompt classification and model selection
2. **Flexibility**: Customizable preferences and constraints
3. **Real Execution**: Not just routing - actual LLM responses
4. **Production Ready**: Error handling, monitoring, responsive design
5. **Multi-Provider**: Unified interface for 12+ models

---

## Technical Details to Mention
- **Classification**: RAG-based with 120 curated examples + LLM fallback
- **Scoring**: Multi-factor optimization (cost, latency, quality)
- **Execution**: Real-time via OpenRouter API
- **Architecture**: React frontend + FastAPI backend
- **Deployment**: Production-ready on Railway

This demo script covers all major functionality while maintaining good pacing and clear value demonstration!