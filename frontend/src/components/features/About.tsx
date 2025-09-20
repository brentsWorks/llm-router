import { Card } from '../ui/Card.tsx';

export const About: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <Card padding="lg">
        <div className="text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            🧠 LLM Router
          </h2>
          <p className="text-lg text-gray-600 mb-6">
            A hybrid LLM routing system that intelligently selects the best language model for your prompts using semantic analysis and AI-powered classification.
          </p>
          <div className="flex justify-center space-x-4 text-sm text-gray-500">
            <span className="flex items-center">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
              Hybrid Classification
            </span>
            <span className="flex items-center">
              <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
              Cost Optimization
            </span>
            <span className="flex items-center">
              <span className="w-2 h-2 bg-purple-500 rounded-full mr-2"></span>
              Multi-Provider
            </span>
          </div>
        </div>
      </Card>

      {/* How It Works */}
      <Card padding="lg">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">
          How It Works
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-2xl">🔍</span>
            </div>
            <h4 className="font-medium text-gray-900 mb-2">1. Analyze</h4>
            <p className="text-sm text-gray-600">
              Uses RAG (Retrieval-Augmented Generation) to find similar examples and classify your prompt semantically.
            </p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-2xl">🤖</span>
            </div>
            <h4 className="font-medium text-gray-900 mb-2">2. Classify</h4>
            <p className="text-sm text-gray-600">
              Falls back to LLM classification when RAG confidence is low, ensuring accurate categorization.
            </p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-2xl">⚡</span>
            </div>
            <h4 className="font-medium text-gray-900 mb-2">3. Route</h4>
            <p className="text-sm text-gray-600">
              Selects the optimal model based on your preferences for cost, latency, and quality.
            </p>
          </div>
        </div>
      </Card>

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card padding="lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            🎯 Key Features
          </h3>
          <ul className="space-y-2 text-sm text-gray-600">
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              <span>Hybrid RAG + LLM classification</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              <span>Multi-provider support (OpenAI, Anthropic, Google, xAI)</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              <span>Cost and latency optimization</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              <span>Constraint-based filtering</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              <span>Real-time model selection metrics</span>
            </li>
          </ul>
        </Card>

        <Card padding="lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            🏗️ Architecture
          </h3>
          <div className="space-y-3 text-sm">
            <div className="flex items-center">
              <span className="w-3 h-3 bg-blue-500 rounded-full mr-3"></span>
              <span className="text-gray-600">FastAPI Backend</span>
            </div>
            <div className="flex items-center">
              <span className="w-3 h-3 bg-green-500 rounded-full mr-3"></span>
              <span className="text-gray-600">React + TypeScript Frontend</span>
            </div>
            <div className="flex items-center">
              <span className="w-3 h-3 bg-purple-500 rounded-full mr-3"></span>
              <span className="text-gray-600">Pinecone Vector Database</span>
            </div>
            <div className="flex items-center">
              <span className="w-3 h-3 bg-orange-500 rounded-full mr-3"></span>
              <span className="text-gray-600">OpenRouter API Integration</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Technical Details */}
      <Card padding="lg">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">
          🔧 Technical Architecture
        </h3>
        <div className="space-y-4">
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Hybrid Classification System</h4>
            <p className="text-sm text-gray-600 mb-2">
              The system uses a two-stage approach for prompt classification:
            </p>
            <ul className="text-sm text-gray-600 ml-4 space-y-1">
              <li>• <strong>RAG Classification:</strong> Semantic similarity search against 120+ curated examples using Pinecone vector database</li>
              <li>• <strong>LLM Fallback:</strong> When RAG confidence is below 60%, uses Gemini Pro for classification</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Model Selection & Scoring</h4>
            <p className="text-sm text-gray-600 mb-2">
              Models are scored based on three weighted factors:
            </p>
            <ul className="text-sm text-gray-600 ml-4 space-y-1">
              <li>• <strong>Quality Score:</strong> How well the model performs for the specific capability needed (e.g., 95% for GPT-5 on code tasks, 89% for Gemini on creative writing)</li>
              <li>• <strong>Cost Efficiency:</strong> Price per 1K tokens with realistic input/output splits</li>
              <li>• <strong>Latency:</strong> Average response time in milliseconds</li>
            </ul>
            <p className="text-xs text-gray-500 mt-2 italic">
              The quality score helps you understand not just that a model was selected, but how well-suited it is for your specific type of prompt.
            </p>
          </div>

          <div>
            <h4 className="font-medium text-gray-900 mb-2">Supported Capabilities</h4>
            <div className="flex flex-wrap gap-2 mt-2">
              {['code', 'creative', 'qa', 'reasoning', 'analysis', 'summarization', 'translation', 'math', 'science', 'writing', 'conversation', 'tool_use'].map((capability) => (
                <span key={capability} className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded-full">
                  {capability}
                </span>
              ))}
            </div>
          </div>

          <div>
            <h4 className="font-medium text-gray-900 mb-2">Advanced Constraints</h4>
            <p className="text-sm text-gray-600 mb-2">
              Fine-tune model selection with precise filtering options:
            </p>
            <ul className="text-sm text-gray-600 ml-4 space-y-1">
              <li>• <strong>Cost Limits:</strong> Set maximum cost per 1K tokens to control spending</li>
              <li>• <strong>Latency Limits:</strong> Specify maximum response time in milliseconds</li>
              <li>• <strong>Context Length:</strong> Ensure model can handle your prompt length</li>
              <li>• <strong>Safety Level:</strong> Choose minimum safety requirements (High/Moderate/Low)</li>
              <li>• <strong>Provider Exclusion:</strong> Block specific providers (OpenAI, Anthropic, Google, xAI)</li>
              <li>• <strong>Model Exclusion:</strong> Exclude specific models from consideration</li>
            </ul>
            <p className="text-xs text-gray-500 mt-2 italic">
              Constraints help you balance performance, cost, and compliance requirements for your specific use case.
            </p>
          </div>
        </div>
      </Card>

      {/* Getting Started */}
      <Card padding="lg">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">
          🚀 Getting Started
        </h3>
        <div className="space-y-3 text-sm text-gray-600">
          <p>
            <strong>1. Enter your prompt</strong> - Describe what you want to accomplish with an LLM
          </p>
          <p>
            <strong>2. Adjust preferences</strong> - Use the sliders to prioritize cost, latency, or quality
          </p>
          <p>
            <strong>3. Set constraints</strong> - Exclude providers, set cost limits, or specify safety levels
          </p>
          <p>
            <strong>4. Execute</strong> - The system will route to the optimal model and execute your prompt
          </p>
        </div>
      </Card>
    </div>
  );
};
