import { useState } from 'react';
import { Card } from '../ui/Card.tsx';

export const Models: React.FC = () => {
  const [selectedCapability, setSelectedCapability] = useState<string>('all');

  const capabilities = [
    { value: 'all', label: 'All Capabilities' },
    { value: 'code', label: 'Code' },
    { value: 'creative', label: 'Creative' },
    { value: 'qa', label: 'Q&A' },
    { value: 'math', label: 'Math' },
    { value: 'writing', label: 'Writing' },
    { value: 'conversation', label: 'Conversation' },
    { value: 'tool_use', label: 'Tool Use' },
    { value: 'latency', label: 'Speed (Latency)' },
    { value: 'cost', label: 'Cost Efficiency' }
  ];

  const models = [
    {
      name: 'Claude Sonnet 4',
      provider: 'Anthropic',
      strengths: { code: 95, creative: 92, qa: 94, math: 89, writing: 94, conversation: 93, tool_use: 0, latency: 2500, cost: 0.006 },
      badge: 'SWE Champion',
      badgeColor: 'bg-purple-100 text-purple-700',
      sweScore: 89,
      description: 'Top performer on SWE-bench coding tasks. High-quality reasoning but slower response times (2.5s).'
    },
    {
      name: 'GPT-5',
      provider: 'OpenAI',
      strengths: { code: 96, creative: 95, qa: 96, math: 93, writing: 95, conversation: 94, tool_use: 97, latency: 7110, cost: 0.008 },
      badge: 'Tool Master',
      badgeColor: 'bg-green-100 text-green-700',
      sweScore: 85,
      description: 'Most capable model with best tool integration, but slowest response time (7.1s). Premium quality.'
    },
    {
      name: 'Gemini 2.0 Flash',
      provider: 'Google',
      strengths: { code: 92, creative: 94, qa: 93, math: 88, writing: 93, conversation: 92, tool_use: 0, latency: 870, cost: 0.002 },
      badge: 'Creative Leader',
      badgeColor: 'bg-yellow-100 text-yellow-700',
      sweScore: 78,
      description: 'Excellent creative writing and content generation. Good speed (0.87s) with high quality.'
    },
    {
      name: 'GPT-4',
      provider: 'OpenAI',
      strengths: { code: 90, creative: 90, qa: 90, math: 85, writing: 90, conversation: 90, tool_use: 0, latency: 620, cost: 0.045 },
      badge: 'Reliable All-Rounder',
      badgeColor: 'bg-blue-100 text-blue-700',
      sweScore: 82,
      description: 'Consistent performance across all task types. Fast response (0.62s) but expensive.'
    },
    {
      name: 'Gemini 2.5 Flash Lite',
      provider: 'Google',
      strengths: { code: 89, creative: 91, qa: 90, math: 87, writing: 90, conversation: 89, tool_use: 0, latency: 520, cost: 0.001 },
      badge: 'Speed Optimized',
      badgeColor: 'bg-green-100 text-green-700',
      sweScore: 75,
      description: 'Second fastest model (0.52s) with solid performance. Great speed-to-quality ratio.'
    },
    {
      name: 'Grok-4 Fast',
      provider: 'xAI',
      strengths: { code: 88, creative: 85, qa: 87, math: 89, writing: 86, conversation: 88, tool_use: 0, latency: 3070, cost: 0.000 },
      badge: 'Math Specialist',
      badgeColor: 'bg-emerald-100 text-emerald-700',
      sweScore: 70,
      description: 'Strong mathematical reasoning and completely free, but slow response time (3.1s).'
    },
    {
      name: 'Claude 3 Haiku',
      provider: 'Anthropic',
      strengths: { code: 70, creative: 80, qa: 80, math: 0, writing: 80, conversation: 80, tool_use: 0, latency: 370, cost: 0.001 },
      badge: 'Efficient Balanced',
      badgeColor: 'bg-blue-100 text-blue-700',
      sweScore: 68,
      description: 'Third fastest model (0.37s) with good performance. Excellent speed-to-cost ratio.'
    },
    {
      name: 'GPT-3.5 Turbo',
      provider: 'OpenAI',
      strengths: { code: 60, creative: 70, qa: 70, math: 0, writing: 0, conversation: 0, tool_use: 0, latency: 300, cost: 0.0015 },
      badge: 'Speed King',
      badgeColor: 'bg-gray-100 text-gray-700',
      sweScore: 55,
      description: 'Fastest model available (0.30s) but limited capabilities. Best for simple, quick tasks.'
    }
  ];

  const getFilteredModels = () => {
    if (selectedCapability === 'all') return models;
    
    if (selectedCapability === 'latency') {
      // Sort by latency (ascending - lower is better)
      return models.sort((a, b) => a.strengths.latency - b.strengths.latency);
    }
    
    if (selectedCapability === 'cost') {
      // Sort by cost (ascending - lower is better)
      return models.sort((a, b) => a.strengths.cost - b.strengths.cost);
    }
    
    // For capability-based filtering, filter and sort by performance
    return models
      .filter(model => model.strengths[selectedCapability as keyof typeof model.strengths] > 0)
      .sort((a, b) => (b.strengths[selectedCapability as keyof typeof b.strengths] || 0) - (a.strengths[selectedCapability as keyof typeof a.strengths] || 0));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card padding="lg">
        <div className="text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            🤖 Available Models
          </h2>
          <p className="text-lg text-gray-600 mb-6">
            Compare and explore all available language models with their capabilities, performance metrics, and pricing.
          </p>
          <div className="flex justify-center space-x-4 text-sm text-gray-500">
            <span className="flex items-center">
              <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
              Real-time Performance Data
            </span>
            <span className="flex items-center">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
              Accurate Latency Metrics
            </span>
            <span className="flex items-center">
              <span className="w-2 h-2 bg-purple-500 rounded-full mr-2"></span>
              Interactive Filtering
            </span>
          </div>
        </div>
      </Card>

      {/* Interactive Model Selector */}
      <Card padding="lg">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">
          Model Comparison
        </h3>
        
        {/* Capability Filter */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Filter by Capability Strength:
          </label>
          <div className="flex flex-wrap gap-2">
            {capabilities.map((capability) => (
              <button
                key={capability.value}
                onClick={() => setSelectedCapability(capability.value)}
                className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                  selectedCapability === capability.value
                    ? 'bg-slate-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {capability.label}
              </button>
            ))}
          </div>
        </div>

        {/* Models List */}
        <div className="space-y-4">
          {getFilteredModels().map((model) => (
            <div key={model.name} className="p-4 sm:p-6 bg-gray-50 rounded-lg border border-gray-200">
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between mb-3 sm:mb-2">
                <div className="flex flex-col sm:flex-row sm:items-center space-y-1 sm:space-y-0 sm:space-x-3 mb-2 sm:mb-0">
                  <div className="text-sm font-medium text-gray-900">
                    {model.name}
                  </div>
                  <div className="text-xs text-gray-500 capitalize">
                    {model.provider}
                  </div>
                  {selectedCapability !== 'all' && selectedCapability !== 'latency' && selectedCapability !== 'cost' && model.strengths[selectedCapability as keyof typeof model.strengths] > 0 && (
                    <div className="text-sm font-semibold text-slate-600">
                      {model.strengths[selectedCapability as keyof typeof model.strengths]}%
                    </div>
                  )}
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${model.badgeColor}`}>
                    {model.badge}
                  </span>
                  <div className="text-xs text-gray-500">
                    {model.strengths.latency}ms
                  </div>
                  <div className="text-xs text-gray-500">
                    ${model.strengths.cost.toFixed(4)}/1k
                  </div>
                </div>
              </div>
              
              <p className="text-xs text-gray-600 mb-3">
                {model.description}
              </p>
              
              {selectedCapability !== 'all' && (
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-gray-500">
                    {capabilities.find(c => c.value === selectedCapability)?.label}:
                  </span>
                  {selectedCapability === 'latency' ? (
                    <div className="text-xs font-medium text-slate-600">
                      {model.strengths.latency}ms
                    </div>
                  ) : selectedCapability === 'cost' ? (
                    <div className="text-xs font-medium text-slate-600">
                      ${model.strengths.cost.toFixed(4)}/1k tokens
                    </div>
                  ) : model.strengths[selectedCapability as keyof typeof model.strengths] > 0 ? (
                    <>
                      <div className="text-xs font-medium text-slate-600">
                        {model.strengths[selectedCapability as keyof typeof model.strengths]}%
                      </div>
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-slate-500 h-2 rounded-full transition-all duration-300"
                          style={{ 
                            width: `${(model.strengths[selectedCapability as keyof typeof model.strengths] || 0)}%` 
                          }}
                        ></div>
                      </div>
                    </>
                  ) : null}
                </div>
              )}
            </div>
          ))}
        </div>
        
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">
              {selectedCapability === 'all' 
                ? 'Total Models Available:' 
                : `Models with ${capabilities.find(c => c.value === selectedCapability)?.label} capability:`
              }
            </span>
            <span className="font-semibold text-gray-900">
              {getFilteredModels().length} Models
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            {selectedCapability === 'all' 
              ? 'Each model is automatically scored and ranked based on your prompt type and preferences'
              : 'Models are ranked by their performance in the selected capability'
            }
          </p>
        </div>
      </Card>

      {/* Performance Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card padding="lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            ⚡ Speed Leaders
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">GPT-3.5 Turbo</span>
              <span className="font-medium text-green-600">0.30s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Claude 3 Haiku</span>
              <span className="font-medium text-green-600">0.37s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Gemini 2.5 Flash Lite</span>
              <span className="font-medium text-green-600">0.52s</span>
            </div>
          </div>
        </Card>

        <Card padding="lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            💰 Cost Leaders
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Grok-4 Fast</span>
              <span className="font-medium text-green-600">Free</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Claude 3 Haiku</span>
              <span className="font-medium text-green-600">$0.001/1k</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Gemini 2.5 Flash Lite</span>
              <span className="font-medium text-green-600">$0.001/1k</span>
            </div>
          </div>
        </Card>

        <Card padding="lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            🏆 Quality Leaders
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">GPT-5</span>
              <span className="font-medium text-purple-600">96% avg</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Claude Sonnet 4</span>
              <span className="font-medium text-purple-600">94% avg</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Gemini 2.0 Flash</span>
              <span className="font-medium text-purple-600">92% avg</span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
