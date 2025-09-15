import { useState } from 'react';
import { PromptInput, RoutingResults, PreferencesPanel, ResponseDisplay } from './components/features';
import { apiService } from './services/api.ts';
import type { RouteResponse, ExecuteResponse } from './types/api.ts';


function App() {
  const [currentPrompt, setCurrentPrompt] = useState<string>('');
  const [routingResults, setRoutingResults] = useState<RouteResponse | null>(null);
  const [executionResults, setExecutionResults] = useState<ExecuteResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preferences, setPreferences] = useState({
    cost_weight: 0.3,
    latency_weight: 0.3,
    quality_weight: 0.4,
  });

  const handlePromptSubmit = async (prompt: string) => {
    setIsLoading(true);
    setError(null);
    setCurrentPrompt(prompt);
    
    try {
      console.log('Routing prompt:', prompt, 'with preferences:', preferences);
      const results = await apiService.routePrompt({
        prompt,
        preferences: {
          cost_weight: preferences.cost_weight,
          latency_weight: preferences.latency_weight,
          quality_weight: preferences.quality_weight,
        }
      });
      setRoutingResults(results);
    } catch (err: any) {
      setError(err.message || 'Failed to route prompt');
      console.error('Routing error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePreferencesChange = (newPreferences: any) => {
    setPreferences(newPreferences);
  };

  const handleResetPreferences = () => {
    setPreferences({
      cost_weight: 0.33,
      latency_weight: 0.33,
      quality_weight: 0.34,
    });
  };

  const handleExecute = async () => {
    if (!routingResults) return;
    
    setIsExecuting(true);
    setError(null);
    
    try {
      console.log('Executing with model:', routingResults.selected_model.model);
      
      // Use the stored prompt from the routing request
      if (!currentPrompt) {
        throw new Error('No prompt available for execution');
      }
      
      const results = await apiService.executePrompt({
        prompt: currentPrompt,
        preferences: {
          cost_weight: preferences.cost_weight,
          latency_weight: preferences.latency_weight,
          quality_weight: preferences.quality_weight,
        }
      });
      setExecutionResults(results);
    } catch (err: any) {
      setError(err.message || 'Failed to execute prompt');
      console.error('Execution error:', err);
    } finally {
      setIsExecuting(false);
    }
  };

  const handleCopyResponse = () => {
    console.log('Response copied to clipboard');
    // Could add toast notification here
  };

  const handleRegenerate = async () => {
    if (!routingResults || !currentPrompt) return;
    
    setIsExecuting(true);
    setError(null);
    
    try {
      console.log('Regenerating response with model:', routingResults.selected_model.model);
      
      const results = await apiService.executePrompt({
        prompt: currentPrompt,
        preferences: {
          cost_weight: preferences.cost_weight,
          latency_weight: preferences.latency_weight,
          quality_weight: preferences.quality_weight,
        }
      });
      setExecutionResults(results);
    } catch (err: any) {
      setError(err.message || 'Failed to regenerate response');
      console.error('Regeneration error:', err);
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header - Fixed */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                LLM Router
              </h1>
              <p className="text-sm text-gray-600">
                Intelligent model selection for optimal LLM routing
              </p>
            </div>
            
            {/* Quick Status */}
            <div className="flex items-center space-x-4">
              {routingResults && (
                <div className="flex items-center space-x-3">
                  <div className="text-sm">
                    <span className="text-gray-500">Selected:</span>
                    <span className="ml-1 font-medium text-gray-900">
                      {routingResults.selected_model.model}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                      Score: {(routingResults.selected_model.score * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              )}
              {executionResults && (
                <div className="text-sm">
                  <span className="text-gray-500">Response:</span>
                  <span className="ml-1 font-medium text-green-600">
                    Ready
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Main Content - Takes up more space */}
          <div className="lg:col-span-3 space-y-4">
            {/* Prompt Input - Compact */}
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <PromptInput 
                onSubmit={handlePromptSubmit}
                isLoading={isLoading}
                placeholder="Describe what you want to accomplish with an LLM..."
              />
            </div>

            {/* Error Display - Compact */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-red-800 text-sm">{error}</p>
              </div>
            )}

            {/* Results Grid - Side by side when available */}
            {(routingResults || executionResults) && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Routing Results - Left side */}
                {routingResults && (
                  <div className="bg-white rounded-lg border border-gray-200 p-4">
                    <RoutingResults 
                      results={routingResults}
                      onExecute={handleExecute}
                      isLoading={isExecuting}
                    />
                  </div>
                )}

                {/* Execution Results - Right side */}
                {executionResults && (
                  <div className="bg-white rounded-lg border border-gray-200 p-4">
                    <ResponseDisplay
                      response={executionResults.llm_response}
                      modelUsed={executionResults.model_used}
                      executionTime={executionResults.execution_time_ms}
                      usage={executionResults.usage as { prompt_tokens: number; completion_tokens: number; total_tokens: number; } | undefined}
                      finishReason={executionResults.finish_reason}
                      onCopy={handleCopyResponse}
                      onRegenerate={handleRegenerate}
                      isLoading={isExecuting}
                    />
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Preferences Sidebar - Compact */}
          <div className="lg:col-span-1">
            <div className="sticky top-24">
              <PreferencesPanel
                preferences={preferences}
                onPreferencesChange={handlePreferencesChange}
                onReset={handleResetPreferences}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
