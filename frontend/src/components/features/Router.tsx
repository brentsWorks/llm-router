import { useState } from 'react';
import { PromptInput, RoutingResults, PreferencesPanel, ResponseDisplay } from './index';
import { apiService } from '../../services/api.ts';
import type { RouteResponse, ExecuteResponse } from '../../types/api.ts';

interface RouterProps {
  preferences: any;
  onPreferencesChange: (preferences: any) => void;
  onResetPreferences: () => void;
  onStatusChange: (status: {
    state: 'ready' | 'routing' | 'routed' | 'executing' | 'executed' | 'error';
    message: string;
  }) => void;
}

export function Router({ preferences, onPreferencesChange, onResetPreferences, onStatusChange }: RouterProps) {
  const [currentPrompt, setCurrentPrompt] = useState<string>('');
  const [routingResults, setRoutingResults] = useState<RouteResponse | null>(null);
  const [executionResults, setExecutionResults] = useState<ExecuteResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePromptSubmit = async (prompt: string) => {
    setIsLoading(true);
    setError(null);
    setCurrentPrompt(prompt);
    onStatusChange({ state: 'routing', message: 'Finding optimal model...' });
    
    try {
      console.log('Routing prompt:', prompt, 'with preferences:', preferences);
      const results = await apiService.routePrompt({
        prompt,
        preferences: {
          cost_weight: preferences.cost_weight,
          latency_weight: preferences.latency_weight,
          quality_weight: preferences.quality_weight,
        },
        constraints: {
          max_cost_per_1k_tokens: preferences.max_cost_per_1k_tokens,
          max_latency_ms: preferences.max_latency_ms,
          max_context_length: preferences.max_context_length,
          min_safety_level: preferences.min_safety_level,
          excluded_providers: preferences.excluded_providers,
          excluded_models: preferences.excluded_models,
        }
      });
      setRoutingResults(results);
      onStatusChange({ 
        state: 'routed', 
        message: `${results.selected_model.provider}/${results.selected_model.model} selected` 
      });
    } catch (err: any) {
      setError(err.message || 'Failed to route prompt');
      onStatusChange({ state: 'error', message: 'Routing failed' });
      console.error('Routing error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExecute = async () => {
    if (!routingResults) return;
    
    setIsExecuting(true);
    setError(null);
    onStatusChange({ state: 'executing', message: 'Generating response...' });
    
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
        },
        constraints: {
          max_cost_per_1k_tokens: preferences.max_cost_per_1k_tokens,
          max_latency_ms: preferences.max_latency_ms,
          max_context_length: preferences.max_context_length,
          min_safety_level: preferences.min_safety_level,
          excluded_providers: preferences.excluded_providers,
          excluded_models: preferences.excluded_models,
        }
      });
      setExecutionResults(results);
      onStatusChange({ state: 'executed', message: 'Response generated successfully' });
    } catch (err: any) {
      setError(err.message || 'Failed to execute prompt');
      onStatusChange({ state: 'error', message: 'Execution failed' });
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
    onStatusChange({ state: 'executing', message: 'Regenerating response...' });
    
    try {
      console.log('Regenerating response with model:', routingResults.selected_model.model);
      
      const results = await apiService.executePrompt({
        prompt: currentPrompt,
        preferences: {
          cost_weight: preferences.cost_weight,
          latency_weight: preferences.latency_weight,
          quality_weight: preferences.quality_weight,
        },
        constraints: {
          max_cost_per_1k_tokens: preferences.max_cost_per_1k_tokens,
          max_latency_ms: preferences.max_latency_ms,
          max_context_length: preferences.max_context_length,
          min_safety_level: preferences.min_safety_level,
          excluded_providers: preferences.excluded_providers,
          excluded_models: preferences.excluded_models,
        }
      });
      setExecutionResults(results);
      onStatusChange({ state: 'executed', message: 'Response regenerated successfully' });
    } catch (err: any) {
      setError(err.message || 'Failed to regenerate response');
      onStatusChange({ state: 'error', message: 'Regeneration failed' });
      console.error('Regeneration error:', err);
    } finally {
      setIsExecuting(false);
    }
  };

  return (
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
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {/* Routing Results - Left side */}
            {routingResults && (
              <div className="bg-white rounded-lg border border-gray-200 p-3 sm:p-4">
                <RoutingResults 
                  results={routingResults}
                  onExecute={handleExecute}
                  isLoading={isExecuting}
                />
              </div>
            )}

            {/* Execution Results - Right side */}
            {executionResults && (
              <div className="bg-white rounded-lg border border-gray-200 p-3 sm:p-4">
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
            onPreferencesChange={onPreferencesChange}
            onReset={onResetPreferences}
          />
        </div>
      </div>
    </div>
  );
}
