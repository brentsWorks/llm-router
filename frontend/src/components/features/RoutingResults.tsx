import type { RouteResponse } from '../../types/api.ts';
import { ModelInfo } from './ModelInfo.tsx';
import { Card } from '../ui/Card.tsx';
import { Button } from '../ui/Button.tsx';

interface RoutingResultsProps {
  results: RouteResponse;
  onExecute?: () => void;
  isLoading?: boolean;
}

export const RoutingResults: React.FC<RoutingResultsProps> = ({
  results,
  onExecute,
  isLoading = false,
}) => {
  const formatConfidence = (confidence: number) => {
    return `${Math.round(confidence * 100)}%`;
  };

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      code: 'bg-green-100 text-green-800',
      creative: 'bg-purple-100 text-purple-800',
      qa: 'bg-blue-100 text-blue-800',
      summarization: 'bg-yellow-100 text-yellow-800',
      tool_use: 'bg-orange-100 text-orange-800',
    };
    return colors[category] || 'bg-gray-100 text-gray-800';
  };

  const getClassification = (classification: Record<string, any>) => {
    return {
      category: classification?.category || 'unknown',
      subcategory: classification?.subcategory,
      confidence: classification?.confidence || 0,
      reasoning: classification?.reasoning
    };
  };

  return (
    <div className="space-y-4">
      {/* Model Selection Metrics */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-blue-900">
            🎯 Model Selection Metrics
          </h3>
          <div className="text-xs text-blue-600">
            {results.routing_time_ms}ms routing time
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          {/* Overall Confidence */}
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-900">
              {formatConfidence(results.confidence)}
            </div>
            <div className="text-xs text-blue-600">Overall Confidence</div>
          </div>
          
          {/* Model Performance Score */}
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {((results.selected_model as any)?.score * 100 || 0).toFixed(0)}%
            </div>
            <div className="text-xs text-green-600">Model Performance</div>
          </div>
        </div>
      </div>

      {/* Classification & Selected Model - Compact */}
      <div className="space-y-3">
        {/* Classification - Inline */}
        {(() => {
          const classification = getClassification(results.classification);
          return (
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">Category:</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getCategoryColor(classification.category)}`}>
                  {classification.category}
                </span>
                <span className="text-xs text-gray-500">
                  {formatConfidence(classification.confidence)}
                </span>
              </div>
              <div className="text-xs text-gray-500">
                {results.routing_time_ms}ms
              </div>
            </div>
          );
        })()}

        {/* Selected Model */}
        <ModelInfo 
          model={results.selected_model as any} 
          isSelected={true}
          showDetails={false}
        />
        
        {/* Execute Button */}
        {onExecute && (
          <div className="pt-2">
            <Button
              variant="primary"
              onClick={onExecute}
              disabled={isLoading}
              size="sm"
              className="w-full"
            >
              {isLoading ? 'Executing...' : 'Execute with this Model'}
            </Button>
          </div>
        )}
      </div>

      {/* Alternative Models - Compact with comparison */}
      {results.fallback_models && results.fallback_models.length > 0 && (
        <div className="border-t pt-3">
          <div className="text-xs text-gray-500 mb-2">Other Options Considered:</div>
          <div className="space-y-2">
            {results.fallback_models.slice(0, 2).map((model: any, index: number) => (
              <div key={`${model.provider}-${model.model}`} className="relative">
                <div className="absolute -top-1 -right-1 z-10">
                  <span className="px-1.5 py-0.5 text-xs font-medium bg-gray-100 text-gray-600 rounded-full">
                    #{index + 2}
                  </span>
                </div>
                <ModelInfo
                  model={model}
                  isSelected={false}
                  showDetails={false}
                />
              </div>
            ))}
            {results.fallback_models.length > 2 && (
              <p className="text-xs text-gray-500 text-center">
                +{results.fallback_models.length - 2} more alternatives
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
