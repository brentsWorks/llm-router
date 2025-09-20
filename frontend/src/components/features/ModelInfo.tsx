import type { ModelInfo as ModelInfoType } from '../../types/api.ts';
import { Card } from '../ui/Card.tsx';

interface ModelInfoProps {
  model: ModelInfoType;
  isSelected?: boolean;
  showDetails?: boolean;
}

export const ModelInfo: React.FC<ModelInfoProps> = ({
  model,
  isSelected = false,
  showDetails = true,
}) => {
  const formatPrice = (price: number) => {
    if (price === 0) {
      return 'Free';
    }
    return `$${price.toFixed(4)}`;
  };

  const formatLatency = (latency: number) => {
    return `${latency}ms`;
  };

  const getQualityScore = (scores: Record<string, any>, category: string) => {
    return scores?.[category] || 0;
  };

  // Handle ModelCandidate structure from backend
  const getModelData = (model: any) => {
    return {
      provider: model?.provider || 'unknown',
      model: model?.model || 'unknown',
      score: model?.score || 0,
      estimatedCost: model?.estimated_cost || 0,
      estimatedLatency: model?.estimated_latency || 0,
      qualityMatch: model?.quality_match || 0,
      constraintViolations: model?.constraint_violations || []
    };
  };

  const modelData = getModelData(model);

  return (
    <Card 
      className={`${isSelected ? 'ring-2 ring-blue-500 bg-blue-50' : ''}`}
      padding="sm"
    >
      <div className="space-y-3">
        <div className="flex justify-between items-start">
          <div>
            <h3 className="font-semibold text-gray-900">
              {modelData.model}
            </h3>
            <p className="text-sm text-gray-600 capitalize">
              {modelData.provider}
            </p>
          </div>
          {isSelected && (
            <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
              Selected
            </span>
          )}
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Score:</span>
            <div className="font-medium flex items-center gap-1">
              {modelData.score.toFixed(3)}
              {isSelected && modelData.score > 0.8 && (
                <span className="text-xs text-green-600">⭐</span>
              )}
            </div>
            <div className="text-gray-600">
              Quality: {(modelData.qualityMatch * 100).toFixed(1)}%
            </div>
          </div>
          
          <div>
            <span className="text-gray-500">Estimates:</span>
            <div className="font-medium">
              {formatLatency(modelData.estimatedLatency)} latency
            </div>
            <div className="text-gray-600">
              {modelData.estimatedCost > 0 ? formatPrice(modelData.estimatedCost) + ' cost' : 'Free'}
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        {isSelected && (
          <div className="mt-3 pt-3 border-t border-gray-200">
            <div className="text-xs text-gray-500 mb-2">Performance Metrics:</div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-500">Quality:</span>
                <span className="font-medium">{(modelData.qualityMatch * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Speed:</span>
                <span className="font-medium">{modelData.estimatedLatency < 1000 ? 'Fast' : 'Moderate'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Cost:</span>
                <span className="font-medium">
                  {modelData.estimatedCost === 0 ? 'Free' : modelData.estimatedCost < 0.01 ? 'Low' : 'Moderate'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Constraints:</span>
                <span className="font-medium">{modelData.constraintViolations.length === 0 ? 'None' : modelData.constraintViolations.length}</span>
              </div>
            </div>
          </div>
        )}

        {showDetails && modelData.constraintViolations.length > 0 && (
          <div className="space-y-2">
            <div>
              <span className="text-sm text-gray-500">Constraint Violations:</span>
              <div className="flex flex-wrap gap-1 mt-1">
                {modelData.constraintViolations.map((violation: string, index: number) => (
                  <span
                    key={index}
                    className="px-2 py-1 text-xs bg-red-100 text-red-700 rounded"
                  >
                    {violation}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};
