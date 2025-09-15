import { useState, useEffect } from 'react';
import { Card } from '../ui/Card.tsx';
import { Slider } from '../ui/Slider.tsx';
import { Button } from '../ui/Button.tsx';

interface Preferences {
  cost_weight: number;
  latency_weight: number;
  quality_weight: number;
  max_cost_per_1k_tokens?: number;
  max_latency_ms?: number;
  max_context_length?: number;
  min_safety_level?: string;
  excluded_providers?: string[];
  excluded_models?: string[];
}

interface PreferencesPanelProps {
  preferences: Preferences;
  onPreferencesChange: (preferences: Preferences) => void;
  onReset: () => void;
  className?: string;
}

const SAFETY_LEVELS = [
  { value: 'low', label: 'Low' },
  { value: 'moderate', label: 'Moderate' },
  { value: 'high', label: 'High' },
];

const PROVIDERS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'google', label: 'Google' },
  { value: 'cohere', label: 'Cohere' },
];

export const PreferencesPanel: React.FC<PreferencesPanelProps> = ({
  preferences,
  onPreferencesChange,
  onReset,
  className = '',
}) => {
  const [localPreferences, setLocalPreferences] = useState<Preferences>(preferences);
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    setLocalPreferences(preferences);
  }, [preferences]);

  const updatePreference = (key: keyof Preferences, value: any) => {
    const updated = { ...localPreferences, [key]: value };
    setLocalPreferences(updated);
    onPreferencesChange(updated);
  };

  const normalizeWeights = () => {
    const { cost_weight, latency_weight, quality_weight } = localPreferences;
    const total = cost_weight + latency_weight + quality_weight;
    
    if (total === 0) return { cost_weight: 0.33, latency_weight: 0.33, quality_weight: 0.34 };
    
    return {
      cost_weight: cost_weight / total,
      latency_weight: latency_weight / total,
      quality_weight: quality_weight / total,
    };
  };

  const normalizedWeights = normalizeWeights();

  const handleWeightChange = (key: 'cost_weight' | 'latency_weight' | 'quality_weight', value: number) => {
    const otherKeys = ['cost_weight', 'latency_weight', 'quality_weight'].filter(k => k !== key) as Array<'cost_weight' | 'latency_weight' | 'quality_weight'>;
    const otherTotal = otherKeys.reduce((sum, k) => sum + localPreferences[k], 0);
    const remaining = 1 - value;
    
    if (remaining <= 0) return;
    
    const updated = { ...localPreferences };
    updated[key] = value;
    
    // Distribute remaining weight proportionally
    otherKeys.forEach(k => {
      updated[k] = (localPreferences[k] / otherTotal) * remaining;
    });
    
    setLocalPreferences(updated);
    onPreferencesChange(updated);
  };

  const toggleProvider = (provider: string) => {
    const excluded = localPreferences.excluded_providers || [];
    const updated = excluded.includes(provider)
      ? excluded.filter(p => p !== provider)
      : [...excluded, provider];
    
    updatePreference('excluded_providers', updated);
  };

  return (
    <Card 
      title="Routing Preferences" 
      className={className}
      padding="sm"
    >
      <div className="space-y-4">
        {/* Weight Controls */}
        <div className="space-y-4">
          <h4 className="font-medium text-gray-900">Optimization Weights</h4>
          <p className="text-sm text-gray-600">
            Adjust how much each factor influences model selection (weights must sum to 1.0)
          </p>
          
          <div className="space-y-3">
            <Slider
              label={`Cost Optimization: ${normalizedWeights.cost_weight.toFixed(2)}`}
              value={normalizedWeights.cost_weight}
              onChange={(value) => handleWeightChange('cost_weight', value)}
              min={0}
              max={1}
              step={0.05}
            />
            
            <Slider
              label={`Latency Optimization: ${normalizedWeights.latency_weight.toFixed(2)}`}
              value={normalizedWeights.latency_weight}
              onChange={(value) => handleWeightChange('latency_weight', value)}
              min={0}
              max={1}
              step={0.05}
            />
            
            <Slider
              label={`Quality Optimization: ${normalizedWeights.quality_weight.toFixed(2)}`}
              value={normalizedWeights.quality_weight}
              onChange={(value) => handleWeightChange('quality_weight', value)}
              min={0}
              max={1}
              step={0.05}
            />
          </div>
        </div>

        {/* Advanced Settings */}
        <div className="border-t pt-4">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center justify-between w-full text-left font-medium text-gray-900 hover:text-gray-700"
          >
            <span>Advanced Constraints</span>
            <span className={`transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}>
              ▼
            </span>
          </button>
          
          {isExpanded && (
            <div className="mt-4 space-y-4">
              {/* Cost Constraint */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Cost per 1K Tokens ($)
                </label>
                <input
                  type="number"
                  value={localPreferences.max_cost_per_1k_tokens || ''}
                  onChange={(e) => updatePreference('max_cost_per_1k_tokens', e.target.value ? parseFloat(e.target.value) : undefined)}
                  placeholder="No limit"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  step="0.001"
                  min="0"
                />
              </div>

              {/* Latency Constraint */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Latency (ms)
                </label>
                <input
                  type="number"
                  value={localPreferences.max_latency_ms || ''}
                  onChange={(e) => updatePreference('max_latency_ms', e.target.value ? parseInt(e.target.value) : undefined)}
                  placeholder="No limit"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  min="0"
                />
              </div>

              {/* Context Length Constraint */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Min Context Length
                </label>
                <input
                  type="number"
                  value={localPreferences.max_context_length || ''}
                  onChange={(e) => updatePreference('max_context_length', e.target.value ? parseInt(e.target.value) : undefined)}
                  placeholder="No limit"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  min="0"
                />
              </div>

              {/* Safety Level */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Min Safety Level
                </label>
                <div className="space-y-2">
                  {SAFETY_LEVELS.map((level) => (
                    <label key={level.value} className="flex items-center">
                      <input
                        type="radio"
                        name="safety_level"
                        value={level.value}
                        checked={localPreferences.min_safety_level === level.value}
                        onChange={(e) => updatePreference('min_safety_level', e.target.value)}
                        className="mr-2"
                      />
                      <span className="text-sm text-gray-700">{level.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Provider Exclusions */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Exclude Providers
                </label>
                <div className="space-y-2">
                  {PROVIDERS.map((provider) => (
                    <label key={provider.value} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={(localPreferences.excluded_providers || []).includes(provider.value)}
                        onChange={() => toggleProvider(provider.value)}
                        className="mr-2"
                      />
                      <span className="text-sm text-gray-700">{provider.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="pt-4 border-t space-y-3">
          <div className="flex justify-center">
            <Button
              variant="outline"
              onClick={onReset}
              size="sm"
            >
              Reset to Defaults
            </Button>
          </div>
          
          <div className="text-center text-xs text-gray-500">
            Weights: {normalizedWeights.cost_weight.toFixed(2)} + {normalizedWeights.latency_weight.toFixed(2)} + {normalizedWeights.quality_weight.toFixed(2)} = {(normalizedWeights.cost_weight + normalizedWeights.latency_weight + normalizedWeights.quality_weight).toFixed(2)}
          </div>
        </div>
      </div>
    </Card>
  );
};
