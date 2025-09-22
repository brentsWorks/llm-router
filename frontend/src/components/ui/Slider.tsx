import React from 'react';

interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  disabled?: boolean;
  className?: string;
}

export const Slider: React.FC<SliderProps> = ({
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.1,
  label,
  disabled = false,
  className = '',
}) => {
  const percentage = ((value - min) / (max - min)) * 100;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    let newValue = parseFloat(e.target.value);
    
    // Snap to edges for easier 0/1 selection
    const snapThreshold = 0.05;
    if (newValue <= min + snapThreshold) {
      newValue = min;
    } else if (newValue >= max - snapThreshold) {
      newValue = max;
    }
    
    onChange(newValue);
  };

  return (
    <div className={`w-full ${className}`}>
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {label}
        </label>
      )}
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleChange}
          disabled={disabled}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
          style={{
            background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${percentage}%, #e5e7eb ${percentage}%, #e5e7eb 100%)`
          }}
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span className={`${value === min ? 'font-bold text-blue-600' : ''}`}>
            {min === 0 ? '0% (Off)' : min}
          </span>
          <span className="font-medium">0.5</span>
          <span className={`${value === max ? 'font-bold text-blue-600' : ''}`}>
            {max === 1 ? '100% (Only)' : max}
          </span>
        </div>
      </div>
    </div>
  );
};
