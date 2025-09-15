import { useState } from 'react';
import { Button } from '../ui/Button.tsx';
import { Card } from '../ui/Card.tsx';

interface PromptInputProps {
  onSubmit: (prompt: string) => void;
  isLoading?: boolean;
  placeholder?: string;
}

export const PromptInput: React.FC<PromptInputProps> = ({
  onSubmit,
  isLoading = false,
  placeholder = "Enter your prompt here...",
}) => {
  const [prompt, setPrompt] = useState('');
  const [error, setError] = useState('');

  const validatePrompt = (text: string): string => {
    if (!text.trim()) {
      return 'Prompt cannot be empty';
    }
    if (text.length < 10) {
      return 'Prompt must be at least 10 characters long';
    }
    if (text.length > 10000) {
      return 'Prompt must be less than 10,000 characters';
    }
    return '';
  };

  const handleSubmit = () => {
    const validationError = validatePrompt(prompt);
    if (validationError) {
      setError(validationError);
      return;
    }
    
    setError('');
    onSubmit(prompt.trim());
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleChange = (value: string) => {
    setPrompt(value);
    if (error) {
      setError('');
    }
  };

  return (
    <Card title="Enter Your Prompt">
      <div className="space-y-4">
        <div>
          <textarea
            value={prompt}
            onChange={(e) => handleChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={isLoading}
            className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none ${
              error ? 'border-red-500' : 'border-gray-300'
            } ${isLoading ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}`}
            rows={6}
          />
          {error && (
            <p className="mt-1 text-sm text-red-600">{error}</p>
          )}
        </div>
        
        <div className="flex justify-between items-center">
          <div className="text-sm text-gray-500">
            {prompt.length}/10,000 characters
            {prompt.length > 0 && (
              <span className={`ml-2 ${prompt.length < 10 ? 'text-red-500' : 'text-green-500'}`}>
                {prompt.length < 10 ? 'Too short' : 'Valid'}
              </span>
            )}
          </div>
          
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => setPrompt('')}
              disabled={isLoading || !prompt}
            >
              Clear
            </Button>
            <Button
              variant="primary"
              onClick={handleSubmit}
              disabled={isLoading || !prompt.trim()}
            >
              {isLoading ? 'Processing...' : 'Route Prompt'}
            </Button>
          </div>
        </div>
        
        <div className="text-xs text-gray-400">
          Tip: Press Ctrl+Enter (or Cmd+Enter) to submit quickly
        </div>
      </div>
    </Card>
  );
};
