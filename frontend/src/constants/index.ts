// Application constants

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const API_ENDPOINTS = {
  ROUTE: '/route',
  EXECUTE: '/execute',
  HEALTH: '/health',
  MODELS: '/models',
  CLASSIFIER: '/classifier',
  METRICS: '/metrics'
} as const

export const PROMPT_CATEGORIES = [
  { value: 'code', label: 'Code Generation', color: 'blue' },
  { value: 'creative', label: 'Creative Writing', color: 'purple' },
  { value: 'qa', label: 'Question Answering', color: 'green' },
  { value: 'analysis', label: 'Analysis', color: 'orange' },
  { value: 'summarization', label: 'Summarization', color: 'teal' },
  { value: 'reasoning', label: 'Reasoning', color: 'indigo' },
  { value: 'translation', label: 'Translation', color: 'pink' },
  { value: 'tool_use', label: 'Tool Use', color: 'gray' }
] as const

export const SAFETY_LEVELS = [
  { value: 'low', label: 'Low', description: 'Minimal safety restrictions' },
  { value: 'medium', label: 'Medium', description: 'Balanced safety measures' },
  { value: 'high', label: 'High', description: 'Strict safety guidelines' },
  { value: 'strict', label: 'Strict', description: 'Maximum safety enforcement' }
] as const

export const DEFAULT_PREFERENCES = {
  cost_weight: 0.3,
  latency_weight: 0.3,
  quality_weight: 0.4
} as const

export const DEFAULT_CONSTRAINTS = {
  max_cost_per_1k_tokens: 0.1,
  max_latency_ms: 5000,
  max_context_length: 4000,
  min_safety_level: 'low' as const
} as const

export const TOAST_DURATION = {
  SUCCESS: 3000,
  ERROR: 5000,
  WARNING: 4000,
  INFO: 3000
} as const

export const ANIMATION_DURATION = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500
} as const
