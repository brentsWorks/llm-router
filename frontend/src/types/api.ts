// API Types for LLM Router - matches backend API structure

export interface RouteRequest {
  prompt: string;
  preferences?: {
    cost_weight?: number;
    latency_weight?: number;
    quality_weight?: number;
  };
  constraints?: {
    max_cost_per_1k_tokens?: number;
    max_latency_ms?: number;
    max_context_length?: number;
    min_safety_level?: string;
    excluded_providers?: string[];
    excluded_models?: string[];
  };
}

// Backend uses Dict[str, Any] for most fields, so we'll use flexible types
export interface ModelInfo {
  provider: string;
  model: string;
  capabilities: string[];
  pricing: Record<string, any>;
  limits: Record<string, any>;
  performance: Record<string, any>;
}

export interface RouteResponse {
  selected_model: Record<string, any>;
  classification: Record<string, any>;
  confidence: number;
  routing_time_ms: number;
  reasoning?: string;
  provider_info?: Record<string, any>;
  fallback_models?: Record<string, any>[];
}

export interface ExecuteResponse {
  selected_model: Record<string, any>;
  classification: Record<string, any>;
  confidence: number;
  routing_time_ms: number;
  reasoning?: string;
  llm_response: string;
  model_used: string;
  execution_time_ms: number;
  usage?: Record<string, any>;
  finish_reason?: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
  total_models: number;
  total_requests: number;
}

export interface ModelsResponse {
  models: ModelInfo[];
  total_count: number;
}

export interface ClassifyRequest {
  prompt: string;
}

export interface ClassifyResponse {
  category: string;
  confidence: number;
  reasoning?: string;
  classification_time_ms: number;
}
