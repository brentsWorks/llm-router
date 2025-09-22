// API Service for LLM Router Backend

import type { 
  RouteRequest, 
  RouteResponse, 
  ExecuteResponse, 
  HealthResponse, 
  ModelsResponse 
} from '../types/api.ts';
import { API_BASE_URL } from '../constants/index.ts';

class ApiError extends Error {
  status: number;
  response?: any;

  constructor(message: string, status: number, response?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.response = response;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
      response.status,
      errorData
    );
  }
  return response.json();
}

export class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async getHealth(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/monitoring/health`);
    return handleResponse<HealthResponse>(response);
  }

  async getModels(): Promise<ModelsResponse> {
    const response = await fetch(`${this.baseUrl}/models`);
    return handleResponse<ModelsResponse>(response);
  }

  async routePrompt(request: RouteRequest): Promise<RouteResponse> {
    const response = await fetch(`${this.baseUrl}/route`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    return handleResponse<RouteResponse>(response);
  }

  async executePrompt(request: RouteRequest): Promise<ExecuteResponse> {
    const response = await fetch(`${this.baseUrl}/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    return handleResponse<ExecuteResponse>(response);
  }
}

// Export singleton instance
export const apiService = new ApiService();
export { ApiError };
