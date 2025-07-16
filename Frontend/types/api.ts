/**
 * API Response Types
 */

export interface ForecastData {
  date: string;
  value: number;
  lower_bound?: number;
  upper_bound?: number;
}

export interface ForecastResponse {
  forecast: ForecastData[];
  metrics?: {
    mape: number;
    rmse: number;
    accuracy: number;
  };
  plot_url?: string;
}

export interface OptimizationResponse {
  optimal_inventory_levels: {
    date: string;
    level: number;
  }[];
  reorder_points: {
    product: string;
    reorder_point: number;
  }[];
  economic_order_quantities: {
    product: string;
    eoq: number;
  }[];
  total_cost: number;
  plot_url?: string;
}

export interface QAResponse {
  answer: string;
  confidence: number;
  sources?: {
    title: string;
    relevance: number;
  }[];
}

export interface Anomaly {
  date: string;
  column: string;
  value: number;
  expected_range: [number, number];
  severity: 'low' | 'medium' | 'high';
}

export interface AnomalyResponse {
  anomalies: Anomaly[];
  summary: string;
  plot_url?: string;
  product_id?: string;
  analysis?: string;
  anomaly_count?: number;
  chart_data?: any[];
}

export interface ScenarioResult {
  name: string;
  kpis: {
    stockout_rate: number;
    inventory_cost: number;
    service_level: number;
    [key: string]: number;
  };
  recommended_actions?: string[];
}

export interface ScenarioResponse {
  scenarios_results: ScenarioResult[];
  comparative_analysis: string;
  plot_url?: string;
  product_id?: string;
  scenario?: string;
  description?: string;
  impact?: string;
  chart_data?: any[];
}

export interface ReportResponse {
  report_url: string;
  sections_included: string[];
  summary: string;
}

export interface DocumentResponse {
  status: 'success' | 'error';
  message: string;
  document_id?: string;
  chunks_extracted?: number;
  vector_db_updated?: boolean;
}

export interface AudioTranscriptionResponse {
  status: 'success' | 'error';
  transcription?: string;
  response?: string;
  speech_file?: string;
  language_detected?: string;
  confidence?: number;
  error?: string;
}

export interface TunnelStatusResponse {
  status: 'running' | 'stopped';
  url?: string;
}

export interface TunnelStartResponse {
  status: 'started' | 'already_running' | 'error';
  message: string;
  url?: string;
}

export interface TunnelStopResponse {
  status: 'stopped' | 'not_running';
  message: string;
}

export interface TunnelTestResponse {
  status: 'success' | 'error';
  message: string;
  timestamp: string;
  tunnel_status: 'running' | 'stopped';
  tunnel_url?: string;
}

/**
 * Language Type
 */
export type Language = 'en' | 'fr' | 'ar';

/**
 * Scenario Type
 */
export type ScenarioType = 
  | 'demand_increase'
  | 'demand_decrease'
  | 'supply_disruption'
  | 'weather_extreme'
  | 'cost_increase'
  | 'new_competitor'
  | 'marketing_campaign'
  | 'seasonal_peak';

/**
 * Report Type
 */
export type ReportType = 'forecast' | 'inventory' | 'anomalies' | 'recommendations';
