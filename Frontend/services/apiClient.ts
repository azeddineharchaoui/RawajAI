import Constants from "expo-constants"
import { Platform } from "react-native"
import { secureStorage } from "../utils/secure-storage"

// Default API URL - for local dev testing
const DEFAULT_API_URL = 'https://reliable-infrared-plc-bl.trycloudflare.com';
const API_URL = Constants.expoConfig?.extra?.apiUrl || DEFAULT_API_URL;
const TUNNEL_KEY = 'tunnel_url'; // Must be a simple string key for SecureStore

// Type definitions for API requests and responses
export interface ForecastParams {
  product_id: string
  days?: number
  language?: string
}

export interface OptimizeInventoryParams {
  forecast_data: any[]
  product_id: string
  holding_cost: number
  ordering_cost: number
  lead_time: number
  service_level: number
  language?: string
}

export interface QuestionParams {
  query: string
  language?: string
}

export interface AnomalyDetectionParams {
  product_id: string
  language?: string
}

export interface ScenarioAnalysisParams {
  product_id: string
  scenario: string
  language?: string
}

export interface ReportParams {
  report_type: "forecast" | "inventory"
  product_id?: string
  language?: string
}

export interface DocumentParams {
  document: string
  language?: string
}

export interface AudioUploadParams {
  uri: string
  name: string
  type: string
  language?: string
}

/**
 * Storage Functions for Cloudflare Tunnel URL - FIXED VERSION
 */
export const saveTunnelUrl = async (url: string) => {
  try {
    await secureStorage.setItemAsync(TUNNEL_KEY, url)
    return true
  } catch (error) {
    console.error("Failed to save tunnel URL:", error)
    return false
  }
}

export const getTunnelUrl = async (): Promise<string | null> => {
  try {
    return await secureStorage.getItemAsync(TUNNEL_KEY)
  } catch (error) {
    console.error("Failed to get tunnel URL:", error)
    return null
  }
}

export const clearTunnelUrl = async () => {
  try {
    await secureStorage.deleteItemAsync(TUNNEL_KEY)
    return true
  } catch (error) {
    console.error("Failed to clear tunnel URL:", error)
    return false
  }
}

/**
 * API Base Functions
 */
const getBaseUrl = async () => {
  // Try to use the tunnel URL first
  const tunnelUrl = await getTunnelUrl()
  if (tunnelUrl && tunnelUrl.startsWith("http")) {
    return tunnelUrl
  }

  // For Android emulators, localhost refers to the emulator itself, not your machine
  if (Platform.OS === "android") {
    return API_URL.replace("localhost", "10.0.2.2")
  }

  return API_URL
}

/**
 * API request function with improved timeout, retry functionality, and error handling
 */
const fetchWithTimeout = async (url: string, options: RequestInit = {}, timeout = 10000) => {
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), timeout)

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    })

    clearTimeout(id)
    return response
  } catch (error) {
    clearTimeout(id)

    if (error instanceof Error && error.name === "AbortError") {
      console.warn(`Request to ${url} timed out after ${timeout}ms`)
      throw new Error(`Request timed out after ${timeout}ms. The server might be overloaded.`)
    }
    
    // Enhanced error detection for specific connection issues
    if (error instanceof TypeError && error.message.includes('Network request failed')) {
      if (url.includes('trycloudflare.com')) {
        console.warn('Cloudflare tunnel connection failed - the tunnel may have expired');
        // Clear the stored tunnel URL to force fallback to default URL on next request
        await clearTunnelUrl();
        throw new Error('Connection to Cloudflare tunnel failed. The tunnel may have expired or been closed. Please restart the tunnel from the settings screen.');
      }
    }
    
    // Re-throw other errors
    throw error;
  }
}

/**
 * API client class - COMPLETE VERSION WITH ALL YOUR METHODS
 */
export class ApiClient {
  /**
   * Base API request method
   */
  async request(
    endpoint: string,
    method = "GET",
    params: Record<string, any> = {},
    body: Record<string, any> = {},
    isFormData = false,
  ) {
    try {
      const baseUrl = await getBaseUrl()
      const url = new URL(`${baseUrl}${endpoint}`)

      // Add query parameters for GET requests
      if (method === "GET" && Object.keys(params).length > 0) {
        Object.keys(params).forEach((key) => {
          if (params[key] !== undefined && params[key] !== null) {
            url.searchParams.append(key, String(params[key]))
          }
        })
      }

      // Prepare headers
      const headers: HeadersInit = {}
      if (!isFormData) {
        headers["Content-Type"] = "application/json"
      }

      // Prepare options
      const options: RequestInit = {
        method,
        headers,
      }

      // Add body for non-GET requests
      if (method !== "GET" && Object.keys(body).length > 0) {
        options.body = isFormData ? (body as unknown as FormData) : JSON.stringify(body)
      }

      // Choose appropriate timeout based on endpoint
      let timeout = 10000 // Default 10 seconds
      if (endpoint === "/ask" || endpoint === "/generate_report") {
        timeout = 30000 // 30 seconds for AI-heavy operations
      }

      console.log(`API ${method} Request to ${url.toString()}`, method !== "GET" ? body : params)

      // Make the request
      const response = await fetchWithTimeout(url.toString(), options, timeout)

      // Handle error responses
      if (!response.ok) {
        let errorMessage = `API Error: ${response.status} ${response.statusText}`
        try {
          const errorData = await response.json()
          if (errorData.error || errorData.message) {
            errorMessage = errorData.error || errorData.message
          }
        } catch {
          // Ignore JSON parsing errors for error responses
        }
        throw new Error(errorMessage)
      }

      // Special case for binary responses (like PDFs)
      const contentType = response.headers.get("content-type") || ""
      if (contentType.includes("application/pdf")) {
        return { blob: await response.blob(), url: response.url }
      }

      // Parse JSON response or return empty object if it fails
      try {
        const data = await response.json()
        return data
      } catch {
        console.warn("Failed to parse JSON response, returning empty object")
        return {}
      }
    } catch (error) {
      console.error(`API request failed:`, error)
      throw error
    }
  }

  /**
   * Tunnel Management API methods
   */
  async getTunnelStatus() {
    try {
      return await this.request("/tunnel/status")
    } catch (error) {
      return { status: "offline", error: error instanceof Error ? error.message : "Unknown error" }
    }
  }

  async startTunnel() {
    return this.request("/tunnel/start", "POST")
  }

  async stopTunnel() {
    return this.request("/tunnel/stop", "POST")
  }

  async testTunnel() {
    return this.request("/tunnel/test")
  }

  /**
   * Supply Chain API Methods
   */
  async getForecast(data: ForecastParams) {
    return this.request("/forecast", "POST", {}, data)
  }

  async optimizeInventory(data: OptimizeInventoryParams) {
    try {
      const demand_forecast: Record<string, number> = {}
      if (Array.isArray(data.forecast_data)) {
        data.forecast_data.forEach((item) => {
          if (item.product_id && item.demand) {
            demand_forecast[item.product_id] = item.demand
          }
        })
      }

      const apiData = {
        demand_forecast: demand_forecast,
        product_id: data.product_id,
        holding_costs: { [data.product_id]: data.holding_cost },
        service_level: data.service_level,
        lead_times: { [data.product_id]: data.lead_time },
        language: data.language || "en",
      }

      const result = await this.request("/optimize", "POST", {}, apiData)

      if (!result || !result.optimization_results) {
        console.warn("API didn't return expected format, using mock data")
        return this.getMockInventoryData(data.product_id)
      }

      return result
    } catch (error) {
      console.error("Inventory optimization error:", error)
      return this.getMockInventoryData(data.product_id)
    }
  }

  async askQuestion(query: string, language = "en") {
    try {
      const normalizedQuery = query.trim().slice(0, 500)

      if (!normalizedQuery) {
        return {
          query: query,
          response: "Please provide a question.",
          language: language,
        }
      }

      const result = await this.request("/ask", "POST", {}, { query: normalizedQuery, language })
      return result
    } catch (error) {
      console.error("Error asking question:", error)

      let errorMessage = "Sorry, I couldn't connect to the AI backend. Please try again later."

      if (error instanceof Error) {
        if (error.message.includes("timed out") || error.message.includes("AbortError")) {
          errorMessage = "The AI is taking too long to respond. Please try a simpler question or try again later."
        } else if (error.message.includes("Network") || error.message.includes("Failed to fetch")) {
          errorMessage = "Network connection issue. Please check your internet connection and try again."
        } else if (error.message.includes("ERR_NAME_NOT_RESOLVED")) {
          errorMessage = "The server URL is not accessible. Please check your tunnel configuration."
        }
      }

      return {
        query: query,
        response: errorMessage,
        language: language,
        error: error instanceof Error ? error.message : "Unknown error",
      }
    }
  }

  async detectAnomalies(data: AnomalyDetectionParams) {
    return this.request("/anomaly_detection", "POST", {}, data)
  }

  async analyzeScenario(data: ScenarioAnalysisParams) {
    return this.request("/scenario_analysis", "POST", {}, data)
  }

  async generateReport(data: ReportParams) {
    try {
      if (data.report_type === "forecast" && !data.product_id) {
        data.product_id = "smartphone"
      }

      const response = await this.request("/generate_report", "POST", {}, data)

      if (!response || !response.report_url) {
        throw new Error("No report URL in response")
      }

      const baseUrl = await getBaseUrl()
      let reportUrl = response.report_url

      if (!reportUrl.startsWith("/")) {
        reportUrl = "/" + reportUrl
      }

      const fullReportUrl = `${baseUrl}${reportUrl}`

      return {
        ...response,
        success: true,
        report_url: fullReportUrl,
        content_type: "application/pdf",
      }
    } catch (error) {
      console.error("Report generation error:", error)

      const baseUrl = await getBaseUrl()
      const mockReportName = `mock_${data.report_type}_report.pdf`
      const mockReportUrl = `${baseUrl}/reports/${mockReportName}`

      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
        report_url: mockReportUrl,
        content_type: "application/pdf",
        mock_url: true,
      }
    }
  }

  async addDocument(document: string, language = "en") {
    return this.request("/add_document", "POST", {}, { document, language })
  }

  async uploadAudio(audioFile: AudioUploadParams) {
    const formData = new FormData()

    formData.append("audio", {
      uri: audioFile.uri,
      name: audioFile.name || "audio.m4a",
      type: audioFile.type || "audio/m4a",
    } as any)

    if (audioFile.language) {
      formData.append("language", audioFile.language)
    }

    return this.request("/upload_audio", "POST", {}, formData as unknown as Record<string, any>, true)
  }
  
  /**
   * Fetches all products with their inventory levels from the backend
   */
  async getProducts() {
    return this.request('/products');
  }
  
  private getMockInventoryData(productId: string = '1001') {
    // Create mock data with the expected structure
    return {
      optimization_results: {
        status: "Optimal",
        total_cost: 12500,
        locations: {
          warehouse_a: {
            total_inventory: 450,
            capacity: 1000,
            capacity_utilization: 45,
            products: {
              smartphone: 150,
              laptop: 100,
              tablet: 100,
              headphones: 50,
              smartwatch: 50,
            },
          },
          warehouse_b: {
            total_inventory: 800,
            capacity: 1500,
            capacity_utilization: 53.3,
            products: {
              smartphone: 300,
              laptop: 150,
              tablet: 150,
              headphones: 100,
              smartwatch: 100,
            },
          },
          warehouse_c: {
            total_inventory: 600,
            capacity: 1200,
            capacity_utilization: 50,
            products: {
              smartphone: 200,
              laptop: 120,
              tablet: 120,
              headphones: 80,
              smartwatch: 80,
            },
          },
        },
      },
      recommendations:
        "Redistribute inventory to optimize warehouse space utilization.\nConsider increasing capacity for warehouses exceeding 85% utilization.\nReduce inventory levels for slow-moving products.\nPrioritize high-demand products in warehouses closer to customers.",
      chart_data: [
        {
          x: ["warehouse_a", "warehouse_b", "warehouse_c"],
          y: [450, 800, 600],
          type: "bar",
          name: "Total Inventory",
        },
        {
          x: ["warehouse_a", "warehouse_b", "warehouse_c"],
          y: [45, 53.3, 50],
          type: "scatter",
          mode: "lines+markers",
          name: "Utilization %",
          yaxis: "y2",
        },
      ],
    }
  }
}

// Export a singleton instance of the API client
export const api = new ApiClient()
