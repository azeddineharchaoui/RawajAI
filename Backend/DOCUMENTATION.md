# RawajAI Supply Chain Management System - Technical Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Dependencies](#dependencies)
4. [Configuration](#configuration)
5. [Core Classes](#core-classes)
   - [ForecastingEngine](#forecastingengine)
   - [InventoryOptimizer](#inventoryoptimizer)
   - [VisualizationEngine](#visualizationengine)
   - [ReportGenerator](#reportgenerator)
   - [CloudflaredTunnel](#cloudflaredtunnel)
6. [API Endpoints](#api-endpoints)
   - [Forecasting APIs](#forecasting-apis)
   - [Inventory Optimization APIs](#inventory-optimization-apis)
   - [Document Management APIs](#document-management-apis)
   - [Natural Language APIs](#natural-language-apis)
   - [Audio Processing APIs](#audio-processing-apis)
   - [Reporting APIs](#reporting-apis)
   - [Tunnel Management APIs](#tunnel-management-apis)
7. [Helper Functions](#helper-functions)
8. [Data Management](#data-management)

## Introduction

The RawajAI Supply Chain Management System is a comprehensive platform that leverages advanced machine learning and artificial intelligence techniques to optimize supply chain operations. The system integrates demand forecasting, inventory optimization, anomaly detection, natural language processing, and document management capabilities to provide a holistic solution for supply chain management.

The backend is built using Flask and provides a RESTful API interface that supports various supply chain operations. It utilizes modern AI techniques including transformer models for natural language understanding, time series forecasting using ARIMA and Prophet models, vector database for document retrieval, and optimization algorithms for inventory management.

## System Architecture

The system follows a modular architecture with specialized components for different aspects of supply chain management:

1. **Forecasting Module**: Handles time series forecasting using ARIMA, SARIMAX, and Prophet models
2. **Inventory Optimization Module**: Provides inventory allocation optimization using linear programming
3. **NLP Module**: Processes natural language queries about supply chain using transformer models
4. **Document Management System**: Stores and retrieves relevant supply chain documents
5. **Visualization Engine**: Creates interactive charts and visualizations
6. **Report Generator**: Generates PDF reports with insights and recommendations
7. **Audio Processing System**: Handles speech-to-text and text-to-speech for voice interaction

The system is designed to be multilingual, supporting English, French, and Arabic languages.

## Dependencies

The system relies on numerous Python libraries for its functionality:

```python
# Core ML and Data Science
transformers, langchain-huggingface, torch, bitsandbytes, autoawq, accelerate 
sentencepiece, protobuf, pulp, scipy, numpy, pandas

# Data Visualization
plotly, matplotlib, seaborn, kaleido

# PDF Generation and Reporting
reportlab

# Audio Processing
gtts, pydub, soundfile, ffmpeg-python

# Web and API
flask, flask-cors

# Other Libraries
pytrends, faiss-cpu, pypdf, langchain_text_splitters, langchain_core
```

## Configuration

The system uses several environment variables and configuration constants:

```python

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL = "openai/whisper-medium"
DATA_PATH = '/content/supply_chain_dataset.csv'
WEATHER_DATA_PATH = '/content/weather_data.csv'
TRENDS_DATA_PATH = '/content/trends_data.csv'
VECTOR_DB_PATH = 'supply_chain_faiss_index'
REPORT_PATH = 'reports'
AUDIO_PATH = 'audio'
```

## Core Classes

### ForecastingEngine

The `ForecastingEngine` class handles demand forecasting using various time series models.

```python
class ForecastingEngine:
    """Engine for training and using different forecasting models"""
```

#### Methods

- **`__init__(self, data)`**: Initializes the forecasting engine with historical data
- **`train_arima_model(self, product_id, exog_features=None)`**: Trains an ARIMA model for a specific product
- **`train_prophet_model(self, product_id)`**: Trains a Prophet model for a specific product
- **`forecast(self, product_id, steps=30, model_type='arima', exog_future=None)`**: Generates forecasts for future periods
- **`train_anomaly_detector(self, product_id)`**: Trains an anomaly detector using Isolation Forest
- **`detect_anomalies(self, product_id, new_data=None)`**: Detects anomalies in historical or new data
- **`analyze_anomalies(self, product_id, language="en")`**: Analyzes detected anomalies and provides insights
- **`simulate_scenario(self, product_id, scenario, language="en")`**: Simulates different scenarios and their impact on forecasts

The `ForecastingEngine` combines multiple forecasting techniques and can handle external factors like weather data and Google Trends data. It supports fallback to simulated data when historical data isn't available for a product.

### InventoryOptimizer

The `InventoryOptimizer` class handles inventory allocation optimization using linear programming.

```python
class InventoryOptimizer:
    """Advanced inventory optimization with multiple objectives and constraints"""
```

#### Methods

- **`__init__(self, warehouse_capacities=None, forecast=None)`**: Initializes the optimizer with warehouse capacities and forecast data
- **`_initialize_default_parameters(self)`**: Sets default values for optimization parameters
- **`optimize_inventory(self, demand_forecast=None, warehouse_capacities=None, holding_costs=None, transportation_costs=None, lead_times=None, service_level=0.95)`**: Optimizes inventory allocation across warehouses
- **`generate_recommendations(self, language="en")`**: Generates inventory optimization recommendations

The optimizer uses scipy's linear programming capabilities to minimize the total cost of inventory management while respecting warehouse capacity constraints and ensuring demand fulfillment.

### VisualizationEngine

The `VisualizationEngine` class provides methods for creating various visualizations.

```python
class VisualizationEngine:
    """Engine for creating various visualizations"""
```

#### Methods

- **`create_forecast_chart(forecast_df, product_id, title=None, language="en")`**: Creates a forecast chart with confidence intervals
- **`create_inventory_chart(inventory_data, title=None, language="en")`**: Creates a chart showing inventory levels
- **`create_product_pie_chart(inventory_data, location, title=None, language="en")`**: Creates a pie chart showing product distribution
- **`create_anomaly_chart(product_data, anomalies, product_id, title=None, language="en")`**: Creates a chart highlighting anomalies

This class leverages Plotly to create interactive visualizations that can be embedded in web applications or exported as static images for reports.

### ReportGenerator

The `ReportGenerator` class handles the generation of PDF reports.

```python
class ReportGenerator:
    """Generate PDF reports with charts and recommendations"""
```

#### Methods

- **`__init__(self)`**: Initializes the report generator
- **`create_forecast_report(self, product_id, forecast_df, recommendations, language="en")`**: Creates a forecast report with charts and recommendations
- **`create_inventory_report(self, inventory_data, recommendations, language="en")`**: Creates an inventory optimization report

The reports include visualizations, tables, and textual insights, and support multiple languages.

### CloudflaredTunnel

The `CloudflaredTunnel` class manages the Cloudflare tunnel for exposing the application publicly.

```python
class CloudflaredTunnel:
    """Helper class to create and manage a Cloudflare tunnel"""
```

#### Methods

- **`__init__(self, port=5000)`**: Initializes the tunnel manager
- **`active`**: Property that returns whether the tunnel is currently running
- **`get_public_url(self)`**: Gets the public URL of the tunnel
- **`find_cloudflared(self)`**: Finds the cloudflared executable
- **`_which(self, program)`**: Utility to find a program on the PATH

## API Endpoints

### Forecasting APIs

#### `/forecast` and `/demand_forecast` [POST]

Generates demand forecasts for specified products.

**Request Body:**
```json
{
  "product_id": "smartphone",
  "steps": 30,
  "model_type": "arima",
  "language": "en",
  "include_chart": true
}
```

**Response:**
```json
{
  "forecast": [...],
  "chart_html": "...",
  "product_id": "smartphone",
  "forecast_dates": [...]
}
```

### Inventory Optimization APIs

#### `/optimize` and `/inventory_optimize` [POST]

Optimizes inventory allocation across warehouses.

**Request Body:**
```json
{
  "demand_forecast": {
    "smartphone": 500,
    "laptop": 200
  },
  "warehouse_capacities": {
    "warehouse_a": 1000,
    "warehouse_b": 1500
  },
  "language": "en"
}
```

**Response:**
```json
{
  "allocation": {
    "smartphone": {
      "warehouse_a": 300,
      "warehouse_b": 200
    },
    "laptop": {
      "warehouse_a": 100,
      "warehouse_b": 100
    }
  },
  "total_cost": 12500,
  "recommendations": [...],
  "chart_html": "...",
  "optimization_parameters": {
    "eoq": {...},
    "reorder_point": {...},
    "safety_stock": {...}
  }
}
```

### Document Management APIs

#### `/add_document` [POST]

Adds a document to the knowledge base.

**Request Body:**
```
FormData with file and metadata
```

**Response:**
```json
{
  "status": "success",
  "message": "Document added to knowledge base",
  "document_id": "doc123"
}
```

#### `/documents` [GET]

Lists all available documents.

**Response:**
```json
{
  "documents": [
    {
      "id": "doc123",
      "title": "Supply Chain Best Practices",
      "file_type": "pdf",
      "added_date": "2023-10-15"
    },
    ...
  ]
}
```

### Natural Language APIs

#### `/ask` [POST]

Answers a supply chain question using the knowledge base and AI models.

**Request Body:**
```json
{
  "query": "How can I optimize my inventory levels?",
  "language": "en"
}
```

**Response:**
```json
{
  "answer": "To optimize inventory levels, you can implement the Economic Order Quantity (EOQ) model which balances ordering and holding costs. Additionally, consider setting appropriate safety stock levels based on demand variability and lead time uncertainty...",
  "sources": [...]
}
```

### Audio Processing APIs

#### `/ask_tts` [POST]

Answers a question and returns the response as audio.

**Request Body:**
```json
{
  "query": "How can I optimize my inventory levels?",
  "language": "en"
}
```

**Response:**
```json
{
  "answer": "...",
  "audio_url": "/audio/response_123.mp3"
}
```

#### `/upload_audio` [POST]

Uploads an audio file with a spoken question.

**Request Body:**
```
FormData with audio file
```

**Response:**
```json
{
  "transcript": "How can I optimize my inventory levels?",
  "audio_id": "audio123"
}
```

#### `/transcribe_audio` [POST]

Transcribes an uploaded audio file.

**Request Body:**
```json
{
  "audio_id": "audio123",
  "target_language": "en"
}
```

**Response:**
```json
{
  "transcript": "How can I optimize my inventory levels?"
}
```

### Reporting APIs

#### `/generate_report` [POST]

Generates a PDF report.

**Request Body:**
```json
{
  "report_type": "forecast",
  "product_id": "smartphone",
  "language": "en"
}
```

**Response:**
```json
{
  "report_url": "/reports/forecast_smartphone_20231015.pdf"
}
```

#### `/reports/<filename>` [GET]

Serves a generated report file.

### Anomaly Detection APIs

#### `/anomaly_detection` [POST]

Detects anomalies in historical data.

**Request Body:**
```json
{
  "product_id": "smartphone",
  "language": "en"
}
```

**Response:**
```json
{
  "anomalies": [...],
  "insights": [...],
  "chart_html": "..."
}
```

### Scenario Analysis APIs

#### `/scenario_analysis` [POST]

Analyzes different scenarios and their impact on forecasts.

**Request Body:**
```json
{
  "product_id": "smartphone",
  "scenario": "demand_increase",
  "language": "en"
}
```

**Response:**
```json
{
  "base_forecast": [...],
  "scenario_forecast": [...],
  "description": "Scenario: 20% increase in demand for smartphone",
  "chart_html": "..."
}
```

### Tunnel Management APIs

#### `/tunnel/status` [GET]

Gets the status of the Cloudflare tunnel.

**Response:**
```json
{
  "active": true,
  "url": "https://example.trycloudflare.com"
}
```

#### `/tunnel/start` [POST]

Starts the Cloudflare tunnel.

**Response:**
```json
{
  "status": "started",
  "url": "https://example.trycloudflare.com"
}
```

#### `/tunnel/stop` [POST]

Stops the Cloudflare tunnel.

**Response:**
```json
{
  "status": "stopped"
}
```

## Helper Functions

### Audio Processing Functions

- **`transcribe_audio(audio_file, target_language="en")`**: Transcribes an audio file using Whisper
- **`generate_speech(text, language="en")`**: Generates speech from text using Google Text-to-Speech

### Data Management Functions

- **`load_data()`**: Loads or creates simulated supply chain, weather, and trends data
- **`merge_data_for_modeling()`**: Merges multiple data sources for modeling
- **`load_and_process_documents()`**: Loads and processes documents for the vector store
- **`process_pdf(file_path)`**: Extracts and processes text from a PDF file
- **`save_document_metadata(metadata)`**: Saves document metadata to the index file
- **`get_document_metadata()`**: Gets document metadata from the index file

### NLP Functions

- **`post_process_response(text, language="en")`**: Post-processes an LLM response for conversation
- **`generate_response(query, context="", language="en")`**: Generates a user-friendly response
- **`get_rag_context(query)`**: Retrieves relevant context from the vector store
- **`enhance_supply_chain_query(query)`**: Enhances a query with supply chain specific terminology

## Data Management

The system works with several types of data:

1. **Supply Chain Data**: Historical demand, inventory levels, lead times, and costs
2. **Weather Data**: Temperature, precipitation, and other weather conditions
3. **Google Trends Data**: Search trends related to products
4. **Documents**: Supply chain documents in various formats (PDF, text)

When actual data files aren't available, the system can generate simulated data for development and testing purposes.

The system also maintains a vector database using FAISS for efficient document retrieval based on semantic similarity.
