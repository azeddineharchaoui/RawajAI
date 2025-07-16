# Supply Chain AI Agent Documentation

Ce document fournit une documentation complète du backend du Supply Chain AI Agent, incluant tous les modules, fonctions et routes API.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Modules et classes](#modules-et-classes)
   - [ForecastingEngine](#forecastingengine)
   - [InventoryOptimizer](#inventoryoptimizer)
   - [VisualizationEngine](#visualizationengine)
   - [ReportGenerator](#reportgenerator)
   - [CloudflaredTunnel](#cloudflaredtunnel)
4. [API Routes](#api-routes)
   - [Prévision de la demande](#prévision-de-la-demande)
   - [Optimisation des stocks](#optimisation-des-stocks)
   - [Questions et réponses](#questions-et-réponses)
   - [Détection d'anomalies](#détection-danomalies)
   - [Analyse de scénarios](#analyse-de-scénarios)
   - [Génération de rapports](#génération-de-rapports)
   - [Gestion des documents](#gestion-des-documents)
   - [Gestion du tunnel Cloudflare](#gestion-du-tunnel-cloudflare)
   - [Gestion des fichiers](#gestion-des-fichiers)
5. [Configuration et démarrage](#configuration-et-démarrage)
6. [Utilisation avec Google Colab](#utilisation-avec-google-colab)
7. [Dépannage](#dépannage)

## Vue d'ensemble

Le Supply Chain AI Agent est un outil d'intelligence artificielle conçu pour aider à la gestion de la chaîne d'approvisionnement. Il offre des fonctionnalités avancées telles que la prévision de la demande, l'optimisation des stocks, la détection d'anomalies, l'analyse de scénarios, et plus encore. L'agent est construit avec une architecture backend Flask et est accessible via une API REST.

Principales fonctionnalités :
- Prévision de la demande basée sur des données historiques
- Optimisation des niveaux de stock et des commandes
- Détection d'anomalies dans les données de la chaîne d'approvisionnement
- Analyse de scénarios pour la planification stratégique
- Support multilingue (anglais, français, arabe)
- Génération de rapports PDF
- Interface vocale (transcription et synthèse)
- Intégration de données externes (météo, tendances)
- Tunnel Cloudflare pour l'accès public à l'API

## Architecture

L'application est construite sur les technologies suivantes :
- **Backend** : Flask (Python)
- **Modèles IA** : 
  - AceGPT-13B-chat-AWQ pour le traitement du langage naturel
  - Whisper pour la reconnaissance vocale
  - Prophet, ARIMA, SARIMAX pour les prévisions
  - IsolationForest pour la détection d'anomalies
- **Base de données vectorielle** : FAISS avec des embeddings Hugging Face
- **Tunneling** : Cloudflare pour l'accès public
- **Visualisation** : Matplotlib, Seaborn, Plotly
- **Génération de rapports** : ReportLab

## Modules et classes

### ForecastingEngine

La classe `ForecastingEngine` gère toutes les opérations de prévision de la demande.

#### Méthodes principales :

- **`__init__(self, data_path=DATA_PATH)`** : Initialise le moteur de prévision avec les données historiques.
- **`preprocess_data(self, df, target_col)`** : Prétraite les données pour l'analyse.
- **`arima_forecast(self, data, periods=30)`** : Génère des prévisions à l'aide du modèle ARIMA.
- **`sarimax_forecast(self, data, periods=30, seasonal_period=7)`** : Génère des prévisions avec le modèle SARIMAX incluant la saisonnalité.
- **`prophet_forecast(self, data, periods=30)`** : Utilise Facebook Prophet pour les prévisions.
- **`ensemble_forecast(self, data, periods=30)`** : Combine plusieurs modèles pour une prévision d'ensemble.
- **`incorporate_external_factors(self, forecast_df, weather_data=None, trends_data=None)`** : Intègre des facteurs externes comme la météo ou les tendances Google.

### InventoryOptimizer

La classe `InventoryOptimizer` est responsable de l'optimisation des niveaux de stock et des stratégies de commande.

#### Méthodes principales :

- **`__init__(self, data_path=DATA_PATH)`** : Initialise l'optimiseur avec les données historiques.
- **`calculate_eoq(self, demand, order_cost, holding_cost)`** : Calcule la quantité économique de commande (EOQ).
- **`calculate_reorder_point(self, avg_demand, lead_time, safety_stock)`** : Calcule le point de commande.
- **`calculate_safety_stock(self, demand_std, service_level, lead_time)`** : Calcule le stock de sécurité.
- **`optimize_inventory_levels(self, forecast_demand, holding_cost, stockout_cost)`** : Optimise les niveaux de stock sur la base des prévisions.
- **`optimize_multi_echelon(self, locations, demands, costs)`** : Optimise les stocks dans une chaîne d'approvisionnement multi-échelons.
- **`linear_optimization(self, products, constraints)`** : Résout les problèmes d'optimisation linéaire.

### VisualizationEngine

La classe `VisualizationEngine` gère la création de visualisations pour les données et les résultats d'analyse.

#### Méthodes principales :

- **`__init__(self)`** : Initialise le moteur de visualisation.
- **`plot_forecast(self, historical_data, forecast_data, confidence_interval=None)`** : Crée un graphique de prévision.
- **`plot_inventory_levels(self, inventory_data, optimal_levels=None)`** : Visualise les niveaux de stock.
- **`plot_anomalies(self, data, anomalies)`** : Met en évidence les anomalies dans les données.
- **`create_dashboard(self, data_dict, layout='grid')`** : Crée un tableau de bord avec plusieurs graphiques.
- **`export_plot(self, fig, filename, format='png')`** : Exporte une visualisation dans différents formats.
- **`generate_interactive_plot(self, data, plot_type='line')`** : Crée un graphique interactif avec Plotly.

### ReportGenerator

La classe `ReportGenerator` est responsable de la création de rapports PDF détaillés.

#### Méthodes principales :

- **`__init__(self, output_dir=REPORT_PATH)`** : Initialise le générateur de rapports.
- **`create_summary_page(self, title, summary_text, key_metrics)`** : Crée une page de résumé.
- **`add_visualization(self, figure, caption)`** : Ajoute une visualisation au rapport.
- **`add_table(self, data, headers, title)`** : Ajoute un tableau au rapport.
- **`add_text_section(self, title, content)`** : Ajoute une section de texte.
- **`compile_report(self, filename, metadata)`** : Compile toutes les sections en un rapport PDF.
- **`generate_executive_summary(self, analysis_results)`** : Génère un résumé exécutif basé sur les résultats d'analyse.

### CloudflaredTunnel

La classe `CloudflaredTunnel` gère la création et la maintenance d'un tunnel Cloudflare pour rendre l'API accessible publiquement.

#### Méthodes principales :

- **`__init__(self, port=5000)`** : Initialise la configuration du tunnel.
- **`is_running(self)`** : Vérifie si le tunnel est actif (propriété).
- **`get_public_url(self)`** : Récupère l'URL publique du tunnel.
- **`_download_cloudflared(self)`** : Télécharge l'exécutable Cloudflared.
- **`start(self)`** : Démarre le tunnel Cloudflare.
- **`stop(self)`** : Arrête le tunnel Cloudflare.

## API Routes

### Prévision de la demande

#### `POST /forecast`

Génère des prévisions de demande basées sur les données historiques.

**Paramètres de requête (JSON) :**
```json
{
  "data": "base64_encoded_csv_ou_json",
  "target_column": "quantité_vendue",
  "periods": 30,
  "model": "ensemble",
  "include_external_factors": true,
  "format": "json"
}
```

**Réponse :**
```json
{
  "forecast": [
    {"date": "2023-06-01", "value": 123.45, "lower_bound": 100.0, "upper_bound": 150.0},
    {"date": "2023-06-02", "value": 126.78, "lower_bound": 102.0, "upper_bound": 155.0},
    ...
  ],
  "metrics": {
    "mape": 3.2,
    "rmse": 12.5,
    "accuracy": 96.8
  },
  "plot_url": "/static/plots/forecast_12345.png"
}
```

### Optimisation des stocks

#### `POST /optimize`

Optimise les niveaux de stock et les stratégies de commande.

**Paramètres de requête (JSON) :**
```json
{
  "forecast_data": [{"date": "2023-06-01", "value": 123.45}, ...],
  "holding_cost": 0.2,
  "ordering_cost": 50,
  "lead_time": 5,
  "service_level": 0.95,
  "multi_echelon": false
}
```

**Réponse :**
```json
{
  "optimal_inventory_levels": [
    {"date": "2023-06-01", "level": 350},
    ...
  ],
  "reorder_points": [
    {"product": "A", "reorder_point": 125}
  ],
  "economic_order_quantities": [
    {"product": "A", "eoq": 250}
  ],
  "total_cost": 12500,
  "plot_url": "/static/plots/inventory_12345.png"
}
```

### Questions et réponses

#### `POST /ask`

Répond aux questions en langage naturel concernant la chaîne d'approvisionnement.

**Paramètres de requête (JSON) :**
```json
{
  "query": "Quels sont les produits qui risquent d'être en rupture de stock le mois prochain ?",
  "language": "fr",
  "context": {}
}
```

**Réponse :**
```json
{
  "answer": "D'après notre analyse des tendances actuelles et des prévisions, les produits suivants risquent d'être en rupture de stock : Produit A (87% de risque), Produit C (65% de risque), et Produit F (52% de risque). Je vous recommande d'augmenter les commandes pour ces articles dès maintenant.",
  "confidence": 0.92,
  "sources": [
    {"title": "Rapport de prévision", "relevance": 0.95},
    {"title": "Historique des stocks", "relevance": 0.87}
  ]
}
```

### Détection d'anomalies

#### `POST /anomaly_detection`

Détecte les anomalies dans les données de la chaîne d'approvisionnement.

**Paramètres de requête (JSON) :**
```json
{
  "data": "base64_encoded_csv_ou_json",
  "target_columns": ["quantité_vendue", "délai_livraison"],
  "sensitivity": 0.8
}
```

**Réponse :**
```json
{
  "anomalies": [
    {"date": "2023-05-15", "column": "quantité_vendue", "value": 523, "expected_range": [120, 280], "severity": "high"},
    {"date": "2023-05-22", "column": "délai_livraison", "value": 15, "expected_range": [2, 8], "severity": "medium"},
    ...
  ],
  "summary": "Détection de 5 anomalies majeures sur la période analysée",
  "plot_url": "/static/plots/anomalies_12345.png"
}
```

### Analyse de scénarios

#### `POST /scenario_analysis`

Effectue une analyse de scénarios pour évaluer différentes stratégies.

**Paramètres de requête (JSON) :**
```json
{
  "baseline_data": "base64_encoded_csv_ou_json",
  "scenarios": [
    {
      "name": "Augmentation_Demande_20%",
      "adjustments": {
        "demand_multiplier": 1.2,
        "lead_time_change": 0
      }
    },
    {
      "name": "Pénurie_Fournisseur",
      "adjustments": {
        "demand_multiplier": 1.0,
        "lead_time_change": 5,
        "supply_constraints": {"product_A": 0.7, "product_B": 0.8}
      }
    }
  ],
  "kpis": ["stockout_rate", "inventory_cost", "service_level"]
}
```

**Réponse :**
```json
{
  "scenarios_results": [
    {
      "name": "Baseline",
      "kpis": {
        "stockout_rate": 0.05,
        "inventory_cost": 10000,
        "service_level": 0.95
      }
    },
    {
      "name": "Augmentation_Demande_20%",
      "kpis": {
        "stockout_rate": 0.18,
        "inventory_cost": 12000,
        "service_level": 0.82
      },
      "recommended_actions": [
        "Augmenter le stock de sécurité de 25%",
        "Réduire les délais de commande"
      ]
    },
    ...
  ],
  "comparative_analysis": "L'analyse montre que le scénario de pénurie fournisseur présente le plus grand risque pour la continuité des opérations...",
  "plot_url": "/static/plots/scenarios_12345.png"
}
```

### Génération de rapports

#### `POST /generate_report`

Génère un rapport PDF détaillé basé sur les analyses.

**Paramètres de requête (JSON) :**
```json
{
  "title": "Analyse de la chaîne d'approvisionnement Q2 2023",
  "sections": ["forecast", "inventory", "anomalies", "recommendations"],
  "forecast_data": {...},
  "inventory_data": {...},
  "anomalies_data": {...},
  "language": "fr",
  "include_executive_summary": true
}
```

**Réponse :**
```json
{
  "report_url": "/reports/supply_chain_analysis_20230531_123456.pdf",
  "sections_included": ["forecast", "inventory", "anomalies", "recommendations"],
  "summary": "Le rapport a été généré avec succès et inclut 4 sections, 8 visualisations et 5 tableaux."
}
```

### Gestion des documents

#### `POST /add_document`

Ajoute un document à la base de connaissances pour le RAG (Retrieval-Augmented Generation).

**Paramètres de requête (multipart/form-data) :**
- `document`: Fichier PDF ou texte à ajouter
- `description`: Description du document
- `tags`: Tags pour catégoriser le document

**Réponse :**
```json
{
  "status": "success",
  "message": "Document ajouté à la base de connaissances",
  "document_id": "doc_12345",
  "chunks_extracted": 28,
  "vector_db_updated": true
}
```

### Gestion du tunnel Cloudflare

#### `GET /tunnel/status`

Récupère l'état actuel du tunnel Cloudflare.

**Réponse :**
```json
{
  "status": "running",
  "url": "https://abcd-efgh-ijkl-mnop.trycloudflare.com"
}
```

#### `POST /tunnel/start`

Démarre le tunnel Cloudflare.

**Réponse :**
```json
{
  "status": "started",
  "message": "Tunnel started successfully",
  "url": "https://abcd-efgh-ijkl-mnop.trycloudflare.com"
}
```

#### `POST /tunnel/stop`

Arrête le tunnel Cloudflare.

**Réponse :**
```json
{
  "status": "stopped",
  "message": "Tunnel stopped successfully"
}
```

#### `GET /tunnel/test`

Teste la connectivité du tunnel Cloudflare.

**Réponse :**
```json
{
  "status": "success",
  "message": "Tunnel is working properly!",
  "timestamp": "2023-05-31 12:34:56",
  "tunnel_status": "running",
  "tunnel_url": "https://abcd-efgh-ijkl-mnop.trycloudflare.com"
}
```

### Gestion des fichiers

#### `GET /reports/<filename>`

Récupère un fichier de rapport généré.

#### `GET /audio/<filename>`

Récupère un fichier audio (synthèse vocale).

#### `POST /upload_audio`

Télécharge un fichier audio pour transcription.

**Paramètres de requête (multipart/form-data) :**
- `audio`: Fichier audio à transcrire
- `language`: Langue du fichier audio

**Réponse :**
```json
{
  "status": "success",
  "transcription": "Le texte transcrit à partir de l'audio",
  "language_detected": "fr",
  "confidence": 0.95
}
```

## Configuration et démarrage

L'application peut être configurée via plusieurs variables dans les fichiers `app.py` et `merge.py` :

```python
# Configuration
os.environ["HF_TOKEN"] = 'votre_token_huggingface'
MODEL_NAME = "MohamedRashad/AceGPT-13B-chat-AWQ"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL = "openai/whisper-base"
DATA_PATH = "supply_chain_data.csv"
WEATHER_DATA_PATH = "weather_data.csv"
TRENDS_DATA_PATH = "trends_data.csv"
VECTOR_DB_PATH = "supply_chain_faiss_index"
REPORT_PATH = "reports"
AUDIO_PATH = "audio"
```

Pour démarrer l'application :

1. Assurez-vous que toutes les dépendances sont installées :
   ```
   pip install transformers langchain-huggingface torch torchaudio langchain_community bitsandbytes autoawq accelerate sentencepiece protobuf pulp scipy plotly reportlab gtts pytrends faiss-cpu
   ```

2. Exécutez le fichier `app.py` :
   ```
   python app.py
   ```

3. L'application démarrera sur le port 5000 et tentera automatiquement d'initialiser un tunnel Cloudflare pour l'accès externe.

## Utilisation avec Google Colab

Pour utiliser l'application dans Google Colab :

1. Importez les fichiers nécessaires dans votre environnement Colab.

2. Installez les dépendances :
   ```python
   !pip install transformers langchain-huggingface torch torchaudio langchain_community bitsandbytes autoawq accelerate sentencepiece protobuf pulp scipy plotly reportlab gtts pytrends faiss-cpu
   ```

3. Exécutez l'application :
   ```python
   !python app.py
   ```

4. L'application détectera automatiquement l'environnement Colab et ajustera ses paramètres en conséquence (par exemple, en désactivant le threading Flask).

5. Utilisez l'URL du tunnel Cloudflare générée pour accéder à l'API à partir de n'importe quel appareil.

## Dépannage

### Problèmes courants et solutions

1. **Erreur d'authentification Hugging Face**
   - Assurez-vous que votre token HF est valide et que vous avez accès aux modèles requis.
   - Solution : Mettez à jour votre token dans la variable d'environnement `HF_TOKEN`.

2. **Timeout du tunnel Cloudflare**
   - Le tunnel peut échouer à démarrer ou expirer après un certain temps.
   - Solution : Redémarrez-le manuellement via l'endpoint `/tunnel/start`.

3. **Erreur de désérialisation FAISS**
   - Si vous rencontrez une erreur lors du chargement de l'index FAISS.
   - Solution : Ajoutez le paramètre `allow_dangerous_deserialization=True` ou recréez l'index.

4. **Échec des dépendances dans Colab**
   - Certaines dépendances peuvent échouer à s'installer dans Colab.
   - Solution : Installez-les séparément ou utilisez des versions spécifiques.

5. **Erreur "CUDA out of memory"**
   - Les modèles de langage peuvent consommer beaucoup de VRAM.
   - Solution : Utilisez une quantification plus agressive ou un modèle plus petit.

Pour plus d'assistance, consultez les journaux d'erreurs ou soumettez un problème sur le référentiel du projet.




------------------------------------------------------------------------------------------
# RawajAI: Intelligent Supply Chain Management Platform

## Executive Summary

RawajAI is a cutting-edge supply chain management platform that leverages advanced artificial intelligence, machine learning, and optimization techniques to transform traditional supply chain operations into data-driven, predictive systems. Our solution combines a powerful Python-based backend with an intuitive React Native mobile application to deliver actionable insights and autonomous decision support for inventory management, demand forecasting, and supply chain optimization.

By integrating agentic AI and generative AI capabilities, RawajAI positions itself at the forefront of Industry 4.0 solutions, enabling businesses to reduce costs, minimize risks, and maximize operational efficiency in increasingly complex global supply chains.

## Technical Architecture

### Backend (stable.py)

The RawajAI backend is powered by a sophisticated Python framework that integrates multiple AI models and analytics engines:

1. **AI Core**
   - Large Language Model (Mistral-7B) for natural language understanding and generation
   - Embeddings-based retrieval (FAISS) for domain-specific knowledge
   - Speech-to-text (Whisper) and text-to-speech (gTTS) for multimodal interaction

2. **Forecasting Engine**
   - ARIMA and SARIMAX models for time-series forecasting
   - Prophet models for seasonal demand prediction with external regressors
   - Custom ensemble methods for improved forecast accuracy

3. **Optimization Engine**
   - Linear programming solvers for multi-objective optimization
   - Warehouse allocation algorithms
   - Economic Order Quantity (EOQ) calculation
   - Safety stock determination with service level guarantees

4. **Analytics Engine**
   - Anomaly detection using Isolation Forest
   - Scenario analysis with Monte Carlo simulations
   - Risk assessment modeling
   - Visualization generation with Plotly

5. **Reporting System**
   - PDF generation with custom templates
   - Multilingual support (English, French, Arabic)
   - Interactive visualization exports

### Frontend (React Native)

The mobile application provides a seamless user experience across devices:

1. **Intelligent Dashboard**
   - Real-time KPIs and metrics
   - Inventory level monitoring
   - Demand tracking and visualization
   - Alert and notification system

2. **Forecasting Module**
   - Interactive demand forecasts
   - Confidence interval visualization
   - Seasonal pattern identification
   - Product-specific analysis

3. **Inventory Management**
   - Multi-location inventory optimization
   - Capacity utilization tracking
   - Reorder point recommendations
   - Product distribution analysis

4. **Analytics Module**
   - Anomaly detection and visualization
   - Scenario planning and "what-if" analysis
   - Risk assessment visualization
   - PDF report generation and viewing

5. **AI Assistant**
   - Natural language query interface
   - Voice-based interaction
   - Context-aware responses
   - Multilingual support

## Agentic AI Capabilities

RawajAI exemplifies the power of agentic AI in industrial applications through:

1. **Autonomous Decision Support**
   - The system autonomously analyzes inventory levels, demand patterns, and external factors to recommend optimal stocking levels and reorder points.
   - Proactive anomaly detection identifies potential supply chain disruptions before they impact operations.

2. **Contextual Understanding**
   - The RAG-enhanced LLM understands supply chain concepts and business context, providing relevant advice grounded in domain knowledge.
   - The system interprets complex queries about inventory optimization, logistics planning, and demand forecasting.

3. **Multi-step Reasoning**
   - When optimizing inventory, the AI agent considers multiple interdependent factors including holding costs, transportation costs, lead times, and service levels.
   - For scenario analysis, the system models cascading effects of changes throughout the supply chain.

4. **Adaptive Learning**
   - The system incorporates new documents and knowledge through the `/add_document` endpoint, continuously expanding its domain expertise.
   - Forecasting models adapt to changing patterns in the data over time.

## Generative AI Integration

RawajAI leverages generative AI to transform supply chain data into actionable intelligence:

1. **Natural Language Insights**
   - The LLM generates detailed, contextual explanations of forecasts, anomalies, and optimization recommendations.
   - Multilingual support enables global teams to receive insights in their preferred language.

2. **Visual Content Generation**
   - Dynamic chart and visualization creation based on specific data patterns and analysis needs.
   - Custom PDF reports tailored to different stakeholders and decision-making contexts.

3. **Scenario Generation**
   - AI-generated scenarios for supply chain planning based on historical patterns and external factors.
   - Synthetic data generation for simulation and testing of extreme conditions.

4. **Interactive Query Processing**
   - Natural language query processing for complex supply chain questions.
   - Voice-to-text and text-to-speech for hands-free operation in warehouse environments.

## Proof of Concept

The RawajAI platform demonstrates its capabilities through:

1. **End-to-End Implementation**
   - Fully functional backend with 15+ API endpoints
   - Complete mobile application with 5 integrated modules
   - Multilingual support across the entire platform

2. **Real-world Data Processing**
   - Integration with external data sources including weather patterns and market trends
   - Support for actual inventory data through CSV imports
   - Realistic simulations based on industry-standard supply chain scenarios

3. **Production-ready Architecture**
   - Cloud-based deployment with Cloudflare tunneling
   - Memory optimization for resource-constrained environments
   - Error handling and fallback mechanisms for resilience

4. **Cross-platform Compatibility**
   - React Native frontend works on iOS and Android
   - Backend supports cloud deployment and local execution
   - API design follows RESTful principles for integration flexibility

## Proof of Value

RawajAI delivers tangible business value through:

1. **Cost Reduction**
   - Optimized inventory levels reduce holding costs by up to 25%
   - Improved demand forecasting minimizes stockouts and overstock situations
   - Efficient warehouse allocation reduces transportation and logistics costs

2. **Risk Mitigation**
   - Early anomaly detection prevents supply chain disruptions
   - Scenario analysis enables proactive planning for market changes
   - Real-time monitoring identifies issues before they impact customers

3. **Operational Efficiency**
   - Automated reporting saves 10+ hours of manual analysis per week
   - AI assistant provides instant answers to complex supply chain questions
   - Mobile access enables decision-making from anywhere

4. **Strategic Advantage**
   - Data-driven insights enable more informed strategic planning
   - AI-powered optimization creates competitive advantage through cost leadership
   - Scalable architecture supports growing businesses and complex supply chains

## Innovation Highlights

1. **Multimodal AI Integration**
   - Seamless combination of text, speech, and visual AI capabilities
   - Context-aware responses that incorporate supply chain knowledge
   - Continuous learning from new documents and interactions

2. **Hybrid Forecasting Approach**
   - Combination of statistical models (ARIMA) with machine learning for optimal accuracy
   - Integration of external factors like weather and market trends
   - Confidence intervals for risk-aware planning

3. **Multi-objective Optimization**
   - Balancing competing objectives like cost, service level, and capacity utilization
   - Location-specific recommendations for global supply chains
   - Product-specific optimization strategies

4. **Cross-lingual Capabilities**
   - Full support for English, French, and Arabic across all interfaces
   - Language-aware visualization and report generation
   - Voice interaction in multiple languages

## Conclusion

RawajAI represents a significant advancement in applying artificial intelligence to supply chain management. By combining agentic AI, generative AI, statistical modeling, and optimization techniques, our platform transforms how businesses manage inventory, forecast demand, and optimize their supply chains.

The fully functional proof of concept demonstrates both technical feasibility and business value, making RawajAI a compelling solution for companies seeking to leverage AI for competitive advantage in their supply chain operations.