# RawajAI: Documentation Technique Complète

## Table des Matières

1. [Vue d'ensemble du système](#vue-densemble-du-système)
2. [Architecture technique](#architecture-technique)
3. [Modèles et Intelligence Artificielle](#modèles-et-intelligence-artificielle)
4. [Modules Backend](#modules-backend)
5. [API REST Endpoints](#api-rest-endpoints)
6. [Frontend Architecture](#frontend-architecture)
7. [Consommation de l'API côté Frontend](#consommation-de-lapi-côté-frontend)
8. [Types de données et interfaces](#types-de-données-et-interfaces)
9. [Gestion des états et contextes](#gestion-des-états-et-contextes)
10. [Sécurité et authentification](#sécurité-et-authentification)
11. [Déploiement et infrastructure](#déploiement-et-infrastructure)
12. [Optimisations et performances](#optimisations-et-performances)

---

## Vue d'ensemble du système

RawajAI est une plateforme intelligente de gestion de chaîne d'approvisionnement qui combine des technologies d'IA avancées avec une architecture moderne pour offrir des solutions prédictives et optimisées. Le système est construit sur une architecture modulaire comprenant :

### Technologies clés
- **Backend**: Flask (Python) avec modèles d'IA intégrés
- **Frontend**: React Native (TypeScript/Expo)
- **IA**: Mistral-7B, Whisper, Prophet, ARIMA/SARIMAX
- **Base de données vectorielle**: FAISS avec embeddings Hugging Face
- **Infrastructure**: Cloudflare Tunnel pour l'accès public

### Capacités principales
- Prévision de la demande multi-modèles
- Optimisation des stocks avec programmation linéaire
- Détection d'anomalies en temps réel
- Analyse de scénarios prédictifs
- Assistant IA conversationnel multilingue
- Génération de rapports automatisés

---

## Architecture technique

### Architecture Backend (stable.py)

Le backend RawajAI est structuré autour de plusieurs modules spécialisés qui interagissent pour fournir des services d'IA complets :

```
Backend/
├── stable.py              # Application Flask principale
├── colab_config.py        # Configuration pour Google Colab
└── requirements.txt       # Dépendances Python
```

#### Configuration et initialisation

```python
# Configuration des modèles IA
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL = "openai/whisper-base"

# Initialisation avec quantification 4-bit pour optimiser la mémoire
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
)
```

### Architecture Frontend (React Native)

Le frontend utilise une architecture basée sur des composants réutilisables et des hooks personnalisés :

```
Frontend/
├── app/                    # Pages et navigation
│   ├── (tabs)/            # Navigation par onglets
│   └── _layout.tsx        # Layout principal
├── components/            # Composants réutilisables
│   └── ui/               # Composants UI de base
├── services/             # Services API
├── types/               # Définitions TypeScript
├── context/            # Gestion d'état global
└── hooks/             # Hooks personnalisés
```

---

## Modèles et Intelligence Artificielle

### 1. Modèle de langage principal (Mistral-7B)

**Utilisation**: Traitement du langage naturel, génération de réponses contextuelles

```python
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)
```

**Optimisations appliquées**:
- Quantification 4-bit pour réduire l'utilisation mémoire
- Configuration optimisée pour les environnements contraints
- Pipeline de génération avec contrôle de température

### 2. Reconnaissance vocale (Whisper)

**Utilisation**: Transcription audio vers texte pour l'interface vocale

```python
def transcribe_audio(audio_file, target_language="en"):
    audio_array, sampling_rate = torchaudio.load(audio_file)
    if audio_array.shape[0] > 1:
        audio_array = torch.mean(audio_array, dim=0, keepdim=True)
    
    input_features = whisper_processor(
        audio_array.squeeze().numpy(), 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features
    
    predicted_ids = whisper_model.generate(
        input_features, 
        forced_decoder_ids=forced_decoder_ids
    )
    return whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
```

### 3. Synthèse vocale (gTTS)

**Utilisation**: Conversion texte vers parole pour les réponses de l'assistant

```python
def generate_speech(text, language="en"):
    lang_map = {"en": "en", "fr": "fr", "ar": "ar"}
    lang = lang_map.get(language, "en")
    
    tts = gTTS(text=text, lang=lang, slow=False)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{AUDIO_PATH}/response_{timestamp}.mp3"
    tts.save(filename)
    return filename
```

### 4. Base de données vectorielle (FAISS)

**Utilisation**: Stockage et recherche sémantique de documents pour RAG

```python
# Documents de chaîne d'approvisionnement multilingues
supply_chain_docs = [
    "Just-in-time inventory reduces holding costs but increases risk of stockouts",
    "La gestion des stocks juste à temps réduit les coûts de stockage",
    "إدارة المخزون في الوقت المناسب تقلل تكاليف الاحتفاظ بالمخزون",
    # ... plus de documents
]

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = FAISS.from_texts(supply_chain_docs, embeddings)
```

---

## Modules Backend

### 1. ForecastingEngine

**Responsabilité**: Gestion des prévisions de demande avec plusieurs modèles statistiques

#### Méthodes principales:

```python
class ForecastingEngine:
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.scalers = {}
        self.anomaly_detectors = {}
    
    def train_arima_model(self, product_id, exog_features=None):
        """Entraîne un modèle ARIMA pour un produit spécifique"""
        if exog is not None:
            model = SARIMAX(
                product_data['demand'], 
                exog=exog, 
                order=(2, 1, 2), 
                seasonal_order=(1, 1, 1, 7)
            )
        else:
            model = ARIMA(product_data['demand'], order=(2, 1, 2))
        
        model_fit = model.fit()
        self.models[product_id] = {
            'model': model_fit,
            'last_date': product_data['date'].max(),
            'exog_features': exog_features
        }
        return model_fit
    
    def train_prophet_model(self, product_id):
        """Entraîne un modèle Prophet avec régresseurs externes"""
        prophet_data = product_data[['date', 'demand']].rename(
            columns={'date': 'ds', 'demand': 'y'}
        )
        
        model = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=True, 
            daily_seasonality=False
        )
        
        # Ajout de régresseurs pour facteurs externes
        for regressor in ['temperature', 'precipitation']:
            if regressor in prophet_data.columns:
                model.add_regressor(regressor)
        
        model.fit(prophet_data)
        self.models[f"prophet_{product_id}"] = model
```

### 2. InventoryOptimizer

**Responsabilité**: Optimisation des niveaux de stock et stratégies de commande

#### Méthodes clés:

```python
class InventoryOptimizer:
    def calculate_eoq(self, demand, order_cost, holding_cost):
        """Calcule la quantité économique de commande"""
        return np.sqrt((2 * demand * order_cost) / holding_cost)
    
    def calculate_reorder_point(self, avg_demand, lead_time, safety_stock):
        """Calcule le point de réapprovisionnement"""
        return (avg_demand * lead_time) + safety_stock
    
    def optimize_multi_echelon(self, locations, demands, costs):
        """Optimisation multi-échelons avec programmation linéaire"""
        prob = pulp.LpProblem("Multi_Echelon_Optimization", pulp.LpMinimize)
        
        # Variables de décision
        x = pulp.LpVariable.dicts("allocation", 
                                 ((i, j) for i in locations for j in demands),
                                 lowBound=0, cat='Continuous')
        
        # Fonction objectif
        prob += pulp.lpSum([costs[i][j] * x[i, j] 
                           for i in locations for j in demands])
        
        # Contraintes
        for j in demands:
            prob += pulp.lpSum([x[i, j] for i in locations]) >= demands[j]
        
        prob.solve()
        return self.extract_solution(x, locations, demands)
```

### 3. VisualizationEngine

**Responsabilité**: Génération de visualisations interactives avec Plotly

```python
class VisualizationEngine:
    def create_forecast_plot(self, data, forecast):
        """Crée un graphique de prévision interactif"""
        fig = go.Figure()
        
        # Données historiques
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['demand'],
            mode='lines+markers',
            name='Données historiques',
            line=dict(color='blue')
        ))
        
        # Prévisions
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['forecast'],
            mode='lines',
            name='Prévision',
            line=dict(color='red', dash='dash')
        ))
        
        # Intervalles de confiance
        if 'upper_bound' in forecast:
            fig.add_trace(go.Scatter(
                x=forecast['date'].tolist() + forecast['date'][::-1].tolist(),
                y=forecast['upper_bound'].tolist() + forecast['lower_bound'][::-1].tolist(),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalle de confiance'
            ))
        
        return fig.to_html(include_plotlyjs='cdn')
```

### 4. ReportGenerator

**Responsabilité**: Génération de rapports PDF avec ReportLab

```python
class ReportGenerator:
    def create_executive_summary(self, analysis_results):
        """Génère un résumé exécutif basé sur les analyses"""
        summary_elements = []
        
        # En-tête du rapport
        title_style = getSampleStyleSheet()['Title']
        summary_elements.append(Paragraph("Résumé Exécutif - Analyse Supply Chain", title_style))
        
        # Métriques clés
        if 'forecast_accuracy' in analysis_results:
            accuracy_text = f"Précision des prévisions: {analysis_results['forecast_accuracy']:.1f}%"
            summary_elements.append(Paragraph(accuracy_text, getSampleStyleSheet()['Normal']))
        
        # Tableau des KPIs
        if 'kpis' in analysis_results:
            kpi_data = [['Métrique', 'Valeur', 'Statut']]
            for kpi, value in analysis_results['kpis'].items():
                status = '✓' if value > 0.8 else '⚠'
                kpi_data.append([kpi, f"{value:.2f}", status])
            
            kpi_table = Table(kpi_data)
            kpi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            summary_elements.append(kpi_table)
        
        return summary_elements
```

### 5. CloudflaredTunnel

**Responsabilité**: Gestion des tunnels Cloudflare pour l'accès public

```python
class CloudflaredTunnel:
    def __init__(self, port=5000):
        self.port = port
        self.process = None
        self.url = None
        self.cloudflared_path = self._get_cloudflared_path()
    
    def start(self):
        """Démarre le tunnel Cloudflare"""
        if self.is_running:
            return self.url
        
        cmd = [self.cloudflared_path, 'tunnel', '--url', f'http://localhost:{self.port}']
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Attendre que l'URL soit disponible
        start_time = time.time()
        while time.time() - start_time < 30:
            if self.process.poll() is not None:
                break
            
            line = self.process.stderr.readline()
            if 'trycloudflare.com' in line:
                self.url = line.split('https://')[1].split(' ')[0].strip()
                return f"https://{self.url}"
            
            time.sleep(0.1)
        
        raise Exception("Failed to start Cloudflare tunnel")
```

---

## API REST Endpoints

### 1. Endpoints de prévision

#### `POST /forecast` et `POST /demand_forecast`

**Utilisation**: Génération de prévisions de demande avec plusieurs modèles

**Paramètres**:
```json
{
  "product_id": "string",
  "days": "number (default: 30)",
  "language": "string (en|fr|ar)"
}
```

**Réponse**:
```json
{
  "forecast": [
    {"date": "2025-07-19", "value": 123.45, "lower_bound": 100.0, "upper_bound": 150.0}
  ],
  "metrics": {
    "mape": 3.2,
    "rmse": 12.5,
    "accuracy": 96.8
  },
  "chart_data": [plotly_chart_object],
  "plot_html": "html_string"
}
```

**Implémentation backend**:
```python
@app.route('/forecast', methods=['POST'])
@app.route('/demand_forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    product_id = data.get('product_id', 'smartphone')
    days = data.get('days', 30)
    language = data.get('language', 'en')
    
    try:
        # Initialiser le moteur de prévision
        forecasting_engine = ForecastingEngine(merged_data)
        
        # Entraîner les modèles
        arima_model = forecasting_engine.train_arima_model(
            product_id, 
            exog_features=['temperature', 'precipitation']
        )
        
        # Générer les prévisions
        forecast_data = forecasting_engine.generate_forecast(
            product_id, 
            days, 
            include_confidence=True
        )
        
        # Créer la visualisation
        viz_engine = VisualizationEngine()
        plot_html = viz_engine.create_forecast_plot(
            historical_data, 
            forecast_data
        )
        
        return jsonify({
            'forecast': forecast_data.to_dict('records'),
            'plot_html': plot_html,
            'metrics': calculate_forecast_metrics(forecast_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### 2. Endpoints d'optimisation

#### `POST /optimize` et `POST /inventory_optimize`

**Utilisation**: Optimisation des niveaux de stock et stratégies de commande

**Paramètres**:
```json
{
  "product_id": "string",
  "holding_cost": "number",
  "ordering_cost": "number", 
  "lead_time": "number",
  "service_level": "number (0-1)",
  "forecast_data": "array"
}
```

**Implémentation**:
```python
@app.route('/optimize', methods=['POST'])
@app.route('/inventory_optimize', methods=['POST'])
def optimize_inventory():
    data = request.get_json()
    
    try:
        optimizer = InventoryOptimizer(merged_data)
        
        # Calculer EOQ
        eoq = optimizer.calculate_eoq(
            demand=data.get('annual_demand', 1000),
            order_cost=data.get('ordering_cost', 100),
            holding_cost=data.get('holding_cost', 10)
        )
        
        # Calculer point de réapprovisionnement
        reorder_point = optimizer.calculate_reorder_point(
            avg_demand=data.get('avg_demand', 50),
            lead_time=data.get('lead_time', 7),
            safety_stock=data.get('safety_stock', 25)
        )
        
        # Optimisation multi-objectifs
        optimization_results = optimizer.optimize_multi_echelon(
            locations=data.get('locations', ['warehouse_a', 'warehouse_b']),
            demands=data.get('demands', [100, 150]),
            costs=data.get('costs', [[10, 15], [12, 8]])
        )
        
        return jsonify({
            'eoq': eoq,
            'reorder_point': reorder_point,
            'optimization_results': optimization_results,
            'total_cost': calculate_total_cost(optimization_results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### 3. Assistant IA conversationnel

#### `POST /ask`

**Utilisation**: Questions en langage naturel avec RAG

**Implémentation avec RAG**:
```python
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('query', '')
    language = data.get('language', 'en')
    
    try:
        # Recherche sémantique dans la base de connaissances
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Construction du prompt avec contexte
        prompt = f"""
        Contexte supply chain: {context}
        
        Question: {query}
        
        Répondez en {language} de manière professionnelle et précise:
        """
        
        # Génération de réponse avec LLM
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=256,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # Génération audio si demandée
        audio_file = None
        if data.get('generate_audio', False):
            audio_file = generate_speech(response, language)
        
        return jsonify({
            'response': response,
            'confidence': 0.85,
            'sources': [{'title': doc.metadata.get('title', 'Document'), 
                        'relevance': 0.9} for doc in docs],
            'audio_url': f"/audio/{os.path.basename(audio_file)}" if audio_file else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### 4. Détection d'anomalies

#### `POST /anomaly_detection`

**Utilisation**: Détection d'anomalies avec Isolation Forest

```python
@app.route('/anomaly_detection', methods=['POST'])
def detect_anomalies():
    data = request.get_json()
    product_id = data.get('product_id')
    
    try:
        # Préparation des données
        product_data = merged_data[merged_data['product_id'] == product_id]
        features = ['demand', 'inventory', 'lead_time']
        X = product_data[features].values
        
        # Entraînement du détecteur d'anomalies
        detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        anomalies = detector.fit_predict(X)
        
        # Identification des anomalies
        anomaly_indices = np.where(anomalies == -1)[0]
        anomaly_data = []
        
        for idx in anomaly_indices:
            row = product_data.iloc[idx]
            anomaly_data.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'demand': row['demand'],
                'expected_range': [row['demand'] * 0.8, row['demand'] * 1.2],
                'severity': 'high' if abs(row['demand'] - row['demand'].mean()) > 2 * row['demand'].std() else 'medium'
            })
        
        # Visualisation
        viz_engine = VisualizationEngine()
        plot_html = viz_engine.create_anomaly_plot(product_data, anomaly_indices)
        
        return jsonify({
            'product_id': product_id,
            'anomalies': anomaly_data,
            'anomaly_count': len(anomaly_data),
            'analysis': f"Détectées {len(anomaly_data)} anomalies sur {len(product_data)} points de données.",
            'plot_html': plot_html
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### 5. Analyse de scénarios

#### `POST /scenario_analysis`

**Utilisation**: Simulation de scénarios "what-if"

```python
@app.route('/scenario_analysis', methods=['POST'])
def analyze_scenario():
    data = request.get_json()
    
    try:
        baseline_data = data.get('baseline_data')
        scenarios = data.get('scenarios', [])
        
        results = []
        
        for scenario in scenarios:
            # Simulation Monte Carlo
            simulation_results = run_monte_carlo_simulation(
                baseline_data,
                scenario['parameters'],
                n_iterations=1000
            )
            
            # Calcul des KPIs
            kpis = calculate_scenario_kpis(simulation_results)
            
            results.append({
                'name': scenario['name'],
                'kpis': kpis,
                'risk_metrics': calculate_risk_metrics(simulation_results),
                'recommended_actions': generate_recommendations(kpis)
            })
        
        # Analyse comparative
        comparative_analysis = generate_comparative_analysis(results)
        
        return jsonify({
            'scenarios_results': results,
            'comparative_analysis': comparative_analysis,
            'plot_html': create_scenario_comparison_plot(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

## Frontend Architecture

### Structure des composants

Le frontend React Native utilise une architecture modulaire avec séparation claire des responsabilités :

#### 1. Services API (apiClient.ts)

**Classe ApiClient**: Gestion centralisée des appels API avec fallbacks

```typescript
export class ApiClient {
  private async request(
    endpoint: string, 
    method = 'GET', 
    params: Record<string, any> = {}, 
    body: Record<string, any> = {}, 
    isFormData = false
  ) {
    try {
      // Configuration de l'URL de base avec tunnel Cloudflare
      const baseUrl = await getBaseUrl();
      const url = new URL(`${baseUrl}${endpoint}`);
      
      // Ajout des paramètres par défaut
      if ((endpoint.includes('forecast') || endpoint.includes('optimize')) && 
          method === 'POST' && body && !body.product_id) {
        body.product_id = body.product_id || "1001";
      }
      
      // Configuration des en-têtes
      const headers: HeadersInit = {};
      if (!isFormData) {
        headers['Content-Type'] = 'application/json';
      }
      
      // Exécution de la requête avec timeout
      const response = await fetchWithTimeout(url.toString(), {
        method,
        headers,
        body: method !== 'GET' && Object.keys(body).length > 0 
          ? (isFormData ? body as unknown as FormData : JSON.stringify(body))
          : undefined
      });
      
      // Gestion des erreurs
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `API Error: ${response.status}`);
      }
      
      // Traitement des réponses PDF
      if (response.headers.get('content-type')?.includes('application/pdf')) {
        return { blob: await response.blob(), url: response.url };
      }
      
      return await response.json();
      
    } catch (error) {
      console.error(`API request failed: ${error.message}`);
      
      // Fallback vers données mock en cas d'échec
      return this.provideMockFallback(endpoint, body);
    }
  }
```

#### 2. Gestion des tunnels Cloudflare

```typescript
// Stockage sécurisé de l'URL du tunnel
export const saveTunnelUrl = async (url: string) => {
  try {
    await SecureStore.setItemAsync(TUNNEL_KEY, url);
    return true;
  } catch (error) {
    console.error('Failed to save tunnel URL:', error);
    return false;
  }
};

// Récupération de l'URL de base
const getBaseUrl = async () => {
  const tunnelUrl = await getTunnelUrl();
  if (tunnelUrl) return tunnelUrl;
  
  // Adaptation pour Android Emulator
  if (Platform.OS === 'android') {
    return API_URL.replace('localhost', '10.0.2.2');
  }
  
  return API_URL;
};
```

### Consommation de l'API côté Frontend

#### 1. Module de prévision (forecast.tsx)

```typescript
export default function ForecastScreen() {
  const [productId, setProductId] = useState('');
  const [days, setDays] = useState('30');
  const [loading, setLoading] = useState(false);
  const [forecastData, setForecastData] = useState<ForecastResult | null>(null);
  
  const generateForecast = async () => {
    if (!productId) {
      setError('Please enter a product ID');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      // Appel API avec gestion d'erreurs
      const response = await api.getForecast({
        product_id: productId,
        days: parseInt(days) || 30,
      });
      
      setForecastData(response);
      
      // Génération de visualisation Plotly
      if (response.chart_data) {
        const htmlContent = generatePlotHtml(response.chart_data);
        setPlotHtml(htmlContent);
      }
    } catch (error) {
      console.error('Forecast error:', error);
      setError('Failed to generate forecast. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  // Génération HTML pour visualisation Plotly
  const generatePlotHtml = (chartData: any) => {
    return `
      <!DOCTYPE html>
      <html>
        <head>
          <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
          <style>
            body { margin: 0; padding: 0; background: ${colorScheme === 'dark' ? '#151718' : '#fff'}; }
          </style>
        </head>
        <body>
          <div id="chart"></div>
          <script>
            const data = ${JSON.stringify(chartData)};
            const layout = {
              margin: { t: 10, r: 10, l: 50, b: 50 },
              paper_bgcolor: '${colorScheme === 'dark' ? '#151718' : '#fff'}',
              font: { color: '${colorScheme === 'dark' ? '#ECEDEE' : '#11181C'}' }
            };
            Plotly.newPlot('chart', data, layout, { responsive: true });
          </script>
        </body>
      </html>
    `;
  };
```

#### 2. Module d'optimisation (inventory.tsx)

```typescript
const optimizeInventory = async () => {
  if (!productId) {
    setError('Please enter a product ID');
    return;
  }
  
  setLoading(true);
  setError('');
  
  try {
    // Appel API d'optimisation avec paramètres
    const response = await api.optimizeInventory({
      forecast_data: [], // Données de prévision précédentes
      product_id: productId,
      holding_cost: parseFloat(holdingCost) || 10,
      ordering_cost: parseFloat(orderingCost) || 100,
      lead_time: parseInt(leadTime) || 7,
      service_level: parseFloat(serviceLevel) || 0.95,
    });
    
    setOptimizationData(response);
    
    // Visualisation des résultats d'optimisation
    if (response.chart_data) {
      const htmlContent = generatePlotHtml(response.chart_data);
      setPlotHtml(htmlContent);
    }
  } catch (error) {
    console.error('Inventory optimization error:', error);
    setError('Failed to optimize inventory. Please try again.');
  } finally {
    setLoading(false);
  }
};
```

#### 3. Assistant IA conversationnel (assistant.tsx)

```typescript
const handleSend = async () => {
  if (!query.trim()) return;
  
  // Ajout du message utilisateur
  const userMessage: Message = {
    id: Date.now().toString(),
    text: query,
    isUser: true,
    timestamp: new Date(),
  };
  
  setMessages(prev => [...prev, userMessage]);
  const userQuery = query;
  setQuery('');
  setLoading(true);
  
  try {
    // Appel API assistant avec support multilingue
    const response = await api.askQuestion(userQuery);
    
    const botMessage: Message = {
      id: (Date.now() + 1).toString(),
      text: response.response || "I'm sorry, I couldn't process your request.",
      isUser: false,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, botMessage]);
    
    // Lecture vocale de la réponse si disponible
    if (response.audio_url) {
      playAudioResponse(response.audio_url);
    }
  } catch (error) {
    console.error('Error asking question:', error);
    
    const errorMessage: Message = {
      id: (Date.now() + 1).toString(),
      text: "Sorry, I encountered an error. Please try again.",
      isUser: false,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, errorMessage]);
  } finally {
    setLoading(false);
  }
};
```

---

## Types de données et interfaces

### Interfaces TypeScript (api.ts)

```typescript
// Interfaces de réponse API
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
  chart_data?: any[];
  plot_html?: string;
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
  chart_data?: any[];
}

export interface QAResponse {
  answer: string;
  confidence: number;
  sources?: {
    title: string;
    relevance: number;
  }[];
  audio_url?: string;
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
```

### Paramètres d'API

```typescript
export interface ForecastParams {
  product_id: string;
  days?: number;
  language?: string;
}

export interface OptimizeInventoryParams {
  forecast_data: any[];
  product_id: string;
  holding_cost: number;
  ordering_cost: number;
  lead_time: number;
  service_level: number;
  language?: string;
}

export interface QuestionParams {
  query: string;
  language?: string;
  generate_audio?: boolean;
}
```

---

## Gestion des états et contextes

### Context de données global (DataContext.tsx)

```typescript
interface DataContextType {
  selectedProduct: string | null;
  forecastData: ForecastData[] | null;
  anomalies: Anomaly[] | null;
  scenario: ScenarioType | null;
  selectProduct: (id: string) => void;
  setForecastData: (data: ForecastData[] | null) => void;
  setAnomalies: (data: Anomaly[] | null) => void;
  setScenario: (type: ScenarioType | null) => void;
}

export const DataProvider = ({ children }: { children: ReactNode }) => {
  const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
  const [forecastData, setForecastData] = useState<ForecastData[] | null>(null);
  const [anomalies, setAnomalies] = useState<Anomaly[] | null>(null);
  const [scenario, setScenario] = useState<ScenarioType | null>(null);

  const selectProduct = (id: string) => {
    setSelectedProduct(id);
    // Déclencher le rechargement des données pour ce produit
    refreshProductData(id);
  };

  return (
    <DataContext.Provider value={{
      selectedProduct,
      forecastData,
      anomalies,
      scenario,
      selectProduct,
      setForecastData,
      setAnomalies,
      setScenario,
    }}>
      {children}
    </DataContext.Provider>
  );
};
```

---

## Sécurité et authentification

### Stockage sécurisé

```typescript
// Utilisation d'Expo SecureStore pour les données sensibles
import * as SecureStore from 'expo-secure-store';

export const saveTunnelUrl = async (url: string) => {
  try {
    await SecureStore.setItemAsync(TUNNEL_KEY, url);
    return true;
  } catch (error) {
    console.error('Failed to save tunnel URL:', error);
    return false;
  }
};

export const getTunnelUrl = async (): Promise<string | null> => {
  try {
    return await SecureStore.getItemAsync(TUNNEL_KEY);
  } catch (error) {
    console.error('Failed to get tunnel URL:', error);
    return null;
  }
};
```

### Validation des entrées

```python
# Validation côté backend
def validate_forecast_params(data):
    """Valide les paramètres de prévision"""
    required_fields = ['product_id']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validation des types
    if 'days' in data:
        try:
            days = int(data['days'])
            if days < 1 or days > 365:
                raise ValueError("Days must be between 1 and 365")
        except (ValueError, TypeError):
            raise ValueError("Days must be a valid integer")
    
    return True
```

---

## Déploiement et infrastructure

### Configuration Cloudflare Tunnel

```python
class CloudflaredTunnel:
    def __init__(self, port=5000):
        self.port = port
        self.process = None
        self.url = None
        self.cloudflared_path = self._get_cloudflared_path()
    
    def _download_cloudflared(self):
        """Télécharge l'exécutable Cloudflare si nécessaire"""
        system = platform.system().lower()
        arch = platform.machine().lower()
        
        if system == 'windows':
            if arch in ['amd64', 'x86_64']:
                url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
                filename = "cloudflared.exe"
            else:
                url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-386.exe"
                filename = "cloudflared.exe"
        elif system == 'darwin':  # macOS
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz"
            filename = "cloudflared"
        else:  # Linux
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            filename = "cloudflared"
        
        # Téléchargement et installation
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        # Permissions d'exécution sur Unix
        if system != 'windows':
            os.chmod(filename, 0o755)
        
        return filename
```

### Configuration pour Google Colab

```python
# colab_config.py
def setup_colab_environment():
    """Configure l'environnement Google Colab pour optimiser les performances"""
    
    if 'google.colab' in sys.modules:
        print("🔧 Configuration de l'environnement Google Colab...")
        
        # Nettoyage de la mémoire GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"✅ GPU détecté: {torch.cuda.get_device_name(0)}")
            print(f"💾 Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        
        # Configuration des variables d'environnement
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Fonction de nettoyage mémoire
        def clean_memory():
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return clean_memory
    
    return lambda: None

def optimize_model_loading(model_name):
    """Optimise le chargement des modèles selon l'environnement"""
    base_config = {
        'torch_dtype': torch.float16,
        'device_map': 'auto',
        'low_cpu_mem_usage': True
    }
    
    if 'google.colab' in sys.modules:
        # Configuration optimisée pour Colab
        base_config.update({
            'max_memory': {0: "13GB"},  # Limite mémoire GPU
            'offload_folder': "./offload",
            'offload_state_dict': True
        })
    
    return base_config
```

---

## Optimisations et performances

### Optimisation des modèles IA

```python
# Quantification 4-bit pour réduire l'empreinte mémoire
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Configuration du pipeline avec optimisations
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
```

### Optimisation des requêtes API

```typescript
// Timeout et retry pour les requêtes
const fetchWithTimeout = async (url: string, options: RequestInit = {}, timeout = 10000) => {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
};

// Mise en cache des réponses API
class ApiCache {
  private cache = new Map<string, {data: any, timestamp: number}>();
  private readonly TTL = 5 * 60 * 1000; // 5 minutes
  
  get(key: string) {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.TTL) {
      return cached.data;
    }
    this.cache.delete(key);
    return null;
  }
  
  set(key: string, data: any) {
    this.cache.set(key, {data, timestamp: Date.now()});
  }
}
```

### Optimisation de la visualisation

```python
def create_optimized_plot(data, max_points=1000):
    """Crée un graphique optimisé en réduisant le nombre de points si nécessaire"""
    if len(data) > max_points:
        # Sous-échantillonnage intelligent
        step = len(data) // max_points
        data = data[::step]
    
    fig = go.Figure()
    
    # Configuration optimisée pour les performances
    fig.update_layout(
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
    )
    
    return fig.to_html(
        include_plotlyjs='cdn',
        config={'displayModeBar': False, 'responsive': True}
    )
```

---

## Conclusion

RawajAI représente une solution complète et moderne pour la gestion intelligente de la chaîne d'approvisionnement. L'architecture modulaire, l'intégration d'IA avancée et l'interface utilisateur intuitive en font un outil puissant pour l'optimisation des opérations de supply chain.

### Points clés de l'architecture :

1. **Backend robuste** avec modèles d'IA optimisés et API REST complète
2. **Frontend réactif** avec interface native mobile et visualisations interactives
3. **IA conversationnelle** avec support multilingue et RAG
4. **Optimisations** pour performances et utilisation en environnements contraints
5. **Infrastructure** scalable avec tunneling Cloudflare

Cette documentation fournit une base solide pour le développement, la maintenance et l'extension de la plateforme RawajAI.
