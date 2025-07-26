fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
requests==2.31.0
python-dotenv==1.0.0

!pip install transformers langchain-huggingface kaleido==0.2.1 torch torchaudio langchain_community bitsandbytes autoawq accelerate sentencepiece protobuf pulp scipy plotly==5.24.1 reportlab gtts pytrends faiss-cpu
!pip install kaleido==0.2.1 plotly==5.24.1

# Import Colab configuration
from colab_config import setup_colab_environment, optimize_model_loading

# Setup Colab environment
clean_memory = setup_colab_environment()

# Rest of imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import base64
from datetime import datetime, timedelta
import pytrends.request
from pytrends.request import TrendReq
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import pulp
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from gtts import gTTS
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import os
import requests
import json
import tempfile
import subprocess
import threading
import socket
import time
import platform
import re

app = Flask(__name__)

# Configuration
os.environ["HF_TOKEN"] = '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'
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL = "openai/whisper-base"
DATA_PATH = "supply_chain_data.csv"
WEATHER_DATA_PATH = "weather_data.csv"
TRENDS_DATA_PATH = "trends_data.csv"
VECTOR_DB_PATH = "supply_chain_faiss_index"
REPORT_PATH = "reports"
AUDIO_PATH = "audio"

# Ensure directories exist
for path in [VECTOR_DB_PATH, REPORT_PATH, AUDIO_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

# Initialize language model with optimizations
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Get optimized model settings
model_settings = optimize_model_loading(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    **model_settings,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
)

# Clean up memory after model loading
clean_memory()

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

# Clean memory again after pipeline setup
clean_memory()

# Initialize Whisper for speech recognition
try:
    whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
    whisper_model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    def transcribe_audio(audio_file, target_language="en"):
        """Transcribe audio file using Whisper"""
        # Load audio
        audio_array, sampling_rate = torchaudio.load(audio_file)
        # Convert to mono if stereo
        if audio_array.shape[0] > 1:
            audio_array = torch.mean(audio_array, dim=0, keepdim=True)
        
        # Resample if needed
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio_array = resampler(audio_array)
            sampling_rate = 16000
        
        # Process audio with Whisper
        input_features = whisper_processor(audio_array.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
        forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language=target_language, task="transcribe")
        
        # Generate transcription
        predicted_ids = whisper_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
except Exception as e:
    print(f"Whisper initialization failed: {e}. Speech recognition will not be available.")
    
    def transcribe_audio(audio_file, target_language="en"):
        return f"Speech recognition unavailable. Error: {str(e)}"

# Text-to-Speech function using gTTS
def generate_speech(text, language="en"):
    """Generate speech from text using Google Text-to-Speech"""
    try:
        # Validate input text
        if not text or not isinstance(text, str):
            print("Invalid text for speech generation")
            return None
            
        # Limit text length to avoid issues with very long text
        max_length = 4000  # gTTS has limitations on text length
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Map language code for gTTS
        lang_map = {
            "en": "en",
            "fr": "fr",
            "ar": "ar"
        }
        lang = lang_map.get(language, "en")
        
        # Ensure AUDIO_PATH directory exists
        if not os.path.exists(AUDIO_PATH):
            os.makedirs(AUDIO_PATH, exist_ok=True)
        
        # Generate speech
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save to file with proper error handling
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{AUDIO_PATH}/response_{timestamp}.mp3"
        
        # Try saving the file
        try:
            tts.save(filename)
        except Exception as save_error:
            print(f"Error saving speech file: {save_error}")
            # Try with a simpler filename if there might be path issues
            simple_filename = f"{AUDIO_PATH}/response.mp3"
            tts.save(simple_filename)
            filename = simple_filename
        
        # Verify the file was created
        if os.path.exists(filename):
            print(f"Speech file successfully created: {filename}")
            # Return relative path for URL construction
            return filename
        else:
            print("Speech file was not created")
            return None
            
    except Exception as e:
        print(f"Speech generation failed: {e}")
        return None

# Load and process data
def load_data():
    """Load all necessary data or create simulated data if files don't exist"""
    # Supply chain data
    if os.path.exists(DATA_PATH):
        supply_chain_data = pd.read_csv(DATA_PATH)
    else:
        # Create simulated data
        print("Creating simulated supply chain data...")
        products = ['smartphone', 'laptop', 'tablet', 'headphones', 'smartwatch']
        locations = ['warehouse_a', 'warehouse_b', 'warehouse_c', 'store_1', 'store_2']
        
        # Generate 1 year of daily data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31')
        rows = []
        
        for product in products:
            base_demand = np.random.randint(50, 200)
            for date in dates:
                # Simulate seasonality and weekly patterns
                seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                weekly_factor = 1.0 + 0.2 * (date.dayofweek < 5)  # Higher on weekdays
                
                # Add some randomness
                random_factor = np.random.normal(1.0, 0.1)
                
                demand = int(base_demand * seasonal_factor * weekly_factor * random_factor)
                
                # Generate inventory levels for each location
                for location in locations:
                    inventory = np.random.randint(demand // 2, demand * 2)
                    cost = np.random.randint(5, 100)
                    lead_time = np.random.randint(1, 10)
                    
                    rows.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'product_id': product,
                        'location': location,
                        'demand': demand,
                        'inventory': inventory,
                        'cost': cost,
                        'lead_time': lead_time
                    })
        
        supply_chain_data = pd.DataFrame(rows)
        supply_chain_data.to_csv(DATA_PATH, index=False)
    
    # Weather data
    if os.path.exists(WEATHER_DATA_PATH):
        weather_data = pd.read_csv(WEATHER_DATA_PATH)
    else:
        # Create simulated weather data
        print("Creating simulated weather data...")
        dates = pd.date_range(start='2024-01-01', end='2024-12-31')
        weather_rows = []
        
        for date in dates:
            # Simulate seasonal temperature changes
            base_temp = 15  # baseline temperature
            seasonal_temp = base_temp + 15 * np.sin(2 * np.pi * date.dayofyear / 365)
            temp_random = np.random.normal(0, 3)
            temperature = seasonal_temp + temp_random
            
            # Simulate precipitation (more in winter, less in summer)
            seasonal_factor = 1 - np.sin(2 * np.pi * date.dayofyear / 365) * 0.5
            precipitation_prob = 0.3 * seasonal_factor
            precipitation = np.random.exponential(5) if np.random.random() < precipitation_prob else 0
            
            # Simulate wind speed
            wind_speed = np.random.gamma(2, 2)
            
            weather_rows.append({
                'date': date.strftime('%Y-%m-%d'),
                'temperature': round(temperature, 1),
                'precipitation': round(precipitation, 1),
                'wind_speed': round(wind_speed, 1)
            })
        
        weather_data = pd.DataFrame(weather_rows)
        weather_data.to_csv(WEATHER_DATA_PATH, index=False)
    
    # Google Trends data
    if os.path.exists(TRENDS_DATA_PATH):
        trends_data = pd.read_csv(TRENDS_DATA_PATH)
    else:
        # Try to fetch real Google Trends data or simulate if it fails
        print("Attempting to fetch Google Trends data...")
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            keywords = ['smartphone', 'laptop', 'tablet', 'headphones', 'smartwatch']
            pytrends.build_payload(keywords, timeframe='today 12-m')
            trends_data = pytrends.interest_over_time()
            trends_data.reset_index(inplace=True)
            trends_data.to_csv(TRENDS_DATA_PATH, index=False)
        except Exception as e:
            print(f"Failed to fetch Google Trends data: {e}. Creating simulated data...")
            # Create simulated trends data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31')
            trends_rows = []
            
            keywords = ['smartphone', 'laptop', 'tablet', 'headphones', 'smartwatch']
            for date in dates:
                row = {'date': date.strftime('%Y-%m-%d')}
                
                for keyword in keywords:
                    # Base interest with seasonality
                    base_interest = 50
                    seasonal_factor = 1.0 + 0.5 * np.sin(2 * np.pi * date.dayofyear / 365)
                    
                    # Add product-specific patterns
                    if keyword == 'smartphone':
                        # Higher interest around new product launches (Q3)
                        product_factor = 1.0 + 0.5 * (date.month in [8, 9])
                    elif keyword == 'laptop':
                        # Higher during back to school season
                        product_factor = 1.0 + 0.6 * (date.month in [7, 8])
                    elif keyword == 'tablet':
                        # Higher during holiday season
                        product_factor = 1.0 + 0.7 * (date.month in [11, 12])
                    else:
                        product_factor = 1.0
                    
                    # Add some randomness
                    random_factor = np.random.normal(1.0, 0.1)
                    
                    interest = int(base_interest * seasonal_factor * product_factor * random_factor)
                    interest = max(0, min(100, interest))  # Clamp between 0-100
                    
                    row[keyword] = interest
                
                trends_rows.append(row)
            
            trends_data = pd.DataFrame(trends_rows)
            trends_data.to_csv(TRENDS_DATA_PATH, index=False)
    
    return supply_chain_data, weather_data, trends_data

# Load data
supply_chain_data, weather_data, trends_data = load_data()

# Merge data for modeling
def merge_data_for_modeling():
    """Merge supply chain, weather, and trends data for modeling"""
    # Convert date columns to datetime
    supply_chain_data['date'] = pd.to_datetime(supply_chain_data['date'])
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    trends_data['date'] = pd.to_datetime(trends_data['date'])
    
    # Merge data
    merged_data = supply_chain_data.merge(weather_data, on='date', how='left')
    merged_data = merged_data.merge(trends_data, on='date', how='left')
    
    # Fill missing values if any
    merged_data.ffill(inplace=True)
    merged_data.bfill(inplace=True)
    
    return merged_data

merged_data = merge_data_for_modeling()

# Sample supply chain documents for RAG
supply_chain_docs = [
    "Just-in-time inventory reduces holding costs but increases risk of stockouts",
    "EOQ (Economic Order Quantity) balances ordering and holding costs",
    "Safety stock protects against demand variability and lead time uncertainty",
    "Cross-docking reduces inventory handling and storage time",
    "ABC analysis prioritizes inventory management based on item value",
    "La gestion des stocks juste à temps réduit les coûts de stockage mais augmente le risque de rupture de stock",
    "Le QEC (Quantité Économique de Commande) équilibre les coûts de commande et de stockage",
    "Le stock de sécurité protège contre la variabilité de la demande et l'incertitude des délais de livraison",
    "Le cross-docking réduit la manipulation et le temps de stockage des inventaires",
    "L'analyse ABC priorise la gestion des stocks en fonction de la valeur des articles",
    "إدارة المخزون في الوقت المناسب تقلل تكاليف الاحتفاظ بالمخزون ولكنها تزيد من خطر نفاد المخزون",
    "كمية الطلب الاقتصادية توازن بين تكاليف الطلب وتكاليف الاحتفاظ بالمخزون",
    "مخزون الأمان يحمي من تقلبات الطلب وعدم اليقين في وقت التسليم",
    "التحميل المتقاطع يقلل من مناولة المخزون ووقت التخزين",
    "تحليل ABC يعطي الأولوية لإدارة المخزون على أساس قيمة العناصر",
    "Adverse weather conditions like heavy rain or snow can delay deliveries and increase lead time",
    "Seasonal demand patterns affect inventory planning, with higher demand during holidays",
    "Supply chain disruptions can occur due to natural disasters, labor strikes, or geopolitical events",
    "Demand forecasting accuracy improves with external data like weather and search trends",
    "Warehouse capacity constraints can limit inventory levels during peak seasons",
    "Les conditions météorologiques défavorables comme la pluie ou la neige peuvent retarder les livraisons",
    "Les tendances de recherche Google peuvent prédire la demande future pour certains produits",
    "L'optimisation des niveaux de stock réduit les coûts tout en maintenant le service client",
    "الظروف الجوية السيئة مثل المطر الغزير أو الثلج يمكن أن تؤخر التسليم وتزيد وقت التسليم",
    "أنماط الطلب الموسمية تؤثر على تخطيط المخزون، مع ارتفاع الطلب خلال العطلات",
    "يمكن أن تحدث اضطرابات في سلسلة التوريد بسبب الكوارث الطبيعية أو الإضرابات العمالية"
]
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Create vector store
vector_store = FAISS.from_texts(supply_chain_docs, embeddings)
vector_store.save_local(VECTOR_DB_PATH)

# Advanced forecasting models
class ForecastingEngine:
    """Engine for training and using different forecasting models"""
    
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.scalers = {}
        self.anomaly_detectors = {}
    
    def train_arima_model(self, product_id, exog_features=None):
        """Train an ARIMA model for a specific product"""
        product_data = self.data[self.data['product_id'] == product_id].copy()
        if product_data.empty:
            raise ValueError(f"No data found for product {product_id}")
        
        # Sort by date
        product_data = product_data.sort_values('date')
        
        # Prepare exogenous features if provided
        exog = None
        if exog_features and all(feature in product_data.columns for feature in exog_features):
            exog = product_data[exog_features]
        
        # Create proper time index
        product_data.reset_index(drop=True, inplace=True)
        
        # Train ARIMA model
        if exog is not None:
            model = SARIMAX(product_data['demand'], exog=exog, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7))
        else:
            model = ARIMA(product_data['demand'], order=(2, 1, 2))
        
        # Try different fit methods based on model type
        try:
            model_fit = model.fit()
        except:
            try:
                # Alternative fit method for some versions
                model_fit = model.fit(disp=False)
            except Exception as e:
                print(f"Model fitting error: {str(e)}")
                raise
        
        # Store the model and the last date for future forecasting
        self.models[product_id] = {
            'model': model_fit,
            'last_date': product_data['date'].max(),
            'exog_features': exog_features,
            'training_data': product_data  # Store training data for reference
        }
        
        return model_fit
    
    def train_prophet_model(self, product_id):
        """Train a Prophet model for a specific product"""
        product_data = self.data[self.data['product_id'] == product_id].copy()
        if product_data.empty:
            raise ValueError(f"No data found for product {product_id}")
        
        # Prepare data for Prophet
        prophet_data = product_data[['date', 'demand']].rename(columns={'date': 'ds', 'demand': 'y'})
        
        # Add regressor columns if available
        regressors = ['temperature', 'precipitation']
        for regressor in regressors:
            if regressor in product_data.columns:
                prophet_data[regressor] = product_data[regressor]
        
        # Add product-specific Google Trends data if available
        if product_id in product_data.columns:
            prophet_data[f'{product_id}_trend'] = product_data[product_id]
        
        # Create and train Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        
        # Add regressors
        for regressor in regressors:
            if regressor in prophet_data.columns:
                model.add_regressor(regressor)
        
        if f'{product_id}_trend' in prophet_data.columns:
            model.add_regressor(f'{product_id}_trend')
        
        model.fit(prophet_data)
        
        # Store the model and data
        self.models[f"prophet_{product_id}"] = {
            'model': model,
            'last_date': product_data['date'].max(),
            'prophet_data': prophet_data
        }
        
        return model
    
    def forecast(self, product_id, steps=30, model_type='arima', exog_future=None):
        """Generate forecast using the trained model"""
        model_key = product_id if model_type == 'arima' else f"prophet_{product_id}"
        
        if model_key not in self.models:
            if model_type == 'arima':
                self.train_arima_model(product_id)
            else:
                self.train_prophet_model(product_id)
        
        model_info = self.models[model_key]
        
        if model_type == 'arima':
            # ARIMA forecasting
            if exog_future is not None and model_info.get('exog_features'):
                forecast = model_info['model'].forecast(steps=steps, exog=exog_future)
            else:
                forecast = model_info['model'].forecast(steps=steps)
                
            # Create forecast dates
            last_date = model_info['last_date']
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=steps)
            
            # Create forecast DataFrame with proper index
            forecast_values = forecast.values if hasattr(forecast, 'values') else forecast
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast_values
            }, index=range(len(forecast_dates)))
            
            return forecast_df
        
        else:
            # Prophet forecasting
            model = model_info['model']
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=steps)
            
            # Add regressor values if provided
            if exog_future is not None:
                for regressor, values in exog_future.items():
                    if regressor in future.columns:
                        future.loc[future.ds > model_info['last_date'], regressor] = values
            
            # Make forecast
            forecast = model.predict(future)
            
            # Filter to only future dates
            forecast = forecast[forecast['ds'] > model_info['last_date']]
            
            # Rename columns for consistency
            forecast = forecast.rename(columns={'ds': 'date', 'yhat': 'forecast', 
                                              'yhat_lower': 'lower_bound', 
                                              'yhat_upper': 'upper_bound'})
            
            return forecast[['date', 'forecast', 'lower_bound', 'upper_bound']]
    
    def train_anomaly_detector(self, product_id):
        """Train an anomaly detector for a specific product"""
        product_data = self.data[self.data['product_id'] == product_id].copy()
        if product_data.empty:
            raise ValueError(f"No data found for product {product_id}")
        
        # Prepare features for anomaly detection
        features = ['demand']
        if 'inventory' in product_data.columns:
            features.append('inventory')
        
        # Add external features if available
        for col in ['temperature', 'precipitation', product_id]:
            if col in product_data.columns:
                features.append(col)
        
        # Scale the features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(product_data[features])
        
        # Train isolation forest
        detector = IsolationForest(contamination=0.05, random_state=42)
        detector.fit(scaled_data)
        
        # Store the model and scaler
        self.anomaly_detectors[product_id] = {
            'detector': detector,
            'scaler': scaler,
            'features': features
        }
        
        return detector
    
    def detect_anomalies(self, product_id, new_data=None):
        """Detect anomalies in historical or new data"""
        if product_id not in self.anomaly_detectors:
            self.train_anomaly_detector(product_id)
        
        detector_info = self.anomaly_detectors[product_id]
        
        if new_data is None:
            # Use historical data
            product_data = self.data[self.data['product_id'] == product_id].copy()
        else:
            product_data = new_data
        
        # Scale the features
        scaled_data = detector_info['scaler'].transform(product_data[detector_info['features']])
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = detector_info['detector'].predict(scaled_data)
        anomaly_scores = detector_info['detector'].decision_function(scaled_data)
        
        # Add results to the data
        product_data['anomaly'] = predictions
        product_data['anomaly_score'] = anomaly_scores
        
        # Filter anomalies
        anomalies = product_data[product_data['anomaly'] == -1].copy()
        
        return anomalies, product_data
    
    def analyze_anomalies(self, product_id, language="en"):
        """Analyze detected anomalies and provide insights"""
        anomalies, all_data = self.detect_anomalies(product_id)
        
        if anomalies.empty:
            if language == "fr":
                return "Aucune anomalie détectée pour ce produit."
            elif language == "ar":
                return "لم يتم اكتشاف أي شذوذ لهذا المنتج."
            else:
                return "No anomalies detected for this product."
        
        # Calculate statistics
        anomaly_pct = len(anomalies) / len(all_data) * 100
        avg_demand = all_data['demand'].mean()
        avg_anomaly_demand = anomalies['demand'].mean()
        demand_diff_pct = (avg_anomaly_demand - avg_demand) / avg_demand * 100
        
        # Generate insights based on the language
        if language == "fr":
            insights = f"Analyse des anomalies pour le produit {product_id}:\n"
            insights += f"- {len(anomalies)} anomalies détectées ({anomaly_pct:.1f}% des données)\n"
            insights += f"- Demande moyenne pendant les anomalies: {avg_anomaly_demand:.1f} "
            if demand_diff_pct > 0:
                insights += f"({demand_diff_pct:.1f}% au-dessus de la moyenne)\n"
            else:
                insights += f"({abs(demand_diff_pct):.1f}% en dessous de la moyenne)\n"
            
            # Add seasonal patterns if detected
            season_counts = anomalies['date'].dt.month.value_counts()
            if len(season_counts) > 0:
                top_month = season_counts.idxmax()
                month_names = {1: 'janvier', 2: 'février', 3: 'mars', 4: 'avril', 5: 'mai', 
                            6: 'juin', 7: 'juillet', 8: 'août', 9: 'septembre', 
                            10: 'octobre', 11: 'novembre', 12: 'décembre'}
                insights += f"- La plupart des anomalies se produisent en {month_names[top_month]}\n"
            
        elif language == "ar":
            insights = f"تحليل الشذوذ للمنتج {product_id}:\n"
            insights += f"- تم اكتشاف {len(anomalies)} حالة شاذة ({anomaly_pct:.1f}٪ من البيانات)\n"
            insights += f"- متوسط الطلب أثناء الشذوذ: {avg_anomaly_demand:.1f} "
            if demand_diff_pct > 0:
                insights += f"({demand_diff_pct:.1f}٪ فوق المتوسط)\n"
            else:
                insights += f"({abs(demand_diff_pct):.1f}٪ تحت المتوسط)\n"
            
            # Add seasonal patterns if detected
            season_counts = anomalies['date'].dt.month.value_counts()
            if len(season_counts) > 0:
                top_month = season_counts.idxmax()
                month_names = {1: 'يناير', 2: 'فبراير', 3: 'مارس', 4: 'أبريل', 5: 'مايو', 
                            6: 'يونيو', 7: 'يوليو', 8: 'أغسطس', 9: 'سبتمبر', 
                            10: 'أكتوبر', 11: 'نوفمبر', 12: 'ديسمبر'}
                insights += f"- تحدث معظم الشذوذ في {month_names[top_month]}\n"
        
        else:  # Default to English
            insights = f"Anomaly analysis for product {product_id}:\n"
            insights += f"- {len(anomalies)} anomalies detected ({anomaly_pct:.1f}% of data)\n"
            insights += f"- Average demand during anomalies: {avg_anomaly_demand:.1f} "
            if demand_diff_pct > 0:
                insights += f"({demand_diff_pct:.1f}% above average)\n"
            else:
                insights += f"({abs(demand_diff_pct):.1f}% below average)\n"
            
            # Add seasonal patterns if detected
            season_counts = anomalies['date'].dt.month.value_counts()
            if len(season_counts) > 0:
                top_month = season_counts.idxmax()
                month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 
                            6: 'June', 7: 'July', 8: 'August', 9: 'September', 
                            10: 'October', 11: 'November', 12: 'December'}
                insights += f"- Most anomalies occur in {month_names[top_month]}\n"
        
        return insights
    
    def simulate_scenario(self, product_id, scenario, language="en"):
        """Simulate different scenarios and their impact on forecasts"""
        # Define scenario impacts
        impacts = {
            "demand_increase": {"demand_factor": 1.2},
            "demand_decrease": {"demand_factor": 0.8},
            "supply_disruption": {"lead_time_factor": 2.0},
            "weather_extreme": {"temperature_factor": 1.5, "precipitation_factor": 2.0},
            "cost_increase": {"cost_factor": 1.3},
            "new_competitor": {"demand_factor": 0.85},
            "marketing_campaign": {"demand_factor": 1.5},
            "seasonal_peak": {"demand_factor": 2.0}
        }
        
        # Get base forecast
        base_forecast = self.forecast(product_id, steps=30)
        
        # Apply scenario impact
        if scenario in impacts:
            impact = impacts[scenario]
            scenario_forecast = base_forecast.copy()
            
            if "demand_factor" in impact:
                scenario_forecast['forecast'] = base_forecast['forecast'] * impact["demand_factor"]
            
            if "lower_bound" in scenario_forecast.columns and "upper_bound" in scenario_forecast.columns:
                scenario_forecast['lower_bound'] = base_forecast['lower_bound'] * impact.get("demand_factor", 1.0)
                scenario_forecast['upper_bound'] = base_forecast['upper_bound'] * impact.get("demand_factor", 1.0)
        else:
            # Unknown scenario, return base forecast
            scenario_forecast = base_forecast
        
        # Generate scenario description
        descriptions = {
            "en": {
                "demand_increase": f"Scenario: 20% increase in demand for {product_id}",
                "demand_decrease": f"Scenario: 20% decrease in demand for {product_id}",
                "supply_disruption": f"Scenario: Supply chain disruption doubling lead time for {product_id}",
                "weather_extreme": f"Scenario: Extreme weather conditions affecting {product_id}",
                "cost_increase": f"Scenario: 30% increase in costs for {product_id}",
                "new_competitor": f"Scenario: New competitor entering market, reducing demand by 15%",
                "marketing_campaign": f"Scenario: Marketing campaign increasing demand by 50%",
                "seasonal_peak": f"Scenario: Seasonal peak doubling demand for {product_id}"
            },
            "fr": {
                "demand_increase": f"Scénario: Augmentation de la demande de 20% pour {product_id}",
                "demand_decrease": f"Scénario: Diminution de la demande de 20% pour {product_id}",
                "supply_disruption": f"Scénario: Perturbation de la chaîne d'approvisionnement doublant le délai de livraison pour {product_id}",
                "weather_extreme": f"Scénario: Conditions météorologiques extrêmes affectant {product_id}",
                "cost_increase": f"Scénario: Augmentation des coûts de 30% pour {product_id}",
                "new_competitor": f"Scénario: Nouveau concurrent entrant sur le marché, réduisant la demande de 15%",
                "marketing_campaign": f"Scénario: Campagne marketing augmentant la demande de 50%",
                "seasonal_peak": f"Scénario: Pic saisonnier doublant la demande pour {product_id}"
            },
            "ar": {
                "demand_increase": f"سيناريو: زيادة الطلب بنسبة 20٪ على {product_id}",
                "demand_decrease": f"سيناريو: انخفاض الطلب بنسبة 20٪ على {product_id}",
                "supply_disruption": f"سيناريو: اضطراب في سلسلة التوريد مضاعفة وقت التسليم لـ {product_id}",
                "weather_extreme": f"سيناريو: ظروف جوية قاسية تؤثر على {product_id}",
                "cost_increase": f"سيناريو: زيادة التكاليف بنسبة 30٪ لـ {product_id}",
                "new_competitor": f"سيناريو: دخول منافس جديد إلى السوق، مما يقلل الطلب بنسبة 15٪",
                "marketing_campaign": f"سيناريو: حملة تسويقية تزيد الطلب بنسبة 50٪",
                "seasonal_peak": f"سيناريو: ذروة موسمية تضاعف الطلب على {product_id}"
            }
        }
        
        lang_key = language if language in descriptions else "en"
        description = descriptions[lang_key].get(scenario, f"Scenario: {scenario}")
        
        return scenario_forecast, base_forecast, description

# Initialize forecasting engine
forecasting_engine = ForecastingEngine(merged_data)

# Advanced Inventory Optimization
class InventoryOptimizer:
    """Advanced inventory optimization with multiple objectives and constraints"""
    
    def __init__(self, warehouse_capacities=None, forecast=None):
        """Initialize the optimizer with warehouse capacities and forecast data
        
        Args:
            warehouse_capacities (dict): Dictionary mapping warehouse IDs to their storage capacities
            forecast (dict): Dictionary mapping product IDs to their demand forecasts
        """
        # Initialize with default values if not provided
        self.warehouse_capacities = warehouse_capacities or {
            'warehouse_a': 1000,
            'warehouse_b': 1500,
            'warehouse_c': 2000
        }
        
        # Initialize with some default forecast if not provided
        self.forecast = forecast or {}
        
        # Extract locations and products
        self.locations = list(self.warehouse_capacities.keys())
        self.products = list(self.forecast.keys()) if self.forecast else []
        
        # Initialize optimization parameters
        self.holding_costs = {}
        self.transportation_costs = {}
        self.production_costs = {}
        self.lead_times = {}
        
        # Set default values for optimization parameters
        self._initialize_default_parameters()
    
    def _initialize_default_parameters(self):
        """Initialize default optimization parameters"""
        for product in self.products:
            # Default holding cost per unit per period
            self.holding_costs[product] = 10
            
            # Default production cost per unit
            self.production_costs[product] = 50
            
            # Default lead time in days
            self.lead_times[product] = 7
            
            # Default transportation costs between locations
            self.transportation_costs[product] = {
                from_loc: {to_loc: 5 for to_loc in self.locations}
                for from_loc in self.locations
            }
        
        # Print initialization for debugging
        print(f"Initialized InventoryOptimizer with {len(self.locations)} locations and {len(self.products)} products")
    
    def optimize_inventory(self, demand_forecast=None, warehouse_capacities=None, holding_costs=None, transportation_costs=None, lead_times=None, service_level=0.95):
        """Optimize inventory allocation across warehouses"""
        from scipy.optimize import linprog
        
        # Update parameters if provided
        if demand_forecast is not None:
            self.forecast = demand_forecast
            self.products = list(demand_forecast.keys())
        
        if warehouse_capacities is not None:
            self.warehouse_capacities = warehouse_capacities
        
        if holding_costs is not None:
            self.holding_costs = holding_costs
        
        if transportation_costs is not None:
            self.transportation_costs = transportation_costs
        
        if lead_times is not None:
            self.lead_times = lead_times
        
        # Coefficients for the objective function (minimize total cost)
        c = []
        for prod in self.products:
            c.append(self.holding_costs.get(prod, 0))  # Holding cost
            c.append(self.production_costs.get(prod, 0))  # Production cost
        
        # Constraints
        A = []
        b = []
        
        # 1. Demand fulfillment constraints
        for prod in self.products:
            constraint = [0] * len(c)
            idx = 0
            for loc in self.locations:
                constraint[idx] = 1  # Inventory
                idx += 1
                constraint[idx] = -self.lead_times.get(prod, 1)  # Production
                idx += 1
            
            A.append(constraint)
            b.append(self.forecast.get(prod, 0))
        
        # 2. Capacity constraints
        for loc in self.locations:
            constraint = [0] * len(c)
            idx = 0
            for prod in self.products:
                constraint[idx] = 1  # Inventory
                idx += 1
                constraint[idx] = 0  # Production
                idx += 1
            
            A.append(constraint)
            b.append(self.warehouse_capacities[loc])
        
        # Convert to linear programming format
        A_eq = None
        b_eq = None
        bounds = [(0, None)] * len(c)  # All variables >= 0
        
        # Solve the linear programming problem
        try:
            result = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if result.success:
                # Extract the optimized inventory and production levels
                optimized_levels = result.x
                
                # Prepare the results dictionary
                results = {
                    "status": "Optimal",
                    "total_cost": result.fun,
                    "inventory": {},
                    "production": {}
                }
                
                # Split into inventory and production plans
                idx = 0
                for prod in self.products:
                    inv_level = optimized_levels[idx]
                    prod_level = optimized_levels[idx + 1]
                    
                    results["inventory"][prod] = inv_level
                    results["production"][prod] = prod_level
                    
                    idx += 2
                
                # Format the results for presentation
                location_results = {}
                for loc in self.locations:
                    total_inv = 0
                    products_inv = {}
                    
                    idx = 0
                    for prod in self.products:
                        inv_level = optimized_levels[idx]
                        products_inv[prod] = float(inv_level)
                        total_inv += inv_level
                        idx += 2
                    
                    # Calculate capacity utilization
                    capacity = self.warehouse_capacities[loc]
                    utilization = (total_inv / capacity) * 100 if capacity > 0 else 0
                    
                    location_results[loc] = {
                        "total_inventory": float(total_inv),
                        "capacity": float(capacity),
                        "capacity_utilization": float(utilization),
                        "products": products_inv
                    }
                
                return {
                    "status": "Optimal",
                    "total_cost": float(result.fun),
                    "locations": location_results
                }
            else:
                return {"status": "No optimal solution found", "problem_status": result.message}
        except Exception as e:
            return {"status": "Error", "message": str(e)}
    
    def generate_recommendations(self, language="en"):
        """Generate inventory optimization recommendations based on the latest results"""
        # Default recommendations in English
        recommendations = [
            "Redistribute inventory to optimize warehouse space utilization",
            "Consider increasing capacity for warehouses exceeding 85% utilization",
            "Reduce inventory levels for slow-moving products",
            "Prioritize high-demand products in warehouses closer to customers",
            "Adjust reorder points based on lead time variability"
        ]
        
        # Translate recommendations based on language
        if language == "fr":
            recommendations = [
                "Redistribuer les stocks pour optimiser l'utilisation de l'espace d'entrepôt",
                "Envisager d'augmenter la capacité des entrepôts dépassant 85% d'utilisation",
                "Réduire les niveaux de stock pour les produits à rotation lente",
                "Prioriser les produits à forte demande dans les entrepôts proches des clients",
                "Ajuster les points de commande en fonction de la variabilité des délais d'approvisionnement"
            ]
        elif language == "ar":
            recommendations = [
                "إعادة توزيع المخزون لتحسين استخدام مساحة المستودع",
                "النظر في زيادة سعة المستودعات التي تتجاوز 85٪ من الاستخدام",
                "تقليل مستويات المخزون للمنتجات بطيئة الحركة",
                "إعطاء الأولوية للمنتجات ذات الطلب المرتفع في المستودعات القريبة من العملاء",
                "ضبط نقاط إعادة الطلب بناءً على تغير وقت التوريد"
            ]
        
        return "\n".join(recommendations)

# Initialize inventory optimizer
inventory_optimizer = InventoryOptimizer()

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Visualization functions
class VisualizationEngine:
    """Engine for creating various visualizations"""
    
    @staticmethod
    def create_forecast_chart(forecast_df, product_id, title=None, language="en"):
        """Create a forecast chart with confidence intervals if available"""
        fig = go.Figure()
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue')
        ))
        
        # Add confidence interval if available
        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['upper_bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['lower_bound'],
                mode='lines',
                name='Lower Bound',
                fill='tonexty',
                fillcolor='rgba(0, 176, 246, 0.2)',
                line=dict(width=0),
                showlegend=False
            ))
        
        # Set chart title based on language
        if title is None:
            if language == "fr":
                title = f"Prévision de demande pour {product_id}"
            elif language == "ar":
                title = f"توقعات الطلب لـ {product_id}"
            else:
                title = f"Demand Forecast for {product_id}"
        
        # Set axis labels based on language
        if language == "fr":
            xaxis_title = "Date"
            yaxis_title = "Demande prévue"
        elif language == "ar":
            xaxis_title = "التاريخ"
            yaxis_title = "الطلب المتوقع"
        else:
            xaxis_title = "Date"
            yaxis_title = "Forecasted Demand"
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend_title="Légende" if language == "fr" else "المفتاح" if language == "ar" else "Legend",
            hovermode="x unified",
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_inventory_chart(inventory_data, title=None, language="en"):
        """Create a bar chart of inventory levels by location"""
        # Verify inventory_data has the required structure
        if not isinstance(inventory_data, dict):
            print(f"Error: inventory_data must be a dict, got {type(inventory_data)}")
            # Create default data
            inventory_data = {"locations": {
                "warehouse_a": {"total_inventory": 0, "capacity_utilization": 0},
                "warehouse_b": {"total_inventory": 0, "capacity_utilization": 0}
            }}
        
        # Handle missing locations key
        if "locations" not in inventory_data:
            print("Error: inventory_data missing 'locations' key")
            inventory_data["locations"] = {
                "warehouse_a": {"total_inventory": 0, "capacity_utilization": 0},
                "warehouse_b": {"total_inventory": 0, "capacity_utilization": 0}
            }
            
        # Prepare data with safe access
        try:
            locations = list(inventory_data["locations"].keys())
            total_inventories = [data.get("total_inventory", 0) for _, data in inventory_data["locations"].items()]
            capacities = [data.get("capacity_utilization", 0) for _, data in inventory_data["locations"].items()]
        except Exception as e:
            print(f"Error extracting data for chart: {e}")
            # Provide fallback data
            locations = ["Location 1", "Location 2"]
            total_inventories = [0, 0]
            capacities = [0, 0]
        
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add inventory bars
        fig.add_trace(
            go.Bar(
                x=locations,
                y=total_inventories,
                name='Total Inventory',
                marker_color='rgb(55, 83, 109)'
            ),
            secondary_y=False,
        )
        
        # Add capacity utilization line
        fig.add_trace(
            go.Scatter(
                x=locations,
                y=capacities,
                name='Utilization %',
                mode='lines+markers',
                marker=dict(size=8),
                line=dict(width=2, color='rgb(255, 128, 0)')
            ),
            secondary_y=True,
        )
        
        # Set chart title based on language
        if title is None:
            if language == "fr":
                title = "Niveaux d'inventaire et utilisation par emplacement"
            elif language == "ar":
                title = "مستويات المخزون والاستخدام حسب الموقع"
            else:
                title = "Inventory Levels and Utilization by Location"
        
        # Set axis labels based on language
        if language == "fr":
            xaxis_title = "Emplacement"
            yaxis_title = "Inventaire total"
            yaxis2_title = "Utilisation de la capacité (%)"
        elif language == "ar":
            xaxis_title = "الموقع"
            yaxis_title = "إجمالي المخزون"
            yaxis2_title = "استخدام السعة (%)"
        else:
            xaxis_title = "Location"
            yaxis_title = "Total Inventory"
            yaxis2_title = "Capacity Utilization (%)"
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            legend_title="Légende" if language == "fr" else "المفتاح" if language == "ar" else "Legend",
            template="plotly_white"
        )
        
        # Update y-axes
        fig.update_yaxes(title_text=yaxis_title, secondary_y=False)
        fig.update_yaxes(title_text=yaxis2_title, secondary_y=True)
        
        return fig
    
    @staticmethod
    def create_product_pie_chart(inventory_data, location, title=None, language="en"):
        """Create a pie chart of product distribution at a specific location"""
        # Check if location exists
        if location not in inventory_data["locations"]:
            return None
        
        # Prepare data
        products = []
        values = []
        
        for prod, prod_data in inventory_data["locations"][location]["products"].items():
            if prod_data["total"] > 0:
                products.append(prod)
                values.append(prod_data["total"])
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=products,
            values=values,
            hole=.3,
            textinfo='percent',
            insidetextorientation='radial'
        )])
        
        # Set chart title based on language
        if title is None:
            if language == "fr":
                title = f"Distribution des produits à {location}"
            elif language == "ar":
                title = f"توزيع المنتجات في {location}"
            else:
                title = f"Product Distribution at {location}"
        
        # Update layout
        fig.update_layout(
            title=title,
            legend_title="Produits" if language == "fr" else "المنتجات" if language == "ar" else "Products",
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_anomaly_chart(product_data, anomalies, product_id, title=None, language="en"):
        """Create a chart highlighting anomalies in demand data"""
        fig = go.Figure()
        
        # Add normal data points
        normal_data = product_data[product_data['anomaly'] == 1]
        fig.add_trace(go.Scatter(
            x=normal_data['date'],
            y=normal_data['demand'],
            mode='lines+markers',
            name='Normal',
            line=dict(color='blue'),
            marker=dict(size=6)
        ))
        
        # Add anomaly points
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['date'],
                y=anomalies['demand'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='circle-open',
                    line=dict(width=2)
                )
            ))
        
        # Set chart title based on language
        if title is None:
            if language == "fr":
                title = f"Détection d'anomalies pour {product_id}"
            elif language == "ar":
                title = f"اكتشاف الشذوذ لـ {product_id}"
            else:
                title = f"Anomaly Detection for {product_id}"
        
        # Set axis labels based on language
        if language == "fr":
            xaxis_title = "Date"
            yaxis_title = "Demande"
        elif language == "ar":
            xaxis_title = "التاريخ"
            yaxis_title = "الطلب"
        else:
            xaxis_title = "Date"
            yaxis_title = "Demand"
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            hovermode="closest",
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_scenario_chart(scenario_forecast, base_forecast, product_id, scenario_name, language="en"):
        """Create a chart comparing baseline forecast with scenario forecast"""
        fig = go.Figure()
        
        # Add base forecast
        fig.add_trace(go.Scatter(
            x=base_forecast['date'],
            y=base_forecast['forecast'],
            mode='lines',
            name='Baseline Forecast',
            line=dict(color='blue')
        ))
        
        # Add scenario forecast
        fig.add_trace(go.Scatter(
            x=scenario_forecast['date'],
            y=scenario_forecast['forecast'],
            mode='lines',
            name='Scenario Forecast',
            line=dict(color='red')
        ))
        
        # Set title based on language
        if language == "fr":
            title = f"Analyse de scénario pour {product_id}: {scenario_name}"
            xaxis_title = "Date"
            yaxis_title = "Demande prévue"
            legend_title = "Scénarios"
        elif language == "ar":
            title = f"تحليل السيناريو لـ {product_id}: {scenario_name}"
            xaxis_title = "التاريخ"
            yaxis_title = "الطلب المتوقع"
            legend_title = "السيناريوهات"
        else:
            title = f"Scenario Analysis for {product_id}: {scenario_name}"
            xaxis_title = "Date"
            yaxis_title = "Forecasted Demand"
            legend_title = "Scenarios"
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend_title=legend_title,
            hovermode="x unified",
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def convert_fig_to_image(fig, format='png'):
        """Convert a plotly figure to an image"""
        try:
            # Try using to_image (requires compatible kaleido)
            img_bytes = fig.to_image(format=format)
            return img_bytes
        except Exception as e:
            print(f"Warning: Could not convert figure to image: {e}")
            # Fallback method: return base64 encoded image
            import base64
            from io import BytesIO
            
            # Convert to JSON
            fig_json = fig.to_json()
            
            # Create a simple placeholder image
            buffer = BytesIO()
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Image generation not available\nPlease install kaleido==0.2.1", 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            plt.savefig(buffer, format=format)
            buffer.seek(0)
            
            return buffer.getvalue()

# Initialize visualization engine
visualization_engine = VisualizationEngine()

# PDF Report generation
class ReportGenerator:
    """Generate PDF reports with charts and recommendations"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        
    def create_forecast_report(self, product_id, forecast_df, recommendations, language="en"):
        """Create a PDF report with forecast and recommendations"""
        # Prepare title and file path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if language == "fr":
            title = f"Rapport de prévision pour {product_id}"
            filename = f"{REPORT_PATH}/rapport_prevision_{product_id}_{timestamp}.pdf"
        elif language == "ar":
            title = f"تقرير التنبؤ لـ {product_id}"
            filename = f"{REPORT_PATH}/taqrir_tanabbu_{product_id}_{timestamp}.pdf"
        else:
            title = f"Forecast Report for {product_id}"
            filename = f"{REPORT_PATH}/forecast_report_{product_id}_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []
        
        # Add title
        title_style = self.styles['Heading1']
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 12))
        
        # Add date
        date_str = datetime.now().strftime("%Y-%m-%d")
        if language == "fr":
            date_text = f"Date du rapport: {date_str}"
        elif language == "ar":
            date_text = f"تاريخ التقرير: {date_str}"
        else:
            date_text = f"Report Date: {date_str}"
        elements.append(Paragraph(date_text, self.styles['Normal']))
        elements.append(Spacer(1, 24))
        
        # Add forecast chart
        fig = visualization_engine.create_forecast_chart(forecast_df, product_id, language=language)
        img_bytes = visualization_engine.convert_fig_to_image(fig)
        img_temp = BytesIO(img_bytes)
        img = Image(img_temp, width=500, height=300)
        elements.append(img)
        elements.append(Spacer(1, 12))
        
        # Add forecast table
        if language == "fr":
            table_title = "Données de prévision"
            date_col = "Date"
            forecast_col = "Prévision"
        elif language == "ar":
            table_title = "بيانات التنبؤ"
            date_col = "التاريخ"
            forecast_col = "التنبؤ"
        else:
            table_title = "Forecast Data"
            date_col = "Date"
            forecast_col = "Forecast"
        
        elements.append(Paragraph(table_title, self.styles['Heading2']))
        elements.append(Spacer(1, 12))
        
        # Create table data
        table_data = [[date_col, forecast_col]]
        for _, row in forecast_df.head(10).iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date']
            forecast_val = f"{row['forecast']:.2f}"
            table_data.append([date_str, forecast_val])
        
        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 24))
        
        # Add recommendations
        if language == "fr":
            rec_title = "Recommandations"
        elif language == "ar":
            rec_title = "التوصيات"
        else:
            rec_title = "Recommendations"
        
        elements.append(Paragraph(rec_title, self.styles['Heading2']))
        elements.append(Spacer(1, 12))
        
        # Split recommendations by line and add as paragraphs
        for line in recommendations.split('\n'):
            if line.strip():
                elements.append(Paragraph(line, self.styles['Normal']))
                elements.append(Spacer(1, 6))
        
        # Generate PDF
        doc.build(elements)
        
        return filename
    
    def create_inventory_report(self, inventory_data, recommendations, language="en"):
        """Create a PDF report with inventory optimization results"""
        # Prepare title and file path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if language == "fr":
            title = "Rapport d'optimisation des stocks"
            filename = f"{REPORT_PATH}/rapport_inventaire_{timestamp}.pdf"
        elif language == "ar":
            title = "تقرير تحسين المخزون"
            filename = f"{REPORT_PATH}/taqrir_makhzun_{timestamp}.pdf"
        else:
            title = "Inventory Optimization Report"
            filename = f"{REPORT_PATH}/inventory_report_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []
        
        # Add title
        title_style = self.styles['Heading1']
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 12))
        
        # Add date
        date_str = datetime.now().strftime("%Y-%m-%d")
        if language == "fr":
            date_text = f"Date du rapport: {date_str}"
        elif language == "ar":
            date_text = f"تاريخ التقرير: {date_str}"
        else:
            date_text = f"Report Date: {date_str}"
        elements.append(Paragraph(date_text, self.styles['Normal']))
        elements.append(Spacer(1, 24))
        
        # Add inventory overview chart
        fig = visualization_engine.create_inventory_chart(inventory_data, language=language)
        img_bytes = visualization_engine.convert_fig_to_image(fig)
        img_temp = BytesIO(img_bytes)
        img = Image(img_temp, width=500, height=300)
        elements.append(img)
        elements.append(Spacer(1, 24))
        
        # Add pie chart for each location
        for location in inventory_data["locations"].keys():
            # Create product distribution pie chart
            fig = visualization_engine.create_product_pie_chart(inventory_data, location, language=language)
            if fig:
                img_bytes = visualization_engine.convert_fig_to_image(fig)
                img_temp = BytesIO(img_bytes)
                img = Image(img_temp, width=400, height=300)
                elements.append(img)
                elements.append(Spacer(1, 12))
        
        # Add recommendations
        if language == "fr":
            rec_title = "Recommandations"
        elif language == "ar":
            rec_title = "التوصيات"
        else:
            rec_title = "Recommendations"
        
        elements.append(Paragraph(rec_title, self.styles['Heading2']))
        elements.append(Spacer(1, 12))
        
        # Split recommendations by line and add as paragraphs
        for line in recommendations.split('\n'):
            if line.strip():
                elements.append(Paragraph(line, self.styles['Normal']))
                elements.append(Spacer(1, 6))
        
        # Generate PDF
        doc.build(elements)
        
        return filename

# Initialize report generator
report_generator = ReportGenerator()

# Multilingual response generation
def generate_response(query, context="", language="en"):
    """Generate a response using the language model with proper error handling"""
    try:
        # Validate inputs
        if not query:
            return "I didn't receive a question to answer."
            
        # Get language-specific prompt
        lang_prompt = {
            "en": "Provide a detailed response in English:",
            "fr": "Fournissez une réponse détaillée en français:",
            "ar": "قدم إجابة مفصلة باللغة العربية:"
        }.get(language, "Provide a detailed response in English:")
        
        # Format prompt with context
        prompt = f"""
        [CONTEXT]
        {context}
        
        [INSTRUCTION]
        {lang_prompt}
        {query}
        
        [RESPONSE]
        """
        
        # Set timeout for LLM generation
        import signal
        
        class TimeoutException(Exception):
            pass
            
        def timeout_handler(signum, frame):
            raise TimeoutException("LLM response generation timed out")
            
        # Set 15-second timeout for LLM generation
        timeout_seconds = 60
        
        try:
            # Set the timeout alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            # Generate response with the LLM
            response = llm_pipeline(
                prompt,
                max_new_tokens=512,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,  # Control randomness
                top_p=0.95,       # Nucleus sampling
                repetition_penalty=1.15  # Reduce repetition
            )
            
            # Cancel the timeout alarm
            signal.alarm(0)
            
            # Extract and clean the response
            if response and len(response) > 0:
                generated_text = response[0]['generated_text']
                # Extract text after [RESPONSE] tag
                if "[RESPONSE]" in generated_text:
                    result = generated_text.split("[RESPONSE]")[-1].strip()
                else:
                    # Fallback to extract response after the last line of the prompt
                    result = "\n".join(generated_text.split("\n")[prompt.count("\n"):]).strip()
                return result
            else:
                return "I couldn't generate a response. Please try asking your question differently."
                
        except TimeoutException:
            print("LLM response generation timed out")
            # Provide a fallback response for timeout
            timeout_responses = {
                "en": "I'm still thinking about your question. Please try again as I might need a moment to process complex queries.",
                "fr": "Je réfléchis toujours à votre question. Veuillez réessayer car il me faut parfois un moment pour traiter des requêtes complexes.",
                "ar": "ما زلت أفكر في سؤالك. يرجى المحاولة مرة أخرى لأنني قد أحتاج إلى لحظة لمعالجة الاستعلامات المعقدة."
            }
            return timeout_responses.get(language, timeout_responses["en"])
            
        finally:
            # Reset the signal handler
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            
    except Exception as e:
        print(f"Error generating response: {e}")
        error_responses = {
            "en": "I encountered an error while processing your question. Please try again with a different query.",
            "fr": "J'ai rencontré une erreur lors du traitement de votre question. Veuillez réessayer avec une autre requête.",
            "ar": "لقد واجهت خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى باستخدام استعلام مختلف."
        }
        return error_responses.get(language, error_responses["en"])

# RAG retrieval function
def get_rag_context(query):
    """Retrieve relevant context from vector store for a given query"""
    try:
        # Handle empty query case
        if not query or not query.strip():
            return ""
        
        # Add timeout handling for vector store search
        docs = []
        try:
            # Attempt to get relevant documents
            docs = vector_store.similarity_search(query, k=2)
        except Exception as search_error:
            print(f"Error in vector search: {search_error}")
            # Return empty context on error
            return ""
        
        # Process retrieved documents
        if not docs:
            return ""
            
        # Extract and join document content
        context = "\n".join([d.page_content for d in docs if hasattr(d, 'page_content') and d.page_content])
        return context
    except Exception as e:
        print(f"Error retrieving RAG context: {e}")
        return ""

# API Endpoints
@app.route('/forecast', methods=['POST'])
@app.route('/demand_forecast', methods=['POST'])  # Support both endpoint names for frontend compatibility
def forecast_demand():
    """Generate demand forecast for a product"""
    data = request.json
    product_id = data['product_id']
    days = data.get('days', 30)
    language = data.get('language', 'en')
    
    try:
        # Generate forecast using our forecasting engine
        model_type = data.get('model_type', 'arima')
        forecast_df = forecasting_engine.forecast(product_id, steps=days, model_type=model_type)
        
        # Create visualization - with error handling
        try:
            fig = visualization_engine.create_forecast_chart(
                forecast_df, product_id, language=language
            )
        except Exception as viz_error:
            print(f"Visualization error: {viz_error}")
            # Continue without visualization if it fails
            fig = None
        
        # Convert to chart data for JSON response
        chart_data = [
            {
                'x': forecast_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'y': forecast_df['forecast'].tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Forecast'
            }
        ]
        
        # Add confidence intervals if available
        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            chart_data.append({
                'x': forecast_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'y': forecast_df['upper_bound'].tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Upper Bound',
                'line': {'width': 0}
            })
            chart_data.append({
                'x': forecast_df['date'].dt.strftime('%Y-%m-%d').tolist() + forecast_df['date'].dt.strftime('%Y-%m-%d').tolist()[::-1],
                'y': forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
                'type': 'scatter',
                'fill': 'toself',
                'fillcolor': 'rgba(0, 176, 246, 0.2)',
                'line': {'width': 0},
                'name': 'Confidence Interval'
            })
        
        # Log successful response for debugging
        print(f"Successfully generated forecast for product {product_id}")
        
        return jsonify({
            "product_id": product_id,
            "forecast": forecast_df['forecast'].tolist(),
            "dates": forecast_df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "chart_data": chart_data
        })
        
    except Exception as e:
        print(f"Forecast error: {str(e)}")
        # Return a more informative error
        return jsonify({
            "error": str(e),
            "message": "Failed to generate forecast. If this is related to Kaleido/Plotly, install kaleido==0.2.1"
        }), 500

@app.route('/optimize', methods=['POST'])
@app.route('/inventory_optimize', methods=['POST'])  # Support both endpoint names for frontend compatibility
def optimize_inventory_route():
    """Optimize inventory allocation across warehouses"""
    data = request.json
    language = data.get('language', 'en')
    
    try:
        # Handle different input formats from frontend
        demand_forecast = data.get('demand_forecast', {})
        if isinstance(demand_forecast, list):
            # Convert list format to dictionary format if needed
            demand_forecast_dict = {}
            for item in demand_forecast:
                if isinstance(item, dict) and 'product_id' in item and 'demand' in item:
                    demand_forecast_dict[item['product_id']] = item['demand']
            if demand_forecast_dict:
                demand_forecast = demand_forecast_dict
        
        # Get warehouse capacities, with fallback to default values
        warehouse_capacities = data.get('warehouse_capacities', {})
        if not warehouse_capacities:
            warehouse_capacities = {
                'warehouse_a': 1000,
                'warehouse_b': 1500,
                'warehouse_c': 1200
            }
            
        # Get holding costs, with fallback to default values
        holding_costs = data.get('holding_costs', {})
        if not holding_costs:
            holding_costs = {
                'smartphone': 10,
                'laptop': 15,
                'tablet': 8,
                'headphones': 5,
                'smartwatch': 7
            }
        
        # Log inputs for debugging
        print(f"Optimization inputs: demand_forecast={demand_forecast}, capacities={warehouse_capacities}")
        
        # If empty demand forecast, use sample data
        if not demand_forecast:
            demand_forecast = {
                'smartphone': 500,
                'laptop': 300,
                'tablet': 400,
                'headphones': 600,
                'smartwatch': 200
            }
        
        
            # Run optimization
            optimization_results = inventory_optimizer.optimize_inventory(
                demand_forecast,
                warehouse_capacities,
                holding_costs,
                transportation_costs=data.get('transportation_costs'),
                lead_times=data.get('lead_times'),
                service_level=data.get('service_level', 0.95)
            )
            
            # Verify optimization results have required structure
            if "locations" not in optimization_results:
                # Add default structure if missing
                optimization_results["locations"] = {}
                for loc in warehouse_capacities.keys():
                    optimization_results["locations"][loc] = {
                        "total_inventory": 0,
                        "capacity_utilization": 0,
                        "products": {}
                    }
            
            # Generate recommendations
            recommendations = inventory_optimizer.generate_recommendations(language)
            
            # Create visualization - with error handling
            try:
                fig = visualization_engine.create_inventory_chart(
                    optimization_results, language=language
                )
            except Exception as viz_error:
                print(f"Visualization error: {viz_error}")
                # Continue without visualization if it fails
                fig = None
        
        # Convert to chart data for JSON response with safety checks
        try:
            # Ensure the optimization results have the expected structure
            locations = list(optimization_results.get("locations", {}).keys())
            if not locations:
                # Create default chart data if no locations available
                locations = list(warehouse_capacities.keys())
                total_inventories = [0] * len(locations)
                utilizations = [0] * len(locations)
            else:
                total_inventories = [
                    loc_data.get("total_inventory", 0) 
                    for loc_data in optimization_results["locations"].values()
                ]
                utilizations = [
                    loc_data.get("capacity_utilization", 0) 
                    for loc_data in optimization_results["locations"].values()
                ]
                
            chart_data = [{
                'x': locations,
                'y': total_inventories,
                'type': 'bar',
                'name': 'Total Inventory'
            }, {
                'x': locations,
                'y': utilizations,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Utilization %',
                'yaxis': 'y2'
            }]
        except Exception as chart_error:
            print(f"Chart data generation error: {chart_error}")
            # Provide fallback chart data
            chart_data = [{
                'x': list(warehouse_capacities.keys()),
                'y': [0] * len(warehouse_capacities),
                'type': 'bar',
                'name': 'Total Inventory (Error)'
            }]
        
        # Log successful response
        print(f"Successfully optimized inventory for {len(demand_forecast)} products")
        
        return jsonify({
            "optimization_results": optimization_results,
            "recommendations": recommendations,
            "chart_data": chart_data
        })
        
    except Exception as e:
        print(f"Optimization error: {str(e)}")
        # Return a more informative error
        return jsonify({
            "error": str(e),
            "message": "Failed to optimize inventory. If this is related to Kaleido/Plotly, install kaleido==0.2.1"
        }), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer supply chain questions using RAG"""
    try:
        # Validate request data
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON data"}), 400
        
        # Extract and validate query
        query = data.get('query')
        if not query or not isinstance(query, str):
            return jsonify({"error": "Missing or invalid 'query' parameter"}), 400
            
        # Extract language with default fallback
        language = data.get('language', 'en')
        if language not in ['en', 'fr', 'ar']:
            # Default to English if unsupported language
            language = 'en'
        
        # Determine if speech generation is requested
        generate_speech_output = data.get('generate_speech', False)
        
        # Track processing steps for debugging
        processing_log = []
        
        try:
            # Set up a timeout for the entire operation
            import signal
            from contextlib import contextmanager
            
            @contextmanager
            def timeout(seconds):
                def handler(signum, frame):
                    raise TimeoutError(f"Operation timed out after {seconds} seconds")
                
                # Set the timeout handler
                original_handler = signal.signal(signal.SIGALRM, handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, original_handler)

            # Use a 60-second timeout for the entire operation
            with timeout(60):
                # Retrieve relevant context with timeout handling
                processing_log.append("Retrieving context")
                context = get_rag_context(query)
                if not context:
                    context = "No specific context available for this query."
                    
                # Generate text response
                processing_log.append("Generating response")
                response = generate_response(query, context, language)
                
                # Generate speech if requested
                speech_file = None
                if generate_speech_output:
                    processing_log.append("Generating speech")
                    speech_file = generate_speech(response, language)
                
                # Build the response
                result = {
                    "query": query,
                    "response": response,
                    "language": language,
                    "success": True
                }
                
                # Add speech file path if available
                if speech_file:
                    result["speech_file"] = speech_file
                    
                return jsonify(result)
                
        except TimeoutError as timeout_error:
            # Handle timeout specifically
            print(f"Request timed out: {str(timeout_error)}")
            
            # Return a user-friendly timeout message
            timeout_responses = {
                "en": "I'm sorry, but your request timed out. Please try a simpler question or try again later.",
                "fr": "Je suis désolé, mais votre demande a expiré. Veuillez essayer une question plus simple ou réessayer plus tard.",
                "ar": "آسف، لقد انتهت مهلة طلبك. يرجى تجربة سؤال أبسط أو المحاولة مرة أخرى لاحقًا."
            }
            
            return jsonify({
                "query": query,
                "response": timeout_responses.get(language, timeout_responses["en"]),
                "language": language,
                "success": False,
                "error": "Request timed out"
            })
            
        except Exception as inner_e:
            # Log the detailed error for debugging
            print(f"Error processing question: {str(inner_e)}")
            print(f"Processing steps completed: {', '.join(processing_log)}")
            
            # Create a user-friendly error message based on the type of error
            error_message = str(inner_e)
            error_response = "I'm having trouble processing your question right now. Please try again later."
            
            # Check if it's an abort error
            if "abort" in error_message.lower() or isinstance(inner_e, TimeoutError):
                error_responses = {
                    "en": "Your request was interrupted. Please try a shorter question or try again later.",
                    "fr": "Votre demande a été interrompue. Veuillez essayer une question plus courte ou réessayer plus tard.",
                    "ar": "تم إيقاف طلبك. يرجى تجربة سؤال أقصر أو المحاولة مرة أخرى لاحقًا."
                }
                error_response = error_responses.get(language, error_responses["en"])
            
            # Return a user-friendly error message
            return jsonify({
                "query": query,
                "response": error_response,
                "language": language,
                "success": False,
                "error": str(inner_e)
            }), 500
            
    except Exception as e:
        # Handle JSON parsing or other request errors
        return jsonify({"error": f"Request error: {str(e)}"}), 400

@app.route('/anomaly_detection', methods=['POST'])
def detect_anomalies_route():
    """Detect and analyze anomalies in product data"""
    data = request.json
    product_id = data['product_id']
    language = data.get('language', 'en')
    
    try:
        # Detect anomalies
        anomalies, product_data = forecasting_engine.detect_anomalies(product_id)
        
        # Generate analysis
        analysis = forecasting_engine.analyze_anomalies(product_id, language)
        
        # Create visualization
        fig = visualization_engine.create_anomaly_chart(
            product_data, anomalies, product_id, language=language
        )
        
        # Convert to chart data for JSON response
        normal_data = product_data[product_data['anomaly'] == 1]
        chart_data = [{
            'x': normal_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'y': normal_data['demand'].tolist(),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Normal',
            'marker': {'color': 'blue'}
        }]
        
        if not anomalies.empty:
            chart_data.append({
                'x': anomalies['date'].dt.strftime('%Y-%m-%d').tolist(),
                'y': anomalies['demand'].tolist(),
                'type': 'scatter',
                'mode': 'markers',
                'name': 'Anomalies',
                'marker': {
                    'color': 'red',
                    'size': 10,
                    'symbol': 'circle-open'
                }
            })
        
        return jsonify({
            "product_id": product_id,
            "analysis": analysis,
            "anomaly_count": len(anomalies),
            "chart_data": chart_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/scenario_analysis', methods=['POST'])
def analyze_scenario_route():
    """Analyze the impact of different scenarios on demand forecast"""
    data = request.json
    product_id = data['product_id']
    scenario = data['scenario']
    language = data.get('language', 'en')
    
    try:
        # Run scenario analysis
        scenario_forecast, base_forecast, description = forecasting_engine.simulate_scenario(
            product_id, scenario, language
        )
        
        # Create visualization
        fig = visualization_engine.create_scenario_chart(
            scenario_forecast, base_forecast, product_id, scenario, language
        )
        
        # Convert to chart data for JSON response
        chart_data = [{

            'x': base_forecast['date'].dt.strftime('%Y-%m-%d').tolist(),
            'y': base_forecast['forecast'].tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Baseline Forecast',
            'line': {'color': 'blue'}

        }, {
            'x': scenario_forecast['date'].dt.strftime('%Y-%m-%d').tolist(),
            'y': scenario_forecast['forecast'].tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Scenario Forecast',
            'line': {'color': 'red'}
        }]
        
        # Calculate the impact
        avg_base = base_forecast['forecast'].mean()
        avg_scenario = scenario_forecast['forecast'].mean()
        impact_pct = (avg_scenario - avg_base) / avg_base * 100
        
        # Generate impact description
        if language == "fr":
            impact_text = f"Impact du scénario: {impact_pct:.1f}% changement dans la demande moyenne"
        elif language == "ar":
            impact_text = f"تأثير السيناريو: {impact_pct:.1f}٪ تغيير في متوسط الطلب"
        else:
            impact_text = f"Scenario impact: {impact_pct:.1f}% change in average demand"
        
        return jsonify({
            "product_id": product_id,
            "scenario": scenario,
            "description": description,
            "impact": impact_text,
            "chart_data": chart_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    """Generate PDF reports"""
    data = request.json
    report_type = data['report_type']
    language = data.get('language', 'en')
    
    try:
        # Make sure the reports directory exists
        if not os.path.exists(REPORT_PATH):
            os.makedirs(REPORT_PATH)
            
        print(f"Generating {report_type} report in {language}")
            
        if report_type == 'forecast':
            product_id = data['product_id']
            days = data.get('days', 30)
            
            try:
                # Generate forecast
                forecast_df = forecasting_engine.forecast(product_id, steps=days)
                
                # Generate recommendations
                if language == "fr":
                    recommendations = f"Recommandations pour {product_id}:\n"
                    recommendations += "- Surveiller les niveaux de stock en fonction des prévisions\n"
                    recommendations += "- Planifier les commandes en avance pour éviter les ruptures de stock\n"
                    recommendations += f"- Prévoir une demande moyenne de {forecast_df['forecast'].mean():.1f} unités par jour"
                elif language == "ar":
                    recommendations = f"توصيات لـ {product_id}:\n"
                    recommendations += "- مراقبة مستويات المخزون بناءً على التوقعات\n"
                    recommendations += "- تخطيط الطلبات مقدمًا لتجنب نفاد المخزون\n"
                    recommendations += f"- توقع متوسط طلب يومي قدره {forecast_df['forecast'].mean():.1f} وحدة"
                else:
                    recommendations = f"Recommendations for {product_id}:\n"
                    recommendations += "- Monitor inventory levels based on forecast\n"
                    recommendations += "- Plan orders ahead to avoid stockouts\n"
                    recommendations += f"- Expect average daily demand of {forecast_df['forecast'].mean():.1f} units"
                
                # Generate report
                report_file = report_generator.create_forecast_report(
                    product_id, forecast_df, recommendations, language
                )
                
            except Exception as forecast_error:
                print(f"Error generating forecast report: {forecast_error}")
                # Return an error response the frontend can handle
                return jsonify({
                    "error": str(forecast_error),
                    "message": "Failed to generate forecast report. Try installing kaleido==0.2.1 and plotly==5.24.1",
                    "report_url": None,
                    "mock_report": True
                }), 500
            
        elif report_type == 'inventory':
            # For demo, create sample optimization results
            sample_forecast = {
                'smartphone': 500,
                'laptop': 300,
                'tablet': 400,
                'headphones': 600,
                'smartwatch': 200
            }
            
            sample_capacities = {
                'warehouse_a': 1000,
                'warehouse_b': 1500,
                'warehouse_c': 1200
            }
            
            sample_holding_costs = {
                'smartphone': 10,
                'laptop': 15,
                'tablet': 8,
                'headphones': 5,
                'smartwatch': 7
            }
            
            try:
                # Generate optimization results
                optimization_results = inventory_optimizer.optimize_inventory(
                    sample_forecast, sample_capacities, sample_holding_costs
                )
                
                # Verify the optimization results have the required structure
                if not isinstance(optimization_results, dict):
                    raise TypeError(f"Expected dict for optimization_results, got {type(optimization_results)}")
                
                if "locations" not in optimization_results:
                    # Create default structure if missing
                    optimization_results["locations"] = {}
                    for loc in sample_capacities.keys():
                        optimization_results["locations"][loc] = {
                            "total_inventory": 0,
                            "capacity_utilization": 0,
                            "products": {}
                        }
                
                # Generate recommendations
                recommendations = inventory_optimizer.generate_recommendations(language)
                
                # Generate report
                report_file = report_generator.create_inventory_report(
                    optimization_results, recommendations, language
                )
            except Exception as inventory_error:
                print(f"Error generating inventory report: {inventory_error}")
                # Return an error response the frontend can handle
                return jsonify({
                    "error": str(inventory_error),
                    "message": "Failed to generate inventory report. Try installing kaleido==0.2.1 and plotly==5.24.1",
                    "report_url": None,
                    "mock_report": True
                }), 500
            
        else:
            return jsonify({"error": "Invalid report type"}), 400
        
        # Return report URL
        report_url = f"/reports/{os.path.basename(report_file)}"
        print(f"Report generated successfully: {report_url}")
        return jsonify({"report_url": report_url})
        
    except Exception as e:
        print(f"Report generation error: {str(e)}")
        return jsonify({
            "error": str(e), 
            "message": "Failed to generate report. Check package compatibility.",
            "mock_report": True
        }), 500

@app.route('/reports/<filename>')
def serve_report(filename):
    """Serve generated PDF reports"""
    return send_file(f"{REPORT_PATH}/{filename}", mimetype='application/pdf')

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve generated audio files"""
    return send_file(f"{AUDIO_PATH}/{filename}", mimetype='audio/mpeg')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Process uploaded audio for speech recognition"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    language = request.form.get('language', 'en')
    
    if audio_file.filename == '':
        return jsonify({"error": "Empty audio file"}), 400
    
    try:
        # Save audio file temporarily
        temp_path = f"{AUDIO_PATH}/temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        audio_file.save(temp_path)
        
        # Transcribe audio
        transcription = transcribe_audio(temp_path, language)
        
        # Process the query
        if transcription:
            context = get_rag_context(transcription)
            response = generate_response(transcription, context, language)
            
            # Generate speech response
            speech_file = generate_speech(response, language)
            
            return jsonify({
                "transcription": transcription,
                "response": response,
                "speech_file": f"/audio/{os.path.basename(speech_file)}" if speech_file else None
            })
        else:
            return jsonify({"error": "Could not transcribe audio"}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/add_document', methods=['POST'])
def add_document():
    """Add a document to the RAG knowledge base"""
    data = request.json
    new_doc = data['document']
    language = data.get('language', 'en')
    
    try:
        # Add to vector store
        global vector_store
        vector_store.add_texts([new_doc])
        vector_store.save_local(VECTOR_DB_PATH)
        
        # Return confirmation
        if language == "fr":
            message = "Document ajouté avec succès à la base de connaissances"
        elif language == "ar":
            message = "تمت إضافة المستند بنجاح إلى قاعدة المعرفة"
        else:
            message = "Document successfully added to knowledge base"
        
        return jsonify({"status": "success", "message": message})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# CloudFlare Tunnel Helper
class CloudflaredTunnel:
    """Helper class to create and manage a Cloudflare tunnel"""
    
    def __init__(self, port=5000):
        self.port = port
        self.process = None
        self.public_url = None
        self.cloudflared_path = self.find_cloudflared()
    
    @property
    def is_running(self):
        """Check if the tunnel is running"""
        return self.process is not None and self.process.poll() is None
    
    def get_public_url(self):
        """Get the public URL of the tunnel"""
        return self.public_url
    
    def find_cloudflared(self):
        """Find the cloudflared executable or download it if needed"""
        # Check if cloudflared is in PATH
        cloudflared_path = self._which('cloudflared')
        if cloudflared_path:
            print(f"Found cloudflared at {cloudflared_path}")
            return cloudflared_path
        
        # If not found, download it
        print("Cloudflared not found in PATH, attempting to download...")
        return self._download_cloudflared()
    
    def _which(self, program):
        """Find executable in PATH, similar to 'which' command in Unix"""
        def is_exe(fpath):
            return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

        fpath, _ = os.path.split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file
        return None
    
    def _download_cloudflared(self):
        """Download cloudflared binary based on platform"""
        import platform
        import tempfile
        import zipfile
        import shutil
        import requests
        
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Determine download URL based on platform
        base_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        
        if system == "windows":
            if "arm" in machine or "aarch64" in machine:
                filename = "cloudflared-windows-arm64.exe"
            else:
                filename = "cloudflared-windows-amd64.exe"
            download_url = f"{base_url}{filename}"
            local_path = os.path.join(tempfile.gettempdir(), "cloudflared.exe")
        elif system == "darwin":  # macOS
            if "arm" in machine or "aarch64" in machine:
                filename = "cloudflared-darwin-aarch64.tgz"
            else:
                filename = "cloudflared-darwin-amd64.tgz"
            download_url = f"{base_url}{filename}"
            local_path = os.path.join(tempfile.gettempdir(), "cloudflared")
        else:  # Linux and others
            if "arm" in machine:
                if "64" in machine or "aarch64" in machine:
                    filename = "cloudflared-linux-arm64"
                else:
                    filename = "cloudflared-linux-arm"
            else:
                filename = "cloudflared-linux-amd64"
            download_url = f"{base_url}{filename}"
            local_path = os.path.join(tempfile.gettempdir(), "cloudflared")
        
        # Download the file
        print(f"Downloading cloudflared from {download_url}")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Save the file
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Make it executable (for Unix systems)
        if system != "windows":
            os.chmod(local_path, 0o755)
        
        print(f"Downloaded cloudflared to {local_path}")
        return local_path
    
    def start(self):
        """Start the cloudflare tunnel"""
        import subprocess
        import threading
        import time
        import queue
        
        if not self.cloudflared_path:
            print("Error: cloudflared not found or couldn't be downloaded.")
            return False
        
        # Create a queue to communicate between threads
        url_queue = queue.Queue()
        
        def run_tunnel():
            try:
                print(f"Starting Cloudflare tunnel for port {self.port}...")
                # Use shell=False for better process management
                self.process = subprocess.Popen(
                    [self.cloudflared_path, 'tunnel', '--url', f'http://localhost:{self.port}'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Wait for tunnel URL
                for line in self.process.stdout:
                    line_str = line.strip()
                    print(line_str)
                    # Look for the URL in the output
                    if 'https://' in line_str and 'trycloudflare.com' in line_str:
                        match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line_str)
                        if match:
                            tunnel_url = match.group(0)
                            # Put the URL in the queue
                            url_queue.put(tunnel_url)
                            break
                
                # Keep the process running
                self.process.wait()
            except Exception as e:
                print(f"Error in tunnel process: {e}")
                url_queue.put(None)  # Signal that no URL was found
        
        # Start the tunnel in a separate thread
        thread = threading.Thread(target=run_tunnel, daemon=True)
        thread.start()
        
        # Wait for the URL with a timeout
        try:
            # Wait up to 20 seconds for the URL to appear
            self.public_url = url_queue.get(timeout=20)
            
            if self.public_url:
                print(f"\n✅ PUBLIC URL: {self.public_url}\n")
                print("Share this URL to access your application from anywhere!")
                print("Press Ctrl+C to stop the server and tunnel")
                return True
            else:
                print("Failed to get tunnel URL")
                self.stop()  # Clean up if we couldn't get a URL
                return False
        except queue.Empty:
            print("Timeout waiting for tunnel URL")
            self.stop()  # Clean up on timeout
            return False
    
    def stop(self):
        """Stop the cloudflare tunnel"""
        if self.process:
            print("Stopping Cloudflare tunnel...")
            try:
                # Try to terminate gracefully first
                self.process.terminate()
                # Wait for up to 5 seconds for process to terminate
                for i in range(10):
                    if self.process.poll() is not None:
                        break
                    time.sleep(0.5)
                
                # If still running, force kill
                if self.process.poll() is None:
                    print("Force killing tunnel process...")
                    if platform.system() == "Windows":
                        import ctypes
                        kernel32 = ctypes.windll.kernel32
                        handle = kernel32.OpenProcess(1, 0, self.process.pid)
                        kernel32.TerminateProcess(handle, 0)
                        kernel32.CloseHandle(handle)
                    else:
                        import signal
                        os.kill(self.process.pid, signal.SIGKILL)
            except Exception as e:
                print(f"Error stopping tunnel: {e}")
            finally:
                self.process = None
                self.public_url = None

# Initialize CloudFlared tunnel
tunnel = CloudflaredTunnel(port=5000)

# API endpoint to get the tunnel status
@app.route('/tunnel/status', methods=['GET'])
def tunnel_status():
    """Get the status of the CloudFlare tunnel"""
    if tunnel.is_running:
        return jsonify({
            "status": "running", 
            "url": tunnel.get_public_url()
        })
    else:
        return jsonify({"status": "stopped"})

# API endpoint to start the tunnel
@app.route('/tunnel/start', methods=['POST'])
def start_tunnel():
    """Start the CloudFlare tunnel"""
    if tunnel.is_running:
        return jsonify({
            "status": "already_running",
            "message": "Tunnel is already running",
            "url": tunnel.get_public_url()
        })
    
    success = tunnel.start()
    if success:
        return jsonify({
            "status": "started",
            "message": "Tunnel started successfully",
            "url": tunnel.get_public_url()
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Failed to start tunnel"
        }), 500

# API endpoint to stop the tunnel
@app.route('/tunnel/stop', methods=['POST'])
def stop_tunnel():
    """Stop the CloudFlare tunnel"""
    if not tunnel.is_running:
        return jsonify({
            "status": "not_running",
            "message": "Tunnel is not running"
        })
    
    tunnel.stop()
    return jsonify({
        "status": "stopped",
        "message": "Tunnel stopped successfully"
    })

# Test route to verify tunnel connectivity
@app.route('/tunnel/test', methods=['GET'])
def test_tunnel():
    """Test route to verify tunnel connectivity"""
    return jsonify({
        "status": "success",
        "message": "Tunnel is working properly!",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tunnel_status": "running" if tunnel.is_running else "stopped",
        "tunnel_url": tunnel.get_public_url() if tunnel.is_running else None
    })

if __name__ == '__main__':
    # Check Plotly and Kaleido compatibility at startup
    try:
        import plotly
        print(f"Plotly version: {plotly.__version__}")
        
        try:
            import kaleido
            print(f"Kaleido version: {kaleido.__version__ if hasattr(kaleido, '__version__') else 'unknown'}")
            
            # Check compatibility
            plotly_version = float('.'.join(plotly.__version__.split('.')[:2]))
            if plotly_version < 5.0:
                print("Warning: Plotly version is below 5.0, which may cause compatibility issues with Kaleido")
            elif plotly_version >= 6.1 and not hasattr(kaleido, '__version__'):
                print("Warning: Using Plotly 6.1+ requires Kaleido 0.2.1 or newer")
            
            # Test image generation
            test_fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 3, 2]))
            try:
                test_fig.to_image(format='png')
                print("✓ Image generation is working correctly")
            except Exception as img_error:
                print(f"× Image generation test failed: {img_error}")
                print("To fix: Install compatible versions with: pip install kaleido==0.2.1 plotly==5.24.1")
        except ImportError:
            print("× Kaleido not installed. Static image generation will not work.")
            print("To fix: Install Kaleido with: pip install kaleido==0.2.1")
    except ImportError:
        print("× Plotly not found. Visualization will not work.")
    
    # Initialize vector store (ensure embeddings are defined before this)
    if os.path.exists(VECTOR_DB_PATH):
        try:
            vector_store = FAISS.load_local(
                VECTOR_DB_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True  # Only use if you trust the source
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Creating new vector store...")
            vector_store = FAISS.from_texts(supply_chain_docs, embeddings)
            vector_store.save_local(VECTOR_DB_PATH)
    else:
        vector_store = FAISS.from_texts(supply_chain_docs, embeddings)
        vector_store.save_local(VECTOR_DB_PATH)
    
    # Configure Flask to handle application termination properly
    import atexit
    
    def cleanup_on_exit():
        """Clean up resources when application exits"""
        if tunnel.is_running:
            print("Shutting down Cloudflare tunnel...")
            tunnel.stop()
    
    # Register the cleanup function
    atexit.register(cleanup_on_exit)
    
    # For Google Colab, set threaded=False to avoid issues
    is_colab = False
    try:
        # Try to detect if running in Google Colab
        import sys
        if 'google.colab' in sys.modules:
            is_colab = True
    except:
        pass
    
    flask_kwargs = {'host': '0.0.0.0', 'port': 5000}
    
    if is_colab:
        flask_kwargs['threaded'] = False
        flask_kwargs['debug'] = False  # Prevent auto-reloader from launching multiple instances
        print("\n=== Running in Google Colab environment ===")
        print("Using non-threaded mode for better stability in Colab")
    
    # Start CloudFlared tunnel automatically
    print("\n=== Starting CloudFlare Tunnel ===")
    print("This will create a public URL to access your application from anywhere")
    try:
        success = tunnel.start()
        if success:
            print(f"Tunnel started successfully! Public URL: {tunnel.get_public_url()}")
        else:
            print("Failed to start tunnel automatically")
            print("You can start it manually later via the /tunnel/start endpoint")
    except Exception as e:
        print(f"Warning: Could not start tunnel automatically: {e}")
        print("You can start it manually later via the /tunnel/start endpoint")
    
    print("\n=== Starting Supply Chain AI Agent Server ===")
    print("Web interface will be available at http://localhost:5000")
    print("Multi-lingual support: English, French, Arabic")
    print("Features: Demand forecasting, Inventory optimization, Anomaly detection, Scenario analysis")
    print("Press Ctrl+C to stop the server and tunnel")
    
    # Disable debug mode when using tunnels to prevent multiple processes
    # In Google Colab, app.run() might not work properly with some Flask versions
    try:
        app.run(**flask_kwargs)
    except Exception as e:
        print(f"Error starting Flask app: {str(e)}")
        print("Trying alternative approach for Colab...")
        # Alternative approach for Colab
        from flask import request
        if 'werkzeug.server.shutdown' in dir():
            try:
                import threading
                server_thread = threading.Thread(target=lambda: app.run(**flask_kwargs))
                server_thread.daemon = True
                server_thread.start()
                print("Flask app started in background thread")
            except Exception as thread_e:
                print(f"Failed to start Flask in thread: {str(thread_e)}")
                print("Please check Flask version and configuration")