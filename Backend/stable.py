# app.py
# Make sure to install these dependencies before running:
# Required packages:
!pip install transformers langchain-huggingface kaleido==0.2.1 torch torchaudio langchain_community bitsandbytes autoawq accelerate sentencepiece protobuf pulp scipy plotly==5.24.1 reportlab gtts pytrends faiss-cpu pydub==0.25.1 soundfile==0.12.1 ffmpeg-python==0.2.0
!pip install kaleido==0.2.1 plotly==5.24.1 langchain_text_splitters==0.0.1 langchain_huggingface
!pip install flask-cors==4.0.0 Flask-CORS==4.0.0 pypdf==3.17.0 langchain_core

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
import time
import traceback
import subprocess
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
import soundfile as sf
from pydub import AudioSegment
from gtts import gTTS
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
import os
import requests
import json
import tempfile
import subprocess
import threading
import traceback
import socket
import time
import platform
import re
import io
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

# Import PDF handling libraries
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    print("Warning: pypdf not installed. PDF document import will not be available.")
    PDF_SUPPORT = False

app = Flask(__name__)

CORS(app) 

# Configuration
os.environ["HF_TOKEN"] = ''
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL = "openai/whisper-medium"
DATA_PATH = '/content/supply_chain_dataset.csv'
WEATHER_DATA_PATH = '/content/weather_data.csv'
TRENDS_DATA_PATH = '/content/trends_data.csv'
VECTOR_DB_PATH = 'supply_chain_faiss_index'
REPORT_PATH = 'reports'
AUDIO_PATH = 'audio'

# Ensure directories exist
for path in [VECTOR_DB_PATH, REPORT_PATH, AUDIO_PATH]:
    try:
        if not os.path.exists(path):
            print(f"Creating directory: {path}")
            os.makedirs(path)
        else:
            print(f"Directory already exists: {path}")
    except Exception as dir_error:
        print(f"Error creating directory {path}: {dir_error}")
        # Try to continue anyway

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

# Initialize Whisper for speech recognition - force CPU to avoid device mismatch errors
whisper_processor = None
whisper_model = None

# Force using CPU for Whisper regardless of CUDA availability
device = "cpu"

# Define the transcribe_audio function at global scope so it's accessible everywhere
def transcribe_audio(audio_file, target_language="en"):
    """Transcribe audio file using Whisper with optimized accuracy and performance"""
    try:
        import os
        import subprocess
        from pydub import AudioSegment
        import soundfile as sf
        import numpy as np
        import torch
        import torchaudio
        
        print(f"Transcribing audio with target language: {target_language}")
        
        # Check file existence
        if not os.path.exists(audio_file):
            print(f"Audio file does not exist: {audio_file}")
            return "Error: Audio file not found"
        
        # Check file size
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            print("Empty audio file received")
            return "Error: Empty audio file"
        
        print(f"Processing audio file: {audio_file} (Size: {file_size} bytes)")
        
        # ===== 1. AUDIO PREPROCESSING - OPTIMIZED =====
        # Prepare optimized audio for Whisper
        temp_wav_path = f"{os.path.splitext(audio_file)[0]}_whisper_compatible.wav"
        
        try:
            # Load and normalize audio with pydub
            audio = AudioSegment.from_file(audio_file)
            
            # Voice activity detection to trim silence (improves accuracy)
            from pydub.silence import detect_nonsilent
            
            # Detect non-silent chunks with lenient parameters
            non_silent_chunks = detect_nonsilent(
                audio, 
                min_silence_len=500,  # 500ms silence threshold
                silence_thresh=-40    # -40 dBFS silence threshold (higher = more aggressive)
            )
            
            # If we found non-silent chunks, keep only those parts
            if non_silent_chunks and len(non_silent_chunks) > 0:
                # Add small padding around non-silent chunks
                padding_ms = 200  # 200ms padding
                chunks = []
                for start, end in non_silent_chunks:
                    chunk_start = max(0, start - padding_ms)
                    chunk_end = min(len(audio), end + padding_ms)
                    chunks.append(audio[chunk_start:chunk_end])
                
                # Combine chunks
                if chunks:
                    audio = sum(chunks, AudioSegment.empty())
                    print(f"Trimmed silence: {len(audio)/1000:.2f}s (from original {len(audio)/1000:.2f}s)")
            
            # Convert to mono and 16kHz with high quality settings
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            
            # Normalize audio volume for better recognition (target_dBFS = -14)
            target_dBFS = -14.0
            change_in_dBFS = target_dBFS - audio.dBFS
            normalized_audio = audio.apply_gain(change_in_dBFS)
            
            # Export with optimal parameters for Whisper
            normalized_audio.export(
                temp_wav_path, 
                format="wav", 
                parameters=[
                    "-acodec", "pcm_s16le", 
                    "-ar", "16000", 
                    "-ac", "1",
                    "-sample_fmt", "s16"
                ]
            )
            
            print(f"Enhanced audio saved to: {temp_wav_path}")
            
            # Verify file was created properly
            if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                audio_file = temp_wav_path
            else:
                print("WARNING: Enhanced audio file is empty, will try alternate methods")
        except Exception as audio_error:
            print(f"Audio enhancement failed: {audio_error}")
            # Fallback to basic conversion
            try:
                print("Falling back to basic ffmpeg conversion...")
                command = f'ffmpeg -y -i "{audio_file}" -ac 1 -ar 16000 -c:a pcm_s16le "{temp_wav_path}"'
                subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE)
                if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                    audio_file = temp_wav_path
            except Exception as ffmpeg_error:
                print(f"FFmpeg conversion failed: {ffmpeg_error}")
                # Continue with original file
        
        # Check model availability
        global whisper_model, whisper_processor
        if whisper_model is None or whisper_processor is None:
            return "Speech recognition model is not initialized. Please try again later."
        
        # ===== 2. AUDIO LOADING - OPTIMIZED =====
        # Load audio with consistent approach
        try:
            # Load with soundfile (optimized for WAV)
            audio_data, sample_rate = sf.read(audio_file)
            
            # Ensure audio is the right format (float32, normalized, mono)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Convert to mono if needed
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio to [-1.0, 1.0] range
            if np.abs(audio_data).max() > 0.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            print(f"Audio loaded successfully: shape={audio_data.shape}, sr={sample_rate}")
        except Exception as load_error:
            print(f"Error loading audio: {load_error}")
            return "Error processing audio: Could not load the audio file"
        
        # ===== 3. WHISPER PROCESSING - OPTIMIZED =====
        print("Processing with Whisper model...")
        try:
            # Move model to appropriate device once
            device = next(whisper_model.parameters()).device
            print(f"Using Whisper model on device: {device}")
            
            # Full language support map - common Whisper languages
            language_map = {
                # Common languages
                "en": "english", "fr": "french", "ar": "arabic", 
                # Special handling
                "auto": None
            }
            
            # Set task type - important for accuracy
            task = "transcribe"  # For transcription (vs. translation)
            
            # Handle language selection with better mapping
            if target_language.lower() == "auto":
                print("Auto language detection enabled")
                forced_decoder_ids = None
                detected_language = "auto"
            else:
                # Check if we can get a direct mapping
                mapped_lang = language_map.get(target_language.lower(), None)
                
                if mapped_lang:
                    print(f"Using specified language: {mapped_lang}")
                    try:
                        # Try to get decoder IDs for this language
                        forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
                            language=mapped_lang, 
                            task=task
                        )
                        detected_language = target_language.lower()
                    except Exception as lang_error:
                        print(f"Error setting language '{mapped_lang}': {lang_error}")
                        print("Falling back to auto-detection")
                        forced_decoder_ids = None
                        detected_language = "auto"
                else:
                    print(f"Language '{target_language}' not recognized, enabling auto-detection")
                    forced_decoder_ids = None
                    detected_language = "auto"
            
            # Process audio features
            input_features = whisper_processor(
                audio_data, 
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features
            
            # Ensure features are on same device as model
            input_features = input_features.to(device)
            
            # Run inference with optimized parameters
            print("Running Whisper inference...")
            with torch.no_grad():
                # Use compatible generation parameters for improved accuracy
                generation_kwargs = {
                    "forced_decoder_ids": forced_decoder_ids,
                    "max_length": 448,        # Longer context
                    "num_beams": 5,           # Beam search for better results
                    "temperature": 0.0        # Greedy decoding for accuracy
                }
                
                # Only add language and task if not auto-detection
                if detected_language != "auto":
                    generation_kwargs["language"] = detected_language
                    generation_kwargs["task"] = task
                
                # We'll use a safer approach - only use known working parameters
                # This avoids the error with condition_on_previous_text and other unsupported params
                
                # Safe parameters that work across all Whisper versions
                print("Using safe generation parameters for maximum compatibility")
                
                # Generate transcription
                predicted_ids = whisper_model.generate(
                    input_features,
                    **generation_kwargs
                )
            
            # Decode the output
            result = whisper_processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True,
                normalize=True  # Normalize text for better readability
            )[0]
            
            # ===== 4. POST-PROCESSING - IMPROVED =====
            # Clean up transcription for better readability
            import re
            
            # Remove repeated spaces
            result = re.sub(r'\s+', ' ', result).strip()
            
            # Fix common transcription artifacts
            result = re.sub(r'\(silence\)', '', result)
            result = re.sub(r'\(music\)', '', result)
            result = re.sub(r'\(laughter\)', '', result)
            result = re.sub(r'\(applause\)', '', result)
            result = re.sub(r'\(inaudible\)', '', result)
            
            # Ensure proper capitalization
            if result and len(result) > 0:
                result = result[0].upper() + result[1:]
                
            # Add proper ending punctuation if missing
            if result and len(result) > 0 and not result[-1] in '.!?':
                result += '.'
                
            print(f"Transcription successful: {result[:100]}...")
            return result
            
        except Exception as whisper_error:
            print(f"Whisper processing error: {whisper_error}")
            import traceback
            traceback.print_exc()
            
            # Try to provide a more user-friendly error message
            error_str = str(whisper_error)
            friendly_message = "Error processing audio with Whisper"
            
            # Known error patterns
            if "device" in error_str.lower():
                friendly_message = "Audio processing device error. Trying again might help."
            elif "memory" in error_str.lower() or "cuda" in error_str.lower():
                friendly_message = "System memory issue with speech recognition. Try a shorter audio clip."
            elif "model" in error_str.lower() and "kwargs" in error_str.lower():
                friendly_message = "Speech recognition configuration issue. This will be fixed in the next update."
            
            print(f"Friendly message: {friendly_message}")
            return friendly_message
            
    except Exception as e:
        print(f"Transcription failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error transcribing audio: {str(e)}"

# Optimized Whisper model initialization
try:
    print(f"Loading optimized Whisper model: {WHISPER_MODEL}")
    
    # Check for better model availability
    # available_models = [
    #     "openai/whisper-large-v3",  # Best accuracy but more resources
    #     "openai/whisper-medium",     # Good balance of accuracy and speed
    #     "openai/whisper-small",      # Faster but still decent accuracy
    #     "openai/whisper-base",       # Default/fallback
    # ]
    
    # Try to load the best available model, falling back to simpler ones
    loaded_model = None
    selected_model = None
    
    # for model_name in available_models:
    #     try:
    #         print(f"Attempting to load {model_name}...")
    #         # Check if we have enough resources for this model
    #         if model_name == "openai/whisper-large-v3" and (
    #             not torch.cuda.is_available() or 
    #             torch.cuda.get_device_properties(0).total_memory < 8 * 1024 * 1024 * 1024  # 8GB
    #         ):
    #             print(f"Skipping {model_name} as it requires more resources")
    #             continue
                
    #         # Load the processor first (lightweight)
    #         whisper_processor = WhisperProcessor.from_pretrained(model_name)
    #         selected_model = model_name
    #         break
    #     except Exception as model_error:
    #         print(f"Failed to load {model_name}: {model_error}")
    #         continue
    
    # If we couldn't load any better model, fall back to the default
    if selected_model is None:
        try:
            print(f"Loading default Whisper model: {WHISPER_MODEL}")
            whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
            selected_model = WHISPER_MODEL
        except Exception as model_error:
            print(f"Error loading default model: {model_error}")
            print("Attempting to load the smallest Whisper model as fallback...")
            try:
                whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
                selected_model = "openai/whisper-tiny"
            except:
                print("Critical: Failed to load any Whisper model")
                raise  # Re-raise the exception to be caught by the outer try-except
    print(f"Selected Whisper model: {selected_model}")
    
    # Determine optimal device with consistent approach
    device = "cpu"
    if torch.cuda.is_available():
        try:
            # Check available GPU memory - require at least 2GB free for stable performance
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            memory_required = 2 * 1024 * 1024 * 1024  # 2GB in bytes
            
            if free_memory >= memory_required:
                device = "cuda"
                print(f"Using CUDA with {free_memory / (1024**3):.1f}GB free memory")

                # Simplified loading for stability
                print("Loading model with standard configuration")
                whisper_model = WhisperForConditionalGeneration.from_pretrained(selected_model)
                whisper_model.to(device)
            else:
                print(f"Insufficient GPU memory ({free_memory / (1024**3):.1f}GB free), using CPU")
                # For CPU inference, we'll load the model directly
                whisper_model = WhisperForConditionalGeneration.from_pretrained(selected_model)
        except Exception as cuda_error:
            print(f"Error configuring CUDA: {cuda_error}, falling back to CPU")
            device = "cpu"
            whisper_model = WhisperForConditionalGeneration.from_pretrained(selected_model)
    else:
        print("CUDA not available, using CPU")
        whisper_model = WhisperForConditionalGeneration.from_pretrained(selected_model)
    
    # Ensure model is on the correct device
    if device == "cpu" and next(whisper_model.parameters()).device.type != "cpu":
        whisper_model = whisper_model.to("cpu")
        
    print(f"Whisper model initialized successfully on {device}")
    
    # Test the model with a tiny sample to ensure it's working
    try:
        print("Testing Whisper model with synthetic sample...")
        # Create a small test sample (silent audio)
        test_audio = np.zeros((16000,), dtype=np.float32)  # 1 second of silence
        
        # Run quick inference to ensure everything is initialized
        with torch.no_grad():
            test_inputs = whisper_processor(test_audio, sampling_rate=16000, return_tensors="pt").input_features
            test_inputs = test_inputs.to(next(whisper_model.parameters()).device)
            _ = whisper_model.generate(test_inputs, max_length=20)
        
        print("Whisper model test successful, ready for transcription")
    except Exception as test_error:
        print(f"Whisper model test failed: {test_error}, but continuing anyway")
        
except Exception as whisper_init_error:
    print(f"Error initializing Whisper model: {whisper_init_error}")
    # Create fallback dummy_transcribe function
    def dummy_transcribe(audio_file, target_language="en"):
        return "Speech recognition is currently unavailable. Please try again later."
    
    # Override transcribe_audio to use dummy_transcribe
    transcribe_audio = dummy_transcribe



# Text-to-Speech function using gTTS
def generate_speech(text, language="en"):
    """Generate speech from text using Google Text-to-Speech with support for long texts"""
    try:
        # Validate input text
        if not text or not isinstance(text, str):
            print("Invalid text for speech generation")
            return None
            
        print(f"Generating speech for {len(text)} characters of text")
        
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
        
        # Clean up old audio files (older than 24 hours) to prevent disk fill-up
        try:
            cleanup_time = time.time() - (24 * 60 * 60)  # 24 hours ago
            for old_file in os.listdir(AUDIO_PATH):
                file_path = os.path.join(AUDIO_PATH, old_file)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < cleanup_time:
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up old audio file: {old_file}")
                    except:
                        pass
        except Exception as cleanup_error:
            print(f"Error during audio cleanup: {cleanup_error}")
        
        # Generate a unique timestamp for this audio file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{AUDIO_PATH}/response_{timestamp}.mp3"
        
        # Check if text exceeds gTTS limitations
        max_chunk_length = 5000  # Safe limit for gTTS
        
        if len(text) <= max_chunk_length:
            # Standard case - text is within limits
            try:
                # Use slow=False for normal speed, but can be adjusted for clarity if needed
                tts = gTTS(text=text, lang=lang, slow=False)
                
                # Add linear backoff retry mechanism for network issues
                max_retries = 3
                retry_delay = 1
                for attempt in range(max_retries):
                    try:
                        tts.save(filename)
                        print(f"Successfully saved speech file on attempt {attempt+1}")
                        break
                    except Exception as retry_error:
                        if attempt < max_retries - 1:
                            retry_delay +=1    # linear backoff: 1s, 2s, 3s
                            print(f"Retrying TTS save after {retry_delay}s delay. Error: {retry_error}")
                            time.sleep(retry_delay)
                        else:
                            raise  # Re-raise on final attempt
                            
            except Exception as save_error:
                print(f"Error saving speech file after retries: {save_error}")
                # Try with a simpler filename if there might be path issues
                simple_filename = f"{AUDIO_PATH}/response.mp3"
                try:
                    tts.save(simple_filename)
                    filename = simple_filename
                except Exception as final_error:
                    print(f"Final attempt to save speech failed: {final_error}")
                    return None
        else:
            # Handle long text by breaking it into chunks
            print(f"Text exceeds max length ({len(text)} chars). Splitting into chunks.")
            
            # Split by sentences to avoid breaking mid-sentence
            import re
            
            # Split text into sentences with improved sentence boundary detection
            # This regex handles common sentence terminators and preserves punctuation
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
            
            # If we couldn't split properly (e.g., no capitalization after periods)
            if len(sentences) <= 1:
                # Fall back to simpler sentence splitting
                sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # If text is still not properly split, use a more aggressive approach
            if len(sentences) <= 3 and len(text) > max_chunk_length * 2:
                # Split on any period, question mark, or exclamation mark
                sentences = re.split(r'(?<=[.!?])', text)
                # Clean up any empty items
                sentences = [s.strip() for s in sentences if s.strip()]
            
            chunks = []
            current_chunk = ""
            
            # Group sentences into chunks under the max length, being careful
            # not to exceed the maximum chunk size
            for sentence in sentences:
                # If the sentence itself exceeds the max length, split it further
                if len(sentence) > max_chunk_length:
                    # If we have a current chunk, add it first
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    
                    # Split long sentence by phrases (commas, semicolons, etc.)
                    phrases = re.split(r'(?<=[,;:])\s+', sentence)
                    phrase_chunk = ""
                    
                    for phrase in phrases:
                        if len(phrase_chunk) + len(phrase) < max_chunk_length:
                            phrase_chunk += phrase + " "
                        else:
                            if phrase_chunk:
                                chunks.append(phrase_chunk.strip())
                            phrase_chunk = phrase + " "
                    
                    # Add the last phrase chunk if not empty
                    if phrase_chunk.strip():
                        current_chunk = phrase_chunk.strip()
                    
                # Normal case - add sentence to current chunk if it fits
                elif len(current_chunk) + len(sentence) < max_chunk_length:
                    current_chunk += sentence + " "
                else:
                    # Current chunk is full, store it and start a new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            # Add the last chunk if not empty
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                
            # Ensure we have at least one chunk
            if not chunks and text:
                # Emergency fallback - split by a fixed size
                chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            
            print(f"Split text into {len(chunks)} chunks for TTS processing")
            
            # Process each chunk and combine the audio files
            chunk_files = []
            
            for i, chunk in enumerate(chunks):
                chunk_filename = f"{AUDIO_PATH}/chunk_{timestamp}_{i}.mp3"
                try:
                    chunk_tts = gTTS(text=chunk, lang=lang, slow=False)
                    chunk_tts.save(chunk_filename)
                    chunk_files.append(chunk_filename)
                    print(f"Generated chunk {i+1}/{len(chunks)}")
                except Exception as chunk_error:
                    print(f"Error processing chunk {i}: {chunk_error}")
            
            # Combine audio files if we have at least one successful chunk
            if chunk_files:
                try:
                    from pydub import AudioSegment
                    
                    combined = AudioSegment.empty()
                    for chunk_file in chunk_files:
                        segment = AudioSegment.from_mp3(chunk_file)
                        combined += segment
                    
                    # Save the combined file
                    combined.export(filename, format="mp3")
                    
                    # Clean up chunk files
                    for chunk_file in chunk_files:
                        try:
                            os.remove(chunk_file)
                        except:
                            pass
                            
                except ImportError:
                    print("pydub not available, using first chunk as response")
                    if chunk_files:
                        filename = chunk_files[0]
                except Exception as combine_error:
                    print(f"Error combining audio chunks: {combine_error}")
                    # Fall back to first chunk if combining fails
                    if chunk_files:
                        filename = chunk_files[0]
            else:
                # If all chunks failed, try with a shortened version of the text
                try:
                    print("All chunks failed, falling back to shortened text")
                    shortened_text = text[:max_chunk_length] + "..."
                    tts = gTTS(text=shortened_text, lang=lang, slow=False)
                    tts.save(filename)
                except Exception as fallback_error:
                    print(f"Fallback TTS also failed: {fallback_error}")
                    return None
        
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
        print("Supply chain data loaded successfully.")
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
        print("Weather data loaded successfully.")
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
        print("Google Trends data loaded successfully.")
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

# Load, process, and index documents from the documents directory
def load_and_process_documents():
    """
    Load documents from the documents directory,
    process them into chunks, and add them to the vector store
    """
    print("Loading documents from 'documents' directory...")
    
    # Base supply chain knowledge - always include these for consistent performance
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
    
    all_docs = list(supply_chain_docs)  # Start with base knowledge
    doc_meta = {}  # Store metadata about loaded documents
    
    # Create documents directory if it doesn't exist
    documents_dir = "documents"
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        print(f"Created documents directory at {documents_dir}")
    
    # Initialize text splitter for document chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Process PDF files
    if PDF_SUPPORT:
        pdf_files = [f for f in os.listdir(documents_dir) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files in documents directory")
        
        for pdf_file in pdf_files:
            file_path = os.path.join(documents_dir, pdf_file)
            try:
                print(f"Processing PDF file: {pdf_file}")
                
                # Extract text from PDF
                pdf_text = process_pdf(file_path)
                if pdf_text:
                    # Split text into chunks
                    chunks = text_splitter.split_text(pdf_text)
                    print(f"  - Extracted {len(chunks)} text chunks from {pdf_file}")
                    
                    # Add chunks to documents list
                    all_docs.extend(chunks)
                    
                    # Store metadata
                    doc_meta[pdf_file] = {
                        'type': 'pdf',
                        'chunks': len(chunks),
                        'size': os.path.getsize(file_path),
                        'added_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                else:
                    print(f"  - No text could be extracted from {pdf_file}")
            except Exception as e:
                print(f"  - Error processing {pdf_file}: {str(e)}")
    else:
        print("PDF support is not available. Install pypdf package to enable PDF processing.")
    
    # Process plain text files
    txt_files = [f for f in os.listdir(documents_dir) if f.lower().endswith('.txt')]
    print(f"Found {len(txt_files)} text files in documents directory")
    
    for txt_file in txt_files:
        file_path = os.path.join(documents_dir, txt_file)
        try:
            print(f"Processing text file: {txt_file}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Split text into chunks
            chunks = text_splitter.split_text(text)
            print(f"  - Extracted {len(chunks)} text chunks from {txt_file}")
            
            # Add chunks to documents list
            all_docs.extend(chunks)
            
            # Store metadata
            doc_meta[txt_file] = {
                'type': 'txt',
                'chunks': len(chunks),
                'size': os.path.getsize(file_path),
                'added_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            print(f"  - Error processing {txt_file}: {str(e)}")
    
    # Save document metadata
    try:
        with open(os.path.join(documents_dir, 'document_metadata.json'), 'w') as f:
            json.dump(doc_meta, f, indent=2)
    except Exception as e:
        print(f"Error saving document metadata: {str(e)}")
    
    print(f"Total documents/chunks for vector store: {len(all_docs)}")
    return all_docs

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Load documents and create vector store
print("Initializing knowledge base...")
supply_chain_docs = load_and_process_documents()
vector_store = FAISS.from_texts(supply_chain_docs, embeddings)
vector_store.save_local(VECTOR_DB_PATH)
print(f"Knowledge base created and saved at {VECTOR_DB_PATH}")

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
        model = Prophet(yearly_seasonality=True, monthly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        
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
        
        # If product_id not found in data, generate realistic mock forecast data
        # First try direct match, then try case-insensitive match
        product_data = self.data[self.data['product_id'] == product_id]
        if product_data.empty:
            # Try case-insensitive match
            product_data = self.data[self.data['product_id'].str.upper() == product_id.upper()]
        
        if product_data.empty:
            print(f"No data found for product {product_id}. Generating mock forecast.")
            # Generate mock forecast data with more realistic values based on product_id
            last_date = datetime.now()
            forecast_dates = pd.date_range(start=last_date, periods=steps)
            
            # Create a sine wave pattern with increasing trend for mock forecast
            import numpy as np
            x = np.arange(steps)
            
            # Base values on product ID to get consistent but different patterns
            if 'PHN' in product_id or 'PHONE' in product_id.upper():
                base_value = 250 # Phones have higher base demand
                trend_factor = 3  # Steeper trend
                season_amp = 20   # Stronger seasonality
            elif 'LAP' in product_id or 'LAPTOP' in product_id.upper():
                base_value = 200
                trend_factor = 2.5
                season_amp = 25
            elif 'TAB' in product_id or 'TABLET' in product_id.upper():
                base_value = 180
                trend_factor = 2
                season_amp = 15
            else:
                base_value = 150
                trend_factor = 1.5
                season_amp = 10
            
            trend = base_value + x * trend_factor  # increasing trend
            seasonality = season_amp * np.sin(x * 2 * np.pi / 7)  # weekly seasonality
            noise = np.random.normal(0, 5, steps)  # random noise
            forecast_values = trend + seasonality + noise
            
            # Add realistic confidence bounds that widen over time
            uncertainty_factor = np.linspace(5, 20, steps)  # Increasing uncertainty
            upper_bound = forecast_values + uncertainty_factor + noise
            lower_bound = forecast_values - uncertainty_factor + noise
            
            # Ensure lower bound doesn't go below zero for products
            lower_bound = np.maximum(lower_bound, 0)
            
            # Create DataFrame
            mock_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast_values,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound
            })
            
            return mock_df
        
        # Train model if not already trained
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
        import numpy as np
        
        # Update parameters if provided
        if demand_forecast is not None:
            self.forecast = demand_forecast
            self.products = list(demand_forecast.keys())
        
        if warehouse_capacities is not None:
            self.warehouse_capacities = warehouse_capacities
            self.locations = list(warehouse_capacities.keys())
        
        if holding_costs is not None:
            self.holding_costs = holding_costs
        
        if transportation_costs is not None:
            self.transportation_costs = transportation_costs
        
        if lead_times is not None:
            self.lead_times = lead_times
            
        # If no products, return a simple default result
        if not self.products:
            print("Warning: No products in demand forecast")
            return {
                "status": "No products specified",
                "total_cost": 0,
                "locations": {loc: {"total_inventory": 0, "capacity": cap, "capacity_utilization": 0, "products": {}} 
                            for loc, cap in self.warehouse_capacities.items()}
            }
        
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
                    
                    # Distribute inventory across locations proportionally
                    # This approach avoids the index out of range error by not directly using idx
                    total_capacity = sum(self.warehouse_capacities.values())
                    loc_capacity = self.warehouse_capacities[loc]
                    loc_ratio = loc_capacity / total_capacity if total_capacity > 0 else 0
                    
                    for prod in self.products:
                        # Use proportional allocation instead of direct indexing
                        if prod in results["inventory"]:
                            inv_level = results["inventory"][prod] * loc_ratio
                            products_inv[prod] = float(inv_level)
                            total_inv += inv_level
                        else:
                            products_inv[prod] = 0.0
                    
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
            # First add upper bound
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['upper_bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            # Then add lower bound with fill between
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['lower_bound'],
                mode='lines',
                name='Confidence Interval',
                fill='tonexty',
                fillcolor='rgba(0, 176, 246, 0.2)',
                line=dict(width=0)
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
def post_process_response(text, language="en"):
    """
    Post-process the LLM response to make it more suitable for conversation and TTS
    
    This function cleans up the response to ensure it's well-formatted for speech synthesis
    and doesn't contain elements that would sound awkward when spoken.
    
    It also ensures responses are formatted in 2-3 paragraphs without numbered phrases.
    """
    import re
    
    # Skip if the text is empty or too short
    if not text or len(text.strip()) < 5:
        return text
    
    # Remove any markdown code blocks
    text = re.sub(r'```(?:[a-zA-Z]+)?\n[\s\S]*?\n```', ' ', text)
    
    # Remove JSON-like structures 
    text = re.sub(r'\{[\s\S]*?\}', ' ', text)
    
    # Remove numbered list markers (1., 2., etc.) 
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove bullet points
    text = re.sub(r'^\s*[\*\-•]\s+', '', text, flags=re.MULTILINE)
    
    # Replace symbols that don't work well in speech
    replacements = {
        '```': ' ',
        '**': ' ',
        '*': ' ',
        '#': ' ',
        '|': ', ',
        '=': ' equals ',
        '->': ' to ',
        '<-': ' from ',
        '>=': ' greater than or equal to ',
        '<=': ' less than or equal to ',
        '>': ' greater than ',
        '<': ' less than ',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Fix spacing issues
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Ensure proper sentence breaks for TTS natural pauses
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Handle numbers for better speech (e.g., "100,000" -> "100 thousand")
    def number_to_words(match):
        num = int(match.group(0).replace(',', ''))
        if num >= 1000000:
            return f"{num // 1000000} million"
        elif num >= 1000:
            return f"{num // 1000} thousand"
        return match.group(0)
    
    text = re.sub(r'\b\d{4,}(?:,\d{3})*\b', number_to_words, text)
    
    # Language-specific post-processing
    if language == "fr":
        # Fix French spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        # Fix spaces before double punctuation in French
        text = re.sub(r'([;:!?])(?!\s)', r'\1 ', text)
    
    # Remove any system prompt leakage phrases
    system_leakage_phrases = [
        "As a supply chain assistant",
        "As an AI assistant",
        "As your AI assistant",
        "As a helpful assistant",
        "I don't have access to",
        "I don't have the ability to",
        "I'm not able to",
        "I cannot access",
        "As RawajAI",
        "As a language model",
        "I'm here to help",
        "I'm an AI",
        "I'm a supply chain",
        "I'm a helpful",
    ]
    
    for phrase in system_leakage_phrases:
        if text.startswith(phrase):
            text = text[len(phrase):].strip()
            # If the text now starts with certain connecting words, clean them up
            text = re.sub(r'^[,.]?\s*(but|however|nevertheless|yet|still|I can|I will|I could|I would)', '', text, flags=re.IGNORECASE).strip()
            
    # Remove any text inside brackets (common metadata marker)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove any XML-like tags (some models output these)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Ensure the text doesn't end with an incomplete sentence
    if not re.search(r'[.!?]$', text):
        text += "."
    
    # Format into 2-3 paragraphs with natural breaks
    paragraphs = re.split(r'\n\s*\n', text)
    
    # If we have too many paragraphs, consolidate them
    if len(paragraphs) > 3:
        # Group paragraphs into 2-3 consolidated paragraphs
        new_paragraphs = []
        current_paragraph = ""
        target_paragraphs = min(3, max(2, len(paragraphs) // 2))
        
        for i, p in enumerate(paragraphs):
            if len(new_paragraphs) < target_paragraphs - 1:
                if current_paragraph:
                    current_paragraph += " " + p.strip()
                    if len(current_paragraph) > 200:  # Natural paragraph size
                        new_paragraphs.append(current_paragraph)
                        current_paragraph = ""
                else:
                    current_paragraph = p.strip()
            else:
                # Add all remaining paragraphs to the last consolidated paragraph
                current_paragraph += " " + p.strip()
        
        # Add any remaining text as the last paragraph
        if current_paragraph:
            new_paragraphs.append(current_paragraph)
            
        # Use consolidated paragraphs
        paragraphs = new_paragraphs
    
    # Join paragraphs with double newlines for proper spacing
    text = "\n\n".join(p for p in paragraphs if p.strip())
    
    # Final clean-up for multiple spaces and ensure good punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\n\2', text)
    
    return text

def generate_response(query, context="", language="en"):
    """Generate a user-friendly response optimized for conversation and audio using prompt engineering"""
    try:
        # Validate inputs
        if not query:
            return "I didn't receive a question to answer."
        
        # Define persona and style guides for different languages - optimized for Mistral 7B
        persona = {
            "en": "You are RawajAI, a specialized supply chain management assistant. NEVER include any meta tags, formatting markers like [], or system prompts in your response. Write in plain text only. Speak directly to the user about supply chain topics as if you're having a natural conversation. Explain supply chain concepts simply. Ensure all responses focus exclusively on supply chain management, inventory optimization, logistics, procurement, demand forecasting, and related business areas.",
            "fr": "Vous êtes RawajAI, un assistant spécialisé en gestion de chaîne d'approvisionnement. N'incluez JAMAIS de balises méta, de marqueurs de formatage comme [], ou d'instructions système dans votre réponse. Écrivez en texte brut uniquement. Parlez directement à l'utilisateur des sujets de chaîne d'approvisionnement comme lors d'une conversation naturelle. Expliquez les concepts de chaîne d'approvisionnement simplement. Assurez-vous que toutes les réponses se concentrent exclusivement sur la gestion de la chaîne d'approvisionnement, l'optimisation des stocks, la logistique, les achats, la prévision de la demande et les domaines commerciaux connexes.",
            "ar": "أنت RawajAI، مساعد متخصص في إدارة سلسلة التوريد. لا تضمن أبدًا أي علامات وصفية، أو علامات تنسيق مثل []، أو تعليمات نظام في ردك. اكتب بنص عادي فقط. تحدث مباشرة إلى المستخدم حول مواضيع سلسلة التوريد كما لو كنت تجري محادثة طبيعية. اشرح مفاهيم سلسلة التوريد ببساطة. تأكد من أن جميع الردود تركز حصريًا على إدارة سلسلة التوريد، وتحسين المخزون، والخدمات اللوجستية، والمشتريات، والتنبؤ بالطلب، ومجالات الأعمال ذات الصلة."
        }
        
        # Audio-friendly response guidelines - optimized for clarity and to prevent metadata
        audio_guidelines = {
            "en": "Create responses that work well when read aloud. Use short, clear sentences under 20 words when possible. Avoid abbreviations, codes, special characters, URLs, or any text that doesn't sound natural in speech. Spell out numbers below 10. Always round statistics (e.g., use 42% instead of 41.7%). NEVER include text inside brackets, parentheses, or any system-style formatting in your final response.",
            "fr": "Créez des réponses qui fonctionnent bien lorsqu'elles sont lues à haute voix. Utilisez des phrases courtes et claires de moins de 20 mots lorsque possible. Évitez les abréviations, codes, caractères spéciaux, URLs, ou tout texte qui ne sonne pas naturel à l'oral. Écrivez en toutes lettres les nombres inférieurs à 10. Arrondissez toujours les statistiques (par exemple, utilisez 42% au lieu de 41,7%). N'incluez JAMAIS de texte entre crochets, parenthèses ou tout formatage de type système dans votre réponse finale.",
            "ar": "أنشئ ردودًا تعمل بشكل جيد عند قراءتها بصوت عالٍ. استخدم جملاً قصيرة وواضحة أقل من 20 كلمة عندما يكون ذلك ممكنًا. تجنب الاختصارات، الرموز، الأحرف الخاصة، عناوين URL، أو أي نص لا يبدو طبيعيًا في الكلام. اكتب الأرقام أقل من 10 بالحروف. قم دائمًا بتقريب الإحصائيات (مثلاً، استخدم 42% بدلاً من 41.7%). لا تضمن أبدًا نصًا داخل أقواس معقوفة أو أقواس أو أي تنسيق على طريقة النظام في ردك النهائي."
        }
        
        # Get language-specific response style - optimized for conversation and domain expertise
        response_style = {
            "en": "Talk like a knowledgeable supply chain expert having a natural conversation. Use practical examples from retail, manufacturing, or distribution. Connect supply chain concepts to business outcomes like cost reduction, efficiency gains, and improved customer satisfaction. Don't use academic language - explain everything as if speaking to a business professional who needs clear advice. Avoid disclaimers about being an AI or language model.",
            "fr": "Parlez comme un expert en chaîne d'approvisionnement ayant une conversation naturelle. Utilisez des exemples pratiques du commerce de détail, de la fabrication ou de la distribution. Reliez les concepts de la chaîne d'approvisionnement aux résultats commerciaux comme la réduction des coûts, les gains d'efficacité et l'amélioration de la satisfaction client. N'utilisez pas de langage académique - expliquez tout comme si vous parliez à un professionnel qui a besoin de conseils clairs. Évitez les avertissements sur le fait d'être une IA ou un modèle de langage.",
            "ar": "تحدث كخبير مطلع في سلسلة التوريد يجري محادثة طبيعية. استخدم أمثلة عملية من البيع بالتجزئة أو التصنيع أو التوزيع. اربط مفاهيم سلسلة التوريد بنتائج الأعمال مثل خفض التكاليف وزيادة الكفاءة وتحسين رضا العملاء. لا تستخدم لغة أكاديمية - اشرح كل شيء كما لو كنت تتحدث إلى محترف أعمال يحتاج إلى نصائح واضحة. تجنب إخلاء المسؤولية عن كونك ذكاءً اصطناعيًا أو نموذج لغة."
        }
        
        # Format prompt with optimized structure for Mistral 7B to prevent metadata leakage
        prompt = f"""<System>
{persona.get(language, persona["en"])}

Remember: You are RawajAI, focusing exclusively on supply chain topics. Never output system tags, never use brackets in output.

Reference Context:
{context}

Style Guidelines:
{audio_guidelines.get(language, audio_guidelines["en"])}
{response_style.get(language, response_style["en"])}

Output Format Instructions:
- Respond in 2-3 short paragraphs of natural conversational text
- Each paragraph should have 3-4 sentences developing one main idea
- Use clear transitions between paragraphs
- Never use bullet points, numbered lists or headings
- Avoid all formatting symbols, brackets, special characters
- Round all numbers and statistics
- Never refer to yourself as an AI or assistant
- Your response should be direct answer only, with no system tags

Important: Your response must ONLY contain the direct answer to the user's question with NO formatting markers.
</System>

<Question>
{query}
</Question>

<Answer>"""
        
        # Generate response with the LLM with better error handling
        try:
            # Log the user query for debugging
            print(f"Generating response for query: {query[:50]}...")
            
            # Generate response with the LLM using improved settings for conversational responses
            response = llm_pipeline(
                prompt,
                max_new_tokens=512,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.72,      # Slightly increased for more natural responses
                top_p=0.92,           # Nucleus sampling
                top_k=40,             # Limit vocabulary diversity
                repetition_penalty=1.18  # Increased to further reduce repetition
            )
            
            # Extract and clean the response
            if response and len(response) > 0:
                generated_text = response[0]['generated_text']
                
                # Extract text after the <Answer> marker for the new format
                if "<Answer>" in generated_text:
                    raw_result = generated_text.split("<Answer>")[-1].strip()
                    # Remove any closing tag if present
                    raw_result = raw_result.split("</Answer>")[0].strip()
                elif "[ASSISTANT RESPONSE]" in generated_text:
                    # Legacy format fallback
                    raw_result = generated_text.split("[ASSISTANT RESPONSE]")[-1].strip()
                else:
                    # Fallback to extract response after the last line of the prompt
                    raw_result = "\n".join(generated_text.split("\n")[prompt.count("\n"):]).strip()
                
                # Log the raw output for debugging
                print(f"Raw LLM output (first 100 chars): {raw_result[:100]}...")
                
                # Check for common hallucination patterns or invalid outputs
                hallucination_patterns = [
                    r"```json",                # JSON code block
                    r"^\s*\{\s*\"",            # JSON object start
                    r"^\s*\[\s*\{",            # JSON array start
                    r"as an AI language model", # Self-reference
                    r"I don't have access to",  # Limitation statement
                    r"I cannot provide",        # Refusal pattern
                    r"<.*?>",                  # HTML tags
                    r"\[.*?\]",                # Content within square brackets
                    r"\[ASSISTANT.*?\]",       # System tags
                    r"\[RESPONSE.*?\]",        # Response tags
                    r"<System>|</System>",     # System tags
                    r"<Answer>|</Answer>",     # Answer tags
                    r"<Question>|</Question>"  # Question tags
                ]
                
                contains_hallucination = any(re.search(pattern, raw_result) for pattern in hallucination_patterns)
                
                if contains_hallucination:
                    print(f"Detected potential hallucination/invalid format, regenerating...")
                    
                    # Add explicit instruction to fix the issue using the same XML-style format
                    retry_prompt = prompt + "\n\nIMPORTANT: DO NOT USE JSON FORMAT, CODE BLOCKS, OR BRACKETS IN YOUR RESPONSE. RESPOND IN PLAIN, NATURAL CONVERSATIONAL TEXT ONLY.\n\n"
                    
                    # Regenerate with stricter settings
                    retry_response = llm_pipeline(
                        retry_prompt,
                        max_new_tokens=512,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.5,      # Lower temperature for more focused response
                        top_p=0.85,          # More constrained sampling
                        repetition_penalty=1.2
                    )
                    
                    if retry_response and len(retry_response) > 0:
                        retry_text = retry_response[0]['generated_text']
                        if "[ASSISTANT RESPONSE]" in retry_text:
                            raw_result = retry_text.split("[ASSISTANT RESPONSE]")[-1].strip()
                        else:
                            raw_result = "\n".join(retry_text.split("\n")[retry_prompt.count("\n"):]).strip()
                            
                        print("Successfully regenerated response")
                
                # Post-process the response to make it more suitable for audio
                result = post_process_response(raw_result, language)
                
                # Ensure we got a meaningful response
                if not result or len(result) < 15:  # Increased minimum length threshold
                    print("Generated response too short or invalid, using fallback")
                    fallback_responses = {
                        "en": "I'm sorry, I couldn't generate a specific answer to your question. Could you please rephrase your question or provide more details about what you'd like to know?",
                        "fr": "Je suis désolé, je n'ai pas pu générer une réponse spécifique à votre question. Pourriez-vous reformuler votre question ou fournir plus de détails sur ce que vous souhaitez savoir?",
                        "ar": "آسف، لم أتمكن من إنشاء إجابة محددة لسؤالك. هل يمكنك إعادة صياغة سؤالك أو تقديم المزيد من التفاصيل حول ما ترغب في معرفته؟"
                    }
                    return fallback_responses.get(language, fallback_responses["en"])
                
                return result
            else:
                return "I couldn't generate a response. Please try asking your question differently."
                
        except Exception as gen_error:
            print(f"Error during LLM generation: {gen_error}")
            error_responses = {
                "en": "I encountered an error while processing your question. Please try again with a different query.",
                "fr": "J'ai rencontré une erreur lors du traitement de votre question. Veuillez réessayer avec une autre requête.",
                "ar": "لقد واجهت خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى باستخدام استعلام مختلف."
            }
            return error_responses.get(language, error_responses["en"])
            
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
        
        print(f"Finding context for query: {query[:50]}...")
        
        # Enhance query with supply chain specific terminology for better matching
        enhanced_query = enhance_supply_chain_query(query)
        
        # Add timeout handling for vector store search
        docs = []
        try:
            # Retrieve more documents for better context coverage
            # Using MMR search for better diversity in results
            docs = vector_store.max_marginal_relevance_search(
                enhanced_query,
                k=5,           # Get 5 documents
                fetch_k=10,    # Fetch 10 initially and select diverse subset
                lambda_mult=0.7 # Balance between relevance and diversity
            )
            print(f"Retrieved {len(docs)} relevant documents from knowledge base")
        except Exception as search_error:
            print(f"Error in vector search: {search_error}")
            
            # Try standard similarity search as fallback
            try:
                docs = vector_store.similarity_search(enhanced_query, k=3)
                print(f"Fallback search retrieved {len(docs)} documents")
            except:
                # Return empty context on error
                return ""
        
        # Process retrieved documents
        if not docs:
            return ""
            
        # Extract and join document content with better source attribution
        context_parts = []
        for i, doc in enumerate(docs):
            if hasattr(doc, 'page_content') and doc.page_content:
                # Default source info
                source_info = f"[Knowledge {i+1}]"
                
                # Extract better source information if available
                if hasattr(doc, 'metadata') and doc.metadata:
                    if 'source' in doc.metadata:
                        source_name = doc.metadata['source']
                        # Extract filename without path
                        if isinstance(source_name, str) and '/' in source_name:
                            source_name = source_name.split('/')[-1]
                        source_info = f"[Source: {source_name}]"
                    elif 'page' in doc.metadata:
                        source_info = f"[Page: {doc.metadata['page']}]"
                
                # Clean the content for better readability
                content = doc.page_content.strip()
                
                # Add document to context with better formatting
                context_parts.append(f"{source_info}\n{content}")
        
        context = "\n\n".join(context_parts)
        
        print(f"RAG context retrieved ({len(docs)} documents, {len(context)} chars)")
        return context
    except Exception as e:
        print(f"Error retrieving RAG context: {e}")
        traceback.print_exc()
        return ""

def enhance_supply_chain_query(query):
    """Enhance the query with supply chain specific terminology for better matching"""
    # Check if query already contains supply chain terminology
    supply_chain_terms = [
        "inventory", "logistics", "procurement", "warehouse", "distribution",
        "supply chain", "forecasting", "lead time", "demand planning", "stockout",
        "backorder", "bullwhip effect", "just-in-time", "jit", "lean", "kpi",
        "supplier", "vendor", "transportation", "shipping", "fulfillment",
        # Additional French terms
        "chaîne d'approvisionnement", "logistique", "entrepôt", "prévision",
        "délai de livraison", "rupture de stock", "fournisseur", "transport",
        # Additional Arabic terms
        "سلسلة التوريد", "المخزون", "اللوجستية", "المشتريات", "المستودع"
    ]
    
    # If query doesn't contain supply chain terms, add context
    has_supply_chain_term = any(term in query.lower() for term in supply_chain_terms)
    
    if not has_supply_chain_term:
        # Add supply chain context to query
        enhanced_query = f"supply chain management: {query}"
        print(f"Enhanced query with supply chain context: {enhanced_query[:50]}...")
        return enhanced_query
    
    return query

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
        
        # Convert to chart data for JSON response with improved styling
        chart_data = [
            {
                'x': forecast_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'y': forecast_df['forecast'].tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Forecast',
                'line': {'width': 3, 'color': 'rgb(31, 119, 180)'},
                'marker': {'size': 6, 'color': 'rgb(31, 119, 180)'},
                'hovertemplate': 'Date: %{x}<br>Demand: %{y:.0f}<extra></extra>'
            }
        ]
        
        # Add confidence intervals if available
        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            # First add upper bound
            chart_data.append({
                'x': forecast_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'y': forecast_df['upper_bound'].tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Upper Bound',
                'line': {'width': 0, 'color': 'rgba(31, 119, 180, 0.3)'},
                'showlegend': False,
                'hoverinfo': 'none'
            })
            
            # Then add lower bound with fill between
            chart_data.append({
                'x': forecast_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'y': forecast_df['lower_bound'].tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Confidence Interval',
                'fill': 'tonexty',
                'fillcolor': 'rgba(31, 119, 180, 0.2)',
                'line': {'width': 0, 'color': 'rgba(31, 119, 180, 0.3)'},
                'hoverinfo': 'none'
            })
        
        # Log successful response for debugging
        print(f"Successfully generated forecast for product {product_id}")
        
        # Calculate basic metrics if possible
        metrics = {}
        if 'actual' in forecast_df.columns and len(forecast_df['actual'].dropna()) > 0:
            from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
            import numpy as np
            
            actual = forecast_df['actual'].dropna()
            predicted = forecast_df['forecast'][:len(actual)]
            
            if len(actual) > 0 and len(predicted) > 0:
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                mape = mean_absolute_percentage_error(actual, predicted) * 100
                metrics = {
                    'rmse': float(rmse),
                    'mape': float(mape),
                    'accuracy': float(100 - mape) if mape < 100 else 0
                }
        
        # Generate recommendations based on forecast trend
        trend_increasing = False
        trend_decreasing = False
        
        if len(forecast_df) >= 3:
            recent_values = forecast_df['forecast'].tail(int(len(forecast_df) / 3))
            if recent_values.is_monotonic_increasing:
                trend_increasing = True
            elif recent_values.is_monotonic_decreasing:
                trend_decreasing = True
                
        recommendations = []
        if trend_increasing:
            recommendations.append(f"Demand for {product_id} is trending upward. Consider increasing inventory levels.")
        elif trend_decreasing:
            recommendations.append(f"Demand for {product_id} is trending downward. Consider reducing inventory to avoid excess.")
        else:
            recommendations.append(f"Demand for {product_id} appears stable. Maintain current inventory levels.")
            
        # Add seasonality detection if possible
        if len(forecast_df) >= 14:
            recommendations.append("Review forecast for potential seasonal patterns and adjust inventory accordingly.")
        
        return jsonify({
            "status": "success",
            "message": f"Demand forecast generated for {product_id}",
            "product_id": product_id,
            "forecast": forecast_df['forecast'].tolist(),
            "dates": forecast_df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "chart_data": chart_data,
            "metrics": metrics,
            "recommendations": recommendations,
            "forecast_period": {
                "start_date": forecast_df['date'].iloc[0].strftime('%Y-%m-%d'),
                "end_date": forecast_df['date'].iloc[-1].strftime('%Y-%m-%d'),
                "days": days
            }
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
        
        # Get ordering cost from the request
        ordering_cost = data.get('ordering_cost', 100)  # Default to 100 if not provided
        
        # Extract product_id for targeted optimization
        product_id = data.get('product_id')
        if product_id and product_id not in holding_costs:
            # Add the product to holding costs if it's not there
            holding_costs[product_id] = data.get('holding_costs', {}).get(product_id, 10)
        
        # Log inputs for debugging
        print(f"Optimization inputs: product_id={product_id}, ordering_cost={ordering_cost}")
        
        # If empty demand forecast, use sample data or generate based on product_id
        if not demand_forecast:
            if product_id:
                demand_forecast = {product_id: 500}  # Default demand for the specified product
            else:
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
            
            # Calculate EOQ, reorder point, and safety stock for the product
            product_id = data.get('product_id')
            eoq = None
            reorder_point = None
            safety_stock = None
            total_cost = optimization_results.get("total_cost", 0)
            
            if product_id:
                # Extract parameters needed for calculations
                holding_cost = holding_costs.get(product_id, 10)
                ordering_cost = data.get('ordering_cost', 100)
                demand = demand_forecast.get(product_id, 500)  # Default to 500 units if not specified
                lead_time = data.get('lead_times', {}).get(product_id, 7)  # Default to 7 days
                service_level = data.get('service_level', 0.95)
                
                # Wilson's EOQ formula: sqrt(2 * D * S / H)
                eoq = round(float(np.sqrt((2 * demand * ordering_cost) / holding_cost)), 2)
                
                # Calculate reorder point based on lead time and demand
                avg_daily_demand = demand / 30  # Assuming monthly demand
                reorder_point = round(float(avg_daily_demand * lead_time), 2)
                
                # Safety stock: lead_time * daily demand * service factor
                service_factor = 1.65 if service_level >= 0.95 else 1.3 if service_level >= 0.9 else 1.0
                safety_stock = round(float(avg_daily_demand * np.sqrt(lead_time) * service_factor), 2)
                
                # Calculate total cost if not provided
                if not total_cost:
                    total_cost = round(float((holding_cost * eoq / 2) + (ordering_cost * demand / eoq)), 2)
            
            # Create visualization - with error handling
            try:
                fig = visualization_engine.create_inventory_chart(
                    optimization_results, language=language
                )
            except Exception as viz_error:
                print(f"Visualization error: {viz_error}")
                # Continue without visualization if it fails
                fig = None
        
        # Add EOQ, reorder point, and safety stock to main response
        if product_id and 'eoq' in locals() and eoq is not None:
            optimization_results['eoq'] = eoq
            optimization_results['reorder_point'] = reorder_point
            optimization_results['safety_stock'] = safety_stock
            optimization_results['total_cost'] = total_cost
            
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
                # Make sure all values are proper numbers and not NaN
                import numpy as np
                total_inventories = []
                utilizations = []
                
                for loc_data in optimization_results["locations"].values():
                    inv_val = loc_data.get("total_inventory", 0)
                    util_val = loc_data.get("capacity_utilization", 0)
                    
                    # Replace NaN with 0
                    total_inventories.append(0 if np.isnan(inv_val) else float(inv_val))
                    utilizations.append(0 if np.isnan(util_val) else float(util_val))
            
            # Create enhanced location-based chart data with better styling    
            chart_data = [{
                'x': locations,
                'y': total_inventories,
                'type': 'bar',
                'name': 'Total Inventory',
                'marker': {
                    'color': 'rgba(58, 71, 80, 0.6)',
                    'line': {'color': 'rgba(58, 71, 80, 1.0)', 'width': 1}
                },
                'hovertemplate': 'Location: %{x}<br>Inventory: %{y:.0f} units<extra></extra>'
            }, {
                'x': locations,
                'y': utilizations,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Utilization %',
                'yaxis': 'y2',
                'line': {'color': 'rgba(231, 76, 60, 1.0)', 'width': 3},
                'marker': {'size': 8, 'color': 'rgba(231, 76, 60, 1.0)'},
                'hovertemplate': 'Location: %{x}<br>Utilization: %{y:.1f}%<extra></extra>'
            }]
            
            # If we have EOQ calculations, add a second chart for inventory parameters
            if eoq and reorder_point and safety_stock:
                param_chart = {
                    'x': ['EOQ', 'Reorder Point', 'Safety Stock'],
                    'y': [eoq, reorder_point, safety_stock],
                    'type': 'bar',
                    'marker': {
                        'color': ['rgba(52, 152, 219, 0.7)', 'rgba(241, 196, 15, 0.7)', 'rgba(46, 204, 113, 0.7)'],
                        'line': {'color': 'rgba(0, 0, 0, 0.5)', 'width': 1}
                    },
                    'name': 'Inventory Parameters',
                    'hovertemplate': '%{x}: %{y:.1f} units<extra></extra>',
                    'xaxis': 'x2',
                    'yaxis': 'y3'
                }
                chart_data.append(param_chart)
                
                # Add cost breakdown visualization
                if 'total_cost' in locals() and total_cost:
                    holding_cost = holding_costs.get(product_id, 10)
                    ordering_cost = data.get('ordering_cost', 100)
                    demand = demand_forecast.get(product_id, 500)
                    
                    annual_holding_cost = holding_cost * eoq / 2
                    annual_ordering_cost = ordering_cost * demand / eoq
                    
                    cost_chart = {
                        'x': ['Holding Cost', 'Ordering Cost'],
                        'y': [annual_holding_cost, annual_ordering_cost],
                        'type': 'bar',
                        'marker': {
                            'color': ['rgba(155, 89, 182, 0.7)', 'rgba(230, 126, 34, 0.7)'],
                            'line': {'color': 'rgba(0, 0, 0, 0.5)', 'width': 1}
                        },
                        'name': 'Annual Cost Breakdown',
                        'hovertemplate': '%{x}: $%{y:.2f}<extra></extra>',
                        'xaxis': 'x3',
                        'yaxis': 'y4'
                    }
                    chart_data.append(cost_chart)
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
            "chart_data": chart_data,
            "eoq": eoq,
            "reorder_point": reorder_point,
            "safety_stock": safety_stock,
            "total_cost": total_cost,
            "product_id": product_id
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
            # Use thread-safe timeout mechanism instead of SIGALRM
            import threading
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            
            # Retrieve relevant context
            processing_log.append("Retrieving context")
            context = get_rag_context(query)
            if not context:
                context = "No specific context available for this query."
            
            # Generate text response with timeout
            processing_log.append("Generating response")
            
            with ThreadPoolExecutor() as executor:
                future = executor.submit(generate_response, query, context, language)
                try:
                    response = future.result(timeout=60)  # 60 second timeout
                except TimeoutError:
                    print("Response generation timed out")
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
            
            # Generate speech if requested
            speech_file = None
            speech_url = None
            
            if generate_speech_output:
                processing_log.append("Generating speech")
                speech_file = generate_speech(response, language)
                if speech_file:
                    speech_url = f"/audio/{os.path.basename(speech_file)}"
            
            # Build the response
            result = {
                "query": query,
                "response": response,
                "language": language,
                "success": True
            }
            
            # Add speech file URL if available
            if speech_url:
                result["speech_url"] = speech_url
                
            return jsonify(result)
                
        except Exception as inner_e:
            # Log the detailed error for debugging
            print(f"Error processing question: {str(inner_e)}")
            print(f"Processing steps completed: {', '.join(processing_log)}")
            
            # Create a user-friendly error message based on the type of error
            error_message = str(inner_e)
            error_response = "I'm having trouble processing your question right now. Please try again later."
            
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
    """Serve generated audio files with proper headers for streaming"""
    from flask import Response, stream_with_context
    import os
    
    filepath = f"{AUDIO_PATH}/{filename}"
    
    # Verify file exists
    if not os.path.exists(filepath):
        return jsonify({"error": "Audio file not found"}), 404
    
    # Get file size for Content-Length header
    file_size = os.path.getsize(filepath)
    
    # Stream file in chunks to avoid loading entire file into memory
    def generate():
        chunk_size = 4096  # 4KB chunks
        with open(filepath, 'rb') as audio_file:
            while True:
                chunk = audio_file.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    # Set headers for proper streaming and caching behavior
    headers = {
        'Content-Type': 'audio/mpeg',
        'Content-Length': str(file_size),
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'public, max-age=86400',  # Cache for a day
        'X-Accel-Buffering': 'yes'  # Enable nginx buffering if using nginx
    }
    
    return Response(
        stream_with_context(generate()),
        headers=headers,
        status=200,
        mimetype='audio/mpeg',
        direct_passthrough=True
    )

@app.route('/ask_tts', methods=['POST'])
def ask_question_with_tts():
    """Answer supply chain questions using RAG and return text-to-speech audio with support for long responses"""
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
        
        # Check if client wants to handle long responses
        handle_long_responses = data.get('handle_long_responses', False)
        
        # Track processing steps and timing for debugging
        processing_log = []
        start_time = datetime.now()
        
        def log_step(step_name):
            elapsed = datetime.now() - start_time
            processing_log.append(f"{step_name} ({elapsed.total_seconds():.2f}s)")
            print(f"TTS Processing Step: {step_name} - {elapsed.total_seconds():.2f}s")
        
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            
            # Retrieve relevant context with timeout protection
            log_step("Start RAG context retrieval")
            context = None
            
            with ThreadPoolExecutor() as executor:
                future = executor.submit(get_rag_context, query)
                try:
                    context = future.result(timeout=20)  # 20 second timeout for context retrieval
                except TimeoutError:
                    print("Context retrieval timed out")
                    context = "Context retrieval timed out. Proceeding with general knowledge."
            
            if not context:
                context = "No specific context available for this query."
            
            log_step("Retrieved context")
            
            # Generate text response with timeout
            with ThreadPoolExecutor() as executor:
                # Use a longer timeout if client can handle long responses
                response_timeout = 90 if handle_long_responses else 60
                
                log_step(f"Start response generation (timeout: {response_timeout}s)")
                future = executor.submit(generate_response, query, context, language)
                try:
                    response = future.result(timeout=response_timeout)
                except TimeoutError:
                    print("Response generation timed out")
                    timeout_responses = {
                        "en": "I'm sorry, but your request timed out. Please try a simpler question or try again later.",
                        "fr": "Je suis désolé, mais votre demande a expiré. Veuillez essayer une question plus simple ou réessayer plus tard.",
                        "ar": "آسف، لقد انتهت مهلة طلبك. يرجى تجربة سؤال أبسط أو المحاولة مرة أخرى لاحقًا."
                    }
                    response = timeout_responses.get(language, timeout_responses["en"])
            
            log_step("Response generated")
            
            # Clean and format the response for better TTS quality
            cleaned_response = post_process_response(response, language)
            print(f"Cleaned response for TTS: {cleaned_response[:100]}...")
            log_step("Response cleaned for TTS")
            
            # Always generate speech for this endpoint
            log_step("Start speech generation")
            
            # Use ThreadPoolExecutor to apply a timeout to speech generation too
            speech_file = None
            with ThreadPoolExecutor() as executor:
                future = executor.submit(generate_speech, cleaned_response, language)
                try:
                    speech_file = future.result(timeout=60)  # 60 second timeout for speech generation
                except TimeoutError:
                    print("Speech generation timed out")
                    # Create a shorter version of the response for speech
                    short_response = cleaned_response[:2000] + "... (Response truncated for audio)"
                    speech_file = generate_speech(short_response, language)
            
            speech_url = f"/audio/{os.path.basename(speech_file)}" if speech_file else None
            log_step("Speech generated")
            
            # Return both text and speech URL along with processing info for debugging
            return jsonify({
                "query": query,
                "response": response,
                "language": language,
                "speech_url": speech_url,
                "success": True,
                "processing_info": {
                    "steps": processing_log,
                    "total_time": (datetime.now() - start_time).total_seconds()
                }
            })
                
        except Exception as inner_e:
            log_step(f"Error: {str(inner_e)}")
            print(f"Error processing TTS question: {str(inner_e)}")
            
            error_responses = {
                "en": "I'm having trouble processing your question right now. Please try again later.",
                "fr": "J'ai du mal à traiter votre question en ce moment. Veuillez réessayer plus tard.",
                "ar": "أواجه صعوبة في معالجة سؤالك الآن. يرجى المحاولة مرة أخرى لاحقًا."
            }
            error_response = error_responses.get(language, error_responses["en"])
            
            # Clean the error response for TTS
            cleaned_error = post_process_response(error_response, language)
            
            # Generate speech for error message too
            try:
                speech_file = generate_speech(cleaned_error, language)
                speech_url = f"/audio/{os.path.basename(speech_file)}" if speech_file else None
            except:
                speech_url = None
            
            return jsonify({
                "query": query,
                "response": error_response,
                "language": language,
                "speech_url": speech_url,
                "success": False,
                "error": str(inner_e),
                "processing_info": {
                    "steps": processing_log,
                    "total_time": (datetime.now() - start_time).total_seconds()
                }
            }), 500
        
    finally:
        # This finally block closes the try block that was opened above
        pass

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Process uploaded audio for speech recognition and return AI response"""
    print("Received audio upload request")
    if 'audio' not in request.files:
        print("No audio file in request")
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    language = request.form.get('language', 'en')
    generate_speech_param = request.form.get('generate_speech', 'true').lower()
    generate_speech_bool = generate_speech_param in ['true', '1', 'yes']
    
    print(f"Audio file: {audio_file.filename}, Language: {language}")
    
    if audio_file.filename == '':
        print("Empty filename received")
        return jsonify({"error": "Empty audio file"}), 400
        
    # Check request content before saving
    if hasattr(audio_file, 'content_length') and audio_file.content_length == 0:
        print("Content length is zero")
        return jsonify({"error": "Empty audio file was uploaded (0 bytes)"}), 400
    
    temp_paths = []  # Keep track of all temporary files
    try:
        # Determine appropriate file extension based on content type or filename
        file_ext = "wav"  # Default
        content_type = audio_file.content_type.lower() if audio_file.content_type else ""
        original_filename = audio_file.filename.lower() if audio_file.filename else ""
        
        # Try to determine format from content type first
        if content_type:
            if "mp3" in content_type:
                file_ext = "mp3"
            elif "m4a" in content_type or "aac" in content_type:
                file_ext = "m4a"
            elif "ogg" in content_type or "opus" in content_type:
                file_ext = "ogg"
            elif "wav" in content_type or "audio/wave" in content_type:
                file_ext = "wav"
            print(f"Detected content type: {content_type}, using extension: {file_ext}")
        # If content type didn't work, try filename
        elif original_filename and "." in original_filename:
            detected_ext = original_filename.split(".")[-1].lower()
            if detected_ext in ["mp3", "wav", "m4a", "aac", "ogg", "opus"]:
                file_ext = detected_ext
                print(f"Using extension from filename: {file_ext}")
        
        # Make sure audio directory exists
        os.makedirs(AUDIO_PATH, exist_ok=True)
        
        # Save audio file with appropriate extension
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        temp_path = f"{AUDIO_PATH}/temp_{timestamp}.{file_ext}"
        temp_paths.append(temp_path)  # Add to list of files to clean up
        
        print(f"Saving uploaded file to: {temp_path}")
        try:
            # Read the entire file into memory first to validate it's not empty
            audio_data = audio_file.read()
            if not audio_data or len(audio_data) == 0:
                return jsonify({"error": "Empty audio file was uploaded (0 bytes)"}), 400
            
            # Write the file to disk
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            # Reset file pointer for potential future use
            audio_file.seek(0)
            
            # Check if file was saved correctly
            if not os.path.exists(temp_path):
                return jsonify({"error": "Failed to save audio file"}), 500
                
            file_size = os.path.getsize(temp_path)
            print(f"File saved successfully. Size: {file_size} bytes")
            if file_size == 0:
                return jsonify({"error": "Empty audio file was uploaded (0 bytes)"}), 400
        except Exception as save_error:
            print(f"Error saving audio file: {save_error}")
            return jsonify({"error": f"Failed to save audio file: {str(save_error)}"}), 500
        
        # If the file isn't already WAV, try to convert it to ensure compatibility
        if file_ext != "wav":
            try:
                wav_path = f"{AUDIO_PATH}/temp_{timestamp}_converted.wav"
                temp_paths.append(wav_path)  # Add to cleanup list
                
                print(f"Converting {file_ext} to WAV format...")
                # Convert using pydub
                audio = AudioSegment.from_file(temp_path)
                audio = audio.set_channels(1)  # Convert to mono
                audio = audio.set_frame_rate(16000)  # Set to 16kHz
                audio.export(wav_path, format="wav")
                
                # Use the converted file if successful
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    print(f"Conversion successful, using: {wav_path}")
                    temp_path = wav_path
                else:
                    print("Conversion produced empty file, using original")
            except Exception as conv_err:
                print(f"Conversion error (will use original file): {conv_err}")
                # Continue with original file if conversion fails
        
        # Transcribe audio
        print(f"Starting transcription of {temp_path}")
        transcription = transcribe_audio(temp_path, language)
        
        if not transcription or transcription.strip() == "":
            return jsonify({
                "status": "error",
                "error": "Could not transcribe audio. Please try again with a clearer recording."
            }), 400
        
        # Process the query using the transcription
        context = get_rag_context(transcription)
        response = generate_response(transcription, context, language)
        
        # Auto-detect language from the response if it was set to "auto"
        detected_language = language
        if language == "auto":
            # Basic language detection based on response
            if any(char in response for char in "éèêëàâîïôùûç"):
                detected_language = "fr"
            elif any("\u0600" <= char <= "\u06FF" for char in response):
                detected_language = "ar"
            else:
                detected_language = "en"
        
        # Generate speech if requested
        speech_url = None
        if generate_speech_bool:
            try:
                speech_file = generate_speech(response, detected_language)
                if speech_file:
                    speech_url = f"/audio/{os.path.basename(speech_file)}"
            except Exception as speech_error:
                print(f"Error generating speech: {speech_error}")
                # Continue processing even if speech generation fails
            
        result = {
            "status": "success",
            "transcription": transcription,
            "response": response,
            "language_detected": detected_language,
            "speech_url": speech_url
        }

        return jsonify(result)
    except Exception as e:
        error_msg = f"Error processing audio: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"status": "error", "error": error_msg}), 500
    finally:
        # Clean up all temp files
        for path in temp_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Cleaned up temporary file: {path}")
                except Exception as cleanup_err:
                    print(f"Error removing temp audio file {path}: {cleanup_err}")

@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio_route():
    """Transcribe audio file to text without generating a response"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    language = request.form.get('language', 'en')
    
    if audio_file.filename == '':
        return jsonify({"error": "Empty audio file"}), 400
    
    temp_paths = []  # Track all temporary files
    try:
        # Determine appropriate file extension based on content type or filename
        file_ext = "wav"  # Default
        content_type = audio_file.content_type.lower() if audio_file.content_type else ""
        original_filename = audio_file.filename.lower() if audio_file.filename else ""
        
        if content_type:
            if "mp3" in content_type:
                file_ext = "mp3"
            elif "m4a" in content_type or "aac" in content_type:
                file_ext = "m4a"
            elif "ogg" in content_type or "opus" in content_type:
                file_ext = "ogg"
            elif "wav" in content_type or "audio/wave" in content_type:
                file_ext = "wav"
            print(f"Detected content type: {content_type}, using extension: {file_ext}")
        elif original_filename and "." in original_filename:
            detected_ext = original_filename.split(".")[-1].lower()
            if detected_ext in ["mp3", "wav", "m4a", "aac", "ogg", "opus"]:
                file_ext = detected_ext
                print(f"Using extension from filename: {file_ext}")
        
        # Save audio file with appropriate extension
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        temp_path = f"{AUDIO_PATH}/temp_{timestamp}.{file_ext}"
        temp_paths.append(temp_path)
        
        print(f"Saving uploaded file to: {temp_path}")
        audio_file.save(temp_path)
        
        if not os.path.exists(temp_path):
            return jsonify({"error": "Failed to save audio file"}), 500
            
        file_size = os.path.getsize(temp_path)
        print(f"File saved successfully. Size: {file_size} bytes")
        if file_size == 0:
            return jsonify({"error": "Empty audio file was uploaded (0 bytes)"}), 400
        
        # If the file isn't already WAV, try to convert it to ensure compatibility
        if file_ext != "wav":
            try:
                wav_path = f"{AUDIO_PATH}/temp_{timestamp}_converted.wav"
                temp_paths.append(wav_path)
                
                print(f"Converting {file_ext} to WAV format...")
                audio = AudioSegment.from_file(temp_path)
                audio = audio.set_channels(1)  # Convert to mono
                audio = audio.set_frame_rate(16000)  # Set to 16kHz
                audio.export(wav_path, format="wav")
                
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    print(f"Conversion successful, using: {wav_path}")
                    temp_path = wav_path
                else:
                    print("Conversion produced empty file, using original")
            except Exception as conv_err:
                print(f"Conversion error (will use original file): {conv_err}")
        
        # Transcribe audio
        print(f"Starting transcription of {temp_path}")
        transcription = transcribe_audio(temp_path, language)
        
        return jsonify({
            "status": "success",
            "transcription": transcription,
            "language_detected": language
        })
        
    except Exception as e:
        error_msg = f"Error transcribing audio: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"status": "error", "error": error_msg}), 500
    finally:
        # Clean up all temp files
        for path in temp_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Cleaned up temporary file: {path}")
                except Exception as cleanup_err:
                    print(f"Error removing temp file {path}: {cleanup_err}")

@app.route('/add_document', methods=['POST'])
def add_document():
    """Add a document to the RAG knowledge base"""
    language = request.form.get('language', 'en')
    
    try:
        # Check if this is a text document or file upload
        if request.json:
            # Text document
            data = request.json
            new_doc = data['document']
            docs_to_add = [new_doc]
            doc_id = f"text_{int(time.time())}"
            chunks_extracted = 1
        elif 'file' in request.files:
            # File upload
            uploaded_file = request.files['file']
            filename = uploaded_file.filename
            
            if not filename:
                return jsonify({"status": "error", "message": "No file provided"}), 400
            
            # Create documents directory if it doesn't exist
            documents_path = "documents"
            if not os.path.exists(documents_path):
                os.makedirs(documents_path)
                
            # Save the file
            file_path = os.path.join(documents_path, filename)
            uploaded_file.save(file_path)
            
            # Process based on file type
            if filename.lower().endswith('.pdf') and PDF_SUPPORT:
                # Extract text from PDF
                docs_to_add, chunks_extracted = process_pdf(file_path)
                doc_id = f"pdf_{os.path.splitext(os.path.basename(filename))[0]}"
                
                # Record document metadata
                doc_metadata = {
                    'id': doc_id,
                    'filename': filename,
                    'path': file_path,
                    'type': 'pdf',
                    'chunks': chunks_extracted,
                    'added_at': datetime.now().isoformat(),
                    'size_kb': round(os.path.getsize(file_path) / 1024, 2)
                }
                
                # Save metadata to documents index
                save_document_metadata(doc_metadata)
            else:
                # Unsupported file type
                return jsonify({
                    "status": "error", 
                    "message": "Unsupported file format. Please upload a PDF file."
                }), 400
        else:
            return jsonify({
                "status": "error", 
                "message": "No document or file provided"
            }), 400
        
        # Add to vector store
        global vector_store
        if docs_to_add:
            vector_store.add_texts(docs_to_add)
            vector_store.save_local(VECTOR_DB_PATH)
        
        # Return confirmation
        if language == "fr":
            message = f"Document ajouté avec succès à la base de connaissances ({chunks_extracted} extraits)"
        elif language == "ar":
            message = f"تمت إضافة المستند بنجاح إلى قاعدة المعرفة ({chunks_extracted} مقتطفات)"
        else:
            message = f"Document successfully added to knowledge base ({chunks_extracted} chunks extracted)"
        
        return jsonify({
            "status": "success", 
            "message": message, 
            "document_id": doc_id,
            "chunks_extracted": chunks_extracted,
            "vector_db_updated": True
        })
        
    except Exception as e:
        print(f"Error adding document: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all documents in the knowledge base"""
    language = request.args.get('language', 'en')
    try:
        documents = get_document_metadata()
        
        # Translate response based on language
        if language == "fr":
            message = f"{len(documents)} documents trouvés dans la base de connaissances"
        elif language == "ar":
            message = f"{len(documents)} مستندات موجودة في قاعدة المعرفة"
        else:
            message = f"{len(documents)} documents found in knowledge base"
            
        return jsonify({
            "status": "success",
            "message": message,
            "documents": documents
        })
    except Exception as e:
        print(f"Error listing documents: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
        
@app.route('/products', methods=['GET'])
def get_products():
    """Get the list of all products with their current inventory levels and categories"""
    try:
        # Get unique products from supply chain data
        unique_products = supply_chain_data['product_id'].unique()
        
        # Prepare response with product details
        product_list = []
        
        for product_id in unique_products:
            # Filter data for this product
            product_data = supply_chain_data[supply_chain_data['product_id'] == product_id]
            
            # Calculate total current inventory across all locations
            total_inventory = product_data['inventory'].sum()
            
            # Determine product category based on product name
            if 'PHN' in product_id:
                category = 'Electronics - Phones'
            elif 'LAP' in product_id:
                category = 'Electronics - Computers'
            elif 'TAB' in product_id:
                category = 'Electronics - Tablets'
            elif 'CPU' in product_id:
                category = 'Processors'
            elif 'GPU' in product_id:
                category = 'Graphics Cards'
            else:
                category = 'Other Electronics'
            
            # Get average cost
            avg_cost = product_data['cost'].mean()
            
            # Get inventory by location
            inventory_by_location = {}
            for location in product_data['location'].unique():
                location_data = product_data[product_data['location'] == location]
                inventory_by_location[location] = int(location_data['inventory'].sum())
            
            # Add product to list
            product_list.append({
                'product_id': product_id,
                'category': category,
                'total_inventory': int(total_inventory),
                'avg_cost': round(float(avg_cost), 2),
                'inventory_by_location': inventory_by_location,
                'avg_lead_time': round(float(product_data['lead_time'].mean()), 1)
            })
        
        return jsonify({
            'products': product_list,
            'total_count': len(product_list)
        })
    except Exception as e:
        print(f"Error in /products endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def save_document_metadata(metadata):
    """Save document metadata to the index file"""
    index_path = "documents/index.json"
    docs = []
    
    # Load existing index if it exists
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                docs = json.load(f)
        except Exception as e:
            print(f"Error loading document index: {e}")
            docs = []
    
    # Add new document metadata
    docs.append(metadata)
    
    # Save updated index
    with open(index_path, 'w') as f:
        json.dump(docs, f, indent=2)

def get_document_metadata():
    """Get all document metadata from the index file"""
    index_path = "documents/index.json"
    
    if not os.path.exists(index_path):
        return []
        
    try:
        with open(index_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading document index: {e}")
        return []

def process_pdf(file_path):
    """Extract and process text from a PDF file"""
    if not PDF_SUPPORT:
        print(f"PDF support is not available. Cannot process {file_path}")
        return None
    
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(file_path)
        text_content = ""
        
        # Extract text from each page
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Add page number for reference
                text_content += f"[Page {i+1}] " + page_text + "\n\n"
        
        # Basic text cleaning
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        text_content = text_content.replace('- ', '')  # Remove hyphenation
        
        if not text_content.strip():
            print(f"  - No extractable text found in {file_path}")
            return None
        
        print(f"  - Extracted {len(text_content)} characters from PDF with {len(pdf_reader.pages)} pages")
        return text_content
        
        chunks = text_splitter.split_text(text_content)
        
        # Add document info as metadata to first chunk
        if chunks:
            info = pdf_reader.metadata
            if info:
                metadata_text = "Document Information:\n"
                if hasattr(info, 'title') and info.title:
                    metadata_text += f"Title: {info.title}\n"
                if hasattr(info, 'author') and info.author:
                    metadata_text += f"Author: {info.author}\n"
                if hasattr(info, 'subject') and info.subject:
                    metadata_text += f"Subject: {info.subject}\n"
                chunks[0] = metadata_text + "\n" + chunks[0]
        
        return chunks, len(chunks)
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        traceback.print_exc()
        raise

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
    
    # Initialize supply chain domain knowledge with core concepts
    supply_chain_docs = [
        """Supply Chain Management: The systematic coordination of traditional business functions within a 
        particular company and across businesses within the supply chain, for the purpose of improving the 
        long-term performance of the individual companies and the supply chain as a whole. Supply chain 
        management involves the integrated planning and execution of processes required to manage the 
        movement of materials, information, and financial capital in activities that broadly include demand 
        planning, sourcing, production, inventory management, and logistics.""",
        
        """Inventory Management Best Practices: Effective inventory management involves balancing the cost 
        of holding inventory against the risk of stockouts. Key metrics include Inventory Turnover Ratio, 
        Days Sales of Inventory (DSI), and Economic Order Quantity (EOQ). Advanced practices include ABC 
        analysis (classifying items by importance), Just-In-Time (JIT) inventory, and establishing safety 
        stock levels. Cycle counting provides ongoing accuracy verification instead of annual physical counts.""",
        
        """Demand Forecasting Techniques: Quantitative forecasting methods include time-series analysis 
        (moving averages, exponential smoothing, ARIMA models), regression analysis, and machine learning 
        techniques. Qualitative methods include Delphi method, market research, and expert opinion. Collaborative 
        forecasting involves sharing data with suppliers and customers to improve accuracy. Key metrics for 
        forecast accuracy include Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Forecast 
        Bias.""",
        
        """Logistics and Transportation: Logistics management optimizes the movement and storage of goods, 
        services, and information throughout the supply chain. Transportation modes include road, rail, air, 
        water, and pipeline, each with distinct cost, speed, and capacity characteristics. Network optimization 
        involves selecting optimal shipping lanes, transportation modes, and carrier selection. Key performance 
        indicators include on-time delivery, transportation cost per unit, and perfect order rate.""",
        
        """Supply Chain Risk Management: Supply chain risks can be categorized as disruption risks (natural 
        disasters, supplier bankruptcy), delay risks (port congestion, quality issues), systems risks (information 
        infrastructure breakdown), forecast risks (inaccurate projections), intellectual property risks, procurement 
        risks (currency fluctuations, price increases), receivables risks (customer insolvency), inventory risks 
        (obsolescence, shrinkage), and capacity risks (insufficient flexibility). Mitigation strategies include 
        multi-sourcing, local sourcing, buffer inventory, flexible manufacturing, and contingency planning.""",
        
        """Supplier Relationship Management: Strategic approaches to working with suppliers range from transactional 
        to collaborative partnerships. Supplier evaluation criteria include quality, delivery performance, cost 
        competitiveness, technical capability, and financial stability. Supplier development programs help improve 
        supplier capabilities through training, technical assistance, and process improvement support. Supplier 
        scorecards provide structured performance evaluation.""",
        
        """Lean Supply Chain: Lean principles focus on eliminating waste (muda) in the supply chain, including 
        overproduction, waiting time, unnecessary transport, over-processing, excess inventory, unnecessary movement, 
        and defects. Techniques include value stream mapping, 5S workplace organization, kanban pull systems, 
        continuous improvement (kaizen), and standardized work procedures. Lean metrics include inventory turns, 
        lead time, cycle time, and perfect order fulfillment."""
    ]
    
    # Add metadata to docs for better context retrieval
    docs_with_metadata = []
    for i, doc in enumerate(supply_chain_docs):
        metadata = {"source": f"Core SCM Knowledge {i+1}", "type": "foundational"}
        docs_with_metadata.append({"page_content": doc, "metadata": metadata})
    
    # Initialize vector store (ensure embeddings are defined before this)
    if os.path.exists(VECTOR_DB_PATH):
        try:
            vector_store = FAISS.load_local(
                VECTOR_DB_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True  # Only use if you trust the source
            )
            print(f"Successfully loaded existing vector store from {VECTOR_DB_PATH}")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Creating new vector store with supply chain domain knowledge...")
            vector_store = FAISS.from_documents(docs_with_metadata, embeddings)
            vector_store.save_local(VECTOR_DB_PATH)
            print(f"Created and saved new vector store with {len(docs_with_metadata)} documents")
    else:
        print("Vector store not found. Creating new vector store with supply chain domain knowledge...")
        vector_store = FAISS.from_documents(docs_with_metadata, embeddings)
        vector_store.save_local(VECTOR_DB_PATH)
        print(f"Created and saved new vector store with {len(docs_with_metadata)} documents")
    
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
    
    # Initialize and update knowledge base with document files
    print("\n=== Loading Supply Chain Documents ===")
    try:
        # Create documents directory if it doesn't exist
        if not os.path.exists("documents"):
            os.makedirs("documents")
            print("Created documents directory")
        
        # Count documents
        pdf_files = [f for f in os.listdir("documents") if f.lower().endswith('.pdf')]
        txt_files = [f for f in os.listdir("documents") if f.lower().endswith('.txt')]
        print(f"Found {len(pdf_files)} PDF files and {len(txt_files)} TXT files in documents directory")
        
        if len(pdf_files) + len(txt_files) > 0:
            print("Processing documents and updating knowledge base...")
            # Process all documents with our document processing function
            all_docs = load_and_process_documents()
            
            # Create documents with metadata
            docs_with_metadata = []
            for i, doc in enumerate(all_docs):
                source_name = f"Document {i+1}"
                if i < len(supply_chain_docs):  # Base knowledge
                    source_name = f"Core SCM Knowledge {i+1}"
                docs_with_metadata.append({"page_content": doc, "metadata": {"source": source_name}})
            
            # Update the vector store with new documents
            print(f"Updating vector store with {len(docs_with_metadata)} documents...")
            vector_store = FAISS.from_documents(docs_with_metadata, embeddings)
            vector_store.save_local(VECTOR_DB_PATH)
            print(f"Knowledge base updated successfully with {len(docs_with_metadata)} documents")
        else:
            print("No document files found. Using default supply chain knowledge.")
    except Exception as doc_error:
        print(f"Error processing documents: {doc_error}")
        print("Continuing with existing knowledge base...")
    
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