"""
AI Voice Detection API Service
------------------------------
This module provides a RESTful API for detecting whether an audio file 
is human-generated or synthesized by AI. It utilizes Librosa for 
digital signal processing and a pre-trained Scikit-Learn/Joblib model.
"""

import base64
import io
import os
import tempfile
import re
import joblib
import numpy as np
import librosa
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict

# --- Configuration & Environment Setup ---
load_dotenv()
API_KEY = os.getenv("API_KEY", "")

# Audio processing constants
SAMPLE_RATE = 16000
SUPPORTED_LANGUAGES = ["English", "Hindi", "Malayalam", "Telugu", "Tamil"]
MODEL_PATH = "artifacts/models/best_ml_model.joblib"

# --- Model Initialization ---
try:
    # Load the serialized machine learning model (e.g., Random Forest, SVM, or XGBoost)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Critical Error: Failed to load model at {MODEL_PATH}. {e}")
    model = None

# Initialize FastAPI application
app = FastAPI(
    title="AI Voice Detection API",
    description="API for classifying audio as Human or AI-Generated",
    version="1.0.0"
)

# --- Schemas ---

class VoiceRequest(BaseModel):
    """Schema for incoming voice detection requests."""
    language: str
    audioFormat: str
    audioBase64: str

# --- Helper Functions ---

def check_api_key(api_key: str):
    """Validates the request against the environment API Key."""
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts acoustic features from raw audio signal.
    
    Calculates:
    - Mel-frequency Spectrogram (Mean & Std)
    - Mel-frequency Cepstral Coefficients (MFCCs)
    """
    # Mel Spectrogram features
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)
    mel_std = np.std(mel_db, axis=1)
    
    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Combine all features into a single vector
    return np.concatenate([mel_mean, mel_std, mfcc_mean, mfcc_std])

def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    """
    Handles file cleanup, ID3 tag stripping, and normalization.
    Returns a reshaped feature vector ready for model prediction.
    """
    try:
        # Step 1: Strip ID3 tags (Metadata)
        # Some MP3 files contain metadata at the start which can crash certain decoders
        if audio_bytes.startswith(b'ID3'):
            if len(audio_bytes) >= 10:
                # Calculate size from synchsafe integers in bytes 6-9
                size = ((audio_bytes[6] & 0x7f) << 21) | \
                       ((audio_bytes[7] & 0x7f) << 14) | \
                       ((audio_bytes[8] & 0x7f) << 7) | \
                       (audio_bytes[9] & 0x7f)
                audio_bytes = audio_bytes[10 + size:]
        
        # Step 2: Write to temporary file for librosa processing
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            # Step 3: Load and Resample
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            
            # Step 4: Normalize Volume
            y = librosa.util.normalize(y)
            
            # Step 5: Feature Engineering
            features = extract_features(y, SAMPLE_RATE)
            return features.reshape(1, -1)
            
        finally:
            # Ensure file is deleted even if processing fails
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or Corrupted Audio File")

def explain_prediction(pred: int, proba: float) -> str:
    """Generates a human-readable explanation based on confidence scores."""
    if pred == 0:  # AI_GENERATED
        if proba > 0.9:
            return "Strong AI Patterns detected: Unnatural Pitch Consistency and Robotic Speech Characteristics"
        elif proba > 0.7:
            return "Moderate AI Patterns detected: Synthetic Voice Artifacts and Irregular Prosody"
        return "Weak AI Patterns detected: Some Unnatural Speech Characteristics present"
    else:  # HUMAN
        if proba > 0.9:
            return "Strong Human Voice Patterns detected: Natural Speech Variations and Organic Vocal Characteristics"
        elif proba > 0.7:
            return "Moderate Human Voice Patterns detected: Typical Human Speech Dynamics detected"
        return "Weak Human Voice Patterns detected: Some Natural Speech Characteristics present"

# --- Exception Handlers ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Standardizes error responses to JSON format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )

# --- Endpoints ---

@app.get("/")
def root():
    """Service Discovery Endpoint."""
    return {
        "service": "AI Voice Detection API",
        "status": "running",
        "endpoints": ["/health", "/api/voice-detection"]
    }

@app.get("/health")
def health_check():
    """System Health and Readiness Probe."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "supported_languages": SUPPORTED_LANGUAGES
    }

@app.post("/api/voice-detection")
def detect_voice(req: VoiceRequest, x_api_key: str = Header(...)):
    """
    Primary Detection Endpoint.
    
    1. Validates API Key & Input parameters
    2. Decodes Base64 audio
    3. Extracts features and runs Inference
    4. Returns classification and confidence score
    """
    check_api_key(x_api_key)
    
    # Request Validation
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language. Choose from: {SUPPORTED_LANGUAGES}")
    
    if req.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format is supported")
    
    # Base64 Decoding
    clean_b64 = re.sub(r"\s+", "", req.audioBase64)
    try:
        audio_bytes = base64.b64decode(clean_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 encoding")
    
    # Audio Processing & ML Inference
    features = preprocess_audio(audio_bytes)
    pred = model.predict(features)[0]

    # Calculate Confidence
    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(features)[0]
        proba = float(proba_all[pred])
    else:
        proba = 0.5 
    
    classification = "HUMAN" if pred == 1 else "AI_GENERATED"
    
    return {
        "status": "success",
        "language": req.language,
        "classification": classification,
        "confidenceScore": round(proba, 5),
        "explanation": explain_prediction(pred, proba)
    }