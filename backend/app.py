"""PhishGuard AI – FastAPI backend."""
import os, json, logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
from scipy.sparse import hstack
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"

# --- Global state ---
models = {}

def load_models():
    try:
        models["tfidf_word"] = joblib.load(MODELS_DIR / "tfidf_word.joblib")
        models["tfidf_char"] = joblib.load(MODELS_DIR / "tfidf_char.joblib")
        models["lr"] = joblib.load(MODELS_DIR / "lr_model.joblib")
        models["cal_lr"] = joblib.load(MODELS_DIR / "cal_lr_model.joblib")
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error("Failed to load models: %s", e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    models.clear()

# Pre-load models on import for reliability across test runners
load_models()

app = FastAPI(title="PhishGuard AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from normalizer import normalize_text
from indicators import detect_indicators
from explainer import get_explanation

# --- Schemas ---
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)

class AnalyzeResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    calibrated_probability: float
    risk_category: str
    indicators: list
    warning: str = "This is an automated assessment, not a guarantee of safety."

class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)

# --- Helpers ---
def _predict(text: str) -> dict:
    if not models:
        raise HTTPException(503, "Models not loaded. Run train_model.py first.")
    normalized = normalize_text(text)
    X_w = models["tfidf_word"].transform([normalized])
    X_c = models["tfidf_char"].transform([normalized])
    X = hstack([X_w, X_c])
    pred = int(models["lr"].predict(X)[0])
    prob_uncal = float(models["lr"].predict_proba(X)[0][1])
    prob_cal = float(models["cal_lr"].predict_proba(X)[0][1])
    return {"prediction": pred, "probability": prob_uncal, "calibrated_probability": prob_cal}

def _risk_category(prob: float) -> str:
    if prob > 0.85: return "high"
    if prob > 0.5: return "medium"
    if prob > 0.3: return "uncertain"
    return "low"

# --- Routes ---
@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": bool(models)}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Text cannot be empty.")
    result = _predict(text)
    inds = detect_indicators(text)
    risk = _risk_category(result["calibrated_probability"])
    return AnalyzeResponse(
        prediction=result["prediction"],
        label="Phishing/Spam" if result["prediction"] == 1 else "Legitimate",
        probability=round(result["probability"], 4),
        calibrated_probability=round(result["calibrated_probability"], 4),
        risk_category=risk,
        indicators=inds,
    )

@app.post("/explain")
async def explain(req: ExplainRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Text cannot be empty.")
    result = _predict(text)
    inds = detect_indicators(text)
    evidence = {
        "prediction": result["prediction"],
        "probability": round(result["calibrated_probability"], 4),
        "indicators": inds,
        "limits": "ML model is a text classifier. Indicators are heuristic signals. No external lookups."
    }
    explanation = get_explanation(evidence)

    # Check for ML vs explanation disagreement
    summary_lower = explanation.get("summary", "").lower()
    if result["prediction"] == 1 and "legitimate" in summary_lower and "phishing" not in summary_lower:
        explanation["disagreement"] = "The explanation suggests legitimate but the ML model predicts phishing."
    elif result["prediction"] == 0 and "phishing" in summary_lower and "legitimate" not in summary_lower:
        explanation["disagreement"] = "The explanation suggests phishing but the ML model predicts legitimate."

    return {**evidence, "explanation": explanation}

@app.get("/metrics")
async def metrics():
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "Metrics not found. Run train_model.py first.")
    return json.loads(path.read_text())

@app.get("/metrics/external")
async def metrics_external():
    path = RESULTS_DIR / "metrics_external.json"
    if not path.exists():
        raise HTTPException(404, "External metrics not found. Run train_external_model.py first.")
    return json.loads(path.read_text())

@app.get("/metrics/cross-dataset")
async def metrics_cross_dataset():
    path = RESULTS_DIR / "cross_dataset_evaluation.json"
    if not path.exists():
        raise HTTPException(404, "Cross-dataset metrics not found. Run train_external_model.py first.")
    return json.loads(path.read_text())

@app.get("/robustness")
async def robustness():
    path = RESULTS_DIR / "robustness.json"
    if not path.exists():
        raise HTTPException(404, "Robustness results not found. Run run_experiments.py first.")
    return json.loads(path.read_text())

@app.get("/calibration")
async def calibration():
    path = RESULTS_DIR / "calibration.json"
    if not path.exists():
        raise HTTPException(404, "Calibration results not found. Run train_model.py first.")
    return json.loads(path.read_text())
