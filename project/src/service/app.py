"""FastAPI service for the Churn prediction model."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.service.schemas import CustomerFeatures, HealthResponse, PredictionResponse

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("churn-service")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "artifacts" / "model.pkl"))
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run `python -m src.models.train` first."
        )
    log.info("Loading model from %s", MODEL_PATH)
    _model = joblib.load(MODEL_PATH)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
        log.info("Model loaded at startup. Threshold = %.2f", THRESHOLD)
    except FileNotFoundError as exc:
        log.warning("Model not loaded at startup: %s", exc)
    yield


app = FastAPI(
    title="Telco Customer Churn API",
    description=(
        "Сервис прогноза оттока клиентов на основе модели, обученной на "
        "открытом датасете Telco Customer Churn (IBM)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    try:
        model = _load_model()
    except FileNotFoundError as exc:
        log.error("Model not available: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    df = pd.DataFrame([features.model_dump()])
    try:
        proba = float(model.predict_proba(df)[0, 1])
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    churn_class = "Yes" if proba >= THRESHOLD else "No"
    log.info(
        "predict: tenure=%d Contract=%s -> proba=%.4f class=%s",
        features.tenure,
        features.Contract,
        proba,
        churn_class,
    )
    return PredictionResponse(
        churn_probability=proba, churn_class=churn_class, threshold=THRESHOLD
    )


@app.get("/")
def root():
    return {
        "service": "Telco Customer Churn API",
        "endpoints": ["/health", "/predict", "/docs"],
    }
