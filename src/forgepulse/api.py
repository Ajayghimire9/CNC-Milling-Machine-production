from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

from .model import ManufacturingModels
from .schema import HealthResponse, MachineRecord, PredictionResponse

MODEL_PATH = Path(os.getenv("FORGEPULSE_MODEL", "artifacts/forgepulse.joblib"))
PREDICTIONS = Counter("forgepulse_predictions_total", "Prediction requests")
LATENCY = Histogram("forgepulse_prediction_latency_seconds", "Prediction latency")

app = FastAPI(title="ForgePulse Industrial ML API", version="3.0.0")
app.mount("/metrics", make_asgi_app())
_models: ManufacturingModels | None = None


def get_models() -> ManufacturingModels:
    global _models
    if _models is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model artifact is not available")
        _models = ManufacturingModels.load(MODEL_PATH)
    return _models


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if MODEL_PATH.exists() else "degraded",
        model_version="3.0.0",
        checks={"model_artifact": MODEL_PATH.exists()},
    )


@app.get("/ready")
def ready() -> dict[str, bool]:
    return {"ready": MODEL_PATH.exists()}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(record: MachineRecord) -> PredictionResponse:
    started = time.perf_counter()
    model = get_models()
    PREDICTIONS.inc()
    frame = pd.DataFrame([record.model_dump()])
    prediction, uncertainty, score, flag = model.predict_with_uncertainty(frame)
    risk = "high" if bool(flag.iloc[0]) else "medium" if float(score.iloc[0]) < 0.1 else "low"
    LATENCY.observe(time.perf_counter() - started)
    return PredictionResponse(
        processing_time=float(prediction.iloc[0]["processing_time"]),
        processing_time_std=float(uncertainty.iloc[0]["processing_time_std"]),
        average_power_consumption=float(prediction.iloc[0]["average_power_consumption"]),
        average_power_consumption_std=float(uncertainty.iloc[0]["average_power_consumption_std"]),
        anomaly_score=float(score.iloc[0]),
        anomaly=bool(flag.iloc[0]),
        risk_level=risk,
        model_version=model.version,
    )
