from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, make_asgi_app

from .model import ManufacturingModels
from .schema import HealthResponse, MachineRecord, PredictionResponse

MODEL_PATH = Path(os.getenv("FORGEPULSE_MODEL", "artifacts/forgepulse.joblib"))
PREDICTIONS = Counter("forgepulse_predictions_total", "Prediction requests")

app = FastAPI(title="ForgePulse API", version="2.0.0")
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
        model_version="2.0.0",
        checks={"model_artifact": MODEL_PATH.exists()},
    )


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(record: MachineRecord) -> PredictionResponse:
    model = get_models()
    PREDICTIONS.inc()
    frame = pd.DataFrame([record.model_dump()])
    prediction, score, flag = model.predict(frame)
    return PredictionResponse(
        processing_time=float(prediction.iloc[0]["processing_time"]),
        average_power_consumption=float(prediction.iloc[0]["average_power_consumption"]),
        anomaly_score=float(score.iloc[0]),
        anomaly=bool(flag.iloc[0]),
        model_version=model.version,
    )
