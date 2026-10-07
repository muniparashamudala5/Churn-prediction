import sys
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException  # type: ignore[import-not-found]

sys.path.insert(0, ".")
from app.schema import CustomerFeatures, PredictionResponse  # noqa: E402

MODEL_PATH = "models/churn_pipeline.joblib"
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        model = None
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running. Visit /docs for the interactive API."}

def risk_bucket(prob: float) -> str:
    if prob < 0.33:
        return "low"
    if prob < 0.66:
        return "medium"
    return "high"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train it first with `python src/train.py`.",
        )
    row = pd.DataFrame([features.model_dump()])
    prob = float(model.predict_proba(row)[0, 1])
    return PredictionResponse(
        churn=prob >= 0.5,
        churn_probability=round(prob, 4),
        risk_level=risk_bucket(prob),
    )
