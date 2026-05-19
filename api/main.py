from fastapi import FastAPI
import joblib
import os
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime

app = FastAPI(title="ML Sentiment API")

# Load Model A
base = os.path.expanduser("~/ml-pipeline-monitor")
model_a = joblib.load(f"{base}/models/model_a_logreg.pkl")
vectorizer = joblib.load(f"{base}/models/vectorizer.pkl")

# Database connection
engine = create_engine("postgresql://localhost/mlpipeline")

# Create predictions table on startup
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            input_text TEXT,
            sentiment VARCHAR(10),
            confidence FLOAT,
            model_version VARCHAR(50),
            timestamp TIMESTAMP
        )
    """))
    conn.commit()

@app.get("/")
def root():
    return {"message": "ML Sentiment API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0",
        "models": ["logistic_regression", "distilbert"]
    }

@app.post("/predict")
def predict(input: str):
    # Vectorize + predict
    X = vectorizer.transform([input])
    prediction = model_a.predict(X)[0]
    confidence = model_a.predict_proba(X)[0].max()
    label = "positive" if prediction == 1 else "negative"

    # Save to database
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO predictions (input_text, sentiment, confidence, model_version, timestamp)
            VALUES (:text, :sentiment, :confidence, :model, :timestamp)
        """), {
            "text": input,
            "sentiment": label,
            "confidence": float(confidence),
            "model": "logistic_regression_v1",
            "timestamp": datetime.utcnow()
        })
        conn.commit()

    return {
        "text": input,
        "sentiment": label,
        "confidence": round(float(confidence), 4),
        "model": "logistic_regression_v1"
    }