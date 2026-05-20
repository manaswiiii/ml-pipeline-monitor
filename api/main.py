from fastapi import FastAPI
import joblib
import os
import random
from sqlalchemy import create_engine, text
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = FastAPI(title="ML Sentiment API")

base = os.path.expanduser("~/ml-pipeline-monitor")

# Load Model A (Logistic Regression)
model_a = joblib.load(f"{base}/models/model_a_logreg.pkl")
vectorizer = joblib.load(f"{base}/models/vectorizer.pkl")

# Load Model B (DistilBERT)
tokenizer = DistilBertTokenizer.from_pretrained(f"{base}/models/model_b_distilbert")
model_b = DistilBertForSequenceClassification.from_pretrained(f"{base}/models/model_b_distilbert")
model_b.eval()

# Database
import os
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/mlpipeline")
engine = create_engine(DATABASE_URL)

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
        "models": ["logistic_regression_v1", "distilbert_v1"]
    }

@app.post("/predict")
def predict(input: str):
    # A/B test: 50/50 split
    use_model_b = random.random() < 0.5

    if use_model_b:
        inputs = tokenizer(input, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model_b(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = probs.argmax().item()
        confidence = probs.max().item()
        model_version = "distilbert_v1"
    else:
        X = vectorizer.transform([input])
        prediction = model_a.predict(X)[0]
        confidence = model_a.predict_proba(X)[0].max()
        model_version = "logistic_regression_v1"

    label = "positive" if prediction == 1 else "negative"

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO predictions (input_text, sentiment, confidence, model_version, timestamp)
            VALUES (:text, :sentiment, :confidence, :model, :timestamp)
        """), {
            "text": input,
            "sentiment": label,
            "confidence": float(confidence),
            "model": model_version,
            "timestamp": datetime.utcnow()
        })
        conn.commit()

    return {
        "text": input,
        "sentiment": label,
        "confidence": round(float(confidence), 4),
        "model": model_version
    }