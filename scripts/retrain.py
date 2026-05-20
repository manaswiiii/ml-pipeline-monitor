import psycopg2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import json
from datetime import datetime

base = os.path.expanduser("~/ml-pipeline-monitor")
ACCURACY_THRESHOLD = 0.78

def get_predictions():
    conn = psycopg2.connect("dbname=mlpipeline")
    df = pd.read_sql("SELECT * FROM predictions WHERE model_version='logistic_regression_v1'", conn)
    conn.close()
    return df

def check_accuracy():
    # Load current metrics
    with open(f"{base}/metrics/model_a_metrics.json") as f:
        metrics = json.load(f)
    return metrics["accuracy"]

def retrain():
    print("Loading training data...")
    train = pd.read_csv(f"{base}/data/processed/train.csv")
    val = pd.read_csv(f"{base}/data/processed/val.csv")

    print("Training new model...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train['sentence'])
    X_val = vectorizer.transform(val['sentence'])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train['label'])

    preds = model.predict(X_val)
    new_accuracy = accuracy_score(val['label'], preds)
    print(f"New model accuracy: {new_accuracy:.4f}")

    # Save new model
    joblib.dump(model, f"{base}/models/model_a_logreg.pkl")
    joblib.dump(vectorizer, f"{base}/models/vectorizer.pkl")

    # Update metrics
    metrics = {
        "model": "logistic_regression",
        "accuracy": new_accuracy,
        "retrained_at": datetime.utcnow().isoformat()
    }
    with open(f"{base}/metrics/model_a_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model retrained and saved!")
    return new_accuracy

def main():
    print("=== Auto-Retraining Check ===")
    current_accuracy = check_accuracy()
    print(f"Current accuracy: {current_accuracy:.4f}")
    print(f"Threshold: {ACCURACY_THRESHOLD:.4f}")

    if current_accuracy < ACCURACY_THRESHOLD:
        print(f"Accuracy below threshold! Retraining...")
        new_accuracy = retrain()
        print(f"Retraining complete. New accuracy: {new_accuracy:.4f}")
    else:
        print(f"Accuracy is fine. No retraining needed.")

if __name__ == "__main__":
    main()