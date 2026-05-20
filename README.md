# Real-Time ML Pipeline with Monitoring

A production-grade machine learning system for sentiment analysis featuring A/B testing, real-time monitoring, data drift detection, and automated model retraining.

## Overview

This system deploys two sentiment analysis models behind a FastAPI service, routes traffic between them for A/B testing, logs every prediction to PostgreSQL, and monitors model health via Prometheus and Grafana.

## Architecture

```
Request → FastAPI → A/B Router (50/50)
                    ├── Model A: Logistic Regression (81.4%)
                    └── Model B: DistilBERT (91.3%)
                         ↓
                    PostgreSQL (predictions log)
                         ↓
                    Prometheus (metrics scraping)
                         ↓
                    Grafana (dashboard)
```

## Features

- **A/B Testing** — 50/50 traffic split between logistic regression and DistilBERT
- **PostgreSQL Logging** — every prediction logged with confidence, model version, timestamp
- **Prometheus Metrics** — request count, latency, prediction distribution, confidence scores
- **Grafana Dashboard** — real-time visualization of model performance
- **Drift Detection** — monitors input distribution and confidence trends, alerts on anomalies
- **Auto-Retraining** — automatically retrains Model A when accuracy drops below threshold
- **Docker** — fully containerized for consistent deployment

## Models

| Model | Accuracy | F1 | Avg Confidence | Training Time |
|-------|----------|----|----------------|---------------|
| Logistic Regression + TF-IDF | 81.4% | 0.81 | 89.7% | <10 seconds |
| DistilBERT (fine-tuned) | 91.3% | 0.91 | 99.7% | ~23 min (GPU) |

Dataset: Stanford SST-2 (67,349 training sentences, binary sentiment)

## Performance

- **Throughput:** 99 requests/second
- **Average latency:** 10.1ms per request
- **Total predictions logged:** 1,200+
- **A/B split:** 614 DistilBERT / 604 Logistic Regression (across test run)

## Tech Stack

Python · FastAPI · PostgreSQL · Prometheus · Grafana · Docker · scikit-learn · Transformers (HuggingFace)

## Setup

**1. Clone and install:**
```bash
git clone https://github.com/manaswiiii/ml-pipeline-monitor.git
cd ml-pipeline-monitor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Start PostgreSQL and create database:**
```bash
brew services start postgresql@15
psql postgres -c "CREATE DATABASE mlpipeline;"
```

**3. Add models to models/ directory** (not included in repo due to size):
- `models/model_a_logreg.pkl`
- `models/vectorizer.pkl`
- `models/model_b_distilbert/`

**4. Start the API:**
```bash
uvicorn api.main:app --reload
```

**5. Start monitoring stack:**
```bash
docker run -d --name prometheus -p 9090:9090 \
  -v ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

docker run -d --name grafana -p 3000:3000 grafana/grafana
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Model status and version info |
| `/predict?input=<text>` | POST | Sentiment prediction with confidence |
| `/metrics` | GET | Prometheus metrics |

## Monitoring

- **Prometheus:** `http://localhost:9090`
- **Grafana:** `http://localhost:3000` (admin/admin)
- **API Docs:** `http://localhost:8000/docs`

## Drift Detection

```bash
python monitoring/drift_detection.py
```

Monitors average confidence and input length distribution. Alerts when confidence drops below 0.7 or >30% of predictions are low confidence.

## Auto-Retraining

```bash
python scripts/retrain.py
```

Checks current model accuracy against threshold (78%). If below, automatically retrains on the full training set and updates the saved model.