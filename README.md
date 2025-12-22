# 🎯 Churn Radar

**Production-Ready ML Platform for Customer Churn Prediction**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.18+-orange.svg)](https://mlflow.org/)
[![CI](https://github.com/yourname/churn-radar/actions/workflows/ci.yml/badge.svg)](https://github.com/yourname/churn-radar/actions)

## ✨ Features

- 🚀 **FastAPI** - High-performance REST API with auth & rate limiting
- 📊 **MLflow** - Model versioning and experiment tracking
- 🔄 **Airflow** - Orchestrated ML pipelines (7 DAGs)
- 📈 **Prometheus + Grafana** - Metrics and dashboards
- 🔍 **SHAP Explainability** - Understand prediction drivers
- ⚡ **Redis Feature Store** - Cached features with TTL
- 🧪 **A/B Testing** - Model comparison and gradual rollout
- 📱 **Notifications** - Slack/Telegram alerts
- 🔧 **Optuna HPO** - Hyperparameter optimization
- 📝 **Model Cards** - Auto-generated documentation
- 🔬 **DVC** - Data versioning and pipelines
- 🎨 **Rich CLI** - Beautiful terminal interface

## 🚀 Quick Start

```bash
git clone https://github.com/yourname/churn-radar.git
cd churn-radar

# Build and start all services
make dev

# Train on demo data (runs in Docker)
make train-demo

# API available at http://localhost:8000
# MLflow at http://localhost:5000
# Airflow at http://localhost:8080 (admin/admin)
```

All commands run inside Docker containers - no local Python setup needed!

## 📊 Data Options

### Demo Data (Synthetic)
```bash
make train-demo
```

### Real Data
Place CSV files in `data/raw/`:
```
data/raw/
├── 2019-Oct.csv
├── 2019-Nov.csv
└── ...
```

Dataset: [eCommerce Behavior Data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)

```bash
make train-real        # Full dataset
make data-sample       # 10% sample
```

## 🔧 Make Commands

```bash
# Setup & Services
make build          # Build Docker image
make dev            # Start all services (builds first)
make dev-down       # Stop services
make logs           # View logs

# Data (runs in Docker)
make data           # Auto-detect data source
make data-demo      # Generate synthetic data
make data-real      # Process real CSV data
make data-sample    # Process 10% sample

# Training (runs in Docker)
make train          # Train model
make train-demo     # Demo data + train
make train-real     # Real data + train
make hpo            # Hyperparameter optimization

# Development (runs in Docker)
make test           # Run tests
make lint           # Run linters
make shell          # Open bash in container
make clean          # Clean artifacts
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check with dependencies |
| `/healthz` | GET | Simple health check |
| `/predict` | POST | Single/batch prediction |
| `/explain` | POST | SHAP explanations |
| `/feature-importance` | GET | Global feature importance |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI |

## 🎯 How to Use the Model

### 1. Start Services
```bash
make dev
```

### 2. Train Model
```bash
# With demo data (quick start)
make train-demo

# With real data (place CSVs in data/raw/)
make train-real
```

### 3. Make Predictions

#### Single Customer
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{
    "user_id": 123,
    "clicks": 50,
    "purchases": 3,
    "avg_session_time": 12.5,
    "active_days": 30,
    "days_since_last_visit": 5,
    "account_age_days": 180,
    "avg_order_value": 75.0,
    "total_spend": 225.0,
    "n_unique_categories": 5,
    "n_unique_brands": 8
  }'
```

**Response:**
```json
{
  "prediction": {
    "user_id": 123,
    "churn_proba": 0.73,
    "churn_label": 1,
    "model_version": "1"
  }
}
```

#### Batch Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{
    "customers": [
      {"user_id": 1, "clicks": 50, "purchases": 3, ...},
      {"user_id": 2, "clicks": 10, "purchases": 0, ...},
      {"user_id": 3, "clicks": 100, "purchases": 10, ...}
    ]
  }'
```

### 4. Get Explanations (SHAP)
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{
    "user_id": 123,
    "clicks": 50,
    "purchases": 3,
    "days_since_last_visit": 15
  }'
```

**Response:**
```json
{
  "prediction": {"churn_proba": 0.73, "churn_label": 1},
  "explanation": {
    "top_features": [
      {"feature": "days_since_last_visit", "impact": "increases", "shap_value": 0.15},
      {"feature": "purchases", "impact": "decreases", "shap_value": -0.08}
    ],
    "text": "High churn risk. Main drivers: long time since last visit (+15%), low purchases (-8%)"
  }
}
```

### 5. View in MLflow
Open http://localhost:5000 to see:
- Experiments and runs
- Model metrics (ROC-AUC, F1, etc.)
- Feature importance
- Registered models

### 6. Python SDK Usage
```python
import requests

API_URL = "http://localhost:8000"
HEADERS = {"X-API-Key": "dev-api-key", "Content-Type": "application/json"}

# Single prediction
response = requests.post(
    f"{API_URL}/predict",
    headers=HEADERS,
    json={
        "user_id": 123,
        "clicks": 50,
        "purchases": 3,
        "days_since_last_visit": 5,
        # ... other features
    }
)
result = response.json()
print(f"Churn probability: {result['prediction']['churn_proba']:.1%}")

# Batch prediction
customers = [{"user_id": i, "clicks": 50, "purchases": 3} for i in range(100)]
response = requests.post(
    f"{API_URL}/predict",
    headers=HEADERS,
    json={"customers": customers}
)
results = response.json()["predictions"]
```

### Feature Columns Reference

| Feature | Description |
|---------|-------------|
| `clicks` | Total click events |
| `purchases` | Total purchase events |
| `avg_session_time` | Average session duration (minutes) |
| `active_days` | Days with activity |
| `days_since_last_visit` | Days since last activity |
| `account_age_days` | Account age in days |
| `avg_order_value` | Average order value |
| `total_spend` | Total spending amount |
| `n_unique_categories` | Unique product categories viewed |
| `n_unique_brands` | Unique brands viewed |

## 📦 Project Structure

```
churn-radar/
├── src/
│   ├── api/           # FastAPI application
│   ├── data/          # Data processing & validation
│   ├── features/      # Feature engineering & store
│   ├── models/        # Training, HPO, explainer, A/B testing
│   ├── common/        # Logging & notifications
│   ├── monitoring/    # Prometheus metrics
│   └── cli.py         # Rich CLI
├── dags/              # Airflow DAGs (7 pipelines)
├── tests/             # pytest tests
├── docs/              # MkDocs documentation
├── grafana/           # Dashboard JSON
├── .github/workflows/ # CI/CD pipelines
├── dvc.yaml           # DVC pipeline
└── mkdocs.yml         # Documentation config
```

## 🔄 Airflow DAGs

| DAG | Schedule | Description |
|-----|----------|-------------|
| `etl_pipeline` | Daily | Demo data processing |
| `etl_real_data` | Manual | Real data processing |
| `training_pipeline` | Weekly | Model training |
| `batch_inference` | Daily | Batch predictions |
| `continuous_training` | Weekly | Full pipeline |
| `train_on_real_data` | Manual | Real data + training |
| `hyperparameter_optimization` | Manual | Optuna HPO |

## 📈 Monitoring

### Grafana Dashboard
Import `grafana/dashboard.json` for:
- Request rate & latency
- Prediction distribution
- Churn rate over time
- Error rates

### Prometheus Metrics
- `churn_api_requests_total`
- `churn_api_request_latency_seconds`
- `churn_predictions_total`
- `churn_prediction_probability`

## 🧪 Development

```bash
# Install dev dependencies
make install-dev

# Pre-commit hooks
make pre-commit

# Run tests
make test
make test-cov

# Linting & formatting
make lint
make format

# Documentation
make docs
```

## 🔬 DVC Pipeline

```bash
# Initialize DVC
make dvc-init

# Run full pipeline
make dvc-repro

# Push data to remote
make dvc-push
```

## 📱 Notifications

Configure in `.env`:
```bash
# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/...

# Telegram
TELEGRAM_BOT_TOKEN=your-token
TELEGRAM_CHAT_ID=your-chat-id
```

Events: Training complete, model promoted, pipeline failed, drift detected.

## 🧪 A/B Testing

```python
from src.models.ab_testing import get_ab_tester, Experiment

tester = get_ab_tester()
tester.register_experiment(Experiment(
    name="model_v2_rollout",
    control_model="v1",
    treatment_model="v2",
    traffic_split=0.1,  # 10% to new model
))

# Get variant for user
result = tester.get_variant("model_v2_rollout", user_id=123)
```

## 📝 Model Card

Auto-generated model documentation:
```bash
# After training, find in:
models/MODEL_CARD.md
models/model_card.json
```

## 🛠️ Tech Stack

- **API**: FastAPI, Pydantic, uvicorn
- **ML**: scikit-learn, LightGBM, XGBoost, SHAP, Optuna
- **MLOps**: MLflow, Airflow, DVC
- **Monitoring**: Prometheus, Grafana, structlog
- **Infra**: Docker, Redis
- **CLI**: Rich, Click
- **Docs**: MkDocs Material
- **CI/CD**: GitHub Actions

## 📄 License

MIT
