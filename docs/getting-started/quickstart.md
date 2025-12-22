# Quick Start

Get up and running with Churn Radar in 5 minutes.

## 1. Setup

```bash
# Clone and install
git clone https://github.com/yourname/churn-radar.git
cd churn-radar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start services
docker compose up -d
```

## 2. Prepare Data

### Option A: Demo Data (fastest)

```bash
make data-demo
```

### Option B: Real Data

Place CSV files in `data/raw/`:

```
data/raw/
├── 2019-Oct.csv
├── 2019-Nov.csv
└── ...
```

Then:

```bash
make data-real
```

## 3. Train Model

```bash
make train
```

This will:

1. Load and validate data
2. Engineer features
3. Train LightGBM model
4. Log to MLflow
5. Save model artifacts

## 4. Start API

```bash
make api
```

API available at http://localhost:8000

## 5. Make Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{
    "user_id": 123,
    "clicks": 50,
    "purchases": 3,
    "days_since_last_visit": 5
  }'
```

Response:

```json
{
  "prediction": {
    "user_id": 123,
    "churn_proba": 0.2341,
    "churn_label": 0,
    "model_version": "1"
  }
}
```

## 6. Get Explanations

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{
    "user_id": 123,
    "clicks": 50,
    "purchases": 3,
    "days_since_last_visit": 5
  }'
```

## Next Steps

- [API Documentation](../user-guide/api.md)
- [Training Guide](../user-guide/training.md)
- [Configuration](configuration.md)
