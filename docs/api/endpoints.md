# API Endpoints

## Authentication

All endpoints (except `/health`) require API key authentication:

```bash
-H "X-API-Key: your-api-key"
```

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/healthz` | GET | Liveness probe |
| `/metrics` | GET | Prometheus metrics |
| `/predict` | POST | Make predictions |
| `/explain` | POST | Get SHAP explanations |
| `/feature-importance` | GET | Global feature importance |

---

## Health Check

### `GET /health`

Returns service health and dependency status.

**Response:**

```json
{
  "status": "ok",
  "details": {
    "uptime_sec": 123.4,
    "dependencies": {
      "mlflow": {"status": "ok"},
      "model": {"status": "ok"},
      "redis": {"status": "ok"}
    }
  }
}
```

---

## Predictions

### `POST /predict`

Make churn predictions.

#### Single Prediction

**Request:**

```json
{
  "user_id": 123,
  "clicks": 50,
  "purchases": 3,
  "avg_session_time": 120.5,
  "active_days": 15,
  "days_since_last_visit": 5,
  "account_age_days": 180,
  "avg_order_value": 45.0,
  "total_spend": 450.0,
  "n_unique_categories": 5,
  "n_unique_brands": 8
}
```

**Response:**

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

#### Batch Prediction

**Request:**

```json
{
  "customers": [
    {"user_id": 1, "clicks": 50, "purchases": 3, ...},
    {"user_id": 2, "clicks": 10, "purchases": 0, ...}
  ]
}
```

**Response:**

```json
{
  "predictions": [
    {"user_id": 1, "churn_proba": 0.23, "churn_label": 0, "model_version": "1"},
    {"user_id": 2, "churn_proba": 0.87, "churn_label": 1, "model_version": "1"}
  ]
}
```

---

## Explainability

### `POST /explain`

Get SHAP-based explanations for predictions.

**Request:**

```json
{
  "user_id": 123,
  "clicks": 50,
  "purchases": 3,
  "days_since_last_visit": 5,
  "top_k": 5,
  "include_text": true
}
```

**Response:**

```json
{
  "result": {
    "user_id": 123,
    "churn_proba": 0.2341,
    "churn_label": 0,
    "explanation": {
      "base_value": 0.35,
      "prediction_contribution": -0.12,
      "top_features": [
        {
          "feature": "purchases",
          "value": 3.0,
          "shap_value": -0.15,
          "impact": "decreases"
        },
        {
          "feature": "days_since_last_visit",
          "value": 5.0,
          "shap_value": 0.08,
          "impact": "increases"
        }
      ]
    },
    "explanation_text": "The model prediction is influenced by:\n  ↓ purchases = 3.00 (decreases churn probability)\n  ↑ days_since_last_visit = 5.00 (increases churn probability)"
  }
}
```

### `GET /feature-importance`

Get global feature importance.

**Response:**

```json
{
  "feature_importance": {
    "days_since_last_visit": 0.245,
    "purchases": 0.189,
    "total_spend": 0.156,
    "clicks": 0.134,
    "active_days": 0.098
  },
  "model_type": "LGBMClassifier"
}
```

---

## Prometheus Metrics

### `GET /metrics`

Returns Prometheus-formatted metrics.

**Available Metrics:**

- `churn_api_requests_total` - Total API requests
- `churn_api_request_latency_seconds` - Request latency histogram
- `churn_predictions_total` - Total predictions by label
- `churn_prediction_probability` - Prediction probability histogram
- `churn_api_uptime_seconds` - API uptime

---

## Error Responses

```json
{
  "detail": "Error message"
}
```

**Status Codes:**

- `400` - Bad request
- `401` - Unauthorized
- `429` - Rate limited
- `500` - Internal error
