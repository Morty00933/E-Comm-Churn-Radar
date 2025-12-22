# Model Explainability

Churn Radar provides SHAP-based model explanations to understand prediction drivers.

## API Endpoints

### Single Prediction Explanation

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{
    "user_id": 123,
    "clicks": 50,
    "purchases": 3,
    "days_since_last_visit": 15,
    "top_k": 5,
    "include_text": true
  }'
```

**Response:**

```json
{
  "result": {
    "user_id": 123,
    "churn_proba": 0.7234,
    "churn_label": 1,
    "explanation": {
      "base_value": 0.32,
      "prediction_contribution": 0.40,
      "top_features": [
        {
          "feature": "days_since_last_visit",
          "value": 15.0,
          "shap_value": 0.25,
          "impact": "increases"
        },
        {
          "feature": "purchases",
          "value": 3.0,
          "shap_value": -0.12,
          "impact": "decreases"
        }
      ]
    },
    "explanation_text": "The model prediction is influenced by:\n  ↑ days_since_last_visit = 15.00 (increases churn probability)\n  ↓ purchases = 3.00 (decreases churn probability)"
  }
}
```

### Batch Explanations

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{
    "customers": [
      {"user_id": 1, "clicks": 50, "purchases": 3},
      {"user_id": 2, "clicks": 5, "purchases": 0}
    ],
    "top_k": 3
  }'
```

### Global Feature Importance

```bash
curl http://localhost:8000/feature-importance \
  -H "X-API-Key: dev-api-key"
```

**Response:**

```json
{
  "feature_importance": {
    "days_since_last_visit": 0.2534,
    "purchases": 0.1892,
    "clicks": 0.1456,
    "total_spend": 0.1234,
    "active_days": 0.0987
  },
  "model_type": "LGBMClassifier"
}
```

## Understanding SHAP Values

### What is SHAP?

SHAP (SHapley Additive exPlanations) provides:

- **Consistent** - Same contribution = same SHAP value
- **Local accuracy** - Sum of SHAP values = prediction
- **Missingness** - Zero impact for missing features

### Interpreting Results

| SHAP Value | Meaning |
|------------|---------|
| Positive | Increases churn probability |
| Negative | Decreases churn probability |
| Near zero | Little impact on prediction |

### Example Interpretation

```
User #123: 72% churn probability

Top factors:
  ↑ days_since_last_visit = 15 (+0.25)
     Long absence signals disengagement
  
  ↓ purchases = 3 (-0.12)
     Purchase history indicates engagement
  
  ↑ clicks = 5 (+0.08)
     Low activity despite visits
```

## Programmatic Usage

```python
from src.models.explainer import ModelExplainer, generate_explanation_text
from src.api.predictor import load_model, get_feature_columns
import pandas as pd

# Load model
model = load_model()
feature_cols = get_feature_columns()

# Create explainer
explainer = ModelExplainer(
    model=model,
    feature_names=feature_cols,
)

# Prepare data
df = pd.DataFrame([{
    "clicks": 50,
    "purchases": 3,
    "days_since_last_visit": 15,
    # ... other features
}])

# Get explanations
explanations = explainer.explain(df, top_k=5)

# Generate human-readable text
for exp in explanations:
    print(generate_explanation_text(exp))
```

## Best Practices

1. **Use top_k wisely** - 3-5 features is usually sufficient
2. **Compare similar users** - Look for patterns in explanations
3. **Monitor drift** - Feature importance changes may indicate model staleness
4. **Validate with domain experts** - Ensure explanations make business sense
