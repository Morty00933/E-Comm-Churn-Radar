from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.monitoring.metrics import record_prediction

logger = get_logger(__name__)

_model_cache: Dict[str, Any] = {}

FEATURE_COLUMNS = [
    "clicks",
    "purchases",
    "avg_session_time",
    "active_days",
    "days_since_last_visit",
    "account_age_days",
    "avg_order_value",
    "total_spend",
    "n_unique_categories",
    "n_unique_brands",
]


def load_model(
    model_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    stage: str = "Production",
) -> Any:
    model_name = model_name or os.getenv("MLFLOW_MODEL_NAME", "churn-model")
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    cache_key = f"{model_name}:{stage}"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    # Try MLflow first
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        
        # Try Production first, then Staging
        for try_stage in [stage, "Staging", "None"]:
            try:
                model_uri = f"models:/{model_name}/{try_stage}"
                model = mlflow.sklearn.load_model(model_uri)  # Use sklearn loader for sklearn models
                _model_cache[cache_key] = model
                logger.info(f"Loaded model {model_name} stage={try_stage}")
                return model
            except Exception as e:
                logger.debug(f"Model {model_name} not found in stage {try_stage}: {e}")
                continue
        
    except Exception as e:
        logger.warning(f"MLflow model not available: {e}")
    
    # Fall back to local model
    local_path = os.getenv("LOCAL_MODEL_PATH", "models/model.pkl")
    if os.path.exists(local_path):
        import pickle
        with open(local_path, "rb") as f:
            model = pickle.load(f)
        _model_cache[cache_key] = model
        logger.info(f"Loaded local model from {local_path}")
        return model
    
    raise RuntimeError("No model available")


def clear_model_cache():
    global _model_cache
    _model_cache.clear()
    logger.info("Model cache cleared")


def get_feature_columns() -> List[str]:
    """Return the list of feature columns used by the model.
    
    Tries to load from saved file first, falls back to default.
    """
    import json
    from pathlib import Path
    
    # Try to load from saved file
    features_path = Path("models/feature_columns.json")
    if features_path.exists():
        try:
            with open(features_path) as f:
                data = json.load(f)
                return data.get("features", FEATURE_COLUMNS.copy())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load feature columns from {features_path}: {e}")
    
    # Fall back to base features + derived
    return FEATURE_COLUMNS + [
        "purchase_rate",
        "spend_per_purchase",
        "engagement_score",
        "recency_score",
    ]


def get_model_version() -> Optional[str]:
    model_name = os.getenv("MLFLOW_MODEL_NAME", "churn-model")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0].version
    except Exception as e:
        logger.debug(f"Could not get model version from MLflow: {e}")

    return "local"


def predict_customers(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        raise ValueError("Empty records list")
    
    from src.features.build_features import prepare_features
    
    model = load_model()
    model_version = get_model_version()
    
    df = pd.DataFrame(records)
    
    # Fill missing base columns
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    
    # Add derived features (model expects 14 features)
    df = prepare_features(df)
    
    # Get all feature columns the model expects
    feature_cols = get_feature_columns()
    
    # Ensure all required columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    features = df[feature_cols].fillna(0)
    
    try:
        if hasattr(model, "predict_proba"):
            proba_output = model.predict_proba(features)
            # Convert to numpy array if needed (handles mock objects returning lists)
            if not isinstance(proba_output, np.ndarray):
                proba_output = np.array(proba_output)
            probas = proba_output[:, 1]
        elif hasattr(model, "predict"):
            predictions = model.predict(features)
            if isinstance(predictions, pd.DataFrame):
                probas = predictions.iloc[:, 0].values
            else:
                probas = np.array(predictions).flatten()
        else:
            raise ValueError("Model has no predict method")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
    
    results = []
    for i, proba in enumerate(probas):
        proba = float(proba)
        label = 1 if proba >= 0.5 else 0
        
        user_id = int(df.iloc[i].get("user_id", i))
        
        record_prediction(proba, label)
        
        results.append({
            "user_id": user_id,
            "churn_proba": round(proba, 4),
            "churn_label": label,
            "model_version": model_version,
        })
    
    return results
