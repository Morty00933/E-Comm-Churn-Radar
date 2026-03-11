"""End-to-end integration tests for the full ML pipeline.

Tests run in-process without Docker. A real model is trained on demo data
and the FastAPI app is exercised via TestClient.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("RATE_LIMIT_ENABLED", "0")
os.environ.setdefault("API_KEY", "test-api-key-for-testing")


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_data_dir(tmp_path_factory):
    from src.data.make_dataset import generate_demo_data

    out_dir = tmp_path_factory.mktemp("data")
    generate_demo_data(n_users=300, n_months=2, output_dir=str(out_dir))
    return out_dir


@pytest.fixture(scope="module")
def trained_artifacts(demo_data_dir, tmp_path_factory):
    import yaml
    from src.models.train import train_model

    model_dir = tmp_path_factory.mktemp("models")
    config_dir = tmp_path_factory.mktemp("configs")

    cfg = {
        "data": {
            "train_path": str(demo_data_dir / "train.csv"),
            "test_path": str(demo_data_dir / "test.csv"),
        },
        "model": {
            "algorithm": "random_forest",
            "cv_folds": 2,
            "candidates": {
                "random_forest": {"n_estimators": 20, "max_depth": 5, "random_state": 42},
            },
        },
        "features": {"clip_quantile": 0.99},
        "artifacts": {
            "model_path": str(model_dir / "model.pkl"),
            "metrics_path": str(model_dir / "metrics.json"),
            "feature_columns_path": str(model_dir / "feature_columns.json"),
        },
        "mlflow": {},
    }

    config_path = config_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    result = train_model(config_path=str(config_path), use_mlflow=False)
    return {"result": result, "model_dir": model_dir}


@pytest.fixture(scope="module")
def e2e_client(trained_artifacts):
    """TestClient with trained model pre-loaded into cache (no MLflow connection)."""
    import json
    import pickle
    import time
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    import src.api.predictor as predictor_mod

    with open(trained_artifacts["model_dir"] / "model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(trained_artifacts["model_dir"] / "feature_columns.json") as f:
        feature_cols = json.load(f)["features"]

    # Pre-populate cache so load_model() never tries to reach MLflow
    predictor_mod._model_cache["churn-model:Production"] = (model, time.time())

    os.environ["LOCAL_MODEL_PATH"] = str(trained_artifacts["model_dir"] / "model.pkl")
    # File-based URI makes get_model_version() fail fast (empty registry, no network)
    os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/e2e_test_mlflow"

    patcher = patch.object(predictor_mod, "get_feature_columns", return_value=feature_cols)
    patcher.start()

    from src.api.app import app
    client = TestClient(app)

    yield client

    patcher.stop()
    predictor_mod.clear_model_cache()
    os.environ.pop("LOCAL_MODEL_PATH", None)


# ---------------------------------------------------------------------------
# TestDataPipeline
# ---------------------------------------------------------------------------

class TestDataPipeline:
    def test_demo_data_creates_csv_files(self, demo_data_dir):
        train = pd.read_csv(demo_data_dir / "train.csv")
        test = pd.read_csv(demo_data_dir / "test.csv")
        assert len(train) > 0 and len(test) > 0
        assert "churn" in train.columns and "user_id" in train.columns

    def test_churn_column_is_binary(self, demo_data_dir):
        train = pd.read_csv(demo_data_dir / "train.csv")
        assert set(train["churn"].unique()).issubset({0, 1})

    def test_feature_preparation_no_nan(self, demo_data_dir):
        from src.features.build_features import prepare_features

        train = pd.read_csv(demo_data_dir / "train.csv")
        df = prepare_features(train)
        numeric = df.select_dtypes(include=[np.number]).columns
        assert df[numeric].isna().sum().sum() == 0

    def test_feature_preparation_adds_derived_columns(self, demo_data_dir):
        from src.features.build_features import prepare_features

        train = pd.read_csv(demo_data_dir / "train.csv")
        df = prepare_features(train)
        assert "purchase_rate" in df.columns
        assert "engagement_score" in df.columns


# ---------------------------------------------------------------------------
# TestTrainingPipeline
# ---------------------------------------------------------------------------

class TestTrainingPipeline:
    def test_training_completes(self, trained_artifacts):
        assert trained_artifacts["result"].model is not None

    def test_roc_auc_above_chance(self, trained_artifacts):
        roc_auc = trained_artifacts["result"].metrics.get("train_roc_auc", 0)
        assert roc_auc > 0.5, f"ROC-AUC too low: {roc_auc}"

    def test_model_pkl_saved(self, trained_artifacts):
        assert (trained_artifacts["model_dir"] / "model.pkl").exists()

    def test_metrics_json_saved(self, trained_artifacts):
        path = trained_artifacts["model_dir"] / "metrics.json"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "metrics" in data

    def test_feature_columns_json_saved(self, trained_artifacts):
        path = trained_artifacts["model_dir"] / "feature_columns.json"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data.get("features", [])) > 0

    def test_saved_model_is_loadable(self, trained_artifacts):
        with open(trained_artifacts["model_dir"] / "model.pkl", "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "predict")


# ---------------------------------------------------------------------------
# TestAPIPipeline
# ---------------------------------------------------------------------------

HEADERS = {"X-API-Key": "test-api-key-for-testing"}
CUSTOMER = {
    "user_id": 1, "clicks": 50, "purchases": 3,
    "avg_session_time": 12.5, "active_days": 20,
    "days_since_last_visit": 5, "account_age_days": 180,
    "avg_order_value": 75.0, "total_spend": 225.0,
    "n_unique_categories": 5, "n_unique_brands": 8,
}


class TestAPIPipeline:
    def test_healthz(self, e2e_client):
        r = e2e_client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_predict_single_returns_200(self, e2e_client):
        r = e2e_client.post("/predict", json=CUSTOMER, headers=HEADERS)
        assert r.status_code == 200

    def test_predict_single_has_churn_proba(self, e2e_client):
        r = e2e_client.post("/predict", json=CUSTOMER, headers=HEADERS)
        proba = r.json()["prediction"]["churn_proba"]
        assert 0.0 <= proba <= 1.0

    def test_predict_single_has_churn_label(self, e2e_client):
        r = e2e_client.post("/predict", json=CUSTOMER, headers=HEADERS)
        assert r.json()["prediction"]["churn_label"] in {0, 1}

    def test_predict_batch_returns_correct_count(self, e2e_client):
        payload = {"customers": [{**CUSTOMER, "user_id": i} for i in range(1, 6)]}
        r = e2e_client.post("/predict", json=payload, headers=HEADERS)
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 5

    def test_predict_without_key_returns_401(self, e2e_client):
        r = e2e_client.post("/predict", json=CUSTOMER)
        assert r.status_code == 401

    def test_predict_empty_batch_returns_422(self, e2e_client):
        r = e2e_client.post("/predict", json={"customers": []}, headers=HEADERS)
        assert r.status_code == 422

    def test_predict_minimal_payload(self, e2e_client):
        r = e2e_client.post("/predict", json={"user_id": 99}, headers=HEADERS)
        assert r.status_code == 200
        assert 0.0 <= r.json()["prediction"]["churn_proba"] <= 1.0

    def test_v1_predict_endpoint(self, e2e_client):
        r = e2e_client.post("/api/v1/predict", json=CUSTOMER, headers=HEADERS)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# TestModelCache
# ---------------------------------------------------------------------------

class TestModelCache:
    def test_cache_cleared_on_demand(self, trained_artifacts):
        import src.api.predictor as predictor_mod

        os.environ["LOCAL_MODEL_PATH"] = str(trained_artifacts["model_dir"] / "model.pkl")
        os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/e2e_test_mlflow"
        predictor_mod.clear_model_cache()
        assert len(predictor_mod._model_cache) == 0

    def test_model_loads_after_cache_clear(self, trained_artifacts):
        import src.api.predictor as predictor_mod

        os.environ["LOCAL_MODEL_PATH"] = str(trained_artifacts["model_dir"] / "model.pkl")
        os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/e2e_test_mlflow"
        predictor_mod.clear_model_cache()
        model = predictor_mod.load_model()
        assert model is not None and hasattr(model, "predict")

    def test_cache_populated_after_load(self, trained_artifacts):
        import src.api.predictor as predictor_mod

        os.environ["LOCAL_MODEL_PATH"] = str(trained_artifacts["model_dir"] / "model.pkl")
        os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/e2e_test_mlflow"
        predictor_mod.clear_model_cache()
        predictor_mod.load_model()
        assert len(predictor_mod._model_cache) == 1

    def test_cache_ttl_expiry(self, trained_artifacts, monkeypatch):
        import src.api.predictor as predictor_mod

        monkeypatch.setenv("LOCAL_MODEL_PATH", str(trained_artifacts["model_dir"] / "model.pkl"))
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/e2e_test_mlflow")
        monkeypatch.setattr(predictor_mod, "MODEL_CACHE_TTL", 1)

        predictor_mod.clear_model_cache()
        predictor_mod.load_model()
        time.sleep(1.1)
        predictor_mod.load_model()

        for _key, (_model, loaded_at) in predictor_mod._model_cache.items():
            assert time.time() - loaded_at < 1.0


# ---------------------------------------------------------------------------
# TestFullPipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_high_risk_scores_higher(self, e2e_client):
        low_risk = {
            "user_id": 1, "clicks": 200, "purchases": 20,
            "avg_session_time": 30.0, "active_days": 60,
            "days_since_last_visit": 1, "account_age_days": 365,
            "avg_order_value": 100.0, "total_spend": 2000.0,
            "n_unique_categories": 10, "n_unique_brands": 15,
        }
        high_risk = {
            "user_id": 2, "clicks": 1, "purchases": 0,
            "avg_session_time": 0.5, "active_days": 1,
            "days_since_last_visit": 60, "account_age_days": 30,
            "avg_order_value": 0.0, "total_spend": 0.0,
            "n_unique_categories": 1, "n_unique_brands": 1,
        }
        proba_low = e2e_client.post("/predict", json=low_risk, headers=HEADERS).json()["prediction"]["churn_proba"]
        proba_high = e2e_client.post("/predict", json=high_risk, headers=HEADERS).json()["prediction"]["churn_proba"]
        assert proba_high > proba_low, f"Expected high-risk ({proba_high}) > low-risk ({proba_low})"

    def test_prediction_is_deterministic(self, e2e_client):
        r1 = e2e_client.post("/predict", json=CUSTOMER, headers=HEADERS)
        r2 = e2e_client.post("/predict", json=CUSTOMER, headers=HEADERS)
        assert r1.json()["prediction"]["churn_proba"] == r2.json()["prediction"]["churn_proba"]
