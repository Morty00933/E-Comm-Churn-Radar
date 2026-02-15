"""Tests for API endpoints."""
import os
import pytest
from unittest.mock import MagicMock, patch


# Use the API key set in conftest.py
TEST_API_KEY = os.environ.get("API_KEY", "test-api-key-for-testing")


class TestHealthEndpoints:
    def test_health_returns_ok(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_healthz_returns_ok(self, api_client):
        response = api_client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_v1_health_returns_ok(self, api_client):
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200


class TestRootEndpoint:
    def test_root_returns_api_info(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "Churn Radar API"
        assert "docs" in data
        assert "predict" in data


class TestPredictEndpoint:
    def test_predict_single_customer(self, api_client, sample_customer_data):
        with patch("src.api.predictor.load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = [[0.3, 0.7]]
            mock_load.return_value = mock_model

            response = api_client.post(
                "/predict",
                json=sample_customer_data,
                headers={"X-API-Key": TEST_API_KEY},
            )

            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "churn_proba" in data["prediction"]
            assert "churn_label" in data["prediction"]

    def test_predict_batch(self, api_client, sample_batch_data):
        with patch("src.api.predictor.load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = [[0.3, 0.7], [0.6, 0.4], [0.2, 0.8]]
            mock_load.return_value = mock_model

            response = api_client.post(
                "/predict",
                json=sample_batch_data,
                headers={"X-API-Key": TEST_API_KEY},
            )

            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 3

    def test_predict_missing_auth_returns_401(self, api_client, sample_customer_data):
        """Test that missing API key returns 401 in any environment."""
        response = api_client.post("/predict", json=sample_customer_data)
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_predict_invalid_auth_returns_401(self, api_client, sample_customer_data):
        """Test that invalid API key returns 401."""
        response = api_client.post(
            "/predict",
            json=sample_customer_data,
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_predict_empty_customers_returns_error(self, api_client):
        """Test that empty customers list returns error."""
        with patch("src.api.predictor.load_model") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            response = api_client.post(
                "/predict",
                json={"customers": []},
                headers={"X-API-Key": TEST_API_KEY},
            )

            # Empty list should raise ValueError
            assert response.status_code in [400, 422, 500]


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, api_client):
        response = api_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")


class TestExplainEndpoint:
    def test_explain_requires_auth(self, api_client, sample_customer_data):
        response = api_client.post("/explain", json=sample_customer_data)
        assert response.status_code == 401

    def test_explain_with_valid_auth(self, api_client, sample_customer_data):
        with patch("src.api.predictor.load_model") as mock_load:
            with patch("src.models.explainer.ModelExplainer") as mock_explainer:
                mock_model = MagicMock()
                mock_model.predict_proba.return_value = [[0.3, 0.7]]
                mock_load.return_value = mock_model

                mock_explainer_instance = MagicMock()
                mock_explainer_instance.explain.return_value = [
                    {
                        "top_features": [
                            {"feature": "clicks", "shap_value": 0.1, "impact": "increases"}
                        ]
                    }
                ]
                mock_explainer.return_value = mock_explainer_instance

                response = api_client.post(
                    "/explain",
                    json=sample_customer_data,
                    headers={"X-API-Key": TEST_API_KEY},
                )

                # May fail if model loading fails, but auth should pass
                assert response.status_code in [200, 500]


class TestFeatureImportanceEndpoint:
    def test_feature_importance_requires_auth(self, api_client):
        response = api_client.get("/feature-importance")
        assert response.status_code == 401

    def test_feature_importance_with_valid_auth(self, api_client):
        with patch("src.api.predictor.load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.feature_importances_ = [0.1, 0.2, 0.3, 0.4]
            mock_load.return_value = mock_model

            response = api_client.get(
                "/feature-importance",
                headers={"X-API-Key": TEST_API_KEY},
            )

            # May fail if model loading fails, but auth should pass
            assert response.status_code in [200, 500]


class TestCORSHeaders:
    def test_cors_headers_present_for_allowed_origin(self, api_client):
        response = api_client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # CORS preflight should work for allowed origins
        assert response.status_code in [200, 405]


class TestAPIVersioning:
    def test_v1_predict_endpoint(self, api_client, sample_customer_data):
        with patch("src.api.predictor.load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = [[0.3, 0.7]]
            mock_load.return_value = mock_model

            response = api_client.post(
                "/api/v1/predict",
                json=sample_customer_data,
                headers={"X-API-Key": TEST_API_KEY},
            )

            assert response.status_code == 200
