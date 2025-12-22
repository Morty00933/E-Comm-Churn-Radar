import pytest
from unittest.mock import MagicMock, patch


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


class TestPredictEndpoint:
    def test_predict_single_customer(self, api_client, sample_customer_data):
        with patch("src.api.predictor.load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = [[0.3, 0.7]]
            mock_load.return_value = mock_model
            
            response = api_client.post(
                "/predict",
                json=sample_customer_data,
                headers={"X-API-Key": "dev-api-key"},
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
                headers={"X-API-Key": "dev-api-key"},
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 3

    def test_predict_missing_auth_in_prod(self, api_client, sample_customer_data):
        with patch.dict("os.environ", {"ENVIRONMENT": "production"}):
            response = api_client.post("/predict", json=sample_customer_data)
            assert response.status_code == 401


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, api_client):
        response = api_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
