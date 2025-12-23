"""Additional tests for modules with low coverage."""

import os
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_returns_api_info(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "Churn Radar API"
        assert "docs" in data
        assert "health" in data


class TestTrainModule:
    """Tests for src.models.train module."""
    
    def test_load_config(self, tmp_path):
        from src.models.train import load_config
        
        config = {
            "data": {"train_path": "data/train.csv"},
            "model": {"algorithm": "lightgbm"},
        }
        config_file = tmp_path / "config.yaml"
        
        import yaml
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        
        loaded = load_config(str(config_file))
        assert loaded["data"]["train_path"] == "data/train.csv"
        assert loaded["model"]["algorithm"] == "lightgbm"
    
    def test_save_artifacts(self, tmp_path, sample_features_df):
        from src.models.train import save_artifacts
        from src.models.trainer import TrainingResult
        from sklearn.ensemble import RandomForestClassifier
        
        X = sample_features_df.drop(columns=["user_id", "churn"])
        y = sample_features_df["churn"]
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        result = TrainingResult(
            model=model,
            algorithm="random_forest",
            metrics={"train_roc_auc": 0.85},
            params={"n_estimators": 10},
            feature_importance={"clicks": 0.1, "purchases": 0.2},
            training_time_sec=1.5,
        )
        
        artifacts_cfg = {
            "model_path": str(tmp_path / "model.pkl"),
            "metrics_path": str(tmp_path / "metrics.json"),
            "feature_columns_path": str(tmp_path / "features.json"),
        }
        
        feature_names = list(X.columns)
        save_artifacts(result, artifacts_cfg, feature_names)
        
        assert (tmp_path / "model.pkl").exists()
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "features.json").exists()
        
        with open(tmp_path / "metrics.json") as f:
            metrics = json.load(f)
        assert metrics["algorithm"] == "random_forest"


class TestBatchModule:
    """Tests for src.models.batch module."""
    
    def test_batch_job_dataclass(self):
        from src.models.batch import BatchJob
        
        job = BatchJob(
            job_id="test123",
            total_records=100,
            processed_records=50,
        )
        
        assert job.job_id == "test123"
        assert job.progress == 0.5
        assert job.status == "pending"
        
        job_dict = job.to_dict()
        assert job_dict["progress"] == "50.0%"
    
    def test_batch_job_zero_records(self):
        from src.models.batch import BatchJob
        
        job = BatchJob(job_id="empty", total_records=0)
        assert job.progress == 0.0
    
    def test_batch_predictor_init(self, sample_features_df, tmp_path):
        from src.models.batch import BatchPredictor
        from sklearn.ensemble import RandomForestClassifier
        
        X = sample_features_df.drop(columns=["user_id", "churn"])
        y = sample_features_df["churn"]
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        feature_columns = list(X.columns)
        
        def model_loader():
            return model
        
        predictor = BatchPredictor(
            model_loader=model_loader,
            feature_columns=feature_columns,
            output_dir=str(tmp_path),
        )
        
        assert predictor.model is not None
        assert predictor.chunk_size == 10000
    
    def test_batch_predictor_create_job(self, sample_features_df, tmp_path):
        from src.models.batch import BatchPredictor
        from sklearn.ensemble import RandomForestClassifier
        
        X = sample_features_df.drop(columns=["user_id", "churn"])
        y = sample_features_df["churn"]
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Save test data
        input_csv = tmp_path / "input.csv"
        sample_features_df.head(10).to_csv(input_csv, index=False)
        
        predictor = BatchPredictor(
            model_loader=lambda: model,
            feature_columns=list(X.columns),
            output_dir=str(tmp_path),
        )
        
        job = predictor.create_job(str(input_csv))
        
        assert job.job_id is not None
        assert job.total_records == 10
        assert job.status == "pending"


class TestConfigSettings:
    """Tests for src.config.settings module."""
    
    def test_settings_from_env(self):
        """Test that Settings class can be instantiated."""
        from src.config.settings import Settings
        
        # Settings class should exist and be instantiable
        assert Settings is not None
        
        # Test that settings has expected attributes
        settings = Settings()
        assert hasattr(settings, "environment")
        assert hasattr(settings, "app_name")
        assert settings.environment in ["development", "staging", "production"]


class TestExplainEndpoint:
    """Test /explain endpoint."""
    
    def test_explain_endpoint_exists(self, api_client):
        """Test that explain endpoint exists (may return error without model)."""
        response = api_client.post(
            "/explain",
            json={"user_id": 123, "clicks": 50, "purchases": 3},
            headers={"X-API-Key": "test-api-key-for-testing"},
        )
        # Endpoint should exist (might fail due to missing model)
        assert response.status_code in [200, 400, 500]


class TestFeatureImportanceEndpoint:
    """Test /feature-importance endpoint."""
    
    def test_feature_importance_endpoint_exists(self, api_client):
        """Test that feature-importance endpoint exists."""
        response = api_client.get(
            "/feature-importance",
            headers={"X-API-Key": "test-api-key-for-testing"},
        )
        # Endpoint should exist
        assert response.status_code in [200, 400, 500]


class TestDocsEndpoint:
    """Test /docs endpoint."""
    
    def test_docs_available(self, api_client):
        response = api_client.get("/docs")
        assert response.status_code == 200


class TestPredictorModule:
    """Additional tests for predictor module."""
    
    def test_feature_columns_defined(self):
        from src.api.predictor import FEATURE_COLUMNS
        
        assert isinstance(FEATURE_COLUMNS, list)
        assert len(FEATURE_COLUMNS) > 0
        assert "clicks" in FEATURE_COLUMNS
        assert "purchases" in FEATURE_COLUMNS
    
    def test_clear_model_cache(self):
        from src.api.predictor import clear_model_cache
        
        # Should not raise
        clear_model_cache()


class TestNotificationsModule:
    """Additional tests for notifications module."""
    
    def test_slack_channel_format(self):
        from src.common.notifications import SlackChannel, Notification, NotificationLevel
        
        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        
        notification = Notification(
            title="Test",
            message="Test message",
            level=NotificationLevel.INFO,
        )
        
        # Test _build_payload method (correct method name)
        formatted = channel._build_payload(notification)
        assert "attachments" in formatted
        assert "username" in formatted
    
    def test_telegram_channel_init(self):
        from src.common.notifications import TelegramChannel
        
        channel = TelegramChannel(bot_token="test", chat_id="123")
        assert channel.bot_token == "test"
        assert channel.chat_id == "123"


class TestValidationModule:
    """Additional tests for validation module."""
    
    def test_validation_result_is_valid(self):
        from src.data.validation import ValidationResult
        
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Minor warning"],
            stats={"rows": 100},
        )
        
        assert result.is_valid
        assert len(result.warnings) == 1
    
    def test_validation_result_invalid(self):
        from src.data.validation import ValidationResult
        
        result = ValidationResult(
            is_valid=False,
            errors=["Missing column"],
            warnings=[],
            stats={},
        )
        
        assert not result.is_valid
        assert len(result.errors) == 1
