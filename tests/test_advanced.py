"""Tests for new modules: explainer, validation, hpo, etc."""
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch


class TestModelExplainer:
    """Tests for SHAP-based model explainer."""
    
    def test_explainer_init_with_tree_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from src.models.explainer import ModelExplainer
        
        # Create simple model
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        
        explainer = ModelExplainer(model=model, feature_names=feature_names)
        
        assert explainer.model == model
        assert explainer.feature_names == feature_names
        assert explainer.explainer is not None
    
    def test_explain_returns_correct_structure(self):
        from sklearn.ensemble import RandomForestClassifier
        from src.models.explainer import ModelExplainer
        
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        explainer = ModelExplainer(model=model, feature_names=feature_names)
        
        # Explain single sample
        df = pd.DataFrame(np.random.rand(1, 5), columns=feature_names)
        explanations = explainer.explain(df, top_k=3)
        
        assert len(explanations) == 1
        exp = explanations[0]
        
        assert "base_value" in exp
        assert "prediction_contribution" in exp
        assert "top_features" in exp
        assert len(exp["top_features"]) <= 3
        
        for feat in exp["top_features"]:
            assert "feature" in feat
            assert "value" in feat
            assert "shap_value" in feat
            assert "impact" in feat
            assert feat["impact"] in ["increases", "decreases"]
    
    def test_generate_explanation_text(self):
        from src.models.explainer import generate_explanation_text
        
        explanation = {
            "base_value": 0.3,
            "prediction_contribution": 0.2,
            "top_features": [
                {"feature": "clicks", "value": 50.0, "shap_value": 0.1, "impact": "increases"},
                {"feature": "purchases", "value": 3.0, "shap_value": -0.05, "impact": "decreases"},
            ]
        }
        
        text = generate_explanation_text(explanation)
        
        assert "clicks" in text
        assert "purchases" in text
        assert "increases" in text or "↑" in text


class TestDataValidation:
    """Tests for data validation module."""
    
    def test_validate_features_valid_data(self):
        from src.data.validation import validate_features
        
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "clicks": [10, 20, 30],
            "purchases": [1, 2, 3],
            "total_spend": [100.0, 200.0, 300.0],
            "avg_order_value": [50.0, 60.0, 70.0],
            "days_since_last_visit": [1.0, 2.0, 3.0],
            "active_days": [10, 20, 30],
            "account_age_days": [100.0, 200.0, 300.0],
            "churn": [0, 1, 0],
        })
        
        result = validate_features(df)
        
        assert result.is_valid or "pandera not installed" in str(result.warnings)
        assert "n_rows" in result.stats
        assert result.stats["n_rows"] == 3
    
    def test_detect_drift(self):
        from src.data.validation import detect_drift
        
        reference = pd.DataFrame({
            "f1": np.random.normal(0, 1, 100),
            "f2": np.random.normal(10, 2, 100),
        })
        
        # Current data with drift in f1
        current = pd.DataFrame({
            "f1": np.random.normal(5, 1, 100),  # Shifted mean
            "f2": np.random.normal(10, 2, 100),  # Same
        })
        
        result = detect_drift(reference, current, ["f1", "f2"])
        
        assert "has_drift" in result
        assert "features_with_drift" in result
        assert "details" in result
        assert "f1" in result["details"]
        assert "f2" in result["details"]


class TestHPO:
    """Tests for hyperparameter optimization."""
    
    def test_hpo_result_dataclass(self):
        from src.models.hpo import HPOResult
        
        result = HPOResult(
            best_params={"n_estimators": 100, "max_depth": 5},
            best_score=0.85,
            n_trials=10,
            study_name="test_study",
            algorithm="lightgbm",
            optimization_history=[],
        )
        
        assert result.best_score == 0.85
        assert result.n_trials == 10
        
        d = result.to_dict()
        assert "best_params" in d
        assert "best_score" in d
    
    @pytest.mark.slow
    def test_optimize_hyperparameters_lightgbm(self):
        from src.models.hpo import optimize_hyperparameters
        
        X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))
        
        result = optimize_hyperparameters(
            X, y,
            algorithm="lightgbm",
            n_trials=3,  # Very few for testing
            cv=2,
        )
        
        assert result.best_score > 0
        assert result.n_trials == 3
        assert "n_estimators" in result.best_params


class TestABTesting:
    """Tests for A/B testing module."""
    
    def test_experiment_is_active(self):
        from src.models.ab_testing import Experiment
        
        exp = Experiment(
            name="test",
            control_model="v1",
            treatment_model="v2",
            enabled=True,
        )
        
        assert exp.is_active()
        
        exp.enabled = False
        assert not exp.is_active()
    
    def test_ab_tester_consistent_assignment(self):
        from src.models.ab_testing import ABTester, Experiment
        
        tester = ABTester()
        tester.register_experiment(Experiment(
            name="test_exp",
            control_model="v1",
            treatment_model="v2",
            traffic_split=0.5,
        ))
        
        # Same user should always get same variant
        result1 = tester.get_variant("test_exp", user_id=12345)
        result2 = tester.get_variant("test_exp", user_id=12345)
        
        assert result1.variant == result2.variant
        assert result1.model_name == result2.model_name
    
    def test_ab_tester_traffic_split(self):
        from src.models.ab_testing import ABTester, Experiment
        
        tester = ABTester()
        tester.register_experiment(Experiment(
            name="split_test",
            control_model="v1",
            treatment_model="v2",
            traffic_split=0.3,  # 30% treatment
        ))
        
        # Test with many users
        treatment_count = 0
        for user_id in range(1000):
            result = tester.get_variant("split_test", user_id=user_id)
            if result.variant == "treatment":
                treatment_count += 1
        
        # Should be roughly 30% (allow some variance)
        treatment_rate = treatment_count / 1000
        assert 0.2 < treatment_rate < 0.4


class TestFeatureStore:
    """Tests for feature store."""
    
    def test_feature_store_without_redis(self):
        from src.features.store import FeatureStore
        
        store = FeatureStore(redis_url="redis://nonexistent:6379")
        
        # Should handle missing Redis gracefully
        result = store.get(user_id=1)
        assert result is None
        
        success = store.set(user_id=1, features={"clicks": 10})
        assert not success
    
    def test_make_key(self):
        from src.features.store import FeatureStore
        
        store = FeatureStore()
        
        key = store._make_key(user_id=123, feature_set="default")
        assert "123" in key
        assert "default" in key
    
    def test_serialize_deserialize(self):
        from src.features.store import FeatureStore
        
        store = FeatureStore()
        
        features = {"clicks": 10, "purchases": 5, "value": 123.45}
        serialized = store._serialize(features)
        deserialized = store._deserialize(serialized)
        
        assert deserialized == features


class TestModelCard:
    """Tests for model card generation."""
    
    def test_model_card_to_dict(self):
        from src.models.model_card import ModelCard
        
        card = ModelCard(
            model_name="Test Model",
            model_version="1.0.0",
            algorithm="RandomForest",
        )
        
        d = card.to_dict()
        
        assert "model_details" in d
        assert d["model_details"]["name"] == "Test Model"
        assert d["model_details"]["version"] == "1.0.0"
    
    def test_model_card_to_markdown(self):
        from src.models.model_card import ModelCard
        
        card = ModelCard(
            model_name="Test Model",
            metrics={"roc_auc": 0.85, "f1": 0.75},
            features=["clicks", "purchases"],
            feature_importance={"clicks": 0.6, "purchases": 0.4},
        )
        
        md = card.to_markdown()
        
        assert "# Model Card: Test Model" in md
        assert "roc_auc" in md
        assert "clicks" in md


class TestNotifications:
    """Tests for notification system."""
    
    @pytest.mark.asyncio
    async def test_console_channel(self):
        from src.common.notifications import ConsoleChannel, Notification, NotificationLevel
        
        channel = ConsoleChannel()
        notification = Notification(
            title="Test",
            message="Test message",
            level=NotificationLevel.INFO,
        )
        
        success = await channel.send(notification)
        assert success
    
    def test_notification_dataclass(self):
        from src.common.notifications import Notification, NotificationLevel
        
        n = Notification(
            title="Training Complete",
            message="Model v2 finished training",
            level=NotificationLevel.SUCCESS,
            metadata={"accuracy": 0.95},
        )
        
        assert n.title == "Training Complete"
        assert n.level == NotificationLevel.SUCCESS
        assert n.timestamp is not None
