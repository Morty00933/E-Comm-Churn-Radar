import numpy as np
import pandas as pd
import pytest


class TestAlgorithmSelection:
    def test_auto_select_algorithm(self):
        from src.models.trainer import auto_select_algorithm
        
        algo = auto_select_algorithm()
        
        assert algo is not None
        assert algo.name in ["lightgbm", "xgboost", "random_forest"]

    def test_get_algorithm(self):
        from src.models.trainer import get_algorithm
        
        rf = get_algorithm("random_forest")
        
        assert rf.name == "random_forest"

    def test_get_algorithm_invalid(self):
        from src.models.trainer import get_algorithm
        
        with pytest.raises(ValueError):
            get_algorithm("invalid_algorithm")


class TestTrainer:
    def test_train_random_forest(self, sample_features_df):
        from src.models.trainer import Trainer
        from src.features.build_features import split_xy
        
        X, y = split_xy(sample_features_df)
        
        trainer = Trainer(algorithm="random_forest")
        result = trainer.train(X, y)
        
        assert result.model is not None
        assert result.algorithm == "random_forest"
        assert "train_roc_auc" in result.metrics
        assert result.training_time_sec > 0

    def test_train_with_validation(self, sample_features_df):
        from src.models.trainer import Trainer
        from src.features.build_features import split_xy
        
        train_df = sample_features_df.iloc[:80]
        val_df = sample_features_df.iloc[80:]
        
        X_train, y_train = split_xy(train_df)
        X_val, y_val = split_xy(val_df)
        
        trainer = Trainer(algorithm="random_forest")
        result = trainer.train(X_train, y_train, X_val, y_val)
        
        assert "val_roc_auc" in result.metrics
        assert "best_threshold" in result.metrics

    def test_trainer_produces_predictions(self, sample_features_df):
        from src.models.trainer import Trainer
        from src.features.build_features import split_xy
        
        X, y = split_xy(sample_features_df)
        
        trainer = Trainer(algorithm="random_forest")
        result = trainer.train(X, y)
        
        probas = result.model.predict_proba(X)
        
        assert probas.shape == (len(X), 2)
        assert all(0 <= p <= 1 for p in probas[:, 1])


class TestTrainingResult:
    def test_to_dict(self, sample_features_df):
        from src.models.trainer import Trainer
        from src.features.build_features import split_xy
        
        X, y = split_xy(sample_features_df)
        
        trainer = Trainer(algorithm="random_forest")
        result = trainer.train(X, y)
        
        result_dict = result.to_dict()
        
        assert "algorithm" in result_dict
        assert "params" in result_dict
        assert "metrics" in result_dict
        assert "training_time_sec" in result_dict
