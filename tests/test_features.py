import numpy as np
import pandas as pd
import pytest


class TestPrepareFeatures:
    def test_fills_missing_columns(self):
        from src.features.build_features import prepare_features, FEATURE_COLUMNS
        
        df = pd.DataFrame({
            "user_id": [1, 2],
            "clicks": [10, 20],
        })
        
        result = prepare_features(df)
        
        for col in FEATURE_COLUMNS:
            assert col in result.columns

    def test_clips_outliers(self):
        from src.features.build_features import clip_outliers
        
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "clicks": [10, 20, 1000],
        })
        
        result = clip_outliers(df, quantile=0.9)
        
        assert result["clicks"].max() < 1000

    def test_adds_derived_features(self):
        from src.features.build_features import add_derived_features
        
        df = pd.DataFrame({
            "clicks": [100],
            "purchases": [10],
            "total_spend": [500],
            "active_days": [30],
            "days_since_last_visit": [5],
        })
        
        result = add_derived_features(df)
        
        assert "purchase_rate" in result.columns
        assert "engagement_score" in result.columns
        assert "recency_score" in result.columns


class TestSplitXY:
    def test_splits_correctly(self):
        from src.features.build_features import split_xy
        
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "clicks": [10, 20, 30],
            "purchases": [1, 2, 3],
            "churn": [0, 1, 0],
        })
        
        X, y = split_xy(df)
        
        assert "churn" not in X.columns
        assert "user_id" not in X.columns
        assert len(y) == 3
        assert list(y) == [0, 1, 0]

    def test_custom_feature_cols(self):
        from src.features.build_features import split_xy
        
        df = pd.DataFrame({
            "user_id": [1],
            "clicks": [10],
            "purchases": [2],
            "extra_col": [100],
            "churn": [1],
        })
        
        X, y = split_xy(df, feature_cols=["clicks", "purchases"])
        
        assert list(X.columns) == ["clicks", "purchases"]
        assert "extra_col" not in X.columns
