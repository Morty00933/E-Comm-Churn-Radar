from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


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


def clip_outliers(df: pd.DataFrame, quantile: float = 0.99) -> pd.DataFrame:
    result = df.copy()
    
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["user_id", "churn"]]
    
    for col in numeric_cols:
        upper = result[col].quantile(quantile)
        result[col] = result[col].clip(upper=upper)
    
    return result


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    
    result["purchase_rate"] = result["purchases"] / (result["clicks"] + 1)
    result["spend_per_purchase"] = result["total_spend"] / (result["purchases"] + 1)
    result["engagement_score"] = (
        result["clicks"] * 0.3 +
        result["purchases"] * 2.0 +
        result["active_days"] * 0.5
    )
    result["recency_score"] = np.exp(-result["days_since_last_visit"] / 14)
    
    return result


def prepare_features(
    df: pd.DataFrame,
    clip_quantile: float = 0.99,
    add_derived: bool = True,
) -> pd.DataFrame:
    result = df.copy()
    
    for col in FEATURE_COLUMNS:
        if col not in result.columns:
            result[col] = 0
    
    result = result.fillna(0)
    result = clip_outliers(result, quantile=clip_quantile)
    
    if add_derived:
        result = add_derived_features(result)
    
    return result


def split_xy(
    df: pd.DataFrame,
    target_col: str = "churn",
    id_col: str = "user_id",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if feature_cols is None:
        exclude = {target_col, id_col}
        feature_cols = [c for c in df.columns if c not in exclude]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y


def get_feature_names(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = exclude or ["user_id", "churn"]
    return [c for c in df.columns if c not in exclude]
