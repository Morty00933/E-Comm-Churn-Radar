"""Data validation schemas and utilities."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema
    from pandera.typing import Series
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    pa = None


# ==================== SCHEMAS ====================

def get_events_schema() -> "DataFrameSchema":
    """Schema for raw events data."""
    if not PANDERA_AVAILABLE:
        raise ImportError("pandera not installed. Run: pip install pandera")
    
    return DataFrameSchema(
        columns={
            "event_time": Column(
                pa.DateTime,
                coerce=True,
                nullable=False,
                description="Event timestamp",
            ),
            "event_type": Column(
                str,
                checks=[
                    Check.isin(["view", "cart", "purchase", "remove_from_cart", "unknown"]),
                ],
                nullable=False,
                description="Type of user action",
            ),
            "user_id": Column(
                int,
                checks=[Check.greater_than(0)],
                coerce=True,
                nullable=False,
                description="Unique user identifier",
            ),
            "product_id": Column(
                int,
                coerce=True,
                nullable=True,
                required=False,
                description="Product identifier",
            ),
            "category_id": Column(
                int,
                coerce=True,
                nullable=True,
                required=False,
            ),
            "category_code": Column(
                str,
                nullable=True,
                required=False,
            ),
            "brand": Column(
                str,
                nullable=True,
                required=False,
            ),
            "price": Column(
                float,
                checks=[Check.greater_than_or_equal_to(0)],
                coerce=True,
                nullable=True,
            ),
            "user_session": Column(
                str,
                nullable=True,
                required=False,
            ),
        },
        strict=False,  # Allow extra columns
        coerce=True,
    )


def get_features_schema() -> "DataFrameSchema":
    """Schema for processed features data."""
    if not PANDERA_AVAILABLE:
        raise ImportError("pandera not installed. Run: pip install pandera")
    
    return DataFrameSchema(
        columns={
            "user_id": Column(
                int,
                checks=[Check.greater_than(0)],
                coerce=True,
                nullable=False,
            ),
            "clicks": Column(
                int,
                checks=[Check.greater_than_or_equal_to(0)],
                coerce=True,
                nullable=False,
            ),
            "purchases": Column(
                int,
                checks=[Check.greater_than_or_equal_to(0)],
                coerce=True,
                nullable=False,
            ),
            "total_spend": Column(
                float,
                checks=[Check.greater_than_or_equal_to(0)],
                coerce=True,
                nullable=False,
            ),
            "avg_order_value": Column(
                float,
                checks=[Check.greater_than_or_equal_to(0)],
                coerce=True,
                nullable=False,
            ),
            "days_since_last_visit": Column(
                float,
                checks=[Check.greater_than_or_equal_to(0)],
                coerce=True,
                nullable=False,
            ),
            "active_days": Column(
                int,
                checks=[Check.greater_than_or_equal_to(0)],
                coerce=True,
                nullable=False,
            ),
            "account_age_days": Column(
                float,
                checks=[Check.greater_than_or_equal_to(0)],
                coerce=True,
                nullable=False,
            ),
            "churn": Column(
                int,
                checks=[Check.isin([0, 1])],
                coerce=True,
                nullable=False,
                required=False,  # Not required for inference
            ),
        },
        strict=False,
        coerce=True,
    )


def get_prediction_input_schema() -> "DataFrameSchema":
    """Schema for prediction API input."""
    if not PANDERA_AVAILABLE:
        raise ImportError("pandera not installed. Run: pip install pandera")
    
    return DataFrameSchema(
        columns={
            "user_id": Column(int, coerce=True, nullable=True, required=False),
            "clicks": Column(int, coerce=True, nullable=True, required=False),
            "purchases": Column(int, coerce=True, nullable=True, required=False),
            "avg_session_time": Column(float, coerce=True, nullable=True, required=False),
            "active_days": Column(int, coerce=True, nullable=True, required=False),
            "days_since_last_visit": Column(float, coerce=True, nullable=True, required=False),
            "account_age_days": Column(float, coerce=True, nullable=True, required=False),
            "avg_order_value": Column(float, coerce=True, nullable=True, required=False),
            "total_spend": Column(float, coerce=True, nullable=True, required=False),
            "n_unique_categories": Column(int, coerce=True, nullable=True, required=False),
            "n_unique_brands": Column(int, coerce=True, nullable=True, required=False),
        },
        strict=False,
        coerce=True,
    )


# ==================== VALIDATION ====================

class ValidationResult:
    """Result of data validation."""
    
    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[Dict[str, Any]]] = None,
        warnings: Optional[List[str]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.stats = stats or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def validate_events(df: pd.DataFrame) -> ValidationResult:
    """Validate events DataFrame."""
    if not PANDERA_AVAILABLE:
        return ValidationResult(is_valid=True, warnings=["pandera not installed, skipping validation"])
    
    schema = get_events_schema()
    errors = []
    warnings = []
    
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        for error in e.failure_cases.to_dict("records"):
            errors.append({
                "column": error.get("column"),
                "check": error.get("check"),
                "failure_case": str(error.get("failure_case"))[:100],
            })
    
    # Compute stats
    stats = {
        "n_rows": len(df),
        "n_users": df["user_id"].nunique() if "user_id" in df.columns else 0,
        "date_range": None,
        "event_types": {},
    }
    
    if "event_time" in df.columns:
        try:
            times = pd.to_datetime(df["event_time"])
            stats["date_range"] = {
                "min": str(times.min()),
                "max": str(times.max()),
            }
        except (ValueError, TypeError) as e:
            warnings.append(f"Could not parse event_time column: {e}")
    
    if "event_type" in df.columns:
        stats["event_types"] = df["event_type"].value_counts().to_dict()
    
    # Warnings
    if len(df) < 100:
        warnings.append("Small dataset (<100 rows)")
    
    if "user_id" in df.columns:
        null_users = df["user_id"].isna().sum()
        if null_users > 0:
            warnings.append(f"{null_users} rows with null user_id")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def validate_features(df: pd.DataFrame) -> ValidationResult:
    """Validate features DataFrame."""
    if not PANDERA_AVAILABLE:
        return ValidationResult(is_valid=True, warnings=["pandera not installed"])
    
    schema = get_features_schema()
    errors = []
    warnings = []
    
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        for error in e.failure_cases.to_dict("records"):
            errors.append({
                "column": error.get("column"),
                "check": error.get("check"),
                "failure_case": str(error.get("failure_case"))[:100],
            })
    
    # Stats
    stats = {
        "n_rows": len(df),
        "n_features": len(df.columns),
        "missing_values": df.isna().sum().to_dict(),
    }
    
    if "churn" in df.columns:
        stats["churn_rate"] = float(df["churn"].mean())
        
        # Check for class imbalance
        if stats["churn_rate"] < 0.05:
            warnings.append(f"Severe class imbalance: {stats['churn_rate']:.1%} churn rate")
        elif stats["churn_rate"] < 0.15:
            warnings.append(f"Class imbalance: {stats['churn_rate']:.1%} churn rate")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


# ==================== DRIFT DETECTION ====================

def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: List[str],
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """Detect data drift between reference and current data.
    
    Uses simple statistical tests to detect distribution changes.
    """
    drift_report = {
        "has_drift": False,
        "features_with_drift": [],
        "details": {},
    }
    
    for col in feature_columns:
        if col not in reference.columns or col not in current.columns:
            continue
        
        ref_values = reference[col].dropna()
        cur_values = current[col].dropna()
        
        if len(ref_values) == 0 or len(cur_values) == 0:
            continue
        
        # Simple drift detection: compare means and stds
        ref_mean = ref_values.mean()
        cur_mean = cur_values.mean()
        ref_std = ref_values.std()
        
        if ref_std > 0:
            normalized_diff = abs(cur_mean - ref_mean) / ref_std
        else:
            normalized_diff = 0 if ref_mean == cur_mean else float("inf")
        
        is_drifted = normalized_diff > threshold * 10  # ~1 std = 10% threshold
        
        drift_report["details"][col] = {
            "reference_mean": float(ref_mean),
            "current_mean": float(cur_mean),
            "normalized_diff": float(normalized_diff),
            "is_drifted": is_drifted,
        }
        
        if is_drifted:
            drift_report["has_drift"] = True
            drift_report["features_with_drift"].append(col)
    
    return drift_report
