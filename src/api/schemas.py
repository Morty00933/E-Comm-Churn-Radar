from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    details: Dict[str, Any] = Field(default_factory=dict)


class CustomerFeatures(BaseModel):
    user_id: int = Field(..., description="User identifier")
    clicks: int = Field(default=0, ge=0)
    purchases: int = Field(default=0, ge=0)
    avg_session_time: float = Field(default=0.0, ge=0)
    active_days: int = Field(default=0, ge=0)
    days_since_last_visit: float = Field(default=0.0, ge=0)
    account_age_days: float = Field(default=0.0, ge=0)
    avg_order_value: float = Field(default=0.0, ge=0)
    total_spend: float = Field(default=0.0, ge=0)
    n_unique_categories: int = Field(default=0, ge=0)
    n_unique_brands: int = Field(default=0, ge=0)


class PredictionResult(BaseModel):
    user_id: int
    churn_proba: float = Field(..., ge=0, le=1)
    churn_label: int = Field(..., ge=0, le=1)
    model_version: Optional[str] = None


class SinglePredictionResponse(BaseModel):
    prediction: PredictionResult


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]


class ModelInfo(BaseModel):
    name: str
    version: str
    stage: str
    metrics: Dict[str, float] = Field(default_factory=dict)
