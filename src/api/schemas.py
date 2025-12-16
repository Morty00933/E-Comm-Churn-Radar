from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    status: str
    details: Dict[str, Any] = Field(default_factory=dict)


class CustomerFeatures(BaseModel):
    """Customer features for churn prediction.

    All numeric fields have reasonable bounds to prevent
    data quality issues and potential attacks.
    """

    user_id: int = Field(..., description="User identifier", ge=0)
    clicks: int = Field(default=0, ge=0, le=1_000_000, description="Total clicks")
    purchases: int = Field(default=0, ge=0, le=100_000, description="Total purchases")
    avg_session_time: float = Field(
        default=0.0, ge=0, le=1440, description="Avg session time in minutes (max 24h)"
    )
    active_days: int = Field(default=0, ge=0, le=3650, description="Active days (max 10 years)")
    days_since_last_visit: float = Field(
        default=0.0, ge=0, le=3650, description="Days since last visit"
    )
    account_age_days: float = Field(
        default=0.0, ge=0, le=7300, description="Account age in days (max 20 years)"
    )
    avg_order_value: float = Field(
        default=0.0, ge=0, le=1_000_000, description="Average order value"
    )
    total_spend: float = Field(
        default=0.0, ge=0, le=100_000_000, description="Total spend amount"
    )
    n_unique_categories: int = Field(
        default=0, ge=0, le=10_000, description="Unique categories viewed"
    )
    n_unique_brands: int = Field(
        default=0, ge=0, le=100_000, description="Unique brands viewed"
    )

    @field_validator("avg_order_value", "total_spend", mode="before")
    @classmethod
    def round_money(cls, v):
        """Round monetary values to 2 decimal places."""
        if v is None:
            return 0.0
        return round(float(v), 2)

    @model_validator(mode="after")
    def validate_consistency(self):
        """Validate that features are logically consistent."""
        # Active days should not exceed account age
        if self.active_days > self.account_age_days and self.account_age_days > 0:
            self.active_days = int(self.account_age_days)

        # Days since last visit should not exceed account age
        if self.days_since_last_visit > self.account_age_days and self.account_age_days > 0:
            self.days_since_last_visit = self.account_age_days

        return self


class PredictionResult(BaseModel):
    user_id: int
    churn_proba: float = Field(..., ge=0, le=1)
    churn_label: int = Field(..., ge=0, le=1)
    model_version: Optional[str] = None


class SinglePredictionResponse(BaseModel):
    prediction: PredictionResult


class BatchPredictionRequest(BaseModel):
    """Batch prediction request with size limits."""

    customers: List[CustomerFeatures] = Field(
        ..., min_length=1, max_length=10_000, description="List of customers (max 10,000)"
    )


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]


class ExplainRequest(BaseModel):
    """Request for model explanation."""

    # Single customer or batch
    customers: Optional[List[CustomerFeatures]] = Field(
        default=None, max_length=100, description="Batch of customers (max 100 for explain)"
    )

    # Single customer fields (alternative to customers list)
    user_id: Optional[int] = Field(default=None, ge=0)
    clicks: Optional[int] = Field(default=None, ge=0, le=1_000_000)
    purchases: Optional[int] = Field(default=None, ge=0, le=100_000)
    avg_session_time: Optional[float] = Field(default=None, ge=0, le=1440)
    active_days: Optional[int] = Field(default=None, ge=0, le=3650)
    days_since_last_visit: Optional[float] = Field(default=None, ge=0, le=3650)
    account_age_days: Optional[float] = Field(default=None, ge=0, le=7300)
    avg_order_value: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    total_spend: Optional[float] = Field(default=None, ge=0, le=100_000_000)
    n_unique_categories: Optional[int] = Field(default=None, ge=0, le=10_000)
    n_unique_brands: Optional[int] = Field(default=None, ge=0, le=100_000)

    # Explanation options
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top features to return")
    include_text: bool = Field(default=True, description="Include text explanation")

    @model_validator(mode="after")
    def validate_has_data(self):
        """Ensure either customers list or single customer fields are provided."""
        if self.customers is None and self.user_id is None:
            raise ValueError("Provide either 'customers' list or single customer fields")
        return self


class ModelInfo(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., min_length=1, max_length=50)
    stage: str = Field(..., pattern=r"^(Production|Staging|Archived|None)$")
    metrics: Dict[str, float] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: Optional[str] = None
    request_id: Optional[str] = None
