import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("RATE_LIMIT_ENABLED", "0")


@pytest.fixture
def sample_customer_data():
    return {
        "user_id": 123,
        "clicks": 50,
        "purchases": 3,
        "avg_session_time": 12.5,
        "active_days": 30,
        "days_since_last_visit": 5,
        "account_age_days": 180,
        "avg_order_value": 75.0,
        "total_spend": 225.0,
        "n_unique_categories": 5,
        "n_unique_brands": 8,
    }


@pytest.fixture
def sample_batch_data(sample_customer_data):
    return {
        "customers": [
            {**sample_customer_data, "user_id": i}
            for i in range(1, 4)
        ]
    }


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient
    from src.api.app import app
    
    return TestClient(app)


@pytest.fixture
def sample_events_df():
    import pandas as pd
    
    return pd.DataFrame({
        "event_time": pd.date_range("2024-01-01", periods=100, freq="H"),
        "event_type": ["view"] * 70 + ["cart"] * 20 + ["purchase"] * 10,
        "user_id": [1] * 30 + [2] * 40 + [3] * 30,
        "category_code": ["electronics.phone"] * 100,
        "brand": ["samsung"] * 50 + ["apple"] * 50,
        "price": [0.0] * 90 + [99.99] * 10,
    })


@pytest.fixture
def sample_features_df():
    import pandas as pd
    import numpy as np
    
    rng = np.random.default_rng(42)
    n = 100
    
    return pd.DataFrame({
        "user_id": range(1, n + 1),
        "clicks": rng.integers(0, 100, n),
        "purchases": rng.integers(0, 10, n),
        "avg_session_time": rng.uniform(0, 30, n),
        "active_days": rng.integers(0, 60, n),
        "days_since_last_visit": rng.uniform(0, 30, n),
        "account_age_days": rng.uniform(30, 365, n),
        "avg_order_value": rng.uniform(0, 200, n),
        "total_spend": rng.uniform(0, 1000, n),
        "n_unique_categories": rng.integers(0, 20, n),
        "n_unique_brands": rng.integers(0, 30, n),
        "churn": rng.integers(0, 2, n),
    })
