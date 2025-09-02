from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

def generate_synthetic(n_users: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clicks = rng.poisson(25, n_users)
    purchases = rng.poisson(2.5, n_users)
    avg_session_time = np.clip(rng.normal(6, 2, n_users), 0.2, None)
    days_since_last_visit = rng.integers(1, 90, n_users)
    aov = np.clip(rng.normal(35, 20, n_users), 1, None)
    churn = ((purchases == 0) & (days_since_last_visit > 30) | ((clicks < 5) & (avg_session_time < 2))).astype(int)
    df = pd.DataFrame({
        "user_id": np.arange(n_users),
        "clicks": clicks,
        "purchases": purchases,
        "avg_session_time": avg_session_time,
        "days_since_last_visit": days_since_last_visit,
        "avg_order_value": aov,
        "churn": churn,
    })
    return df

def load_raw(csv_path: Optional[str]) -> pd.DataFrame:
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        return df
    return generate_synthetic(n_users=10000)
