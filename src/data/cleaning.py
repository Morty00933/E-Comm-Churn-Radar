from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def clean_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    cleaned = df.copy()
    report: Dict[str, Dict[str, float]] = {
        "dropped_rows": {},
        "filled_values": {},
    }

    cleaned["event_time"] = pd.to_datetime(
        cleaned.get("event_time"), errors="coerce", utc=True
    )
    fallback_time = cleaned["event_time"].dropna().min()
    if pd.isna(fallback_time):
        fallback_time = pd.Timestamp.utcnow().tz_localize("UTC")
    
    filled_ts = cleaned["event_time"].isna().sum()
    cleaned["event_time"] = cleaned["event_time"].fillna(fallback_time)
    
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["user_id"])
    report["dropped_rows"]["invalid_time_or_user"] = float(before - len(cleaned))
    report["filled_values"]["event_time"] = float(filled_ts)

    if "event_type" in cleaned.columns:
        cleaned["event_type"] = cleaned["event_type"].fillna("unknown")
    else:
        cleaned["event_type"] = "unknown"
    
    if "user_session" in cleaned.columns:
        cleaned["user_session"] = cleaned["user_session"].fillna("no_session")
    
    if "category_code" in cleaned.columns:
        cleaned["category_code"] = cleaned["category_code"].fillna("unknown")
    
    if "brand" in cleaned.columns:
        cleaned["brand"] = cleaned["brand"].fillna("unknown")

    if "price" in cleaned.columns:
        cleaned["price"] = pd.to_numeric(cleaned["price"], errors="coerce")
        purchase_mask = cleaned["event_type"] == "purchase"
        purchase_median = cleaned.loc[purchase_mask, "price"].median()
        if np.isnan(purchase_median):
            purchase_median = 0.0
        cleaned.loc[purchase_mask, "price"] = cleaned.loc[
            purchase_mask, "price"
        ].fillna(purchase_median)
        cleaned["price"] = cleaned["price"].fillna(0.0)
        report["filled_values"]["price"] = float(cleaned["price"].isna().sum())
    else:
        report["filled_values"]["price"] = 0.0

    return cleaned, report
