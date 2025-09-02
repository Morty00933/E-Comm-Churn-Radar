from __future__ import annotations
import pandas as pd
import numpy as np


def aggregate_events_to_users(
    events: pd.DataFrame,
    inactive_days: int = 30,
    period_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    ev = events.copy()

    required = {"event_time", "event_type", "user_id"}
    if not required.issubset(ev.columns):
        raise ValueError(f"aggregate_events_to_users: нет обязательных колонок {required}")

    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce", utc=True)
    ev = ev.dropna(subset=["event_time"])
    ev["user_id"] = ev["user_id"].astype("int64")

    if period_end is None:
        period_end = ev["event_time"].max()
    period_end = pd.to_datetime(period_end, utc=True)

    clicks = (ev[ev["event_type"] != "purchase"]
              .groupby("user_id").size().rename("clicks").astype("int64"))
    purchases = (ev[ev["event_type"] == "purchase"]
                 .groupby("user_id").size().rename("purchases").astype("int64"))

    if "user_session" in ev.columns:
        sess = ev.groupby("user_session")["event_time"].agg(["min", "max"]).reset_index()
        sess["session_minutes"] = (sess["max"] - sess["min"]).dt.total_seconds() / 60.0
        uid_map = ev.sort_values("event_time").drop_duplicates("user_session")[["user_session", "user_id"]]
        sess = sess.merge(uid_map, on="user_session", how="left")
        avg_session_time = sess.groupby("user_id")["session_minutes"].mean().rename("avg_session_time")
        active_days = (ev.groupby(["user_id", ev["event_time"].dt.date]).size()
                         .groupby("user_id").size().rename("active_days").astype("int64"))
    else:
        avg_session_time = pd.Series(dtype="float64", name="avg_session_time")
        active_days = pd.Series(dtype="int64", name="active_days")

    last_seen = ev.groupby("user_id")["event_time"].max().rename("last_event_time")
    first_seen = ev.groupby("user_id")["event_time"].min().rename("first_event_time")
    days_since = ((period_end - last_seen).dt.total_seconds() / 86400.0).rename("days_since_last_visit")
    account_age_days = ((period_end - first_seen).dt.total_seconds() / 86400.0).rename("account_age_days")

    if "price" in ev.columns:
        avg_order_value = (ev[ev["event_type"] == "purchase"]
                           .groupby("user_id")["price"].mean().rename("avg_order_value"))
        total_spend = (ev[ev["event_type"] == "purchase"]
                       .groupby("user_id")["price"].sum().rename("total_spend"))
    else:
        avg_order_value = pd.Series(dtype="float64", name="avg_order_value")
        total_spend = pd.Series(dtype="float64", name="total_spend")

    n_cats = (ev.dropna(subset=["category_code"])
                .groupby("user_id")["category_code"].nunique().rename("n_unique_categories"))
    n_brands = (ev.dropna(subset=["brand"])
                  .groupby("user_id")["brand"].nunique().rename("n_unique_brands"))

    df = pd.DataFrame(index=ev["user_id"].unique())
    df.index.name = "user_id"
    for s in [clicks, purchases, avg_session_time, active_days, days_since,
              account_age_days, avg_order_value, total_spend, n_cats, n_brands]:
        df = df.join(s, how="left")

    df = df.reset_index()

    for c in ["clicks", "purchases", "active_days"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    for c in ["avg_session_time", "avg_order_value", "total_spend", "account_age_days", "days_since_last_visit"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    df["days_since_last_visit"] = df["days_since_last_visit"].replace(0, inactive_days + 1)

    df["churn"] = 0

    return df


def enrich_features(df: pd.DataFrame, clip_q: float = 0.99, label_noise_pct: float = 0.0) -> pd.DataFrame:
    df = df.copy()

    if "days_since_last_visit" in df.columns:
        cat = pd.Categorical(
            pd.cut(df["days_since_last_visit"], bins=[-1, 7, 30, 999999],
                   labels=["recent", "medium", "long"]),
            categories=["recent", "medium", "long"], ordered=True
        )
        df["days_bin"] = cat
    else:
        df["days_bin"] = pd.Categorical(["recent"] * len(df), categories=["recent", "medium", "long"])

    def safe_div(a, b): return a / (b + 1e-9)
    df["is_buyer"] = (df.get("purchases", 0) > 0).astype(int)
    df["high_value_user"] = (df.get("avg_order_value", 0.0) > 50).astype(int)
    df["conversion_rate"] = safe_div(df.get("purchases", 0), (df.get("clicks", 0) + 1))
    df["engagement"] = df.get("clicks", 0) * df.get("avg_session_time", 0.0)
    df["clicks_per_time"] = safe_div(df.get("clicks", 0), (df.get("avg_session_time", 0.0) + 1))
    df["order_value_per_time"] = safe_div(df.get("avg_order_value", 0.0), (df.get("avg_session_time", 0.0) + 1))
    df["spend_per_day"] = safe_div(df.get("total_spend", 0.0), (df.get("active_days", 0) + 1))
    df["purchases_per_day"] = safe_div(df.get("purchases", 0), (df.get("active_days", 0) + 1))

    df = pd.get_dummies(df, columns=["days_bin"], drop_first=False)

    for col in ["clicks", "avg_session_time", "avg_order_value", "total_spend",
                "days_since_last_visit", "account_age_days", "n_unique_categories", "n_unique_brands",
                "engagement", "spend_per_day"]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            cap = df[col].quantile(clip_q)
            df[col] = np.minimum(df[col], cap)

    if "churn" in df.columns and 0 < label_noise_pct <= 0.5:
        rng = np.random.default_rng(42)
        flip_idx = rng.choice(df.index, size=int(label_noise_pct * len(df)), replace=False)
        df.loc[flip_idx, "churn"] = 1 - df.loc[flip_idx, "churn"]

    return df


def split_Xy(df: pd.DataFrame):
    X = df.drop(columns=["user_id", "churn"], errors="ignore")
    y = df["churn"].astype(int) if "churn" in df.columns else None
    return X, y
