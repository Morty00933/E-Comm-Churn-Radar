from __future__ import annotations

import argparse
import glob
import os
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# Dataset: eCommerce Behavior Data from Multi Category Store
# Source: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
# Files: 2019-Oct.csv, 2019-Nov.csv, 2019-Dec.csv, 2020-Jan.csv, 2020-Feb.csv
#
# Place CSV files in data/raw/ directory

REQUIRED_COLUMNS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "brand",
    "price",
    "user_id",
    "user_session",
]


def load_real_data(
    raw_dir: str = "data/raw",
    months: Optional[List[str]] = None,
    sample_frac: Optional[float] = None,
    max_rows_per_file: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load real eCommerce data from CSV files.
    
    Args:
        raw_dir: Directory with CSV files
        months: List of months to load, e.g. ["2019-Oct", "2019-Nov"]
                If None, loads all available files
        sample_frac: Fraction of data to sample (0.0-1.0)
        max_rows_per_file: Max rows to read per file (for quick testing)
        seed: Random seed for sampling
    
    Returns:
        Combined DataFrame with all events
    """
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_path}\n"
            "Create data/raw/ directory and place CSV files there.\n"
            "Or use --demo to generate synthetic data."
        )
    
    csv_files = sorted(glob.glob(str(raw_path / "*.csv")))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")
    
    # Filter by months if specified
    if months:
        csv_files = [
            f for f in csv_files
            if any(m in Path(f).stem for m in months)
        ]
        if not csv_files:
            raise FileNotFoundError(f"No files found for months: {months}")
    
    print(f"Loading {len(csv_files)} files: {[Path(f).name for f in csv_files]}")
    
    dfs = []
    for csv_file in csv_files:
        print(f"  Reading {Path(csv_file).name}...", end=" ")
        
        df = pd.read_csv(
            csv_file,
            nrows=max_rows_per_file,
            dtype={
                "product_id": "int64",
                "category_id": "int64",
                "user_id": "int64",
            },
            parse_dates=["event_time"],
        )
        
        print(f"{len(df):,} rows")
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(combined):,} events")
    
    # Sample if requested
    if sample_frac and sample_frac < 1.0:
        combined = combined.sample(frac=sample_frac, random_state=seed)
        print(f"Sampled {sample_frac:.1%}: {len(combined):,} events")
    
    return combined


def process_real_data(
    raw_dir: str = "data/raw",
    output_dir: str = "data",
    months: Optional[List[str]] = None,
    train_months: Optional[List[str]] = None,
    test_months: Optional[List[str]] = None,
    sample_frac: Optional[float] = None,
    max_users: Optional[int] = None,
    churn_days_threshold: int = 14,
    seed: int = 42,
) -> dict:
    """Process real Kaggle data into train/test datasets.
    
    Churn definition: User has no activity in the last `churn_days_threshold` days
    of the observation period.
    
    Args:
        raw_dir: Directory with raw CSV files
        output_dir: Directory for processed data
        months: Months to use (default: all available)
        train_months: Months for training (default: all except last)
        test_months: Months for testing (default: last month)
        sample_frac: Fraction of events to sample
        max_users: Maximum number of users (for quick testing)
        churn_days_threshold: Days without activity = churn
        seed: Random seed
    
    Returns:
        Dict with paths and statistics
    """
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    events = load_real_data(
        raw_dir=raw_dir,
        months=months,
        sample_frac=sample_frac,
        seed=seed,
    )
    
    # Clean events
    from src.data.cleaning import clean_events
    events, cleaning_report = clean_events(events)
    print(f"After cleaning: {len(events):,} events")
    
    # Determine train/test split by time
    events = events.sort_values("event_time")
    max_date = events["event_time"].max()
    min_date = events["event_time"].min()
    
    # Default: last 2 weeks for churn labeling, rest for features
    cutoff_date = max_date - pd.Timedelta(days=churn_days_threshold)
    label_start = cutoff_date
    label_end = max_date
    
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print(f"Feature window: {min_date.date()} to {cutoff_date.date()}")
    print(f"Churn label window: {cutoff_date.date()} to {max_date.date()}")
    
    # Split events
    feature_events = events[events["event_time"] < cutoff_date]
    label_events = events[events["event_time"] >= cutoff_date]
    
    # Aggregate features
    features = aggregate_events(feature_events)
    print(f"Users with features: {len(features):,}")
    
    # Generate churn labels based on activity in label window
    users_active_in_label_window = set(label_events["user_id"].unique())
    features["churn"] = (~features["user_id"].isin(users_active_in_label_window)).astype(int)
    
    print(f"Churn rate: {features['churn'].mean():.2%}")
    
    # Limit users if requested
    if max_users and len(features) > max_users:
        features = features.sample(n=max_users, random_state=seed)
        print(f"Sampled {max_users:,} users")
    
    # Train/test split (80/20)
    features = features.sample(frac=1.0, random_state=seed)  # shuffle
    split_idx = int(len(features) * 0.8)
    
    train_df = features.iloc[:split_idx]
    test_df = features.iloc[split_idx:]
    
    # Save
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved {len(train_df):,} train users -> {train_path}")
    print(f"Saved {len(test_df):,} test users -> {test_path}")
    print(f"Train churn rate: {train_df['churn'].mean():.2%}")
    print(f"Test churn rate: {test_df['churn'].mean():.2%}")
    
    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_churn_rate": float(train_df["churn"].mean()),
        "test_churn_rate": float(test_df["churn"].mean()),
        "date_range": f"{min_date.date()} to {max_date.date()}",
        "cleaning_report": cleaning_report,
    }


def generate_demo_data(
    n_users: int = 1000,
    n_months: int = 3,
    output_dir: str = "data",
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_users = np.arange(1, n_users + 1)
    test_users = np.arange(n_users + 1, n_users + int(n_users * 0.3) + 1)
    
    categories = [
        "electronics.smartphone",
        "electronics.audio",
        "appliances.kitchen",
        "computers.notebook",
        "sports.fitness",
    ]
    brands = ["apple", "samsung", "xiaomi", "lenovo", "asus", "sony"]
    
    def make_events(user_ids: Sequence[int], months: int, start_date: str) -> pd.DataFrame:
        rows = []
        base_date = pd.Timestamp(start_date)
        
        for user_id in user_ids:
            activity_level = rng.choice([0.5, 1.0, 2.0], p=[0.3, 0.5, 0.2])
            
            for month in range(months):
                month_start = base_date + pd.DateOffset(months=month)
                n_events = int(max(1, rng.poisson(15 * activity_level)))
                
                for _ in range(n_events):
                    offset_days = rng.uniform(0, 28)
                    event_time = month_start + timedelta(days=offset_days)
                    event_type = rng.choice(
                        ["view", "cart", "purchase"], p=[0.7, 0.2, 0.1]
                    )
                    price = (
                        float(max(0.0, rng.normal(80, 35)))
                        if event_type == "purchase"
                        else 0.0
                    )
                    
                    rows.append({
                        "event_time": event_time,
                        "event_type": event_type,
                        "user_id": int(user_id),
                        "category_code": rng.choice(categories),
                        "brand": rng.choice(brands),
                        "price": round(price, 2),
                        "user_session": f"{user_id}-{month}",
                    })
        
        return pd.DataFrame(rows)
    
    train_events = make_events(train_users, n_months, "2024-01-01")
    test_events = make_events(test_users, 1, "2024-04-01")
    
    train_features = aggregate_events(train_events)
    test_features = aggregate_events(test_events)
    
    train_features["churn"] = generate_churn_labels(train_features, rng)
    test_features["churn"] = generate_churn_labels(test_features, rng)
    
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    
    train_features.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    
    print(f"Generated {len(train_features)} train users -> {train_path}")
    print(f"Generated {len(test_features)} test users -> {test_path}")
    print(f"Train churn rate: {train_features['churn'].mean():.2%}")
    print(f"Test churn rate: {test_features['churn'].mean():.2%}")
    
    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_size": len(train_features),
        "test_size": len(test_features),
    }


def calculate_session_times(df: pd.DataFrame, session_gap_minutes: int = 30) -> pd.DataFrame:
    """Calculate actual session times from event timestamps.

    A new session starts when there's a gap of more than `session_gap_minutes`
    between consecutive events for the same user.

    Args:
        df: DataFrame with event_time and user_id columns
        session_gap_minutes: Minutes of inactivity to define a new session

    Returns:
        DataFrame with user_id and avg_session_time (in minutes)
    """
    df = df.sort_values(["user_id", "event_time"])

    # Calculate time difference between consecutive events per user
    df["time_diff"] = df.groupby("user_id")["event_time"].diff()

    # Mark new sessions (gap > threshold)
    gap_threshold = pd.Timedelta(minutes=session_gap_minutes)
    df["new_session"] = (df["time_diff"] > gap_threshold) | df["time_diff"].isna()

    # Assign session IDs
    df["session_id"] = df.groupby("user_id")["new_session"].cumsum()

    # Calculate session duration (time between first and last event in session)
    session_stats = df.groupby(["user_id", "session_id"]).agg(
        session_start=("event_time", "min"),
        session_end=("event_time", "max"),
        n_events=("event_time", "count"),
    ).reset_index()

    # Session duration in minutes
    session_stats["session_duration"] = (
        session_stats["session_end"] - session_stats["session_start"]
    ).dt.total_seconds() / 60

    # For single-event sessions, estimate duration based on event type (avg 2 minutes)
    session_stats.loc[session_stats["n_events"] == 1, "session_duration"] = 2.0

    # Average session time per user
    avg_session_time = session_stats.groupby("user_id").agg(
        avg_session_time=("session_duration", "mean"),
        total_sessions=("session_id", "count"),
    ).reset_index()

    # Clip to reasonable bounds (0-120 minutes)
    avg_session_time["avg_session_time"] = avg_session_time["avg_session_time"].clip(0, 120)

    return avg_session_time[["user_id", "avg_session_time"]]


def aggregate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw events into user-level features.

    Args:
        df: DataFrame with event data

    Returns:
        DataFrame with one row per user and aggregated features
    """
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])

    # Handle different column names (category_code vs category_id)
    category_col = "category_code" if "category_code" in df.columns else "category_id"

    agg = df.groupby("user_id").agg(
        clicks=("event_type", lambda x: (x == "view").sum()),
        purchases=("event_type", lambda x: (x == "purchase").sum()),
        total_spend=("price", "sum"),
        avg_order_value=("price", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        n_unique_categories=(category_col, "nunique"),
        n_unique_brands=("brand", "nunique"),
        first_event=("event_time", "min"),
        last_event=("event_time", "max"),
        n_events=("event_type", "count"),
    ).reset_index()

    now = df["event_time"].max()
    agg["days_since_last_visit"] = (now - agg["last_event"]).dt.days
    agg["account_age_days"] = (now - agg["first_event"]).dt.days

    # Calculate active_days (unique days with activity)
    active_days = df.groupby("user_id").agg(
        active_days=("event_time", lambda x: x.dt.date.nunique())
    ).reset_index()
    agg = agg.merge(active_days, on="user_id", how="left")

    # Calculate actual average session time from event data
    session_times = calculate_session_times(df)
    agg = agg.merge(session_times, on="user_id", how="left")

    # Fill missing session times with median (for users with insufficient data)
    median_session_time = agg["avg_session_time"].median()
    if pd.isna(median_session_time):
        median_session_time = 5.0  # Default 5 minutes if no data
    agg["avg_session_time"] = agg["avg_session_time"].fillna(median_session_time)

    # Drop intermediate columns
    agg = agg.drop(columns=["first_event", "last_event", "n_events"], errors="ignore")
    agg = agg.fillna(0)

    return agg


def generate_churn_labels(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    base_prob = 0.15
    
    probs = np.full(len(df), base_prob)
    
    probs += np.where(df["days_since_last_visit"] > 14, 0.3, 0)
    probs += np.where(df["purchases"] == 0, 0.2, 0)
    probs -= np.where(df["purchases"] >= 3, 0.1, 0)
    probs -= np.where(df["total_spend"] > 200, 0.1, 0)
    
    probs = np.clip(probs, 0.05, 0.85)
    
    return (rng.random(len(df)) < probs).astype(int)


def main():
    parser = argparse.ArgumentParser(
        description="Generate or process dataset for churn prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate demo data (synthetic)
  python -m src.data.make_dataset --demo --n-users 1000
  
  # Process real data from data/raw/
  python -m src.data.make_dataset --real
  
  # Process with sampling (faster)
  python -m src.data.make_dataset --real --sample 0.1 --max-users 5000
  
  # Process specific months
  python -m src.data.make_dataset --real --months 2019-Nov 2019-Dec
  
  # Auto mode: use real data if available, otherwise generate demo
  python -m src.data.make_dataset --auto
        """
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Generate synthetic demo data")
    mode.add_argument("--real", action="store_true", help="Process real data from data/raw/")
    mode.add_argument("--auto", action="store_true", help="Use real data if available, otherwise demo")
    
    # Common options
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Demo mode options
    parser.add_argument("--n-users", type=int, default=1000, help="Number of users (demo mode)")
    
    # Real data options
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--months", nargs="+", help="Months to process, e.g. 2019-Oct 2019-Nov")
    parser.add_argument("--sample", type=float, help="Sample fraction (0.0-1.0)")
    parser.add_argument("--max-users", type=int, help="Maximum users to include")
    parser.add_argument("--churn-days", type=int, default=14, help="Days threshold for churn label")
    
    args = parser.parse_args()
    
    if args.demo:
        generate_demo_data(
            n_users=args.n_users,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    
    elif args.real:
        process_real_data(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            months=args.months,
            sample_frac=args.sample,
            max_users=args.max_users,
            churn_days_threshold=args.churn_days,
            seed=args.seed,
        )
    
    elif args.auto:
        # Check if real data exists
        raw_path = Path(args.raw_dir)
        csv_files = list(raw_path.glob("*.csv")) if raw_path.exists() else []
        
        if csv_files:
            print(f"Found {len(csv_files)} CSV files in {raw_path}, processing real data...")
            process_real_data(
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                months=args.months,
                sample_frac=args.sample,
                max_users=args.max_users,
                churn_days_threshold=args.churn_days,
                seed=args.seed,
            )
        else:
            print(f"No CSV files in {raw_path}, generating demo data...")
            generate_demo_data(
                n_users=args.n_users,
                output_dir=args.output_dir,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()
