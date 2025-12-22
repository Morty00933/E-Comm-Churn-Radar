import numpy as np
import pandas as pd
import pytest


class TestCleanEvents:
    def test_coerces_event_time(self):
        from src.data.cleaning import clean_events
        
        raw = pd.DataFrame({
            "event_time": ["2024-01-01T00:00:00Z", "2024-01-02T12:00:00Z"],
            "event_type": ["view", "purchase"],
            "user_id": [1, 2],
        })
        
        cleaned, report = clean_events(raw)
        
        assert pd.api.types.is_datetime64_any_dtype(cleaned["event_time"])

    def test_drops_invalid_user_id(self):
        from src.data.cleaning import clean_events
        
        raw = pd.DataFrame({
            "event_time": ["2024-01-01T00:00:00Z"] * 3,
            "event_type": ["view", "view", "view"],
            "user_id": [1, None, 3],
        })
        
        cleaned, report = clean_events(raw)
        
        assert len(cleaned) == 2
        assert report["dropped_rows"]["invalid_time_or_user"] == 1.0

    def test_fills_missing_event_type(self):
        from src.data.cleaning import clean_events
        
        raw = pd.DataFrame({
            "event_time": ["2024-01-01T00:00:00Z"],
            "event_type": [None],
            "user_id": [1],
        })
        
        cleaned, report = clean_events(raw)
        
        assert cleaned["event_type"].iloc[0] == "unknown"

    def test_fills_missing_price(self):
        from src.data.cleaning import clean_events
        
        raw = pd.DataFrame({
            "event_time": ["2024-01-01T00:00:00Z"],
            "event_type": ["view"],
            "user_id": [1],
            "price": [None],
        })
        
        cleaned, report = clean_events(raw)
        
        assert cleaned["price"].iloc[0] == 0.0


class TestMakeDataset:
    def test_generate_demo_data(self, tmp_path):
        from src.data.make_dataset import generate_demo_data
        
        result = generate_demo_data(
            n_users=50,
            n_months=2,
            output_dir=str(tmp_path),
            seed=42,
        )
        
        assert "train_path" in result
        assert "test_path" in result
        assert result["train_size"] == 50
        
        train_df = pd.read_csv(result["train_path"])
        assert "churn" in train_df.columns
        assert len(train_df) == 50

    def test_aggregate_events(self):
        from src.data.make_dataset import aggregate_events
        
        events = pd.DataFrame({
            "event_time": pd.date_range("2024-01-01", periods=10, freq="H"),
            "event_type": ["view"] * 7 + ["purchase"] * 3,
            "user_id": [1] * 5 + [2] * 5,
            "category_code": ["electronics"] * 10,
            "brand": ["samsung"] * 10,
            "price": [0.0] * 7 + [100.0] * 3,
            "user_session": ["s1"] * 5 + ["s2"] * 5,
        })
        
        features = aggregate_events(events)
        
        assert len(features) == 2
        assert "clicks" in features.columns
        assert "purchases" in features.columns
        assert "total_spend" in features.columns


class TestRealDataProcessing:
    """Tests for real Kaggle data processing."""
    
    def test_load_real_data_missing_dir(self, tmp_path):
        from src.data.make_dataset import load_real_data
        
        with pytest.raises(FileNotFoundError, match="Raw data directory not found"):
            load_real_data(raw_dir=str(tmp_path / "nonexistent"))
    
    def test_load_real_data_empty_dir(self, tmp_path):
        from src.data.make_dataset import load_real_data
        
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        
        with pytest.raises(FileNotFoundError, match="No CSV files found"):
            load_real_data(raw_dir=str(raw_dir))
    
    def test_load_real_data_with_sample_file(self, tmp_path):
        from src.data.make_dataset import load_real_data
        
        # Create sample CSV
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        
        sample_data = pd.DataFrame({
            "event_time": ["2019-12-01 00:00:00 UTC"] * 100,
            "event_type": ["view"] * 70 + ["cart"] * 20 + ["purchase"] * 10,
            "product_id": range(100),
            "category_id": [123] * 100,
            "category_code": ["electronics.phone"] * 100,
            "brand": ["samsung"] * 50 + ["apple"] * 50,
            "price": [0.0] * 90 + [99.99] * 10,
            "user_id": [1] * 30 + [2] * 40 + [3] * 30,
            "user_session": ["s1"] * 30 + ["s2"] * 40 + ["s3"] * 30,
        })
        sample_data.to_csv(raw_dir / "2019-Dec.csv", index=False)
        
        # Load data
        events = load_real_data(raw_dir=str(raw_dir))
        
        assert len(events) == 100
        assert "event_time" in events.columns
        assert "user_id" in events.columns
    
    def test_process_real_data_full_pipeline(self, tmp_path):
        from src.data.make_dataset import process_real_data
        
        # Create sample data spanning 30 days
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        
        dates = pd.date_range("2019-12-01", periods=30, freq="D")
        events = []
        
        for i, date in enumerate(dates):
            for user_id in range(1, 51):  # 50 users
                n_events = np.random.randint(1, 10)
                for _ in range(n_events):
                    events.append({
                        "event_time": date + pd.Timedelta(hours=np.random.randint(0, 24)),
                        "event_type": np.random.choice(["view", "cart", "purchase"], p=[0.7, 0.2, 0.1]),
                        "product_id": np.random.randint(1000, 9999),
                        "category_id": np.random.randint(100, 999),
                        "category_code": "electronics.phone",
                        "brand": np.random.choice(["samsung", "apple", "xiaomi"]),
                        "price": np.random.uniform(0, 200) if np.random.random() > 0.9 else 0.0,
                        "user_id": user_id,
                        "user_session": f"s{user_id}_{i}",
                    })
        
        df = pd.DataFrame(events)
        df.to_csv(raw_dir / "2019-Dec.csv", index=False)
        
        # Process data
        result = process_real_data(
            raw_dir=str(raw_dir),
            output_dir=str(tmp_path / "processed"),
            max_users=30,
            churn_days_threshold=7,
            seed=42,
        )
        
        assert "train_path" in result
        assert "test_path" in result
        assert result["train_size"] > 0
        assert result["test_size"] > 0
        assert 0 <= result["train_churn_rate"] <= 1
        
        # Verify files exist
        train_df = pd.read_csv(result["train_path"])
        test_df = pd.read_csv(result["test_path"])
        
        assert "churn" in train_df.columns
        assert "churn" in test_df.columns
        assert len(train_df) + len(test_df) == 30  # max_users
