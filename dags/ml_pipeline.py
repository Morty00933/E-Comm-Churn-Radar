from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


def get_project_root() -> Path:
    candidates = [
        os.getenv("PROJECT_ROOT"),
        "/opt/airflow/app",
        Path(__file__).resolve().parents[1],
    ]
    for candidate in candidates:
        if candidate:
            p = Path(candidate)
            if p.exists() and (p / "src").exists():
                return p
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-model")

ETL_SCHEDULE = "@daily"
TRAINING_SCHEDULE = "@weekly"
BATCH_INFERENCE_SCHEDULE = "@daily"


def get_default_args() -> Dict:
    return {
        "owner": "mlops",
        "depends_on_past": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=2),
    }


with DAG(
    dag_id="etl_pipeline",
    description="ETL: Generate and validate data",
    default_args=get_default_args(),
    schedule_interval=ETL_SCHEDULE,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["etl", "data"],
) as etl_dag:
    
    @task(task_id="validate_config")
    def validate_config() -> str:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
        return str(CONFIG_PATH)
    
    generate_data = BashOperator(
        task_id="generate_data",
        bash_command=f"""
            set -e
            cd {PROJECT_ROOT}
            export PYTHONPATH={PROJECT_ROOT}:{PROJECT_ROOT}/src
            python -m src.data.make_dataset --demo --n-users 2000
        """,
    )
    
    @task(task_id="validate_data")
    def validate_data() -> Dict:
        import pandas as pd
        
        data_dir = PROJECT_ROOT / "data"
        train_path = data_dir / "train.csv"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Train data not found: {train_path}")
        
        train = pd.read_csv(train_path)
        
        if train.empty:
            raise ValueError("Train data is empty")
        
        if "churn" not in train.columns:
            raise ValueError("Missing 'churn' column")
        
        return {
            "train_rows": len(train),
            "churn_rate": float(train["churn"].mean()),
            "validated_at": datetime.utcnow().isoformat(),
        }
    
    config_task = validate_config()
    config_task >> generate_data >> validate_data()


with DAG(
    dag_id="training_pipeline",
    description="Train and register churn prediction model",
    default_args=get_default_args(),
    schedule_interval=TRAINING_SCHEDULE,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["training", "ml"],
) as training_dag:
    
    @task(task_id="prepare_environment")
    def prepare_environment() -> Dict:
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
        os.environ["MLFLOW_MODEL_NAME"] = MLFLOW_MODEL_NAME
        
        return {
            "project_root": str(PROJECT_ROOT),
            "config_path": str(CONFIG_PATH),
            "mlflow_uri": MLFLOW_TRACKING_URI,
        }
    
    @task(task_id="train_model")
    def train_model(env: Dict) -> Dict:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        
        from src.models.train import train_model as run_training
        
        result = run_training(
            config_path=env["config_path"],
            use_mlflow=True,
        )
        
        return {
            "algorithm": result.algorithm,
            "metrics": result.metrics,
            "training_time_sec": result.training_time_sec,
        }
    
    @task(task_id="validate_model")
    def validate_model(training_result: Dict) -> Dict:
        metrics = training_result.get("metrics", {})
        
        roc_auc = metrics.get("val_roc_auc") or metrics.get("train_roc_auc", 0)
        
        if roc_auc < 0.6:
            raise ValueError(f"Model quality too low: ROC-AUC = {roc_auc:.4f}")
        
        return {
            "status": "passed",
            "roc_auc": roc_auc,
            "algorithm": training_result.get("algorithm"),
        }
    
    env = prepare_environment()
    result = train_model(env)
    validate_model(result)


with DAG(
    dag_id="batch_inference",
    description="Batch scoring using production model",
    default_args=get_default_args(),
    schedule_interval=BATCH_INFERENCE_SCHEDULE,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["inference", "ml", "batch"],
) as batch_dag:
    
    @task(task_id="load_input_data")
    def load_input_data() -> str:
        import pandas as pd
        
        input_path = PROJECT_ROOT / "data" / "test.csv"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input data not found: {input_path}")
        
        df = pd.read_csv(input_path)
        
        if df.empty:
            raise ValueError("Input data is empty")
        
        return str(input_path)
    
    @task(task_id="run_predictions")
    def run_predictions(input_path: str) -> str:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        
        import pandas as pd
        from src.api.predictor import predict_customers
        from src.features.build_features import prepare_features
        
        df = pd.read_csv(input_path)
        df = prepare_features(df)
        
        records = df.to_dict(orient="records")
        predictions = predict_customers(records)
        
        output_df = pd.DataFrame(predictions)
        output_path = PROJECT_ROOT / "data" / "predictions.csv"
        output_df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    @task(task_id="log_stats")
    def log_stats(output_path: str) -> Dict:
        import pandas as pd
        import logging
        
        df = pd.read_csv(output_path)
        
        stats = {
            "total_scored": len(df),
            "predicted_churners": int((df["churn_label"] == 1).sum()),
            "churn_rate": float(df["churn_label"].mean()),
            "avg_probability": float(df["churn_proba"].mean()),
            "output_path": output_path,
        }
        
        logging.getLogger(__name__).info(f"Batch scoring complete: {stats}")
        
        return stats
    
    input_path = load_input_data()
    output_path = run_predictions(input_path)
    log_stats(output_path)


with DAG(
    dag_id="continuous_training",
    description="End-to-end pipeline: ETL -> Train -> Inference",
    default_args=get_default_args(),
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["orchestration", "ml"],
) as orchestrator_dag:
    
    trigger_etl = TriggerDagRunOperator(
        task_id="trigger_etl",
        trigger_dag_id="etl_pipeline",
        wait_for_completion=True,
        poke_interval=60,
    )
    
    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="training_pipeline",
        wait_for_completion=True,
        poke_interval=60,
    )
    
    trigger_inference = TriggerDagRunOperator(
        task_id="trigger_inference",
        trigger_dag_id="batch_inference",
        wait_for_completion=True,
        poke_interval=60,
    )
    
    @task(task_id="notify_completion")
    def notify_completion() -> str:
        import logging
        logging.getLogger(__name__).info("Continuous training pipeline completed")
        return "completed"
    
    trigger_etl >> trigger_training >> trigger_inference >> notify_completion()


with DAG(
    dag_id="etl_real_data",
    description="ETL: Process real Kaggle eCommerce data",
    default_args=get_default_args(),
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["etl", "data", "kaggle"],
    params={
        "sample_frac": 0.1,
        "max_users": 10000,
        "churn_days": 14,
    },
) as etl_real_dag:
    
    @task(task_id="check_raw_data")
    def check_raw_data() -> Dict:
        import glob
        
        raw_dir = PROJECT_ROOT / "data" / "raw"
        
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {raw_dir}\n"
                "Create directory and place CSV files there."
            )
        
        csv_files = sorted(glob.glob(str(raw_dir / "*.csv")))
        
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files in {raw_dir}\n"
                "Place eCommerce CSV files (2019-Oct.csv, etc.) in data/raw/"
            )
        
        return {
            "raw_dir": str(raw_dir),
            "files": [Path(f).name for f in csv_files],
            "file_count": len(csv_files),
        }
    
    @task(task_id="process_real_data")
    def process_real_data(raw_info: Dict, **context) -> Dict:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        
        from src.data.make_dataset import process_real_data as process_data
        
        params = context["params"]
        
        result = process_data(
            raw_dir=raw_info["raw_dir"],
            output_dir=str(PROJECT_ROOT / "data"),
            sample_frac=params.get("sample_frac"),
            max_users=params.get("max_users"),
            churn_days_threshold=params.get("churn_days", 14),
        )
        
        return result
    
    @task(task_id="validate_processed_data")
    def validate_processed_data(process_result: Dict) -> Dict:
        import pandas as pd
        
        train_path = process_result["train_path"]
        test_path = process_result["test_path"]
        
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        
        # Validation checks
        assert len(train) > 0, "Train data is empty"
        assert len(test) > 0, "Test data is empty"
        assert "churn" in train.columns, "Missing 'churn' column in train"
        assert "churn" in test.columns, "Missing 'churn' column in test"
        
        return {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_churn_rate": float(train["churn"].mean()),
            "test_churn_rate": float(test["churn"].mean()),
            "train_features": list(train.columns),
            "status": "validated",
        }
    
    raw_info = check_raw_data()
    process_result = process_real_data(raw_info)
    validate_processed_data(process_result)


with DAG(
    dag_id="train_on_real_data",
    description="Full pipeline: Process real data -> Train -> Evaluate",
    default_args=get_default_args(),
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["training", "ml", "kaggle"],
    params={
        "sample_frac": 0.1,
        "max_users": 10000,
        "churn_days": 14,
    },
) as train_real_dag:
    
    trigger_etl_real = TriggerDagRunOperator(
        task_id="trigger_etl_real",
        trigger_dag_id="etl_real_data",
        wait_for_completion=True,
        poke_interval=60,
        conf={
            "sample_frac": "{{ params.sample_frac }}",
            "max_users": "{{ params.max_users }}",
            "churn_days": "{{ params.churn_days }}",
        },
    )
    
    trigger_training_real = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="training_pipeline",
        wait_for_completion=True,
        poke_interval=60,
    )
    
    @task(task_id="log_completion")
    def log_completion() -> Dict:
        import logging
        from datetime import datetime
        
        logging.getLogger(__name__).info("Real data training pipeline completed")
        
        return {
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
        }
    
    trigger_etl_real >> trigger_training_real >> log_completion()


@dag(
    dag_id="hyperparameter_optimization",
    description="Run hyperparameter optimization with Optuna",
    schedule=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "hpo", "optuna"],
    params={
        "algorithm": "lightgbm",
        "n_trials": 50,
        "timeout": None,
    },
)
def hpo_pipeline():
    
    @task(task_id="check_data")
    def check_data() -> str:
        train_path = PROJECT_ROOT / "data" / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError("Training data not found. Run ETL first.")
        return str(train_path)
    
    @task(task_id="run_hpo")
    def run_hpo(train_path: str, **context) -> Dict:
        import pandas as pd
        from src.models.hpo import optimize_hyperparameters
        from src.features.build_features import split_xy
        
        params = context["params"]
        algorithm = params.get("algorithm", "lightgbm")
        n_trials = params.get("n_trials", 50)
        timeout = params.get("timeout")
        
        train_df = pd.read_csv(train_path)
        X, y = split_xy(train_df)
        
        result = optimize_hyperparameters(
            X, y,
            algorithm=algorithm,
            n_trials=n_trials,
            timeout=timeout,
        )
        
        return result.to_dict()
    
    @task(task_id="save_results")
    def save_results(hpo_result: Dict) -> str:
        import json
        
        output_path = PROJECT_ROOT / "models" / "hpo_result.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(hpo_result, f, indent=2)
        
        print(f"Best score: {hpo_result['best_score']:.4f}")
        print(f"Best params: {hpo_result['best_params']}")
        
        return str(output_path)
    
    @task(task_id="log_to_mlflow")
    def log_to_mlflow(hpo_result: Dict):
        import mlflow
        
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("churn-hpo")
        
        with mlflow.start_run(run_name=f"hpo_{hpo_result['algorithm']}"):
            mlflow.log_params(hpo_result["best_params"])
            mlflow.log_metric("best_score", hpo_result["best_score"])
            mlflow.log_metric("n_trials", hpo_result["n_trials"])
    
    train_path = check_data()
    result = run_hpo(train_path)
    save_results(result)
    log_to_mlflow(result)


hpo_dag = hpo_pipeline()
