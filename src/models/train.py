from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from src.common.logging import get_logger
from src.features.build_features import prepare_features, split_xy, get_feature_names
from .trainer import Trainer, TrainingResult

logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_model(
    config_path: str = "configs/config.yaml",
    use_mlflow: bool = True,
) -> TrainingResult:
    logger.info(f"Loading config from {config_path}")
    cfg = load_config(config_path)
    
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    artifacts_cfg = cfg.get("artifacts", {})
    mlflow_cfg = cfg.get("mlflow", {})
    
    train_path = data_cfg.get("train_path", "data/train.csv")
    test_path = data_cfg.get("test_path", "data/test.csv")
    
    logger.info(f"Loading train data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    test_df = None
    if os.path.exists(test_path):
        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)
    
    clip_q = cfg.get("features", {}).get("clip_quantile", 0.99)
    train_df = prepare_features(train_df, clip_quantile=clip_q)
    if test_df is not None:
        test_df = prepare_features(test_df, clip_quantile=clip_q)
    
    X_train, y_train = split_xy(train_df)
    
    X_val, y_val = None, None
    if test_df is not None:
        X_val, y_val = split_xy(test_df)
    
    logger.info(f"Training set: {len(X_train)} samples, {X_train.shape[1]} features")
    logger.info(f"Churn rate: {y_train.mean():.2%}")
    
    algorithm = model_cfg.get("algorithm", "auto")
    candidates = model_cfg.get("candidates", {})
    
    params = {}
    if algorithm != "auto" and algorithm in candidates:
        params = candidates[algorithm]
    
    trainer = Trainer(
        algorithm=algorithm,
        params=params,
        cv_folds=model_cfg.get("cv_folds", 5),
    )
    
    logger.info(f"Training with algorithm: {trainer.algo.name}")
    result = trainer.train(X_train, y_train, X_val, y_val)
    
    logger.info(f"Training complete in {result.training_time_sec:.1f}s")
    logger.info(f"Train ROC-AUC: {result.metrics.get('train_roc_auc', 0):.4f}")
    if "val_roc_auc" in result.metrics:
        logger.info(f"Val ROC-AUC: {result.metrics['val_roc_auc']:.4f}")
    
    save_artifacts(result, artifacts_cfg, get_feature_names(train_df))
    
    if use_mlflow:
        log_to_mlflow(result, mlflow_cfg, train_df)
    
    return result


def save_artifacts(
    result: TrainingResult,
    artifacts_cfg: Dict[str, Any],
    feature_names: list,
) -> None:
    model_path = Path(artifacts_cfg.get("model_path", "models/model.pkl"))
    metrics_path = Path(artifacts_cfg.get("metrics_path", "models/metrics.json"))
    features_path = Path(artifacts_cfg.get("feature_columns_path", "models/feature_columns.json"))
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, "wb") as f:
        pickle.dump(result.model, f)
    logger.info(f"Model saved to {model_path}")
    
    with open(metrics_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    with open(features_path, "w") as f:
        json.dump({"features": feature_names}, f, indent=2)
    logger.info(f"Features saved to {features_path}")


def log_to_mlflow(
    result: TrainingResult,
    mlflow_cfg: Dict[str, Any],
    train_df: pd.DataFrame,
) -> Optional[str]:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("MLflow not available, skipping")
        return None
    
    # Environment variable takes precedence over config
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        mlflow_cfg.get("tracking_uri", "http://localhost:5000")
    )
    experiment_name = mlflow_cfg.get("experiment_name", "churn-prediction")
    model_name = os.getenv("MLFLOW_MODEL_NAME", mlflow_cfg.get("model_name", "churn-model"))
    register_stage = mlflow_cfg.get("register_stage", "Staging")
    
    logger.info(f"Connecting to MLflow at {tracking_uri}")
    
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Could not connect to MLflow: {e}")
        return None
    
    with mlflow.start_run() as run:
        mlflow.log_params(result.params)
        mlflow.log_metrics(result.metrics)
        
        mlflow.log_param("algorithm", result.algorithm)
        mlflow.log_param("n_features", train_df.shape[1])
        mlflow.log_param("n_samples", len(train_df))
        
        mlflow.sklearn.log_model(
            result.model,
            "model",
            registered_model_name=model_name,
        )
        
        if result.feature_importance:
            importance_df = pd.DataFrame([
                {"feature": k, "importance": v}
                for k, v in sorted(
                    result.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ])
            mlflow.log_table(importance_df, "feature_importance.json")
        
        logger.info(f"MLflow run ID: {run.info.run_id}")
        
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=["None"])
            if versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=versions[0].version,
                    stage=register_stage,
                )
                logger.info(f"Model promoted to {register_stage}")
        except Exception as e:
            logger.warning(f"Could not promote model: {e}")
        
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    
    args = parser.parse_args()
    
    train_model(
        config_path=args.config,
        use_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
