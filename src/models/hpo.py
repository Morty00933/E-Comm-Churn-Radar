"""Hyperparameter optimization with Optuna."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class HPOResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    study_name: str
    algorithm: str
    optimization_history: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "study_name": self.study_name,
            "algorithm": self.algorithm,
            "optimization_history": self.optimization_history,
        }
    
    def save(self, path: str) -> None:
        """Save HPO result to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def get_lightgbm_search_space(trial: "optuna.Trial") -> Dict[str, Any]:
    """Define LightGBM hyperparameter search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def get_xgboost_search_space(trial: "optuna.Trial") -> Dict[str, Any]:
    """Define XGBoost hyperparameter search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def get_random_forest_search_space(trial: "optuna.Trial") -> Dict[str, Any]:
    """Define Random Forest hyperparameter search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


SEARCH_SPACES = {
    "lightgbm": get_lightgbm_search_space,
    "xgboost": get_xgboost_search_space,
    "random_forest": get_random_forest_search_space,
}


def create_objective(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm: str,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> Callable:
    """Create Optuna objective function."""
    
    def objective(trial: "optuna.Trial") -> float:
        # Get search space
        search_space_fn = SEARCH_SPACES.get(algorithm)
        if search_space_fn is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        params = search_space_fn(trial)
        
        # Create model
        if algorithm == "lightgbm":
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(**params, random_state=42, verbosity=-1)
        elif algorithm == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(**params, random_state=42, verbosity=0)
        elif algorithm == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        return scores.mean()
    
    return objective


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm: str = "lightgbm",
    n_trials: int = 50,
    timeout: Optional[int] = None,
    cv: int = 5,
    scoring: str = "roc_auc",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    n_jobs: int = 1,
) -> HPOResult:
    """Run hyperparameter optimization.
    
    Args:
        X: Features DataFrame
        y: Target Series
        algorithm: Algorithm to optimize (lightgbm, xgboost, random_forest)
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        cv: Number of cross-validation folds
        scoring: Scoring metric
        study_name: Optuna study name
        storage: Optuna storage URL (e.g., sqlite:///optuna.db)
        n_jobs: Number of parallel jobs
    
    Returns:
        HPOResult with best parameters and optimization history
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna not installed. Run: pip install optuna")
    
    study_name = study_name or f"churn_{algorithm}"
    
    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    
    # Create objective
    objective = create_objective(X, y, algorithm, cv, scoring)
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    
    # Build optimization history
    history = []
    for trial in study.trials:
        history.append({
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": str(trial.state),
        })
    
    return HPOResult(
        best_params=study.best_params,
        best_score=study.best_value,
        n_trials=len(study.trials),
        study_name=study_name,
        algorithm=algorithm,
        optimization_history=history,
    )


def log_hpo_to_mlflow(result: HPOResult, experiment_name: str = "churn-hpo") -> None:
    """Log HPO results to MLflow."""
    import mlflow
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"hpo_{result.algorithm}"):
        # Log best params
        mlflow.log_params(result.best_params)
        
        # Log metrics
        mlflow.log_metric("best_score", result.best_score)
        mlflow.log_metric("n_trials", result.n_trials)
        
        # Log history as artifact
        history_path = f"/tmp/hpo_history_{result.algorithm}.json"
        with open(history_path, "w") as f:
            json.dump(result.optimization_history, f, indent=2)
        mlflow.log_artifact(history_path)


# CLI entry point
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument("--algorithm", default="lightgbm", choices=["lightgbm", "xgboost", "random_forest"])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--output", default="models/hpo_result.json")
    parser.add_argument("--mlflow", action="store_true", help="Log to MLflow")
    
    args = parser.parse_args()
    
    # Load data
    train_path = Path("data/train.csv")
    if not train_path.exists():
        print("No training data found. Run: make data")
        return
    
    train_df = pd.read_csv(train_path)
    
    from src.features.build_features import split_xy
    X, y = split_xy(train_df)
    
    print(f"Running HPO for {args.algorithm} with {args.n_trials} trials...")
    
    result = optimize_hyperparameters(
        X, y,
        algorithm=args.algorithm,
        n_trials=args.n_trials,
        timeout=args.timeout,
        cv=args.cv,
    )
    
    print(f"\nBest score: {result.best_score:.4f}")
    print(f"Best params: {result.best_params}")
    
    result.save(args.output)
    print(f"\nSaved to {args.output}")
    
    if args.mlflow:
        log_hpo_to_mlflow(result)
        print("Logged to MLflow")


if __name__ == "__main__":
    main()
