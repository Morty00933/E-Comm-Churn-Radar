from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold


@dataclass
class TrainingResult:
    model: BaseEstimator
    algorithm: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    training_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "params": self.params,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "training_time_sec": self.training_time_sec,
        }


class BaseAlgorithm(ABC):
    name: str = "base"
    
    @abstractmethod
    def create_model(self, params: Dict[str, Any]) -> BaseEstimator:
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        pass
    
    def get_feature_importance(
        self, model: BaseEstimator, feature_names: List[str]
    ) -> Dict[str, float]:
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance.tolist()))
        return {}


class LightGBMAlgorithm(BaseAlgorithm):
    name = "lightgbm"
    
    def create_model(self, params: Dict[str, Any]) -> BaseEstimator:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            "objective": "binary",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }


class XGBoostAlgorithm(BaseAlgorithm):
    name = "xgboost"
    
    def create_model(self, params: Dict[str, Any]) -> BaseEstimator:
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            "objective": "binary:logistic",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "use_label_encoder": False,
            "verbosity": 0,
        }


class RandomForestAlgorithm(BaseAlgorithm):
    name = "random_forest"
    
    def create_model(self, params: Dict[str, Any]) -> BaseEstimator:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }


class LogRegAlgorithm(BaseAlgorithm):
    name = "logreg"
    
    def create_model(self, params: Dict[str, Any]) -> BaseEstimator:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**params)),
        ])
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }


ALGORITHMS: Dict[str, Type[BaseAlgorithm]] = {
    "lightgbm": LightGBMAlgorithm,
    "xgboost": XGBoostAlgorithm,
    "random_forest": RandomForestAlgorithm,
    "logreg": LogRegAlgorithm,
}


def get_algorithm(name: str) -> BaseAlgorithm:
    if name not in ALGORITHMS:
        available = ", ".join(ALGORITHMS.keys())
        raise ValueError(f"Unknown algorithm: {name}. Available: {available}")
    return ALGORITHMS[name]()


def auto_select_algorithm() -> BaseAlgorithm:
    for name in ["lightgbm", "xgboost", "random_forest"]:
        try:
            algo = get_algorithm(name)
            algo.create_model(algo.get_default_params())
            return algo
        except ImportError:
            continue
    
    return get_algorithm("random_forest")


class Trainer:
    def __init__(
        self,
        algorithm: str = "auto",
        params: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        if algorithm == "auto":
            self.algo = auto_select_algorithm()
        else:
            self.algo = get_algorithm(algorithm)
        
        self.user_params = params or {}
        self.cv_folds = cv_folds
        self.random_state = random_state
        self._feature_names: List[str] = []
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> TrainingResult:
        start_time = time.time()
        
        self._feature_names = list(X_train.columns)
        
        params = {**self.algo.get_default_params(), **self.user_params}
        model = self.algo.create_model(params)
        
        model.fit(X_train, y_train)
        
        metrics = self._evaluate(model, X_train, y_train, X_val, y_val)
        
        feature_importance = self.algo.get_feature_importance(
            model, self._feature_names
        )
        
        training_time = time.time() - start_time
        
        return TrainingResult(
            model=model,
            algorithm=self.algo.name,
            params=params,
            metrics=metrics,
            feature_importance=feature_importance,
            training_time_sec=training_time,
        )
    
    def _evaluate(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> Dict[str, float]:
        from sklearn.metrics import (
            roc_auc_score, f1_score, precision_score, recall_score
        )
        
        metrics = {}
        
        train_proba = model.predict_proba(X_train)[:, 1]
        train_pred = (train_proba >= 0.5).astype(int)
        
        metrics["train_roc_auc"] = float(roc_auc_score(y_train, train_proba))
        metrics["train_f1"] = float(f1_score(y_train, train_pred))
        metrics["train_precision"] = float(precision_score(y_train, train_pred))
        metrics["train_recall"] = float(recall_score(y_train, train_pred))
        
        if X_val is not None and y_val is not None:
            val_proba = model.predict_proba(X_val)[:, 1]
            val_pred = (val_proba >= 0.5).astype(int)
            
            metrics["val_roc_auc"] = float(roc_auc_score(y_val, val_proba))
            metrics["val_f1"] = float(f1_score(y_val, val_pred))
            metrics["val_precision"] = float(precision_score(y_val, val_pred))
            metrics["val_recall"] = float(recall_score(y_val, val_pred))
            
            best_threshold = self._find_optimal_threshold(y_val, val_proba)
            metrics["best_threshold"] = float(best_threshold)
        
        return metrics
    
    def _find_optimal_threshold(
        self, y_true: pd.Series, proba: np.ndarray
    ) -> float:
        from sklearn.metrics import f1_score
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            pred = (proba >= threshold).astype(int)
            f1 = f1_score(y_true, pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
