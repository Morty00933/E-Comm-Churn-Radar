"""Model explainability using SHAP."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelExplainer:
    """SHAP-based model explainer."""
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_data: Optional[pd.DataFrame] = None,
        max_background_samples: int = 100,
    ):
        if not SHAP_AVAILABLE:
            raise ImportError("shap package not installed. Run: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer = None
        self.max_background_samples = max_background_samples
        
        self._init_explainer()
    
    def _init_explainer(self) -> None:
        """Initialize SHAP explainer based on model type."""
        model_type = type(self.model).__name__.lower()
        
        if "lightgbm" in model_type or "lgbm" in model_type:
            self.explainer = shap.TreeExplainer(self.model)
        elif "xgboost" in model_type or "xgb" in model_type:
            self.explainer = shap.TreeExplainer(self.model)
        elif "randomforest" in model_type or "gradientboosting" in model_type:
            self.explainer = shap.TreeExplainer(self.model)
        elif self.background_data is not None:
            # Use KernelExplainer for other models
            background = self.background_data
            if len(background) > self.max_background_samples:
                background = background.sample(
                    n=self.max_background_samples,
                    random_state=42
                )
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background[self.feature_names].values,
            )
        else:
            raise ValueError(
                "Cannot create explainer: unsupported model type and no background data"
            )
    
    def explain(
        self,
        X: pd.DataFrame,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate SHAP explanations for predictions.
        
        Args:
            X: Features DataFrame
            top_k: Number of top features to return
        
        Returns:
            List of explanation dicts per sample
        """
        X_values = X[self.feature_names].values
        
        shap_values = self.explainer.shap_values(X_values)
        
        # Handle different SHAP output formats
        # For binary classification, we want positive class SHAP values
        if isinstance(shap_values, list):
            # List of arrays per class - take positive class
            shap_values = np.array(shap_values[1])
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                # Shape: (samples, features, classes) - take positive class
                shap_values = shap_values[:, :, 1]
            # If 2D, use as-is (samples, features)
        
        # Ensure 2D shape
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        
        explanations = []
        for i in range(len(X)):
            sample_shap = shap_values[i].flatten()  # Ensure 1D
            sample_features = X_values[i].flatten()
            
            # Sort by absolute SHAP value
            sorted_indices = np.argsort(np.abs(sample_shap))[::-1][:top_k]
            
            contributions = []
            for idx in sorted_indices:
                idx_int = int(idx)  # Convert numpy int to Python int
                contributions.append({
                    "feature": self.feature_names[idx_int],
                    "value": float(sample_features[idx_int]),
                    "shap_value": float(sample_shap[idx_int]),
                    "impact": "increases" if sample_shap[idx_int] > 0 else "decreases",
                })
            
            # Handle expected_value which can be array or scalar
            expected_val = self.explainer.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                if len(expected_val) > 1:
                    base_value = float(expected_val[1])  # Positive class
                else:
                    base_value = float(expected_val[0])
            else:
                base_value = float(expected_val)
            
            explanations.append({
                "base_value": base_value,
                "prediction_contribution": float(np.sum(sample_shap)),
                "top_features": contributions,
            })
        
        return explanations

    async def explain_async(
        self,
        X: pd.DataFrame,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Run explain() in a thread pool to avoid blocking the FastAPI event loop."""
        return await asyncio.to_thread(self.explain, X, top_k)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get global feature importance from SHAP values.
        
        Requires background data to be set.
        """
        if self.background_data is None:
            raise ValueError("Background data required for global importance")
        
        X = self.background_data[self.feature_names]
        if len(X) > self.max_background_samples:
            X = X.sample(n=self.max_background_samples, random_state=42)
        
        shap_values = self.explainer.shap_values(X.values)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        importance = np.abs(shap_values).mean(axis=0)
        
        return {
            name: float(imp)
            for name, imp in sorted(
                zip(self.feature_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        }


def generate_explanation_text(explanation: Dict[str, Any]) -> str:
    """Generate human-readable explanation text."""
    lines = ["The model prediction is influenced by:"]
    
    for feat in explanation["top_features"][:3]:
        direction = "↑" if feat["impact"] == "increases" else "↓"
        lines.append(
            f"  {direction} {feat['feature']} = {feat['value']:.2f} "
            f"({feat['impact']} churn probability)"
        )
    
    return "\n".join(lines)


def save_explanation(
    explanation: Dict[str, Any],
    output_path: str,
) -> None:
    """Save explanation to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(explanation, f, indent=2)
