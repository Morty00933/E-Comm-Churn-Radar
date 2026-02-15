"""Model Card generation following ML Model Cards best practices."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ModelCard:
    """ML Model Card following industry best practices.
    
    Reference: https://arxiv.org/abs/1810.03993
    """
    
    # Model Details
    model_name: str = "Churn Prediction Model"
    model_version: str = "1.0.0"
    model_type: str = "Binary Classification"
    algorithm: str = "LightGBM"
    framework: str = "scikit-learn compatible"
    
    # Dates
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Developers
    developers: List[str] = field(default_factory=lambda: ["ML Team"])
    contact: str = "mlops@example.com"
    
    # Intended Use
    primary_use: str = "Predict customer churn probability for retention campaigns"
    primary_users: List[str] = field(default_factory=lambda: [
        "Marketing team",
        "Customer success",
        "Data analysts",
    ])
    out_of_scope_uses: List[str] = field(default_factory=lambda: [
        "Credit scoring",
        "Fraud detection",
        "Individual customer decisions without human review",
    ])
    
    # Training Data
    training_data_description: str = "eCommerce customer behavior data"
    training_data_size: Optional[int] = None
    training_date_range: Optional[str] = None
    preprocessing_steps: List[str] = field(default_factory=lambda: [
        "Feature aggregation from event logs",
        "Outlier clipping at 99th percentile",
        "Missing value imputation with zeros",
    ])
    
    # Evaluation Data
    evaluation_data_description: str = "Held-out test set (20% of data)"
    evaluation_data_size: Optional[int] = None
    
    # Features
    features: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    
    # Performance by segment
    performance_by_segment: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Ethical Considerations
    ethical_considerations: List[str] = field(default_factory=lambda: [
        "Model should not be used as sole factor for customer decisions",
        "Predictions should be reviewed by human operators",
        "Model may have lower accuracy for new customers with limited history",
    ])
    
    # Limitations
    limitations: List[str] = field(default_factory=lambda: [
        "Trained on historical data which may not reflect future patterns",
        "Performance may degrade over time due to data drift",
        "Limited to features available at prediction time",
        "May underperform for customers with unusual behavior patterns",
    ])
    
    # Caveats and Recommendations
    caveats: List[str] = field(default_factory=lambda: [
        "Retrain model periodically (recommended: monthly)",
        "Monitor prediction distribution for drift",
        "Validate predictions against actual churn outcomes",
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_details": {
                "name": self.model_name,
                "version": self.model_version,
                "type": self.model_type,
                "algorithm": self.algorithm,
                "framework": self.framework,
                "created_at": self.created_at,
                "last_updated": self.last_updated,
                "developers": self.developers,
                "contact": self.contact,
            },
            "intended_use": {
                "primary_use": self.primary_use,
                "primary_users": self.primary_users,
                "out_of_scope_uses": self.out_of_scope_uses,
            },
            "training_data": {
                "description": self.training_data_description,
                "size": self.training_data_size,
                "date_range": self.training_date_range,
                "preprocessing": self.preprocessing_steps,
            },
            "evaluation": {
                "data_description": self.evaluation_data_description,
                "data_size": self.evaluation_data_size,
                "threshold": self.threshold,
                "metrics": self.metrics,
            },
            "features": {
                "list": self.features,
                "importance": self.feature_importance,
            },
            "performance_by_segment": self.performance_by_segment,
            "ethical_considerations": self.ethical_considerations,
            "limitations": self.limitations,
            "caveats": self.caveats,
        }
    
    def to_markdown(self) -> str:
        """Generate Markdown Model Card."""
        lines = [
            f"# Model Card: {self.model_name}",
            "",
            "## Model Details",
            "",
            f"- **Model Version**: {self.model_version}",
            f"- **Model Type**: {self.model_type}",
            f"- **Algorithm**: {self.algorithm}",
            f"- **Framework**: {self.framework}",
            f"- **Created**: {self.created_at}",
            f"- **Last Updated**: {self.last_updated}",
            f"- **Developers**: {', '.join(self.developers)}",
            f"- **Contact**: {self.contact}",
            "",
            "## Intended Use",
            "",
            f"**Primary Use**: {self.primary_use}",
            "",
            "**Primary Users**:",
        ]
        
        for user in self.primary_users:
            lines.append(f"- {user}")
        
        lines.extend([
            "",
            "**Out of Scope Uses**:",
        ])
        
        for use in self.out_of_scope_uses:
            lines.append(f"- {use}")
        
        lines.extend([
            "",
            "## Training Data",
            "",
            f"- **Description**: {self.training_data_description}",
            f"- **Size**: {self.training_data_size or 'N/A'} samples",
            f"- **Date Range**: {self.training_date_range or 'N/A'}",
            "",
            "**Preprocessing Steps**:",
        ])
        
        for step in self.preprocessing_steps:
            lines.append(f"1. {step}")
        
        lines.extend([
            "",
            "## Evaluation",
            "",
            f"- **Test Data**: {self.evaluation_data_description}",
            f"- **Test Size**: {self.evaluation_data_size or 'N/A'} samples",
            f"- **Classification Threshold**: {self.threshold}",
            "",
            "### Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])
        
        for name, value in self.metrics.items():
            lines.append(f"| {name} | {value:.4f} |")
        
        if self.features:
            lines.extend([
                "",
                "## Features",
                "",
                "| Feature | Importance |",
                "|---------|------------|",
            ])
            
            for feat in self.features:
                imp = self.feature_importance.get(feat, 0)
                lines.append(f"| {feat} | {imp:.4f} |")
        
        if self.performance_by_segment:
            lines.extend([
                "",
                "## Performance by Segment",
                "",
            ])
            
            for segment, metrics in self.performance_by_segment.items():
                lines.append(f"### {segment}")
                lines.append("")
                for name, value in metrics.items():
                    lines.append(f"- {name}: {value:.4f}")
                lines.append("")
        
        lines.extend([
            "",
            "## Ethical Considerations",
            "",
        ])
        
        for consideration in self.ethical_considerations:
            lines.append(f"- {consideration}")
        
        lines.extend([
            "",
            "## Limitations",
            "",
        ])
        
        for limitation in self.limitations:
            lines.append(f"- {limitation}")
        
        lines.extend([
            "",
            "## Caveats and Recommendations",
            "",
        ])
        
        for caveat in self.caveats:
            lines.append(f"- {caveat}")
        
        return "\n".join(lines)
    
    def save(self, output_dir: str = "models") -> Dict[str, str]:
        """Save Model Card as JSON and Markdown."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # JSON
        json_path = output_path / "model_card.json"
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Markdown
        md_path = output_path / "MODEL_CARD.md"
        with open(md_path, "w") as f:
            f.write(self.to_markdown())
        
        return {
            "json": str(json_path),
            "markdown": str(md_path),
        }


def generate_model_card(
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: Dict[str, float],
    feature_columns: List[str],
    algorithm: str = "LightGBM",
    version: str = "1.0.0",
) -> ModelCard:
    """Generate Model Card from training artifacts."""
    
    # Feature importance
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        for name, imp in zip(feature_columns, model.feature_importances_):
            feature_importance[name] = float(imp)
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
    
    # Date range
    date_range = None
    if "event_time" in train_df.columns:
        try:
            times = pd.to_datetime(train_df["event_time"])
            date_range = f"{times.min().date()} to {times.max().date()}"
        except (ValueError, TypeError):
            date_range = "Unable to parse event_time"
    
    # Performance by segment (example: by customer tenure)
    performance_by_segment = {}
    if "account_age_days" in test_df.columns and "churn" in test_df.columns:
        try:
            from sklearn.metrics import roc_auc_score

            # New customers (< 30 days)
            new_mask = test_df["account_age_days"] < 30
            if new_mask.sum() > 10:
                new_y = test_df.loc[new_mask, "churn"]
                new_X = test_df.loc[new_mask, feature_columns]
                new_proba = model.predict_proba(new_X)[:, 1]
                performance_by_segment["New Customers (<30 days)"] = {
                    "count": int(new_mask.sum()),
                    "roc_auc": float(roc_auc_score(new_y, new_proba)),
                }

            # Established customers (>= 30 days)
            est_mask = test_df["account_age_days"] >= 30
            if est_mask.sum() > 10:
                est_y = test_df.loc[est_mask, "churn"]
                est_X = test_df.loc[est_mask, feature_columns]
                est_proba = model.predict_proba(est_X)[:, 1]
                performance_by_segment["Established Customers (>=30 days)"] = {
                    "count": int(est_mask.sum()),
                    "roc_auc": float(roc_auc_score(est_y, est_proba)),
                }
        except (ValueError, KeyError) as e:
            # Insufficient data for segment analysis or missing columns
            performance_by_segment["error"] = f"Could not compute segment performance: {e}"
    
    return ModelCard(
        model_version=version,
        algorithm=algorithm,
        training_data_size=len(train_df),
        training_date_range=date_range,
        evaluation_data_size=len(test_df),
        features=feature_columns,
        feature_importance=feature_importance,
        metrics=metrics,
        performance_by_segment=performance_by_segment,
    )
