"""A/B Testing for model comparison and gradual rollout."""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Experiment:
    """A/B experiment configuration."""
    name: str
    control_model: str  # Model name/version for control
    treatment_model: str  # Model name/version for treatment
    traffic_split: float = 0.5  # Fraction of traffic to treatment
    enabled: bool = True
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        if not self.enabled:
            return False
        
        now = datetime.utcnow()
        
        if self.start_time:
            start = datetime.fromisoformat(self.start_time)
            if now < start:
                return False
        
        if self.end_time:
            end = datetime.fromisoformat(self.end_time)
            if now > end:
                return False
        
        return True


@dataclass
class ExperimentResult:
    """Result of an A/B experiment assignment."""
    experiment_name: str
    variant: str  # "control" or "treatment"
    model_name: str
    user_id: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ABTester:
    """A/B Testing manager for model rollouts.
    
    Features:
    - Consistent user assignment (same user always gets same variant)
    - Traffic splitting
    - Shadow mode (run both models, use control)
    - Gradual rollout
    """
    
    def __init__(
        self,
        experiments: Optional[List[Experiment]] = None,
        redis_url: Optional[str] = None,
    ):
        self.experiments: Dict[str, Experiment] = {}
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self._redis = None
        
        if experiments:
            for exp in experiments:
                self.register_experiment(exp)
    
    @property
    def redis(self):
        """Lazy Redis client for result logging."""
        if self._redis is None and self.redis_url:
            try:
                import redis
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis not available for A/B logging: {e}")
        return self._redis
    
    def register_experiment(self, experiment: Experiment) -> None:
        """Register an experiment."""
        self.experiments[experiment.name] = experiment
        logger.info(f"Registered experiment: {experiment.name}")
    
    def get_variant(
        self,
        experiment_name: str,
        user_id: int,
    ) -> Optional[ExperimentResult]:
        """Assign user to experiment variant.
        
        Uses consistent hashing to ensure same user always gets same variant.
        """
        experiment = self.experiments.get(experiment_name)
        
        if not experiment or not experiment.is_active():
            return None
        
        # Consistent hashing for user assignment
        hash_input = f"{experiment_name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 100) / 100.0
        
        if bucket < experiment.traffic_split:
            variant = "treatment"
            model_name = experiment.treatment_model
        else:
            variant = "control"
            model_name = experiment.control_model
        
        result = ExperimentResult(
            experiment_name=experiment_name,
            variant=variant,
            model_name=model_name,
            user_id=user_id,
        )
        
        # Log assignment
        self._log_assignment(result)
        
        return result
    
    def _log_assignment(self, result: ExperimentResult) -> None:
        """Log experiment assignment to Redis."""
        if self.redis:
            try:
                key = f"ab:{result.experiment_name}:{result.variant}"
                self.redis.hincrby(key, "count", 1)
                
                # Also log individual assignment
                detail_key = f"ab:assignment:{result.experiment_name}:{result.user_id}"
                self.redis.setex(detail_key, 86400, result.variant)  # 24h TTL
                
            except Exception as e:
                logger.warning(f"Failed to log A/B assignment: {e}")
    
    def log_outcome(
        self,
        experiment_name: str,
        user_id: int,
        outcome: str,  # e.g., "churn", "no_churn", "converted"
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log experiment outcome for analysis."""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return
        
        # Get user's variant
        result = self.get_variant(experiment_name, user_id)
        if not result:
            return
        
        if self.redis:
            try:
                key = f"ab:{experiment_name}:{result.variant}:outcomes:{outcome}"
                self.redis.hincrby(key, "count", 1)
                
                if metadata:
                    # Log metadata for detailed analysis
                    detail_key = f"ab:outcome:{experiment_name}:{user_id}"
                    self.redis.setex(
                        detail_key,
                        86400 * 7,  # 7 days
                        json.dumps({
                            "variant": result.variant,
                            "outcome": outcome,
                            "metadata": metadata,
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                    )
            except Exception as e:
                logger.warning(f"Failed to log outcome: {e}")
    
    def get_results(self, experiment_name: str) -> Dict[str, Any]:
        """Get experiment results summary."""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return {"error": "Experiment not found"}
        
        results = {
            "experiment": experiment_name,
            "control_model": experiment.control_model,
            "treatment_model": experiment.treatment_model,
            "traffic_split": experiment.traffic_split,
            "is_active": experiment.is_active(),
            "control": {"count": 0, "outcomes": {}},
            "treatment": {"count": 0, "outcomes": {}},
        }
        
        if self.redis:
            try:
                for variant in ["control", "treatment"]:
                    # Get counts
                    key = f"ab:{experiment_name}:{variant}"
                    count = self.redis.hget(key, "count")
                    results[variant]["count"] = int(count) if count else 0
                    
                    # Get outcomes
                    for outcome in ["churn", "no_churn", "converted"]:
                        outcome_key = f"ab:{experiment_name}:{variant}:outcomes:{outcome}"
                        outcome_count = self.redis.hget(outcome_key, "count")
                        if outcome_count:
                            results[variant]["outcomes"][outcome] = int(outcome_count)
                            
            except Exception as e:
                results["error"] = str(e)
        
        return results


class ShadowMode:
    """Shadow mode for comparing model predictions without affecting users.
    
    Runs both models but only uses primary model for response.
    Logs comparison metrics.
    """
    
    def __init__(
        self,
        primary_model_loader,
        shadow_model_loader,
        comparison_threshold: float = 0.1,
    ):
        self.primary_loader = primary_model_loader
        self.shadow_loader = shadow_model_loader
        self.comparison_threshold = comparison_threshold
        self._discrepancy_count = 0
        self._total_count = 0
    
    def predict(self, features) -> Dict[str, Any]:
        """Run both models and return primary prediction with comparison."""
        import numpy as np
        
        # Primary prediction (used for response)
        primary_model = self.primary_loader()
        primary_proba = primary_model.predict_proba(features)[:, 1]
        
        # Shadow prediction (logged only)
        try:
            shadow_model = self.shadow_loader()
            shadow_proba = shadow_model.predict_proba(features)[:, 1]
            
            # Compare predictions
            discrepancy = np.abs(primary_proba - shadow_proba).mean()
            has_discrepancy = discrepancy > self.comparison_threshold
            
            self._total_count += 1
            if has_discrepancy:
                self._discrepancy_count += 1
            
            comparison = {
                "shadow_proba": shadow_proba.tolist() if len(shadow_proba) <= 10 else None,
                "discrepancy": float(discrepancy),
                "has_significant_discrepancy": has_discrepancy,
            }
            
        except Exception as e:
            logger.warning(f"Shadow model failed: {e}")
            comparison = {"error": str(e)}
        
        return {
            "proba": primary_proba,
            "shadow_comparison": comparison,
        }
    
    @property
    def discrepancy_rate(self) -> float:
        """Get rate of significant discrepancies."""
        if self._total_count == 0:
            return 0.0
        return self._discrepancy_count / self._total_count


# Global A/B tester instance
_ab_tester: Optional[ABTester] = None


def get_ab_tester() -> ABTester:
    """Get global A/B tester instance."""
    global _ab_tester
    if _ab_tester is None:
        _ab_tester = ABTester()
    return _ab_tester


def setup_default_experiment():
    """Setup default model comparison experiment."""
    tester = get_ab_tester()
    
    tester.register_experiment(Experiment(
        name="model_v2_rollout",
        control_model="churn-model/Production",
        treatment_model="churn-model/Staging",
        traffic_split=0.1,  # 10% to new model
        enabled=True,
    ))
