from __future__ import annotations

import os
import time
from typing import Any, Dict, Union

from fastapi import Body, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware

from src.common.logging import get_logger
from src.monitoring.metrics import (
    setup_metrics_endpoint,
    MetricsMiddleware,
    start_timestamp,
    uptime_gauge,
)
from .schemas import (
    HealthResponse,
    SinglePredictionResponse,
    BatchPredictionResponse,
    PredictionResult,
)
from .auth import authorize_request
from .rate_limiter import (
    rate_limit_middleware,
    rate_limit_startup,
    rate_limit_shutdown,
    RATE_LIMIT_ENABLED,
)
from .predictor import predict_customers, clear_model_cache

logger = get_logger(__name__)

APP_START_TIME = time.time()

app = FastAPI(
    title="Churn Radar API",
    description="Customer Churn Prediction API",
    version=os.getenv("APP_VERSION", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc",
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type", "Authorization"],
)

app.add_middleware(MetricsMiddleware)
app.middleware("http")(rate_limit_middleware)

setup_metrics_endpoint(app)


@app.on_event("startup")
async def startup():
    logger.info("Starting Churn Radar API...")
    
    start_timestamp.set(APP_START_TIME)
    uptime_gauge.set(0)
    
    await rate_limit_startup()
    
    try:
        from .predictor import load_model
        load_model()
        logger.info("Model loaded during startup")
    except Exception as e:
        logger.warning(f"Model warm-up failed: {e}")
    
    logger.info("API started successfully")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down API...")
    await rate_limit_shutdown()
    clear_model_cache()


def _probe_dependencies() -> Dict[str, Dict[str, Any]]:
    deps = {}
    
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        client = mlflow.tracking.MlflowClient()
        client.search_experiments(max_results=1)
        deps["mlflow"] = {"status": "ok"}
    except Exception as e:
        deps["mlflow"] = {"status": "error", "error": str(e)[:100]}
    
    try:
        from .predictor import load_model
        load_model()
        deps["model"] = {"status": "ok"}
    except Exception as e:
        deps["model"] = {"status": "error", "error": str(e)[:100]}
    
    if RATE_LIMIT_ENABLED:
        try:
            from .rate_limiter import get_rate_limiter
            client = get_rate_limiter()
            if client:
                deps["redis"] = {"status": "ok"}
            else:
                deps["redis"] = {"status": "degraded", "note": "Using local rate limiting"}
        except Exception as e:
            deps["redis"] = {"status": "error", "error": str(e)[:100]}
    
    return deps


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Churn Radar API",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "docs": "/docs",
        "health": "/healthz",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    deps = _probe_dependencies()
    
    details = {
        "uptime_sec": round(time.time() - APP_START_TIME, 1),
        "dependencies": deps,
    }
    
    critical = ["mlflow", "model"]
    critical_errors = [k for k in critical if deps.get(k, {}).get("status") == "error"]
    
    if critical_errors:
        status = "unhealthy"
    elif any(v.get("status") == "error" for v in deps.values()):
        status = "degraded"
    else:
        status = "ok"
    
    return HealthResponse(status=status, details=details)


@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    return HealthResponse(status="ok", details={"probe": "liveness"})


PredictionResponse = Union[SinglePredictionResponse, BatchPredictionResponse]


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: Dict[str, Any] = Body(...),
    _auth: str = Security(authorize_request),
) -> PredictionResponse:
    try:
        if "customers" in payload:
            records = payload["customers"]
            if not records:
                raise HTTPException(status_code=422, detail="customers list cannot be empty")
            predictions = predict_customers(records)
            return BatchPredictionResponse(
                predictions=[PredictionResult(**p) for p in predictions]
            )
        else:
            predictions = predict_customers([payload])
            return SinglePredictionResponse(
                prediction=PredictionResult(**predictions[0])
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_v1():
    return HealthResponse(status="ok", details={"version": "v1"})


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_v1(
    payload: Dict[str, Any] = Body(...),
    _auth: str = Security(authorize_request),
) -> PredictionResponse:
    return await predict(payload, _auth)


@app.post("/explain")
async def explain(
    payload: Dict[str, Any] = Body(...),
    _auth: str = Security(authorize_request),
) -> Dict[str, Any]:
    """Explain model predictions using SHAP values.
    
    Returns top contributing features for each prediction.
    """
    import pandas as pd
    from src.features.build_features import prepare_features
    from src.models.explainer import ModelExplainer, generate_explanation_text
    from .predictor import load_model, get_feature_columns
    
    try:
        # Handle single or batch
        if "customers" in payload:
            records = payload["customers"]
        else:
            records = [payload]
        
        top_k = payload.get("top_k", 5)
        include_text = payload.get("include_text", True)
        
        # Prepare features
        df = pd.DataFrame(records)
        df = prepare_features(df)
        
        model = load_model()
        feature_cols = get_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]

        sklearn_model = model
        if hasattr(model, "_model_impl"):
            sklearn_model = model._model_impl.python_model
        if hasattr(sklearn_model, "get_raw_model"):
            sklearn_model = sklearn_model.get_raw_model()
        
        explainer = ModelExplainer(
            model=sklearn_model,
            feature_names=available_cols,
        )
        
        # Get predictions
        X = df[available_cols]
        
        if hasattr(sklearn_model, "predict_proba"):
            probas = sklearn_model.predict_proba(X)[:, 1]
        else:
            # For MLflow pyfunc models
            probas = model.predict(X)
            if isinstance(probas, pd.DataFrame):
                probas = probas.iloc[:, 0].values
        
        # Get explanations (async — не блокирует event loop)
        explanations = await explainer.explain_async(df[available_cols], top_k=top_k)
        
        # Build response
        results = []
        for i, (record, proba, explanation) in enumerate(zip(records, probas, explanations)):
            result = {
                "user_id": record.get("user_id", i),
                "churn_proba": round(float(proba), 4),
                "churn_label": int(proba >= 0.5),
                "explanation": explanation,
            }
            if include_text:
                result["explanation_text"] = generate_explanation_text(explanation)
            results.append(result)
        
        if len(results) == 1:
            return {"result": results[0]}
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Explain failed: {e}")
        raise


@app.get("/feature-importance")
async def feature_importance(
    _auth: str = Security(authorize_request),
) -> Dict[str, Any]:
    """Get global feature importance from the model."""
    import pandas as pd
    from pathlib import Path
    from .predictor import load_model, get_feature_columns
    from src.models.explainer import ModelExplainer
    from src.features.build_features import prepare_features
    
    try:
        model = load_model()
        feature_cols = get_feature_columns()
        
        sklearn_model = model
        if hasattr(model, "_model_impl"):
            sklearn_model = model._model_impl.python_model
        if hasattr(sklearn_model, "get_raw_model"):
            sklearn_model = sklearn_model.get_raw_model()

        if hasattr(sklearn_model, "feature_importances_"):
            n_features = len(sklearn_model.feature_importances_)
            names = feature_cols[:n_features] if n_features <= len(feature_cols) else feature_cols
            importance = {
                name: float(imp)
                for name, imp in zip(names, sklearn_model.feature_importances_)
            }
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            return {"feature_importance": importance, "source": "model"}
        
        train_path = Path("data/train.csv")
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            train_df = prepare_features(train_df)
            
            available_cols = [c for c in feature_cols if c in train_df.columns]
            background = train_df[available_cols].head(100)
            
            explainer = ModelExplainer(
                model=sklearn_model,
                feature_names=available_cols,
                background_data=background,
            )
            
            importance = explainer.get_feature_importance()
            return {"feature_importance": importance, "source": "shap"}
        
        return {"error": "No feature importance available"}
    
    except Exception as e:
        logger.error(f"Feature importance failed: {e}")
        return {"error": str(e)}
