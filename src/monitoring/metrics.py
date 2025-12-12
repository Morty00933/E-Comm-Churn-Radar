from __future__ import annotations

import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

request_total = Counter(
    "churn_api_requests_total",
    "Total API requests",
    labelnames=["method", "endpoint", "status"],
)

request_latency = Histogram(
    "churn_api_request_latency_seconds",
    "Request latency in seconds",
    labelnames=["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

prediction_total = Counter(
    "churn_predictions_total",
    "Total predictions made",
    labelnames=["label"],
)

prediction_probability = Histogram(
    "churn_prediction_probability",
    "Distribution of churn probabilities",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

model_info = Gauge(
    "churn_model_info",
    "Model information",
    labelnames=["name", "version", "stage"],
)

rate_limit_total = Counter(
    "churn_rate_limit_total",
    "Rate limit events",
    labelnames=["action"],
)

start_timestamp = Gauge("churn_api_start_timestamp", "API start time")
uptime_gauge = Gauge("churn_api_uptime_seconds", "API uptime")


def record_prediction(proba: float, label: int) -> None:
    prediction_total.labels(label=str(label)).inc()
    prediction_probability.observe(proba)


def record_request(method: str, endpoint: str, status: int, duration: float) -> None:
    request_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    request_latency.labels(method=method, endpoint=endpoint).observe(duration)


def set_model_info(name: str, version: str, stage: str) -> None:
    model_info.labels(name=name, version=version, stage=stage).set(1)


def setup_metrics_endpoint(app):
    from fastapi import Response
    
    @app.get("/metrics")
    async def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )


class MetricsMiddleware:
    def __init__(self, app, exclude_paths: list = None):
        self.app = app
        self.exclude_paths = exclude_paths or ["/metrics", "/health", "/healthz"]
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        
        if any(path.startswith(p) for p in self.exclude_paths):
            return await self.app(scope, receive, send)
        
        start = time.perf_counter()
        status_code = 500
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start
            record_request(method, path, status_code, duration)
