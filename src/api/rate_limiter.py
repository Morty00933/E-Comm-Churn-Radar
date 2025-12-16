from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse

from src.common.logging import get_logger
from src.monitoring.metrics import rate_limit_total

logger = get_logger(__name__)

RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1") == "1"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))

# Лимиты для «тяжёлых» эндпоинтов (SHAP медленный — даём запас, но не снимаем защиту)
ENDPOINT_LIMITS: Dict[str, Tuple[int, int]] = {
    "/explain": (int(os.getenv("RATE_LIMIT_EXPLAIN_REQUESTS", "30")), 60),          # 30 req/min
    "/feature-importance": (int(os.getenv("RATE_LIMIT_IMPORTANCE_REQUESTS", "60")), 60),  # 60 req/min
}

_redis_client = None
_local_cache: dict = {}


async def rate_limit_startup():
    global _redis_client

    if not RATE_LIMIT_ENABLED:
        return

    try:
        import redis.asyncio as redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        await _redis_client.ping()
        logger.info("Redis connected for rate limiting")
    except Exception as e:
        logger.warning(f"Redis unavailable, using local rate limiting: {e}")
        _redis_client = None


async def rate_limit_shutdown():
    global _redis_client
    if _redis_client:
        await _redis_client.close()


def _get_client_id(request: Request) -> str:
    """Extract client identifier from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take first IP, strip whitespace
        return forwarded.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


def _get_rate_limit_for_path(path: str) -> Tuple[int, int]:
    """Get rate limit config for specific endpoint."""
    for endpoint_path, limits in ENDPOINT_LIMITS.items():
        if path.startswith(endpoint_path):
            return limits
    return RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW


async def _check_rate_limit_redis(
    client_id: str, max_requests: int, window: int, key_suffix: str = ""
) -> Tuple[bool, int]:
    """Check rate limit using Redis."""
    key = f"rate_limit:{client_id}{key_suffix}"

    pipe = _redis_client.pipeline()
    pipe.incr(key)
    pipe.expire(key, window)
    results = await pipe.execute()

    current = results[0]
    remaining = max(0, max_requests - current)

    return current <= max_requests, remaining


def _check_rate_limit_local(
    client_id: str, max_requests: int, window: int, key_suffix: str = ""
) -> Tuple[bool, int]:
    """Check rate limit using local memory."""
    now = time.time()
    key = f"rate_limit:{client_id}{key_suffix}"

    # Clean old entries periodically
    if len(_local_cache) > 10000:
        cutoff = now - max(RATE_LIMIT_WINDOW, 300)
        to_delete = [k for k, (_, t) in _local_cache.items() if t < cutoff]
        for k in to_delete:
            del _local_cache[k]

    if key in _local_cache:
        count, window_start = _local_cache[key]
        if now - window_start > window:
            _local_cache[key] = (1, now)
            return True, max_requests - 1
        else:
            _local_cache[key] = (count + 1, window_start)
            remaining = max(0, max_requests - count - 1)
            return count + 1 <= max_requests, remaining
    else:
        _local_cache[key] = (1, now)
        return True, max_requests - 1


async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware with per-endpoint limits."""
    if not RATE_LIMIT_ENABLED:
        return await call_next(request)

    # Skip rate limiting for health and metrics endpoints
    skip_paths = ["/health", "/healthz", "/metrics", "/docs", "/redoc", "/openapi.json"]
    if any(request.url.path.startswith(p) for p in skip_paths):
        return await call_next(request)

    client_id = _get_client_id(request)
    path = request.url.path

    # Get rate limit for this endpoint
    max_requests, window = _get_rate_limit_for_path(path)

    # Use path-specific key suffix for per-endpoint limits
    key_suffix = ""
    if path in ENDPOINT_LIMITS:
        key_suffix = f":{path}"

    try:
        if _redis_client:
            allowed, remaining = await _check_rate_limit_redis(
                client_id, max_requests, window, key_suffix
            )
        else:
            allowed, remaining = _check_rate_limit_local(
                client_id, max_requests, window, key_suffix
            )
    except Exception as e:
        logger.warning(f"Rate limit check failed: {e}")
        # Fail open - allow request if rate limiting fails
        return await call_next(request)

    if not allowed:
        rate_limit_total.labels(action="blocked").inc()
        logger.info(f"Rate limit exceeded for {client_id} on {path}")
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Rate limit exceeded. Try again in {window} seconds.",
                "limit": max_requests,
                "window_seconds": window,
            },
            headers={
                "Retry-After": str(window),
                "X-RateLimit-Limit": str(max_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + window),
            },
        )

    rate_limit_total.labels(action="allowed").inc()
    response = await call_next(request)

    # Add rate limit headers to response
    response.headers["X-RateLimit-Limit"] = str(max_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)

    return response


def get_rate_limiter():
    """Get Redis client for external use."""
    return _redis_client
