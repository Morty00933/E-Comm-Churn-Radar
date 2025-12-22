from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from src.common.logging import get_logger
from src.monitoring.metrics import rate_limit_total

logger = get_logger(__name__)

RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1") == "1"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))

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
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    if request.client:
        return request.client.host
    
    return "unknown"


async def _check_rate_limit_redis(client_id: str) -> tuple[bool, int]:
    key = f"rate_limit:{client_id}"
    
    pipe = _redis_client.pipeline()
    pipe.incr(key)
    pipe.expire(key, RATE_LIMIT_WINDOW)
    results = await pipe.execute()
    
    current = results[0]
    remaining = max(0, RATE_LIMIT_REQUESTS - current)
    
    return current <= RATE_LIMIT_REQUESTS, remaining


def _check_rate_limit_local(client_id: str) -> tuple[bool, int]:
    now = time.time()
    key = f"rate_limit:{client_id}"
    
    if key in _local_cache:
        count, window_start = _local_cache[key]
        if now - window_start > RATE_LIMIT_WINDOW:
            _local_cache[key] = (1, now)
            return True, RATE_LIMIT_REQUESTS - 1
        else:
            _local_cache[key] = (count + 1, window_start)
            remaining = max(0, RATE_LIMIT_REQUESTS - count - 1)
            return count + 1 <= RATE_LIMIT_REQUESTS, remaining
    else:
        _local_cache[key] = (1, now)
        return True, RATE_LIMIT_REQUESTS - 1


async def rate_limit_middleware(request: Request, call_next):
    if not RATE_LIMIT_ENABLED:
        return await call_next(request)
    
    if request.url.path in ["/health", "/healthz", "/metrics"]:
        return await call_next(request)
    
    client_id = _get_client_id(request)
    
    try:
        if _redis_client:
            allowed, remaining = await _check_rate_limit_redis(client_id)
        else:
            allowed, remaining = _check_rate_limit_local(client_id)
    except Exception as e:
        logger.warning(f"Rate limit check failed: {e}")
        return await call_next(request)
    
    if not allowed:
        rate_limit_total.labels(action="blocked").inc()
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded. Try again in {RATE_LIMIT_WINDOW} seconds."},
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
        )
    
    rate_limit_total.labels(action="allowed").inc()
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    
    return response


def get_rate_limiter():
    return _redis_client
