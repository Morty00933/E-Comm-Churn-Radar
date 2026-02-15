from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from src.common.logging import get_logger

logger = get_logger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Track failed authentication attempts for rate limiting
_auth_failures: dict = {}
AUTH_FAILURE_WINDOW = 300  # 5 minutes
AUTH_FAILURE_MAX = 10  # Max failures before lockout


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _check_auth_rate_limit(client_ip: str) -> bool:
    """Check if client is rate limited due to auth failures."""
    now = time.time()

    # Clean old entries periodically
    if len(_auth_failures) > 1000:
        cutoff = now - AUTH_FAILURE_WINDOW
        to_delete = [k for k, (_, t) in _auth_failures.items() if t < cutoff]
        for k in to_delete:
            del _auth_failures[k]

    if client_ip in _auth_failures:
        failures, first_failure = _auth_failures[client_ip]
        if now - first_failure > AUTH_FAILURE_WINDOW:
            # Window expired, reset
            del _auth_failures[client_ip]
            return True
        if failures >= AUTH_FAILURE_MAX:
            return False
    return True


def _record_auth_failure(client_ip: str):
    """Record an authentication failure."""
    now = time.time()
    if client_ip in _auth_failures:
        failures, first_failure = _auth_failures[client_ip]
        _auth_failures[client_ip] = (failures + 1, first_failure)
    else:
        _auth_failures[client_ip] = (1, now)


def get_api_key() -> str:
    """Get API key from environment.

    Security: In production, use a strong random key.
    Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"
    """
    key = os.getenv("API_KEY")
    if not key:
        raise RuntimeError("API_KEY environment variable is not set")
    return key


def secure_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


async def authorize_request(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
) -> str:
    """Authorize API request with security best practices.

    Security features:
    - Timing-safe comparison to prevent timing attacks
    - Rate limiting on failed attempts
    - Logging of auth failures (without exposing key)
    - No dev bypass in any environment
    """
    client_ip = _get_client_ip(request)

    # Check rate limit on auth failures
    if not _check_auth_rate_limit(client_ip):
        logger.warning(f"Auth rate limit exceeded for {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Too many authentication failures. Try again later.",
        )

    # Require API key in all environments
    if not api_key:
        _record_auth_failure(client_ip)
        logger.warning(f"Missing API key from {client_ip} for {request.url.path}")
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )

    try:
        expected_key = get_api_key()
    except RuntimeError:
        # API_KEY not configured - fail closed
        logger.error("API_KEY environment variable not set")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error.",
        )

    # Use timing-safe comparison
    if not secure_compare(api_key, expected_key):
        _record_auth_failure(client_ip)
        logger.warning(f"Invalid API key from {client_ip} for {request.url.path}")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    return api_key


def generate_api_key() -> str:
    """Generate a secure API key.

    Use this to generate a new key:
    python -c "from src.api.auth import generate_api_key; print(generate_api_key())"
    """
    return secrets.token_urlsafe(32)
