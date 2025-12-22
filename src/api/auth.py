from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key() -> str:
    return os.getenv("API_KEY", "dev-api-key")


async def authorize_request(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
) -> str:
    if os.getenv("ENVIRONMENT") == "development" and not api_key:
        return "dev-bypass"
    
    expected_key = get_api_key()
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )
    
    if api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )
    
    return api_key
