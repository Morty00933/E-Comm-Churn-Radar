"""Tests for rate limiter module."""
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.api.rate_limiter import (
    _get_client_id,
    _get_rate_limit_for_path,
    _check_rate_limit_local,
    _local_cache,
    RATE_LIMIT_REQUESTS,
    ENDPOINT_LIMITS,
)


class TestGetClientId:
    def test_returns_forwarded_ip(self):
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}
        request.client = None

        assert _get_client_id(request) == "1.2.3.4"

    def test_strips_whitespace_from_forwarded(self):
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "  1.2.3.4  , 5.6.7.8"}
        request.client = None

        assert _get_client_id(request) == "1.2.3.4"

    def test_returns_client_host_if_no_forwarded(self):
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        assert _get_client_id(request) == "192.168.1.1"

    def test_returns_unknown_if_no_client(self):
        request = MagicMock()
        request.headers = {}
        request.client = None

        assert _get_client_id(request) == "unknown"


class TestGetRateLimitForPath:
    def test_returns_default_for_unknown_path(self):
        limit, window = _get_rate_limit_for_path("/predict")
        assert limit == RATE_LIMIT_REQUESTS

    def test_returns_custom_limit_for_explain(self):
        limit, window = _get_rate_limit_for_path("/explain")
        assert limit == ENDPOINT_LIMITS["/explain"][0]
        assert window == 60

    def test_returns_custom_limit_for_feature_importance(self):
        limit, window = _get_rate_limit_for_path("/feature-importance")
        assert limit == ENDPOINT_LIMITS["/feature-importance"][0]
        assert window == 60


class TestCheckRateLimitLocal:
    def setup_method(self):
        """Clear local cache before each test."""
        _local_cache.clear()

    def test_allows_first_request(self):
        allowed, remaining = _check_rate_limit_local("client1", 10, 60)
        assert allowed is True
        assert remaining == 9

    def test_allows_requests_under_limit(self):
        for i in range(9):
            allowed, remaining = _check_rate_limit_local("client2", 10, 60)
            assert allowed is True
            assert remaining == 10 - i - 1

    def test_blocks_after_limit_reached(self):
        # First 10 requests should be allowed
        for i in range(10):
            allowed, _ = _check_rate_limit_local("client3", 10, 60)

        # 11th request should be blocked
        allowed, remaining = _check_rate_limit_local("client3", 10, 60)
        assert allowed is False
        assert remaining == 0

    def test_different_clients_have_separate_limits(self):
        # Exhaust client4's limit
        for _ in range(10):
            _check_rate_limit_local("client4", 10, 60)

        # client5 should still be allowed
        allowed, _ = _check_rate_limit_local("client5", 10, 60)
        assert allowed is True

    def test_different_key_suffixes_are_separate(self):
        # Exhaust limit for one endpoint
        for _ in range(10):
            _check_rate_limit_local("client6", 10, 60, ":/explain")

        # Different endpoint should still be allowed
        allowed, _ = _check_rate_limit_local("client6", 10, 60, ":/predict")
        assert allowed is True


class TestRateLimitMiddleware:
    @pytest.fixture
    def mock_request(self):
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.url.path = "/predict"
        return request

    def setup_method(self):
        """Clear local cache before each test."""
        _local_cache.clear()

    @pytest.mark.asyncio
    async def test_skips_health_endpoints(self, mock_request):
        from src.api.rate_limiter import rate_limit_middleware

        mock_request.url.path = "/health"
        call_next = AsyncMock(return_value=MagicMock())

        with patch.dict(os.environ, {"RATE_LIMIT_ENABLED": "1"}):
            await rate_limit_middleware(mock_request, call_next)

        call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_metrics_endpoint(self, mock_request):
        from src.api.rate_limiter import rate_limit_middleware

        mock_request.url.path = "/metrics"
        call_next = AsyncMock(return_value=MagicMock())

        with patch.dict(os.environ, {"RATE_LIMIT_ENABLED": "1"}):
            await rate_limit_middleware(mock_request, call_next)

        call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_disabled_allows_all(self, mock_request):
        from src.api.rate_limiter import rate_limit_middleware

        call_next = AsyncMock(return_value=MagicMock())

        with patch("src.api.rate_limiter.RATE_LIMIT_ENABLED", False):
            await rate_limit_middleware(mock_request, call_next)

        call_next.assert_called_once()
