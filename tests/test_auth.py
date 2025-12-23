"""Tests for authentication module."""
import os
import pytest
from unittest.mock import MagicMock, patch

from src.api.auth import (
    secure_compare,
    generate_api_key,
    get_api_key,
    authorize_request,
    _check_auth_rate_limit,
    _record_auth_failure,
    _auth_failures,
    AUTH_FAILURE_MAX,
)


class TestSecureCompare:
    def test_equal_strings_return_true(self):
        assert secure_compare("test123", "test123") is True

    def test_different_strings_return_false(self):
        assert secure_compare("test123", "test456") is False

    def test_empty_strings_return_true(self):
        assert secure_compare("", "") is True

    def test_one_empty_string_returns_false(self):
        assert secure_compare("test", "") is False
        assert secure_compare("", "test") is False

    def test_unicode_strings(self):
        assert secure_compare("тест", "тест") is True
        assert secure_compare("тест", "test") is False


class TestGenerateApiKey:
    def test_generates_string(self):
        key = generate_api_key()
        assert isinstance(key, str)

    def test_generates_sufficient_length(self):
        key = generate_api_key()
        assert len(key) >= 32

    def test_generates_unique_keys(self):
        keys = [generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100


class TestGetApiKey:
    def test_returns_env_variable(self):
        with patch.dict(os.environ, {"API_KEY": "my-secret-key"}):
            assert get_api_key() == "my-secret-key"

    def test_raises_when_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove API_KEY if it exists
            os.environ.pop("API_KEY", None)
            with pytest.raises(RuntimeError, match="API_KEY environment variable is not set"):
                get_api_key()


class TestAuthRateLimit:
    def setup_method(self):
        """Clear auth failures before each test."""
        _auth_failures.clear()

    def test_allows_first_request(self):
        assert _check_auth_rate_limit("192.168.1.1") is True

    def test_allows_requests_under_limit(self):
        client_ip = "192.168.1.2"
        for _ in range(AUTH_FAILURE_MAX - 1):
            _record_auth_failure(client_ip)

        assert _check_auth_rate_limit(client_ip) is True

    def test_blocks_after_max_failures(self):
        client_ip = "192.168.1.3"
        for _ in range(AUTH_FAILURE_MAX):
            _record_auth_failure(client_ip)

        assert _check_auth_rate_limit(client_ip) is False

    def test_different_clients_independent(self):
        client1 = "192.168.1.4"
        client2 = "192.168.1.5"

        for _ in range(AUTH_FAILURE_MAX):
            _record_auth_failure(client1)

        assert _check_auth_rate_limit(client1) is False
        assert _check_auth_rate_limit(client2) is True


class TestAuthorizeRequest:
    @pytest.fixture
    def mock_request(self):
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.url.path = "/predict"
        return request

    def setup_method(self):
        """Clear auth failures before each test."""
        _auth_failures.clear()

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_401(self, mock_request):
        from fastapi import HTTPException

        with patch.dict(os.environ, {"API_KEY": "secret-key"}):
            with pytest.raises(HTTPException) as exc_info:
                await authorize_request(mock_request, api_key=None)

            assert exc_info.value.status_code == 401
            assert "Missing API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_api_key_returns_401(self, mock_request):
        from fastapi import HTTPException

        with patch.dict(os.environ, {"API_KEY": "correct-key"}):
            with pytest.raises(HTTPException) as exc_info:
                await authorize_request(mock_request, api_key="wrong-key")

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_valid_api_key_returns_key(self, mock_request):
        with patch.dict(os.environ, {"API_KEY": "correct-key"}):
            result = await authorize_request(mock_request, api_key="correct-key")
            assert result == "correct-key"

    @pytest.mark.asyncio
    async def test_rate_limit_after_failures(self, mock_request):
        from fastapi import HTTPException

        with patch.dict(os.environ, {"API_KEY": "secret-key"}):
            # Exhaust rate limit
            for _ in range(AUTH_FAILURE_MAX):
                try:
                    await authorize_request(mock_request, api_key="wrong-key")
                except HTTPException:
                    pass

            # Next request should be rate limited
            with pytest.raises(HTTPException) as exc_info:
                await authorize_request(mock_request, api_key="wrong-key")

            assert exc_info.value.status_code == 429
            assert "Too many authentication failures" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_no_dev_bypass(self, mock_request):
        """Ensure there's no development bypass."""
        from fastapi import HTTPException

        with patch.dict(os.environ, {"ENVIRONMENT": "development", "API_KEY": "secret"}):
            with pytest.raises(HTTPException) as exc_info:
                await authorize_request(mock_request, api_key=None)

            assert exc_info.value.status_code == 401
