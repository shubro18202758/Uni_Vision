"""Tests for API authentication, rate limiting, and security headers."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Ensure real httpx is available
import httpx as _real_httpx

if isinstance(sys.modules.get("httpx"), MagicMock):
    sys.modules["httpx"] = _real_httpx
    for key in list(sys.modules):
        if key.startswith("httpx."):
            sys.modules.pop(key, None)

from fastapi.testclient import TestClient

from uni_vision.api import create_app
from uni_vision.common.config import AppConfig

# ── Helpers ───────────────────────────────────────────────────────


def _make_client(api_keys: str = "", rate_limit_rpm: int = 0) -> TestClient:
    """Create a test client with specified auth/rate-limit settings."""
    config = AppConfig()
    config.api.api_keys = api_keys
    config.api.rate_limit_rpm = rate_limit_rpm
    app = create_app(config)
    return TestClient(app, raise_server_exceptions=False)


# ── Auth Tests ────────────────────────────────────────────────────


class TestAPIKeyAuth:
    def test_no_auth_when_keys_empty(self) -> None:
        """When api_keys is empty, all requests pass through."""
        client = _make_client(api_keys="")
        resp = client.get("/detections?page=1&page_size=1")
        # May fail on DB but should NOT be 401/403
        assert resp.status_code != 401
        assert resp.status_code != 403

    def test_health_always_public(self) -> None:
        """Health endpoint never requires auth."""
        client = _make_client(api_keys="secret-key-123")
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_metrics_always_public(self) -> None:
        """Metrics endpoint never requires auth."""
        client = _make_client(api_keys="secret-key-123")
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_missing_key_returns_401(self) -> None:
        """Request without X-API-Key header gets 401."""
        client = _make_client(api_keys="valid-key")
        resp = client.get("/detections")
        assert resp.status_code == 401
        assert "Missing" in resp.json()["detail"]

    def test_invalid_key_returns_403(self) -> None:
        """Request with wrong key gets 403."""
        client = _make_client(api_keys="valid-key")
        resp = client.get("/detections", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403
        assert "Invalid" in resp.json()["detail"]

    def test_valid_key_passes(self) -> None:
        """Request with correct key passes through auth."""
        client = _make_client(api_keys="valid-key")
        resp = client.get("/detections", headers={"X-API-Key": "valid-key"})
        # Should NOT be 401/403 — may be 500 due to no DB, that's fine
        assert resp.status_code not in (401, 403)

    def test_multiple_keys_supported(self) -> None:
        """Comma-separated keys are all valid."""
        client = _make_client(api_keys="key-a,key-b,key-c")
        for key in ("key-a", "key-b", "key-c"):
            resp = client.get("/stats", headers={"X-API-Key": key})
            assert resp.status_code not in (401, 403), f"Key {key} should be valid"


# ── Rate Limit Tests ──────────────────────────────────────────────


class TestRateLimit:
    def test_no_rate_limit_when_zero(self) -> None:
        """When rate_limit_rpm=0, no limiting occurs."""
        client = _make_client(rate_limit_rpm=0)
        for _ in range(200):
            resp = client.get("/health")
            assert resp.status_code != 429

    def test_rate_limit_triggers_429(self) -> None:
        """Exceeding the limit returns 429."""
        client = _make_client(rate_limit_rpm=5)
        statuses = []
        for _ in range(10):
            resp = client.get("/stats")
            statuses.append(resp.status_code)

        assert 429 in statuses, "Should hit rate limit within 10 requests at 5 RPM"

    def test_health_exempt_from_rate_limit(self) -> None:
        """Health endpoint is exempt from rate limiting."""
        client = _make_client(rate_limit_rpm=3)
        for _ in range(20):
            resp = client.get("/health")
            assert resp.status_code == 200


# ── Security Headers Tests ────────────────────────────────────────


class TestSecurityHeaders:
    def test_security_headers_present(self) -> None:
        """All security headers are set on responses."""
        client = _make_client()
        resp = client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("Cache-Control") == "no-store"
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


# ── Key Generation ────────────────────────────────────────────────


class TestKeyGeneration:
    def test_generate_api_key_length(self) -> None:
        """Generated keys are 64 hex characters (32 bytes)."""
        from uni_vision.api.middleware.auth import generate_api_key

        key = generate_api_key()
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_generate_api_key_unique(self) -> None:
        """Each call generates a unique key."""
        from uni_vision.api.middleware.auth import generate_api_key

        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100
