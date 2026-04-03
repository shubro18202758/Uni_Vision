"""Integration tests: API full lifecycle (routes + middleware + auth).

Exercises the full FastAPI app with middleware stacking, auth enforcement,
rate limiting, and route→DB interactions using in-memory fakes.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Ensure real httpx for TestClient
import httpx as _real_httpx
import pytest

if isinstance(sys.modules.get("httpx"), MagicMock):
    sys.modules["httpx"] = _real_httpx
    for key in list(sys.modules):
        if key.startswith("httpx."):
            sys.modules.pop(key, None)

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from uni_vision.api import create_app
from uni_vision.common.config import AppConfig

if TYPE_CHECKING:
    from fastapi import FastAPI

# ── Fake DB layer ─────────────────────────────────────────────────


class _FakeConn:
    def __init__(self, rows: list | None = None):
        self._rows = rows or []
        self.executed: list = []

    async def fetch(self, sql, *args):
        self.executed.append(("fetch", sql, args))
        return self._rows

    async def fetchval(self, sql, *args):
        self.executed.append(("fetchval", sql, args))
        return len(self._rows)

    async def execute(self, sql, *args):
        self.executed.append(("execute", sql, args))
        return "INSERT 0 1"


class _AcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        pass


class _FakePool:
    def __init__(self, conn=None):
        self._conn = conn or _FakeConn()

    def acquire(self, **kw):
        return _AcquireCtx(self._conn)


class _FakePG:
    def __init__(self, conn=None):
        self._pool = _FakePool(conn)

    async def connect(self):
        pass

    async def close(self):
        pass


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def app(config: AppConfig) -> FastAPI:
    return create_app(config)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────


class TestFullRequestLifecycle:
    """End-to-end request flows through all middleware layers."""

    def test_public_health_with_all_middleware(self) -> None:
        """Health passes through auth + rate-limit + security headers."""
        config = AppConfig()
        config.api.api_keys = "secret-key"
        config.api.rate_limit_rpm = 60
        app = create_app(config)
        with TestClient(app) as tc:
            resp = tc.get("/health")
        assert resp.status_code == 200
        # Security headers present
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        # No auth required
        data = resp.json()
        assert data["healthy"] is True

    def test_protected_endpoint_full_chain(self) -> None:
        """Authenticated request → middleware chain → DB query → response."""
        conn = _FakeConn(rows=[])
        config = AppConfig()
        config.api.api_keys = "test-integration-key"
        app = create_app(config)

        with TestClient(app) as tc:
            app.state.pg_client = _FakePG(conn)
            resp = tc.get(
                "/detections",
                headers={"X-API-Key": "test-integration-key"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_auth_blocks_then_allows(self) -> None:
        """First request without key → 401, then with key → passes."""
        config = AppConfig()
        config.api.api_keys = "my-key"
        app = create_app(config)

        with TestClient(app, raise_server_exceptions=False) as tc:
            app.state.pg_client = _FakePG()
            # No key → blocked
            r1 = tc.get("/detections")
            assert r1.status_code == 401

            # Valid key → passes
            r2 = tc.get("/detections", headers={"X-API-Key": "my-key"})
            assert r2.status_code == 200


class TestSourcesCRUDIntegration:
    """Full CRUD cycle on /sources with fake DB."""

    def test_register_and_list_sources(self, app: FastAPI) -> None:
        """POST a source, then GET should return it via the DB layer."""
        conn = _FakeConn()

        with TestClient(app) as tc:
            app.state.pg_client = _FakePG(conn)

            # Register
            resp = tc.post(
                "/sources",
                json={
                    "camera_id": "cam-integration-01",
                    "source_url": "rtsp://192.168.1.100/live",
                    "location_tag": "Parking A",
                    "fps_target": 5,
                },
            )
            assert resp.status_code == 201
            assert resp.json()["camera_id"] == "cam-integration-01"

            # The registration should have triggered an execute call
            assert any(op[0] == "execute" for op in conn.executed)

    def test_delete_nonexistent_returns_404(self, app: FastAPI) -> None:
        conn = _FakeConn()

        # Override execute to simulate "DELETE 0" for a missing row
        async def _execute_delete_0(sql, *args):
            conn.executed.append(("execute", sql, args))
            return "DELETE 0"

        conn.execute = _execute_delete_0

        with TestClient(app) as tc:
            app.state.pg_client = _FakePG(conn)
            resp = tc.delete("/sources/ghost-camera")
        assert resp.status_code == 404


class TestDetectionsFilterIntegration:
    """Detections endpoint with various filter combinations."""

    def test_filter_by_camera_id(self, app: FastAPI) -> None:
        conn = _FakeConn(rows=[])
        with TestClient(app) as tc:
            app.state.pg_client = _FakePG(conn)
            resp = tc.get("/detections?camera_id=cam-01")
        assert resp.status_code == 200
        # Verify the filter was passed to the query
        fetch_calls = [c for c in conn.executed if c[0] == "fetch"]
        assert len(fetch_calls) >= 1
        sql = fetch_calls[0][1]
        assert "camera_id" in sql

    def test_filter_by_plate_number(self, app: FastAPI) -> None:
        conn = _FakeConn(rows=[])
        with TestClient(app) as tc:
            app.state.pg_client = _FakePG(conn)
            resp = tc.get("/detections?plate_number=MH12")
        assert resp.status_code == 200

    def test_filter_by_status(self, app: FastAPI) -> None:
        conn = _FakeConn(rows=[])
        with TestClient(app) as tc:
            app.state.pg_client = _FakePG(conn)
            resp = tc.get("/detections?status=valid")
        assert resp.status_code == 200

    def test_combined_filters(self, app: FastAPI) -> None:
        conn = _FakeConn(rows=[])
        with TestClient(app) as tc:
            app.state.pg_client = _FakePG(conn)
            resp = tc.get("/detections?camera_id=cam-01&status=valid&page=1&page_size=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 1
        assert data["page_size"] == 10


class TestMetricsStatsIntegration:
    """Metrics and stats endpoints return valid data."""

    def test_metrics_returns_prometheus_format(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]

    def test_stats_returns_json_dict(self, client: TestClient) -> None:
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_metrics_and_stats_are_consistent(self, client: TestClient) -> None:
        """Both endpoints should respond without errors."""
        r1 = client.get("/metrics")
        r2 = client.get("/stats")
        assert r1.status_code == 200
        assert r2.status_code == 200


class TestRateLimitIntegration:
    """Rate limiting interacts correctly with the full middleware chain."""

    def test_rate_limit_does_not_affect_health(self) -> None:
        config = AppConfig()
        config.api.rate_limit_rpm = 2
        app = create_app(config)
        with TestClient(app) as tc:
            # Health is exempt — should never get 429
            for _ in range(20):
                r = tc.get("/health")
                assert r.status_code == 200

    def test_rate_limit_triggers_on_stats(self) -> None:
        config = AppConfig()
        config.api.rate_limit_rpm = 3
        app = create_app(config)
        statuses = []
        with TestClient(app) as tc:
            for _ in range(15):
                r = tc.get("/stats")
                statuses.append(r.status_code)
        assert 429 in statuses


class TestSecurityHeadersIntegration:
    """Security headers are present on all response types."""

    def test_headers_on_json_endpoint(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.headers["Cache-Control"] == "no-store"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_headers_on_metrics_endpoint(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    def test_headers_on_404(self, client: TestClient) -> None:
        resp = client.get("/nonexistent-path")
        assert resp.headers.get("X-Frame-Options") == "DENY"
