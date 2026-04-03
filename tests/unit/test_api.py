"""Integration tests for the FastAPI REST API layer.

These tests exercise the route handlers in isolation using FastAPI's
``TestClient`` (synchronous, backed by ``httpx``).  The database is
**not** hit — all PostgreSQL interactions are monkey-patched.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ── Ensure real httpx is available for TestClient ─────────────────
# The main conftest.py stubs httpx as MagicMock for unit tests that
# don't need it.  API tests *do* need it, so we force-import the real
# httpx before conftest runs.  If httpx was already stubbed, replace
# it with the real module.
import httpx as _real_httpx
import pytest

if isinstance(sys.modules.get("httpx"), MagicMock):
    sys.modules["httpx"] = _real_httpx
    # Also restore sub-modules that starlette / fastapi might need
    for key in list(sys.modules):
        if key.startswith("httpx."):
            sys.modules.pop(key, None)

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from uni_vision.api import create_app
from uni_vision.common.config import AppConfig

if TYPE_CHECKING:
    from fastapi import FastAPI

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def app() -> FastAPI:
    """Create a test FastAPI application with default config."""
    return create_app(AppConfig())


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Return a synchronous test client wired to the app."""
    return TestClient(app)


# ── Health endpoint ───────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["healthy"] is True

    def test_health_standalone_mode(self, client: TestClient) -> None:
        """Without a HealthService in state, returns standalone mode."""
        resp = client.get("/health")
        data = resp.json()
        assert data.get("mode") == "standalone"

    def test_health_with_service(self, app: FastAPI) -> None:
        """When HealthService is wired, returns full health payload."""
        from uni_vision.contracts.dtos import HealthStatus, OffloadMode

        mock_status = HealthStatus(
            healthy=True,
            gpu_available=True,
            ollama_reachable=True,
            database_connected=True,
            streams_connected=2,
            streams_total=3,
            offload_mode=OffloadMode.GPU_PRIMARY,
        )

        class _FakeHealthService:
            async def check(self):
                return mock_status

        app.state.health_service = _FakeHealthService()

        with TestClient(app) as tc:
            resp = tc.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["healthy"] is True
        assert data["gpu_available"] is True
        assert data["streams_connected"] == 2

    def test_health_unhealthy_returns_503(self, app: FastAPI) -> None:
        from uni_vision.contracts.dtos import HealthStatus, OffloadMode

        mock_status = HealthStatus(
            healthy=False,
            gpu_available=False,
            ollama_reachable=False,
            database_connected=False,
            streams_connected=0,
            streams_total=0,
            offload_mode=OffloadMode.FULL_CPU,
            details={"gpu": "unavailable"},
        )

        class _FakeHealthService:
            async def check(self):
                return mock_status

        app.state.health_service = _FakeHealthService()

        with TestClient(app, raise_server_exceptions=False) as tc:
            resp = tc.get("/health")
        assert resp.status_code == 503
        assert resp.json()["healthy"] is False


# ── Metrics endpoint ──────────────────────────────────────────────


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        ct = resp.headers.get("content-type", "")
        # Prometheus text format MIME type
        assert "text/plain" in ct or "text/plain" in ct


# ── Stats endpoint ────────────────────────────────────────────────


class TestStatsEndpoint:
    def test_stats_returns_200(self, client: TestClient) -> None:
        resp = client.get("/stats")
        assert resp.status_code == 200

    def test_stats_returns_json(self, client: TestClient) -> None:
        resp = client.get("/stats")
        data = resp.json()
        assert isinstance(data, dict)


# ── Fake asyncpg pool / connection for DB-dependent tests ─────────


class _FakeConn:
    """Mimics an asyncpg Connection with recording."""

    def __init__(self) -> None:
        self.executed: list = []

    async def fetch(self, sql: str, *args: object) -> list:
        self.executed.append(("fetch", sql, args))
        return []

    async def fetchval(self, sql: str, *args: object) -> int:
        self.executed.append(("fetchval", sql, args))
        return 0

    async def execute(self, sql: str, *args: object) -> str:
        self.executed.append(("execute", sql, args))
        return "DELETE 0"


class _AcquireCtx:
    """Async context manager returned by ``_FakePool.acquire``."""

    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *a: object) -> None:
        pass


class _FakePool:
    def __init__(self, conn: _FakeConn | None = None) -> None:
        self._conn = conn or _FakeConn()

    def acquire(self, **kw: object) -> _AcquireCtx:
        return _AcquireCtx(self._conn)


class _FakePG:
    def __init__(self, conn: _FakeConn | None = None) -> None:
        self._pool = _FakePool(conn)

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass


# ── Sources endpoint ──────────────────────────────────────────────


class TestSourcesEndpoint:
    def test_sources_list_returns_empty_when_no_db(self, app: FastAPI) -> None:
        with TestClient(app) as tc:
            app.state.pg_client = _FakePG()
            resp = tc.get("/sources")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_sources_post_registers_camera(self, app: FastAPI) -> None:
        conn = _FakeConn()

        with TestClient(app) as tc:
            app.state.pg_client = _FakePG(conn)
            resp = tc.post(
                "/sources",
                json={
                    "camera_id": "cam-01",
                    "source_url": "rtsp://10.0.0.1/stream",
                    "location_tag": "Gate A",
                    "fps_target": 5,
                    "enabled": True,
                },
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["camera_id"] == "cam-01"
        assert data["status"] == "registered"
        assert len(conn.executed) == 1

    def test_sources_post_rejects_empty_camera_id(self, client: TestClient) -> None:
        resp = client.post(
            "/sources",
            json={"camera_id": "", "source_url": "rtsp://x"},
        )
        assert resp.status_code == 422  # validation error

    def test_sources_delete_not_found(self, app: FastAPI) -> None:
        with TestClient(app) as tc:
            app.state.pg_client = _FakePG()
            resp = tc.delete("/sources/nonexistent-cam")
        assert resp.status_code == 404


# ── Detections endpoint ───────────────────────────────────────────


class TestDetectionsEndpoint:
    def test_detections_empty_page(self, app: FastAPI) -> None:
        with TestClient(app) as tc:
            app.state.pg_client = _FakePG()
            resp = tc.get("/detections")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1

    def test_detections_pagination_params(self, app: FastAPI) -> None:
        with TestClient(app) as tc:
            app.state.pg_client = _FakePG()
            resp = tc.get("/detections?page=2&page_size=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 2
        assert data["page_size"] == 10

    def test_detections_invalid_page_size(self, client: TestClient) -> None:
        resp = client.get("/detections?page_size=999")
        assert resp.status_code == 422


# ── Middleware ────────────────────────────────────────────────────


class TestMiddleware:
    def test_cors_headers_present(self, client: TestClient) -> None:
        resp = client.options(
            "/health",
            headers={
                "origin": "http://localhost:3000",
                "access-control-request-method": "GET",
            },
        )
        # CORS middleware should respond (may be 200 or 400 depending
        # on exact config, but the header should be present)
        assert "access-control-allow-origin" in resp.headers

    def test_request_logging_does_not_break_response(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
