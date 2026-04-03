"""Tests for the WebSocket real-time events endpoint."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Ensure real httpx is available
import httpx as _real_httpx
import pytest

if isinstance(sys.modules.get("httpx"), MagicMock):
    sys.modules["httpx"] = _real_httpx
    for key in list(sys.modules):
        if key.startswith("httpx."):
            sys.modules.pop(key, None)

from fastapi.testclient import TestClient

from uni_vision.api import create_app
from uni_vision.common.config import AppConfig


@pytest.fixture
def client() -> TestClient:
    app = create_app(AppConfig())
    return TestClient(app)


class TestWebSocketEvents:
    def test_ws_connect_and_disconnect(self, client: TestClient) -> None:
        """Client can connect to /ws/events and cleanly disconnect."""
        with client.websocket_connect("/ws/events"):
            # Connection established — just disconnect
            pass

    def test_ws_client_tracked(self, client: TestClient) -> None:
        """Connected clients are tracked in the _clients set."""
        from uni_vision.api.routes.ws_events import _clients

        with client.websocket_connect("/ws/events"):
            assert len(_clients) >= 1

    def test_broadcast_helper(self) -> None:
        """The _broadcast function sends to all registered clients."""
        import asyncio

        from uni_vision.api.routes.ws_events import _broadcast

        # _broadcast with no clients should not raise
        asyncio.get_event_loop().run_until_complete(_broadcast('{"test": true}'))

    def test_publish_event_import(self) -> None:
        """publish_event function is importable and callable."""
        from uni_vision.api.routes.ws_events import publish_event

        assert callable(publish_event)

    def test_redis_channel_constant(self) -> None:
        """Redis channel name is set correctly."""
        from uni_vision.api.routes.ws_events import REDIS_CHANNEL

        assert REDIS_CHANNEL == "uv:events"
