"""WebSocket endpoint for real-time plate detection events.

Clients connect to ``/ws/events`` and receive JSON messages for
every detection as it flows through the pipeline.  The bridge
between the pipeline and WebSocket clients uses Redis Pub/Sub.

Architecture:
  Pipeline → MultiDispatcher → Redis PUBLISH "uv:events"
  API /ws/events ← Redis SUBSCRIBE "uv:events" → WebSocket clients
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])

logger = logging.getLogger(__name__)

REDIS_CHANNEL = "uv:events"

# Connected WebSocket clients (managed per-process)
_clients: Set[WebSocket] = set()


async def _broadcast(message: str) -> None:
    """Send a message to all connected WebSocket clients."""
    dead: Set[WebSocket] = set()
    for ws in _clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


async def start_redis_subscriber(redis_url: str) -> asyncio.Task[None]:
    """Start a background task that subscribes to Redis and broadcasts.

    Parameters
    ----------
    redis_url:
        Redis connection URL (e.g. ``redis://localhost:6379/0``).

    Returns
    -------
    asyncio.Task
        The subscriber task. Cancel it on shutdown.
    """
    import redis.asyncio as aioredis

    async def _subscriber() -> None:
        client = aioredis.from_url(redis_url, decode_responses=True)
        try:
            pubsub = client.pubsub()
            await pubsub.subscribe(REDIS_CHANNEL)
            logger.info("redis_subscriber_started channel=%s", REDIS_CHANNEL)
            async for message in pubsub.listen():
                if message["type"] == "message":
                    await _broadcast(message["data"])
        except asyncio.CancelledError:
            logger.info("redis_subscriber_stopped")
        except Exception:
            logger.exception("redis_subscriber_error")
        finally:
            await client.aclose()

    return asyncio.create_task(_subscriber())


async def publish_event(redis_url: str, event_data: dict) -> None:
    """Publish a detection event to the Redis channel.

    Called by the dispatcher after a successful write.
    """
    import redis.asyncio as aioredis

    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await client.publish(REDIS_CHANNEL, json.dumps(event_data, default=str))
    finally:
        await client.aclose()


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time detection event streaming.

    Clients receive JSON messages with detection details as they
    are processed by the pipeline. No authentication is required
    for WebSocket connections in this version.
    """
    await websocket.accept()
    _clients.add(websocket)
    logger.info("ws_client_connected total=%d", len(_clients))

    try:
        # Keep the connection alive; client can send pings
        while True:
            # Wait for any message from client (ping/pong keep-alive)
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("ws_client_error", exc_info=True)
    finally:
        _clients.discard(websocket)
        logger.info("ws_client_disconnected total=%d", len(_clients))
