"""WebSocket endpoint for real-time pipeline processing visibility.

Clients connect to ``/ws/pipeline`` and receive JSON messages for
every stage transition as frames flow through the CV pipeline.

This gives users full transparency into what the backend is doing:
frame ingestion, vehicle detection, plate localisation, OCR, etc.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from uni_vision.orchestrator.pipeline_events import pipeline_broadcaster

router = APIRouter(tags=["websocket"])

logger = logging.getLogger(__name__)


@router.websocket("/ws/pipeline")
async def websocket_pipeline(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time pipeline stage event streaming.

    Clients receive per-stage JSON events including:
    - frame_accepted: New frame enters pipeline with thumbnail
    - stage_started / stage_completed: Per-stage timing and details
    - frame_preview: Intermediate image thumbnails (base64 JPEG)
    - pipeline_complete: Full frame processing finished
    - flag_raised: Anomaly detected, triggers chain-of-thought analysis
    - queue_status: Queue depth and throttle state
    """
    await websocket.accept()
    await pipeline_broadcaster.register(websocket)

    try:
        while True:
            try:
                # Wait for client message (ping) with a 30s timeout
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # No client message received — send heartbeat to keep connection alive
                # and detect dead clients (send will raise if connection is gone)
                try:
                    await websocket.send_text(json.dumps({"type": "heartbeat", "timestamp": time.time()}))
                except Exception:
                    break  # Connection is dead, exit loop
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("pipeline_ws_client_error", exc_info=True)
    finally:
        await pipeline_broadcaster.unregister(websocket)
