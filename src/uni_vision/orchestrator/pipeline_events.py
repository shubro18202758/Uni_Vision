"""Real-time pipeline event broadcaster for processing visibility.

Emits per-stage events during frame processing so the frontend
can display a live, transparent view of the CV pipeline.

Architecture:
  Pipeline._process_event() → PipelineEventBroadcaster.emit_*()
  PipelineEventBroadcaster → WebSocket clients via /ws/pipeline
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class PipelineEventType(str, Enum):
    FRAME_ACCEPTED = "frame_accepted"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    FRAME_PREVIEW = "frame_preview"
    PIPELINE_COMPLETE = "pipeline_complete"
    FLAG_RAISED = "flag_raised"
    PIPELINE_IDLE = "pipeline_idle"
    QUEUE_STATUS = "queue_status"
    ANALYSIS_RESULT = "analysis_result"
    # Manager Agent job lifecycle events
    JOB_CREATED = "job_created"
    JOB_PHASE_CHANGED = "job_phase_changed"
    COMPONENT_PROVISIONED = "component_provisioned"
    JOB_FLUSHING = "job_flushing"
    JOB_FLUSH_COMPLETE = "job_flush_complete"


# Ordered stage definitions for the generic CV analysis pipeline
PIPELINE_STAGES = [
    {"index": 0, "id": "S0_ingest", "name": "Frame Ingestion", "description": "Accept frame from video source into inference queue"},
    {"index": 1, "id": "S1_preprocess", "name": "Preprocessing", "description": "Frame quality enhancement, denoising, and normalisation"},
    {"index": 2, "id": "S2_scene_analysis", "name": "Scene Analysis", "description": "LLM vision analysis — object inventory and scene understanding"},
    {"index": 3, "id": "S3_anomaly_detection", "name": "Anomaly Detection", "description": "Identify anomalies, deviations, and risk indicators"},
    {"index": 4, "id": "S4_deep_analysis", "name": "Deep Analysis", "description": "Chain-of-thought reasoning, risk & impact assessment"},
    {"index": 5, "id": "S5_results", "name": "Results & Dispatch", "description": "Aggregate findings, persist results, and notify"},
]


@dataclass
class PipelineEvent:
    """Single pipeline event for WebSocket broadcast."""
    type: str
    frame_id: str
    camera_id: str
    timestamp: float
    stage_id: Optional[str] = None
    stage_index: Optional[int] = None
    stage_name: Optional[str] = None
    stage_description: Optional[str] = None
    status: Optional[str] = None
    latency_ms: Optional[float] = None
    total_stages: int = len(PIPELINE_STAGES)
    details: Dict[str, Any] = field(default_factory=dict)
    thumbnail_b64: Optional[str] = None
    stages_definition: Optional[List[Dict]] = None
    queue_depth: Optional[int] = None
    detection: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(d, default=str)


def _encode_thumbnail(image: np.ndarray, max_width: int = 320, quality: int = 70) -> str:
    """Encode a CV2 image as a base64 JPEG thumbnail."""
    try:
        import cv2
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / w
            image = cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        logger.debug("thumbnail_encode_failed", exc_info=True)
        return ""


class PipelineEventBroadcaster:
    """Manages WebSocket clients and broadcasts pipeline stage events."""

    def __init__(self) -> None:
        self._clients: Set[Any] = set()  # WebSocket instances
        self._lock = asyncio.Lock()
        self._frames_processed: int = 0
        self._current_frame_id: Optional[str] = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def register(self, ws: Any) -> None:
        async with self._lock:
            self._clients.add(ws)
        logger.info("pipeline_ws_client_connected total=%d", len(self._clients))

    async def unregister(self, ws: Any) -> None:
        async with self._lock:
            self._clients.discard(ws)
        logger.info("pipeline_ws_client_disconnected total=%d", len(self._clients))

    async def _broadcast(self, event: PipelineEvent) -> None:
        if not self._clients:
            return
        payload = event.to_json()
        dead: Set[Any] = set()
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self._clients.difference_update(dead)

    def generate_frame_id(self) -> str:
        self._frames_processed += 1
        fid = f"frm-{self._frames_processed:06d}-{uuid.uuid4().hex[:8]}"
        self._current_frame_id = fid
        return fid

    # ── Event emitters ────────────────────────────────────────────

    async def emit_frame_accepted(
        self,
        frame_id: str,
        camera_id: str,
        frame_index: int,
        queue_depth: int,
        image: Optional[np.ndarray] = None,
    ) -> None:
        thumbnail = ""
        if image is not None and self._clients:
            thumbnail = _encode_thumbnail(image, max_width=480, quality=60)

        event = PipelineEvent(
            type=PipelineEventType.FRAME_ACCEPTED.value,
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=time.time(),
            stage_id="S0_ingest",
            stage_index=0,
            stage_name="Frame Ingestion",
            status=StageStatus.COMPLETED.value,
            queue_depth=queue_depth,
            stages_definition=PIPELINE_STAGES,
            details={
                "frame_index": frame_index,
                "queue_depth": queue_depth,
                "total_frames_processed": self._frames_processed,
            },
            thumbnail_b64=thumbnail or None,
        )
        await self._broadcast(event)

    async def emit_stage_started(
        self,
        frame_id: str,
        camera_id: str,
        stage_id: str,
    ) -> None:
        stage_def = next((s for s in PIPELINE_STAGES if s["id"] == stage_id), None)
        event = PipelineEvent(
            type=PipelineEventType.STAGE_STARTED.value,
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=time.time(),
            stage_id=stage_id,
            stage_index=stage_def["index"] if stage_def else None,
            stage_name=stage_def["name"] if stage_def else stage_id,
            stage_description=stage_def["description"] if stage_def else None,
            status=StageStatus.RUNNING.value,
        )
        await self._broadcast(event)

    async def emit_stage_completed(
        self,
        frame_id: str,
        camera_id: str,
        stage_id: str,
        latency_ms: float,
        details: Optional[Dict[str, Any]] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        thumbnail = None
        if image is not None and self._clients:
            thumbnail = _encode_thumbnail(image, max_width=320) or None

        stage_def = next((s for s in PIPELINE_STAGES if s["id"] == stage_id), None)
        event = PipelineEvent(
            type=PipelineEventType.STAGE_COMPLETED.value,
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=time.time(),
            stage_id=stage_id,
            stage_index=stage_def["index"] if stage_def else None,
            stage_name=stage_def["name"] if stage_def else stage_id,
            stage_description=stage_def["description"] if stage_def else None,
            status=StageStatus.COMPLETED.value,
            latency_ms=round(latency_ms, 2),
            details=details or {},
            thumbnail_b64=thumbnail,
        )
        await self._broadcast(event)

    async def emit_stage_failed(
        self,
        frame_id: str,
        camera_id: str,
        stage_id: str,
        latency_ms: float,
        error: str,
    ) -> None:
        stage_def = next((s for s in PIPELINE_STAGES if s["id"] == stage_id), None)
        event = PipelineEvent(
            type=PipelineEventType.STAGE_COMPLETED.value,
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=time.time(),
            stage_id=stage_id,
            stage_index=stage_def["index"] if stage_def else None,
            stage_name=stage_def["name"] if stage_def else stage_id,
            status=StageStatus.FAILED.value,
            latency_ms=round(latency_ms, 2),
            details={"error": error},
        )
        await self._broadcast(event)

    async def emit_pipeline_complete(
        self,
        frame_id: str,
        camera_id: str,
        total_latency_ms: float,
        detections_count: int,
    ) -> None:
        event = PipelineEvent(
            type=PipelineEventType.PIPELINE_COMPLETE.value,
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=time.time(),
            status="complete",
            latency_ms=round(total_latency_ms, 2),
            details={
                "detections_count": detections_count,
                "total_frames_processed": self._frames_processed,
            },
        )
        await self._broadcast(event)

    async def emit_flag_raised(
        self,
        frame_id: str,
        camera_id: str,
        detection: Dict[str, Any],
        validation_status: str,
    ) -> None:
        event = PipelineEvent(
            type=PipelineEventType.FLAG_RAISED.value,
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=time.time(),
            status="flagged",
            details={
                "validation_status": validation_status,
                "scene_description": detection.get("scene_description", ""),
                "confidence": detection.get("confidence", 0.0),
                "risk_level": detection.get("risk_level", "unknown"),
                "anomaly_detected": detection.get("anomaly_detected", False),
            },
            detection=detection,
        )
        await self._broadcast(event)

    async def emit_queue_status(
        self,
        queue_depth: int,
        throttled: bool,
    ) -> None:
        event = PipelineEvent(
            type=PipelineEventType.QUEUE_STATUS.value,
            frame_id="",
            camera_id="",
            timestamp=time.time(),
            queue_depth=queue_depth,
            details={
                "throttled": throttled,
                "total_frames_processed": self._frames_processed,
            },
        )
        await self._broadcast(event)

    async def emit_analysis_result(
        self,
        frame_id: str,
        camera_id: str,
        analysis: Dict[str, Any],
        image: Optional[np.ndarray] = None,
    ) -> None:
        """Broadcast a structured anomaly analysis result for a frame."""
        thumbnail = None
        if image is not None and self._clients:
            thumbnail = _encode_thumbnail(image, max_width=480, quality=70) or None

        event = PipelineEvent(
            type=PipelineEventType.ANALYSIS_RESULT.value,
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=time.time(),
            status="flagged" if analysis.get("anomaly_detected") else "complete",
            details=analysis,
            thumbnail_b64=thumbnail,
        )
        await self._broadcast(event)

    async def emit_custom(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Broadcast a custom event (job lifecycle, component status, etc.)."""
        event = PipelineEvent(
            type=event_type,
            frame_id="",
            camera_id=data.get("camera_id", ""),
            timestamp=time.time(),
            details=data,
        )
        await self._broadcast(event)


# Module-level singleton
pipeline_broadcaster = PipelineEventBroadcaster()
