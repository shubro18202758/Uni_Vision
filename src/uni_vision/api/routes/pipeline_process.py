"""Pipeline processing endpoint — bridges uploaded videos to the live pipeline.

Provides REST endpoints to start, monitor, and stop video processing
jobs.  Each job creates an ``RTSPFrameSource`` (which supports local
file paths via OpenCV) and a ``TemporalSampler`` that feeds frames
into the running ``Pipeline`` instance.  Real-time progress is
broadcast over the ``/ws/pipeline`` WebSocket via the existing
``PipelineEventBroadcaster``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from uni_vision.contracts.dtos import CameraSource
from uni_vision.ingestion.rtsp_source import RTSPFrameSource
from uni_vision.ingestion.sampler import TemporalSampler

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = logging.getLogger(__name__)

# Active processing jobs (in-memory; fine for single-process server)
_active_jobs: Dict[str, Dict[str, Any]] = {}


# ── Request / response schemas ────────────────────────────────────

class ProcessRequest(BaseModel):
    source_url: str
    camera_id: str = ""
    location_tag: str = ""
    fps_target: int = 3


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/process", status_code=202)
async def start_processing(
    request: Request,
    body: ProcessRequest,
) -> Dict[str, Any]:
    """Start processing a video file through the vision pipeline.

    Creates an ``RTSPFrameSource`` for the video file and feeds frames
    through the pipeline stages with real-time WebSocket events.
    Returns immediately; processing runs in background threads.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not started")

    # Validate source file exists
    source_path = Path(body.source_url)
    if not source_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found: {body.source_url}",
        )

    # Generate identifiers
    job_id = uuid.uuid4().hex[:12]
    camera_id = body.camera_id or f"video-{job_id}"

    # Build CameraSource DTO
    camera = CameraSource(
        camera_id=camera_id,
        source_url=str(source_path.resolve()),
        location_tag=body.location_tag or "upload",
        fps_target=body.fps_target,
        enabled=True,
    )

    # Create frame source (max_reconnect=0 → stop on EOF)
    source = RTSPFrameSource(camera, max_reconnect_attempts=0)
    try:
        source.start()
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to open video file: {exc}",
        ) from exc

    # Create temporal sampler wired to the running pipeline
    loop = asyncio.get_running_loop()
    sampler = TemporalSampler(
        sources=[source],
        enqueue_callback=pipeline.enqueue_frame,
        event_loop=loop,
        hamming_threshold=5,
    )
    sampler.start()

    # Track the job
    _active_jobs[job_id] = {
        "job_id": job_id,
        "camera_id": camera_id,
        "source_url": str(source_path),
        "status": "processing",
        "source": source,
        "sampler": sampler,
    }

    # Background task: auto-cleanup when video ends
    asyncio.create_task(_wait_for_completion(job_id, source, sampler))

    logger.info(
        "video_processing_started job_id=%s camera_id=%s source=%s",
        job_id,
        camera_id,
        str(source_path),
    )

    return {
        "job_id": job_id,
        "camera_id": camera_id,
        "source_url": str(source_path),
        "status": "processing",
    }


@router.get("/process/status")
async def list_jobs() -> List[Dict[str, Any]]:
    """List all active and recently completed processing jobs."""
    return [
        {
            "job_id": j["job_id"],
            "camera_id": j["camera_id"],
            "source_url": j["source_url"],
            "status": j["status"],
        }
        for j in _active_jobs.values()
    ]


@router.delete("/process/{job_id}")
async def stop_job(job_id: str) -> Dict[str, str]:
    """Stop an active processing job."""
    job = _active_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    _cleanup_job(job)
    return {"job_id": job_id, "status": "stopped"}


@router.delete("/process")
async def stop_all_jobs() -> Dict[str, Any]:
    """Stop all active processing jobs."""
    stopped = []
    for job_id, job in list(_active_jobs.items()):
        if job["status"] == "processing":
            _cleanup_job(job)
            stopped.append(job_id)
    return {"stopped": stopped, "count": len(stopped)}


# ── Internal helpers ──────────────────────────────────────────────

def _cleanup_job(job: Dict[str, Any]) -> None:
    """Stop source and sampler for a single job."""
    sampler = job.get("sampler")
    source = job.get("source")
    if sampler is not None:
        sampler.stop()
    if source is not None:
        source.release()
    job["status"] = "stopped"


async def _wait_for_completion(
    job_id: str,
    source: RTSPFrameSource,
    sampler: TemporalSampler,
) -> None:
    """Background coroutine — waits for video EOF then cleans up."""
    max_wait_s = 600  # 10-minute safety timeout
    elapsed = 0.0
    try:
        # Poll until the reader thread exits (EOF or error)
        while elapsed < max_wait_s:
            thread = getattr(source, "_reader_thread", None)
            if thread is None or not thread.is_alive():
                break
            await asyncio.sleep(0.5)
            elapsed += 0.5

        if elapsed >= max_wait_s:
            logger.warning("video_processing_timeout job_id=%s", job_id)

        # Allow remaining queued frames to flush through the sampler
        await asyncio.sleep(2.0)

        try:
            sampler.stop()
        except Exception:
            logger.debug("sampler_stop_error job_id=%s", job_id, exc_info=True)
        try:
            source.release()
        except Exception:
            logger.debug("source_release_error job_id=%s", job_id, exc_info=True)

        job = _active_jobs.get(job_id)
        if job is not None:
            job["status"] = "completed" if elapsed < max_wait_s else "timeout"

        logger.info("video_processing_completed job_id=%s elapsed=%.0fs", job_id, elapsed)
    except Exception:
        logger.exception("video_processing_error job_id=%s", job_id)
        job = _active_jobs.get(job_id)
        if job is not None:
            job["status"] = "error"
