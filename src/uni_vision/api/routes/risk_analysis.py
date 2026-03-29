"""Risk-analysis & flag-reasoning API endpoints.

GET /risk-analysis/{detection_id}  → full risk analysis
GET /flag-reasoning/{detection_id} → flag reasoning chain
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from uni_vision.postprocessing.flag_reasoning import FlagReasoningEngine
from uni_vision.postprocessing.impact_analysis import ImpactAnalysisEngine
from uni_vision.postprocessing.risk_analysis import RiskAnalysisEngine

router = APIRouter(prefix="/api/analysis", tags=["risk-analysis"])

_reasoning_engine = FlagReasoningEngine()
_risk_engine = RiskAnalysisEngine()
_impact_engine = ImpactAnalysisEngine()


# ── Helpers ───────────────────────────────────────────────────────

async def _get_detection(request: Request, detection_id: str) -> Dict[str, Any]:
    """Fetch a single detection row from PostgreSQL."""
    pg = getattr(request.app.state, "pg_client", None)
    if pg is None:
        from uni_vision.storage.postgres import PostgresClient
        config = request.app.state.config
        pg = PostgresClient(config.database, config.dispatch)
        await pg.connect()
        request.app.state.pg_client = pg

    pool = pg._pool  # noqa: SLF001
    assert pool is not None

    sql = """\
    SELECT id, camera_id, plate_number, raw_ocr_text,
           ocr_confidence, ocr_engine, vehicle_class,
           vehicle_image_path, plate_image_path,
           detected_at_utc, validation_status, location_tag
    FROM detection_events WHERE id = $1
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, detection_id)

    if row is None:
        raise HTTPException(status_code=404, detail="Detection not found")

    return dict(row)


async def _get_recent_detections(
    request: Request, camera_id: str, limit: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch recent detections from the same camera for context."""
    pg = getattr(request.app.state, "pg_client", None)
    if pg is None:
        return []

    pool = pg._pool  # noqa: SLF001
    if pool is None:
        return []

    sql = """\
    SELECT id, camera_id, plate_number, raw_ocr_text,
           ocr_confidence, ocr_engine, vehicle_class,
           validation_status, detected_at_utc
    FROM detection_events
    WHERE camera_id = $1
    ORDER BY detected_at_utc DESC
    LIMIT $2
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, camera_id, limit)

    return [
        {
            "id": r["id"],
            "camera_id": r["camera_id"],
            "plate_number": r["plate_number"],
            "raw_ocr_text": r["raw_ocr_text"],
            "ocr_confidence": float(r["ocr_confidence"]) if r["ocr_confidence"] else 0.0,
            "ocr_engine": r["ocr_engine"],
            "vehicle_class": r["vehicle_class"],
            "validation_status": r["validation_status"],
            "detected_at_utc": (
                r["detected_at_utc"].isoformat() if r["detected_at_utc"] else None
            ),
        }
        for r in rows
    ]


def _build_telemetry_stub() -> Dict[str, Any]:
    """Basic telemetry stub when live metrics are not available."""
    return {
        "detection_latency_ms": 80,
        "ocr_latency_ms": 150,
        "postprocess_latency_ms": 30,
        "adjudication_latency_ms": 800,
        "pipeline_latency_ms": 1200,
        "vram_utilisation_pct": 65,
        "component_error_rate": 0.02,
    }


# ── Flag Reasoning ────────────────────────────────────────────────

@router.get("/flag-reasoning/{detection_id}")
async def get_flag_reasoning(request: Request, detection_id: str) -> Dict[str, Any]:
    """Return the reasoning chain explaining why a detection was flagged."""
    det = await _get_detection(request, detection_id)

    if det["validation_status"] == "valid":
        return {
            "detection_id": detection_id,
            "flagged": False,
            "message": "This detection passed all validation checks — no flag was raised.",
        }

    recent = await _get_recent_detections(request, det["camera_id"])
    telemetry = _build_telemetry_stub()

    reasoning = _reasoning_engine.generate(
        detection_id=detection_id,
        validation_status=det["validation_status"],
        ocr_confidence=float(det["ocr_confidence"]) if det["ocr_confidence"] else 0.0,
        ocr_engine=det["ocr_engine"] or "unknown",
        camera_id=det["camera_id"],
        plate_number=det["plate_number"],
        raw_ocr_text=det["raw_ocr_text"] or "",
        vehicle_class=det["vehicle_class"] or "unknown",
        char_corrections=None,
        pipeline_telemetry=telemetry,
    )

    data = reasoning.to_dict()
    data["flagged"] = True
    return data


# ── Risk Analysis ─────────────────────────────────────────────────

@router.get("/risk-analysis/{detection_id}")
async def get_risk_analysis(request: Request, detection_id: str) -> Dict[str, Any]:
    """Return full multi-dimensional risk analysis for a detection."""
    det = await _get_detection(request, detection_id)

    recent = await _get_recent_detections(request, det["camera_id"])
    telemetry = _build_telemetry_stub()

    analysis = _risk_engine.analyze(
        detection_id=detection_id,
        validation_status=det["validation_status"],
        ocr_confidence=float(det["ocr_confidence"]) if det["ocr_confidence"] else 0.0,
        ocr_engine=det["ocr_engine"] or "unknown",
        camera_id=det["camera_id"],
        plate_number=det["plate_number"],
        raw_ocr_text=det["raw_ocr_text"] or "",
        vehicle_class=det["vehicle_class"] or "unknown",
        detected_at=det["detected_at_utc"].isoformat() if det["detected_at_utc"] else "",
        char_corrections=None,
        recent_detections=recent,
        pipeline_telemetry=telemetry,
    )

    return analysis.to_dict()


# ── Impact Analysis ───────────────────────────────────────────────

@router.get("/impact-analysis/{detection_id}")
async def get_impact_analysis(request: Request, detection_id: str) -> Dict[str, Any]:
    """Return exhaustive impact analysis for a flagged detection."""
    det = await _get_detection(request, detection_id)

    recent = await _get_recent_detections(request, det["camera_id"])
    telemetry = _build_telemetry_stub()

    analysis = _impact_engine.analyze(
        detection_id=detection_id,
        validation_status=det["validation_status"],
        plate_number=det["plate_number"],
        ocr_confidence=float(det["ocr_confidence"]) if det["ocr_confidence"] else 0.0,
        camera_id=det["camera_id"],
        telemetry=telemetry,
        recent_detections=recent,
    )

    return analysis.to_dict()
