"""GET /detections — paginated query of detection events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

if TYPE_CHECKING:
    from datetime import datetime

router = APIRouter(prefix="/detections", tags=["detections"])


# ── Response schema ───────────────────────────────────────────────


class DetectionOut(BaseModel):
    id: str
    camera_id: str
    plate_number: str
    raw_ocr_text: str
    ocr_confidence: float
    ocr_engine: str
    vehicle_class: str
    vehicle_image_path: str
    plate_image_path: str
    detected_at_utc: str | None = None
    validation_status: str
    location_tag: str


class DetectionPage(BaseModel):
    items: list[DetectionOut]
    total: int
    page: int
    page_size: int


# ── Endpoint ──────────────────────────────────────────────────────

_BASE_QUERY = """\
SELECT id, camera_id, plate_number, raw_ocr_text,
       ocr_confidence, ocr_engine, vehicle_class,
       vehicle_image_path, plate_image_path,
       detected_at_utc, validation_status, location_tag
FROM detection_events
"""

_COUNT_QUERY = "SELECT count(*) FROM detection_events"


def _build_where(
    camera_id: str | None,
    plate_number: str | None,
    status: str | None,
    since: datetime | None,
    until: datetime | None,
) -> tuple[str, list[Any]]:
    """Return a WHERE clause fragment and matching positional params."""
    clauses: list[str] = []
    params: list[Any] = []
    idx = 1

    if camera_id:
        clauses.append(f"camera_id = ${idx}")
        params.append(camera_id)
        idx += 1
    if plate_number:
        clauses.append(f"plate_number ILIKE ${idx}")
        params.append(f"%{plate_number}%")
        idx += 1
    if status:
        clauses.append(f"validation_status = ${idx}")
        params.append(status)
        idx += 1
    if since:
        clauses.append(f"detected_at_utc >= ${idx}")
        params.append(since)
        idx += 1
    if until:
        clauses.append(f"detected_at_utc <= ${idx}")
        params.append(until)
        idx += 1

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


@router.get("", response_model=DetectionPage)
async def list_detections(
    request: Request,
    camera_id: str | None = Query(None),
    plate_number: str | None = Query(None),
    status: str | None = Query(None),
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
) -> dict[str, Any]:
    """Return a paginated list of detection events with optional filters."""
    pg = getattr(request.app.state, "pg_client", None)
    if pg is None:
        from uni_vision.storage.postgres import PostgresClient

        config = request.app.state.config
        pg = PostgresClient(config.database, config.dispatch)
        await pg.connect()
        request.app.state.pg_client = pg

    pool = pg._pool
    assert pool is not None

    where, params = _build_where(camera_id, plate_number, status, since, until)

    # Total count
    async with pool.acquire() as conn:
        total = await conn.fetchval(_COUNT_QUERY + where, *params)

    # Paginated rows
    offset = (page - 1) * page_size
    order_clause = " ORDER BY detected_at_utc DESC"
    limit_idx = len(params) + 1
    offset_idx = limit_idx + 1
    paginated_sql = _BASE_QUERY + where + order_clause + f" LIMIT ${limit_idx} OFFSET ${offset_idx}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(paginated_sql, *params, page_size, offset)

    items = [
        {
            "id": r["id"],
            "camera_id": r["camera_id"],
            "plate_number": r["plate_number"],
            "raw_ocr_text": r["raw_ocr_text"],
            "ocr_confidence": r["ocr_confidence"],
            "ocr_engine": r["ocr_engine"],
            "vehicle_class": r["vehicle_class"],
            "vehicle_image_path": r["vehicle_image_path"],
            "plate_image_path": r["plate_image_path"],
            "detected_at_utc": (r["detected_at_utc"].isoformat() if r["detected_at_utc"] else None),
            "validation_status": r["validation_status"],
            "location_tag": r["location_tag"],
        }
        for r in rows
    ]

    return {"items": items, "total": total, "page": page, "page_size": page_size}
