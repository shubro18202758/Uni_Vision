"""Camera source CRUD — POST / GET / DELETE /sources."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/sources", tags=["sources"])


# ── Request / response schemas ────────────────────────────────────


class CameraSourceIn(BaseModel):
    camera_id: str = Field(..., min_length=1, max_length=128)
    source_url: str = Field(..., min_length=1)
    location_tag: str = ""
    fps_target: int = Field(default=3, ge=1, le=60)
    enabled: bool = True


class CameraSourceOut(BaseModel):
    camera_id: str
    source_url: str
    location_tag: str
    fps_target: int
    enabled: bool
    added_at: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────


async def _get_pool(request: Request):  # type: ignore[no-untyped-def]
    """Acquire (or lazily create) the asyncpg connection pool."""
    pg = getattr(request.app.state, "pg_client", None)
    if pg is not None:
        return pg

    # Import lazily to avoid hard dependency at module scope.
    from uni_vision.storage.postgres import PostgresClient
    from uni_vision.common.config import AppConfig

    config: AppConfig = request.app.state.config
    pg = PostgresClient(config.database, config.dispatch)
    await pg.connect()
    request.app.state.pg_client = pg
    return pg


# ── Endpoints ─────────────────────────────────────────────────────


@router.post("", status_code=201)
async def register_source(body: CameraSourceIn, request: Request) -> Dict[str, Any]:
    """Register or update a camera source (upsert)."""
    pg = await _get_pool(request)
    pool = pg._pool  # noqa: SLF001 — direct pool access for lightweight queries
    assert pool is not None

    from uni_vision.storage.models import INSERT_CAMERA_SOURCE_SQL

    async with pool.acquire() as conn:
        await conn.execute(
            INSERT_CAMERA_SOURCE_SQL,
            body.camera_id,
            body.source_url,
            body.location_tag,
            body.fps_target,
            body.enabled,
        )
    return {"camera_id": body.camera_id, "status": "registered"}


@router.get("", response_model=List[CameraSourceOut])
async def list_sources(request: Request) -> List[Dict[str, Any]]:
    """Return all enabled camera sources ordered by registration date."""
    try:
        pg = await _get_pool(request)
    except Exception:
        # Database not available — return empty list instead of 500
        return []
    pool = getattr(pg, "_pool", None)
    if pool is None:
        return []

    from uni_vision.storage.models import SELECT_CAMERA_SOURCES_SQL

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(SELECT_CAMERA_SOURCES_SQL)
    except Exception:
        return []

    return [
        {
            "camera_id": r["camera_id"],
            "source_url": r["source_url"],
            "location_tag": r["location_tag"],
            "fps_target": r["fps_target"],
            "enabled": r["enabled"],
            "added_at": r["added_at"].isoformat() if r["added_at"] else None,
        }
        for r in rows
    ]


@router.delete("/{camera_id}", status_code=204)
async def delete_source(camera_id: str, request: Request) -> None:
    """Remove a camera source by ID."""
    pg = await _get_pool(request)
    pool = pg._pool  # noqa: SLF001
    assert pool is not None

    from uni_vision.storage.models import DELETE_CAMERA_SOURCE_SQL

    async with pool.acquire() as conn:
        result = await conn.execute(DELETE_CAMERA_SOURCE_SQL, camera_id)

    # asyncpg returns e.g. "DELETE 1" or "DELETE 0"
    if result.endswith("0"):
        raise HTTPException(status_code=404, detail="Camera source not found")
