"""Video upload endpoint — accepts video files as pipeline data sources."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

router = APIRouter(prefix="/sources/upload", tags=["sources"])

# Allowed video MIME types → extensions
_ALLOWED_TYPES: dict[str, str] = {
    "video/mp4": ".mp4",
    "video/avi": ".avi",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-flv": ".flv",
    "video/mpeg": ".mpeg",
    "video/3gpp": ".3gp",
    "video/x-ms-wmv": ".wmv",
    "video/ogg": ".ogv",
    "application/octet-stream": "",  # fallback — validated by extension
}

_ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".flv", ".mpeg", ".mpg", ".3gp", ".wmv", ".ogv", ".ts"}

# 500 MB hard limit
_MAX_FILE_SIZE = 500 * 1024 * 1024

# Upload directory (relative to project root)
_UPLOAD_DIR = Path(__file__).resolve().parents[4] / "data" / "uploads" / "videos"

# Sanitise camera ID: alphanumeric, hyphens, underscores only
_CAMERA_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _safe_filename(original: str) -> str:
    """Sanitise an uploaded filename — strip path components, keep extension."""
    name = Path(original).name
    # Replace anything that isn't alphanumeric, hyphens, underscores, or dots
    name = re.sub(r"[^\w.\-]", "_", name)
    return name


@router.post("", status_code=201)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    camera_id: str = Form(default=""),
    location_tag: str = Form(default=""),
    fps_target: int = Form(default=3, ge=1, le=60),
) -> dict[str, Any]:
    """Upload a video file and register it as a camera source for the pipeline.

    The file is saved to ``data/uploads/videos/`` and a corresponding
    camera source entry is created so the existing RTSP/file ingestion
    pipeline can pick it up.
    """

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # ── Extension validation ──────────────────────────────────────
    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported video format '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    # ── Content-type validation (lenient for octet-stream) ────────
    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_TYPES and content_type != "":
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported content type '{content_type}'.",
        )

    # ── Generate safe identifiers ─────────────────────────────────
    upload_id = uuid.uuid4().hex[:12]
    safe_name = _safe_filename(file.filename)
    if not camera_id:
        camera_id = f"upload-{upload_id}"
    else:
        if not _CAMERA_ID_RE.match(camera_id):
            raise HTTPException(status_code=422, detail="Invalid camera_id (alphanumeric, hyphens, underscores only)")

    dest_filename = f"{upload_id}_{safe_name}"
    dest_path = _UPLOAD_DIR / dest_filename

    # ── Stream to disk with size guard ────────────────────────────
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 256)  # 256 KB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > _MAX_FILE_SIZE:
                    out.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {_MAX_FILE_SIZE // (1024 * 1024)} MB.",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    if total_bytes == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # ── Register as a camera source (file:// path) ───────────────
    source_url = str(dest_path.resolve())
    pg = getattr(request.app.state, "pg_client", None)
    registered = False

    if pg is not None and getattr(pg, "_pool", None) is not None:
        try:
            from uni_vision.storage.models import INSERT_CAMERA_SOURCE_SQL

            async with pg._pool.acquire() as conn:
                await conn.execute(
                    INSERT_CAMERA_SOURCE_SQL,
                    camera_id,
                    source_url,
                    location_tag,
                    fps_target,
                    True,
                )
            registered = True
        except Exception:
            # DB may not be available — still return success for the upload
            pass

    return {
        "camera_id": camera_id,
        "filename": dest_filename,
        "original_name": file.filename,
        "size_bytes": total_bytes,
        "source_url": source_url,
        "format": ext.lstrip("."),
        "fps_target": fps_target,
        "location_tag": location_tag,
        "db_registered": registered,
        "status": "uploaded",
    }


@router.get("")
async def list_uploads() -> list[dict[str, Any]]:
    """List all uploaded video files currently on disk."""
    if not _UPLOAD_DIR.exists():
        return []

    uploads = []
    for f in sorted(_UPLOAD_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in _ALLOWED_EXTENSIONS:
            uploads.append(
                {
                    "filename": f.name,
                    "size_bytes": f.stat().st_size,
                    "format": f.suffix.lstrip("."),
                    "source_url": str(f.resolve()),
                }
            )
    return uploads


@router.delete("/{filename}")
async def delete_upload(filename: str) -> dict[str, str]:
    """Delete an uploaded video file from disk."""
    # Prevent path traversal
    safe = Path(filename).name
    if safe != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    target = _UPLOAD_DIR / safe
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    target.unlink()
    return {"deleted": safe, "status": "ok"}
