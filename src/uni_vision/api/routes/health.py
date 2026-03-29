"""GET /health — system-wide liveness and readiness probe."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> JSONResponse:
    """Return aggregated health status.

    When the ``HealthService`` is available in app state (i.e. the
    full pipeline is running), a deep check is performed.  Otherwise
    a lightweight 200 with ``{"healthy": true}`` is returned — useful
    during development and integration testing.
    """
    health_svc = getattr(request.app.state, "health_service", None)
    if health_svc is None:
        return JSONResponse({"healthy": True, "mode": "standalone"})

    status = await health_svc.check()
    payload: Dict[str, Any] = asdict(status)
    code = 200 if status.healthy else 503
    return JSONResponse(payload, status_code=code)
