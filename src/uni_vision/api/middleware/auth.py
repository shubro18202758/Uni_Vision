"""API key authentication middleware.

Validates requests against a configurable set of API keys.
Health and metrics endpoints are excluded from authentication
to allow Kubernetes probes and Prometheus scraping.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from typing import FrozenSet, Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Paths that never require authentication
_PUBLIC_PATHS: FrozenSet[str] = frozenset({"/health", "/metrics", "/openapi.json", "/docs", "/redoc"})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid ``X-API-Key`` header.

    Parameters
    ----------
    app:
        The ASGI application to wrap.
    api_keys:
        Set of valid API key strings.  When empty, authentication is
        **disabled** (all requests pass through) — suitable for local dev.
    """

    def __init__(self, app: object, api_keys: Optional[Set[str]] = None) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._enabled = bool(api_keys)
        # Store SHA-256 hashes to avoid holding plain keys in memory
        self._key_hashes: FrozenSet[str] = frozenset(
            hashlib.sha256(k.encode()).hexdigest() for k in (api_keys or set())
        )

    async def dispatch(self, request: Request, call_next: object) -> Response:  # type: ignore[override]
        # Skip auth for public endpoints
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)  # type: ignore[misc]

        if not self._enabled:
            return await call_next(request)  # type: ignore[misc]

        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            return JSONResponse(
                {"detail": "Missing X-API-Key header"},
                status_code=401,
            )

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if not any(hmac.compare_digest(key_hash, h) for h in self._key_hashes):
            logger.warning("api_key_rejected path=%s", request.url.path)
            return JSONResponse(
                {"detail": "Invalid API key"},
                status_code=403,
            )

        return await call_next(request)  # type: ignore[misc]


def generate_api_key() -> str:
    """Generate a cryptographically secure 32-byte hex API key."""
    return secrets.token_hex(32)
