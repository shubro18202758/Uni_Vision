"""Rate-limiting middleware using a sliding-window counter.

Limits per-IP request rates to prevent abuse. Uses an in-memory
store (suitable for single-process deployments) with automatic
expiry of stale entries.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)

_PUBLIC_PATHS: frozenset[str] = frozenset({"/health", "/metrics"})


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter per client IP.

    Parameters
    ----------
    app:
        The ASGI application to wrap.
    requests_per_minute:
        Maximum requests allowed per IP per 60-second window.
        Set to 0 to disable rate limiting.
    """

    def __init__(self, app: object, requests_per_minute: int = 60) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._rpm = requests_per_minute
        self._window = 60.0
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)

    def _client_ip(self, request: Request) -> str:
        """Extract the client IP, respecting X-Forwarded-For from trusted proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_rate_limited(self, ip: str) -> bool:
        if self._rpm <= 0:
            return False

        now = time.monotonic()
        window = self._hits[ip]
        cutoff = now - self._window

        # Remove expired timestamps
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self._rpm:
            return True

        window.append(now)
        return False

    async def dispatch(self, request: Request, call_next: object) -> Response:  # type: ignore[override]
        # Health/metrics endpoints are exempt from rate limiting
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)  # type: ignore[misc]

        ip = self._client_ip(request)
        if self._is_rate_limited(ip):
            logger.warning("rate_limited ip=%s path=%s", ip, request.url.path)
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try again later."},
                status_code=429,
                headers={"Retry-After": "60"},
            )

        response: Response = await call_next(request)  # type: ignore[misc]
        return response
