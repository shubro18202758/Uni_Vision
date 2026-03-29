"""API middleware — request logging and error handling."""

from __future__ import annotations

import logging
import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs method, path, status code and latency for every request."""

    async def dispatch(
        self, request: Request, call_next: Callable  # type: ignore[type-arg]
    ) -> Response:
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_error method=%s path=%s",
                request.method,
                request.url.path,
            )
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request method=%s path=%s status=%d latency_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response