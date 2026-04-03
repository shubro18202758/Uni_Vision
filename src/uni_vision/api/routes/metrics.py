"""GET /metrics — Prometheus-compatible scrape endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def prometheus_metrics() -> PlainTextResponse:
    """Export all Prometheus counters, gauges and histograms in text format."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
