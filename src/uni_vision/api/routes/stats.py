"""GET /stats — pipeline telemetry summary."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["stats"])


@router.get("/stats")
async def pipeline_stats(request: Request) -> JSONResponse:
    """Return aggregated pipeline statistics from Prometheus metrics.

    Reads the in-process metric values and returns a flat JSON summary.
    """
    from prometheus_client import REGISTRY

    stats: Dict[str, Any] = {}

    # Iterate through all metrics in the default registry
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            name = sample.name
            # Skip internal metrics
            if name.startswith("python_") or name.startswith("process_"):
                continue
            if not name.startswith("uv_"):
                continue

            labels = sample.labels
            value = sample.value

            if labels:
                key = f"{name}|{'|'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
            else:
                key = name

            # For histograms pick _count and _sum, skip individual buckets
            if name.endswith("_bucket"):
                continue

            stats[key] = value

    # Compute friendly summary keys
    active = sum(1 for k in stats if k.startswith("uv_frames_total|"))
    stats["active_streams"] = active
    stats["total_detections"] = stats.get("uv_detections_total_count", 0)
    stats["total_frames"] = stats.get("uv_frames_total_count", 0)

    return JSONResponse(stats)
