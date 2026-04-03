"""Databricks integration API routes.

Exposes Delta Lake, MLflow, Spark analytics, and FAISS vector
search endpoints for the frontend Databricks Insights dashboard.
All endpoints are gated behind ``databricks.enabled`` — they
return 503 when the integration is disabled.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/databricks", tags=["databricks"])


# ── Request / Response Models ─────────────────────────────────────


class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 20
    threshold: float = 0.65
    camera_id: str | None = None


class VectorTimeRangeRequest(BaseModel):
    query: str
    start_ts: float
    end_ts: float
    top_k: int = 20
    threshold: float = 0.65


class AnalyticsRequest(BaseModel):
    query_type: str
    hours_back: int = 24
    top_n: int = 20
    min_cameras: int = 2
    bucket_hours: int = 1
    bucket_minutes: int = 5
    z_threshold: float = 2.0


# ── Helpers ───────────────────────────────────────────────────────


def _require_databricks(request: Request, component: str) -> Any:
    """Retrieve a Databricks service from app state, or raise 503."""
    svc = getattr(request.app.state, f"databricks_{component}", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail=f"Databricks {component} integration is not enabled",
        )
    return svc


# ── Overview ──────────────────────────────────────────────────────


@router.get("/overview")
async def databricks_overview(request: Request) -> JSONResponse:
    """Return a combined status overview of all Databricks integrations."""
    result: dict[str, Any] = {"enabled": True}

    delta = getattr(request.app.state, "databricks_delta", None)
    if delta:
        try:
            result["delta"] = delta.get_table_stats()
        except Exception:
            result["delta"] = {"status": "error"}

    mlflow = getattr(request.app.state, "databricks_mlflow", None)
    if mlflow:
        try:
            result["mlflow"] = mlflow.get_experiment_summary()
        except Exception:
            result["mlflow"] = {"status": "error"}

    spark = getattr(request.app.state, "databricks_spark", None)
    if spark:
        try:
            result["spark"] = spark.get_analytics_overview()
        except Exception:
            result["spark"] = {"status": "error"}

    vector = getattr(request.app.state, "databricks_vector", None)
    if vector:
        try:
            result["vector"] = vector.get_stats()
        except Exception:
            result["vector"] = {"status": "error"}

    return JSONResponse(result)


# ── Delta Lake ────────────────────────────────────────────────────


@router.get("/delta/stats")
async def delta_stats(request: Request) -> JSONResponse:
    """Return Delta Lake table statistics."""
    delta = _require_databricks(request, "delta")
    try:
        stats = delta.get_table_stats()
        audit = delta.get_audit_stats()
        return JSONResponse({"detections": stats, "audits": audit})
    except Exception as exc:
        logger.error("delta_stats_error err=%s", exc, exc_info=True)
        raise HTTPException(500, detail=str(exc)) from exc


@router.get("/delta/history")
async def delta_history(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> JSONResponse:
    """Return Delta Lake version history."""
    delta = _require_databricks(request, "delta")
    try:
        history = delta.get_version_history(limit=limit)
        return JSONResponse({"versions": history})
    except Exception as exc:
        logger.error("delta_history_error err=%s", exc, exc_info=True)
        raise HTTPException(500, detail=str(exc)) from exc


@router.get("/delta/recent")
async def delta_recent(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
) -> JSONResponse:
    """Return recent detections from Delta Lake."""
    delta = _require_databricks(request, "delta")
    try:
        records = delta.read_recent(limit=limit)
        return JSONResponse({"records": records, "count": len(records)})
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc


# ── MLflow ────────────────────────────────────────────────────────


@router.get("/mlflow/summary")
async def mlflow_summary(request: Request) -> JSONResponse:
    """Return MLflow experiment summary."""
    mlflow = _require_databricks(request, "mlflow")
    try:
        summary = mlflow.get_experiment_summary()
        return JSONResponse(summary)
    except Exception as exc:
        logger.error("mlflow_summary_error err=%s", exc, exc_info=True)
        raise HTTPException(500, detail=str(exc)) from exc


@router.get("/mlflow/metrics/{metric_name}")
async def mlflow_metric_history(
    request: Request,
    metric_name: str,
    limit: int = Query(default=50, ge=1, le=500),
) -> JSONResponse:
    """Return metric history for a specific MLflow metric."""
    mlflow = _require_databricks(request, "mlflow")
    try:
        history = mlflow.get_metric_history(metric_name, limit=limit)
        return JSONResponse({"metric": metric_name, "history": history})
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc


# ── Spark Analytics ───────────────────────────────────────────────


@router.post("/spark/analytics")
async def spark_analytics(request: Request, body: AnalyticsRequest) -> JSONResponse:
    """Run a Spark analytics query."""
    spark = _require_databricks(request, "spark")
    query_map = {
        "hourly_rollup": lambda: spark.hourly_rollup(hours_back=body.hours_back),
        "plate_frequency": lambda: spark.plate_frequency(top_n=body.top_n),
        "cross_camera": lambda: spark.cross_camera_correlation(min_cameras=body.min_cameras),
        "confidence_trend": lambda: spark.confidence_trend(bucket_hours=body.bucket_hours),
        "camera_performance": lambda: spark.camera_performance(),
        "temporal_pattern": lambda: spark.temporal_pattern(),
        "anomaly_detection": lambda: spark.anomaly_detection(z_threshold=body.z_threshold),
        "detection_rate": lambda: spark.detection_rate(bucket_minutes=body.bucket_minutes),
    }

    handler = query_map.get(body.query_type)
    if handler is None:
        raise HTTPException(
            400,
            detail=f"Unknown query_type: {body.query_type}. Valid: {', '.join(query_map.keys())}",
        )

    try:
        results = handler()
        return JSONResponse(
            {
                "query_type": body.query_type,
                "results": results,
                "count": len(results),
            }
        )
    except Exception as exc:
        logger.error("spark_analytics_error query=%s err=%s", body.query_type, exc, exc_info=True)
        raise HTTPException(500, detail=str(exc)) from exc


@router.get("/spark/overview")
async def spark_overview(request: Request) -> JSONResponse:
    """Return combined Spark analytics overview."""
    spark = _require_databricks(request, "spark")
    try:
        overview = spark.get_analytics_overview()
        return JSONResponse(overview)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc


# ── FAISS Vector Search ──────────────────────────────────────────


@router.post("/vector/search")
async def vector_search(request: Request, body: VectorSearchRequest) -> JSONResponse:
    """Search for similar plates using FAISS vector similarity."""
    vector = _require_databricks(request, "vector")
    try:
        if body.camera_id:
            results = vector.search_by_camera(
                body.query,
                body.camera_id,
                top_k=body.top_k,
            )
        else:
            results = vector.search_similar_plates(
                body.query,
                top_k=body.top_k,
                threshold=body.threshold,
            )
        return JSONResponse(
            {
                "query": body.query,
                "results": results,
                "count": len(results),
            }
        )
    except Exception as exc:
        logger.error("vector_search_error query=%s err=%s", body.query, exc, exc_info=True)
        raise HTTPException(500, detail=str(exc)) from exc


@router.get("/vector/stats")
async def vector_stats(request: Request) -> JSONResponse:
    """Return FAISS vector index statistics."""
    vector = _require_databricks(request, "vector")
    try:
        stats = vector.get_stats()
        return JSONResponse(stats)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc


@router.get("/vector/duplicates")
async def vector_duplicates(
    request: Request,
    threshold: float = Query(default=0.85, ge=0.5, le=1.0),
) -> JSONResponse:
    """Find potential OCR duplicate plates via similarity."""
    vector = _require_databricks(request, "vector")
    try:
        dupes = vector.find_potential_duplicates(similarity_threshold=threshold)
        return JSONResponse({"duplicates": dupes, "count": len(dupes)})
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc


@router.post("/vector/search/time-range")
async def vector_time_range_search(request: Request, body: VectorTimeRangeRequest) -> JSONResponse:
    """Search for similar plates within a time window."""
    vector = _require_databricks(request, "vector")
    try:
        results = vector.search_by_time_range(
            body.query,
            start_ts=body.start_ts,
            end_ts=body.end_ts,
            top_k=body.top_k,
            threshold=body.threshold,
        )
        return JSONResponse(
            {
                "query": body.query,
                "time_range": {"start": body.start_ts, "end": body.end_ts},
                "results": results,
                "count": len(results),
            }
        )
    except Exception as exc:
        logger.error("vector_time_range_error err=%s", exc, exc_info=True)
        raise HTTPException(500, detail=str(exc)) from exc


@router.get("/vector/clusters")
async def vector_clusters(
    request: Request,
    n_clusters: int = Query(default=8, ge=2, le=50),
) -> JSONResponse:
    """Run K-Means cluster analysis on the plate embedding space."""
    vector = _require_databricks(request, "vector")
    try:
        analysis = vector.get_cluster_analysis(n_clusters=n_clusters)
        return JSONResponse(analysis)
    except Exception as exc:
        logger.error("vector_cluster_error err=%s", exc, exc_info=True)
        raise HTTPException(500, detail=str(exc)) from exc


# ── Delta Lake Maintenance ────────────────────────────────────────


@router.post("/delta/compact")
async def delta_compact(request: Request) -> JSONResponse:
    """Compact Delta Lake Parquet files for better read performance."""
    delta = _require_databricks(request, "delta")
    try:
        result = delta.compact()
        return JSONResponse(result)
    except Exception as exc:
        logger.error("delta_compact_error err=%s", exc, exc_info=True)
        raise HTTPException(500, detail=str(exc)) from exc


# ── Health ────────────────────────────────────────────────────────


@router.get("/health")
async def databricks_health(request: Request) -> JSONResponse:
    """Return health status for all Databricks components."""
    health: dict[str, Any] = {"overall": "ok"}

    for name in ("delta", "mlflow", "vector"):
        svc = getattr(request.app.state, f"databricks_{name}", None)
        if svc and hasattr(svc, "get_health"):
            try:
                health[name] = svc.get_health()
            except Exception as exc:
                health[name] = {"status": "error", "error": str(exc)}
                health["overall"] = "degraded"
        else:
            health[name] = {"status": "disabled"}

    spark = getattr(request.app.state, "databricks_spark", None)
    if spark:
        try:
            health["spark"] = spark.get_analytics_overview()
        except Exception as exc:
            health["spark"] = {"status": "error", "error": str(exc)}
            health["overall"] = "degraded"
    else:
        health["spark"] = {"status": "disabled"}

    return JSONResponse(health)
