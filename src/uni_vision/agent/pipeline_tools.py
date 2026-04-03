"""Concrete tool implementations for querying and controlling the pipeline.

These tools provide the agent with read/write access to the system:
  * Query detection history from PostgreSQL.
  * Get real-time pipeline statistics from Prometheus.
  * Manage cameras (list, enable/disable).
  * Adjust pipeline thresholds dynamically.
  * Perform system health checks.
  * Search the OCR audit log for failures.
  * Analyse detection patterns and anomalies.
  * Re-process detections with adjusted parameters.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from uni_vision.agent.tools import tool

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Detection Query Tools
# ══════════════════════════════════════════════════════════════════


@tool(
    name="query_detections",
    description=(
        "Query recent detection records from the database. "
        "Supports filtering by camera_id, identifier (partial match), "
        "validation_status, and time range. Returns up to 'limit' results."
    ),
    param_descriptions={
        "camera_id": "Filter by camera ID (exact match), empty for all cameras",
        "plate_number": "Filter by detection identifier (partial/ILIKE match), empty for all",
        "status": "Filter by validation status (valid, low_confidence, regex_fail, etc.)",
        "hours_back": "How many hours back to search (default: 1)",
        "limit": "Maximum number of results to return (default: 20, max: 100)",
    },
)
async def query_detections(
    camera_id: str = "",
    plate_number: str = "",
    status: str = "",
    hours_back: float = 1.0,
    limit: int = 20,
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Query detection records from PostgreSQL."""
    pg_client = _get_pg_client(context)
    if pg_client is None:
        return {"error": "Database not connected", "results": []}

    limit = min(max(1, limit), 100)

    clauses: List[str] = []
    params: List[Any] = []
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

    clauses.append(f"detected_at_utc >= NOW() - INTERVAL '{max(0.01, hours_back)} hours'")

    where = " WHERE " + " AND ".join(clauses) if clauses else ""

    query = (
        f"SELECT id, camera_id, plate_number, ocr_confidence, "
        f"ocr_engine, vehicle_class, validation_status, detected_at_utc "
        f"FROM detection_events {where} "
        f"ORDER BY detected_at_utc DESC LIMIT ${idx}"
    )
    params.append(limit)

    pool = getattr(pg_client, "_pool", None)
    if pool is None:
        return {"error": "Database pool not available", "results": []}

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    results = [dict(row) for row in rows]
    # Convert datetimes to strings for JSON serialization
    for r in results:
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                r[k] = v.isoformat()

    return {"count": len(results), "results": results}


@tool(
    name="get_detection_summary",
    description=(
        "Get aggregated detection statistics — total count, average confidence, "
        "breakdown by camera, by status, and by vehicle class for a time window."
    ),
    param_descriptions={
        "hours_back": "Time window in hours (default: 24)",
    },
)
async def get_detection_summary(
    hours_back: float = 24.0,
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Aggregate detection statistics."""
    pg_client = _get_pg_client(context)
    if pg_client is None:
        return {"error": "Database not connected"}

    pool = getattr(pg_client, "_pool", None)
    if pool is None:
        return {"error": "Database pool not available"}

    interval = f"{max(0.01, hours_back)} hours"

    async with pool.acquire() as conn:
        # Total count and average confidence
        row = await conn.fetchrow(
            "SELECT COUNT(*) as total, "
            "COALESCE(AVG(ocr_confidence), 0) as avg_confidence, "
            "COALESCE(MIN(ocr_confidence), 0) as min_confidence, "
            "COALESCE(MAX(ocr_confidence), 0) as max_confidence "
            f"FROM detection_events WHERE detected_at_utc >= NOW() - INTERVAL '{interval}'"
        )

        # By camera
        camera_rows = await conn.fetch(
            "SELECT camera_id, COUNT(*) as count, "
            "COALESCE(AVG(ocr_confidence), 0) as avg_conf "
            f"FROM detection_events WHERE detected_at_utc >= NOW() - INTERVAL '{interval}' "
            "GROUP BY camera_id ORDER BY count DESC LIMIT 20"
        )

        # By status
        status_rows = await conn.fetch(
            "SELECT validation_status, COUNT(*) as count "
            f"FROM detection_events WHERE detected_at_utc >= NOW() - INTERVAL '{interval}' "
            "GROUP BY validation_status ORDER BY count DESC"
        )

        # By vehicle class
        class_rows = await conn.fetch(
            "SELECT vehicle_class, COUNT(*) as count "
            f"FROM detection_events WHERE detected_at_utc >= NOW() - INTERVAL '{interval}' "
            "GROUP BY vehicle_class ORDER BY count DESC"
        )

    return {
        "time_window_hours": hours_back,
        "total_detections": row["total"] if row else 0,
        "avg_confidence": round(float(row["avg_confidence"]) if row else 0, 4),
        "min_confidence": round(float(row["min_confidence"]) if row else 0, 4),
        "max_confidence": round(float(row["max_confidence"]) if row else 0, 4),
        "by_camera": [
            {"camera_id": r["camera_id"], "count": r["count"],
             "avg_confidence": round(float(r["avg_conf"]), 4)}
            for r in camera_rows
        ],
        "by_status": [
            {"status": r["validation_status"], "count": r["count"]}
            for r in status_rows
        ],
        "by_vehicle_class": [
            {"class": r["vehicle_class"], "count": r["count"]}
            for r in class_rows
        ],
    }


# ══════════════════════════════════════════════════════════════════
# Pipeline Statistics Tools
# ══════════════════════════════════════════════════════════════════


@tool(
    name="get_pipeline_stats",
    description=(
        "Get real-time pipeline performance metrics including latency, "
        "throughput, queue depth, OCR confidence distribution, and error rates."
    ),
    param_descriptions={
        "metric_filter": "Optional filter — only return metrics containing this substring",
    },
)
async def get_pipeline_stats(
    metric_filter: str = "",
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Read Prometheus metrics from the in-process registry."""
    try:
        from prometheus_client import REGISTRY
    except ImportError:
        return {"error": "prometheus_client not available"}

    stats: Dict[str, Any] = {}

    for metric in REGISTRY.collect():
        for sample in metric.samples:
            name = sample.name
            if name.startswith("python_") or name.startswith("process_"):
                continue
            if not name.startswith("uv_"):
                continue
            if name.endswith("_bucket"):
                continue
            if metric_filter and metric_filter not in name:
                continue

            labels = sample.labels
            value = sample.value
            key = (
                f"{name}|{'|'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
                if labels
                else name
            )
            stats[key] = value

    return stats


@tool(
    name="get_system_health",
    description=(
        "Comprehensive system health check — pipeline status, queue depths, "
        "circuit breaker state, VRAM usage, database connectivity, and recent error rates."
    ),
)
async def get_system_health(
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Perform a system health check."""
    health: Dict[str, Any] = {
        "pipeline_running": False,
        "database_connected": False,
        "inference_queue_depth": 0,
        "circuit_breaker_state": "unknown",
        "vram_usage_mb": 0,
        "recent_errors": 0,
    }

    pipeline = _get_pipeline(context)
    if pipeline is not None:
        health["pipeline_running"] = not getattr(pipeline, "_shutting_down", True)
        health["inference_queue_depth"] = getattr(
            pipeline, "_inference_queue", asyncio.Queue()
        ).qsize()
        health["throttled"] = getattr(pipeline, "_throttled", False)

        # Circuit breaker state from OCR strategy
        ocr_strategy = getattr(pipeline, "_ocr_strategy", None)
        if ocr_strategy is not None:
            primary = getattr(ocr_strategy, "_primary", None)
            if primary is not None:
                cb_state = getattr(primary, "_cb_state", None)
                if cb_state is not None:
                    health["circuit_breaker_state"] = cb_state.value

    pg_client = _get_pg_client(context)
    if pg_client is not None:
        pool = getattr(pg_client, "_pool", None)
        health["database_connected"] = pool is not None

    # Read error metrics
    try:
        from prometheus_client import REGISTRY
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if "error" in sample.name and sample.name.startswith("uv_"):
                    health["recent_errors"] += int(sample.value)
    except ImportError:
        pass

    return health


# ══════════════════════════════════════════════════════════════════
# Camera Management Tools
# ══════════════════════════════════════════════════════════════════


@tool(
    name="list_cameras",
    description="List all registered camera sources with their status and configuration.",
)
async def list_cameras(
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """List camera sources from the database."""
    pg_client = _get_pg_client(context)
    if pg_client is None:
        return {"error": "Database not connected", "cameras": []}

    pool = getattr(pg_client, "_pool", None)
    if pool is None:
        return {"error": "Database pool not available", "cameras": []}

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT camera_id, source_url, location_tag, fps_target, enabled, added_at "
            "FROM camera_sources ORDER BY added_at"
        )

    cameras = []
    for row in rows:
        cam = dict(row)
        for k, v in cam.items():
            if hasattr(v, "isoformat"):
                cam[k] = v.isoformat()
        cameras.append(cam)

    return {"count": len(cameras), "cameras": cameras}


@tool(
    name="manage_camera",
    description=(
        "Enable or disable a camera source. "
        "Use action='enable' or action='disable'."
    ),
    param_descriptions={
        "camera_id": "The camera ID to manage",
        "action": "Action to perform: 'enable' or 'disable'",
    },
)
async def manage_camera(
    camera_id: str,
    action: str = "enable",
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Enable or disable a camera source."""
    if action not in ("enable", "disable"):
        return {"error": f"Invalid action: {action}. Use 'enable' or 'disable'."}

    pg_client = _get_pg_client(context)
    if pg_client is None:
        return {"error": "Database not connected"}

    pool = getattr(pg_client, "_pool", None)
    if pool is None:
        return {"error": "Database pool not available"}

    enabled = action == "enable"

    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE camera_sources SET enabled = $1 WHERE camera_id = $2",
            enabled,
            camera_id,
        )

    if "UPDATE 0" in result:
        return {"error": f"Camera '{camera_id}' not found"}

    return {"camera_id": camera_id, "action": action, "success": True}


# ══════════════════════════════════════════════════════════════════
# Threshold & Configuration Tools
# ══════════════════════════════════════════════════════════════════


@tool(
    name="adjust_threshold",
    description=(
        "Dynamically adjust a pipeline threshold. Supported thresholds: "
        "'ocr_confidence' (0.0-1.0), 'adjudication_confidence' (0.0-1.0), "
        "'detection_confidence' (0.0-1.0), 'temperature' (0.0-2.0)."
    ),
    param_descriptions={
        "threshold_name": "Name of the threshold to adjust",
        "value": "New value for the threshold",
    },
)
async def adjust_threshold(
    threshold_name: str,
    value: float,
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Dynamically adjust a pipeline threshold at runtime."""
    config = _get_config(context)
    if config is None:
        return {"error": "Configuration not available"}

    _THRESHOLD_MAP = {
        "ocr_confidence": ("validation", "adjudication_confidence_threshold", 0.0, 1.0),
        "adjudication_confidence": ("validation", "adjudication_confidence_threshold", 0.0, 1.0),
        "detection_confidence": ("_models_conf", None, 0.0, 1.0),
        "temperature": ("ollama", "temperature", 0.0, 2.0),
    }

    spec = _THRESHOLD_MAP.get(threshold_name)
    if spec is None:
        return {
            "error": f"Unknown threshold: {threshold_name}",
            "available": list(_THRESHOLD_MAP.keys()),
        }

    section_name, attr_name, min_val, max_val = spec

    if value < min_val or value > max_val:
        return {"error": f"Value {value} out of range [{min_val}, {max_val}]"}

    old_value = None

    if section_name == "validation" and attr_name:
        section = getattr(config, section_name, None)
        if section is not None:
            old_value = getattr(section, attr_name, None)
            object.__setattr__(section, attr_name, value)
    elif section_name == "ollama" and attr_name:
        section = getattr(config, section_name, None)
        if section is not None:
            old_value = getattr(section, attr_name, None)
            object.__setattr__(section, attr_name, value)

    logger.info(
        "threshold_adjusted name=%s old=%s new=%s",
        threshold_name,
        old_value,
        value,
    )

    return {
        "threshold": threshold_name,
        "old_value": old_value,
        "new_value": value,
        "success": True,
    }


@tool(
    name="get_current_config",
    description=(
        "Get the current values of key pipeline configuration parameters "
        "including thresholds, timeouts, and model settings."
    ),
)
async def get_current_config(
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Return current pipeline configuration."""
    config = _get_config(context)
    if config is None:
        return {"error": "Configuration not available"}

    return {
        "ollama": {
            "model": config.ollama.model,
            "temperature": config.ollama.temperature,
            "num_ctx": config.ollama.num_ctx,
            "num_predict": config.ollama.num_predict,
            "timeout_s": config.ollama.timeout_s,
            "max_retries": config.ollama.max_retries,
        },
        "validation": {
            "adjudication_confidence_threshold": config.validation.adjudication_confidence_threshold,
            "default_locale": config.validation.default_locale,
        },
        "adjudication": {
            "enabled": config.adjudication.enabled,
            "timeout_s": config.adjudication.timeout_s,
            "max_retries": config.adjudication.max_retries,
        },
        "pipeline": {
            "inference_queue_maxsize": config.pipeline.inference_queue_maxsize,
            "inference_queue_high_water": config.pipeline.inference_queue_high_water,
            "inference_queue_low_water": config.pipeline.inference_queue_low_water,
        },
        "circuit_breaker": {
            "failure_threshold": config.circuit_breaker.failure_threshold,
            "recovery_timeout_s": config.circuit_breaker.recovery_timeout_s,
        },
    }


# ══════════════════════════════════════════════════════════════════
# Audit & Search Tools
# ══════════════════════════════════════════════════════════════════


@tool(
    name="search_audit_log",
    description=(
        "Search the OCR audit log for failed or low-confidence readings. "
        "Useful for diagnosing OCR quality issues on specific cameras or detections."
    ),
    param_descriptions={
        "camera_id": "Filter by camera ID (empty for all)",
        "failure_reason": "Filter by failure reason substring",
        "hours_back": "Time window in hours (default: 24)",
        "limit": "Maximum results (default: 20, max: 100)",
    },
)
async def search_audit_log(
    camera_id: str = "",
    failure_reason: str = "",
    hours_back: float = 24.0,
    limit: int = 20,
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Search the OCR audit log."""
    pg_client = _get_pg_client(context)
    if pg_client is None:
        return {"error": "Database not connected", "results": []}

    pool = getattr(pg_client, "_pool", None)
    if pool is None:
        return {"error": "Database pool not available", "results": []}

    limit = min(max(1, limit), 100)
    interval = f"{max(0.01, hours_back)} hours"

    clauses: List[str] = [f"created_at >= NOW() - INTERVAL '{interval}'"]
    params: List[Any] = []
    idx = 1

    if camera_id:
        clauses.append(f"camera_id = ${idx}")
        params.append(camera_id)
        idx += 1
    if failure_reason:
        clauses.append(f"failure_reason ILIKE ${idx}")
        params.append(f"%{failure_reason}%")
        idx += 1

    where = " WHERE " + " AND ".join(clauses) if clauses else ""

    query = (
        f"SELECT id, camera_id, raw_ocr_text, ocr_confidence, "
        f"failure_reason, created_at "
        f"FROM ocr_audit_log {where} "
        f"ORDER BY created_at DESC LIMIT ${idx}"
    )
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    results = [dict(row) for row in rows]
    for r in results:
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                r[k] = v.isoformat()

    return {"count": len(results), "results": results}


@tool(
    name="analyze_detection_patterns",
    description=(
        "Analyse detection patterns — find most/least common detections, "
        "frequency distributions, and anomalies like detections seen across "
        "multiple cameras simultaneously."
    ),
    param_descriptions={
        "hours_back": "Time window for analysis (default: 24)",
        "top_n": "Number of top detections to return (default: 10)",
    },
)
async def analyze_detection_patterns(
    hours_back: float = 24.0,
    top_n: int = 10,
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Analyse detection patterns."""
    pg_client = _get_pg_client(context)
    if pg_client is None:
        return {"error": "Database not connected"}

    pool = getattr(pg_client, "_pool", None)
    if pool is None:
        return {"error": "Database pool not available"}

    interval = f"{max(0.01, hours_back)} hours"
    top_n = min(max(1, top_n), 50)

    async with pool.acquire() as conn:
        # Most frequently seen plates
        freq_rows = await conn.fetch(
            "SELECT plate_number, COUNT(*) as sightings, "
            "COALESCE(AVG(ocr_confidence), 0) as avg_confidence, "
            "array_agg(DISTINCT camera_id) as cameras "
            f"FROM detection_events WHERE detected_at_utc >= NOW() - INTERVAL '{interval}' "
            "GROUP BY plate_number ORDER BY sightings DESC LIMIT $1",
            top_n,
        )

        # Plates seen on multiple cameras (potential pass-through traffic)
        multi_cam = await conn.fetch(
            "SELECT plate_number, COUNT(DISTINCT camera_id) as camera_count, "
            "COUNT(*) as total_sightings "
            f"FROM detection_events WHERE detected_at_utc >= NOW() - INTERVAL '{interval}' "
            "GROUP BY plate_number HAVING COUNT(DISTINCT camera_id) > 1 "
            "ORDER BY camera_count DESC LIMIT $1",
            top_n,
        )

        # Low-confidence plate readings (potential misreads)
        low_conf = await conn.fetch(
            "SELECT plate_number, camera_id, ocr_confidence, validation_status "
            f"FROM detection_events WHERE detected_at_utc >= NOW() - INTERVAL '{interval}' "
            "AND ocr_confidence < 0.5 "
            "ORDER BY ocr_confidence ASC LIMIT $1",
            top_n,
        )

    return {
        "time_window_hours": hours_back,
        "most_frequent_plates": [
            {
                "plate": r["plate_number"],
                "sightings": r["sightings"],
                "avg_confidence": round(float(r["avg_confidence"]), 4),
                "cameras": list(r["cameras"]),
            }
            for r in freq_rows
        ],
        "multi_camera_plates": [
            {
                "plate": r["plate_number"],
                "camera_count": r["camera_count"],
                "total_sightings": r["total_sightings"],
            }
            for r in multi_cam
        ],
        "low_confidence_reads": [
            {
                "plate": r["plate_number"],
                "camera_id": r["camera_id"],
                "confidence": round(float(r["ocr_confidence"]), 4),
                "status": r["validation_status"],
            }
            for r in low_conf
        ],
    }


# ══════════════════════════════════════════════════════════════════
# Diagnostic & Self-Healing Tools
# ══════════════════════════════════════════════════════════════════


@tool(
    name="diagnose_camera",
    description=(
        "Run diagnostics for a specific camera — recent error rate, "
        "average confidence, detection count, and identify patterns "
        "in failures."
    ),
    param_descriptions={
        "camera_id": "The camera ID to diagnose",
        "hours_back": "Time window for diagnosis (default: 1)",
    },
)
async def diagnose_camera(
    camera_id: str,
    hours_back: float = 1.0,
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Run diagnostics for a specific camera."""
    pg_client = _get_pg_client(context)
    if pg_client is None:
        return {"error": "Database not connected"}

    pool = getattr(pg_client, "_pool", None)
    if pool is None:
        return {"error": "Database pool not available"}

    interval = f"{max(0.01, hours_back)} hours"

    async with pool.acquire() as conn:
        # Detection summary for this camera
        summary = await conn.fetchrow(
            "SELECT COUNT(*) as total, "
            "COALESCE(AVG(ocr_confidence), 0) as avg_conf, "
            "COUNT(*) FILTER (WHERE validation_status = 'valid') as valid_count, "
            "COUNT(*) FILTER (WHERE validation_status != 'valid') as error_count "
            f"FROM detection_events WHERE camera_id = $1 "
            f"AND detected_at_utc >= NOW() - INTERVAL '{interval}'",
            camera_id,
        )

        # Recent failures
        failures = await conn.fetch(
            "SELECT validation_status, COUNT(*) as count "
            f"FROM detection_events WHERE camera_id = $1 "
            f"AND detected_at_utc >= NOW() - INTERVAL '{interval}' "
            "AND validation_status != 'valid' "
            "GROUP BY validation_status ORDER BY count DESC",
            camera_id,
        )

        # Audit log entries for this camera
        audit_count = await conn.fetchval(
            "SELECT COUNT(*) FROM ocr_audit_log "
            f"WHERE camera_id = $1 AND created_at >= NOW() - INTERVAL '{interval}'",
            camera_id,
        )

    total = summary["total"] if summary else 0
    error_count = summary["error_count"] if summary else 0
    error_rate = (error_count / total * 100) if total > 0 else 0

    diagnosis: Dict[str, Any] = {
        "camera_id": camera_id,
        "time_window_hours": hours_back,
        "total_detections": total,
        "valid_detections": summary["valid_count"] if summary else 0,
        "error_detections": error_count,
        "error_rate_pct": round(error_rate, 2),
        "avg_confidence": round(float(summary["avg_conf"]) if summary else 0, 4),
        "audit_log_entries": audit_count or 0,
        "failure_breakdown": [
            {"status": r["validation_status"], "count": r["count"]}
            for r in failures
        ],
    }

    # Generate recommendations
    recommendations: List[str] = []
    if error_rate > 50:
        recommendations.append(
            "High error rate (>50%). Check camera focus, lighting, and angle."
        )
    if float(summary["avg_conf"] if summary else 0) < 0.5:
        recommendations.append(
            "Low average confidence. Consider adjusting camera position or "
            "reducing the adjudication confidence threshold."
        )
    if audit_count and audit_count > total * 0.3:
        recommendations.append(
            "Many audit log entries. The OCR engine may be struggling with "
            "this camera's image quality."
        )
    if total == 0:
        recommendations.append(
            "No detections in the time window. Verify the camera is online "
            "and the stream URL is correct."
        )

    diagnosis["recommendations"] = recommendations

    return diagnosis


@tool(
    name="reset_circuit_breaker",
    description=(
        "Reset the OCR circuit breaker from OPEN to CLOSED state. "
        "Use when the LLM service has recovered after an outage."
    ),
)
async def reset_circuit_breaker(
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Reset the circuit breaker to CLOSED."""
    from uni_vision.contracts.dtos import CircuitState

    pipeline = _get_pipeline(context)
    if pipeline is None:
        return {"error": "Pipeline not running"}

    ocr_strategy = getattr(pipeline, "_ocr_strategy", None)
    if ocr_strategy is None:
        return {"error": "OCR strategy not found"}

    primary = getattr(ocr_strategy, "_primary", None)
    if primary is None:
        return {"error": "Primary OCR engine not found"}

    old_state = getattr(primary, "_cb_state", None)
    primary._cb_state = CircuitState.CLOSED
    primary._failure_timestamps = []

    return {
        "success": True,
        "old_state": old_state.value if old_state else "unknown",
        "new_state": "closed",
    }


# ══════════════════════════════════════════════════════════════════
# Natural Language to SQL Translation
# ══════════════════════════════════════════════════════════════════


@tool(
    name="run_analytics_query",
    description=(
        "Execute a read-only analytics SQL query against the detection database. "
        "Only SELECT queries are permitted. Use for custom analysis that "
        "other tools don't cover. NEVER execute INSERT/UPDATE/DELETE."
    ),
    param_descriptions={
        "query_description": "Natural language description of the data you want",
    },
)
async def run_analytics_query(
    query_description: str,
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """The agent describes what it wants; we map to a safe query.

    This tool accepts a natural language description and translates it
    to a safe, parameterised query. The agent cannot execute arbitrary
    SQL — only pre-approved query patterns.
    """
    # Pre-approved query patterns (safe, read-only)
    _QUERY_PATTERNS = {
        "hourly_trend": (
            "SELECT date_trunc('hour', detected_at_utc) as hour, "
            "COUNT(*) as count, AVG(ocr_confidence) as avg_conf "
            "FROM detection_events "
            "WHERE detected_at_utc >= NOW() - INTERVAL '24 hours' "
            "GROUP BY hour ORDER BY hour"
        ),
        "camera_performance": (
            "SELECT camera_id, COUNT(*) as total, "
            "AVG(ocr_confidence) as avg_conf, "
            "COUNT(*) FILTER (WHERE validation_status = 'valid') as valid, "
            "COUNT(*) FILTER (WHERE validation_status != 'valid') as invalid "
            "FROM detection_events "
            "WHERE detected_at_utc >= NOW() - INTERVAL '24 hours' "
            "GROUP BY camera_id ORDER BY total DESC"
        ),
        "engine_comparison": (
            "SELECT ocr_engine, COUNT(*) as total, "
            "AVG(ocr_confidence) as avg_conf "
            "FROM detection_events "
            "WHERE detected_at_utc >= NOW() - INTERVAL '24 hours' "
            "GROUP BY ocr_engine ORDER BY total DESC"
        ),
        "recent_errors": (
            "SELECT validation_status, failure_reason, COUNT(*) as count "
            "FROM ocr_audit_log "
            "WHERE created_at >= NOW() - INTERVAL '24 hours' "
            "GROUP BY validation_status, failure_reason "
            "ORDER BY count DESC LIMIT 20"
        ),
    }

    desc_lower = query_description.lower()

    # Simple keyword matching to select the right pattern
    if "hourly" in desc_lower or "trend" in desc_lower or "over time" in desc_lower:
        pattern_key = "hourly_trend"
    elif "camera" in desc_lower and ("performance" in desc_lower or "comparison" in desc_lower):
        pattern_key = "camera_performance"
    elif "engine" in desc_lower or "ocr" in desc_lower and "comparison" in desc_lower:
        pattern_key = "engine_comparison"
    elif "error" in desc_lower or "failure" in desc_lower or "audit" in desc_lower:
        pattern_key = "recent_errors"
    else:
        return {
            "error": "Could not match query pattern. Available patterns: "
            + ", ".join(_QUERY_PATTERNS.keys()),
            "hint": "Try describing your query using keywords like: "
            "hourly trend, camera performance, engine comparison, recent errors",
        }

    pg_client = _get_pg_client(context)
    if pg_client is None:
        return {"error": "Database not connected"}

    pool = getattr(pg_client, "_pool", None)
    if pool is None:
        return {"error": "Database pool not available"}

    query = _QUERY_PATTERNS[pattern_key]

    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    results = []
    for row in rows:
        r = dict(row)
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                r[k] = v.isoformat()
            elif isinstance(v, float):
                r[k] = round(v, 4)
        results.append(r)

    return {
        "pattern": pattern_key,
        "query_description": query_description,
        "result_count": len(results),
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════
# Context helpers (extract objects from the agent's execution context)
# ══════════════════════════════════════════════════════════════════


def _get_pg_client(context: Any) -> Any:
    """Extract the PostgresClient from the tool execution context."""
    if context is None:
        return None
    return getattr(context, "pg_client", None)


def _get_pipeline(context: Any) -> Any:
    """Extract the Pipeline from the tool execution context."""
    if context is None:
        return None
    return getattr(context, "pipeline", None)


def _get_config(context: Any) -> Any:
    """Extract the AppConfig from the tool execution context."""
    if context is None:
        return None
    return getattr(context, "config", None)
