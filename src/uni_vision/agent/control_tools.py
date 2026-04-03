"""Intelligent pipeline control tools — self-tuning and self-healing.

These tools give the agent the ability to:
  * Dynamically tune pipeline parameters based on observed quality.
  * Perform self-healing when anomalies are detected.
  * Run stage-level analytics for targeted optimisation.
  * Manage the OCR strategy (primary vs fallback balance).
  * Control inference queue pressure and adaptive throttling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from uni_vision.agent.tools import tool

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Self-Tuning Tools
# ══════════════════════════════════════════════════════════════════


@tool(
    name="auto_tune_confidence",
    description=(
        "Analyze recent detection quality and recommend or apply optimal "
        "confidence thresholds. Examines validation pass/fail rates over "
        "the time window and adjusts the adjudication threshold to balance "
        "precision (fewer false positives) and recall (fewer missed detections)."
    ),
    param_descriptions={
        "hours_back": "Time window for analysis (default: 1)",
        "apply": "If true, automatically apply the recommended threshold",
        "target_valid_rate": "Target percentage of valid detections (default: 0.80)",
    },
)
async def auto_tune_confidence(
    hours_back: float = 1.0,
    apply: bool = False,
    target_valid_rate: float = 0.80,
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Analyzes detection quality and tunes adjudication threshold."""
    pg_client = _get_pg_client(context)
    if pg_client is None:
        return {"error": "Database not connected"}

    pool = getattr(pg_client, "_pool", None)
    if pool is None:
        return {"error": "Database pool not available"}

    interval = f"{max(0.01, hours_back)} hours"

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT "
            "  COUNT(*) as total, "
            "  COUNT(*) FILTER (WHERE validation_status = 'valid') as valid_count, "
            "  COALESCE(AVG(ocr_confidence), 0) as avg_conf, "
            "  COALESCE(percentile_cont(0.25) WITHIN GROUP (ORDER BY ocr_confidence), 0) as p25, "
            "  COALESCE(percentile_cont(0.50) WITHIN GROUP (ORDER BY ocr_confidence), 0) as p50, "
            "  COALESCE(percentile_cont(0.75) WITHIN GROUP (ORDER BY ocr_confidence), 0) as p75 "
            f"FROM detection_events WHERE detected_at_utc >= NOW() - INTERVAL '{interval}'"
        )

    total = row["total"] if row else 0
    valid = row["valid_count"] if row else 0
    current_rate = valid / total if total > 0 else 0

    # Get current threshold
    config = _get_config(context)
    current_threshold = 0.75
    if config is not None:
        current_threshold = config.validation.adjudication_confidence_threshold

    # Determine recommendation
    recommended = current_threshold
    reason = "No change needed"

    if total < 10:
        reason = "Insufficient data (< 10 detections) — no change recommended"
    elif current_rate < target_valid_rate * 0.8:
        # Too many failures — lower the threshold to be less strict
        recommended = max(0.3, float(row["p25"]) - 0.05)
        reason = (
            f"Valid rate {current_rate:.1%} is well below target {target_valid_rate:.1%}. "
            f"Lowering threshold from {current_threshold:.2f} to {recommended:.2f} "
            "to accept more readings."
        )
    elif current_rate < target_valid_rate:
        # Slightly below target — small adjustment
        recommended = max(0.3, current_threshold - 0.05)
        reason = (
            f"Valid rate {current_rate:.1%} slightly below target {target_valid_rate:.1%}. "
            f"Small decrease from {current_threshold:.2f} to {recommended:.2f}."
        )
    elif current_rate > 0.95:
        # Very high valid rate — can afford to be stricter
        recommended = min(0.95, current_threshold + 0.05)
        reason = (
            f"Valid rate {current_rate:.1%} very high — tightening threshold "
            f"from {current_threshold:.2f} to {recommended:.2f} for better precision."
        )

    result = {
        "time_window_hours": hours_back,
        "total_detections": total,
        "valid_detections": valid,
        "valid_rate": round(current_rate, 4),
        "target_valid_rate": target_valid_rate,
        "current_threshold": current_threshold,
        "recommended_threshold": round(recommended, 4),
        "confidence_p25": round(float(row["p25"]) if row else 0, 4),
        "confidence_p50": round(float(row["p50"]) if row else 0, 4),
        "confidence_p75": round(float(row["p75"]) if row else 0, 4),
        "reason": reason,
        "applied": False,
    }

    if apply and recommended != current_threshold and config is not None:
        old = current_threshold
        object.__setattr__(
            config.validation,
            "adjudication_confidence_threshold",
            recommended,
        )
        result["applied"] = True
        logger.info(
            "auto_tune_applied old=%.4f new=%.4f reason=%s",
            old, recommended, reason,
        )

    return result


@tool(
    name="get_stage_analytics",
    description=(
        "Get per-stage latency analytics — identifies which pipeline stages "
        "(S0-S8) are slowest and contributing most to end-to-end latency."
    ),
)
async def get_stage_analytics(
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Read per-stage latency metrics from Prometheus."""
    try:
        from prometheus_client import REGISTRY
    except ImportError:
        return {"error": "prometheus_client not available"}

    stages: Dict[str, Dict[str, float]] = {}

    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if not sample.name.startswith("uv_stage_latency_seconds"):
                continue
            stage = sample.labels.get("stage", "unknown")
            if stage not in stages:
                stages[stage] = {}

            if sample.name.endswith("_sum"):
                stages[stage]["total_seconds"] = sample.value
            elif sample.name.endswith("_count"):
                stages[stage]["invocations"] = sample.value
            elif sample.name.endswith("_bucket"):
                continue  # skip histogram buckets

    # Calculate averages
    results = []
    for stage, data in sorted(stages.items()):
        total = data.get("total_seconds", 0)
        count = data.get("invocations", 0)
        avg = total / count if count > 0 else 0
        results.append({
            "stage": stage,
            "invocations": int(count),
            "total_seconds": round(total, 4),
            "avg_latency_ms": round(avg * 1000, 2),
        })

    # Sort by average latency descending
    results.sort(key=lambda x: x["avg_latency_ms"], reverse=True)

    # E2E pipeline metric
    e2e_total = 0.0
    e2e_count = 0
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == "uv_pipeline_latency_seconds_sum":
                e2e_total = sample.value
            elif sample.name == "uv_pipeline_latency_seconds_count":
                e2e_count = int(sample.value)

    e2e_avg = e2e_total / e2e_count if e2e_count > 0 else 0

    return {
        "stages": results,
        "e2e_avg_latency_ms": round(e2e_avg * 1000, 2),
        "e2e_total_events": e2e_count,
        "bottleneck": results[0]["stage"] if results else "unknown",
    }


@tool(
    name="get_ocr_strategy_stats",
    description=(
        "Get OCR strategy performance — primary vs fallback usage, "
        "success rates, circuit breaker state, and confidence differences."
    ),
)
async def get_ocr_strategy_stats(
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Analyze OCR engine performance."""
    try:
        from prometheus_client import REGISTRY
    except ImportError:
        return {"error": "prometheus_client not available"}

    stats: Dict[str, Any] = {
        "primary_requests": 0,
        "primary_success": 0,
        "fallback_requests": 0,
        "fallback_total": 0,
        "circuit_breaker_state": "unknown",
    }

    for metric in REGISTRY.collect():
        for sample in metric.samples:
            name = sample.name
            engine = sample.labels.get("engine", "")

            if name == "uv_ocr_requests_total" and engine == "ollama":
                stats["primary_requests"] = int(sample.value)
            elif name == "uv_ocr_requests_total" and engine == "easyocr":
                stats["fallback_requests"] = int(sample.value)
            elif name == "uv_ocr_success_total" and engine == "ollama":
                stats["primary_success"] = int(sample.value)
            elif name == "uv_ocr_success_total" and engine == "easyocr":
                stats["fallback_success"] = int(sample.value)
            elif name == "uv_ocr_fallback_total":
                stats["fallback_total"] = int(sample.value)

    # Get circuit breaker state from pipeline
    pipeline = _get_pipeline(context)
    if pipeline is not None:
        ocr_strategy = getattr(pipeline, "_ocr_strategy", None)
        if ocr_strategy:
            primary = getattr(ocr_strategy, "_primary", None)
            if primary:
                cb_state = getattr(primary, "_cb_state", None)
                if cb_state:
                    stats["circuit_breaker_state"] = cb_state.value
                failures = getattr(primary, "_failure_timestamps", [])
                stats["recent_failures"] = len(failures)

    # Calculate rates
    prim_req = stats["primary_requests"]
    prim_succ = stats["primary_success"]
    stats["primary_success_rate"] = round(
        prim_succ / prim_req if prim_req > 0 else 0, 4
    )
    stats["fallback_rate"] = round(
        stats["fallback_total"] / (prim_req + stats["fallback_requests"])
        if (prim_req + stats["fallback_requests"]) > 0
        else 0,
        4,
    )

    return stats


# ══════════════════════════════════════════════════════════════════
# Self-Healing Tools
# ══════════════════════════════════════════════════════════════════


@tool(
    name="self_heal_pipeline",
    description=(
        "Run an automated self-healing check. Detects common issues "
        "(circuit breaker open, high error rates, queue backpressure) "
        "and takes corrective action automatically."
    ),
)
async def self_heal_pipeline(
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Detect and auto-fix common pipeline issues."""
    pipeline = _get_pipeline(context)
    actions_taken: List[str] = []
    issues_found: List[str] = []

    if pipeline is None:
        return {"error": "Pipeline not running", "actions": [], "issues": ["Pipeline not found"]}

    # Check circuit breaker
    ocr_strategy = getattr(pipeline, "_ocr_strategy", None)
    if ocr_strategy:
        primary = getattr(ocr_strategy, "_primary", None)
        if primary:
            from uni_vision.contracts.dtos import CircuitState

            cb_state = getattr(primary, "_cb_state", None)
            if cb_state == CircuitState.OPEN:
                issues_found.append("Circuit breaker is OPEN — OCR using fallback only")
                # Check if Ollama is back by doing a lightweight probe
                try:
                    import httpx

                    async with httpx.AsyncClient(timeout=3.0) as client:
                        resp = await client.get(f"{primary._base_url}/api/tags")
                        if resp.status_code == 200:
                            primary._cb_state = CircuitState.CLOSED
                            primary._failure_timestamps = []
                            actions_taken.append(
                                "Reset circuit breaker to CLOSED — Ollama is responsive"
                            )
                except Exception:
                    actions_taken.append(
                        "Ollama still unreachable — circuit breaker remains OPEN"
                    )

    # Check queue backpressure
    inference_q = getattr(pipeline, "_inference_queue", None)
    if inference_q is not None:
        depth = inference_q.qsize()
        maxsize = getattr(inference_q, "maxsize", 10)
        if depth >= maxsize * 0.8:
            issues_found.append(
                f"Inference queue near full ({depth}/{maxsize})"
            )
            # Enable throttle
            if not getattr(pipeline, "_throttled", False):
                pipeline._throttled = True
                actions_taken.append("Enabled adaptive throttle for queue pressure relief")

    # Check if too many frames are being dropped
    try:
        from prometheus_client import REGISTRY

        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if sample.name == "uv_frames_dropped_total":
                    if sample.value > 100:
                        issues_found.append(
                            f"High frame drop count: {int(sample.value)}"
                        )
    except ImportError:
        pass

    return {
        "issues_found": issues_found,
        "actions_taken": actions_taken,
        "healthy": len(issues_found) == 0,
    }


@tool(
    name="get_queue_pressure",
    description=(
        "Get detailed inference queue metrics — current depth, max size, "
        "high/low water marks, throttle state, and frame drop counts."
    ),
)
async def get_queue_pressure(
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Get inference queue pressure details."""
    pipeline = _get_pipeline(context)
    if pipeline is None:
        return {"error": "Pipeline not running"}

    config = _get_config(context)

    inference_q = getattr(pipeline, "_inference_queue", None)
    stream_q = getattr(pipeline, "_stream_queue", None)

    result: Dict[str, Any] = {
        "inference_queue_depth": inference_q.qsize() if inference_q else 0,
        "inference_queue_maxsize": getattr(inference_q, "maxsize", 0) if inference_q else 0,
        "stream_queue_depth": stream_q.qsize() if stream_q else 0,
        "throttled": getattr(pipeline, "_throttled", False),
    }

    if config is not None:
        result["high_water_mark"] = config.pipeline.inference_queue_high_water
        result["low_water_mark"] = config.pipeline.inference_queue_low_water
        result["adaptive_throttle_factor"] = config.pipeline.adaptive_throttle_factor

    return result


@tool(
    name="flush_inference_queue",
    description=(
        "Drain the inference queue by discarding pending frames. "
        "Use when queue is stuck or filled with stale frames."
    ),
)
async def flush_inference_queue(
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Drain the inference queue."""
    pipeline = _get_pipeline(context)
    if pipeline is None:
        return {"error": "Pipeline not running"}

    inference_q = getattr(pipeline, "_inference_queue", None)
    if inference_q is None:
        return {"error": "Inference queue not found"}

    flushed = 0
    while not inference_q.empty():
        try:
            inference_q.get_nowait()
            flushed += 1
        except asyncio.QueueEmpty:
            break

    return {"flushed_frames": flushed, "success": True}


# ══════════════════════════════════════════════════════════════════
# Context helpers
# ══════════════════════════════════════════════════════════════════


def _get_pg_client(context: Any) -> Any:
    if context is None:
        return None
    return getattr(context, "pg_client", None)


def _get_pipeline(context: Any) -> Any:
    if context is None:
        return None
    return getattr(context, "pipeline", None)


def _get_config(context: Any) -> Any:
    if context is None:
        return None
    return getattr(context, "config", None)
