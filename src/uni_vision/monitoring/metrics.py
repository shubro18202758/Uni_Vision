"""Prometheus metric definitions — spec §12.1.

All 14 metrics from the architecture specification are defined here
as module-level singletons.  Import individual metrics where needed
rather than passing a registry around.

Usage::

    from uni_vision.monitoring.metrics import FRAMES_INGESTED, PIPELINE_LATENCY

    FRAMES_INGESTED.inc()
    PIPELINE_LATENCY.observe(elapsed)
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Counters ──────────────────────────────────────────────────────

FRAMES_INGESTED = Counter(
    "uv_frames_ingested_total",
    "Total frames read from all camera streams",
    ["camera_id"],
)

FRAMES_DEDUPLICATED = Counter(
    "uv_frames_deduplicated_total",
    "Frames discarded by pHash deduplication",
    ["camera_id"],
)

FRAMES_DROPPED = Counter(
    "uv_frames_dropped_total",
    "Frames dropped due to inference queue backpressure",
    ["camera_id"],
)

DETECTIONS_TOTAL = Counter(
    "uv_detections_total",
    "Successful vehicle + plate detection events",
    ["camera_id"],
)

OCR_REQUESTS = Counter(
    "uv_ocr_requests_total",
    "OCR requests sent to an engine",
    ["engine"],
)

OCR_SUCCESS = Counter(
    "uv_ocr_success_total",
    "OCR results with validation status 'valid'",
    ["engine"],
)

OCR_FALLBACK = Counter(
    "uv_ocr_fallback_total",
    "OCR requests routed to the fallback engine",
)

# ── Histograms ────────────────────────────────────────────────────

PIPELINE_LATENCY = Histogram(
    "uv_pipeline_latency_seconds",
    "End-to-end latency per detection event (S0 → S8)",
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0),
)

STAGE_LATENCY = Histogram(
    "uv_stage_latency_seconds",
    "Per-stage latency in seconds",
    ["stage"],
    buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

OCR_CONFIDENCE = Histogram(
    "uv_ocr_confidence",
    "Distribution of OCR confidence scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

# ── Gauges ────────────────────────────────────────────────────────

VRAM_USAGE = Gauge(
    "uv_vram_usage_bytes",
    "Current VRAM usage per region",
    ["region"],
)

INFERENCE_QUEUE_DEPTH = Gauge(
    "uv_inference_queue_depth",
    "Current number of items in the inference queue",
)

STREAM_STATUS = Gauge(
    "uv_stream_status",
    "Per-camera stream status (1=connected, 0=disconnected)",
    ["camera_id"],
)

# ── Agent metrics ─────────────────────────────────────────────────

AGENT_REQUESTS = Counter(
    "uv_agent_requests_total",
    "Total agent chat requests received",
)

AGENT_TOOL_CALLS = Counter(
    "uv_agent_tool_calls_total",
    "Agent tool invocations",
    ["tool_name", "success"],
)

AGENT_LATENCY = Histogram(
    "uv_agent_latency_seconds",
    "End-to-end agent response latency",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)

AGENT_STEPS = Histogram(
    "uv_agent_steps_per_request",
    "Number of ReAct steps per agent request",
    buckets=(1, 2, 3, 5, 7, 10, 15),
)

# ── Dispatch & dedup counters ─────────────────────────────────────

DETECTIONS_DEDUPLICATED = Counter(
    "uv_detections_deduplicated_total",
    "Detections suppressed by sliding-window deduplication",
)

DISPATCH_SUCCESS = Counter(
    "uv_dispatch_success_total",
    "Records successfully persisted to database + object store",
)

DISPATCH_ERRORS = Counter(
    "uv_dispatch_errors_total",
    "Dispatch failures (DB write or image upload)",
    ["target"],
)
