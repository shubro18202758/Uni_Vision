"""Pipeline Stats — real-time dashboard: FPS, detection rate, OCR accuracy, queue depth.

Reads Prometheus metrics from the /metrics endpoint and the API health
endpoint to populate live gauges and charts.
"""

from __future__ import annotations

import re

import streamlit as st

from uni_vision.visualizer.helpers import fetch_json, get_api_base, get_api_key


def _parse_prometheus_text(text: str) -> dict[str, float]:
    """Extract metric name → latest value from Prometheus text format."""
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:{}=",]*)\s+([\d.eE+-]+)', line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def _fetch_metrics(base: str, api_key: str) -> dict[str, float]:
    """Fetch raw Prometheus metrics and parse them."""
    import httpx

    from uni_vision.visualizer.helpers import api_headers

    try:
        resp = httpx.get(
            f"{base}/metrics",
            headers=api_headers(api_key),
            timeout=10.0,
        )
        resp.raise_for_status()
        return _parse_prometheus_text(resp.text)
    except Exception as exc:
        st.error(f"Failed to fetch metrics: {exc}")
        return {}


def render() -> None:
    """Render the pipeline stats dashboard."""
    st.header("📊 Pipeline Stats")

    base = get_api_base()
    api_key = get_api_key()

    col_refresh, _ = st.columns([1, 3])
    with col_refresh:
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)

    if auto_refresh:
        st.empty()
        import time

        time.sleep(5)
        st.rerun()

    # ── Health endpoint ───────────────────────────────────────────
    health = fetch_json(f"{base}/health", api_key)

    if health:
        st.subheader("System Health")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", "✅ Healthy" if health.get("healthy") else "❌ Unhealthy")
        c2.metric("GPU Available", "Yes" if health.get("gpu_available") else "No")
        c3.metric("Ollama", "✅" if health.get("ollama_reachable") else "❌")
        c4.metric("Database", "✅" if health.get("database_connected") else "❌")

        stream_info = f"{health.get('streams_connected', 0)}/{health.get('streams_total', 0)}"
        st.metric("Streams Connected", stream_info)

    st.divider()

    # ── Prometheus metrics ────────────────────────────────────────
    metrics = _fetch_metrics(base, api_key)

    if not metrics:
        st.info("No metrics available. Is the pipeline running?")
        return

    st.subheader("Throughput Counters")
    c1, c2, c3 = st.columns(3)

    # Sum across all camera_id labels
    frames_ingested = sum(v for k, v in metrics.items() if "uv_frames_ingested_total" in k)
    frames_deduped = sum(v for k, v in metrics.items() if "uv_frames_deduplicated_total" in k)
    detections = sum(v for k, v in metrics.items() if "uv_detections_total" in k)

    c1.metric("Frames Ingested", f"{int(frames_ingested):,}")
    c2.metric("Frames Deduplicated", f"{int(frames_deduped):,}")
    c3.metric("Detections", f"{int(detections):,}")

    st.subheader("OCR Performance")
    c1, c2, c3 = st.columns(3)

    ocr_requests = sum(v for k, v in metrics.items() if "uv_ocr_requests_total" in k)
    ocr_success = sum(v for k, v in metrics.items() if "uv_ocr_success_total" in k)
    ocr_fallback = metrics.get("uv_ocr_fallback_total", 0.0)

    c1.metric("OCR Requests", f"{int(ocr_requests):,}")
    c2.metric("OCR Success", f"{int(ocr_success):,}")
    c3.metric("OCR Fallbacks", f"{int(ocr_fallback):,}")

    if ocr_requests > 0:
        accuracy_pct = (ocr_success / ocr_requests) * 100
        st.progress(min(accuracy_pct / 100, 1.0), text=f"OCR Accuracy: {accuracy_pct:.1f}%")

    st.subheader("Infrastructure Gauges")
    c1, c2, c3 = st.columns(3)

    queue_depth = metrics.get("uv_inference_queue_depth", 0.0)
    dispatch_ok = metrics.get("uv_dispatch_success_total", 0.0)
    dispatch_err = sum(v for k, v in metrics.items() if "uv_dispatch_errors_total" in k)

    c1.metric("Queue Depth", int(queue_depth))
    c2.metric("Dispatched OK", f"{int(dispatch_ok):,}")
    c3.metric("Dispatch Errors", int(dispatch_err))

    # VRAM usage
    vram_entries = {k: v for k, v in metrics.items() if "uv_vram_usage_bytes" in k}
    if vram_entries:
        st.subheader("VRAM Usage")
        for key, value in vram_entries.items():
            label = re.search(r'region="([^"]+)"', key)
            region_name = label.group(1) if label else key
            st.metric(f"VRAM — {region_name}", f"{value / 1024 / 1024:.1f} MB")
