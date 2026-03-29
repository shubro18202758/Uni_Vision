"""Plate Detection visualizer — highlighted plate region within vehicle ROI.

Shows plate detection results from S3 with location info, confidence,
and bounding-box metadata.
"""

from __future__ import annotations

import streamlit as st

from uni_vision.visualizer.helpers import fetch_json, get_api_base, get_api_key


def render() -> None:
    """Render the plate detection inspection page."""
    st.header("🔲 Plate Detection (S3)")

    base = get_api_base()
    api_key = get_api_key()

    st.markdown(
        """
        Stage S3 locates **licence plate regions** within each detected
        vehicle's bounding box using a dedicated plate detector model.
        This page shows detection results with plate text and confidence.
        """
    )

    # ── Fetch detections ──────────────────────────────────────────
    camera_filter = st.text_input("Filter by Camera ID (optional)")
    page_size = st.slider("Results per page", 5, 50, 20, key="plate_page_size")

    url = f"{base}/detections?page=1&page_size={page_size}"
    if camera_filter:
        url += f"&camera_id={camera_filter}"

    data = fetch_json(url, api_key)

    if not data or not data.get("items"):
        st.info("No plate detection records found.")
        return

    items = data["items"]
    total = data.get("total", 0)
    st.caption(f"Showing {len(items)} of {total:,} total records")

    # ── Plate gallery ─────────────────────────────────────────────
    st.subheader("Detected Plates")
    cols = st.columns(4)

    for idx, item in enumerate(items):
        with cols[idx % 4]:
            plate = item["plate_number"]
            conf = item["ocr_confidence"]

            # Color-code by confidence
            if conf >= 0.85:
                color = "🟢"
            elif conf >= 0.6:
                color = "🟡"
            else:
                color = "🔴"

            st.markdown(f"### {color} `{plate}`")
            st.caption(
                f"Camera: {item['camera_id']}  \n"
                f"Confidence: {conf:.2f}  \n"
                f"Vehicle: {item.get('vehicle_class', 'N/A')}  \n"
                f"Status: {item.get('validation_status', 'N/A')}"
            )

            plate_path = item.get("plate_image_path", "")
            if plate_path and plate_path != "upload_pending":
                st.caption(f"📷 {plate_path}")

            st.divider()
