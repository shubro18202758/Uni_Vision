"""Frame Sampler visualizer — displays sampled frames with pHash similarity score.

Shows the latest frames from the pipeline with deduplication info,
allowing operators to verify that the frame sampler is correctly
filtering redundant frames.
"""

from __future__ import annotations

import streamlit as st

from uni_vision.visualizer.helpers import fetch_json, get_api_base, get_api_key


def render() -> None:
    """Render the frame sampler inspection page."""
    st.header("🎞️ Frame Sampler")

    base = get_api_base()
    api_key = get_api_key()

    st.markdown(
        """
        The frame sampler (S1) applies **perceptual hashing (pHash)** to
        drop near-duplicate frames before they reach the inference queue.
        This page shows recent sampled frames and their deduplication
        statistics.
        """
    )

    # ── Recent detections as a proxy for sampled frames ──────────
    data = fetch_json(f"{base}/detections?page=1&page_size=12", api_key)

    if not data or not data.get("items"):
        st.info("No detection frames available yet. Start the pipeline to see sampled frames.")
        return

    st.subheader("Recent Sampled Frames")

    # Display summary stats
    total = data.get("total", 0)
    st.metric("Total Detections (sampled frames passed)", total)

    # Display items in a grid
    items = data["items"]
    cols = st.columns(3)
    for idx, item in enumerate(items):
        with cols[idx % 3]:
            st.markdown(f"**Camera:** `{item['camera_id']}`")
            st.markdown(f"**Plate:** `{item['plate_number']}`")
            st.markdown(f"**Confidence:** {item['ocr_confidence']:.2f}")
            st.markdown(f"**Time:** {item.get('detected_at_utc', 'N/A')}")

            # Show plate image if available
            plate_path = item.get("plate_image_path", "")
            if plate_path and plate_path != "upload_pending":
                st.caption(f"Image: {plate_path}")
            st.divider()

    st.caption(
        "Full frame images are stored in the object store (S3/MinIO). "
        "Enable the debug store to capture intermediate pipeline frames."
    )
