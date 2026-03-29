"""Crop & Straighten visualizer — side-by-side before/after crop and deskew.

Displays plate cropping and geometric correction results from stages S4/S5.
Since intermediate images are stored in the debug store / object store,
this module fetches metadata and shows image paths alongside text results.
"""

from __future__ import annotations

import streamlit as st

from uni_vision.visualizer.helpers import fetch_json, get_api_base, get_api_key


def render() -> None:
    """Render the crop & straighten inspection page."""
    st.header("✂️ Crop & Straighten (S4/S5)")

    base = get_api_base()
    api_key = get_api_key()

    st.markdown(
        """
        **S4 — Plate Crop:** Extracts the plate ROI from the full frame
        with configurable padding.

        **S5 — Geometric Correction:** Applies deskewing, rotation
        correction, and perspective transform to straighten the plate
        for optimal OCR accuracy.

        Enable the debug image store to capture before/after images
        for visual comparison.
        """
    )

    # ── Recent detections ─────────────────────────────────────────
    data = fetch_json(f"{base}/detections?page=1&page_size=10", api_key)

    if not data or not data.get("items"):
        st.info("No detection records available.")
        return

    items = data["items"]

    st.subheader("Recent Plate Crops")

    for item in items:
        plate_path = item.get("plate_image_path", "")
        vehicle_path = item.get("vehicle_image_path", "")

        with st.expander(
            f"🔖 {item['plate_number']} — Camera {item['camera_id']}", expanded=False
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Vehicle ROI**")
                if vehicle_path:
                    st.caption(f"Path: `{vehicle_path}`")
                    st.info("Enable debug store to view vehicle image")
                else:
                    st.caption("No vehicle image stored")

            with col2:
                st.markdown("**Cropped & Straightened Plate**")
                if plate_path and plate_path != "upload_pending":
                    st.caption(f"Path: `{plate_path}`")
                    st.info("Enable debug store to view plate image")
                else:
                    st.caption("No plate image stored")

            st.markdown(
                f"**Confidence:** {item['ocr_confidence']:.2f} | "
                f"**Engine:** {item.get('ocr_engine', 'N/A')} | "
                f"**Status:** {item.get('validation_status', 'N/A')}"
            )
