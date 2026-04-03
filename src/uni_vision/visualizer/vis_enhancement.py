"""Enhancement visualizer — side-by-side raw vs enhanced plate image.

Shows S6 photometric enhancement results: CLAHE, sharpening, and
denoising applied to plate crops before OCR.
"""

from __future__ import annotations

import streamlit as st

from uni_vision.visualizer.helpers import fetch_json, get_api_base, get_api_key


def render() -> None:
    """Render the enhancement inspection page."""
    st.header("✨ Enhancement (S6)")

    base = get_api_base()
    api_key = get_api_key()

    st.markdown(
        """
        Stage S6 applies **photometric enhancement** to plate crops:

        - **CLAHE** — Adaptive histogram equalisation for contrast
        - **Unsharp masking** — Edge sharpening for character clarity
        - **Non-local means denoising** — Noise reduction for night/rain

        Enhanced images are passed to OCR (S7). Enable the debug store
        to capture raw vs enhanced comparisons.
        """
    )

    # ── Detection records with confidence analysis ────────────────
    data = fetch_json(f"{base}/detections?page=1&page_size=20", api_key)

    if not data or not data.get("items"):
        st.info("No detection records available. Start the pipeline to generate data.")
        return

    items = data["items"]

    # ── Confidence bucketing — enhancement quality proxy ──────────
    buckets = {"High (≥0.85)": 0, "Medium (0.6–0.85)": 0, "Low (<0.6)": 0}
    for item in items:
        conf = item["ocr_confidence"]
        if conf >= 0.85:
            buckets["High (≥0.85)"] += 1
        elif conf >= 0.6:
            buckets["Medium (0.6–0.85)"] += 1
        else:
            buckets["Low (<0.6)"] += 1

    st.subheader("Enhancement Quality Proxy")
    st.caption("Higher OCR confidence suggests effective preprocessing")

    import pandas as pd

    df_buckets = pd.DataFrame(list(buckets.items()), columns=["Quality Tier", "Count"])
    st.bar_chart(df_buckets.set_index("Quality Tier"))

    # ── Per-plate details ─────────────────────────────────────────
    st.subheader("Enhancement Details")

    for item in items:
        conf = item["ocr_confidence"]
        indicator = "🟢" if conf >= 0.85 else ("🟡" if conf >= 0.6 else "🔴")

        with st.expander(f"{indicator} {item['plate_number']} — conf={conf:.2f}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Raw Plate Crop**")
                plate_path = item.get("plate_image_path", "")
                if plate_path:
                    st.caption(f"Image: `{plate_path}`")
                st.info("Enable debug store for raw image capture")
            with col2:
                st.markdown("**Enhanced Plate**")
                st.info("Enable debug store for enhanced image capture")

            st.markdown(
                f"**OCR Text:** `{item.get('raw_ocr_text', 'N/A')}` → "
                f"`{item['plate_number']}`  \n"
                f"**Engine:** {item.get('ocr_engine', 'N/A')} | "
                f"**Status:** {item.get('validation_status', 'N/A')}"
            )
