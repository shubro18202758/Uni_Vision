"""Vehicle Detection visualizer — annotated bounding boxes with class labels and confidence.

Renders detection records in a structured table and provides
confidence distribution charts for operator review.
"""

from __future__ import annotations

import streamlit as st

from uni_vision.visualizer.helpers import fetch_json, get_api_base, get_api_key


def render() -> None:
    """Render the vehicle detection inspection page."""
    st.header("🚗 Vehicle Detection (S2)")

    base = get_api_base()
    api_key = get_api_key()

    st.markdown(
        """
        Stage S2 runs **YOLOv8** vehicle detection on sampled frames.
        This page shows recent detection results with vehicle class
        labels and confidence scores.
        """
    )

    # ── Fetch detections via API ──────────────────────────────────
    page_size = st.slider("Results per page", 5, 50, 20)
    page_num = st.number_input("Page", min_value=1, value=1)

    data = fetch_json(f"{base}/detections?page={page_num}&page_size={page_size}", api_key)

    if not data or not data.get("items"):
        st.info("No vehicle detection records found.")
        return

    items = data["items"]
    total = data.get("total", 0)
    st.caption(f"Showing {len(items)} of {total:,} total detections")

    # ── Summary by vehicle class ──────────────────────────────────
    class_counts: dict[str, int] = {}
    confidences: list[float] = []

    for item in items:
        vc = item.get("vehicle_class", "unknown") or "unknown"
        class_counts[vc] = class_counts.get(vc, 0) + 1
        confidences.append(item["ocr_confidence"])

    st.subheader("Vehicle Class Distribution")
    if class_counts:
        import pandas as pd

        df_classes = pd.DataFrame(list(class_counts.items()), columns=["Vehicle Class", "Count"])
        st.bar_chart(df_classes.set_index("Vehicle Class"))

    # ── Confidence distribution ───────────────────────────────────
    st.subheader("Confidence Distribution")
    if confidences:
        import pandas as pd

        df_conf = pd.DataFrame({"Confidence": confidences})
        st.bar_chart(df_conf["Confidence"].value_counts(bins=10).sort_index())

    # ── Detail table ──────────────────────────────────────────────
    st.subheader("Detection Records")
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "Camera": item["camera_id"],
                "Plate": item["plate_number"],
                "Vehicle": item.get("vehicle_class", ""),
                "Confidence": f"{item['ocr_confidence']:.2f}",
                "Engine": item.get("ocr_engine", ""),
                "Status": item.get("validation_status", ""),
                "Time": item.get("detected_at_utc", ""),
            }
            for item in items
        ]
    )
    st.dataframe(df, use_container_width=True)
