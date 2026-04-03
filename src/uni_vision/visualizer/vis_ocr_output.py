"""OCR Output visualizer — plate image with OCR text overlay and confidence score.

Displays S7 OCR results with engine info, raw vs cleaned text,
confidence distribution, and validation status breakdown.
"""

from __future__ import annotations

import streamlit as st

from uni_vision.visualizer.helpers import fetch_json, get_api_base, get_api_key


def render() -> None:
    """Render the OCR output inspection page."""
    st.header("🔤 OCR Output (S7)")

    base = get_api_base()
    api_key = get_api_key()

    st.markdown(
        """
        Stage S7 runs **OCR extraction** via the primary engine
        (Gemma 4 E2B LLM via Ollama) with automatic fallback to
        EasyOCR on failure. This page shows OCR results, raw text,
        confidence scores, and engine routing.
        """
    )

    # ── Filters ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        engine_filter = st.selectbox("Engine Filter", ["All", "gemma4", "easyocr"])
    with col2:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

    page_size = st.slider("Results per page", 5, 50, 20, key="ocr_page_size")
    data = fetch_json(f"{base}/detections?page=1&page_size={page_size}", api_key)

    if not data or not data.get("items"):
        st.info("No OCR results available.")
        return

    items = data["items"]

    # Apply client-side filters
    if engine_filter != "All":
        items = [i for i in items if i.get("ocr_engine", "") == engine_filter]
    items = [i for i in items if i["ocr_confidence"] >= min_conf]

    if not items:
        st.warning("No results match the current filters.")
        return

    st.caption(f"Showing {len(items)} results")

    # ── Engine breakdown ──────────────────────────────────────────
    st.subheader("Engine Distribution")
    engine_counts: dict[str, int] = {}
    for item in items:
        eng = item.get("ocr_engine", "unknown") or "unknown"
        engine_counts[eng] = engine_counts.get(eng, 0) + 1

    import pandas as pd

    if engine_counts:
        df_eng = pd.DataFrame(
            list(engine_counts.items()), columns=["Engine", "Count"]
        )
        st.bar_chart(df_eng.set_index("Engine"))

    # ── Confidence histogram ──────────────────────────────────────
    st.subheader("Confidence Distribution")
    confidences = [i["ocr_confidence"] for i in items]
    df_conf = pd.DataFrame({"Confidence": confidences})
    st.bar_chart(df_conf["Confidence"].value_counts(bins=10).sort_index())

    # ── OCR results table ─────────────────────────────────────────
    st.subheader("OCR Results")
    df = pd.DataFrame(
        [
            {
                "Camera": item["camera_id"],
                "Raw OCR": item.get("raw_ocr_text", ""),
                "Plate": item["plate_number"],
                "Confidence": f"{item['ocr_confidence']:.2f}",
                "Engine": item.get("ocr_engine", ""),
                "Status": item.get("validation_status", ""),
                "Time": item.get("detected_at_utc", ""),
            }
            for item in items
        ]
    )
    st.dataframe(df, use_container_width=True)

    # ── Individual OCR card view ──────────────────────────────────
    st.subheader("Detailed OCR Cards")
    for item in items[:10]:
        conf = item["ocr_confidence"]
        indicator = "🟢" if conf >= 0.85 else ("🟡" if conf >= 0.6 else "🔴")
        raw = item.get("raw_ocr_text", "")
        plate = item["plate_number"]

        with st.expander(f"{indicator} {plate} — {item['camera_id']}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Raw OCR Text:** `{raw}`")
                st.markdown(f"**Cleaned Plate:** `{plate}`")
            with col2:
                st.markdown(f"**Confidence:** {conf:.3f}")
                st.markdown(f"**Engine:** {item.get('ocr_engine', 'N/A')}")
                st.markdown(f"**Status:** {item.get('validation_status', 'N/A')}")

            # Highlight character differences
            if raw and raw != plate:
                st.markdown("**Corrections Applied:**")
                diffs = []
                for i, (a, b) in enumerate(zip(raw.ljust(len(plate)), plate)):
                    if a != b:
                        diffs.append(f"pos {i}: `{a}` → `{b}`")
                if len(raw) != len(plate):
                    diffs.append(f"length: {len(raw)} → {len(plate)}")
                st.code("\n".join(diffs) if diffs else "No character-level diffs")
