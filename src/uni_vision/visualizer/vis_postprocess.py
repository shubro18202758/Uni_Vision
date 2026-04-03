"""Post-Processing visualizer — input vs output text with corrections highlighted.

Shows S8 validation and post-processing results: regex checks,
character corrections, and final validation status breakdown.
"""

from __future__ import annotations

import streamlit as st

from uni_vision.visualizer.helpers import fetch_json, get_api_base, get_api_key


def render() -> None:
    """Render the post-processing inspection page."""
    st.header("🔧 Post-Processing (S8)")

    base = get_api_base()
    api_key = get_api_key()

    st.markdown(
        """
        Stage S8 applies **post-processing validation** to OCR output:

        - **Regex validation** against known plate formats
        - **Character substitution** (e.g., O→0, I→1)
        - **Confidence thresholding** for quality gating
        - **Deduplication** via sliding window

        Records that fail validation are routed to the **OCR audit log**
        for manual review.
        """
    )

    # ── Validation status breakdown ───────────────────────────────
    data = fetch_json(f"{base}/detections?page=1&page_size=50", api_key)

    if not data or not data.get("items"):
        st.info("No detection records available.")
        return

    items = data["items"]
    total = data.get("total", 0)

    status_counts: dict[str, int] = {}
    corrections: list[dict] = []

    for item in items:
        status = item.get("validation_status", "unknown") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

        raw = item.get("raw_ocr_text", "")
        plate = item["plate_number"]
        if raw and raw != plate:
            corrections.append(
                {
                    "camera": item["camera_id"],
                    "raw": raw,
                    "corrected": plate,
                    "confidence": item["ocr_confidence"],
                    "status": status,
                }
            )

    # ── Status pie chart ──────────────────────────────────────────
    st.subheader("Validation Status Breakdown")
    import pandas as pd

    df_status = pd.DataFrame(list(status_counts.items()), columns=["Status", "Count"])
    st.bar_chart(df_status.set_index("Status"))

    st.caption(f"Based on {len(items)} of {total:,} total records")

    # ── Correction analysis ───────────────────────────────────────
    st.subheader(f"Character Corrections ({len(corrections)} found)")

    if corrections:
        df_corr = pd.DataFrame(corrections)
        st.dataframe(df_corr, use_container_width=True)

        # Common substitutions
        st.subheader("Common Character Substitutions")
        sub_counts: dict[str, int] = {}
        for corr in corrections:
            raw = corr["raw"]
            fixed = corr["corrected"]
            for a, b in zip(raw.ljust(len(fixed)), fixed, strict=False):
                if a != b:
                    key = f"`{a}` → `{b}`"
                    sub_counts[key] = sub_counts.get(key, 0) + 1

        if sub_counts:
            df_subs = pd.DataFrame(
                sorted(sub_counts.items(), key=lambda x: -x[1]),
                columns=["Substitution", "Count"],
            )
            st.dataframe(df_subs, use_container_width=True)
    else:
        st.success("No character corrections needed — all OCR output matches final plates.")

    # ── Stats summary ─────────────────────────────────────────────
    st.divider()
    st.subheader("Quality Summary")
    c1, c2, c3 = st.columns(3)

    valid = status_counts.get("valid", 0)
    total_shown = len(items)
    pass_rate = (valid / total_shown * 100) if total_shown > 0 else 0.0

    c1.metric("Valid Plates", valid)
    c2.metric("Corrections Applied", len(corrections))
    c3.metric("Pass Rate", f"{pass_rate:.1f}%")
