"""Uni_Vision Streamlit Visualizer — main entry point.

Launch with: streamlit run src/uni_vision/visualizer/app.py

Provides 8 pipeline debugging views as per PRD §14.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Uni_Vision Pipeline Debugger",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.sidebar.title("Uni_Vision")
    st.sidebar.markdown("Pipeline Stage Visualizer")

    page = st.sidebar.radio(
        "Select Stage",
        [
            "Pipeline Stats",
            "Frame Sampler",
            "Vehicle Detection",
            "Plate Detection",
            "Crop & Straighten",
            "Enhancement",
            "OCR Output",
            "Post-Processing",
        ],
    )

    if page == "Pipeline Stats":
        from uni_vision.visualizer.vis_pipeline_stats import render

        render()
    elif page == "Frame Sampler":
        from uni_vision.visualizer.vis_frame_sampler import render

        render()
    elif page == "Vehicle Detection":
        from uni_vision.visualizer.vis_vehicle_detection import render

        render()
    elif page == "Plate Detection":
        from uni_vision.visualizer.vis_plate_detection import render

        render()
    elif page == "Crop & Straighten":
        from uni_vision.visualizer.vis_crop_straighten import render

        render()
    elif page == "Enhancement":
        from uni_vision.visualizer.vis_enhancement import render

        render()
    elif page == "OCR Output":
        from uni_vision.visualizer.vis_ocr_output import render

        render()
    elif page == "Post-Processing":
        from uni_vision.visualizer.vis_postprocess import render

        render()


if __name__ == "__main__":
    main()
