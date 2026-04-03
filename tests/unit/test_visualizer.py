"""Tests for the Streamlit visualizer modules — import and structure checks."""

from __future__ import annotations

import importlib
import pathlib

import pytest

_streamlit_available = importlib.util.find_spec("streamlit") is not None
_skip_no_streamlit = pytest.mark.skipif(not _streamlit_available, reason="streamlit not installed")

# All 8 visualizer modules + helpers + app.
_MODULES = [
    "uni_vision.visualizer.app",
    "uni_vision.visualizer.helpers",
    "uni_vision.visualizer.vis_pipeline_stats",
    "uni_vision.visualizer.vis_frame_sampler",
    "uni_vision.visualizer.vis_vehicle_detection",
    "uni_vision.visualizer.vis_plate_detection",
    "uni_vision.visualizer.vis_crop_straighten",
    "uni_vision.visualizer.vis_enhancement",
    "uni_vision.visualizer.vis_ocr_output",
    "uni_vision.visualizer.vis_postprocess",
]

_RENDER_MODULES = [m for m in _MODULES if m.startswith("uni_vision.visualizer.vis_")]


class TestVisualizerImports:
    """Ensure all visualizer modules are importable."""

    @_skip_no_streamlit
    @pytest.mark.parametrize("module_name", _MODULES)
    def test_module_importable(self, module_name: str) -> None:
        mod = importlib.import_module(module_name)
        assert mod is not None

    @_skip_no_streamlit
    @pytest.mark.parametrize("module_name", _RENDER_MODULES)
    def test_render_function_exists(self, module_name: str) -> None:
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "render"), f"{module_name} must export a render() function"
        assert callable(mod.render)


class TestVisualizerFiles:
    """Verify all visualizer source files physically exist."""

    _VIS_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "uni_vision" / "visualizer"

    _EXPECTED_FILES = [
        "__init__.py",
        "app.py",
        "helpers.py",
        "vis_pipeline_stats.py",
        "vis_frame_sampler.py",
        "vis_vehicle_detection.py",
        "vis_plate_detection.py",
        "vis_crop_straighten.py",
        "vis_enhancement.py",
        "vis_ocr_output.py",
        "vis_postprocess.py",
    ]

    @pytest.mark.parametrize("filename", _EXPECTED_FILES)
    def test_file_exists(self, filename: str) -> None:
        assert (self._VIS_DIR / filename).is_file(), f"Missing: {filename}"

    def test_all_render_modules_present(self) -> None:
        """Verify all 8 PRD §14 visualizer modules exist as files."""
        vis_files = [f.stem for f in self._VIS_DIR.glob("vis_*.py")]
        assert len(vis_files) == 8, f"Expected 8 vis_* modules, found {len(vis_files)}: {vis_files}"


class TestHelpers:
    """Test helper utilities (no streamlit import needed at function level)."""

    @_skip_no_streamlit
    def test_api_headers_empty_key(self) -> None:
        from uni_vision.visualizer.helpers import api_headers

        h = api_headers("")
        assert "X-API-Key" not in h
        assert h["Accept"] == "application/json"

    @_skip_no_streamlit
    def test_api_headers_with_key(self) -> None:
        from uni_vision.visualizer.helpers import api_headers

        h = api_headers("test-key-123")
        assert h["X-API-Key"] == "test-key-123"
        assert h["Accept"] == "application/json"


class TestAppStructure:
    """Test app.py main structure."""

    @_skip_no_streamlit
    def test_main_function_exists(self) -> None:
        from uni_vision.visualizer.app import main

        assert callable(main)

    def test_module_count(self) -> None:
        """Verify all 8 PRD §14 visualizer modules are registered."""
        assert len(_RENDER_MODULES) == 8
