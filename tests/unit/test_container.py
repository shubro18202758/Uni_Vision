"""Tests for the DI container — orchestrator/container.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from uni_vision.common.config import AppConfig
from uni_vision.orchestrator.pipeline import Pipeline


class TestBuildPipeline:
    """Verify ``build_pipeline`` wires all stages correctly."""

    def _build(self, config: AppConfig | None = None) -> Pipeline:
        # Patch cv2 inside the enhance module so createCLAHE is available
        mock_cv2 = MagicMock()
        mock_cv2.createCLAHE.return_value = MagicMock()
        with patch("uni_vision.preprocessing.enhance.cv2", mock_cv2):
            from uni_vision.orchestrator.container import build_pipeline
            return build_pipeline(config or AppConfig())

    def test_returns_pipeline_instance(self) -> None:
        pipeline = self._build()
        assert isinstance(pipeline, Pipeline)

    def test_all_stages_wired(self) -> None:
        pipeline = self._build()
        assert pipeline._vehicle_detector is not None
        assert pipeline._plate_detector is not None
        assert pipeline._straightener is not None
        assert pipeline._enhancer is not None
        assert pipeline._ocr_strategy is not None
        assert pipeline._validator is not None
        assert pipeline._dispatcher is not None
        assert pipeline._vram_monitor is not None

    def test_loads_config_when_none(self) -> None:
        mock_cv2 = MagicMock()
        mock_cv2.createCLAHE.return_value = MagicMock()
        with patch("uni_vision.preprocessing.enhance.cv2", mock_cv2), \
             patch(
                "uni_vision.orchestrator.container.load_config",
                return_value=AppConfig(),
             ) as mock_load:
            from uni_vision.orchestrator.container import build_pipeline
            pipeline = build_pipeline(None)
            mock_load.assert_called_once()
            assert isinstance(pipeline, Pipeline)

    def test_uses_provided_config(self) -> None:
        config = AppConfig()
        pipeline = self._build(config)
        assert pipeline._config is config

    def test_ocr_strategy_has_engines(self) -> None:
        pipeline = self._build()
        strategy = pipeline._ocr_strategy
        assert hasattr(strategy, "_engines")
        assert len(strategy._engines) >= 1

