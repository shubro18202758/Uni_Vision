"""Integration tests: Pipeline smoke test with mock stages.

Wires the real Pipeline class with lightweight mock stage objects
and verifies the S2→S8 processing chain runs end-to-end without
external dependencies (no GPU, no Ollama, no DB).
"""

from __future__ import annotations

import asyncio
import logging
import sys

import numpy as np
import pytest

# Patch the structlog stub to support keyword arguments (structlog-style)
# The conftest stub returns a stdlib Logger which rejects kwargs.
_orig_structlog = sys.modules.get("structlog")
if _orig_structlog is not None:

    class _KWLogger:
        """Logger proxy that accepts keyword arguments like structlog."""

        def __init__(self):
            self._log = logging.getLogger("pipeline_smoke_test")

        def _do(self, fn, msg, *args, **kwargs):
            fn(msg, *args)

        def debug(self, msg, *args, **kwargs):
            self._do(self._log.debug, msg, *args, **kwargs)

        def info(self, msg, *args, **kwargs):
            self._do(self._log.info, msg, *args, **kwargs)

        def warning(self, msg, *args, **kwargs):
            self._do(self._log.warning, msg, *args, **kwargs)

        def error(self, msg, *args, **kwargs):
            self._do(self._log.error, msg, *args, **kwargs)

        def exception(self, msg, *args, **kwargs):
            self._do(self._log.exception, msg, *args, **kwargs)

    _kw_logger_instance = _KWLogger()
    _orig_structlog.get_logger = lambda *a, **kw: _kw_logger_instance  # type: ignore[attr-defined]

# Re-import pipeline so it picks up the patched structlog
import importlib

import uni_vision.orchestrator.pipeline as _pipeline_mod
from uni_vision.common.config import AppConfig
from uni_vision.contracts.dtos import (
    BoundingBox,
    FramePacket,
    OCRResult,
    ProcessedResult,
    ValidationStatus,
)

importlib.reload(_pipeline_mod)
from uni_vision.orchestrator.pipeline import Pipeline

# ── Mock stage objects ────────────────────────────────────────────


class _MockVehicleDetector:
    def detect(self, image):
        return [BoundingBox(x1=10, y1=10, x2=200, y2=200, confidence=0.95, class_id=0, class_name="car")]

    def warmup(self):
        pass

    def release(self):
        pass


class _MockPlateDetector:
    def detect(self, image):
        return [BoundingBox(x1=50, y1=120, x2=180, y2=155, confidence=0.90, class_id=0, class_name="plate")]

    def warmup(self):
        pass

    def release(self):
        pass


class _MockStraightener:
    def process(self, image):
        return image


class _MockEnhancer:
    def process(self, image):
        return image


class _MockOCRStrategy:
    async def extract(self, plate_image, context):
        return OCRResult(
            plate_text="MH12AB1234",
            raw_text="MH12AB1234",
            confidence=0.92,
            reasoning="Mock OCR",
            engine="mock_engine",
            status=ValidationStatus.VALID,
        )


class _MockValidator:
    async def validate(self, ocr_result, plate_image):
        return ProcessedResult(
            plate_text=ocr_result.plate_text,
            raw_ocr_text=ocr_result.raw_text,
            confidence=ocr_result.confidence,
            validation_status=ValidationStatus.VALID,
        )


class _MockDispatcher:
    def __init__(self):
        self.dispatched: list = []

    async def dispatch(self, record, plate_image=None):
        self.dispatched.append(record)


class _MockVRAMMonitor:
    offload_mode = "gpu_primary"

    async def run(self):
        pass

    async def shutdown(self):
        pass


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def mock_dispatcher():
    return _MockDispatcher()


@pytest.fixture
def pipeline(mock_dispatcher) -> Pipeline:
    config = AppConfig()
    return Pipeline(
        config=config,
        vram_monitor=_MockVRAMMonitor(),
        vehicle_detector=_MockVehicleDetector(),
        plate_detector=_MockPlateDetector(),
        straightener=_MockStraightener(),
        enhancer=_MockEnhancer(),
        ocr_strategy=_MockOCRStrategy(),
        validator=_MockValidator(),
        dispatcher=mock_dispatcher,
    )


@pytest.fixture
def sample_frame():
    rng = np.random.default_rng(42)
    return FramePacket(
        camera_id="cam-int-01",
        timestamp_utc=1700000000.0,
        frame_index=1,
        image=rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
    )


# ── Tests ─────────────────────────────────────────────────────────


class TestPipelineSmoke:
    """Verify the pipeline processes a frame through all stages."""

    def test_process_event_dispatches_record(
        self, pipeline: Pipeline, mock_dispatcher: _MockDispatcher, sample_frame
    ) -> None:
        """A single frame should flow through S2→S8 and produce a dispatch."""
        asyncio.get_event_loop().run_until_complete(pipeline._process_event(sample_frame))
        assert len(mock_dispatcher.dispatched) == 1
        record = mock_dispatcher.dispatched[0]
        assert record.plate_number == "MH12AB1234"
        assert record.camera_id == "cam-int-01"
        assert record.ocr_confidence == 0.92
        assert record.validation_status == "valid"

    def test_no_vehicle_no_dispatch(self, mock_dispatcher: _MockDispatcher, sample_frame) -> None:
        """When vehicle detector returns no detections, dispatcher is not called."""

        class _EmptyDetector:
            def detect(self, image):
                return []

            def warmup(self):
                pass

            def release(self):
                pass

        pipeline = Pipeline(
            config=AppConfig(),
            vram_monitor=_MockVRAMMonitor(),
            vehicle_detector=_EmptyDetector(),
            plate_detector=_MockPlateDetector(),
            straightener=_MockStraightener(),
            enhancer=_MockEnhancer(),
            ocr_strategy=_MockOCRStrategy(),
            validator=_MockValidator(),
            dispatcher=mock_dispatcher,
        )
        asyncio.get_event_loop().run_until_complete(pipeline._process_event(sample_frame))
        assert len(mock_dispatcher.dispatched) == 0

    def test_no_plate_no_dispatch(self, mock_dispatcher: _MockDispatcher, sample_frame) -> None:
        """When plate detector returns nothing, no record dispatched."""

        class _EmptyPlateDetector:
            def detect(self, image):
                return []

            def warmup(self):
                pass

            def release(self):
                pass

        pipeline = Pipeline(
            config=AppConfig(),
            vram_monitor=_MockVRAMMonitor(),
            vehicle_detector=_MockVehicleDetector(),
            plate_detector=_EmptyPlateDetector(),
            straightener=_MockStraightener(),
            enhancer=_MockEnhancer(),
            ocr_strategy=_MockOCRStrategy(),
            validator=_MockValidator(),
            dispatcher=mock_dispatcher,
        )
        asyncio.get_event_loop().run_until_complete(pipeline._process_event(sample_frame))
        assert len(mock_dispatcher.dispatched) == 0


class TestPipelineEnqueueAndThrottle:
    """Test the queue-based frame ingestion and backpressure logic."""

    def test_enqueue_frame_accepted(self, pipeline: Pipeline, sample_frame) -> None:
        """Frame should be accepted when queue is not full."""
        accepted = asyncio.get_event_loop().run_until_complete(pipeline.enqueue_frame(sample_frame))
        assert accepted is True
        assert pipeline._inference_queue.qsize() == 1

    def test_enqueue_frame_dropped_when_full(self, pipeline: Pipeline, sample_frame) -> None:
        """When queue is at max capacity, frame should be dropped."""
        # Fill the queue
        for _ in range(pipeline._config.pipeline.inference_queue_maxsize):
            asyncio.get_event_loop().run_until_complete(pipeline.enqueue_frame(sample_frame))
        # Next enqueue should fail
        dropped = asyncio.get_event_loop().run_until_complete(pipeline.enqueue_frame(sample_frame))
        assert dropped is False

    def test_throttle_engaged_at_high_water(self, pipeline: Pipeline, sample_frame) -> None:
        """Throttle flag should engage when queue hits high-water mark."""
        for _ in range(pipeline._high_water + 1):
            asyncio.get_event_loop().run_until_complete(pipeline.enqueue_frame(sample_frame))
        assert pipeline._throttled is True


class TestPipelineLifecycle:
    """Test start and shutdown lifecycle."""

    def test_shutdown_sets_flag(self, pipeline: Pipeline) -> None:
        """Shutdown should set the _shutting_down flag."""
        asyncio.get_event_loop().run_until_complete(pipeline.shutdown())
        assert pipeline._shutting_down is True

    def test_multiple_frames_processed(
        self, pipeline: Pipeline, mock_dispatcher: _MockDispatcher, sample_frame
    ) -> None:
        """Multiple frames should each produce a dispatched record."""
        for _ in range(3):
            asyncio.get_event_loop().run_until_complete(pipeline._process_event(sample_frame))
        assert len(mock_dispatcher.dispatched) == 3
