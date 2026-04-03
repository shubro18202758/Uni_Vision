"""Shared test fixtures for the Uni_Vision test suite."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Stub out heavy third-party modules before any uni_vision import ──
# This lets the test suite run in a lightweight CI environment without
# GPU drivers, pynvml, OpenCV-headless, structlog, asyncpg, etc.

_STUB_MODULES = [
    "cv2",
    "pynvml",
    "asyncpg",
    "structlog",
    "tensorrt",
    "pycuda",
    "pycuda.driver",
    "pycuda.autoinit",
    "easyocr",
    "torch",
    "ultralytics",
    "onnxruntime",
    "httpx",
    "aioboto3",
    "boto3",
    "redis",
    "prometheus_client",
]


def _install_stubs() -> None:
    """Insert lightweight MagicMock stubs for heavyweight dependencies."""
    for mod_name in _STUB_MODULES:
        if mod_name not in sys.modules:
            stub = MagicMock(spec=[])
            # cv2 needs real-enough functions so phash tests work
            if mod_name == "cv2":
                stub = _build_cv2_stub()
            # prometheus_client needs real Counter/Histogram/Gauge stubs
            elif mod_name == "prometheus_client":
                stub = _build_prometheus_stub()
            # structlog needs get_logger returning a real-enough logger
            elif mod_name == "structlog":
                stub = _build_structlog_stub()
            sys.modules[mod_name] = stub


def _build_cv2_stub() -> ModuleType:
    """Build a minimal cv2 stub with working cvtColor and resize."""
    mod = ModuleType("cv2")
    mod.COLOR_BGR2GRAY = 6  # type: ignore[attr-defined]
    mod.INTER_AREA = 3  # type: ignore[attr-defined]

    def cvtColor(src, code):  # noqa: N802
        if code == 6:  # COLOR_BGR2GRAY
            # Simple luminance conversion
            return (0.299 * src[:, :, 2] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 0]).astype(np.uint8)
        return src

    def resize(src, dsize, interpolation=None):
        from PIL import Image

        img = Image.fromarray(src)
        img = img.resize(dsize, Image.BILINEAR)
        return np.array(img)

    mod.cvtColor = cvtColor  # type: ignore[attr-defined]
    mod.resize = resize  # type: ignore[attr-defined]
    return mod


def _build_prometheus_stub() -> ModuleType:
    """Build a minimal prometheus_client stub with callable metric classes."""
    mod = ModuleType("prometheus_client")

    class _FakeMetric:
        def __init__(self, *args, **kwargs):
            self._children = {}

        def inc(self, amount=1):
            pass

        def dec(self, amount=1):
            pass

        def set(self, value):
            pass

        def observe(self, value):
            pass

        def labels(self, **kwargs):
            return self

    mod.Counter = _FakeMetric  # type: ignore[attr-defined]
    mod.Histogram = _FakeMetric  # type: ignore[attr-defined]
    mod.Gauge = _FakeMetric  # type: ignore[attr-defined]
    mod.Summary = _FakeMetric  # type: ignore[attr-defined]

    # Functions / objects used by the /metrics and /stats endpoints
    def generate_latest(registry=None):
        return b"# HELP fake metric\n"

    mod.generate_latest = generate_latest  # type: ignore[attr-defined]
    mod.CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"  # type: ignore[attr-defined]

    class _FakeRegistry:
        def collect(self):
            return []

    mod.REGISTRY = _FakeRegistry()  # type: ignore[attr-defined]
    return mod


def _build_structlog_stub() -> ModuleType:
    """Build a minimal structlog stub that returns stdlib-compatible loggers.

    structlog loggers accept arbitrary **kwargs (key=value context).
    The stdlib Logger._log() does NOT accept extra kwargs, so we wrap
    it with a thin adapter that silently drops them.
    """
    import logging

    mod = ModuleType("structlog")

    class _StructlogAdapter:
        """Minimal adapter matching structlog's .info/.warning/.error/.debug API."""

        def __init__(self, name: str = "structlog_stub"):
            self._logger = logging.getLogger(name)

        def _log(self, level: int, msg: str, **kwargs):
            # Drop structlog-style kwargs; just log the message
            if self._logger.isEnabledFor(level):
                self._logger.log(level, msg)

        def debug(self, msg, *args, **kwargs):
            self._log(logging.DEBUG, msg, **kwargs)

        def info(self, msg, *args, **kwargs):
            self._log(logging.INFO, msg, **kwargs)

        def warning(self, msg, *args, **kwargs):
            self._log(logging.WARNING, msg, **kwargs)

        warn = warning

        def error(self, msg, *args, **kwargs):
            self._log(logging.ERROR, msg, **kwargs)

        def critical(self, msg, *args, **kwargs):
            self._log(logging.CRITICAL, msg, **kwargs)

        def exception(self, msg, *args, **kwargs):
            self._log(logging.ERROR, msg, **kwargs)

        def msg(self, msg, *args, **kwargs):
            self._log(logging.INFO, msg, **kwargs)

        def bind(self, **kwargs):
            return self

        def unbind(self, *keys):
            return self

        def new(self, **kwargs):
            return self

    def get_logger(*args, **kwargs):
        return _StructlogAdapter()

    mod.get_logger = get_logger  # type: ignore[attr-defined]
    return mod


# Install stubs before pytest collects any test modules
_install_stubs()


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_bgr_image() -> np.ndarray:
    """A small 64×64 BGR test image with some variation."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """A small 64×64 grayscale test image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture
def validation_config():
    """A ValidationConfig with default Indian locale patterns."""
    from uni_vision.common.config import ValidationConfig

    return ValidationConfig()


@pytest.fixture
def dedup_config():
    """A DeduplicationConfig with a short window for testing."""
    from uni_vision.common.config import DeduplicationConfig

    return DeduplicationConfig(window_seconds=2.0, purge_interval_seconds=1.0)


@pytest.fixture
def dispatch_config():
    """Default DispatchConfig for tests."""
    from uni_vision.common.config import DispatchConfig

    return DispatchConfig()


@pytest.fixture
def sample_detection_record():
    """A typical valid DetectionRecord for testing."""
    from uni_vision.contracts.dtos import DetectionRecord

    return DetectionRecord(
        id="test-001",
        camera_id="cam_01",
        plate_number="MH12AB1234",
        raw_ocr_text="MH12AB1234",
        ocr_confidence=0.92,
        ocr_engine="ollama_llm",
        vehicle_class="car",
        vehicle_image_path="",
        plate_image_path="",
        detected_at_utc="2025-01-01T00:00:00Z",
        validation_status="valid",
        location_tag="toll_gate_1",
    )
