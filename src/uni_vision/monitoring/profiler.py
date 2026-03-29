"""Stage-level profiling decorators and VRAM telemetry hooks — spec §7.3, §12.

Provides two composable decorators and a VRAM-sampling context manager
that dynamically measure execution time and GPU memory allocation of
every pipeline module at runtime.

Design goals
~~~~~~~~~~~~
* **Zero-overhead when disabled** — the ``@profile_stage`` decorator
  short-circuits to the raw callable when ``UV_PROFILING_ENABLED=0``.
* **No import-time GPU dependency** — ``pynvml`` and ``torch`` are
  imported lazily inside the sampling functions.
* **Works with sync and async** — ``@profile_stage`` inspects the
  wrapped callable and dispatches accordingly.
* **Atomic region snapshots** — ``VRAMSampler`` captures used VRAM
  before and after a code block, attributing the delta to a named
  memory region.

Usage::

    from uni_vision.monitoring.profiler import profile_stage, vram_sampler

    @profile_stage("S2_vehicle_detect")
    def detect_vehicles(image): ...

    @profile_stage("S7_ocr")
    async def run_ocr(plate_image, context): ...

    with vram_sampler("region_C"):
        vehicles = detector.detect(frame)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    List,
    Optional,
)

from uni_vision.monitoring.metrics import STAGE_LATENCY

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────

_PROFILING_ENABLED: bool = True


def set_profiling_enabled(enabled: bool) -> None:
    """Toggle profiling at runtime (e.g. from env var at startup)."""
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = enabled


# ── VRAM query helpers (lazy imports) ─────────────────────────────


# Cached pynvml handles — avoids nvmlInit/nvmlShutdown on every probe.
_NVML_HANDLES: Dict[int, Any] = {}
_NVML_INITED: bool = False


def _ensure_nvml() -> bool:
    """Lazily initialise pynvml once; return True if available."""
    global _NVML_INITED
    if _NVML_INITED:
        return True
    try:
        import pynvml

        pynvml.nvmlInit()
        _NVML_INITED = True
        return True
    except Exception:
        return False


def _get_nvml_handle(device_index: int) -> Any:
    """Return a cached pynvml device handle."""
    if device_index not in _NVML_HANDLES:
        import pynvml

        _NVML_HANDLES[device_index] = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    return _NVML_HANDLES[device_index]


def _query_vram_used_mb(device_index: int = 0) -> float:
    """Return current VRAM usage in MB.  Falls back gracefully."""
    try:
        if _ensure_nvml():
            import pynvml

            handle = _get_nvml_handle(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / (1024 * 1024)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device_index)
            return allocated / (1024 * 1024)
    except Exception:
        pass
    return -1.0


def _query_torch_allocated_mb(device_index: int = 0) -> float:
    """Return torch-level allocated VRAM in MB."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(device_index) / (1024 * 1024)
    except Exception:
        pass
    return -1.0


def _query_torch_reserved_mb(device_index: int = 0) -> float:
    """Return torch-level reserved (cached) VRAM in MB."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_reserved(device_index) / (1024 * 1024)
    except Exception:
        pass
    return -1.0


# ── Stage profile record ─────────────────────────────────────────


@dataclass
class StageProfile:
    """Captures a single stage execution measurement."""

    stage: str
    wall_time_ms: float
    vram_before_mb: float
    vram_after_mb: float
    vram_delta_mb: float
    torch_allocated_before_mb: float = -1.0
    torch_allocated_after_mb: float = -1.0
    torch_reserved_mb: float = -1.0


# ── Profile history (ring buffer — O(1) append via deque) ─────────

_MAX_HISTORY: int = 512
_profile_history: Deque[StageProfile] = deque(maxlen=_MAX_HISTORY)


def get_profile_history() -> List[StageProfile]:
    """Return a **copy** of the most recent stage profiles."""
    return list(_profile_history)


def clear_profile_history() -> None:
    """Reset the profile history buffer."""
    _profile_history.clear()


def _record_profile(profile: StageProfile) -> None:
    """Append a profile record; oldest auto-evicted by deque maxlen."""
    _profile_history.append(profile)


# ── Core profiling decorator ─────────────────────────────────────


def profile_stage(
    stage_name: str,
    *,
    device_index: int = 0,
    track_vram: bool = True,
) -> Callable:
    """Decorator that records wall-time and VRAM delta for a pipeline stage.

    Works seamlessly with both synchronous and asynchronous callables.

    Parameters
    ----------
    stage_name:
        Label used in Prometheus ``STAGE_LATENCY`` histogram and in the
        in-memory profile ring buffer.
    device_index:
        CUDA device to sample (default 0).
    track_vram:
        If ``False``, skip VRAM sampling (useful for CPU-only stages).
    """

    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _PROFILING_ENABLED:
                    return await fn(*args, **kwargs)

                vram_before = _query_vram_used_mb(device_index) if track_vram else -1.0
                torch_alloc_before = _query_torch_allocated_mb(device_index) if track_vram else -1.0

                t0 = time.perf_counter()
                result = await fn(*args, **kwargs)
                elapsed = time.perf_counter() - t0

                vram_after = _query_vram_used_mb(device_index) if track_vram else -1.0
                torch_alloc_after = _query_torch_allocated_mb(device_index) if track_vram else -1.0
                torch_reserved = _query_torch_reserved_mb(device_index) if track_vram else -1.0

                elapsed_ms = elapsed * 1000.0
                STAGE_LATENCY.labels(stage=stage_name).observe(elapsed)

                profile = StageProfile(
                    stage=stage_name,
                    wall_time_ms=round(elapsed_ms, 2),
                    vram_before_mb=round(vram_before, 1),
                    vram_after_mb=round(vram_after, 1),
                    vram_delta_mb=round(vram_after - vram_before, 1) if vram_before >= 0 else 0.0,
                    torch_allocated_before_mb=round(torch_alloc_before, 1),
                    torch_allocated_after_mb=round(torch_alloc_after, 1),
                    torch_reserved_mb=round(torch_reserved, 1),
                )
                _record_profile(profile)

                logger.debug(
                    "stage_profile stage=%s wall_ms=%.2f vram_delta_mb=%.1f",
                    stage_name,
                    elapsed_ms,
                    profile.vram_delta_mb,
                )
                return result

            return async_wrapper

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _PROFILING_ENABLED:
                    return fn(*args, **kwargs)

                vram_before = _query_vram_used_mb(device_index) if track_vram else -1.0
                torch_alloc_before = _query_torch_allocated_mb(device_index) if track_vram else -1.0

                t0 = time.perf_counter()
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - t0

                vram_after = _query_vram_used_mb(device_index) if track_vram else -1.0
                torch_alloc_after = _query_torch_allocated_mb(device_index) if track_vram else -1.0
                torch_reserved = _query_torch_reserved_mb(device_index) if track_vram else -1.0

                elapsed_ms = elapsed * 1000.0
                STAGE_LATENCY.labels(stage=stage_name).observe(elapsed)

                profile = StageProfile(
                    stage=stage_name,
                    wall_time_ms=round(elapsed_ms, 2),
                    vram_before_mb=round(vram_before, 1),
                    vram_after_mb=round(vram_after, 1),
                    vram_delta_mb=round(vram_after - vram_before, 1) if vram_before >= 0 else 0.0,
                    torch_allocated_before_mb=round(torch_alloc_before, 1),
                    torch_allocated_after_mb=round(torch_alloc_after, 1),
                    torch_reserved_mb=round(torch_reserved, 1),
                )
                _record_profile(profile)

                logger.debug(
                    "stage_profile stage=%s wall_ms=%.2f vram_delta_mb=%.1f",
                    stage_name,
                    elapsed_ms,
                    profile.vram_delta_mb,
                )
                return result

            return sync_wrapper

    return decorator


# ── VRAM sampling context manager ─────────────────────────────────


@dataclass
class VRAMSnapshot:
    """Result of a ``vram_sampler`` context block."""

    region: str
    used_before_mb: float
    used_after_mb: float
    delta_mb: float
    peak_mb: float
    torch_allocated_before_mb: float
    torch_allocated_after_mb: float


@contextmanager
def vram_sampler(
    region: str,
    *,
    device_index: int = 0,
) -> Generator[VRAMSnapshot, None, None]:
    """Context manager that brackets a code block with VRAM samples.

    Yields a mutable ``VRAMSnapshot`` that is populated on exit.

    Usage::

        with vram_sampler("region_C") as snap:
            vehicles = detector.detect(frame)
        print(f"Region C delta: {snap.delta_mb:.1f} MB")
    """
    before = _query_vram_used_mb(device_index)
    torch_before = _query_torch_allocated_mb(device_index)

    snapshot = VRAMSnapshot(
        region=region,
        used_before_mb=round(before, 1),
        used_after_mb=0.0,
        delta_mb=0.0,
        peak_mb=0.0,
        torch_allocated_before_mb=round(torch_before, 1),
        torch_allocated_after_mb=0.0,
    )

    yield snapshot

    after = _query_vram_used_mb(device_index)
    torch_after = _query_torch_allocated_mb(device_index)

    snapshot.used_after_mb = round(after, 1)
    snapshot.delta_mb = round(after - before, 1) if (before >= 0 and after >= 0) else 0.0
    snapshot.peak_mb = snapshot.used_after_mb  # best estimate without polling
    snapshot.torch_allocated_after_mb = round(torch_after, 1)

    logger.debug(
        "vram_sample region=%s delta_mb=%.1f before=%.1f after=%.1f",
        region,
        snapshot.delta_mb,
        before,
        after,
    )


# ── Pipeline-level telemetry hook ─────────────────────────────────


@dataclass
class EventTelemetry:
    """Aggregated telemetry for a single pipeline event (S0→S8)."""

    event_id: str = ""
    camera_id: str = ""
    total_wall_ms: float = 0.0
    stage_profiles: List[StageProfile] = field(default_factory=list)
    peak_vram_mb: float = 0.0
    vram_at_fence_mb: float = 0.0


class PipelineTelemetryHook:
    """Hooks into the pipeline to collect per-event telemetry.

    The pipeline calls ``begin_event()`` at the start of
    ``_process_event`` and ``end_event()`` at the end.
    Stage profiles accumulated during the event are snapshotted.
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._current: Optional[EventTelemetry] = None
        self._history: List[EventTelemetry] = []
        self._max_history: int = 128

    def begin_event(self, camera_id: str, event_id: str = "") -> None:
        """Mark the start of a pipeline event."""
        self._current = EventTelemetry(
            event_id=event_id,
            camera_id=camera_id,
        )

    def record_fence_vram(self) -> None:
        """Snapshot VRAM at the memory fence point (after vision, before LLM)."""
        if self._current is not None:
            self._current.vram_at_fence_mb = round(
                _query_vram_used_mb(self._device_index), 1
            )

    def end_event(self, elapsed_ms: float) -> Optional[EventTelemetry]:
        """Finalise the current event telemetry and archive it."""
        if self._current is None:
            return None

        self._current.total_wall_ms = round(elapsed_ms, 2)
        self._current.peak_vram_mb = round(
            _query_vram_used_mb(self._device_index), 1
        )

        # Capture stage profiles accumulated during this event
        recent = get_profile_history()
        self._current.stage_profiles = list(recent[-10:])  # last N stages

        event = self._current
        self._current = None

        if len(self._history) >= self._max_history:
            self._history.pop(0)
        self._history.append(event)

        logger.info(
            "event_telemetry camera=%s total_ms=%.2f peak_vram_mb=%.1f fence_vram_mb=%.1f",
            event.camera_id,
            event.total_wall_ms,
            event.peak_vram_mb,
            event.vram_at_fence_mb,
        )
        return event

    def get_event_history(self) -> List[EventTelemetry]:
        """Return a copy of archived event telemetry records."""
        return list(self._history)
