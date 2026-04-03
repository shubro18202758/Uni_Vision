"""GPU Memory Profiler — runtime VRAM measurement and leak detection.

Unlike the lifecycle manager which uses *estimated* VRAM from model
metadata, this profiler measures *actual* GPU memory before and after
component load/unload operations.  It provides:

  * **Per-component actual VRAM** — measured by diffing GPU memory.
  * **Fragmentation estimation** — gap between allocated and in-use memory.
  * **Leak detection** — memory that isn't freed after component unload.
  * **Feedback to lifecycle** — corrects VRAM estimates with measurements.

Falls back gracefully when `pynvml` or `torch` isn't available.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Generator

log = structlog.get_logger(__name__)

# Try NVML first, then torch, else stub
_HAS_NVML = False
_HAS_TORCH = False

try:
    import pynvml

    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    pass

if not _HAS_NVML:
    try:
        import torch

        if torch.cuda.is_available():
            _HAS_TORCH = True
    except (ImportError, AttributeError):
        pass


def _get_gpu_memory_mb(device_index: int = 0) -> dict[str, float] | None:
    """Return current GPU memory stats in MB."""
    if _HAS_NVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "total_mb": info.total / (1024 * 1024),
                "used_mb": info.used / (1024 * 1024),
                "free_mb": info.free / (1024 * 1024),
            }
        except Exception as exc:
            log.debug("nvml_query_failed", error=str(exc))
            return None

    if _HAS_TORCH:
        import torch

        try:
            return {
                "total_mb": torch.cuda.get_device_properties(device_index).total_mem / (1024 * 1024),
                "used_mb": torch.cuda.memory_allocated(device_index) / (1024 * 1024),
                "free_mb": (
                    torch.cuda.get_device_properties(device_index).total_mem - torch.cuda.memory_allocated(device_index)
                )
                / (1024 * 1024),
                "reserved_mb": torch.cuda.memory_reserved(device_index) / (1024 * 1024),
            }
        except Exception as exc:
            log.debug("torch_cuda_query_failed", error=str(exc))
            return None

    return None


@dataclass
class VRAMSnapshot:
    """A point-in-time GPU memory snapshot."""

    timestamp: float
    used_mb: float
    free_mb: float
    total_mb: float
    label: str = ""

    @property
    def utilisation(self) -> float:
        if self.total_mb <= 0:
            return 0.0
        return self.used_mb / self.total_mb


@dataclass
class ComponentVRAMMeasurement:
    """Measured VRAM usage for a component load/unload."""

    component_id: str
    estimated_mb: int
    measured_mb: float
    delta: float  # measured - estimated
    load_time_ms: float
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def accuracy_pct(self) -> float:
        """How accurate the estimate was (100% = perfect)."""
        if self.estimated_mb == 0:
            return 0.0
        return max(0.0, 100.0 * (1.0 - abs(self.delta) / self.estimated_mb))


@dataclass
class LeakSuspect:
    """A suspected memory leak."""

    component_id: str
    expected_free_mb: float
    actual_free_mb: float
    leak_mb: float
    timestamp: float = field(default_factory=time.monotonic)


class GPUProfiler:
    """Runtime GPU memory profiler with leak detection.

    Parameters
    ----------
    device_index:
        CUDA device index to monitor.
    leak_threshold_mb:
        Minimum unreleased memory to flag as a leak.
    snapshot_interval_s:
        Minimum seconds between background snapshots.
    """

    def __init__(
        self,
        *,
        device_index: int = 0,
        leak_threshold_mb: float = 50.0,
        snapshot_interval_s: float = 10.0,
    ) -> None:
        self._device = device_index
        self._leak_threshold = leak_threshold_mb
        self._snapshot_interval = snapshot_interval_s
        self._available = _HAS_NVML or _HAS_TORCH

        # History
        self._measurements: dict[str, ComponentVRAMMeasurement] = {}
        self._leaks: list[LeakSuspect] = []
        self._snapshots: list[VRAMSnapshot] = []
        self._last_snapshot: float = 0.0

    @property
    def available(self) -> bool:
        """Whether GPU profiling is available."""
        return self._available

    def snapshot(self, label: str = "") -> VRAMSnapshot | None:
        """Take a GPU memory snapshot."""
        mem = _get_gpu_memory_mb(self._device)
        if mem is None:
            return None
        snap = VRAMSnapshot(
            timestamp=time.monotonic(),
            used_mb=mem["used_mb"],
            free_mb=mem["free_mb"],
            total_mb=mem["total_mb"],
            label=label,
        )
        self._snapshots.append(snap)
        # Limit history
        if len(self._snapshots) > 500:
            self._snapshots = self._snapshots[-250:]
        self._last_snapshot = snap.timestamp
        return snap

    @contextmanager
    def measure_load(
        self,
        component_id: str,
        estimated_mb: int,
    ) -> Generator[None, None, None]:
        """Context manager that measures actual VRAM consumed by a load.

        Usage::

            with profiler.measure_load("yolov8n", 200):
                component.load(device="cuda")
        """
        before = _get_gpu_memory_mb(self._device)
        t0 = time.monotonic()
        try:
            yield
        finally:
            t1 = time.monotonic()
            after = _get_gpu_memory_mb(self._device)
            if before is not None and after is not None:
                measured = after["used_mb"] - before["used_mb"]
                delta = measured - estimated_mb
                measurement = ComponentVRAMMeasurement(
                    component_id=component_id,
                    estimated_mb=estimated_mb,
                    measured_mb=measured,
                    delta=delta,
                    load_time_ms=(t1 - t0) * 1000,
                )
                self._measurements[component_id] = measurement
                log.info(
                    "vram_measured",
                    component=component_id,
                    estimated_mb=estimated_mb,
                    measured_mb=round(measured, 1),
                    delta_mb=round(delta, 1),
                    load_ms=round(measurement.load_time_ms, 1),
                )

    @contextmanager
    def measure_unload(
        self,
        component_id: str,
        expected_free_mb: float,
    ) -> Generator[None, None, None]:
        """Context manager that checks memory is properly freed on unload."""
        before = _get_gpu_memory_mb(self._device)
        try:
            yield
        finally:
            after = _get_gpu_memory_mb(self._device)
            if before is not None and after is not None:
                actual_freed = before["used_mb"] - after["used_mb"]
                leak = expected_free_mb - actual_freed
                if leak > self._leak_threshold:
                    suspect = LeakSuspect(
                        component_id=component_id,
                        expected_free_mb=expected_free_mb,
                        actual_free_mb=actual_freed,
                        leak_mb=leak,
                    )
                    self._leaks.append(suspect)
                    log.warning(
                        "vram_leak_suspected",
                        component=component_id,
                        expected_free_mb=round(expected_free_mb, 1),
                        actual_freed_mb=round(actual_freed, 1),
                        leak_mb=round(leak, 1),
                    )
                # Clean up measurement
                self._measurements.pop(component_id, None)

    def get_measured_vram(self, component_id: str) -> float | None:
        """Return the actual measured VRAM for a loaded component."""
        m = self._measurements.get(component_id)
        return m.measured_mb if m else None

    def get_vram_correction(self, component_id: str) -> int | None:
        """Return corrected VRAM estimate based on measurement."""
        m = self._measurements.get(component_id)
        if m is None:
            return None
        return max(1, int(m.measured_mb))

    def get_current_usage(self) -> dict[str, float] | None:
        """Return current GPU memory usage."""
        return _get_gpu_memory_mb(self._device)

    def status(self) -> dict[str, Any]:
        current = _get_gpu_memory_mb(self._device)
        return {
            "available": self._available,
            "device_index": self._device,
            "current_usage": ({k: round(v, 1) for k, v in current.items()} if current else None),
            "tracked_components": len(self._measurements),
            "measurements": {
                cid: {
                    "estimated_mb": m.estimated_mb,
                    "measured_mb": round(m.measured_mb, 1),
                    "delta_mb": round(m.delta, 1),
                    "accuracy_pct": round(m.accuracy_pct, 1),
                }
                for cid, m in self._measurements.items()
            },
            "suspected_leaks": len(self._leaks),
            "leak_details": [
                {
                    "component": lk.component_id,
                    "leak_mb": round(lk.leak_mb, 1),
                    "age_s": round(time.monotonic() - lk.timestamp, 1),
                }
                for lk in self._leaks[-5:]
            ],
            "total_snapshots": len(self._snapshots),
        }
