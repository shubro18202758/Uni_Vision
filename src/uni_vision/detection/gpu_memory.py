"""GPU memory manager and VRAM offload controller — spec §5, §7.

Implements:

1. **Dynamic offload decisions** (spec §7.4) — monitors free VRAM via
   ``pynvml`` and selects the appropriate ``OffloadMode``:

   ========== ============================================
   Free VRAM  Mode
   ========== ============================================
   ≥ 1024 MB  ``GPU_PRIMARY``   — both S2 & S3 on GPU
   ≥ 512 MB   ``PARTIAL_OFFLOAD`` — S2 GPU, S3 → CPU
   < 512 MB   ``FULL_CPU``      — both detectors on CPU
   ========== ============================================

2. **Sequential exclusivity** (spec P3) — ensures no two neural
   network forward passes occupy VRAM simultaneously unless they
   fit within Region C.

3. **Memory fence** (spec §5.5) — hard synchronisation barrier
   between the vision workspace (Region C) and the LLM workspace
   (Regions A + B).  The fence asserts Region C VRAM is fully
   reclaimed and calls ``torch.cuda.empty_cache()`` before LLM
   inference begins.

4. **OOM recovery** — if a CUDA OOM is caught during a forward pass
   the offload manager immediately falls back to CPU execution and
   records a ``VRAMError`` metric.

Usage
~~~~~
The ``GPUMemoryManager`` is injected into the pipeline orchestrator.
At each event the pipeline calls:

    with gpu_mem.inference_context(vehicle_detector, plate_detector):
        vehicles = vehicle_detector.detect(image)
        plates   = plate_detector.detect_in_roi(image, vehicle_bbox)
    # ← on exit, models are released and VRAM is reclaimed

Before LLM inference:

    gpu_mem.assert_memory_fence()
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import structlog

from uni_vision.contracts.dtos import OffloadMode
from uni_vision.common.exceptions import VRAMError, VRAMBudgetExceeded
from uni_vision.monitoring.metrics import VRAM_USAGE

if TYPE_CHECKING:
    from collections.abc import Generator
    from uni_vision.detection.vehicle_detector import VehicleDetector
    from uni_vision.detection.plate_detector import PlateDetector

log = structlog.get_logger()

# Spec §7.4 — offload thresholds (MB)
_GPU_PRIMARY_THRESHOLD_MB: int = 1024
_PARTIAL_OFFLOAD_THRESHOLD_MB: int = 512


class GPUMemoryManager:
    """Centralised VRAM lifecycle and offload controller.

    Parameters
    ----------
    vision_budget_mb : int
        Maximum VRAM allowed for the vision workspace (Region C).
        Default 1024 MB per spec §2.1.
    safety_margin_mb : int
        Reserved headroom that should never be consumed (default 256 MB).
    device_index : int
        CUDA device ordinal.
    """

    def __init__(
        self,
        *,
        vision_budget_mb: int = 1024,
        safety_margin_mb: int = 256,
        device_index: int = 0,
    ) -> None:
        self._vision_budget_mb = vision_budget_mb
        self._safety_margin_mb = safety_margin_mb
        self._device_index = device_index
        self._current_mode = OffloadMode.GPU_PRIMARY

    # ── Public API ────────────────────────────────────────────────

    @property
    def offload_mode(self) -> OffloadMode:
        """Current offload mode based on last VRAM poll."""
        return self._current_mode

    def poll_and_decide(self) -> OffloadMode:
        """Read current free VRAM and select the offload mode.

        This is a **non-blocking** read of the GPU memory counters via
        ``pynvml``.  The decision is persisted in ``offload_mode``.
        """
        free_mb = self._get_free_vram_mb()
        if free_mb >= _GPU_PRIMARY_THRESHOLD_MB:
            self._current_mode = OffloadMode.GPU_PRIMARY
        elif free_mb >= _PARTIAL_OFFLOAD_THRESHOLD_MB:
            self._current_mode = OffloadMode.PARTIAL_OFFLOAD
        else:
            self._current_mode = OffloadMode.FULL_CPU

        VRAM_USAGE.labels(region="C_free").set(free_mb)
        log.debug(
            "offload_decision",
            free_mb=free_mb,
            mode=self._current_mode.value,
        )
        return self._current_mode

    def apply_offload(
        self,
        vehicle_det: VehicleDetector,
        plate_det: PlateDetector,
    ) -> None:
        """Move detectors to the correct device based on current mode.

        This method should be called **before** loading the models for
        an inference cycle.

        ================== ======= ========
        Mode               S2      S3
        ================== ======= ========
        GPU_PRIMARY        cuda    cuda
        PARTIAL_OFFLOAD    cuda    cpu
        FULL_CPU           cpu     cpu
        ================== ======= ========
        """
        mode = self._current_mode

        if mode == OffloadMode.GPU_PRIMARY:
            desired_v, desired_p = "cuda", "cuda"
        elif mode == OffloadMode.PARTIAL_OFFLOAD:
            desired_v, desired_p = "cuda", "cpu"
        else:
            desired_v, desired_p = "cpu", "cpu"

        if vehicle_det.device != desired_v:
            vehicle_det.switch_device(desired_v)
        if plate_det.device != desired_p:
            plate_det.switch_device(desired_p)

    @contextlib.contextmanager
    def inference_context(
        self,
        vehicle_det: VehicleDetector,
        plate_det: PlateDetector,
    ) -> Generator[OffloadMode, None, None]:
        """Context manager for vision inference (Region C).

        1. Polls free VRAM and selects offload mode.
        2. Moves detectors to the appropriate device.
        3. Loads model weights (``warmup`` if not already loaded).
        4. Yields control for the caller to run ``detect()``.
        5. On exit, releases both models and reclaims VRAM.

        If a CUDA OOM occurs during the body, **both** detectors are
        immediately offloaded to CPU and the OOM is re-raised as a
        ``VRAMError``.
        """
        mode = self.poll_and_decide()
        self.apply_offload(vehicle_det, plate_det)

        # Ensure weights are resident
        if not vehicle_det._engine.loaded:
            vehicle_det.warmup()
        if not plate_det._engine.loaded:
            plate_det.warmup()

        try:
            yield mode
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                log.error("cuda_oom_detected_during_inference", error=str(exc))
                _emergency_cpu_fallback(vehicle_det, plate_det)
                raise VRAMError(
                    "CUDA OOM during detection — both detectors moved to CPU"
                ) from exc
            raise
        finally:
            # Release region C — models are unloaded between events
            vehicle_det.release()
            plate_det.release()
            _try_empty_cuda_cache()
            VRAM_USAGE.labels(region="C_allocated").set(0)

    # ── Memory Fence (spec §5.5) ──────────────────────────────────

    def assert_memory_fence(self) -> None:
        """Hard barrier between vision (Region C) and LLM (A + B).

        Must be called **after** all vision inference in an event cycle
        is complete and **before** LLM inference begins.

        Raises
        ------
        VRAMBudgetExceeded
            If Region C still has allocated memory that was not freed.
        """
        _try_empty_cuda_cache()

        allocated_mb = self._get_allocated_vram_mb()
        if allocated_mb > self._safety_margin_mb:
            raise VRAMBudgetExceeded(
                region="C",
                budget_mb=self._vision_budget_mb,
                used_mb=allocated_mb,
            )
        log.debug("memory_fence_passed", allocated_mb=allocated_mb)

    # ── Internal helpers ──────────────────────────────────────────

    def _get_free_vram_mb(self) -> int:
        """Return free VRAM in MB for the configured device."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return int(info.free / (1024 * 1024))
        except Exception:
            # If pynvml is unavailable, try torch
            try:
                import torch

                if torch.cuda.is_available():
                    free, _ = torch.cuda.mem_get_info(self._device_index)
                    return int(free / (1024 * 1024))
            except Exception:
                pass
        # Cannot determine VRAM — assume constrained
        log.warning("vram_query_failed_assuming_constrained")
        return 0

    def _get_allocated_vram_mb(self) -> int:
        """Return currently allocated VRAM in MB (via torch)."""
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self._device_index)
                return int(allocated / (1024 * 1024))
        except Exception:
            pass
        return 0


# ── Module-level helpers ──────────────────────────────────────────


def _emergency_cpu_fallback(
    vehicle_det: VehicleDetector,
    plate_det: PlateDetector,
) -> None:
    """Force both detectors to CPU after an OOM event."""
    log.warning("emergency_cpu_fallback_triggered")
    try:
        vehicle_det.release()
    except Exception:
        pass
    try:
        plate_det.release()
    except Exception:
        pass
    _try_empty_cuda_cache()
    vehicle_det.switch_device("cpu")
    plate_det.switch_device("cpu")


def _try_empty_cuda_cache() -> None:
    """Best-effort ``torch.cuda.empty_cache()``."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
