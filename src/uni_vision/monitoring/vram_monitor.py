"""Real-time GPU VRAM & PCIe bus telemetry monitor — spec §7.3.

Runs as a background ``asyncio.Task`` polling NVIDIA GPU state via
``pynvml`` (NVIDIA Management Library) at a configurable interval
(default 500 ms).  Publishes telemetry to Prometheus gauges and emits
structured logs on threshold breaches.

Key responsibilities:
  * Track per-region VRAM allocation against the static budget.
  * Track PCIe TX/RX throughput (bus saturation).
  * Trigger CPU-fallback decisions per spec §7.4 offloading rules.
  * Emit ``GPUTelemetry`` DTOs consumable by the health service.

Usage::

    monitor = VRAMMonitor(hardware_cfg, vram_budgets)
    asyncio.create_task(monitor.run())     # fire-and-forget background
    ...
    snapshot = monitor.latest_telemetry()  # non-blocking read
    await monitor.shutdown()               # graceful stop
"""

from __future__ import annotations

import asyncio
import time

import structlog

from uni_vision.contracts.dtos import (
    GPUTelemetry,
    OffloadMode,
    VRAMRegionSnapshot,
)
from uni_vision.monitoring.metrics import VRAM_USAGE

log = structlog.get_logger()

# pynvml is optional — the monitor degrades gracefully when the GPU
# is unavailable (e.g. CI, CPU-only hosts).
try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYNVML_AVAILABLE = False


class VRAMMonitor:
    """Background daemon that polls GPU VRAM and PCIe metrics.

    Parameters
    ----------
    device_index:
        CUDA device ordinal (default 0).
    vram_ceiling_mb:
        Hard VRAM ceiling in MB (8192 for RTX 4070).
    safety_margin_mb:
        Minimum free VRAM before CPU fallback is triggered (256 MB).
    poll_interval_ms:
        Polling interval in milliseconds (500 ms default).
    region_budgets:
        Dict mapping region name → budget in MB.  Used for per-region
        utilisation reporting.  If ``None``, a default 4-region budget
        matching spec §2.1 is used.
    """

    def __init__(
        self,
        *,
        device_index: int = 0,
        vram_ceiling_mb: int = 8192,
        safety_margin_mb: int = 256,
        poll_interval_ms: int = 500,
        region_budgets: dict[str, int] | None = None,
    ) -> None:
        self._device_index = device_index
        self._vram_ceiling_mb = vram_ceiling_mb
        self._safety_margin_mb = safety_margin_mb
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._region_budgets: dict[str, int] = region_budgets or {
            "llm_weights": 5632,
            "kv_cache": 1024,
            "vision_workspace": 1024,
            "system_overhead": 512,
        }

        self._running = False
        self._handle: object | None = None  # pynvml device handle
        self._latest: GPUTelemetry | None = None
        self._offload_mode: OffloadMode = OffloadMode.GPU_PRIMARY

    # ── Lifecycle ─────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the polling loop.  Blocks until ``shutdown()`` is called."""
        if not _PYNVML_AVAILABLE:
            log.warning("pynvml_unavailable", msg="GPU monitoring disabled — pynvml not installed")
            return

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        self._running = True

        log.info(
            "vram_monitor_started",
            device_index=self._device_index,
            poll_interval_ms=int(self._poll_interval_s * 1000),
        )

        try:
            while self._running:
                self._poll()
                await asyncio.sleep(self._poll_interval_s)
        finally:
            pynvml.nvmlShutdown()
            log.info("vram_monitor_stopped")

    async def shutdown(self) -> None:
        """Signal the polling loop to exit."""
        self._running = False

    # ── Query interface ───────────────────────────────────────────

    def latest_telemetry(self) -> GPUTelemetry | None:
        """Return the most recently captured ``GPUTelemetry`` snapshot."""
        return self._latest

    @property
    def offload_mode(self) -> OffloadMode:
        """Current offloading decision based on the latest poll."""
        return self._offload_mode

    # ── Internal polling ──────────────────────────────────────────

    def _poll(self) -> None:
        """Execute a single telemetry collection cycle."""
        assert self._handle is not None

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        util_rates = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        temperature = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)

        # PCIe throughput (KB/s) — NVML counter type constants
        pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(self._handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
        pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(self._handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)

        vram_total_mb = mem_info.total / (1024 * 1024)
        vram_used_mb = mem_info.used / (1024 * 1024)
        vram_free_mb = mem_info.free / (1024 * 1024)
        vram_pct = (vram_used_mb / vram_total_mb) * 100 if vram_total_mb else 0

        # Device name (cached on first call by pynvml internally)
        device_name = pynvml.nvmlDeviceGetName(self._handle)
        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8")

        # Build per-region snapshots.  Since pynvml reports aggregate
        # VRAM, we distribute proportionally based on the configured
        # budgets.  In a production build with TensorRT, per-allocation
        # tracking would replace this heuristic.
        total_budget = sum(self._region_budgets.values())
        regions = []
        for name, budget in self._region_budgets.items():
            # Proportional estimate
            proportion = budget / total_budget if total_budget else 0
            region_used = vram_used_mb * proportion
            region_pct = (region_used / budget) * 100 if budget else 0
            regions.append(
                VRAMRegionSnapshot(
                    region_name=name,
                    budget_mb=float(budget),
                    used_mb=round(region_used, 1),
                    utilisation_pct=round(region_pct, 1),
                )
            )

        telemetry = GPUTelemetry(
            timestamp_utc=time.time(),
            device_index=self._device_index,
            device_name=device_name,
            vram_total_mb=round(vram_total_mb, 1),
            vram_used_mb=round(vram_used_mb, 1),
            vram_free_mb=round(vram_free_mb, 1),
            vram_utilisation_pct=round(vram_pct, 1),
            gpu_utilisation_pct=float(util_rates.gpu),
            temperature_c=temperature,
            pcie_tx_kbps=pcie_tx,
            pcie_rx_kbps=pcie_rx,
            regions=regions,
        )

        self._latest = telemetry

        # Publish Prometheus gauges
        for region in regions:
            VRAM_USAGE.labels(region=region.region_name).set(
                region.used_mb * 1024 * 1024  # gauge is in bytes
            )

        # Offloading decision — spec §7.4
        self._evaluate_offload(vram_free_mb)

        # Alert if any region exceeds 90% of its budget
        for region in regions:
            if region.utilisation_pct > 90:
                log.warning(
                    "vram_region_high",
                    region=region.region_name,
                    used_mb=region.used_mb,
                    budget_mb=region.budget_mb,
                    utilisation_pct=region.utilisation_pct,
                )

    def _evaluate_offload(self, free_mb: float) -> None:
        """Apply the CPU↔GPU offloading rules from spec §7.4.

        | Free VRAM   | Decision                              |
        |-------------|---------------------------------------|
        | ≥ 1024 MB   | Normal GPU path (GPU_PRIMARY)         |
        | < 1024 MB   | S2 on GPU, S3 on CPU (PARTIAL)        |
        | < 512 MB    | S2 + S3 on CPU (FULL_CPU)             |
        """
        previous = self._offload_mode

        if free_mb >= 1024:
            self._offload_mode = OffloadMode.GPU_PRIMARY
        elif free_mb >= 512:
            self._offload_mode = OffloadMode.PARTIAL_OFFLOAD
        else:
            self._offload_mode = OffloadMode.FULL_CPU

        if self._offload_mode != previous:
            log.warning(
                "offload_mode_changed",
                previous=previous.value,
                current=self._offload_mode.value,
                free_vram_mb=round(free_mb, 1),
            )
