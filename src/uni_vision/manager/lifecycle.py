"""Lifecycle Manager — VRAM-aware component load/unload/swap.

On an RTX 4070 with 8 GiB, every megabyte counts.  The Lifecycle
Manager decides:
  * When to load a component to GPU
  * When to offload a component to CPU (suspend)
  * When to unload entirely (free memory)
  * LRU eviction when VRAM is tight

The Ollama server (Gemma 4 E2B Q4_K_M) takes ~5.0 GiB.  That
leaves ~3.0 GiB for all other components.  The lifecycle manager
keeps a running tally and enforces the budget.

Integrates with ``GPUProfiler`` (when available) for real VRAM
measurement instead of relying solely on estimates.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

import structlog

from uni_vision.components.base import ComponentState, CVComponent

if TYPE_CHECKING:
    from uni_vision.manager.component_registry import ComponentRegistry
    from uni_vision.manager.gpu_profiler import GPUProfiler

log = structlog.get_logger(__name__)

# Default timeout for a single component load operation (seconds)
_LOAD_TIMEOUT_SECONDS = 120


class LifecycleManager:
    """VRAM-aware component lifecycle controller.

    Parameters
    ----------
    registry:
        Component registry.
    vram_total_mb:
        Total GPU VRAM available.
    vram_reserved_mb:
        VRAM reserved for non-component use (Ollama, CUDA overhead).
    device:
        Default device for loading (``"cuda:0"`` or ``"cpu"``).
    gpu_profiler:
        Optional GPUProfiler for real VRAM measurement.
    load_timeout:
        Per-component load timeout in seconds.
    """

    def __init__(
        self,
        registry: ComponentRegistry,
        *,
        vram_total_mb: int = 8192,
        vram_reserved_mb: int = 5800,
        device: str = "cuda:0",
        gpu_profiler: GPUProfiler | None = None,
        load_timeout: int = _LOAD_TIMEOUT_SECONDS,
    ) -> None:
        self._registry = registry
        self._vram_total = vram_total_mb
        self._vram_reserved = vram_reserved_mb
        self._device = device
        self._gpu_profiler = gpu_profiler
        self._load_timeout = load_timeout

        # LRU tracking: component_id → last_used_timestamp
        self._lru: OrderedDict[str, float] = OrderedDict()

        # Lock for serialising lifecycle operations
        self._lock = asyncio.Lock()

    @property
    def vram_budget_mb(self) -> int:
        """VRAM available for components (total - reserved)."""
        return self._vram_total - self._vram_reserved

    @property
    def vram_used_mb(self) -> int:
        """VRAM currently consumed by loaded components."""
        return self._registry.get_loaded_vram_mb()

    @property
    def vram_free_mb(self) -> int:
        """VRAM available for new components.

        Uses real GPU measurement if profiler is available;
        falls back to estimate-based accounting.
        """
        if self._gpu_profiler and self._gpu_profiler.available:
            real = self._gpu_profiler.get_current_usage()
            if real is not None:
                # real["free_mb"] is total GPU free; subtract reservation
                return max(0, int(real["free_mb"]) - self._vram_reserved)
        return max(0, self.vram_budget_mb - self.vram_used_mb)

    # ── Load ──────────────────────────────────────────────────────

    async def load_component(
        self,
        component_id: str,
        *,
        device: str | None = None,
    ) -> bool:
        """Load a component into VRAM with timeout and health check.

        If there isn't enough VRAM, evicts LRU components first.
        Returns True on success.
        """
        async with self._lock:
            comp = self._registry.get(component_id)
            if comp is None:
                log.error("load_unknown_component component_id=%s", component_id)
                return False

            if comp.state == ComponentState.READY:
                self._touch(component_id)
                return True

            needed = comp.metadata.resource_estimate.vram_mb
            target_device = device or self._device

            # If loading to GPU and not enough VRAM, evict
            if "cuda" in target_device and needed > self.vram_free_mb:
                freed = await self._evict_for_vram(needed)
                if freed < needed and needed > self.vram_free_mb:
                    log.warning(
                        "insufficient_vram needed=%d free=%d component=%s",
                        needed,
                        self.vram_free_mb,
                        component_id,
                    )
                    if comp.metadata.resource_estimate.supports_cpu:
                        target_device = "cpu"
                        log.info("fallback_to_cpu component=%s", component_id)
                    else:
                        return False

            try:
                # Load with timeout and optional GPU profiling
                await self._load_with_profiling(comp, target_device)
                self._touch(component_id)
                log.info(
                    "component_loaded id=%s device=%s vram_used=%d vram_free=%d",
                    component_id,
                    target_device,
                    self.vram_used_mb,
                    self.vram_free_mb,
                )
                return True

            except asyncio.TimeoutError:
                log.error("component_load_timeout id=%s timeout=%ds", component_id, self._load_timeout)
                comp._set_state(ComponentState.FAILED)
                comp._load_error = f"Load timed out after {self._load_timeout}s"
                return False
            except Exception as exc:
                log.error("component_load_failed id=%s error=%s", component_id, exc)
                return False

    async def _load_with_profiling(
        self,
        comp: CVComponent,
        device: str,
    ) -> None:
        """Load a component with timeout, GPU profiling, and health check."""
        cid = comp.metadata.component_id
        estimated = comp.metadata.resource_estimate.vram_mb

        if self._gpu_profiler and self._gpu_profiler.available and "cuda" in device:
            # Use synchronous profiler context manager around async load
            loop = asyncio.get_event_loop()

            def _profiled_load() -> None:
                with self._gpu_profiler.measure_load(cid, estimated):
                    # We need to run the async load in this sync context
                    import asyncio as _aio

                    _loop = _aio.new_event_loop()
                    try:
                        _loop.run_until_complete(comp.load(device=device))
                    finally:
                        _loop.close()

            await asyncio.wait_for(
                loop.run_in_executor(None, _profiled_load),
                timeout=self._load_timeout,
            )
        else:
            await asyncio.wait_for(
                comp.load(device=device),
                timeout=self._load_timeout,
            )

        # Health check — if the component supports it
        if hasattr(comp, "_health_check"):
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, comp._health_check)
            except Exception:
                log.debug("post_load_health_check skipped: %s", cid)

    # ── Unload ────────────────────────────────────────────────────

    async def unload_component(self, component_id: str) -> bool:
        """Unload a component and free its resources."""
        async with self._lock:
            comp = self._registry.get(component_id)
            if comp is None:
                return False

            if comp.state not in (ComponentState.READY, ComponentState.SUSPENDED):
                return True  # Already unloaded

            try:
                await comp.unload()
                self._lru.pop(component_id, None)
                log.info(
                    "component_unloaded",
                    component_id=component_id,
                    vram_free=self.vram_free_mb,
                )
                return True

            except Exception as exc:
                log.error("component_unload_failed", component_id=component_id, error=str(exc))
                return False

    # ── Swap (atomic unload + load) ───────────────────────────────

    async def swap_component(
        self,
        old_id: str,
        new_id: str,
    ) -> bool:
        """Unload old component and load new one.

        This is atomic — if the new load fails, the old component
        is restored.
        """
        async with self._lock:
            old = self._registry.get(old_id)
            new = self._registry.get(new_id)

            if new is None:
                log.error("swap_new_not_found", old_id=old_id, new_id=new_id)
                return False

            # Unload old
            if old and old.state == ComponentState.READY:
                try:
                    await old.unload()
                    self._lru.pop(old_id, None)
                except Exception:
                    log.warning("swap_old_unload_failed", old_id=old_id)

            # Load new
            try:
                target = self._device
                needed = new.metadata.resource_estimate.vram_mb
                if "cuda" in target and needed > self.vram_free_mb:
                    if new.metadata.resource_estimate.supports_cpu:
                        target = "cpu"
                    else:
                        # Restore old
                        if old:
                            await old.load(device=self._device)
                        return False

                await new.load(device=target)
                self._touch(new_id)
                log.info("component_swapped", old=old_id, new=new_id)
                return True

            except Exception as exc:
                log.error("swap_new_load_failed", new_id=new_id, error=str(exc))
                # Try restoring old
                if old:
                    try:
                        await old.load(device=self._device)
                        self._touch(old_id)
                    except Exception:
                        pass
                return False

    # ── Batch operations ──────────────────────────────────────────

    async def ensure_loaded(
        self,
        component_ids: list[str],
    ) -> dict[str, bool]:
        """Ensure a list of components are loaded.

        Returns a mapping of component_id → success.
        """
        results: dict[str, bool] = {}
        for cid in component_ids:
            results[cid] = await self.load_component(cid)
        return results

    async def unload_all(self) -> int:
        """Unload all loaded components.  Returns count unloaded."""
        loaded = self._registry.get_loaded()
        count = 0
        for comp in loaded:
            if await self.unload_component(comp.metadata.component_id):
                count += 1
        return count

    # ── LRU eviction ──────────────────────────────────────────────

    async def _evict_for_vram(self, needed_mb: int) -> int:
        """Evict LRU components until we have enough VRAM.

        Returns total VRAM freed.
        """
        freed = 0
        # Iterate LRU order (oldest first)
        eviction_candidates = list(self._lru.keys())

        for cid in eviction_candidates:
            if self.vram_free_mb >= needed_mb:
                break

            comp = self._registry.get(cid)
            if comp is None or comp.state != ComponentState.READY:
                continue

            vram = comp.metadata.resource_estimate.vram_mb
            try:
                await comp.unload()
                self._lru.pop(cid, None)
                freed += vram
                log.info("lru_evicted", component_id=cid, freed_mb=vram)
            except Exception:
                log.warning("lru_eviction_failed", component_id=cid)

        return freed

    def _touch(self, component_id: str) -> None:
        """Mark component as recently used (moves to end of LRU)."""
        self._lru.pop(component_id, None)
        self._lru[component_id] = time.monotonic()

    # ── Status ────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return VRAM status for LLM prompts."""
        s = {
            "vram_total_mb": self._vram_total,
            "vram_reserved_mb": self._vram_reserved,
            "vram_budget_mb": self.vram_budget_mb,
            "vram_used_mb": self.vram_used_mb,
            "vram_free_mb": self.vram_free_mb,
            "loaded_count": len(self._registry.get_loaded()),
            "lru_order": list(self._lru.keys()),
        }
        if self._gpu_profiler and self._gpu_profiler.available:
            real = self._gpu_profiler.get_current_usage()
            if real is not None:
                s["gpu_real_used_mb"] = round(real.get("used_mb", 0), 1)
                s["gpu_real_free_mb"] = round(real.get("free_mb", 0), 1)
        return s
