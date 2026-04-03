"""Performance Feedback Loop — closed-loop telemetry for pipeline tuning.

Collects per-component, per-pipeline, and per-camera execution telemetry
and builds statistical profiles.  This data feeds back into:

  * **Adaptation Engine** — triggering runtime pipeline changes.
  * **Component Quality Scorer** — updating reliability/speed scores.
  * **Pipeline Composer** — informing stage selection for new pipelines.

Design notes:
  * All state is in-memory with configurable retention (no DB dependency).
  * Thread-safe via asyncio Lock (single-writer, multi-reader).
  * Exposes telemetry via a status dict for Prometheus / API consumption.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uni_vision.manager.schemas import (
        FrameContext,
        PipelineExecutionResult,
        SceneType,
    )

log = logging.getLogger(__name__)

_WINDOW = 100  # default sliding window size


@dataclass
class _EWMATracker:
    """Exponentially-weighted moving average for a single metric."""

    alpha: float = 0.1
    value: float = 0.0
    count: int = 0

    def update(self, sample: float) -> None:
        if self.count == 0:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1 - self.alpha) * self.value
        self.count += 1


@dataclass
class ComponentProfile:
    """Statistical performance profile for a single component."""

    component_id: str

    # Sliding window raw values
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))
    successes: deque[bool] = field(default_factory=lambda: deque(maxlen=_WINDOW))
    confidences: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))

    # EWMA trackers
    latency_ewma: _EWMATracker = field(default_factory=_EWMATracker)
    confidence_ewma: _EWMATracker = field(default_factory=_EWMATracker)

    # Counters
    total_runs: int = 0
    total_failures: int = 0
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = 0.0

    def record(self, latency_ms: float, success: bool, confidence: float | None = None) -> None:
        self.latencies.append(latency_ms)
        self.successes.append(success)
        self.latency_ewma.update(latency_ms)
        self.total_runs += 1
        self.last_seen = time.monotonic()
        if not success:
            self.total_failures += 1
        if confidence is not None:
            self.confidences.append(confidence)
            self.confidence_ewma.update(confidence)

    # ── Derived metrics ───────────────────────────────────────

    @property
    def reliability(self) -> float:
        """Fraction of successful runs (0.0–1.0)."""
        if self.total_runs == 0:
            return 1.0
        return 1.0 - (self.total_failures / self.total_runs)

    @property
    def recent_reliability(self) -> float:
        """Reliability over the sliding window."""
        if not self.successes:
            return 1.0
        return sum(1 for s in self.successes if s) / len(self.successes)

    @property
    def median_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = int(0.99 * len(s))
        return s[min(idx, len(s) - 1)]

    @property
    def latency_std(self) -> float:
        if len(self.latencies) < 2:
            return 0.0
        mean = sum(self.latencies) / len(self.latencies)
        variance = sum((x - mean) ** 2 for x in self.latencies) / (len(self.latencies) - 1)
        return math.sqrt(variance)

    def summary(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "total_runs": self.total_runs,
            "reliability": round(self.reliability, 4),
            "recent_reliability": round(self.recent_reliability, 4),
            "latency_ewma_ms": round(self.latency_ewma.value, 1),
            "median_latency_ms": round(self.median_latency, 1),
            "p99_latency_ms": round(self.p99_latency, 1),
            "latency_std_ms": round(self.latency_std, 1),
            "confidence_ewma": round(self.confidence_ewma.value, 3) if self.confidence_ewma.count else None,
            "uptime_s": round(time.monotonic() - self.first_seen, 1),
        }


@dataclass
class PipelineProfile:
    """Aggregate telemetry for a pipeline blueprint (by hash)."""

    blueprint_hash: str
    total_runs: int = 0
    total_latencies: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))
    success_count: int = 0
    scene_type_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, total_ms: float, success: bool, scene: SceneType | None = None) -> None:
        self.total_runs += 1
        self.total_latencies.append(total_ms)
        if success:
            self.success_count += 1
        if scene:
            self.scene_type_counts[scene.value] += 1

    @property
    def avg_latency(self) -> float:
        if not self.total_latencies:
            return 0.0
        return sum(self.total_latencies) / len(self.total_latencies)

    @property
    def success_rate(self) -> float:
        if self.total_runs == 0:
            return 1.0
        return self.success_count / self.total_runs


class FeedbackLoop:
    """Closed-loop performance feedback collector.

    Thread-safe, lock-free reads for hot path (profiles are append-only deques).
    """

    def __init__(self, *, retention_window: int = _WINDOW) -> None:
        self._retention = retention_window
        self._components: dict[str, ComponentProfile] = {}
        self._pipelines: dict[str, PipelineProfile] = {}
        self._total_frames: int = 0
        self._lock = asyncio.Lock()

    async def record_result(
        self,
        result: PipelineExecutionResult,
        context: FrameContext,
        blueprint_hash: str | None = None,
    ) -> None:
        """Record a full pipeline execution result."""
        async with self._lock:
            self._total_frames += 1

            # Per-component recording
            for sr in result.stage_results:
                cid = sr.component_id
                if cid not in self._components:
                    self._components[cid] = ComponentProfile(component_id=cid)
                conf = None
                if isinstance(sr.output, dict) and "confidence" in sr.output:
                    conf = float(sr.output["confidence"])
                self._components[cid].record(sr.elapsed_ms, sr.success, conf)

            # Per-pipeline recording
            if blueprint_hash:
                if blueprint_hash not in self._pipelines:
                    self._pipelines[blueprint_hash] = PipelineProfile(blueprint_hash=blueprint_hash)
                all_success = all(sr.success for sr in result.stage_results)
                self._pipelines[blueprint_hash].record(
                    result.total_elapsed_ms,
                    all_success,
                    context.scene_type,
                )

    def get_component_profile(self, component_id: str) -> ComponentProfile | None:
        return self._components.get(component_id)

    def get_pipeline_profile(self, blueprint_hash: str) -> PipelineProfile | None:
        return self._pipelines.get(blueprint_hash)

    def rank_components_by_reliability(self) -> list[tuple[str, float]]:
        """Return components ranked by reliability (worst first)."""
        items = [(cid, prof.recent_reliability) for cid, prof in self._components.items() if prof.total_runs >= 5]
        items.sort(key=lambda x: x[1])
        return items

    def rank_components_by_latency(self) -> list[tuple[str, float]]:
        """Return components ranked by avg latency (slowest first)."""
        items = [(cid, prof.latency_ewma.value) for cid, prof in self._components.items() if prof.total_runs >= 5]
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def get_degraded_components(
        self,
        min_reliability: float = 0.7,
        max_latency_ms: float = 500.0,
    ) -> list[str]:
        """Return component IDs that are underperforming."""
        degraded = []
        for cid, prof in self._components.items():
            if prof.total_runs < 5:
                continue
            if prof.recent_reliability < min_reliability or prof.latency_ewma.value > max_latency_ms:
                degraded.append(cid)
        return degraded

    def status(self) -> dict[str, Any]:
        """Full feedback loop status for monitoring."""
        return {
            "total_frames_processed": self._total_frames,
            "tracked_components": len(self._components),
            "tracked_pipelines": len(self._pipelines),
            "component_profiles": {cid: prof.summary() for cid, prof in self._components.items()},
            "pipeline_profiles": {
                h: {
                    "total_runs": p.total_runs,
                    "avg_latency_ms": round(p.avg_latency, 1),
                    "success_rate": round(p.success_rate, 4),
                }
                for h, p in self._pipelines.items()
            },
        }
