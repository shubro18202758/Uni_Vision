"""Adaptation Engine — real-time pipeline adaptation based on feedback.

The Adaptation Engine is the heart of the self-assembling pipeline.
It continuously monitors inference results and environmental changes
to dynamically adjust the active pipeline.  Key capabilities:

  * **Scene drift detection** — spots when the camera feed shifts from
    one scene type to another (e.g., a PTZ camera panning from a
    parking lot to a highway) and triggers pipeline recomposition.
  * **Quality-aware fallback** — when a component's output quality
    degrades (confidence drops, latency spikes) it triggers a
    fallback or replacement.
  * **VRAM pressure adaptation** — proactively downgrades components
    to lighter variants before an OOM crash.
  * **Temporal coherence** — tracks cross-frame consistency to avoid
    thrashing between pipeline configurations.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

from uni_vision.manager.schemas import (
    FrameContext,
    PipelineExecutionResult,
    SceneType,
    StageResult,
    TaskPriority,
)

if TYPE_CHECKING:
    from uni_vision.components.base import ComponentCapability

log = structlog.get_logger(__name__)


# ── Adaptation signals ────────────────────────────────────────────


class AdaptationSignal(str, Enum):
    """Signals that trigger pipeline adaptation."""

    SCENE_DRIFT = "scene_drift"
    QUALITY_DROP = "quality_drop"
    LATENCY_SPIKE = "latency_spike"
    VRAM_PRESSURE = "vram_pressure"
    COMPONENT_FAILURE = "component_failure"
    NEW_CAPABILITY_NEEDED = "new_capability_needed"
    CONFIDENCE_DEGRADATION = "confidence_degradation"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"


@dataclass
class AdaptationEvent:
    """A concrete adaptation signal with context."""

    signal: AdaptationSignal
    severity: TaskPriority
    source_component: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class AdaptationAction:
    """A concrete action the adaptation engine wants to perform."""

    action_type: str  # "swap_component" | "add_stage" | "remove_stage" | "downgrade" | "recompose"
    target_component: str | None = None
    replacement_component: str | None = None
    capability: ComponentCapability | None = None
    priority: TaskPriority = TaskPriority.NORMAL
    reasoning: str = ""


# ── Performance tracking ──────────────────────────────────────────


@dataclass
class ComponentPerformanceWindow:
    """Sliding-window performance tracker for a single component."""

    component_id: str
    latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    confidences: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    errors: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    last_success: float = 0.0

    @property
    def avg_latency(self) -> float:
        return sum(self.latency_ms) / max(len(self.latency_ms), 1)

    @property
    def avg_confidence(self) -> float:
        return sum(self.confidences) / max(len(self.confidences), 1)

    @property
    def error_rate(self) -> float:
        if not self.errors:
            return 0.0
        return sum(self.errors) / len(self.errors)

    @property
    def p95_latency(self) -> float:
        if not self.latency_ms:
            return 0.0
        sorted_lat = sorted(self.latency_ms)
        idx = int(0.95 * len(sorted_lat))
        return sorted_lat[min(idx, len(sorted_lat) - 1)]


@dataclass
class SceneHistory:
    """Tracks scene transitions over time."""

    scenes: deque[SceneType] = field(default_factory=lambda: deque(maxlen=30))
    timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=30))

    @property
    def current_scene(self) -> SceneType | None:
        return self.scenes[-1] if self.scenes else None

    @property
    def is_stable(self) -> bool:
        """True if the scene has been consistent for the last N frames."""
        if len(self.scenes) < 5:
            return True
        return len(set(list(self.scenes)[-5:])) == 1

    def dominant_scene(self, window: int = 10) -> SceneType | None:
        """Most common scene in the last N entries."""
        recent = list(self.scenes)[-window:]
        if not recent:
            return None
        from collections import Counter

        counts = Counter(recent)
        return counts.most_common(1)[0][0]


# ── Adaptation Engine ─────────────────────────────────────────────


class AdaptationEngine:
    """Real-time pipeline adaptation controller.

    Monitors pipeline execution results and environmental signals
    to proactively adapt the pipeline configuration.

    Parameters
    ----------
    latency_threshold_ms:
        Per-stage latency threshold — spikes beyond this trigger adaptation.
    confidence_threshold:
        Minimum acceptable average confidence before triggering quality alerts.
    error_rate_threshold:
        Component error rate above which a swap is triggered.
    scene_stability_window:
        Number of frames before a scene change is considered stable.
    vram_pressure_threshold:
        Fraction of VRAM budget at which proactive downgrade kicks in (0.0–1.0).
    cooldown_s:
        Minimum seconds between adaptation actions to prevent thrashing.
    """

    def __init__(
        self,
        *,
        latency_threshold_ms: float = 200.0,
        confidence_threshold: float = 0.4,
        error_rate_threshold: float = 0.3,
        scene_stability_window: int = 5,
        vram_pressure_threshold: float = 0.85,
        cooldown_s: float = 5.0,
    ) -> None:
        self._latency_threshold = latency_threshold_ms
        self._confidence_threshold = confidence_threshold
        self._error_rate_threshold = error_rate_threshold
        self._scene_stability_window = scene_stability_window
        self._vram_pressure_threshold = vram_pressure_threshold
        self._cooldown_s = cooldown_s

        # Per-component performance tracking
        self._perf: dict[str, ComponentPerformanceWindow] = {}

        # Per-camera scene history
        self._scene_history: dict[str, SceneHistory] = {}

        # Event log (last N adaptation events)
        self._events: deque[AdaptationEvent] = deque(maxlen=200)

        # Cooldown tracking
        self._last_adaptation: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────

    def ingest_result(
        self,
        result: PipelineExecutionResult,
        context: FrameContext,
    ) -> list[AdaptationAction]:
        """Ingest a pipeline execution result and return adaptation actions.

        This is the main hot-path method called after every frame.
        """
        actions: list[AdaptationAction] = []

        # Update component performance windows
        for sr in result.stage_results:
            self._update_perf(sr)

        # Update scene history
        camera_id = context.camera_id or "unknown"
        if camera_id not in self._scene_history:
            self._scene_history[camera_id] = SceneHistory()
        sh = self._scene_history[camera_id]
        sh.scenes.append(context.scene_type)
        sh.timestamps.append(time.monotonic())

        # Run adaptation checks
        actions.extend(self._check_scene_drift(camera_id))
        actions.extend(self._check_quality_degradation())
        actions.extend(self._check_latency_spikes())
        actions.extend(self._check_error_rates())

        # Filter by cooldown
        actions = self._apply_cooldown(actions)

        if actions:
            log.info(
                "adaptation_actions_generated",
                count=len(actions),
                types=[a.action_type for a in actions],
            )

        return actions

    def check_vram_pressure(
        self,
        vram_used_mb: int,
        vram_budget_mb: int,
    ) -> list[AdaptationAction]:
        """Check for VRAM pressure and suggest proactive downgrades."""
        if vram_budget_mb <= 0:
            return []

        utilisation = vram_used_mb / vram_budget_mb
        if utilisation < self._vram_pressure_threshold:
            return []

        event = AdaptationEvent(
            signal=AdaptationSignal.VRAM_PRESSURE,
            severity=TaskPriority.HIGH,
            details={
                "vram_used_mb": vram_used_mb,
                "vram_budget_mb": vram_budget_mb,
                "utilisation": utilisation,
            },
        )
        self._events.append(event)

        # Find the highest-VRAM component with acceptable alternatives
        heaviest = self._find_heaviest_components()
        actions = []
        for comp_id, _avg_lat in heaviest[:2]:
            actions.append(
                AdaptationAction(
                    action_type="downgrade",
                    target_component=comp_id,
                    priority=TaskPriority.HIGH,
                    reasoning=f"VRAM at {utilisation:.0%} — proactive downgrade of {comp_id}",
                )
            )

        return self._apply_cooldown(actions)

    def get_component_health(self, component_id: str) -> dict[str, Any]:
        """Return health metrics for a specific component."""
        perf = self._perf.get(component_id)
        if perf is None:
            return {"status": "unknown"}
        return {
            "status": "healthy" if perf.error_rate < self._error_rate_threshold else "degraded",
            "avg_latency_ms": round(perf.avg_latency, 1),
            "p95_latency_ms": round(perf.p95_latency, 1),
            "avg_confidence": round(perf.avg_confidence, 3),
            "error_rate": round(perf.error_rate, 3),
            "samples": len(perf.latency_ms),
        }

    def get_scene_status(self, camera_id: str) -> dict[str, Any]:
        """Return scene tracking status for a camera."""
        sh = self._scene_history.get(camera_id)
        if sh is None:
            return {"status": "no_data"}
        return {
            "current_scene": sh.current_scene.value if sh.current_scene else "unknown",
            "dominant_scene": sh.dominant_scene().value if sh.dominant_scene() else "unknown",
            "is_stable": sh.is_stable,
            "history_len": len(sh.scenes),
        }

    def reset_component(self, component_id: str) -> None:
        """Reset performance tracking for a component (e.g. after swap)."""
        self._perf.pop(component_id, None)
        self._last_adaptation.pop(component_id, None)

    # ── Internal checks ───────────────────────────────────────────

    def _update_perf(self, sr: StageResult) -> None:
        """Update the performance window for a component."""
        cid = sr.component_id
        if cid not in self._perf:
            self._perf[cid] = ComponentPerformanceWindow(component_id=cid)
        perf = self._perf[cid]
        perf.latency_ms.append(sr.elapsed_ms)
        perf.errors.append(0.0 if sr.success else 1.0)
        if sr.success:
            perf.last_success = time.monotonic()
            # Extract confidence if available
            if isinstance(sr.output, dict) and "confidence" in sr.output:
                perf.confidences.append(float(sr.output["confidence"]))

    def _check_scene_drift(self, camera_id: str) -> list[AdaptationAction]:
        """Detect if the scene has drifted and a recomposition is needed."""
        sh = self._scene_history.get(camera_id)
        if sh is None or len(sh.scenes) < self._scene_stability_window:
            return []

        list(sh.scenes)[-self._scene_stability_window :]
        dominant = sh.dominant_scene(self._scene_stability_window)

        # Check if there was a change
        older = list(sh.scenes)[-(self._scene_stability_window * 2) : -self._scene_stability_window]
        if not older:
            return []

        old_dominant = max(set(older), key=older.count) if older else None
        if dominant == old_dominant:
            return []

        # Scene has changed
        event = AdaptationEvent(
            signal=AdaptationSignal.SCENE_DRIFT,
            severity=TaskPriority.HIGH,
            details={
                "camera_id": camera_id,
                "old_scene": old_dominant.value if old_dominant else "unknown",
                "new_scene": dominant.value if dominant else "unknown",
            },
        )
        self._events.append(event)

        return [
            AdaptationAction(
                action_type="recompose",
                priority=TaskPriority.HIGH,
                reasoning=f"Scene drift: {old_dominant.value if old_dominant else '?'} → {dominant.value if dominant else '?'}",
            )
        ]

    def _check_quality_degradation(self) -> list[AdaptationAction]:
        """Check for components with degrading output quality."""
        actions = []
        for cid, perf in self._perf.items():
            if len(perf.confidences) < 10:
                continue
            if perf.avg_confidence < self._confidence_threshold:
                event = AdaptationEvent(
                    signal=AdaptationSignal.CONFIDENCE_DEGRADATION,
                    severity=TaskPriority.HIGH,
                    source_component=cid,
                    details={"avg_confidence": perf.avg_confidence},
                )
                self._events.append(event)
                actions.append(
                    AdaptationAction(
                        action_type="swap_component",
                        target_component=cid,
                        priority=TaskPriority.HIGH,
                        reasoning=f"Confidence degraded to {perf.avg_confidence:.2f} (threshold {self._confidence_threshold})",
                    )
                )
        return actions

    def _check_latency_spikes(self) -> list[AdaptationAction]:
        """Check for components with sustained latency spikes."""
        actions = []
        for cid, perf in self._perf.items():
            if len(perf.latency_ms) < 5:
                continue
            if perf.p95_latency > self._latency_threshold:
                event = AdaptationEvent(
                    signal=AdaptationSignal.LATENCY_SPIKE,
                    severity=TaskPriority.NORMAL,
                    source_component=cid,
                    details={"p95_latency_ms": perf.p95_latency},
                )
                self._events.append(event)
                actions.append(
                    AdaptationAction(
                        action_type="downgrade",
                        target_component=cid,
                        priority=TaskPriority.NORMAL,
                        reasoning=f"p95 latency {perf.p95_latency:.1f} ms > threshold {self._latency_threshold}",
                    )
                )
        return actions

    def _check_error_rates(self) -> list[AdaptationAction]:
        """Check for components with high error rates."""
        actions = []
        for cid, perf in self._perf.items():
            if len(perf.errors) < 5:
                continue
            if perf.error_rate > self._error_rate_threshold:
                event = AdaptationEvent(
                    signal=AdaptationSignal.COMPONENT_FAILURE,
                    severity=TaskPriority.CRITICAL,
                    source_component=cid,
                    details={"error_rate": perf.error_rate},
                )
                self._events.append(event)
                actions.append(
                    AdaptationAction(
                        action_type="swap_component",
                        target_component=cid,
                        priority=TaskPriority.CRITICAL,
                        reasoning=f"Error rate {perf.error_rate:.0%} > threshold {self._error_rate_threshold:.0%}",
                    )
                )
        return actions

    def _find_heaviest_components(self) -> list[tuple]:
        """Return components sorted by avg latency (proxy for resource use)."""
        items = [(cid, perf.avg_latency) for cid, perf in self._perf.items() if len(perf.latency_ms) > 0]
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def _apply_cooldown(self, actions: list[AdaptationAction]) -> list[AdaptationAction]:
        """Filter out actions that are within cooldown period."""
        now = time.monotonic()
        filtered = []
        for action in actions:
            key = f"{action.action_type}:{action.target_component or 'global'}"
            last = self._last_adaptation.get(key, 0.0)
            if now - last >= self._cooldown_s:
                filtered.append(action)
                self._last_adaptation[key] = now
        return filtered

    # ── Telemetry ─────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return adaptation engine status for monitoring."""
        return {
            "tracked_components": len(self._perf),
            "tracked_cameras": len(self._scene_history),
            "total_events": len(self._events),
            "recent_events": [
                {
                    "signal": e.signal.value,
                    "severity": e.severity.value,
                    "component": e.source_component,
                    "age_s": round(time.monotonic() - e.timestamp, 1),
                }
                for e in list(self._events)[-5:]
            ],
            "component_health": {cid: self.get_component_health(cid) for cid in self._perf},
        }
