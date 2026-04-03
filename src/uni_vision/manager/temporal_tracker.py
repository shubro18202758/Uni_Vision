"""Temporal Context Tracker — multi-frame awareness for pipeline decisions.

Unlike single-frame analysis, this module maintains a rolling temporal
context that helps the pipeline understand:

  * **Object persistence** — which objects have been tracked across frames.
  * **Activity patterns** — parking-lot traffic density over time.
  * **Environmental trends** — lighting transitions (dusk/dawn), weather.
  * **Pipeline performance trends** — latency and confidence trajectories.

This context is fed to the ManagerAgent's LLM for more informed
pipeline composition decisions.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uni_vision.manager.schemas import FrameContext, SceneType

log = logging.getLogger(__name__)

_DEFAULT_WINDOW = 30  # frames


@dataclass
class ObjectTrack:
    """A tracked object across multiple frames."""

    track_id: str
    object_class: str
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    frame_count: int = 1
    positions: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=60))
    confidences: deque[float] = field(default_factory=lambda: deque(maxlen=60))

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.first_seen

    @property
    def is_stationary(self) -> bool:
        """True if object hasn't moved significantly in recent frames."""
        if len(self.positions) < 5:
            return True
        recent = list(self.positions)[-5:]
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        return x_range < 10 and y_range < 10  # pixels

    @property
    def avg_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)


@dataclass
class TemporalFrame:
    """A snapshot of one frame's state for temporal analysis."""

    timestamp: float
    camera_id: str
    scene_type: SceneType
    object_count: int
    avg_brightness: float = 0.0
    pipeline_latency_ms: float = 0.0
    detection_confidence: float = 0.0


@dataclass
class EnvironmentTrend:
    """Detected environmental trend."""

    trend_type: str  # "lighting_change" | "traffic_density" | "weather"
    direction: str  # "increasing" | "decreasing" | "stable"
    magnitude: float = 0.0  # 0.0–1.0
    since: float = field(default_factory=time.monotonic)


class TemporalTracker:
    """Multi-frame temporal context tracker.

    Maintains a rolling window of frame observations and provides
    summarised context for pipeline composition decisions.

    Parameters
    ----------
    window_size:
        Number of frames to keep in the temporal window.
    track_timeout_s:
        Seconds after which an unseen object track is expired.
    """

    def __init__(
        self,
        *,
        window_size: int = _DEFAULT_WINDOW,
        track_timeout_s: float = 10.0,
    ) -> None:
        self._window_size = window_size
        self._track_timeout = track_timeout_s

        # Per-camera frame history
        self._frames: dict[str, deque[TemporalFrame]] = {}

        # Per-camera object tracks
        self._tracks: dict[str, dict[str, ObjectTrack]] = {}

        # Per-camera environment trends
        self._trends: dict[str, list[EnvironmentTrend]] = {}

    def record_frame(
        self,
        camera_id: str,
        context: FrameContext,
        *,
        object_count: int = 0,
        avg_brightness: float = 0.0,
        pipeline_latency_ms: float = 0.0,
        detection_confidence: float = 0.0,
    ) -> None:
        """Record a new frame observation."""
        if camera_id not in self._frames:
            self._frames[camera_id] = deque(maxlen=self._window_size)

        frame = TemporalFrame(
            timestamp=time.monotonic(),
            camera_id=camera_id,
            scene_type=context.scene_type,
            object_count=object_count,
            avg_brightness=avg_brightness,
            pipeline_latency_ms=pipeline_latency_ms,
            detection_confidence=detection_confidence,
        )
        self._frames[camera_id].append(frame)

        # Detect trends
        self._update_trends(camera_id)

    def update_track(
        self,
        camera_id: str,
        track_id: str,
        object_class: str,
        *,
        position: tuple[float, float] | None = None,
        confidence: float = 0.5,
    ) -> None:
        """Update or create an object track."""
        if camera_id not in self._tracks:
            self._tracks[camera_id] = {}

        tracks = self._tracks[camera_id]
        if track_id not in tracks:
            tracks[track_id] = ObjectTrack(
                track_id=track_id,
                object_class=object_class,
            )

        track = tracks[track_id]
        track.last_seen = time.monotonic()
        track.frame_count += 1
        if position:
            track.positions.append(position)
        track.confidences.append(confidence)

    def expire_tracks(self, camera_id: str) -> list[str]:
        """Remove stale tracks and return their IDs."""
        if camera_id not in self._tracks:
            return []
        now = time.monotonic()
        expired = [
            tid for tid, track in self._tracks[camera_id].items() if (now - track.last_seen) > self._track_timeout
        ]
        for tid in expired:
            del self._tracks[camera_id][tid]
        return expired

    def get_context_summary(self, camera_id: str) -> dict[str, Any]:
        """Get a temporal context summary for pipeline decisions.

        This summary is designed to be included in LLM prompts for
        context-aware pipeline composition.
        """
        frames = self._frames.get(camera_id, deque())
        tracks = self._tracks.get(camera_id, {})
        trends = self._trends.get(camera_id, [])

        if not frames:
            return {"status": "no_data", "camera_id": camera_id}

        recent = list(frames)[-10:]
        scene_counts = Counter(f.scene_type for f in recent)
        dominant_scene = scene_counts.most_common(1)[0][0] if scene_counts else None

        # Object class distribution
        class_counts: dict[str, int] = Counter()
        stationary_count = 0
        for track in tracks.values():
            class_counts[track.object_class] += 1
            if track.is_stationary:
                stationary_count += 1

        # Brightness trend
        brightness_values = [f.avg_brightness for f in recent if f.avg_brightness > 0]
        brightness_trend = "stable"
        if len(brightness_values) >= 5:
            first_half = sum(brightness_values[: len(brightness_values) // 2]) / max(1, len(brightness_values) // 2)
            second_half = sum(brightness_values[len(brightness_values) // 2 :]) / max(
                1, len(brightness_values) - len(brightness_values) // 2
            )
            if second_half > first_half * 1.15:
                brightness_trend = "brightening"
            elif second_half < first_half * 0.85:
                brightness_trend = "darkening"

        # Performance trend
        latencies = [f.pipeline_latency_ms for f in recent if f.pipeline_latency_ms > 0]
        avg_latency = sum(latencies) / max(len(latencies), 1)

        return {
            "camera_id": camera_id,
            "frames_in_window": len(frames),
            "dominant_scene": dominant_scene.value if dominant_scene else "unknown",
            "scene_stability": scene_counts.most_common(1)[0][1] / len(recent) if recent else 1.0,
            "active_tracks": len(tracks),
            "stationary_objects": stationary_count,
            "object_classes": dict(class_counts),
            "brightness_trend": brightness_trend,
            "avg_pipeline_latency_ms": round(avg_latency, 1),
            "active_trends": [
                {
                    "type": t.trend_type,
                    "direction": t.direction,
                    "magnitude": round(t.magnitude, 3),
                }
                for t in trends
            ],
        }

    def get_capability_hints(self, camera_id: str) -> set[str]:
        """Suggest capabilities needed based on temporal context.

        For example, if many objects are moving fast, suggest tracking.
        If it's getting dark, suggest low-light enhancement.
        """
        hints: set[str] = set()
        summary = self.get_context_summary(camera_id)

        if summary.get("status") == "no_data":
            return hints

        # Many moving objects → suggest tracking
        active = summary.get("active_tracks", 0)
        stationary = summary.get("stationary_objects", 0)
        moving = active - stationary
        if moving > 3:
            hints.add("multi_object_tracking")

        # Darkening → suggest low-light enhancement
        if summary.get("brightness_trend") == "darkening":
            hints.add("image_enhance")
            hints.add("low_light_enhancement")

        # High latency → suggest lighter models
        if summary.get("avg_pipeline_latency_ms", 0) > 150:
            hints.add("lightweight_models")

        # Many vehicles → suggest vehicle-specific detection
        classes = summary.get("object_classes", {})
        if classes.get("vehicle", 0) + classes.get("car", 0) + classes.get("truck", 0) > 5:
            hints.add("plate_detection")
            hints.add("plate_ocr")

        return hints

    def status(self) -> dict[str, Any]:
        return {
            "tracked_cameras": len(self._frames),
            "cameras": {
                cam_id: {
                    "frames": len(frames),
                    "tracks": len(self._tracks.get(cam_id, {})),
                    "trends": len(self._trends.get(cam_id, [])),
                }
                for cam_id, frames in self._frames.items()
            },
        }

    # ── Internal ──────────────────────────────────────────────────

    def _update_trends(self, camera_id: str) -> None:
        """Detect environmental trends from recent frames."""
        frames = list(self._frames.get(camera_id, []))
        if len(frames) < 10:
            return

        if camera_id not in self._trends:
            self._trends[camera_id] = []

        trends = self._trends[camera_id]

        # Traffic density trend
        counts = [f.object_count for f in frames[-10:]]
        first_half = sum(counts[:5]) / 5
        second_half = sum(counts[5:]) / max(1, len(counts) - 5)

        if second_half > first_half * 1.3:
            self._upsert_trend(trends, "traffic_density", "increasing", (second_half - first_half) / max(first_half, 1))
        elif second_half < first_half * 0.7:
            self._upsert_trend(trends, "traffic_density", "decreasing", (first_half - second_half) / max(first_half, 1))
        else:
            self._upsert_trend(trends, "traffic_density", "stable", 0.0)

    @staticmethod
    def _upsert_trend(
        trends: list[EnvironmentTrend],
        trend_type: str,
        direction: str,
        magnitude: float,
    ) -> None:
        """Update an existing trend or create a new one."""
        for t in trends:
            if t.trend_type == trend_type:
                t.direction = direction
                t.magnitude = magnitude
                return
        trends.append(
            EnvironmentTrend(
                trend_type=trend_type,
                direction=direction,
                magnitude=magnitude,
            )
        )
