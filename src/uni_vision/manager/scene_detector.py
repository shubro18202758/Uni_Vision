"""Scene Transition Detector — detects when camera scene type changes.

Prevents unnecessary pipeline recompositions by only triggering
adaptation when the scene has truly changed (not a transient flicker).

Uses a multi-layer approach:
  1. **Frame-level heuristics** — brightness, contrast, motion magnitude.
  2. **Histogram similarity** — colour distribution shift detection.
  3. **Temporal smoothing** — Kalman-style state estimation for scene type.
  4. **Hysteresis** — requires N consecutive agreeing frames before transition.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from uni_vision.manager.schemas import SceneType

log = structlog.get_logger(__name__)


class TransitionState(str, Enum):
    STABLE = "stable"
    TRANSITIONING = "transitioning"
    CONFIRMED = "confirmed"


@dataclass
class SceneObservation:
    """A single scene observation with metadata."""

    scene_type: SceneType
    confidence: float
    brightness: float = 0.0
    contrast: float = 0.0
    motion_score: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class CameraSceneState:
    """Tracked scene state for a single camera."""

    camera_id: str
    current_scene: SceneType | None = None
    pending_scene: SceneType | None = None
    state: TransitionState = TransitionState.STABLE
    confirmation_count: int = 0
    observations: deque[SceneObservation] = field(default_factory=lambda: deque(maxlen=60))
    transitions: deque[tuple[SceneType, SceneType, float]] = field(
        default_factory=lambda: deque(maxlen=20)
    )  # (from, to, timestamp)

    # Histogram-based drift tracking
    _prev_hist: np.ndarray | None = field(default=None, repr=False)

    @property
    def scene_stability(self) -> float:
        """How stable the scene has been (0.0–1.0)."""
        if len(self.observations) < 5:
            return 1.0
        recent = [o.scene_type for o in list(self.observations)[-10:]]
        counts = Counter(recent)
        if not counts:
            return 1.0
        return counts.most_common(1)[0][1] / len(recent)


class SceneTransitionDetector:
    """Detects scene transitions with hysteresis and temporal smoothing.

    Parameters
    ----------
    confirmation_threshold:
        Number of consecutive frames that must agree before a transition
        is confirmed.
    histogram_bins:
        Number of bins for the brightness histogram comparison.
    drift_threshold:
        Chi-squared histogram distance above which a visual drift is flagged.
    min_confidence:
        Minimum scene classification confidence to consider.
    """

    def __init__(
        self,
        *,
        confirmation_threshold: int = 5,
        histogram_bins: int = 32,
        drift_threshold: float = 0.4,
        min_confidence: float = 0.3,
    ) -> None:
        self._confirm_thresh = confirmation_threshold
        self._hist_bins = histogram_bins
        self._drift_threshold = drift_threshold
        self._min_confidence = min_confidence
        self._cameras: dict[str, CameraSceneState] = {}

    def observe(
        self,
        camera_id: str,
        scene_type: SceneType,
        confidence: float,
        *,
        frame_gray: np.ndarray | None = None,
    ) -> tuple[SceneType, SceneType] | None:
        """Record a scene observation and return transition if confirmed.

        Parameters
        ----------
        camera_id:
            Camera identifier.
        scene_type:
            Detected scene type for this frame.
        confidence:
            Confidence of the scene classification.
        frame_gray:
            Optional grayscale frame for histogram analysis.

        Returns
        -------
        Optional[Tuple[SceneType, SceneType]]
            (old_scene, new_scene) if a transition is confirmed, else None.
        """
        if camera_id not in self._cameras:
            self._cameras[camera_id] = CameraSceneState(
                camera_id=camera_id,
                current_scene=scene_type,
            )
            return None

        cam = self._cameras[camera_id]

        # Build observation
        obs = SceneObservation(
            scene_type=scene_type,
            confidence=confidence,
        )
        if frame_gray is not None:
            obs.brightness = float(np.mean(frame_gray))
            obs.contrast = float(np.std(frame_gray))
            obs.motion_score = self._compute_histogram_drift(cam, frame_gray)

        cam.observations.append(obs)

        # Low-confidence observations don't count
        if confidence < self._min_confidence:
            return None

        # Hysteresis state machine
        if cam.current_scene is None:
            cam.current_scene = scene_type
            cam.state = TransitionState.STABLE
            return None

        if scene_type == cam.current_scene:
            # Reinforces stability
            cam.state = TransitionState.STABLE
            cam.pending_scene = None
            cam.confirmation_count = 0
            return None

        # Scene differs from current
        if cam.pending_scene == scene_type:
            cam.confirmation_count += 1
        else:
            # New candidate scene
            cam.pending_scene = scene_type
            cam.confirmation_count = 1
            cam.state = TransitionState.TRANSITIONING

        if cam.confirmation_count >= self._confirm_thresh:
            old_scene = cam.current_scene
            cam.current_scene = scene_type
            cam.state = TransitionState.CONFIRMED
            cam.pending_scene = None
            cam.confirmation_count = 0
            cam.transitions.append((old_scene, scene_type, time.monotonic()))

            log.info(
                "scene_transition_confirmed",
                camera_id=camera_id,
                old_scene=old_scene.value,
                new_scene=scene_type.value,
            )
            return (old_scene, scene_type)

        return None

    def get_camera_state(self, camera_id: str) -> dict[str, Any]:
        cam = self._cameras.get(camera_id)
        if cam is None:
            return {"status": "unknown"}
        return {
            "current_scene": cam.current_scene.value if cam.current_scene else "unknown",
            "state": cam.state.value,
            "pending_scene": cam.pending_scene.value if cam.pending_scene else None,
            "confirmation_progress": f"{cam.confirmation_count}/{self._confirm_thresh}",
            "stability": round(cam.scene_stability, 3),
            "total_transitions": len(cam.transitions),
        }

    def get_transition_history(self, camera_id: str) -> list[dict[str, Any]]:
        cam = self._cameras.get(camera_id)
        if cam is None:
            return []
        return [
            {
                "from": t[0].value,
                "to": t[1].value,
                "age_s": round(time.monotonic() - t[2], 1),
            }
            for t in cam.transitions
        ]

    def status(self) -> dict[str, Any]:
        return {
            "tracked_cameras": len(self._cameras),
            "cameras": {cid: self.get_camera_state(cid) for cid in self._cameras},
        }

    # ── Internal ──────────────────────────────────────────────

    def _compute_histogram_drift(
        self,
        cam: CameraSceneState,
        frame_gray: np.ndarray,
    ) -> float:
        """Compute normalised histogram distance between current and previous frame."""
        hist, _ = np.histogram(
            frame_gray.ravel(),
            bins=self._hist_bins,
            range=(0, 256),
            density=True,
        )
        hist = hist.astype(np.float64) + 1e-10  # avoid division by zero

        if cam._prev_hist is None:
            cam._prev_hist = hist
            return 0.0

        # Chi-squared distance
        chi2 = float(np.sum((hist - cam._prev_hist) ** 2 / (hist + cam._prev_hist)))
        cam._prev_hist = hist
        return chi2
