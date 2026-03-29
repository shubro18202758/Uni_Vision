"""Knowledge base and learning loop for the agentic subsystem.

Provides persistent and session-scoped knowledge that the agent
accumulates over time:

  * **Plate pattern knowledge** — tracks frequently seen plates,
    common OCR error patterns per camera, and correction histories.
  * **Feedback collection** — records operator confirmations or
    corrections so the agent can learn from ground truth.
  * **Error pattern analysis** — identifies systematic OCR failures
    (e.g., a specific camera always confuses O↔0) and feeds these
    back into dynamic prompt tuning.
  * **Anomaly signals** — flags unusual plate activity such as
    sudden frequency spikes or never-before-seen formats.

Storage: In-memory with periodic PostgreSQL persistence.  The KB
survives agent restarts by reloading from the ``agent_knowledge``
table at startup.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────


@dataclass
class PlateObservation:
    """A single observation of a plate number."""

    plate_text: str
    camera_id: str
    confidence: float
    engine: str
    validation_status: str
    timestamp: float = field(default_factory=time.time)
    was_corrected: bool = False
    original_ocr_text: str = ""


@dataclass
class CameraErrorProfile:
    """Aggregated error profile for a specific camera."""

    camera_id: str
    total_detections: int = 0
    error_count: int = 0
    common_confusions: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def error_rate(self) -> float:
        if self.total_detections == 0:
            return 0.0
        return self.error_count / self.total_detections

    def top_confusions(self, n: int = 5) -> List[Tuple[str, int]]:
        return sorted(
            self.common_confusions.items(), key=lambda x: -x[1]
        )[:n]


@dataclass
class FeedbackEntry:
    """Operator feedback on an agent or OCR result."""

    detection_id: str
    original_plate: str
    corrected_plate: str
    feedback_type: str  # "confirm" | "correct" | "reject"
    camera_id: str
    timestamp: float = field(default_factory=time.time)
    notes: str = ""


# ── Knowledge Base ────────────────────────────────────────────────


class KnowledgeBase:
    """In-memory knowledge base with periodic persistence.

    Parameters
    ----------
    max_observations : int
        Maximum plate observations to retain in memory (FIFO).
    max_feedback : int
        Maximum feedback entries to retain.
    """

    def __init__(
        self,
        *,
        max_observations: int = 10_000,
        max_feedback: int = 5_000,
    ) -> None:
        self._max_observations = max_observations
        self._max_feedback = max_feedback

        # Core knowledge stores
        self._observations: List[PlateObservation] = []
        self._feedback: List[FeedbackEntry] = []
        self._camera_profiles: Dict[str, CameraErrorProfile] = {}

        # Optional Databricks FAISS vector engine (set externally)
        self._vector_engine = None

        # Plate frequency tracking (plate_text → count)
        self._plate_frequency: Dict[str, int] = defaultdict(int)

        # Camera → plate set for cross-camera tracking
        self._camera_plates: Dict[str, set] = defaultdict(set)

        # Error pattern accumulator: "char_from→char_to" → count per camera
        self._error_patterns: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    # ── Observation Recording ─────────────────────────────────────

    def record_observation(self, obs: PlateObservation) -> None:
        """Record a plate detection observation."""
        self._observations.append(obs)
        if len(self._observations) > self._max_observations:
            self._observations = self._observations[-self._max_observations:]

        self._plate_frequency[obs.plate_text] += 1
        self._camera_plates[obs.camera_id].add(obs.plate_text)

        # Update camera profile
        profile = self._camera_profiles.get(obs.camera_id)
        if profile is None:
            profile = CameraErrorProfile(camera_id=obs.camera_id)
            self._camera_profiles[obs.camera_id] = profile

        profile.total_detections += 1
        if obs.validation_status not in ("valid",):
            profile.error_count += 1

        # Track character confusions
        if obs.was_corrected and obs.original_ocr_text:
            raw = obs.original_ocr_text.upper()
            corrected = obs.plate_text.upper()
            for a, b in zip(raw, corrected):
                if a != b:
                    key = f"{a}→{b}"
                    profile.common_confusions[key] = (
                        profile.common_confusions.get(key, 0) + 1
                    )
                    self._error_patterns[obs.camera_id][key] += 1

        # Running average confidence
        n = profile.total_detections
        profile.avg_confidence = (
            (profile.avg_confidence * (n - 1) + obs.confidence) / n
        )
        profile.last_updated = time.time()

        # FAISS vector embedding (Databricks add-on)
        if self._vector_engine is not None:
            try:
                self._vector_engine.add_plate_observation(
                    plate_text=obs.plate_text,
                    camera_id=obs.camera_id,
                    confidence=obs.confidence,
                    engine=obs.engine,
                    validation_status=obs.validation_status,
                    timestamp=obs.timestamp,
                )
            except Exception:
                logger.debug("faiss_embedding_failed plate=%s", obs.plate_text)

    # ── Feedback Collection ───────────────────────────────────────

    def record_feedback(self, entry: FeedbackEntry) -> None:
        """Record operator feedback on a detection."""
        self._feedback.append(entry)
        if len(self._feedback) > self._max_feedback:
            self._feedback = self._feedback[-self._max_feedback:]

        # If the feedback provides a correction, record as an observation
        if entry.feedback_type == "correct" and entry.corrected_plate:
            self.record_observation(
                PlateObservation(
                    plate_text=entry.corrected_plate,
                    camera_id=entry.camera_id,
                    confidence=1.0,
                    engine="human_feedback",
                    validation_status="valid",
                    was_corrected=True,
                    original_ocr_text=entry.original_plate,
                )
            )

        logger.info(
            "kb_feedback_recorded type=%s camera=%s plate=%s",
            entry.feedback_type,
            entry.camera_id,
            entry.corrected_plate or entry.original_plate,
        )

    # ── Queries ───────────────────────────────────────────────────

    def get_plate_frequency(
        self, *, top_n: int = 20
    ) -> List[Tuple[str, int]]:
        """Return the most frequently seen plates."""
        return sorted(
            self._plate_frequency.items(), key=lambda x: -x[1]
        )[:top_n]

    def get_camera_profile(self, camera_id: str) -> Optional[CameraErrorProfile]:
        """Get the error profile for a specific camera."""
        return self._camera_profiles.get(camera_id)

    def get_all_camera_profiles(self) -> Dict[str, CameraErrorProfile]:
        """Return all camera error profiles."""
        return dict(self._camera_profiles)

    def get_cross_camera_plates(
        self, *, min_cameras: int = 2
    ) -> Dict[str, List[str]]:
        """Find plates seen across multiple cameras."""
        # Invert: plate → list of cameras
        plate_cameras: Dict[str, List[str]] = defaultdict(list)
        for cam_id, plates in self._camera_plates.items():
            for plate in plates:
                plate_cameras[plate].append(cam_id)

        return {
            plate: cams
            for plate, cams in plate_cameras.items()
            if len(cams) >= min_cameras
        }

    def get_error_patterns(
        self, camera_id: Optional[str] = None, *, top_n: int = 10
    ) -> Dict[str, List[Tuple[str, int]]]:
        """Get common character confusion patterns.

        Parameters
        ----------
        camera_id : str, optional
            Filter to a specific camera. If None, return all cameras.
        top_n : int
            Maximum patterns per camera.
        """
        if camera_id:
            patterns = self._error_patterns.get(camera_id, {})
            return {
                camera_id: sorted(
                    patterns.items(), key=lambda x: -x[1]
                )[:top_n]
            }

        result: Dict[str, List[Tuple[str, int]]] = {}
        for cam_id, patterns in self._error_patterns.items():
            result[cam_id] = sorted(
                patterns.items(), key=lambda x: -x[1]
            )[:top_n]
        return result

    def get_recent_feedback(
        self, *, hours_back: float = 24, limit: int = 50
    ) -> List[FeedbackEntry]:
        """Return recent feedback entries."""
        cutoff = time.time() - (hours_back * 3600)
        recent = [f for f in self._feedback if f.timestamp >= cutoff]
        return recent[-limit:]

    def get_anomalies(
        self, *, hours_back: float = 1, spike_threshold: float = 3.0
    ) -> List[Dict[str, Any]]:
        """Detect anomalous plate activity in the recent window.

        Looks for:
        - Frequency spikes: plate seen spike_threshold× more than average
        - New plates: plates never seen before this window
        """
        cutoff = time.time() - (hours_back * 3600)
        recent = [o for o in self._observations if o.timestamp >= cutoff]

        if not recent:
            return []

        anomalies: List[Dict[str, Any]] = []

        # Frequency analysis in the window
        window_freq: Dict[str, int] = defaultdict(int)
        for obs in recent:
            window_freq[obs.plate_text] += 1

        # Compare to historical average per hour
        total_hours = max(1.0, len(self._observations) / max(1, len(recent)))
        for plate, count in window_freq.items():
            historical = self._plate_frequency.get(plate, 0)
            avg_per_hour = historical / total_hours if total_hours > 0 else 0

            if avg_per_hour > 0 and count > avg_per_hour * spike_threshold:
                anomalies.append({
                    "type": "frequency_spike",
                    "plate": plate,
                    "count_in_window": count,
                    "historical_avg_per_hour": round(avg_per_hour, 2),
                    "spike_factor": round(count / avg_per_hour, 1),
                })

        return anomalies[:20]

    # ── Statistics ────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of the knowledge base."""
        return {
            "total_observations": len(self._observations),
            "total_feedback": len(self._feedback),
            "unique_plates": len(self._plate_frequency),
            "cameras_profiled": len(self._camera_profiles),
            "max_observations": self._max_observations,
            "max_feedback": self._max_feedback,
        }

    # ── Prompt Enhancement ────────────────────────────────────────

    def get_camera_hints(self, camera_id: str) -> str:
        """Generate camera-specific hints for prompt injection.

        If a camera systematically confuses certain characters, this
        generates a hint string that can be injected into the OCR
        system prompt to improve accuracy.
        """
        profile = self._camera_profiles.get(camera_id)
        if profile is None:
            return ""

        top = profile.top_confusions(3)
        if not top:
            return ""

        hints = [f"This camera commonly confuses: "]
        for confusion, count in top:
            hints.append(f"  {confusion} ({count} times)")

        return "\n".join(hints)

    # ── Persistence ───────────────────────────────────────────────

    async def save_to_db(self, pg_client: Any) -> None:
        """Persist knowledge summaries to PostgreSQL."""
        if pg_client is None:
            return

        pool = getattr(pg_client, "_pool", None)
        if pool is None:
            return

        try:
            async with pool.acquire() as conn:
                # Store camera profiles as JSON
                for cam_id, profile in self._camera_profiles.items():
                    await conn.execute(
                        """
                        INSERT INTO agent_knowledge (key, category, data)
                        VALUES ($1, $2, $3::jsonb)
                        ON CONFLICT (key) DO UPDATE SET
                            data = $3::jsonb,
                            updated_at = NOW()
                        """,
                        f"camera_profile:{cam_id}",
                        "camera_profile",
                        _profile_to_json(profile),
                    )

                # Store plate frequency top 100
                top_plates = self.get_plate_frequency(top_n=100)
                import json

                await conn.execute(
                    """
                    INSERT INTO agent_knowledge (key, category, data)
                    VALUES ($1, $2, $3::jsonb)
                    ON CONFLICT (key) DO UPDATE SET
                        data = $3::jsonb,
                        updated_at = NOW()
                    """,
                    "plate_frequency_top100",
                    "plate_stats",
                    json.dumps({"plates": [{"text": p, "count": c} for p, c in top_plates]}),
                )

            logger.info(
                "kb_persisted cameras=%d", len(self._camera_profiles)
            )
        except Exception as exc:
            logger.warning("kb_persist_failed error=%s", exc)

    async def load_from_db(self, pg_client: Any) -> None:
        """Restore knowledge from PostgreSQL at startup."""
        if pg_client is None:
            return

        pool = getattr(pg_client, "_pool", None)
        if pool is None:
            return

        try:
            import json

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT key, category, data FROM agent_knowledge"
                )

            for row in rows:
                key = row["key"]
                category = row["category"]
                data = row["data"] if isinstance(row["data"], dict) else json.loads(row["data"])

                if category == "camera_profile" and key.startswith("camera_profile:"):
                    cam_id = key.split(":", 1)[1]
                    self._camera_profiles[cam_id] = _json_to_profile(cam_id, data)
                elif key == "plate_frequency_top100":
                    for item in data.get("plates", []):
                        self._plate_frequency[item["text"]] = item["count"]

            logger.info(
                "kb_loaded cameras=%d plates=%d",
                len(self._camera_profiles),
                len(self._plate_frequency),
            )
        except Exception as exc:
            logger.warning("kb_load_failed error=%s", exc)

    def clear(self) -> None:
        """Reset all in-memory knowledge (for testing)."""
        self._observations.clear()
        self._feedback.clear()
        self._camera_profiles.clear()
        self._plate_frequency.clear()
        self._camera_plates.clear()
        self._error_patterns.clear()


# ── Serialisation helpers ─────────────────────────────────────────


def _profile_to_json(profile: CameraErrorProfile) -> str:
    """Serialize a CameraErrorProfile to JSON string."""
    import json

    return json.dumps({
        "total_detections": profile.total_detections,
        "error_count": profile.error_count,
        "common_confusions": profile.common_confusions,
        "avg_confidence": profile.avg_confidence,
        "last_updated": profile.last_updated,
    })


def _json_to_profile(camera_id: str, data: Dict[str, Any]) -> CameraErrorProfile:
    """Deserialize a CameraErrorProfile from a JSON dict."""
    return CameraErrorProfile(
        camera_id=camera_id,
        total_detections=data.get("total_detections", 0),
        error_count=data.get("error_count", 0),
        common_confusions=data.get("common_confusions", {}),
        avg_confidence=data.get("avg_confidence", 0.0),
        last_updated=data.get("last_updated", 0.0),
    )
