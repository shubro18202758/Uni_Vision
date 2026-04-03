"""Context Analyzer — determines what CV capabilities a frame requires.

This module provides two analysis strategies:

1. **Lightweight heuristic** (fast path):
   Uses basic image stats (resolution, brightness, motion hint) and
   camera metadata to infer the scene type without an LLM call.

2. **LLM-assisted** (deep path):
   Sends a low-res thumbnail to Gemma 4 E2B with the
   CONTEXT_ANALYSIS_PROMPT to get a structured scene description
   and required capabilities.

The Manager Agent decides which strategy to use based on latency
requirements and whether the scene type is ambiguous.
"""

from __future__ import annotations

import json
import structlog
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Set

import numpy as np

from uni_vision.components.base import ComponentCapability
from uni_vision.manager.schemas import (
    DiscoveryQuery,
    FrameContext,
    SceneType,
    TaskPriority,
)

log = structlog.get_logger(__name__)


# ── Scene heuristics (seed hints — NOT the ceiling) ──────────────

# Camera ID patterns that hint at scene type
_CAMERA_HINTS: Dict[str, SceneType] = {
    "traffic": SceneType.TRAFFIC,
    "highway": SceneType.TRAFFIC,
    "parking": SceneType.PARKING,
    "gate": SceneType.PARKING,
    "indoor": SceneType.INDOOR,
    "warehouse": SceneType.INDUSTRIAL,
    "factory": SceneType.INDUSTRIAL,
}

# Scene → default required capabilities (SEED KNOWLEDGE — used as fast-path
# fallback when LLM is not available.  NOT an upper bound on what the
# system can discover.)
_SCENE_CAPABILITIES: Dict[SceneType, FrozenSet[ComponentCapability]] = {
    SceneType.TRAFFIC: frozenset({
        ComponentCapability.VEHICLE_DETECTION,
        ComponentCapability.PLATE_DETECTION,
        ComponentCapability.PLATE_OCR,
        ComponentCapability.TRACKING,
    }),
    SceneType.PARKING: frozenset({
        ComponentCapability.VEHICLE_DETECTION,
        ComponentCapability.PLATE_DETECTION,
        ComponentCapability.PLATE_OCR,
    }),
    SceneType.SURVEILLANCE: frozenset({
        ComponentCapability.PERSON_DETECTION,
        ComponentCapability.FACE_DETECTION,
        ComponentCapability.TRACKING,
    }),
    SceneType.INDUSTRIAL: frozenset({
        ComponentCapability.OBJECT_DETECTION,
        ComponentCapability.ANOMALY_DETECTION,
    }),
    SceneType.INDOOR: frozenset({
        ComponentCapability.PERSON_DETECTION,
        ComponentCapability.SCENE_CLASSIFICATION,
    }),
    SceneType.UNKNOWN: frozenset({
        ComponentCapability.OBJECT_DETECTION,
        ComponentCapability.ANOMALY_DETECTION,
        ComponentCapability.SCENE_CLASSIFICATION,
    }),
    SceneType.GENERAL: frozenset({
        ComponentCapability.OBJECT_DETECTION,
        ComponentCapability.ANOMALY_DETECTION,
        ComponentCapability.SCENE_CLASSIFICATION,
    }),
}


class ContextAnalyzer:
    """Analyze frames to determine required CV pipeline capabilities.

    Parameters
    ----------
    llm_client:
        Optional async LLM client for deep analysis.  If None, only
        heuristic analysis is available.
    default_scene:
        Fallback scene type when heuristics can't determine one.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        *,
        default_scene: SceneType = SceneType.UNKNOWN,
    ) -> None:
        self._llm = llm_client
        self._default_scene = default_scene

        # Cache of camera_id → last-seen scene type
        self._camera_scene_cache: Dict[str, SceneType] = {}

    # ── Public API ────────────────────────────────────────────────

    async def analyze(
        self,
        frame: np.ndarray,
        *,
        camera_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_llm: bool = False,
    ) -> FrameContext:
        """Analyze a frame and return a FrameContext.

        Parameters
        ----------
        frame:
            BGR or RGB numpy array.
        camera_id:
            Source camera identifier (used for hint lookup).
        metadata:
            Additional metadata (e.g., GPS, timestamp).
        use_llm:
            If True, use Gemma 4 E2B for deep scene understanding.
        """
        metadata = metadata or {}

        # Heuristic path
        scene = self._heuristic_scene(frame, camera_id)

        if use_llm and self._llm is not None:
            llm_ctx = await self._llm_analysis(frame, camera_id, metadata)
            if llm_ctx is not None:
                return llm_ctx

        required = set(_SCENE_CAPABILITIES.get(scene, frozenset()))
        optional = self._optional_capabilities(frame, scene)

        context = FrameContext(
            scene_type=scene,
            required_capabilities=frozenset(required),
            optional_capabilities=frozenset(optional),
            priority=self._infer_priority(scene, metadata),
            camera_id=camera_id or "unknown",
            metadata=metadata,
        )

        # Update cache
        if camera_id:
            self._camera_scene_cache[camera_id] = scene

        log.info(
            "context_analyzed",
            scene=scene.value,
            required=[c.value for c in required],
            camera_id=camera_id,
        )
        return context

    # ── Heuristic analysis ────────────────────────────────────────

    def _heuristic_scene(
        self,
        frame: np.ndarray,
        camera_id: Optional[str],
    ) -> SceneType:
        """Infer scene type from camera ID hints and image stats."""
        # 1. Camera ID hint
        if camera_id:
            # Check cache first
            if camera_id in self._camera_scene_cache:
                return self._camera_scene_cache[camera_id]

            # Check keyword hints
            cam_lower = camera_id.lower()
            for keyword, scene in _CAMERA_HINTS.items():
                if keyword in cam_lower:
                    return scene

        # 2. Image statistics heuristics
        h, w = frame.shape[:2]
        brightness = float(np.mean(frame))

        # Wide aspect ratio + outdoor brightness — don't assume traffic
        # for uploaded videos; default to UNKNOWN so we get broad detection.
        aspect = w / max(h, 1)

        # Dark scene → surveillance at night
        if brightness < 40:
            return SceneType.SURVEILLANCE

        # Very high brightness with warm tones could indicate fire/industrial
        if brightness > 150 and frame.shape[2] >= 3:
            # Check red-channel dominance (fire heuristic)
            mean_bgr = np.mean(frame, axis=(0, 1))
            if len(mean_bgr) >= 3 and mean_bgr[2] > mean_bgr[0] * 1.3:
                return SceneType.INDUSTRIAL

        return self._default_scene

    def _optional_capabilities(
        self,
        frame: np.ndarray,
        scene: SceneType,
    ) -> Set[ComponentCapability]:
        """Suggest optional capabilities based on frame quality."""
        optional: Set[ComponentCapability] = set()

        h, w = frame.shape[:2]

        # Low resolution → suggest super resolution
        if max(h, w) < 480:
            optional.add(ComponentCapability.SUPER_RESOLUTION)

        # Potentially noisy image → denoising
        brightness = float(np.mean(frame))
        if brightness < 50:
            optional.add(ComponentCapability.IMAGE_ENHANCE)

        return optional

    # ── LLM-assisted analysis (OPEN-ENDED) ─────────────────────────

    async def _llm_analysis(
        self,
        frame: np.ndarray,
        camera_id: Optional[str],
        metadata: Dict[str, Any],
    ) -> Optional[FrameContext]:
        """Use Gemma 4 E2B for OPEN-ENDED scene understanding.

        Unlike the heuristic path, this method does NOT constrain the
        returned capabilities to the fixed ComponentCapability enum.
        Gemma 4 reasons freely about what the frame needs, producing both
        standard capabilities (mapped to the enum) and dynamic
        capabilities (free-form strings for open internet discovery).
        It also generates search queries that drive unbounded component
        discovery from HuggingFace, PyPI, and GitHub.
        """
        from uni_vision.manager.prompts import OPEN_DISCOVERY_PROMPT

        # Build a text description (we don't send raw pixels to text LLM)
        h, w = frame.shape[:2]
        brightness = float(np.mean(frame))

        # Look up previous context from cache
        prev = self._camera_scene_cache.get(camera_id or "", None)
        previous_ctx = prev.value if prev else "none"

        # Provide the known capabilities as HINTS (not a ceiling)
        known_caps = ", ".join(c.value for c in ComponentCapability)

        # Loaded components summary
        loaded_str = metadata.get("loaded_components_summary", "none")

        prompt = OPEN_DISCOVERY_PROMPT.format(
            camera_id=camera_id or "unknown",
            width=w,
            height=h,
            timestamp=metadata.get("timestamp_utc", "unknown"),
            brightness=f"{brightness:.0f}",
            previous_context=previous_ctx,
            scene_hints=metadata.get("scene_hints", f"avg_brightness={brightness:.0f}/255"),
            vram_available_mb=metadata.get("vram_available_mb", 1024),
            loaded_components=loaded_str,
            known_capabilities=known_caps,
        )

        try:
            response = await self._llm.generate(prompt)
            parsed = json.loads(response)

            # Parse scene type — accept custom scene types from the LLM
            scene_raw = parsed.get("scene_type", "unknown")
            try:
                scene = SceneType(scene_raw)
            except ValueError:
                # LLM returned a custom scene type — map to UNKNOWN but
                # preserve the original label in metadata
                scene = SceneType.UNKNOWN

            # Separate capabilities into standard (enum) and dynamic (free-form)
            standard_required: Set[ComponentCapability] = set()
            dynamic_required: Set[str] = set()

            for cap_entry in parsed.get("required_capabilities", []):
                cap_name = cap_entry if isinstance(cap_entry, str) else cap_entry.get("name", "")
                if not cap_name:
                    continue
                # Try to match against the known enum
                matched = self._match_capability(cap_name)
                if matched is not None:
                    standard_required.add(matched)
                else:
                    dynamic_required.add(cap_name)

            standard_optional: Set[ComponentCapability] = set()
            dynamic_optional: Set[str] = set()

            for cap_entry in parsed.get("optional_capabilities", []):
                cap_name = cap_entry if isinstance(cap_entry, str) else cap_entry.get("name", "")
                if not cap_name:
                    continue
                matched = self._match_capability(cap_name)
                if matched is not None:
                    standard_optional.add(matched)
                else:
                    dynamic_optional.add(cap_name)

            # Parse LLM-generated discovery queries
            discovery_queries: list[DiscoveryQuery] = []
            for dq in parsed.get("discovery_queries", []):
                discovery_queries.append(DiscoveryQuery(
                    query=dq.get("query", ""),
                    source=dq.get("source", "all"),
                    capability_hint=dq.get("capability_hint", ""),
                    context_rationale=dq.get("context_rationale", ""),
                    priority=dq.get("priority", 0),
                ))

            priority = TaskPriority(parsed.get("priority", "normal"))

            # Ensure at least object detection if nothing was identified
            if not standard_required and not dynamic_required:
                standard_required.add(ComponentCapability.OBJECT_DETECTION)

            return FrameContext(
                scene_type=scene,
                required_capabilities=frozenset(standard_required),
                optional_capabilities=frozenset(standard_optional),
                dynamic_required=frozenset(dynamic_required),
                dynamic_optional=frozenset(dynamic_optional),
                discovery_queries=discovery_queries,
                priority=priority,
                camera_id=camera_id or "unknown",
                metadata={
                    **metadata,
                    "llm_scene_label": scene_raw,
                    "llm_reasoning": parsed.get("reasoning", ""),
                },
            )

        except Exception as exc:
            log.warning("llm_context_analysis_failed", error=str(exc))
            return None

    @staticmethod
    def _match_capability(name: str) -> Optional[ComponentCapability]:
        """Attempt to match a free-form capability name to the enum.

        Returns the enum member if matched, else None (dynamic cap).
        """
        # Direct value match
        normalized = name.lower().strip().replace(" ", "_").replace("-", "_")
        for member in ComponentCapability:
            if member.value == normalized:
                return member
        # Also check enum member names (e.g., "VEHICLE_DETECTION")
        for member in ComponentCapability:
            if member.name.lower() == normalized:
                return member
        return None

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _infer_priority(
        scene: SceneType,
        metadata: Dict[str, Any],
    ) -> TaskPriority:
        """Infer task priority from scene type and metadata."""
        if metadata.get("alert") or metadata.get("alarm"):
            return TaskPriority.CRITICAL
        if scene == SceneType.TRAFFIC:
            return TaskPriority.HIGH
        if scene == SceneType.SURVEILLANCE:
            return TaskPriority.HIGH
        return TaskPriority.NORMAL
