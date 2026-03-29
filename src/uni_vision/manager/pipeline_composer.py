"""Pipeline Composer — assembles dynamic PipelineBlueprints.

Given a FrameContext (what capabilities are needed) and the resolved
components (what's available), the composer creates a PipelineBlueprint
that describes the ordered stages to execute.

Supports BOTH standard ``ComponentCapability`` enum stages AND free-form
dynamic capabilities discovered by the LLM.  For dynamic capabilities,
sensible defaults are used for ordering and IO keys.

The composer can also disassemble a blueprint for teardown.
"""

from __future__ import annotations

import structlog
import uuid
from typing import Dict, List, Optional, Set, Union

from uni_vision.components.base import ComponentCapability, ComponentState
from uni_vision.manager.component_registry import ComponentRegistry
from uni_vision.manager.schemas import (
    FrameContext,
    PipelineBlueprint,
    StageSpec,
    TaskPriority,
)

log = structlog.get_logger(__name__)


# ── Stage ordering (SEED KNOWLEDGE — performance shortcuts) ──────
# Defines the canonical execution order for KNOWN capabilities.
# Dynamic capabilities not in this dict get a default order of 99
# (post-processing), or the LLM can specify custom ordering.

_STAGE_ORDER: Dict[ComponentCapability, int] = {
    ComponentCapability.IMAGE_DENOISING: 10,
    ComponentCapability.IMAGE_ENHANCE: 15,
    ComponentCapability.SUPER_RESOLUTION: 20,
    ComponentCapability.GEOMETRIC_CORRECTION: 25,
    ComponentCapability.SCENE_CLASSIFICATION: 30,
    ComponentCapability.OBJECT_DETECTION: 40,
    ComponentCapability.VEHICLE_DETECTION: 41,
    ComponentCapability.PERSON_DETECTION: 42,
    ComponentCapability.FACE_DETECTION: 43,
    ComponentCapability.ZERO_SHOT_DETECTION: 44,
    ComponentCapability.INSTANCE_SEGMENTATION: 50,
    ComponentCapability.SEMANTIC_SEGMENTATION: 51,
    ComponentCapability.PLATE_DETECTION: 60,
    ComponentCapability.DEPTH_ESTIMATION: 65,
    ComponentCapability.TRACKING: 70,
    ComponentCapability.PLATE_OCR: 80,
    ComponentCapability.SCENE_TEXT_OCR: 81,
    ComponentCapability.DOCUMENT_OCR: 82,
    ComponentCapability.ANOMALY_DETECTION: 90,
    ComponentCapability.ACTION_RECOGNITION: 95,
    ComponentCapability.POSE_ESTIMATION: 96,
}

# IO key naming convention (SEED KNOWLEDGE — dynamic capabilities use
# sensible defaults: input_key="frame", output_key="output")
_INPUT_KEYS: Dict[ComponentCapability, str] = {
    ComponentCapability.IMAGE_DENOISING: "frame",
    ComponentCapability.IMAGE_ENHANCE: "frame",
    ComponentCapability.SUPER_RESOLUTION: "frame",
    ComponentCapability.GEOMETRIC_CORRECTION: "frame",
    ComponentCapability.SCENE_CLASSIFICATION: "frame",
    ComponentCapability.OBJECT_DETECTION: "frame",
    ComponentCapability.VEHICLE_DETECTION: "frame",
    ComponentCapability.PERSON_DETECTION: "frame",
    ComponentCapability.FACE_DETECTION: "frame",
    ComponentCapability.ZERO_SHOT_DETECTION: "frame",
    ComponentCapability.INSTANCE_SEGMENTATION: "frame",
    ComponentCapability.SEMANTIC_SEGMENTATION: "frame",
    ComponentCapability.PLATE_DETECTION: "frame",
    ComponentCapability.TRACKING: "detections",
    ComponentCapability.PLATE_OCR: "plate_crops",
    ComponentCapability.SCENE_TEXT_OCR: "text_regions",
    ComponentCapability.DOCUMENT_OCR: "frame",
    ComponentCapability.ANOMALY_DETECTION: "frame",
    ComponentCapability.DEPTH_ESTIMATION: "frame",
}

_OUTPUT_KEYS: Dict[ComponentCapability, str] = {
    ComponentCapability.IMAGE_DENOISING: "frame",
    ComponentCapability.IMAGE_ENHANCE: "frame",
    ComponentCapability.SUPER_RESOLUTION: "frame",
    ComponentCapability.GEOMETRIC_CORRECTION: "frame",
    ComponentCapability.SCENE_CLASSIFICATION: "scene_label",
    ComponentCapability.OBJECT_DETECTION: "detections",
    ComponentCapability.VEHICLE_DETECTION: "detections",
    ComponentCapability.PERSON_DETECTION: "detections",
    ComponentCapability.FACE_DETECTION: "face_detections",
    ComponentCapability.ZERO_SHOT_DETECTION: "detections",
    ComponentCapability.INSTANCE_SEGMENTATION: "masks",
    ComponentCapability.SEMANTIC_SEGMENTATION: "seg_map",
    ComponentCapability.PLATE_DETECTION: "plate_crops",
    ComponentCapability.TRACKING: "tracks",
    ComponentCapability.PLATE_OCR: "plate_texts",
    ComponentCapability.SCENE_TEXT_OCR: "text_results",
    ComponentCapability.DOCUMENT_OCR: "doc_text",
    ComponentCapability.ANOMALY_DETECTION: "anomalies",
    ComponentCapability.DEPTH_ESTIMATION: "depth_map",
}


class PipelineComposer:
    """Compose PipelineBlueprints from context + available components.

    Parameters
    ----------
    registry:
        Component registry for looking up component IDs.
    """

    def __init__(self, registry: ComponentRegistry) -> None:
        self._registry = registry

    # ── Public API ────────────────────────────────────────────────

    def compose(
        self,
        context: FrameContext,
        *,
        resolved_components: Optional[Dict[ComponentCapability, str]] = None,
        resolved_dynamic: Optional[Dict[str, str]] = None,
    ) -> PipelineBlueprint:
        """Create a pipeline blueprint for the given context.

        Parameters
        ----------
        context:
            The FrameContext describing what capabilities are needed.
            Includes both standard enum capabilities and dynamic string
            capabilities discovered by the LLM.
        resolved_components:
            Optional mapping of standard capability → component_id.
        resolved_dynamic:
            Optional mapping of dynamic capability label → component_id.

        Returns
        -------
        PipelineBlueprint ready for execution.
        """
        resolved = resolved_components or {}
        dyn_resolved = resolved_dynamic or {}
        stages: List[StageSpec] = []
        total_vram = 0

        # ── Standard (enum) capabilities ──
        all_caps = list(context.required_capabilities) + list(context.optional_capabilities)
        all_caps.sort(key=lambda c: _STAGE_ORDER.get(c, 100))

        for cap in all_caps:
            is_optional = cap in context.optional_capabilities

            component_id = resolved.get(cap)
            if not component_id:
                component_id = self._find_component_for_capability(cap)

            if not component_id:
                if is_optional:
                    log.debug("optional_cap_skipped capability=%s", cap.value)
                    continue
                log.warning("required_cap_unresolved capability=%s", cap.value)
                component_id = f"__unresolved__.{cap.value}"

            stage = StageSpec(
                stage_name=f"S_{cap.value}",
                required_capability=cap,
                component_id=component_id,
                is_optional=is_optional,
                input_key=_INPUT_KEYS.get(cap, "frame"),
                output_key=_OUTPUT_KEYS.get(cap, "output"),
            )
            stages.append(stage)

            comp = self._registry.get(component_id)
            if comp:
                total_vram += comp.metadata.resource_estimate.vram_mb

        # ── Dynamic (LLM-discovered) capabilities ──
        dyn_required = getattr(context, "dynamic_required", frozenset())
        dyn_optional = getattr(context, "dynamic_optional", frozenset())
        dyn_all = [(label, False) for label in dyn_required] + \
                  [(label, True) for label in dyn_optional]

        for dyn_label, is_optional in dyn_all:
            component_id = dyn_resolved.get(dyn_label)

            if not component_id:
                if is_optional:
                    log.debug("optional_dynamic_cap_skipped label=%s", dyn_label)
                    continue
                log.warning("required_dynamic_cap_unresolved label=%s", dyn_label)
                component_id = f"__unresolved__.dyn.{dyn_label}"

            # Dynamic stages: use default order 99 (after standard stages),
            # input_key="frame", output_key="output"
            stage = StageSpec(
                stage_name=f"S_dyn_{dyn_label}",
                required_capability=None,
                dynamic_capability=dyn_label,
                component_id=component_id,
                is_optional=is_optional,
                input_key="frame",
                output_key="output",
            )
            stages.append(stage)

            comp = self._registry.get(component_id) if component_id else None
            if comp:
                total_vram += comp.metadata.resource_estimate.vram_mb

        blueprint = PipelineBlueprint(
            blueprint_id=str(uuid.uuid4()),
            name=f"auto_{context.scene_type.value}_{len(stages)}stages",
            context=context,
            stages=tuple(stages),
            estimated_vram_mb=total_vram,
        )

        log.info(
            "blueprint_composed",
            name=blueprint.name,
            stages=len(stages),
            vram_mb=total_vram,
        )
        return blueprint

    def compose_anpr_default(self) -> PipelineBlueprint:
        """Shortcut: compose the default ANPR pipeline.

        This recreates the original hardcoded S0-S8 pipeline using
        the new dynamic system — useful for backward compatibility.
        """
        context = FrameContext(
            scene_type=_scene_type_import("TRAFFIC"),
            required_capabilities=frozenset({
                ComponentCapability.IMAGE_ENHANCE,
                ComponentCapability.GEOMETRIC_CORRECTION,
                ComponentCapability.VEHICLE_DETECTION,
                ComponentCapability.PLATE_DETECTION,
                ComponentCapability.PLATE_OCR,
            }),
            optional_capabilities=frozenset({
                ComponentCapability.TRACKING,
                ComponentCapability.SUPER_RESOLUTION,
            }),
            priority=TaskPriority.HIGH,
            camera_id="default_anpr",
        )
        return self.compose(context)

    # ── Internals ─────────────────────────────────────────────────

    def _find_component_for_capability(
        self,
        capability: ComponentCapability,
    ) -> Optional[str]:
        """Search registry for a component providing this capability."""
        # Prefer READY components, then any registered
        components = self._registry.get_by_capability(capability, only_ready=True)
        if components:
            return components[0].metadata.component_id

        components = self._registry.get_by_capability(capability)
        if components:
            return components[0].metadata.component_id

        return None


def _scene_type_import(value: str):
    """Lazy import helper to avoid circular imports."""
    from uni_vision.manager.schemas import SceneType
    return SceneType(value.lower())
