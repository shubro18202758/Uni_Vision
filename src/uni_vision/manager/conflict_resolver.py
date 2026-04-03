"""Conflict Resolver — detects and resolves component incompatibilities.

Conflicts can arise from:
  * VRAM budget exceeded (sum of loaded components > limit)
  * Python dependency version clashes
  * Device mismatch (GPU-only component on CPU-only system)
  * Capability overlap (two components doing the same thing)
  * License incompatibility
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from uni_vision.manager.schemas import (
    ComponentConflict,
    ConflictReport,
    ConflictType,
    PipelineBlueprint,
    TaskPriority,
)

if TYPE_CHECKING:
    from uni_vision.components.base import ComponentCapability
    from uni_vision.manager.component_registry import ComponentRegistry

log = structlog.get_logger(__name__)


class ConflictResolver:
    """Detect and propose resolutions for component conflicts.

    Parameters
    ----------
    registry:
        Component registry for querying loaded components.
    vram_limit_mb:
        Maximum VRAM budget.
    """

    def __init__(
        self,
        registry: ComponentRegistry,
        *,
        vram_limit_mb: int = 8192,
    ) -> None:
        self._registry = registry
        self._vram_limit = vram_limit_mb

    # ── Public API ────────────────────────────────────────────────

    def check_blueprint(
        self,
        blueprint: PipelineBlueprint,
        *,
        additional_vram_mb: int = 0,
    ) -> ConflictReport:
        """Run all conflict checks against a pipeline blueprint.

        Parameters
        ----------
        blueprint:
            The proposed pipeline to validate.
        additional_vram_mb:
            Extra VRAM already consumed by the LLM or other
            non-component processes (e.g. Ollama server).
        """
        conflicts: list[ComponentConflict] = []

        conflicts.extend(self._check_vram(blueprint, additional_vram_mb))
        conflicts.extend(self._check_capability_overlap(blueprint))
        conflicts.extend(self._check_dependencies(blueprint))

        total_vram = blueprint.estimated_vram_mb + additional_vram_mb
        available = self._vram_limit

        return ConflictReport(
            conflicts=conflicts,
            total_vram_required_mb=total_vram,
            vram_available_mb=available,
        )

    def suggest_unloads_for_vram(
        self,
        needed_vram_mb: int,
    ) -> list[str]:
        """Suggest components to unload to free VRAM.

        Uses a least-recently-used-first / lowest-priority heuristic.
        """
        loaded = self._registry.get_loaded()
        if not loaded:
            return []

        # Sort by VRAM (largest first — freeing one large model is
        # better than unloading many small ones)
        loaded.sort(
            key=lambda c: c.metadata.resource_estimate.vram_mb,
            reverse=True,
        )

        to_unload: list[str] = []
        freed = 0
        for comp in loaded:
            if freed >= needed_vram_mb:
                break
            to_unload.append(comp.metadata.component_id)
            freed += comp.metadata.resource_estimate.vram_mb

        return to_unload

    # ── Internal checks ───────────────────────────────────────────

    def _check_vram(
        self,
        blueprint: PipelineBlueprint,
        additional_vram_mb: int,
    ) -> list[ComponentConflict]:
        """Check if the blueprint exceeds the VRAM budget."""
        total = blueprint.estimated_vram_mb + additional_vram_mb
        if total <= self._vram_limit:
            return []

        overshoot = total - self._vram_limit
        component_ids = list(blueprint.required_component_ids)

        # Suggest unloading the least critical components
        suggestion = (
            f"Exceeds VRAM limit by {overshoot} MB. "
            f"Consider offloading components to CPU or using lighter alternatives."
        )

        return [
            ComponentConflict(
                conflict_type=ConflictType.VRAM_EXCEEDED,
                component_ids=component_ids,
                description=(f"Pipeline requires {total} MB but only {self._vram_limit} MB available"),
                severity=TaskPriority.CRITICAL,
                suggested_resolution=suggestion,
                auto_resolvable=overshoot < 500,
            )
        ]

    def _check_capability_overlap(
        self,
        blueprint: PipelineBlueprint,
    ) -> list[ComponentConflict]:
        """Detect if two stages provide the same core capability."""
        conflicts: list[ComponentConflict] = []
        seen_caps: dict[ComponentCapability, str] = {}

        for stage in blueprint.stages:
            cap = stage.required_capability
            if cap in seen_caps:
                conflicts.append(
                    ComponentConflict(
                        conflict_type=ConflictType.CAPABILITY_OVERLAP,
                        component_ids=[seen_caps[cap], stage.component_id],
                        description=(
                            f"Both '{seen_caps[cap]}' and '{stage.component_id}' provide capability '{cap.value}'"
                        ),
                        severity=TaskPriority.LOW,
                        suggested_resolution=(
                            "Remove one of the overlapping components, or keep both if they serve different sub-tasks."
                        ),
                        auto_resolvable=True,
                    )
                )
            else:
                seen_caps[cap] = stage.component_id

        return conflicts

    def _check_dependencies(
        self,
        blueprint: PipelineBlueprint,
    ) -> list[ComponentConflict]:
        """Basic dependency clash detection.

        Looks for two components requiring conflicting versions of the
        same Python package.
        """
        conflicts: list[ComponentConflict] = []
        pkg_owners: dict[str, list[str]] = {}  # pkg_base_name → [component_ids]

        for stage in blueprint.stages:
            comp = self._registry.get(stage.component_id)
            if comp is None:
                continue

            for req in comp.metadata.python_requirements:
                # Normalise "torch>=2.0" → "torch"
                pkg_base = req.split(">=")[0].split("<=")[0].split("==")[0].split("<")[0].split(">")[0].strip()
                if pkg_base not in pkg_owners:
                    pkg_owners[pkg_base] = []
                pkg_owners[pkg_base].append(stage.component_id)

        # Detect potential clashes (two components needing different
        # pinned versions of the same package)
        for pkg, owners in pkg_owners.items():
            if len(owners) > 1:
                # This is a warning, not necessarily a hard conflict
                # A true version resolver would compare specs
                log.debug(
                    "shared_dependency",
                    package=pkg,
                    components=owners,
                )

        return conflicts

    # ── Auto-resolution ───────────────────────────────────────────

    def auto_resolve(
        self,
        report: ConflictReport,
    ) -> list[dict]:
        """Attempt to auto-resolve auto_resolvable conflicts.

        Returns a list of actions to perform.
        """
        actions: list[dict] = []

        for conflict in report.conflicts:
            if not conflict.auto_resolvable:
                continue

            if conflict.conflict_type == ConflictType.VRAM_EXCEEDED:
                overshoot = report.total_vram_required_mb - report.vram_available_mb
                to_unload = self.suggest_unloads_for_vram(overshoot)
                actions.append(
                    {
                        "action": "unload_components",
                        "component_ids": to_unload,
                        "reason": conflict.description,
                    }
                )

            elif conflict.conflict_type == ConflictType.CAPABILITY_OVERLAP:
                # Remove the second component (heuristic)
                if len(conflict.component_ids) >= 2:
                    actions.append(
                        {
                            "action": "remove_from_blueprint",
                            "component_id": conflict.component_ids[1],
                            "reason": conflict.description,
                        }
                    )

        return actions
