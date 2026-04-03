"""Pipeline Validator — validate assembled pipeline blueprints.

Before executing a dynamically-composed pipeline, the validator:
  1. Checks all required components are registered and loadable.
  2. Verifies stage IO chaining (output of stage N feeds input of N+1).
  3. Confirms VRAM estimates fit within the budget.
  4. Optionally runs a dry-run with a dummy frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from uni_vision.components.base import ComponentState

if TYPE_CHECKING:
    from uni_vision.manager.component_registry import ComponentRegistry
    from uni_vision.manager.schemas import PipelineBlueprint

log = structlog.get_logger(__name__)


@dataclass
class ValidationIssue:
    """A single validation finding."""

    stage_name: str
    severity: str  # "error" | "warning"
    message: str


@dataclass
class ValidationReport:
    """Result of pipeline validation."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    dry_run_ms: float | None = None

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


class PipelineValidator:
    """Validate pipeline blueprints before execution.

    Parameters
    ----------
    registry:
        Component registry for checking component availability.
    vram_budget_mb:
        Maximum VRAM for components (after LLM reservation).
    """

    def __init__(
        self,
        registry: ComponentRegistry,
        *,
        vram_budget_mb: int = 2200,
    ) -> None:
        self._registry = registry
        self._vram_budget = vram_budget_mb

    # ── Public API ────────────────────────────────────────────────

    def validate(
        self,
        blueprint: PipelineBlueprint,
    ) -> ValidationReport:
        """Run all static validation checks."""
        issues: list[ValidationIssue] = []

        issues.extend(self._check_components_exist(blueprint))
        issues.extend(self._check_io_chaining(blueprint))
        issues.extend(self._check_vram(blueprint))
        issues.extend(self._check_no_empty_pipeline(blueprint))

        is_valid = not any(i.severity == "error" for i in issues)

        report = ValidationReport(is_valid=is_valid, issues=issues)

        log.info(
            "pipeline_validated",
            blueprint=blueprint.name,
            valid=is_valid,
            errors=len(report.errors),
            warnings=len(report.warnings),
        )
        return report

    async def dry_run(
        self,
        blueprint: PipelineBlueprint,
        *,
        frame_shape: tuple = (480, 640, 3),
    ) -> ValidationReport:
        """Run a quick dry-run with a dummy frame.

        Only executes on components that are already READY.
        """
        report = self.validate(blueprint)
        if not report.is_valid:
            return report

        dummy = np.zeros(frame_shape, dtype=np.uint8)
        data: dict[str, Any] = {"frame": dummy}
        t0 = time.monotonic()

        for stage in blueprint.stages:
            comp = self._registry.get(stage.component_id)
            if comp is None or comp.state != ComponentState.READY:
                report.issues.append(
                    ValidationIssue(
                        stage_name=stage.stage_name,
                        severity="warning",
                        message=f"Component {stage.component_id} not READY — skipped dry run",
                    )
                )
                continue

            try:
                input_data = data.get(stage.input_key, dummy)
                result = await comp.execute(input_data, context={"dry_run": True})
                if result is not None:
                    data[stage.output_key] = result
            except Exception as exc:
                report.issues.append(
                    ValidationIssue(
                        stage_name=stage.stage_name,
                        severity="error",
                        message=f"Dry run failed: {exc}",
                    )
                )
                report.is_valid = False

        report.dry_run_ms = (time.monotonic() - t0) * 1000
        return report

    # ── Checks ────────────────────────────────────────────────────

    def _check_components_exist(
        self,
        blueprint: PipelineBlueprint,
    ) -> list[ValidationIssue]:
        """Every stage must reference a registered component."""
        issues = []
        for stage in blueprint.stages:
            if stage.component_id.startswith("__unresolved__"):
                issues.append(
                    ValidationIssue(
                        stage_name=stage.stage_name,
                        severity="error" if not stage.is_optional else "warning",
                        message=f"Unresolved component for capability {stage.required_capability.value}",
                    )
                )
                continue

            comp = self._registry.get(stage.component_id)
            if comp is None:
                issues.append(
                    ValidationIssue(
                        stage_name=stage.stage_name,
                        severity="error" if not stage.is_optional else "warning",
                        message=f"Component {stage.component_id} not in registry",
                    )
                )
        return issues

    def _check_io_chaining(
        self,
        blueprint: PipelineBlueprint,
    ) -> list[ValidationIssue]:
        """Verify that each stage's input can be produced by a prior stage."""
        issues = []
        available_keys = {"frame"}  # Frame is always available as base input

        for stage in blueprint.stages:
            if stage.input_key not in available_keys:
                issues.append(
                    ValidationIssue(
                        stage_name=stage.stage_name,
                        severity="warning",
                        message=(
                            f"Input key '{stage.input_key}' not produced by any "
                            f"prior stage. Available: {available_keys}"
                        ),
                    )
                )
            available_keys.add(stage.output_key)

        return issues

    def _check_vram(
        self,
        blueprint: PipelineBlueprint,
    ) -> list[ValidationIssue]:
        """Check VRAM estimate against budget."""
        if blueprint.estimated_vram_mb > self._vram_budget:
            return [
                ValidationIssue(
                    stage_name="__pipeline__",
                    severity="warning",
                    message=(
                        f"Estimated VRAM {blueprint.estimated_vram_mb} MB exceeds "
                        f"budget {self._vram_budget} MB. LRU eviction may occur."
                    ),
                )
            ]
        return []

    def _check_no_empty_pipeline(
        self,
        blueprint: PipelineBlueprint,
    ) -> list[ValidationIssue]:
        """Pipeline must have at least one non-optional stage."""
        required_stages = [s for s in blueprint.stages if not s.is_optional]
        if not required_stages:
            return [
                ValidationIssue(
                    stage_name="__pipeline__",
                    severity="error",
                    message="Pipeline has no required stages",
                )
            ]
        return []
