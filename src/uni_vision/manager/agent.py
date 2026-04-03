"""Manager Agent — the Gemma 4 E2B meta-orchestrator.

This is the brain of the self-assembling CV platform.  Gemma 4 E2B
no longer does OCR or direct reasoning on frames.  Instead it:

  1. Receives a frame + camera context.
  2. Analyses what capabilities are needed (via ContextAnalyzer).
  3. Checks the ComponentRegistry for loaded components.
  4. Discovers missing capabilities on HuggingFace Hub.
  5. Downloads/loads new components (via LifecycleManager).
  6. Resolves conflicts (via ConflictResolver).
  7. Composes a PipelineBlueprint (via PipelineComposer).
  8. Validates and executes the pipeline.
  9. Returns the final result.

The Manager Agent operates in a ReAct-style loop with structured
JSON actions, powered by the prompts in ``manager/prompts.py``.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import structlog

from uni_vision.components.base import ComponentCapability, ComponentState
from uni_vision.manager.adaptation_engine import AdaptationAction, AdaptationEngine
from uni_vision.manager.compatibility import CompatibilityMatrix
from uni_vision.manager.fallback_chain import FallbackChainManager
from uni_vision.manager.feedback_loop import FeedbackLoop
from uni_vision.manager.gpu_profiler import GPUProfiler
from uni_vision.manager.prompts import build_manager_system_prompt
from uni_vision.manager.quality_scorer import QualityScorer
from uni_vision.manager.scene_detector import SceneTransitionDetector
from uni_vision.manager.schemas import (
    FrameContext,
    ManagerAction,
    ManagerDecision,
    PipelineBlueprint,
    PipelineExecutionResult,
    StageResult,
)
from uni_vision.manager.temporal_tracker import TemporalTracker

if TYPE_CHECKING:
    import numpy as np

    from uni_vision.manager.component_registry import ComponentRegistry
    from uni_vision.manager.component_resolver import ComponentResolver
    from uni_vision.manager.conflict_resolver import ConflictResolver
    from uni_vision.manager.context_analyzer import ContextAnalyzer
    from uni_vision.manager.hub_client import HubClient
    from uni_vision.manager.job_lifecycle import JobLifecycleManager
    from uni_vision.manager.lifecycle import LifecycleManager
    from uni_vision.manager.pipeline_composer import PipelineComposer
    from uni_vision.manager.pipeline_validator import PipelineValidator

log = structlog.get_logger(__name__)

# Maximum ReAct iterations before the agent gives up
_MAX_ITERATIONS = 6


class ManagerAgent:
    """The central Manager Agent that orchestrates the CV pipeline.

    This is the brain of the self-assembling adaptive CV platform.
    Beyond basic pipeline composition, it maintains closed-loop
    feedback, performs real-time adaptation, handles fallbacks, and
    uses temporal awareness for smarter decisions.

    Parameters
    ----------
    llm_client:
        Async LLM client for Gemma 4 E2B.
    registry:
        Component registry tracking all available components.
    hub_client:
        Multi-source component discovery client.
    resolver:
        Capability → component resolver.
    conflict_resolver:
        Conflict detection and resolution.
    context_analyzer:
        Frame context analysis.
    composer:
        Pipeline blueprint composer.
    lifecycle:
        VRAM-aware component lifecycle controller.
    validator:
        Pipeline validation.
    adaptation_engine:
        Real-time pipeline adaptation controller.
    feedback_loop:
        Closed-loop performance telemetry.
    fallback_manager:
        Ordered fallback chains per capability.
    quality_scorer:
        Bayesian component quality scoring.
    scene_detector:
        Hysteresis scene transition detection.
    gpu_profiler:
        Runtime VRAM measurement and leak detection.
    compat_matrix:
        Inter-component compatibility tracking.
    temporal_tracker:
        Multi-frame temporal context tracker.
    """

    def __init__(
        self,
        *,
        llm_client: Any,
        registry: ComponentRegistry,
        hub_client: HubClient,
        resolver: ComponentResolver,
        conflict_resolver: ConflictResolver,
        context_analyzer: ContextAnalyzer,
        composer: PipelineComposer,
        lifecycle: LifecycleManager,
        validator: PipelineValidator,
        # ── Adaptive subsystems (all optional for backward compat) ──
        adaptation_engine: AdaptationEngine | None = None,
        feedback_loop: FeedbackLoop | None = None,
        fallback_manager: FallbackChainManager | None = None,
        quality_scorer: QualityScorer | None = None,
        scene_detector: SceneTransitionDetector | None = None,
        gpu_profiler: GPUProfiler | None = None,
        compat_matrix: CompatibilityMatrix | None = None,
        temporal_tracker: TemporalTracker | None = None,
        job_lifecycle: JobLifecycleManager | None = None,
    ) -> None:
        self._llm = llm_client
        self._registry = registry
        self._hub = hub_client
        self._resolver = resolver
        self._conflicts = conflict_resolver
        self._analyzer = context_analyzer
        self._composer = composer
        self._lifecycle = lifecycle
        self._validator = validator

        # Adaptive subsystems
        self._adaptation = adaptation_engine or AdaptationEngine()
        self._feedback = feedback_loop or FeedbackLoop()
        self._fallbacks = fallback_manager or FallbackChainManager()
        self._quality = quality_scorer or QualityScorer()
        self._scene_det = scene_detector or SceneTransitionDetector()
        self._gpu = gpu_profiler or GPUProfiler()
        self._compat = compat_matrix or CompatibilityMatrix()
        self._temporal = temporal_tracker or TemporalTracker()

        # Job lifecycle (tracks per-job dynamic components + anomaly state)
        self._job_lifecycle: JobLifecycleManager | None = job_lifecycle

        # Cache last blueprint per camera to avoid re-composing
        self._blueprint_cache: dict[str, PipelineBlueprint] = {}

        # Cache component IDs that already failed provisioning (skip retries)
        self._provision_failed: set[str] = set()

        # Cameras where ALL components failed — skip expensive pipeline build
        # Maps camera_id → monotonic timestamp.  Retries after _DEGRADED_RETRY_S.
        self._degraded_cameras: dict[str, float] = {}
        self._DEGRADED_RETRY_S: float = 300.0  # 5 minutes

        # Track consecutive adaptation actions to avoid thrash
        self._adaptation_count: int = 0
        self._max_adaptations_per_frame: int = 3

    # ── Main entry point ──────────────────────────────────────────

    async def process_frame(
        self,
        frame: np.ndarray,
        *,
        camera_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineExecutionResult:
        """Process a single frame through the adaptive dynamic pipeline.

        The hot path — called for every frame from the orchestrator.

        Steps:
          1. Analyze frame context (+ scene transition detection)
          2. Enrich with temporal context and capability hints
          3. Check if cached blueprint still valid
          4. If not, compose a new blueprint (may involve LLM)
          5. Pre-flight: compatibility check + VRAM validation
          6. Ensure components are loaded (with GPU profiling)
          7. Execute pipeline stages (with per-stage feedback)
          8. Post-execution: record telemetry, run adaptation engine
          9. Apply adaptations (fallbacks, swaps) for next frame
        """
        t0 = time.monotonic()
        metadata = metadata or {}
        self._adaptation_count = 0
        cam = camera_id or "default"

        # 0. Degraded-mode early exit — all components missing, skip expensive build
        degraded_since = self._degraded_cameras.get(cam)
        if degraded_since is not None:
            if (time.monotonic() - degraded_since) < self._DEGRADED_RETRY_S:
                return PipelineExecutionResult(
                    blueprint_id="degraded-skip",
                    stage_results=[],
                    final_output={},
                    success=False,
                    total_elapsed_ms=(time.monotonic() - t0) * 1000,
                )
            else:
                # Retry interval elapsed — clear caches and try again
                del self._degraded_cameras[cam]
                self._provision_failed.clear()
                self.invalidate_cache(cam)
                log.info("degraded_retry", camera=cam)

        # 1. Analyze context
        context = await self._analyzer.analyze(
            frame,
            camera_id=camera_id,
            metadata=metadata,
        )

        # 1b. Scene transition detection (hysteresis)
        scene_change = self._scene_det.observe(
            camera_id=cam,
            scene_type=context.scene_type.value,
            confidence=0.8,
            frame_gray=None,  # full histogram drift requires gray frame
        )
        if scene_change is not None:
            old_scene, new_scene = scene_change
            log.info("scene_transition", camera=cam, old=old_scene, new=new_scene)
            # Invalidate cache — scene changed, pipeline probably needs rebuilding
            self.invalidate_cache(cam)

        # 2. Enrich context with temporal hints
        temporal_hints = self._temporal.get_capability_hints(cam)
        self._temporal.get_context_summary(cam)

        # 3. Check cache
        blueprint = self._get_cached_blueprint(context)

        if blueprint is None:
            # 4. Build new pipeline (may consult quality scores + compat)
            blueprint = await self._build_pipeline(context, temporal_hints, camera_id=camera_id)
            if camera_id:
                self._blueprint_cache[camera_id] = blueprint

        # 5. Pre-flight compatibility check
        comp_ids = list(blueprint.required_component_ids)
        compat_issues = self._compat.check_set(comp_ids)
        if compat_issues:
            log.warning("compatibility_issues", issues=[str(i) for i in compat_issues])
            # Try to find an alternative blueprint
            blueprint = await self._resolve_compat_issues(blueprint, compat_issues)

        # 6. Ensure all components are loaded (with GPU profiling)
        load_results = await self._lifecycle.ensure_loaded(list(blueprint.required_component_ids))
        failed_loads = [cid for cid, ok in load_results.items() if not ok]

        if failed_loads:
            log.warning("components_failed_to_load", failed=failed_loads)
            # Try fallback components
            blueprint = await self._apply_fallbacks_for_failed(blueprint, set(failed_loads))

        # 7. Execute with per-stage telemetry
        result = await self._execute_blueprint(blueprint, frame)
        result.total_elapsed_ms = (time.monotonic() - t0) * 1000

        # 7b. Mark camera degraded when zero stages succeeded
        if not result.success and not any(sr.success for sr in result.stage_results):
            self._degraded_cameras[cam] = time.monotonic()
            log.info("camera_degraded", camera=cam, reason="all_stages_failed")

        # 8. Post-execution: record telemetry across all subsystems
        await self._record_telemetry(result, context, blueprint, cam)

        # 9. Adaptation: check for performance issues, apply for next frame
        adaptations = self._adaptation.ingest_result(result, context)
        if adaptations:
            await self._apply_adaptations(adaptations, cam)

        # Update temporal tracker
        self._temporal.record_frame(
            cam,
            context,
            object_count=self._count_detections(result),
            pipeline_latency_ms=result.total_elapsed_ms,
            detection_confidence=self._avg_confidence(result),
        )

        log.info(
            "frame_processed",
            camera_id=camera_id,
            stages=len(blueprint.stages),
            success=result.success,
            elapsed_ms=f"{result.total_elapsed_ms:.1f}",
            adaptations=len(adaptations) if adaptations else 0,
        )

        return result

    # ── Pipeline building ─────────────────────────────────────────

    async def _build_pipeline(
        self,
        context: FrameContext,
        temporal_hints: set[str] | None = None,
        camera_id: str | None = None,
    ) -> PipelineBlueprint:
        """Build a new pipeline for the given context.

        This may involve:
          * Checking the registry for existing components
          * Consulting quality scores to pick the best component per slot
          * Using temporal hints to request additional capabilities
          * Searching multi-source Hub for missing capabilities
          * LLM-driven open internet search for dynamic capabilities
          * Downloading and provisioning new components
          * Checking pairwise compatibility
          * Resolving conflicts
          * Validating the final blueprint
        """
        # Expand required capabilities with temporal hints
        required: set[ComponentCapability] = set(context.required_capabilities)
        if temporal_hints:
            for hint in temporal_hints:
                try:
                    cap = ComponentCapability(hint)
                    required.add(cap)
                except ValueError:
                    pass  # hint isn't a known capability — OK

        # Check what capabilities are missing
        missing = self._registry.get_missing_capabilities(required)

        resolved_map: dict[ComponentCapability, str] = {}

        # For capabilities already available locally, pick the best by quality
        for cap in required - missing:
            best = self._quality.get_best_for_capability(cap.value)
            if best:
                resolved_map[cap] = best

        if missing:
            log.info("resolving_missing_capabilities", missing=[c.value for c in missing])

            # Resolve missing capabilities (searches multi-source Hub if needed)
            results = await self._resolver.resolve_capabilities(
                missing,
                available_vram_mb=self._lifecycle.vram_free_mb,
            )

            for res in results:
                if res.selected_candidate:
                    candidate = res.selected_candidate

                    # Compatibility check against existing loaded components
                    loaded_ids = [c for c in self._registry.get_loaded()]
                    compat_ok = True
                    for lid in loaded_ids:
                        if not self._compat.is_compatible(candidate.component_id, lid):
                            log.warning(
                                "incompatible_candidate",
                                candidate=candidate.component_id,
                                conflict_with=lid,
                            )
                            compat_ok = False
                            break

                    if not compat_ok:
                        continue

                    # Provision if not already in registry
                    if candidate.component_id not in self._registry:
                        # Skip if previously failed provisioning
                        if candidate.component_id in self._provision_failed:
                            continue
                        try:
                            await self._resolver.provision_candidate(candidate)
                            # Track as dynamic component for job lifecycle
                            if self._job_lifecycle and camera_id:
                                job = self._job_lifecycle.get_job_for_camera(camera_id)
                                if job:
                                    pip_pkg = (
                                        candidate.python_requirements[0] if candidate.python_requirements else None
                                    )
                                    await self._job_lifecycle.register_dynamic_component(
                                        job.job_id,
                                        candidate.component_id,
                                        pip_package=pip_pkg,
                                    )
                        except Exception as exc:
                            log.warning(
                                "provision_failed",
                                candidate=candidate.component_id,
                                error=str(exc),
                            )
                            self._provision_failed.add(candidate.component_id)
                            continue

                    # Verify component reached READY state
                    prov_comp = self._registry.get(candidate.component_id)
                    if prov_comp is None or prov_comp.state != ComponentState.READY:
                        log.warning(
                            "component_not_ready_after_provision",
                            cid=candidate.component_id,
                            state=prov_comp.state.value if prov_comp else "missing",
                        )
                        self._provision_failed.add(candidate.component_id)

                    resolved_map[res.capability] = candidate.component_id

                    # Register in fallback chain (enum + full candidate)
                    self._fallbacks.register_candidate(
                        res.capability,
                        candidate,
                    )

                    # Auto-wire provisioned OCR components into OCR strategy
                    if res.capability in (
                        ComponentCapability.PLATE_OCR,
                        ComponentCapability.SCENE_TEXT_OCR,
                        ComponentCapability.DOCUMENT_OCR,
                    ):
                        self._wire_ocr_engine(prov_comp, candidate.name)

        # ── Dynamic capability resolution (LLM-driven) ──
        dyn_required = getattr(context, "dynamic_required", frozenset())
        dyn_optional = getattr(context, "dynamic_optional", frozenset())
        discovery_queries = getattr(context, "discovery_queries", [])
        resolved_dynamic: dict[str, str] = {}

        all_dynamic = set(dyn_required) | set(dyn_optional)
        if all_dynamic:
            log.info(
                "resolving_dynamic_capabilities",
                dynamic=[str(d) for d in all_dynamic],
                queries=len(discovery_queries),
            )
            dyn_results = await self._resolver.resolve_dynamic_capabilities(
                all_dynamic,
                discovery_queries=discovery_queries,
                available_vram_mb=self._lifecycle.vram_free_mb,
            )

            for dres in dyn_results:
                if dres.selected_candidate:
                    cand = dres.selected_candidate
                    # Provision dynamic components
                    if cand.component_id not in self._registry:
                        if cand.component_id in self._provision_failed:
                            continue
                        try:
                            await self._resolver.provision_candidate(cand)
                            # Track as dynamic component for job lifecycle
                            if self._job_lifecycle and camera_id:
                                job = self._job_lifecycle.get_job_for_camera(camera_id)
                                if job:
                                    pip_pkg = cand.python_requirements[0] if cand.python_requirements else None
                                    await self._job_lifecycle.register_dynamic_component(
                                        job.job_id,
                                        cand.component_id,
                                        pip_package=pip_pkg,
                                    )
                        except Exception as exc:
                            log.warning(
                                "dynamic_provision_failed",
                                candidate=cand.component_id,
                                error=str(exc),
                            )
                            self._provision_failed.add(cand.component_id)
                            continue

                    # Determine which dynamic label this resolves
                    # Match via the DiscoveryQuery capability_hint
                    for dl in all_dynamic:
                        if dl not in resolved_dynamic:
                            resolved_dynamic[dl] = cand.component_id
                            break

        # Compose the blueprint (including dynamic stages)
        blueprint = self._composer.compose(
            context,
            resolved_components=resolved_map,
            resolved_dynamic=resolved_dynamic,
        )

        # Check for conflicts
        vram_status = self._lifecycle.status()
        report = self._conflicts.check_blueprint(
            blueprint,
            additional_vram_mb=vram_status["vram_reserved_mb"],
        )

        if report.has_conflicts:
            # Try auto-resolution
            actions = self._conflicts.auto_resolve(report)
            for action in actions:
                if action["action"] == "unload_components":
                    for cid in action["component_ids"]:
                        await self._lifecycle.unload_component(cid)

            # If still blocking conflicts, ask LLM for help
            if report.blocking_conflicts:
                blueprint = await self._llm_resolve_conflicts(blueprint, report)

        # Validate
        validation = self._validator.validate(blueprint)
        if not validation.is_valid:
            log.warning(
                "blueprint_validation_failed",
                errors=[i.message for i in validation.errors],
            )

        return blueprint

    # ── LLM-assisted operations ───────────────────────────────────

    async def _llm_resolve_conflicts(
        self,
        blueprint: PipelineBlueprint,
        report: Any,
    ) -> PipelineBlueprint:
        """Ask LLM for help resolving blocking conflicts."""
        from uni_vision.manager.prompts import CONFLICT_RESOLUTION_PROMPT

        conflicts_desc = "\n".join(f"- [{c.conflict_type.value}] {c.description}" for c in report.blocking_conflicts)

        vram_status = self._lifecycle.status()
        loaded_summary = json.dumps(self._registry.loaded_summary(), indent=2)
        proposed = json.dumps(
            [s.component_id for s in blueprint.stages],
            indent=2,
        )

        prompt = CONFLICT_RESOLUTION_PROMPT.format(
            conflicts_description=conflicts_desc,
            vram_total_mb=vram_status["vram_total_mb"],
            vram_used_mb=vram_status["vram_used_mb"],
            vram_available_mb=vram_status["vram_free_mb"],
            loaded_components=loaded_summary,
            proposed_components=proposed,
        )

        try:
            response = await self._llm.generate(prompt)
            parsed = json.loads(response)

            for action in parsed.get("actions", []):
                if action.get("action") == "unload":
                    cid = action.get("component_id")
                    if cid:
                        await self._lifecycle.unload_component(cid)
                elif action.get("action") == "offload_to_cpu":
                    cid = action.get("component_id")
                    if cid:
                        comp = self._registry.get(cid)
                        if comp:
                            await comp.unload()
                            await comp.load(device="cpu")

            # Recompose after conflict resolution
            return self._composer.compose(blueprint.context)

        except Exception as exc:
            log.warning("llm_conflict_resolution_failed", error=str(exc))
            return blueprint

    async def llm_decide(
        self,
        query: str,
        *,
        context: FrameContext | None = None,
    ) -> ManagerAction:
        """Use the LLM to make a strategic decision.

        This is used for complex decisions that heuristics can't handle,
        such as choosing between two equally-scored components, or
        deciding whether to download a large model.
        """
        vram_status = self._lifecycle.status()
        system = build_manager_system_prompt(
            vram_ceiling_mb=vram_status["vram_total_mb"],
            vram_used_mb=vram_status["vram_used_mb"],
            loaded_components=self._registry.loaded_summary(),
            registry_summary=self._registry.summary(),
            tool_descriptions="",
        )

        try:
            response = await self._llm.generate(query, system_prompt=system)
            parsed = json.loads(response)

            return ManagerAction(
                decision=ManagerDecision(parsed.get("decision", "use_existing")),
                components_to_load=parsed.get("components_to_load", []),
                components_to_unload=parsed.get("components_to_unload", []),
                components_to_download=parsed.get("components_to_download", []),
                reasoning=parsed.get("reasoning", ""),
            )

        except Exception as exc:
            log.warning("llm_decide_failed", error=str(exc))
            return ManagerAction(
                decision=ManagerDecision.USE_EXISTING,
                reasoning=f"LLM decision failed: {exc}",
            )

    # ── Pipeline execution ────────────────────────────────────────

    async def _execute_blueprint(
        self,
        blueprint: PipelineBlueprint,
        frame: np.ndarray,
    ) -> PipelineExecutionResult:
        """Execute all stages of a pipeline blueprint."""
        data: dict[str, Any] = {"frame": frame}
        stage_results: list[StageResult] = []
        success = True

        for stage in blueprint.stages:
            comp = self._registry.get(stage.component_id)
            if comp is None or comp.state != ComponentState.READY:
                if stage.is_optional:
                    continue
                stage_results.append(
                    StageResult(
                        stage_name=stage.stage_name,
                        component_id=stage.component_id,
                        success=False,
                        error=f"Component {stage.component_id} not ready",
                    )
                )
                success = False
                break

            t_stage = time.monotonic()
            try:
                input_data = data.get(stage.input_key, frame)
                ctx = {**(stage.config_overrides or {}), "_pipeline_data": data}

                output = await comp.execute(input_data, context=ctx)
                elapsed = (time.monotonic() - t_stage) * 1000

                if output is not None:
                    data[stage.output_key] = output

                stage_results.append(
                    StageResult(
                        stage_name=stage.stage_name,
                        component_id=stage.component_id,
                        output=output,
                        elapsed_ms=elapsed,
                        success=True,
                    )
                )

            except Exception as exc:
                elapsed = (time.monotonic() - t_stage) * 1000
                log.error(
                    "stage_execution_failed",
                    stage=stage.stage_name,
                    component=stage.component_id,
                    error=str(exc),
                )

                stage_results.append(
                    StageResult(
                        stage_name=stage.stage_name,
                        component_id=stage.component_id,
                        elapsed_ms=elapsed,
                        success=False,
                        error=str(exc),
                    )
                )

                if not stage.is_optional:
                    success = False
                    break

        return PipelineExecutionResult(
            blueprint_id=blueprint.blueprint_id,
            stage_results=stage_results,
            final_output=data,
            success=success,
        )

    # ── Helpers ────────────────────────────────────────────────────

    def _wire_ocr_engine(self, component: Any, name: str) -> None:
        """Add a provisioned OCR component to the builtin OCR strategy.

        When the Manager Agent provisions a new OCR model (PaddleOCR,
        TrOCR, etc.), this helper wraps it in ``ComponentOCREngine``
        and adds it to the ``OCRStrategy`` engine list so both the
        dynamic pipeline and the legacy fallback path can use it.
        """
        try:
            from uni_vision.ocr.llm_ocr import ComponentOCREngine

            ocr_comp = self._registry.get("builtin.ocr_strategy")
            if ocr_comp is None:
                return

            strategy = getattr(ocr_comp, "_ocr", None)
            if strategy is None or not hasattr(strategy, "add_engine"):
                return

            adapter = ComponentOCREngine(component, name=name)
            strategy.add_engine(adapter)
            log.info("ocr_engine_auto_wired", engine=name)
        except Exception as exc:
            log.warning("ocr_engine_wire_failed", engine=name, error=str(exc))

    def _get_cached_blueprint(
        self,
        context: FrameContext,
    ) -> PipelineBlueprint | None:
        """Check if a cached blueprint is still valid for this context."""
        cached = self._blueprint_cache.get(context.camera_id)
        if cached is None:
            return None

        # Blueprint is valid if it covers all required capabilities (both standard and dynamic)
        cached_caps = {s.required_capability for s in cached.stages if s.required_capability}
        cached_dyn = {s.dynamic_capability for s in cached.stages if s.dynamic_capability}

        if not context.required_capabilities.issubset(cached_caps):
            return None  # Missing standard capabilities

        dyn_required = getattr(context, "dynamic_required", frozenset())
        if dyn_required and not dyn_required.issubset(cached_dyn):
            return None  # Missing dynamic capabilities

        # Verify all referenced components are still loaded
        for stage in cached.stages:
            comp = self._registry.get(stage.component_id)
            if comp is None or comp.state != ComponentState.READY:
                if not stage.is_optional:
                    return None  # Need to rebuild

        return cached

    @staticmethod
    def _strip_failed_stages(
        blueprint: PipelineBlueprint,
        failed_ids: set,
    ) -> PipelineBlueprint:
        """Remove stages with failed component loads."""
        remaining = tuple(s for s in blueprint.stages if s.component_id not in failed_ids or s.is_optional)
        return PipelineBlueprint(
            blueprint_id=blueprint.blueprint_id,
            name=blueprint.name,
            context=blueprint.context,
            stages=remaining,
            estimated_vram_mb=blueprint.estimated_vram_mb,
        )

    def invalidate_cache(self, camera_id: str | None = None) -> None:
        """Clear blueprint cache (all or specific camera)."""
        if camera_id:
            self._blueprint_cache.pop(camera_id, None)
        else:
            self._blueprint_cache.clear()

    def status(self) -> dict:
        """Return full manager status including adaptive subsystems."""
        return {
            "registry_size": len(self._registry),
            "loaded_components": len(self._registry.get_loaded()),
            "cached_blueprints": len(self._blueprint_cache),
            "vram": self._lifecycle.status(),
            "temporal": self._temporal.status(),
            "gpu_profiler": self._gpu.status() if hasattr(self._gpu, "status") else {},
            "degraded_components": self._feedback.get_degraded_components(),
        }

    # ── Adaptive helpers ──────────────────────────────────────────

    async def _record_telemetry(
        self,
        result: PipelineExecutionResult,
        context: FrameContext,
        blueprint: PipelineBlueprint,
        camera_id: str,
    ) -> None:
        """Record per-stage telemetry to all feedback subsystems."""
        bp_hash = blueprint.blueprint_id
        for sr in result.stage_results:
            # Feedback loop — raw telemetry
            await self._feedback.record_result(sr, context, bp_hash)

            # Find matching stage to get capability info
            cap_str = ""
            cap_enum: ComponentCapability | None = None
            for stage in blueprint.stages:
                if stage.component_id == sr.component_id:
                    cap_enum = stage.required_capability
                    cap_str = stage.capability_label  # handles both enum and dynamic
                    break

            # Quality scorer — Bayesian update
            self._quality.record_execution(
                component_id=sr.component_id,
                capability=cap_str,
                latency_ms=sr.elapsed_ms,
                success=sr.success,
                confidence=getattr(sr, "confidence", 0.5),
                vram_mb=0.0,
            )

            # Fallback chain — record success/failure (uses enum key)
            if cap_enum is not None:
                if sr.success:
                    self._fallbacks.record_success(cap_enum, sr.component_id)
                else:
                    self._fallbacks.record_failure(cap_enum, sr.component_id)

            # Compatibility — record outcome if pair loaded
            if sr.success:
                for other in result.stage_results:
                    if other.component_id != sr.component_id and other.success:
                        self._compat.record_success(sr.component_id, other.component_id)

    async def _apply_adaptations(
        self,
        actions: list[AdaptationAction],
        camera_id: str,
    ) -> None:
        """Apply adaptation actions (swap, downgrade, recompose)."""
        for action in actions:
            if self._adaptation_count >= self._max_adaptations_per_frame:
                log.warning("adaptation_limit_reached", camera=camera_id)
                break

            if action.action_type == "swap_component":
                old_id = action.details.get("old_component_id", "")
                cap_str = action.details.get("capability", "")
                try:
                    cap_enum = ComponentCapability(cap_str)
                except ValueError:
                    log.warning("unknown_capability_in_swap", cap=cap_str)
                    continue
                fallback = self._fallbacks.get_next_fallback(
                    cap_enum,
                    exclude={old_id},
                )
                if fallback:
                    fid = fallback.component_id
                    # Provision fallback if not yet in registry
                    if fid not in self._registry:
                        try:
                            await self._resolver.provision_candidate(fallback)
                        except Exception as exc:
                            log.warning("swap_provision_failed", fallback=fid, error=str(exc))
                            continue

                    # Atomic swap: unload old, load new (with rollback)
                    swapped = await self._lifecycle.swap_component(old_id, fid)
                    if swapped:
                        log.info("adapting_swap_done", old=old_id, new=fid, cap=cap_str)
                        self.invalidate_cache(camera_id)
                        self._adaptation_count += 1
                    else:
                        log.warning("adapting_swap_failed", old=old_id, new=fid)

            elif action.action_type == "recompose":
                log.info("adapting_recompose", camera=camera_id)
                self.invalidate_cache(camera_id)
                self._adaptation_count += 1

            elif action.action_type == "downgrade":
                # Prefer a lighter component for VRAM pressure
                cap_str = action.details.get("capability", "")
                current_id = action.details.get("old_component_id", "")
                if cap_str:
                    ranked = self._quality.rank_by_capability(cap_str)
                    for cid, score in ranked:
                        if cid == current_id:
                            continue  # skip current component
                        if score.vram_score > 0.7:
                            load_ok = await self._lifecycle.ensure_loaded([cid])
                            if load_ok.get(cid, False):
                                if current_id:
                                    await self._lifecycle.unload_component(current_id)
                                log.info("adapting_downgrade_done", cap=cap_str, old=current_id, to=cid)
                                self.invalidate_cache(camera_id)
                                self._adaptation_count += 1
                            break

    async def _apply_fallbacks_for_failed(
        self,
        blueprint: PipelineBlueprint,
        failed_ids: set[str],
    ) -> PipelineBlueprint:
        """Replace failed stages with fallback components."""
        new_stages = []
        for stage in blueprint.stages:
            if stage.component_id in failed_ids:
                cap_enum = stage.required_capability
                fallback_cand = self._fallbacks.get_next_fallback(
                    cap_enum,
                    exclude=failed_ids,
                )
                if fallback_cand:
                    fid = fallback_cand.component_id
                    # Skip if this fallback already failed provisioning
                    if fid in self._provision_failed:
                        if not stage.is_optional:
                            new_stages.append(stage)
                        continue
                    log.info(
                        "fallback_swap",
                        original=stage.component_id,
                        fallback=fid,
                    )
                    # Provision fallback if not yet in registry
                    if fid not in self._registry:
                        try:
                            await self._resolver.provision_candidate(fallback_cand)
                        except Exception as exc:
                            log.warning("fallback_provision_failed", fallback=fid, error=str(exc))
                            self._provision_failed.add(fid)
                            if not stage.is_optional:
                                new_stages.append(stage)
                            continue

                    # Try loading the fallback
                    load_result = await self._lifecycle.ensure_loaded([fid])
                    if load_result.get(fid, False):
                        from uni_vision.manager.schemas import StageSpec

                        new_stages.append(
                            StageSpec(
                                stage_name=stage.stage_name,
                                required_capability=stage.required_capability,
                                component_id=fid,
                                is_optional=stage.is_optional,
                                input_key=stage.input_key,
                                output_key=stage.output_key,
                                config_overrides=stage.config_overrides,
                            )
                        )
                        continue
                # No fallback available — keep original (will fail) or skip if optional
                if not stage.is_optional:
                    new_stages.append(stage)
            else:
                new_stages.append(stage)

        return PipelineBlueprint(
            blueprint_id=blueprint.blueprint_id,
            name=blueprint.name,
            context=blueprint.context,
            stages=new_stages,
            estimated_vram_mb=blueprint.estimated_vram_mb,
        )

    async def _resolve_compat_issues(
        self,
        blueprint: PipelineBlueprint,
        issues: list,
    ) -> PipelineBlueprint:
        """Try to rebuild the blueprint to avoid compatibility issues."""
        # Collect problematic component IDs
        bad_ids: set[str] = set()
        for issue in issues:
            if hasattr(issue, "component_a"):
                bad_ids.add(issue.component_a)
            if hasattr(issue, "component_b"):
                bad_ids.add(issue.component_b)

        if not bad_ids:
            return blueprint

        log.info("resolving_compat_issues", problematic=list(bad_ids))
        # Invalidate and rebuild from scratch
        if blueprint.context:
            return await self._build_pipeline(blueprint.context, camera_id=None)
        return self._strip_failed_stages(blueprint, bad_ids)

    @staticmethod
    def _count_detections(result: PipelineExecutionResult) -> int:
        """Count total detections across all stage results."""
        count = 0
        for sr in result.stage_results:
            if sr.output and isinstance(sr.output, dict):
                count += len(sr.output.get("detections", []))
        return count

    @staticmethod
    def _avg_confidence(result: PipelineExecutionResult) -> float:
        """Average confidence across all stage results."""
        confs = []
        for sr in result.stage_results:
            if sr.output and isinstance(sr.output, dict):
                for det in sr.output.get("detections", []):
                    if isinstance(det, dict) and "confidence" in det:
                        confs.append(det["confidence"])
        return sum(confs) / max(len(confs), 1)
