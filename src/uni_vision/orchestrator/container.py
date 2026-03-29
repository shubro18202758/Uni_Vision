"""Dependency-injection container — assembles the live pipeline.

Provides a single factory function ``build_pipeline()`` that reads the
validated ``AppConfig``, instantiates every concrete stage
implementation, and returns a fully wired ``Pipeline`` ready to be
started.

Wiring order (respects dependency graph):
  1. VRAMMonitor          — no dependencies
  2. VehicleDetector       — config only
  3. PlateDetector         — config only
  4. HoughStraightener     — DeskewConfig
  5. PhotometricEnhancer   — EnhanceConfig
  6. EasyOCR primary       — FallbackOCRConfig (default engine)
  7. OCRStrategy            — multi-engine list (Manager adds more at runtime)
  8. ConsensusAdjudicator  — OCRStrategy + AdjudicationConfig
  9. CognitiveOrchestrator — ValidationConfig + Adjudicator + AdjudicationConfig
 10. MultiDispatcher        — DB + Storage + Dispatch + Dedup configs
 11. Manager Agent subsystem (optional, when manager.enabled)
 12. Pipeline              — all of the above

This module is the **only** place that imports concrete implementations.
The rest of the codebase programs against Protocol contracts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from uni_vision.common.config import AppConfig, load_config
from uni_vision.detection.plate_detector import PlateDetector
from uni_vision.detection.vehicle_detector import VehicleDetector
from uni_vision.monitoring.vram_monitor import VRAMMonitor
from uni_vision.ocr.fallback_ocr import EasyOCRFallback
from uni_vision.ocr.strategy import OCRStrategy
from uni_vision.orchestrator.pipeline import Pipeline
from uni_vision.postprocessing.adjudicator import ConsensusAdjudicator
from uni_vision.postprocessing.dispatcher import MultiDispatcher
from uni_vision.postprocessing.orchestrator import CognitiveOrchestrator
from uni_vision.preprocessing.deskew import HoughStraightener
from uni_vision.preprocessing.enhance import PhotometricEnhancer



logger = logging.getLogger(__name__)


def build_pipeline(config: Optional[AppConfig] = None) -> Pipeline:
    """Assemble all concrete stage implementations into a live Pipeline.

    Parameters
    ----------
    config:
        Pre-loaded application config.  If *None*, ``load_config()``
        is called to read YAML files and env overrides.

    Returns
    -------
    Pipeline
        A fully wired, ready-to-start pipeline instance.
    """
    if config is None:
        config = load_config()

    # ── 1. VRAM Monitor ───────────────────────────────────────────
    vram_monitor = VRAMMonitor(
        device_index=config.hardware.cuda_device_index,
        vram_ceiling_mb=config.hardware.vram_ceiling_mb,
        safety_margin_mb=config.hardware.vram_safety_margin_mb,
        poll_interval_ms=config.hardware.vram_poll_interval_ms,
        region_budgets={
            "llm_weights": config.vram_budgets.llm_weights_mb,
            "kv_cache": config.vram_budgets.kv_cache_mb,
            "vision_workspace": config.vram_budgets.vision_workspace_mb,
            "system_overhead": config.vram_budgets.system_overhead_mb,
        },
    )

    # ── 2. Vehicle Detector (YOLOv8n — Region C) ─────────────────
    vehicle_model_cfg = config.models.get("vehicle_detector")
    vehicle_detector = VehicleDetector(
        model_path=vehicle_model_cfg.model_path if vehicle_model_cfg else "models/vehicle_detector.engine",
        model_format=vehicle_model_cfg.model_format if vehicle_model_cfg else "onnx",
        input_size=tuple(vehicle_model_cfg.input_size) if vehicle_model_cfg else (640, 640),
        confidence_threshold=vehicle_model_cfg.confidence_threshold if vehicle_model_cfg else 0.60,
        nms_iou_threshold=vehicle_model_cfg.nms_iou_threshold if vehicle_model_cfg else 0.45,
    )

    # ── 3. Plate Detector (YOLOv8n — Region C) ───────────────────
    plate_model_cfg = config.models.get("plate_detector")
    plate_detector = PlateDetector(
        model_path=plate_model_cfg.model_path if plate_model_cfg else "models/plate_detector.engine",
        model_format=plate_model_cfg.model_format if plate_model_cfg else "onnx",
        input_size=tuple(plate_model_cfg.input_size) if plate_model_cfg else (640, 640),
        confidence_threshold=plate_model_cfg.confidence_threshold if plate_model_cfg else 0.60,
        nms_iou_threshold=plate_model_cfg.nms_iou_threshold if plate_model_cfg else 0.45,
        multi_plate_policy=(
            plate_model_cfg.multi_plate_policy if plate_model_cfg else "highest_confidence"
        ),
    )

    # ── 4. Geometric Correction (CPU — no VRAM) ──────────────────
    straightener = HoughStraightener(config=config.preprocessing.deskew)

    # ── 5. Photometric Enhancement (CPU — no VRAM) ───────────────
    enhancer = PhotometricEnhancer(config=config.preprocessing.enhance)

    # ── 6. Primary OCR — EasyOCR (CPU only, default engine) ────────
    primary_ocr = EasyOCRFallback(config=config.fallback_ocr)

    # ── 7. OCR Strategy (multi-engine, Manager adds more at runtime) ─
    ocr_strategy = OCRStrategy(engines=[primary_ocr])

    # ── 8. Consensus Adjudicator (multi-engine voting for Layer 2) ─
    adjudicator = ConsensusAdjudicator(
        ocr_strategy=ocr_strategy,
        adj_config=config.adjudication,
    )

    # ── 9. Cognitive Orchestrator (deterministic + consensus) ────
    validator = CognitiveOrchestrator(
        validation_config=config.validation,
        adjudicator=adjudicator,
        adjudication_config=config.adjudication,
    )

    # ── 10. Multi-target Dispatcher (Postgres + S3 + dedup) ──────
    dispatcher = MultiDispatcher(
        db_config=config.database,
        storage_config=config.storage,
        dispatch_config=config.dispatch,
        dedup_config=config.deduplication,
    )

    # ── 11. Manager Agent subsystem (dynamic pipeline) ──────────
    manager_agent = None

    if config.manager.enabled:
        manager_agent = _build_manager_agent(
            config=config,
            ocr_strategy=ocr_strategy,
            vehicle_detector=vehicle_detector,
            plate_detector=plate_detector,
            straightener=straightener,
            enhancer=enhancer,
            validator=validator,
            dispatcher=dispatcher,
        )
        logger.info("manager_agent_assembled")

    # ── 12. Assemble Pipeline ─────────────────────────────────────
    pipeline = Pipeline(
        config=config,
        vram_monitor=vram_monitor,
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        straightener=straightener,
        enhancer=enhancer,
        ocr_strategy=ocr_strategy,
        validator=validator,
        dispatcher=dispatcher,
        manager_agent=manager_agent,
    )

    logger.info(
        "container_assembled stages=[VehicleDetector, PlateDetector, "
        "HoughStraightener, PhotometricEnhancer, OCRStrategy("
        "EasyOCR+dynamic), ConsensusAdjudicator, CognitiveOrchestrator, MultiDispatcher]"
        + (", ManagerAgent" if manager_agent else "")
    )

    return pipeline


def _build_manager_agent(
    *,
    config: AppConfig,
    ocr_strategy: object,
    vehicle_detector: object,
    plate_detector: object,
    straightener: object,
    enhancer: object,
    validator: object,
    dispatcher: object,
) -> object:
    """Assemble all Manager Agent subsystems and register builtin components.

    Imports are deferred to keep the container lightweight when the
    manager is disabled.
    """
    from uni_vision.agent.llm_client import AgentLLMClient
    from uni_vision.components.base import ComponentCapability, ComponentType, ResourceEstimate, ComponentMetadata
    from uni_vision.components.wrappers import (
        BuiltinDetectorComponent,
        BuiltinOCRComponent,
        BuiltinPreprocessorComponent,
        BuiltinPostprocessorComponent,
    )
    from uni_vision.manager.adaptation_engine import AdaptationEngine
    from uni_vision.manager.agent import ManagerAgent
    from uni_vision.manager.compatibility import CompatibilityMatrix
    from uni_vision.manager.component_registry import ComponentRegistry
    from uni_vision.manager.component_resolver import ComponentResolver
    from uni_vision.manager.conflict_resolver import ConflictResolver
    from uni_vision.manager.context_analyzer import ContextAnalyzer
    from uni_vision.manager.fallback_chain import FallbackChainManager
    from uni_vision.manager.feedback_loop import FeedbackLoop
    from uni_vision.manager.gpu_profiler import GPUProfiler
    from uni_vision.manager.hub_client import HubClient
    from uni_vision.manager.lifecycle import LifecycleManager
    from uni_vision.manager.pipeline_composer import PipelineComposer
    from uni_vision.manager.pipeline_validator import PipelineValidator
    from uni_vision.manager.quality_scorer import QualityScorer
    from uni_vision.manager.scene_detector import SceneTransitionDetector
    from uni_vision.manager.temporal_tracker import TemporalTracker

    mgr_cfg = config.manager

    # 1. Component Registry
    registry = ComponentRegistry()

    # 2. Register builtin ANPR components as CVComponent wrappers
    builtin_vehicle_det = BuiltinDetectorComponent(
        component_id="builtin.vehicle_detector",
        name="YOLOv8n Vehicle Detector",
        capabilities=frozenset({ComponentCapability.VEHICLE_DETECTION}),
        detector_instance=vehicle_detector,
        vram_mb=200,
    )
    builtin_plate_det = BuiltinDetectorComponent(
        component_id="builtin.plate_detector",
        name="YOLOv8n Plate Detector",
        capabilities=frozenset({
            ComponentCapability.PLATE_DETECTION,
            ComponentCapability.PLATE_LOCALIZATION,
        }),
        detector_instance=plate_detector,
        vram_mb=200,
    )
    builtin_ocr = BuiltinOCRComponent(
        ocr_strategy_instance=ocr_strategy,
    )
    builtin_straightener = BuiltinPreprocessorComponent(
        component_id="builtin.straightener",
        name="Hough Straightener",
        capabilities=frozenset({ComponentCapability.GEOMETRIC_CORRECTION}),
        preprocessor_instance=straightener,
    )
    builtin_enhancer = BuiltinPreprocessorComponent(
        component_id="builtin.enhancer",
        name="Photometric Enhancer",
        capabilities=frozenset({
            ComponentCapability.IMAGE_ENHANCE,
            ComponentCapability.SUPER_RESOLUTION,
        }),
        preprocessor_instance=enhancer,
    )
    builtin_validator = BuiltinPostprocessorComponent(
        component_id="builtin.cognitive_validator",
        name="Cognitive Orchestrator",
        capabilities=frozenset({ComponentCapability.PLATE_VALIDATION}),
        postprocessor_instance=validator,
    )

    for comp in (
        builtin_vehicle_det,
        builtin_plate_det,
        builtin_ocr,
        builtin_straightener,
        builtin_enhancer,
        builtin_validator,
    ):
        registry.register(comp)

    # 3. HuggingFace Hub client
    hub_client = HubClient(
        cache_dir=Path(mgr_cfg.hub_cache_dir).expanduser(),
        http_timeout=mgr_cfg.http_timeout_s,
        max_search_results=mgr_cfg.max_search_results,
    )

    # 3b. GPU Profiler (needed by LifecycleManager — no deps)
    gpu_profiler = GPUProfiler(
        device_index=config.hardware.cuda_device_index,
    )

    # 4. Lifecycle Manager (with GPU profiler for real VRAM measurement)
    lifecycle = LifecycleManager(
        registry,
        vram_total_mb=mgr_cfg.vram_total_mb,
        vram_reserved_mb=mgr_cfg.vram_reserved_for_llm_mb,
        device=f"cuda:{config.hardware.cuda_device_index}",
        gpu_profiler=gpu_profiler,
    )

    # 5. Component Resolver (with lifecycle for auto-loading after provision)
    resolver = ComponentResolver(
        registry=registry,
        hub_client=hub_client,
        vram_limit_mb=lifecycle.vram_budget_mb,
        prefer_trusted=mgr_cfg.prefer_trusted,
        lifecycle=lifecycle,
    )

    # 6. Conflict Resolver
    conflict_resolver = ConflictResolver(
        registry=registry,
        vram_limit_mb=lifecycle.vram_budget_mb,
    )

    # 7. Context Analyzer (uses LLM for deep analysis)
    #    The manager modules expect .generate(prompt, system_prompt=)
    #    while AgentLLMClient exposes .chat(messages).  Bridge the gap.
    _raw_llm = AgentLLMClient(config.ollama, timeout_s=mgr_cfg.http_timeout_s)

    class _LLMAdapter:
        """Thin adapter: generate(prompt, system_prompt) → chat(messages)."""

        def __init__(self, client: AgentLLMClient) -> None:
            self._c = client

        async def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
        ) -> str:
            msgs: list[dict[str, str]] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt})
            resp = await self._c.chat(msgs)
            return resp.content

    llm_client = _LLMAdapter(_raw_llm)
    from uni_vision.manager.schemas import SceneType as _SceneType
    context_analyzer = ContextAnalyzer(
        llm_client=llm_client,
        default_scene=_SceneType(mgr_cfg.default_scene),
    )

    # 8. Pipeline Composer
    composer = PipelineComposer(registry=registry)

    # 9. Pipeline Validator
    pipeline_validator = PipelineValidator(
        registry=registry,
        vram_budget_mb=lifecycle.vram_budget_mb,
    )

    # 10. Adaptive subsystems
    adaptation_engine = AdaptationEngine(
        latency_threshold_ms=getattr(mgr_cfg, "latency_threshold_ms", 200),
        vram_pressure_threshold=getattr(mgr_cfg, "vram_pressure_threshold", 0.85),
    )
    feedback_loop = FeedbackLoop()
    fallback_manager = FallbackChainManager(
        max_consecutive_failures=getattr(mgr_cfg, "max_consecutive_failures", 5),
    )
    quality_scorer = QualityScorer(
        latency_budget_ms=getattr(mgr_cfg, "latency_budget_ms", 100.0),
        vram_budget_mb=lifecycle.vram_budget_mb,
    )
    scene_detector = SceneTransitionDetector()
    # gpu_profiler already created above (step 3b) before lifecycle
    compat_matrix = CompatibilityMatrix()
    temporal_tracker = TemporalTracker()

    # 11. Manager Agent
    agent = ManagerAgent(
        llm_client=llm_client,
        registry=registry,
        hub_client=hub_client,
        resolver=resolver,
        conflict_resolver=conflict_resolver,
        context_analyzer=context_analyzer,
        composer=composer,
        lifecycle=lifecycle,
        validator=pipeline_validator,
        adaptation_engine=adaptation_engine,
        feedback_loop=feedback_loop,
        fallback_manager=fallback_manager,
        quality_scorer=quality_scorer,
        scene_detector=scene_detector,
        gpu_profiler=gpu_profiler,
        compat_matrix=compat_matrix,
        temporal_tracker=temporal_tracker,
    )

    return agent
