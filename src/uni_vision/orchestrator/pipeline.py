"""Main asynchronous pipeline loop — spec §3, §6, §7.

Implements the S0→S5 sequential processing chain with:
  * Two-layer bounded queue architecture (stream queues + inference queue)
  * Single-consumer GPU inference (sequential exclusivity — P3)
  * LLM vision analysis for multipurpose anomaly detection
  * Adaptive FPS throttling on queue backpressure (§6.4)
  * Graceful degradation via OffloadMode awareness (§7.2)
  * Structured observability for every stage

This module is the application entry-point (``main``).
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from typing import List, Optional

import structlog

from uni_vision.common.config import AppConfig, load_config
from uni_vision.common.exceptions import (
    PipelineShutdownError,
    UniVisionError,
)
from uni_vision.common.logging import configure_logging
from uni_vision.contracts.dtos import (
    DetectionRecord,
    FramePacket,
    OffloadMode,
)
from uni_vision.monitoring.metrics import (
    DETECTIONS_TOTAL,
    FRAMES_DROPPED,
    INFERENCE_QUEUE_DEPTH,
    PIPELINE_LATENCY,
    STAGE_LATENCY,
)
from uni_vision.monitoring.profiler import (
    PipelineTelemetryHook,
    profile_stage,
    set_profiling_enabled,
    vram_sampler,
)
from uni_vision.monitoring.vram_budget import validate_budget
from uni_vision.monitoring.vram_monitor import VRAMMonitor
from uni_vision.orchestrator.pipeline_events import pipeline_broadcaster

# Type-only import to avoid circular dependency at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uni_vision.manager.agent import ManagerAgent

log = structlog.get_logger()


class Pipeline:
    """Asynchronous pipeline controller.

    All stage dependencies are injected via protocol-typed references
    (see ``contracts/``).  Concrete implementations are wired in
    ``orchestrator/container.py`` — this class knows nothing about
    YOLOv8, Ollama, or PostgreSQL directly.

    Parameters follow the contract interfaces; type hints are omitted
    here to avoid circular imports at module level.  The container
    enforces type safety through ``runtime_checkable`` Protocol checks.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        vram_monitor: VRAMMonitor,
        # Protocol-typed stage references (injected by container)
        vehicle_detector: object,
        plate_detector: object,
        straightener: object,
        enhancer: object,
        ocr_strategy: object,
        validator: object,
        dispatcher: object,
        # Manager Agent for dynamic pipeline composition (optional)
        manager_agent: Optional["ManagerAgent"] = None,
    ) -> None:
        self._config = config
        self._vram_monitor = vram_monitor

        self._vehicle_detector = vehicle_detector
        self._plate_detector = plate_detector
        self._straightener = straightener
        self._enhancer = enhancer
        self._ocr_strategy = ocr_strategy
        self._validator = validator
        self._dispatcher = dispatcher
        self._manager_agent = manager_agent
        self._vision_analyzer = None  # Lazy-init in _process_event

        # Layer 2 — single-consumer inference queue (spec §6.3)
        self._inference_queue: asyncio.Queue[FramePacket] = asyncio.Queue(
            maxsize=config.pipeline.inference_queue_maxsize,
        )
        self._high_water = config.pipeline.inference_queue_high_water
        self._low_water = config.pipeline.inference_queue_low_water

        self._shutting_down = False
        self._throttled = False

        # Profiling & telemetry (§7.3, §12)
        self._telemetry = PipelineTelemetryHook(
            device_index=config.hardware.cuda_device_index,
        )

    # ── Public lifecycle ──────────────────────────────────────────

    async def start(self) -> None:
        """Warm up models, start the VRAM monitor, and begin processing."""
        log.info("pipeline_starting")

        # Initialise profiling subsystem from config
        set_profiling_enabled(self._config.profiling.enabled)

        # Validate VRAM budget before allocating any GPU memory
        if self._config.profiling.validate_vram_budget_on_start:
            budget = validate_budget()
            log.info(
                "vram_budget_validated",
                total_mb=budget.total_allocated_mb,
                headroom_mb=budget.headroom_mb,
                context_tokens=budget.context_window_tokens,
            )

        # Warm up vision models (loads weights onto GPU Region C)
        await self._warmup_models()

        # Start VRAM telemetry as a background task
        asyncio.create_task(self._vram_monitor.run())

        # Start the single inference consumer
        asyncio.create_task(self._inference_consumer())

        log.info("pipeline_ready")

    async def enqueue_frame(self, frame: FramePacket) -> bool:
        """Submit a frame to the inference queue (called by stream dispatchers).

        Returns ``True`` if the frame was accepted, ``False`` if dropped
        due to backpressure.
        """
        # Adaptive throttle check — spec §6.4
        depth = self._inference_queue.qsize()
        INFERENCE_QUEUE_DEPTH.set(depth)

        if depth >= self._high_water and not self._throttled:
            self._throttled = True
            log.warning("throttle_engaged", queue_depth=depth)

        if self._inference_queue.full():
            FRAMES_DROPPED.labels(camera_id=frame.camera_id).inc()
            log.debug("frame_dropped", camera_id=frame.camera_id, queue_depth=depth)
            return False

        await self._inference_queue.put(frame)
        return True

    async def shutdown(self) -> None:
        """Initiate graceful shutdown — drain queue, release resources."""
        self._shutting_down = True
        log.info("pipeline_shutdown_initiated")

        await self._vram_monitor.shutdown()

        # Drain remaining items with a timeout
        try:
            await asyncio.wait_for(self._inference_queue.join(), timeout=10.0)
        except asyncio.TimeoutError:
            log.warning("pipeline_drain_timeout", remaining=self._inference_queue.qsize())

        await self._release_models()
        log.info("pipeline_shutdown_complete")

    # ── Inference consumer ────────────────────────────────────────

    async def _inference_consumer(self) -> None:
        """Single-consumer loop — processes one detection event at a time.

        Sequential execution ensures GPU single-tenancy (P3).
        """
        log.info("inference_consumer_started")
        while not self._shutting_down:
            try:
                frame = await asyncio.wait_for(
                    self._inference_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            # Un-throttle when queue drains below low-water mark
            depth = self._inference_queue.qsize()
            INFERENCE_QUEUE_DEPTH.set(depth)
            if depth <= self._low_water and self._throttled:
                self._throttled = False
                log.info("throttle_released", queue_depth=depth)

            t0 = time.perf_counter()
            try:
                # Assign a unique frame ID for pipeline visibility tracking
                _frame_id = pipeline_broadcaster.generate_frame_id()
                log.info(
                    "consumer_processing_frame",
                    frame_id=_frame_id,
                    camera_id=frame.camera_id,
                    queue_depth=depth,
                )
                await pipeline_broadcaster.emit_frame_accepted(
                    frame_id=_frame_id,
                    camera_id=frame.camera_id,
                    frame_index=getattr(frame, "frame_index", 0),
                    queue_depth=depth,
                    image=frame.image,
                )

                # Route to dynamic pipeline if Manager Agent is wired
                if self._manager_agent is not None:
                    await self._process_event_dynamic(frame, _frame_id)
                else:
                    await self._process_event(frame, _frame_id)
            except UniVisionError as exc:
                log.error(
                    "pipeline_event_error",
                    error=str(exc),
                    camera_id=frame.camera_id,
                    exc_info=True,
                )
            except Exception as exc:
                log.error(
                    "pipeline_event_unexpected_error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    camera_id=frame.camera_id,
                    exc_info=True,
                )
            finally:
                elapsed = time.perf_counter() - t0
                PIPELINE_LATENCY.observe(elapsed)
                self._telemetry.end_event(elapsed_ms=elapsed * 1000.0)
                self._inference_queue.task_done()
        log.info("inference_consumer_stopped")

    # ── Per-event processing — LLM Vision Analysis ─────────────────

    async def _process_event(self, frame: FramePacket, frame_id: str = "unknown") -> None:
        """Execute anomaly analysis pipeline for a single frame.

        Uses LLM vision to analyze frames for anomalies, producing
        structured results with chain-of-thought reasoning, risk
        assessment, and impact analysis.
        """
        from dataclasses import asdict

        camera_id = frame.camera_id
        self._telemetry.begin_event(camera_id=camera_id)
        pipeline_start = time.perf_counter()

        # S1 — Preprocessing
        await pipeline_broadcaster.emit_stage_started(frame_id, camera_id, "S1_preprocess")
        t = time.perf_counter()
        processed_image = frame.image
        try:
            processed_image = self._enhancer.process(frame.image)  # type: ignore[union-attr]
        except Exception:
            processed_image = frame.image  # use original on failure
        s1_ms = (time.perf_counter() - t) * 1000
        STAGE_LATENCY.labels(stage="S1_preprocess").observe(s1_ms / 1000)
        await pipeline_broadcaster.emit_stage_completed(
            frame_id, camera_id, "S1_preprocess", s1_ms,
            details={"enhanced": processed_image is not frame.image},
        )

        # S2 — Scene Analysis (LLM Vision)
        await pipeline_broadcaster.emit_stage_started(frame_id, camera_id, "S2_scene_analysis")
        t = time.perf_counter()

        if self._vision_analyzer is None:
            from uni_vision.detection.vision_analyzer import VisionAnalyzer
            self._vision_analyzer = VisionAnalyzer(
                ollama_base_url=self._config.ollama.base_url,
                model=self._config.ollama.model,
                timeout_s=self._config.ollama.timeout_s,
                num_predict=1024,
                temperature=0.15,
            )

        try:
            analysis = await self._vision_analyzer.analyze_frame(
                processed_image,
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp_utc=str(frame.timestamp_utc),
            )
        except Exception as exc:
            log.error("vision_analysis_failed camera=%s frame=%s error=%s", camera_id, frame_id, exc)
            from uni_vision.contracts.dtos import AnalysisResult
            analysis = AnalysisResult(
                frame_id=frame_id,
                camera_id=camera_id,
                timestamp_utc=str(frame.timestamp_utc),
                scene_description=f"Analysis failed: {exc}",
                anomaly_detected=False,
                risk_level="low",
                confidence=0.0,
            )
        s2_ms = (time.perf_counter() - t) * 1000
        STAGE_LATENCY.labels(stage="S2_scene_analysis").observe(s2_ms / 1000)
        await pipeline_broadcaster.emit_stage_completed(
            frame_id, camera_id, "S2_scene_analysis", s2_ms,
            details={
                "objects_count": len(analysis.objects_detected),
                "scene": analysis.scene_description[:80] if analysis.scene_description else "",
            },
        )

        # S3 — Anomaly Detection
        await pipeline_broadcaster.emit_stage_started(frame_id, camera_id, "S3_anomaly_detection")
        t = time.perf_counter()
        anomaly_count = len(analysis.anomalies)
        s3_ms = (time.perf_counter() - t) * 1000
        STAGE_LATENCY.labels(stage="S3_anomaly_detection").observe(s3_ms / 1000)
        await pipeline_broadcaster.emit_stage_completed(
            frame_id, camera_id, "S3_anomaly_detection", s3_ms,
            details={
                "anomaly_detected": analysis.anomaly_detected,
                "anomalies_count": anomaly_count,
                "risk_level": analysis.risk_level,
            },
        )

        # S4 — Deep Analysis (chain-of-thought, risk, impact)
        await pipeline_broadcaster.emit_stage_started(frame_id, camera_id, "S4_deep_analysis")
        t = time.perf_counter()
        s4_ms = (time.perf_counter() - t) * 1000
        STAGE_LATENCY.labels(stage="S4_deep_analysis").observe(s4_ms / 1000)
        await pipeline_broadcaster.emit_stage_completed(
            frame_id, camera_id, "S4_deep_analysis", s4_ms,
            details={
                "confidence": round(analysis.confidence, 3),
                "chain_of_thought": analysis.chain_of_thought[:120] if analysis.chain_of_thought else "",
                "risk_analysis": analysis.risk_analysis[:120] if analysis.risk_analysis else "",
                "impact_analysis": analysis.impact_analysis[:120] if analysis.impact_analysis else "",
            },
        )

        # S5 — Results & Dispatch
        await pipeline_broadcaster.emit_stage_started(frame_id, camera_id, "S5_results")
        t = time.perf_counter()

        # Broadcast full analysis result to subscribers
        analysis_dict = asdict(analysis)
        await pipeline_broadcaster.emit_analysis_result(
            frame_id, camera_id, analysis_dict, image=frame.image,
        )

        # Raise flag if anomaly detected
        if analysis.anomaly_detected:
            DETECTIONS_TOTAL.labels(camera_id=camera_id).inc()
            await pipeline_broadcaster.emit_flag_raised(
                frame_id=frame_id,
                camera_id=camera_id,
                detection=analysis_dict,
                validation_status=analysis.risk_level,
            )

        s5_ms = (time.perf_counter() - t) * 1000
        STAGE_LATENCY.labels(stage="S5_results").observe(s5_ms / 1000)
        await pipeline_broadcaster.emit_stage_completed(
            frame_id, camera_id, "S5_results", s5_ms,
            details={
                "anomaly_detected": analysis.anomaly_detected,
                "dispatched": True,
                "recommendations": len(analysis.recommendations),
            },
        )

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        await pipeline_broadcaster.emit_pipeline_complete(
            frame_id, camera_id, total_ms,
            1 if analysis.anomaly_detected else 0,
        )

        log.info(
            "analysis_complete",
            camera_id=camera_id,
            frame_id=frame_id,
            anomaly=analysis.anomaly_detected,
            risk=analysis.risk_level,
            confidence=analysis.confidence,
            objects=len(analysis.objects_detected),
            total_ms=round(total_ms, 1),
        )

    # ── OCR invocation with circuit-breaker fallback ──────────────

    async def _invoke_ocr(self, plate_image: object, context: object) -> object:
        """Delegate to the OCR strategy (handles primary/fallback routing)."""
        return await self._ocr_strategy.extract(plate_image, context)  # type: ignore[union-attr]

    # ── Dynamic pipeline processing (Manager Agent) ───────────────

    async def _process_event_dynamic(self, frame: FramePacket, frame_id: str = "unknown") -> None:
        """Execute a frame through the Manager Agent's dynamic pipeline.

        Instead of the hardcoded S2→S8 stage sequence, the Manager
        Agent analyses the frame context, discovers and composes the
        optimal set of CV components, and executes them dynamically.

        Falls back to the legacy ``_process_event`` when the Manager
        Agent is unavailable or the dynamic pipeline fails.
        """
        assert self._manager_agent is not None
        camera_id = frame.camera_id

        try:
            import numpy as np

            # Emit synthetic stage events so the frontend can track progress
            await pipeline_broadcaster.emit_stage_started(frame_id, camera_id, "S1_preprocess")
            await pipeline_broadcaster.emit_stage_completed(
                frame_id, camera_id, "S1_preprocess", 0.0,
                details={"enhanced": False, "dynamic": True},
            )

            await pipeline_broadcaster.emit_stage_started(frame_id, camera_id, "S2_scene_analysis")
            t0 = time.perf_counter()

            result = await self._manager_agent.process_frame(
                frame.image if isinstance(frame.image, np.ndarray) else frame.image,
                camera_id=camera_id,
                metadata={"timestamp_utc": str(frame.timestamp_utc)},
            )

            agent_ms = (time.perf_counter() - t0) * 1000

            if result.success:
                await pipeline_broadcaster.emit_stage_completed(
                    frame_id, camera_id, "S2_scene_analysis", agent_ms,
                    details={"dynamic": True, "stages_run": len(result.stage_results)},
                )

                # S3–S5 completed atomically by the dynamic agent
                for stage in ("S3_anomaly_detection", "S4_deep_analysis", "S5_results"):
                    await pipeline_broadcaster.emit_stage_started(frame_id, camera_id, stage)
                    await pipeline_broadcaster.emit_stage_completed(
                        frame_id, camera_id, stage, 0.0,
                        details={"dynamic": True},
                    )

                # Bridge dynamic pipeline results → DetectionRecord dispatch
                await self._dispatch_dynamic_results(result.final_output, frame)

                total_ms = (time.perf_counter() - t0) * 1000
                await pipeline_broadcaster.emit_pipeline_complete(
                    frame_id, camera_id, total_ms,
                    1 if (result.final_output or {}).get("anomaly_detected") else 0,
                )

                log.info(
                    "dynamic_detection_complete",
                    camera_id=camera_id,
                    stages_run=len(result.stage_results),
                    total_ms=result.total_elapsed_ms,
                    success=True,
                )
            else:
                await pipeline_broadcaster.emit_stage_completed(
                    frame_id, camera_id, "S2_scene_analysis", agent_ms,
                    details={"dynamic": True, "degraded": True},
                )
                log.warning(
                    "dynamic_pipeline_degraded",
                    camera_id=camera_id,
                    stages_run=len(result.stage_results),
                    error=str(result.error) if result.error else None,
                )
                # Fallback to legacy pipeline
                await self._process_event(frame, frame_id)

        except Exception as exc:
            log.error(
                "dynamic_pipeline_error",
                camera_id=camera_id,
                error=str(exc),
                exc_info=True,
            )
            # Fallback to legacy pipeline on any dynamic failure
            await self._process_event(frame, frame_id)

    async def _dispatch_dynamic_results(
        self,
        final_output: object,
        frame: FramePacket,
    ) -> None:
        """Extract analysis results from dynamic pipeline output and dispatch.

        ``final_output`` is the data dict accumulated by
        ``ManagerAgent._execute_blueprint``.  Handles both generic
        analysis results and legacy plate detection outputs.
        """
        from dataclasses import asdict

        if not isinstance(final_output, dict):
            return

        # Check for generic analysis results first
        analysis_data = final_output.get("analysis_result")
        if analysis_data and isinstance(analysis_data, dict):
            await pipeline_broadcaster.emit_analysis_result(
                frame_id=str(id(frame)),
                camera_id=frame.camera_id,
                analysis=analysis_data,
                image=frame.image,
            )
            if analysis_data.get("anomaly_detected"):
                DETECTIONS_TOTAL.labels(camera_id=frame.camera_id).inc()
                await pipeline_broadcaster.emit_flag_raised(
                    frame_id=str(id(frame)),
                    camera_id=frame.camera_id,
                    detection=analysis_data,
                    validation_status=analysis_data.get("risk_level", "unknown"),
                )
            log.info(
                "dynamic_analysis_dispatched",
                camera_id=frame.camera_id,
                anomaly=analysis_data.get("anomaly_detected", False),
                risk=analysis_data.get("risk_level", "unknown"),
            )
            return

        # Legacy: plate detection results
        plate_texts: list = final_output.get("plate_texts", [])
        plate_crops: list = final_output.get("plate_crops", [])

        if not plate_texts:
            log.debug("dynamic_no_results", camera_id=frame.camera_id)
            return

        for idx, pt in enumerate(plate_texts):
            if not isinstance(pt, dict):
                continue

            plate_text = pt.get("plate_text", "")
            if not plate_text:
                continue

            plate_image = None
            if idx < len(plate_crops) and isinstance(plate_crops[idx], dict):
                plate_image = plate_crops[idx].get("crop")

            record = DetectionRecord(
                camera_id=frame.camera_id,
                plate_number=plate_text,
                raw_ocr_text=pt.get("raw_ocr_text", plate_text),
                ocr_confidence=float(pt.get("confidence", 0.0)),
                ocr_engine=pt.get("engine", "unknown"),
                vehicle_class=pt.get("vehicle_class", "unknown"),
                detected_at_utc=str(frame.timestamp_utc),
                validation_status="dynamic",
            )

            await self._dispatcher.dispatch(record, plate_image=plate_image)  # type: ignore[union-attr]
            DETECTIONS_TOTAL.labels(camera_id=frame.camera_id).inc()

            log.info(
                "dynamic_plate_dispatched",
                camera_id=frame.camera_id,
                plate=plate_text,
                confidence=pt.get("confidence", 0.0),
                engine=pt.get("engine", "unknown"),
            )

    # ── Model lifecycle helpers ───────────────────────────────────

    async def _warmup_models(self) -> None:
        """Load and warm up detector models (allocates GPU Region C).

        If model files are missing the pipeline starts in **degraded** mode
        (no CV inference) but the API / agent / Navarasa remain functional.
        """
        log.info("model_warmup_start")
        try:
            self._vehicle_detector.warmup()  # type: ignore[union-attr]
            self._plate_detector.warmup()  # type: ignore[union-attr]
        except Exception as exc:
            log.warning(
                "model_warmup_failed_degraded_mode",
                error=str(exc),
            )
            return  # continue in degraded mode
        log.info("model_warmup_complete")

    async def _release_models(self) -> None:
        """Release all model resources (free GPU VRAM)."""
        for model in (self._vehicle_detector, self._plate_detector):
            try:
                model.release()  # type: ignore[union-attr]
            except Exception as exc:
                log.error("model_release_error", error=str(exc))


# ── Application entry-point ───────────────────────────────────────


async def _async_main() -> None:
    """Bootstrap configuration, logging, monitoring, and the pipeline."""
    from uni_vision.orchestrator.container import build_pipeline

    config = load_config()
    configure_logging(level=config.logging.log_level, fmt=config.logging.log_format)

    # Initialise profiling subsystem early
    set_profiling_enabled(config.profiling.enabled)

    # Validate VRAM budget before any GPU allocation
    if config.profiling.validate_vram_budget_on_start:
        budget = validate_budget()
        log.info(
            "vram_budget_preflight",
            total_mb=budget.total_allocated_mb,
            headroom_mb=budget.headroom_mb,
            fits=budget.fits,
        )

    # Assemble and start the full pipeline via DI container
    pipeline = build_pipeline(config)
    await pipeline.start()

    # Graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        log.info("shutdown_signal_received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows does not support add_signal_handler for SIGTERM
            signal.signal(sig, lambda s, f: _signal_handler())

    log.info("pipeline_running")
    await shutdown_event.wait()

    log.info("shutting_down")
    await pipeline.shutdown()
    log.info("application_exited")


def main() -> None:
    """Synchronous wrapper for the async entry-point."""
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
