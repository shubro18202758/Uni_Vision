"""Cognitive Orchestrator — the definitive S8 decision engine.

Composes two processing layers into a single ``PostProcessor``
implementation:

1. **Deterministic layer** (``DeterministicValidator``) — fast, code-only
   character confusion correction and locale-aware regex validation.
   Runs entirely on the CPU with sub-millisecond latency.

2. **Consensus layer** (``ConsensusAdjudicator``) — multi-engine
   voting adjudication.  Invoked *only* when the deterministic
   layer fails (``REGEX_FAIL``) or when OCR confidence falls below
   the configured threshold (``LOW_CONFIDENCE``).

The orchestrator **never blocks** the high-speed processing queue:
strict timeouts on the adjudicator, and graceful degradation to the
best available deterministic result on any engine failure.

Spec references: §4 S8, §8.1–§8.4 agentic orchestration, §9.2
PostProcessor protocol.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from uni_vision.contracts.dtos import (
    OCRResult,
    ProcessedResult,
    ValidationStatus,
)
from uni_vision.postprocessing.validator import (
    DeterministicValidator,
    ValidationVerdict,
    Verdict,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from uni_vision.common.config import (
        AdjudicationConfig,
        ValidationConfig,
    )
    from uni_vision.postprocessing.adjudicator import (
        AdjudicationResult,
        ConsensusAdjudicator,
    )

logger = logging.getLogger(__name__)


# ── Verdict → ValidationStatus mapping ────────────────────────────

_VERDICT_STATUS = {
    Verdict.ACCEPTED: ValidationStatus.VALID,
    Verdict.CORRECTED: ValidationStatus.VALID,
    Verdict.LOW_CONFIDENCE: ValidationStatus.LOW_CONFIDENCE,
    Verdict.REGEX_FAIL: ValidationStatus.REGEX_FAIL,
}


class CognitiveOrchestrator:
    """Central decision engine implementing the ``PostProcessor`` protocol.

    Parameters
    ----------
    validation_config : ValidationConfig
        Settings for the deterministic validator.
    adjudicator : ConsensusAdjudicator
        Pre-built consensus adjudicator (multi-engine voting).
    adjudication_config : AdjudicationConfig
        Toggle, timeout, and retry settings for the consensus layer.
    """

    def __init__(
        self,
        validation_config: ValidationConfig,
        adjudicator: ConsensusAdjudicator,
        adjudication_config: AdjudicationConfig,
    ) -> None:
        self._deterministic = DeterministicValidator(validation_config)
        self._adjudicator = adjudicator
        self._adj_enabled = adjudication_config.enabled
        self._confidence_threshold = validation_config.adjudication_confidence_threshold

    # ── PostProcessor protocol ────────────────────────────────────

    async def validate(
        self,
        result: OCRResult,
        plate_image: NDArray[np.uint8],
    ) -> ProcessedResult:
        """Apply deterministic validation, then agentic adjudication if needed.

        Flow
        ----
        1. Run ``DeterministicValidator.validate()`` on the raw text.
        2. If verdict is ``ACCEPTED`` or ``CORRECTED`` with high
           confidence → return immediately.
        3. If verdict is ``REGEX_FAIL`` or ``LOW_CONFIDENCE`` and the
           adjudicator is enabled → invoke the multimodal LLM.
        4. If the LLM adjudicator succeeds → use its result.
        5. If the LLM adjudicator fails → return the best
           deterministic result with a degraded status.
        """
        t0 = time.perf_counter()

        # ── Layer 1: deterministic validation ─────────────────────
        verdict: ValidationVerdict = self._deterministic.validate(
            result.plate_text,
            result.confidence,
        )

        det_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "S8 deterministic: verdict=%s text=%s→%s (%.1f ms)",
            verdict.verdict.value,
            verdict.original_text,
            verdict.corrected_text,
            det_ms,
        )

        # Fast path — deterministic pass with high confidence
        if verdict.verdict in (Verdict.ACCEPTED, Verdict.CORRECTED):
            return ProcessedResult(
                plate_text=verdict.corrected_text,
                raw_ocr_text=result.raw_text,
                confidence=result.confidence,
                validation_status=_VERDICT_STATUS[verdict.verdict],
                char_corrections=verdict.corrections_applied,
            )

        # ── Layer 2: agentic adjudication ─────────────────────────
        if not self._adj_enabled:
            return self._fallback_result(result, verdict)

        adj_result = await self._invoke_adjudicator(
            plate_image=plate_image,
            ocr_text=verdict.corrected_text,
            ocr_confidence=result.confidence,
            corrections=verdict.corrections_applied,
            engine=result.engine,
        )

        if adj_result is not None:
            return self._build_adjudicated_result(result, adj_result, verdict)

        # LLM failed — return degraded deterministic result
        return self._fallback_result(result, verdict)

    # ── Adjudicator invocation ────────────────────────────────────

    async def _invoke_adjudicator(
        self,
        plate_image: NDArray[np.uint8],
        ocr_text: str,
        ocr_confidence: float,
        corrections: dict,
        engine: str,
    ) -> AdjudicationResult | None:
        """Call the LLM adjudicator with timing and error isolation."""
        t0 = time.perf_counter()
        try:
            adj = await self._adjudicator.adjudicate(
                plate_image=plate_image,
                ocr_text=ocr_text,
                ocr_confidence=ocr_confidence,
                corrections_attempted=corrections,
                ocr_engine=engine,
            )
        except Exception as exc:
            logger.error("S8 adjudicator unexpected error: %s", exc, exc_info=True)
            adj = None

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if adj is not None:
            logger.info(
                "S8 adjudicated: text=%s conf=%.3f (%.1f ms)",
                adj.plate_text,
                adj.confidence,
                elapsed_ms,
            )
        else:
            logger.warning("S8 adjudicator returned None (%.1f ms)", elapsed_ms)

        return adj

    # ── Result builders ───────────────────────────────────────────

    @staticmethod
    def _build_adjudicated_result(
        ocr: OCRResult,
        adj: AdjudicationResult,
        det: ValidationVerdict,
    ) -> ProcessedResult:
        """Map a successful adjudication to ``ProcessedResult``."""
        plate = adj.plate_text.strip().upper()

        if plate == "UNREADABLE" or adj.confidence < 0.1:
            return ProcessedResult(
                plate_text=det.corrected_text,
                raw_ocr_text=ocr.raw_text,
                confidence=adj.confidence,
                validation_status=ValidationStatus.UNREADABLE,
                char_corrections=det.corrections_applied,
            )

        return ProcessedResult(
            plate_text=plate,
            raw_ocr_text=ocr.raw_text,
            confidence=adj.confidence,
            validation_status=ValidationStatus.VALID,
            char_corrections=det.corrections_applied,
        )

    @staticmethod
    def _fallback_result(
        ocr: OCRResult,
        det: ValidationVerdict,
    ) -> ProcessedResult:
        """Return the best deterministic result with degraded status."""
        return ProcessedResult(
            plate_text=det.corrected_text,
            raw_ocr_text=ocr.raw_text,
            confidence=ocr.confidence,
            validation_status=_VERDICT_STATUS.get(det.verdict, ValidationStatus.REGEX_FAIL),
            char_corrections=det.corrections_applied,
        )

    async def close(self) -> None:
        """Release resources held by the adjudicator."""
        await self._adjudicator.close()
