"""Multi-engine consensus adjudicator — no LLM image processing.

When the deterministic validator (``validator.py``) fails or the OCR
confidence falls below the adjudication threshold, this module
re-evaluates the plate by running it through **all available OCR
engines** and using consensus voting to determine the most likely
correct plate text.

Gemma 4 E2B is the Manager Agent brain only — it does NOT process
images.  All OCR is performed by dedicated pre-trained models and
libraries (EasyOCR, PaddleOCR, TrOCR, etc.) that the Manager Agent
provisions at runtime.

Spec references: §8.1–§8.4 agentic orchestration, §4 S8 dispatch ≤ 2s.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from uni_vision.common.config import AdjudicationConfig
from uni_vision.contracts.dtos import DetectionContext, OCRResult, ValidationStatus

logger = logging.getLogger(__name__)


class AdjudicationResult:
    """Result of multi-engine consensus adjudication."""

    __slots__ = ("plate_text", "confidence", "corrections", "reasoning")

    def __init__(
        self,
        plate_text: str,
        confidence: float,
        corrections: str,
        reasoning: str,
    ) -> None:
        self.plate_text = plate_text
        self.confidence = confidence
        self.corrections = corrections
        self.reasoning = reasoning


class ConsensusAdjudicator:
    """Multi-engine consensus adjudicator for ambiguous plates.

    Instead of sending images to an LLM, this adjudicator runs the
    plate crop through all available OCR engines and uses voting to
    determine the most likely correct text.

    Parameters
    ----------
    ocr_strategy : OCRStrategy
        The OCR strategy with access to all provisioned engines.
    adj_config : AdjudicationConfig
        Toggle, timeout, and retry settings.
    """

    def __init__(
        self,
        ocr_strategy: object,
        adj_config: AdjudicationConfig,
    ) -> None:
        self._ocr_strategy = ocr_strategy
        self._cfg = adj_config

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    async def adjudicate(
        self,
        plate_image: NDArray[np.uint8],
        ocr_text: str,
        ocr_confidence: float,
        corrections_attempted: Dict[str, str],
        ocr_engine: str,
    ) -> Optional[AdjudicationResult]:
        """Run multi-engine consensus on the plate image.

        Returns ``None`` if consensus cannot be reached (e.g. only
        one engine available, or all engines fail).
        """
        engines = getattr(self._ocr_strategy, "engines", [])

        if len(engines) < 2:
            # Cannot do consensus with fewer than 2 engines.
            # Return the original OCR result as-is.
            logger.debug(
                "Consensus skipped: only %d engine(s) available",
                len(engines),
            )
            return None

        # Run all engines (except the one that already produced the result)
        context = DetectionContext(
            camera_id="consensus",
            vehicle_class="unknown",
        )

        tasks = []
        engine_names = []
        for eng in engines:
            eng_name = getattr(eng, "engine_name", "unknown")
            tasks.append(self._safe_extract(eng, plate_image, context))
            engine_names.append(eng_name)

        raw_results = await asyncio.gather(*tasks)

        # Collect valid readings
        valid: List[OCRResult] = []
        for r in raw_results:
            if r is not None and r.plate_text != "UNREADABLE":
                valid.append(r)

        if not valid:
            logger.warning("Consensus: all %d engines returned UNREADABLE", len(engines))
            return None

        # Include the original OCR text in the vote (it was produced
        # by one of the engines already)
        all_texts = [r.plate_text for r in valid]
        if ocr_text and ocr_text != "UNREADABLE":
            all_texts.append(ocr_text.strip().upper())

        # Vote on the best text
        votes = Counter(all_texts)
        best_text, count = votes.most_common(1)[0]
        total = len(all_texts)

        # Consensus confidence: agreement ratio weighted by average engine confidence
        agreement_ratio = count / total
        matching_confs = [
            r.confidence for r in valid if r.plate_text == best_text
        ]
        avg_conf = sum(matching_confs) / len(matching_confs) if matching_confs else 0.5
        consensus_confidence = min(1.0, agreement_ratio * 0.5 + avg_conf * 0.5)

        logger.info(
            "Consensus: '%s' — %d/%d engines agree (confidence=%.3f)",
            best_text,
            count,
            total,
            consensus_confidence,
        )

        return AdjudicationResult(
            plate_text=best_text,
            confidence=consensus_confidence,
            corrections=f"consensus: {count}/{total} votes",
            reasoning=(
                f"Multi-engine consensus: {count}/{total} engines agree on "
                f"'{best_text}' (engines: {', '.join(engine_names)})"
            ),
        )

    @staticmethod
    async def _safe_extract(
        engine: object,
        image: NDArray[np.uint8],
        context: DetectionContext,
    ) -> Optional[OCRResult]:
        """Run a single engine with error isolation."""
        try:
            return await engine.extract(image, context)  # type: ignore[union-attr]
        except Exception as exc:
            eng_name = getattr(engine, "engine_name", "unknown")
            logger.debug("Consensus engine %s failed: %s", eng_name, exc)
            return None

    async def close(self) -> None:
        """No resources to release (engines are owned by OCRStrategy)."""
        pass
