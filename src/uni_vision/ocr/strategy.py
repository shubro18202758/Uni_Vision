"""OCR Strategy manager — multi-engine selection via priority ordering.

The ``OCRStrategy`` implements the ``OCREngine`` Protocol and manages
a list of OCR engines in priority order.  The default engine is
EasyOCR; additional engines are provisioned at runtime by the Manager
Agent (e.g. PaddleOCR, TrOCR, etc. from HuggingFace/PyPI).

Engine selection is simple:
  * Try engines in priority order (index 0 = highest priority).
  * First engine to return a non-UNREADABLE result wins.
  * If all engines fail → return UNREADABLE.

The Manager Agent can dynamically add/remove engines via
``add_engine()`` and ``remove_engine()`` as it adapts
the pipeline to changing conditions.

Spec reference: §9.1 OCR strategy.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from uni_vision.contracts.dtos import (
    DetectionContext,
    OCRResult,
    ValidationStatus,
)
from uni_vision.monitoring.metrics import (
    OCR_FALLBACK,
    OCR_REQUESTS,
    OCR_SUCCESS,
    STAGE_LATENCY,
)

logger = logging.getLogger(__name__)


class OCRStrategy:
    """Multi-engine OCR strategy with priority-ordered fallback.

    Parameters
    ----------
    engines : list[OCREngine]
        Ordered list of OCR engines (index 0 = highest priority).
        At minimum, one engine must be provided (typically EasyOCR).
    """

    def __init__(self, engines: List[object]) -> None:
        if not engines:
            raise ValueError("OCRStrategy requires at least one engine")
        self._engines: List[object] = list(engines)

    @property
    def engine_name(self) -> str:
        return "ocr_strategy"

    @property
    def engines(self) -> List[object]:
        """Return a copy of the current engine list."""
        return list(self._engines)

    def add_engine(
        self,
        engine: object,
        priority: Optional[int] = None,
    ) -> None:
        """Add an engine at the given priority position.

        Parameters
        ----------
        engine : OCREngine
            Engine to add.
        priority : int, optional
            Insert position (0 = highest priority).
            ``None`` appends to the end (lowest priority).
        """
        if priority is None:
            self._engines.append(engine)
        else:
            self._engines.insert(priority, engine)
        logger.info(
            "OCR engine added: %s (position %d/%d)",
            getattr(engine, "engine_name", "unknown"),
            priority if priority is not None else len(self._engines) - 1,
            len(self._engines),
        )

    def remove_engine(self, engine_name: str) -> bool:
        """Remove an engine by name.  Returns True if found."""
        for i, eng in enumerate(self._engines):
            if getattr(eng, "engine_name", None) == engine_name:
                self._engines.pop(i)
                logger.info("OCR engine removed: %s", engine_name)
                return True
        return False

    async def extract(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext,
    ) -> OCRResult:
        """Try engines in priority order; return first valid result."""
        last_error: Optional[Exception] = None

        for idx, engine in enumerate(self._engines):
            eng_name = getattr(engine, "engine_name", f"engine_{idx}")
            OCR_REQUESTS.labels(engine=eng_name).inc()

            try:
                result = await engine.extract(image, context)  # type: ignore[union-attr]
                OCR_SUCCESS.labels(engine=eng_name).inc()

                if result.plate_text != "UNREADABLE":
                    return result

                # Engine returned UNREADABLE — try next
                logger.debug("Engine %s returned UNREADABLE, trying next", eng_name)

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "OCR engine %s failed (%s) — trying next",
                    eng_name,
                    exc,
                )
                if idx < len(self._engines) - 1:
                    OCR_FALLBACK.inc()

        # All engines exhausted
        logger.error("All %d OCR engines failed", len(self._engines))
        return OCRResult(
            plate_text="UNREADABLE",
            raw_text="",
            confidence=0.0,
            reasoning=f"All {len(self._engines)} OCR engines failed. Last error: {last_error}",
            engine=self.engine_name,
            status=ValidationStatus.UNREADABLE,
        )
