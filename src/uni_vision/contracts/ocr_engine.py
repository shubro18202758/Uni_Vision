"""Protocol contract for OCR engines (S7)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from uni_vision.contracts.dtos import DetectionContext, OCRResult


@runtime_checkable
class OCREngine(Protocol):
    """Extracts text from an enhanced image region.

    Implementations: EasyOCRFallback (default), ComponentOCREngine
    (adapter for Manager-provisioned models — PaddleOCR, TrOCR, etc.).

    The engines are managed by OCRStrategy which orders them by
    priority.  The Manager Agent provisions additional engines at
    runtime based on scene requirements.
    """

    @property
    def engine_name(self) -> str:
        """Identifier embedded in ``OCRResult.engine`` for provenance."""
        ...

    async def extract(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext,
    ) -> OCRResult:
        """Perform OCR on the plate image.

        Args:
            image: Enhanced plate crop, ``(H, W, 3)`` BGR uint8.
            context: Upstream detection metadata (camera, timestamp,
                     vehicle class, bounding boxes).

        Returns:
            Structured OCR result including text, confidence, reasoning
            trace, and validation status.

        The implementation is async because some engines
        (e.g. ComponentOCREngine) may delegate to async components.
        """
        ...
