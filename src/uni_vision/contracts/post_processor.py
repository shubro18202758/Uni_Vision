"""Protocol contract for post-processing validation (S8 — validation phase)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from uni_vision.contracts.dtos import OCRResult, ProcessedResult


@runtime_checkable
class PostProcessor(Protocol):
    """Validates, corrects, and optionally adjudicates raw OCR output.

    Implementations: ``CognitiveOrchestrator`` — deterministic char-
    substitution + regex first, then agentic LLM adjudication on failure.

    The ``plate_image`` is carried through so the LLM adjudicator can
    perform deep spatial reasoning when the deterministic layer fails.

    This protocol covers only the validation/correction step.
    Dispatch is handled by the separate ``Dispatcher`` protocol.
    """

    async def validate(
        self,
        result: OCRResult,
        plate_image: NDArray[np.uint8],
    ) -> ProcessedResult:
        """Apply character corrections, regex validation, and optional LLM adjudication.

        Args:
            result: Raw OCR result from the engine.
            plate_image: Enhanced plate crop ``(H, W, 3)`` BGR uint8 —
                used for multimodal LLM adjudication if deterministic
                validation fails.

        Returns:
            Processed result with corrected text and validation status.
        """
        ...
