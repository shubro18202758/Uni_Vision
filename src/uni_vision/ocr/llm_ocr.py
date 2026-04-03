"""Component-based OCR engine — wraps Manager-provisioned CV components.

Provides ``ComponentOCREngine`` which adapts any CVComponent with
PLATE_OCR capability into the ``OCREngine`` Protocol.  The Manager
Agent provisions OCR models from HuggingFace, PyPI, or TorchHub,
and this adapter makes them usable by the pipeline.

Gemma 4 E2B is the Manager Agent brain ONLY — it reasons about which
components to provision, resolves conflicts, and monitors health.
ALL image processing (including OCR) is performed by dedicated
pre-trained models and libraries (EasyOCR, PaddleOCR, TrOCR, etc.).

Spec reference: §9.1 OCR strategy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from uni_vision.contracts.dtos import DetectionContext, OCRResult, ValidationStatus

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ComponentOCREngine:
    """OCR engine adapter for Manager-provisioned CV components.

    Adapts a ``CVComponent`` (e.g. ``HuggingFaceModelComponent`` wrapping
    TrOCR, ``PipPackageComponent`` wrapping PaddleOCR) into the
    ``OCREngine`` Protocol so the pipeline can use it transparently.

    Parameters
    ----------
    component : CVComponent
        A loaded CV component with PLATE_OCR capability.
    name : str
        Human-readable engine name for logging and metrics.
    """

    def __init__(self, component: Any, name: str = "component_ocr") -> None:
        self._component = component
        self._name = name

    @property
    def engine_name(self) -> str:
        return self._name

    async def extract(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext,
    ) -> OCRResult:
        """Run OCR via the wrapped component."""
        try:
            result = await self._component.execute(
                image,
                context={
                    "camera_id": context.camera_id,
                    "vehicle_class": context.vehicle_class,
                },
            )
            return self._normalize_result(result)
        except Exception as exc:
            logger.error(
                "Component OCR %s failed: %s",
                self._name,
                exc,
                exc_info=True,
            )
            return OCRResult(
                plate_text="UNREADABLE",
                raw_text="",
                confidence=0.0,
                reasoning=f"Component {self._name} error: {exc}",
                engine=self.engine_name,
                status=ValidationStatus.UNREADABLE,
            )

    def _normalize_result(self, result: Any) -> OCRResult:
        """Convert component output to ``OCRResult``."""
        if isinstance(result, OCRResult):
            return result

        if isinstance(result, str):
            plate_text = result.strip().upper()
            return OCRResult(
                plate_text=plate_text or "UNREADABLE",
                raw_text=result,
                confidence=0.7 if plate_text else 0.0,
                reasoning=f"Component {self._name} returned raw text",
                engine=self.engine_name,
                status=(ValidationStatus.VALID if plate_text else ValidationStatus.UNREADABLE),
            )

        if isinstance(result, dict):
            plate_text = str(result.get("plate_text", result.get("text", ""))).strip().upper()
            confidence = float(result.get("confidence", 0.7))
            return OCRResult(
                plate_text=plate_text or "UNREADABLE",
                raw_text=str(result),
                confidence=confidence,
                reasoning=str(result.get("reasoning", f"Component {self._name}")),
                engine=self.engine_name,
                status=(ValidationStatus.VALID if plate_text else ValidationStatus.UNREADABLE),
            )

        text = str(result).strip().upper()
        return OCRResult(
            plate_text=text or "UNREADABLE",
            raw_text=str(result),
            confidence=0.5,
            reasoning=f"Component {self._name}: {type(result).__name__}",
            engine=self.engine_name,
            status=(ValidationStatus.VALID if text else ValidationStatus.UNREADABLE),
        )
