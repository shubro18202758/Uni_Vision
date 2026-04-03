"""EasyOCR-based lightweight fallback OCR engine — spec §9.1 Strategy.

Runs exclusively on **CPU** (``gpu=False``) so it never competes for
VRAM with the LLM or vision models on the RTX 4070.  The ``Reader``
is lazily initialised on first call and reused thereafter (single-
allocation principle — no per-frame overhead).

The engine produces results compatible with the ``OCREngine`` Protocol
including per-character bounding boxes extracted from EasyOCR's native
``readtext()`` output.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from uni_vision.contracts.dtos import DetectionContext, OCRResult, ValidationStatus

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from uni_vision.common.config import FallbackOCRConfig

logger = logging.getLogger(__name__)


class EasyOCRFallback:
    """Lightweight CPU-only OCR engine implementing ``OCREngine`` Protocol.

    Parameters
    ----------
    config : FallbackOCRConfig
        Engine configuration (languages, gpu flag, allowlist, threshold).
    """

    def __init__(self, config: FallbackOCRConfig) -> None:
        self._config = config
        self._reader: object | None = None  # lazily initialised
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr_fb")

    @property
    def engine_name(self) -> str:
        return "easyocr_fallback"

    # ── Protocol method ───────────────────────────────────────────

    async def extract(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext,
    ) -> OCRResult:
        """Run EasyOCR on ``image`` in a thread executor (blocking I/O)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._extract_sync,
            image,
            context,
        )

    # ── Internal sync implementation ──────────────────────────────

    def _extract_sync(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext,
    ) -> OCRResult:
        reader = self._ensure_reader()

        results: list = reader.readtext(  # type: ignore[union-attr]
            image,
            allowlist=self._config.allowlist,
            detail=1,
        )

        if not results:
            return OCRResult(
                plate_text="UNREADABLE",
                raw_text="",
                confidence=0.0,
                reasoning="EasyOCR returned no detections",
                engine=self.engine_name,
                status=ValidationStatus.UNREADABLE,
            )

        # Aggregate text and character bounding boxes
        texts: list[str] = []
        confidences: list[float] = []
        bbox_strs: list[str] = []

        for bbox, text, conf in results:
            texts.append(text.upper())
            confidences.append(float(conf))
            # EasyOCR bboxes are 4-point polygons [[x1,y1],…]
            # Convert to axis-aligned x1,y1,x2,y2
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            bbox_strs.append(f"{int(min(xs))},{int(min(ys))},{int(max(xs))},{int(max(ys))}")

        plate_text = "".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        char_bbox_str = ";".join(bbox_strs) if bbox_strs else "NONE"

        # Filter by confidence threshold
        if avg_confidence < self._config.confidence_threshold:
            status = ValidationStatus.LOW_CONFIDENCE
        else:
            status = ValidationStatus.FALLBACK

        reasoning = (
            f"EasyOCR detected {len(results)} region(s), "
            f"avg confidence {avg_confidence:.2f}. "
            f"char_bboxes: {char_bbox_str}"
        )

        return OCRResult(
            plate_text=plate_text,
            raw_text="".join(texts),
            confidence=round(avg_confidence, 4),
            reasoning=reasoning,
            engine=self.engine_name,
            status=status,
        )

    # ── Lazy initialisation ───────────────────────────────────────

    def _ensure_reader(self) -> object:
        """Instantiate the EasyOCR ``Reader`` once, on first invocation."""
        if self._reader is None:
            import easyocr  # deferred import — heavy dependency

            logger.info(
                "Initialising EasyOCR reader (languages=%s, gpu=%s)",
                self._config.languages,
                self._config.gpu,
            )
            self._reader = easyocr.Reader(
                self._config.languages,
                gpu=self._config.gpu,
            )
        return self._reader
