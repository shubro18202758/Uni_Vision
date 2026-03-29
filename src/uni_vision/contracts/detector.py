"""Protocol contract for object detectors (S2, S3)."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from uni_vision.contracts.dtos import BoundingBox


@runtime_checkable
class Detector(Protocol):
    """Runs inference on an image and returns bounding boxes.

    Implementations: YOLOv8 vehicle detector, YOLO-variant LPD,
    ONNX Runtime fallback, CPU-only stub.

    Vehicle and plate detectors share this interface — they differ
    only in their trained weights, class maps, and thresholds.
    """

    @property
    def model_name(self) -> str:
        """Human-readable identifier for logging and metrics."""
        ...

    @property
    def device(self) -> str:
        """Execution device: ``"cuda"`` or ``"cpu"``."""
        ...

    def warmup(self) -> None:
        """Run a dummy forward pass to pre-allocate CUDA memory.

        Must be called once after model load, before the first real
        inference call, to avoid first-frame latency spikes.
        """
        ...

    def detect(self, image: NDArray[np.uint8]) -> List[BoundingBox]:
        """Run detection on a BGR uint8 image.

        Args:
            image: ``(H, W, 3)`` BGR array on CPU.  The implementation
                   is responsible for upload, resize, letterbox, and
                   normalisation.

        Returns:
            Bounding boxes that pass the configured confidence and NMS
            thresholds, sorted by descending confidence.
        """
        ...

    def release(self) -> None:
        """Unload model weights and free all associated GPU/CPU memory."""
        ...
