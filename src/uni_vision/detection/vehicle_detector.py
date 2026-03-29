"""S2 Vehicle Detector — spec §4 S2.

Edge-optimised real-time vehicle detection using YOLOv8n INT8
quantised weights via the unified ``InferenceEngine`` backend.

Classifies broad vehicle categories (car, truck, bus, motorcycle)
and returns bounding boxes sorted by descending confidence.

This module implements the ``Detector`` protocol defined in
``contracts.detector``.

VRAM lifecycle
~~~~~~~~~~~~~~
* ``warmup()`` loads the model onto the GPU, allocating ≤ 400 MB
  inside Region C.
* ``detect()`` runs the full letterbox → infer → NMS pipeline.
* ``release()`` unloads all weights and frees Region C memory.

The pipeline orchestrator may call ``release()`` between events to
yield VRAM for the plate sub-detector or to satisfy the memory fence
before LLM inference.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
from numpy.typing import NDArray

import structlog

from uni_vision.contracts.dtos import BoundingBox
from uni_vision.monitoring.metrics import STAGE_LATENCY, DETECTIONS_TOTAL
from uni_vision.detection.engine import EngineConfig, InferenceEngine

log = structlog.get_logger()

# Spec §4 S2 — vehicle class map (matches config/models.yaml)
_VEHICLE_CLASSES: dict[int, str] = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "motorcycle",
}


class VehicleDetector:
    """Two-tiered strategy — Tier 1: broad vehicle localisation.

    Parameters
    ----------
    model_path : str
        Filesystem path to the TensorRT ``.engine`` or ONNX ``.onnx`` file.
    model_format : str
        ``"tensorrt"`` or ``"onnx"``.
    input_size : tuple[int, int]
        Model input resolution, default ``(640, 640)``.
    confidence_threshold : float
        Minimum confidence to keep a detection (spec default: 0.60).
    nms_iou_threshold : float
        NMS IoU overlap limit (spec default: 0.45).
    device : str
        ``"cuda"`` or ``"cpu"``.
    device_index : int
        CUDA device ordinal.
    """

    def __init__(
        self,
        *,
        model_path: str,
        model_format: str = "tensorrt",
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.60,
        nms_iou_threshold: float = 0.45,
        device: str = "cuda",
        device_index: int = 0,
    ) -> None:
        self._model_path = model_path
        self._device = device

        self._engine = InferenceEngine(
            EngineConfig(
                model_path=model_path,
                model_format=model_format,
                input_size=input_size,
                confidence_threshold=confidence_threshold,
                nms_iou_threshold=nms_iou_threshold,
                classes=_VEHICLE_CLASSES,
                device=device,
                device_index=device_index,
            ),
            stage_label="S2_vehicle_detect",
        )

    # ── Detector Protocol ─────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return f"vehicle_detector:{self._model_path}"

    @property
    def device(self) -> str:
        return self._device

    def warmup(self) -> None:
        """Load model weights onto the device and run a dummy forward pass."""
        self._engine.load()
        # Dummy inference to trigger TensorRT kernel autotuning / JIT
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._engine.predict(dummy)
        log.info("vehicle_detector_warmed_up", device=self._device)

    def detect(self, image: NDArray[np.uint8]) -> List[BoundingBox]:
        """Detect vehicles in *image* (BGR, uint8).

        Returns bounding boxes for car / truck / bus / motorcycle
        sorted by descending confidence.  Only boxes with
        ``confidence >= threshold`` and surviving NMS are returned.

        Raises
        ------
        VRAMError
            If the engine has not been loaded (``warmup()`` not called).
        """
        t0 = time.perf_counter()
        detections = self._engine.predict(
            image,
            class_filter=list(_VEHICLE_CLASSES.keys()),
        )
        elapsed = time.perf_counter() - t0
        STAGE_LATENCY.labels(stage="S2_vehicle_detect").observe(elapsed)
        DETECTIONS_TOTAL.labels(stage="S2", cls="vehicle").inc(len(detections))

        log.debug(
            "vehicle_detect_done",
            count=len(detections),
            elapsed_ms=round(elapsed * 1000, 1),
        )
        return detections

    def release(self) -> None:
        """Unload model weights and free all device memory."""
        self._engine.unload()
        log.info("vehicle_detector_released")

    # ── Convenience ───────────────────────────────────────────────

    def switch_device(self, new_device: str) -> None:
        """Re-initialise the engine on a different device (offloading)."""
        was_loaded = self._engine.loaded
        self._engine.unload()
        self._device = new_device
        self._engine = InferenceEngine(
            EngineConfig(
                model_path=self._model_path,
                model_format="onnx" if new_device == "cpu" else self._engine._cfg.model_format,
                input_size=self._engine._cfg.input_size,
                confidence_threshold=self._engine._cfg.confidence_threshold,
                nms_iou_threshold=self._engine._cfg.nms_iou_threshold,
                classes=_VEHICLE_CLASSES,
                device=new_device,
                device_index=self._engine._cfg.device_index,
            ),
            stage_label="S2_vehicle_detect",
        )
        if was_loaded:
            self._engine.load()
        log.info("vehicle_detector_device_switched", new_device=new_device)
