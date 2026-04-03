"""S3 Plate Sub-Detector — spec §4 S3.

Localised licence-plate detection that operates **strictly** within the
bounding box coordinates provided by the S2 vehicle detector.

The sub-detector receives the full-frame image together with a vehicle
``BoundingBox`` from S2, crops the region of interest (ROI), runs
plate detection inside that crop, then maps resulting coordinates back
to the original frame space.

Multi-plate policy
~~~~~~~~~~~~~~~~~~
When multiple plates are detected inside a single vehicle ROI the
``highest_confidence`` policy is applied: only the single plate with
the greatest confidence score is returned (spec §4 S3).

VRAM lifecycle
~~~~~~~~~~~~~~
Same load / unload semantics as ``VehicleDetector`` — the pipeline
orchestrator controls Region C time-slicing between S2 and S3.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import structlog

from uni_vision.contracts.dtos import BoundingBox
from uni_vision.detection.engine import EngineConfig, InferenceEngine
from uni_vision.monitoring.metrics import DETECTIONS_TOTAL, STAGE_LATENCY

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = structlog.get_logger()

# Plate detector has a single class (licence plate = 0)
_PLATE_CLASSES: dict[int, str] = {0: "plate"}


class PlateDetector:
    """Two-tiered strategy — Tier 2: licence-plate localisation within ROI.

    Parameters
    ----------
    model_path : str
        Path to the TensorRT ``.engine`` or ONNX ``.onnx`` plate model.
    model_format : str
        ``"tensorrt"`` or ``"onnx"``.
    input_size : tuple[int, int]
        Model input resolution, default ``(640, 640)``.
    confidence_threshold : float
        Minimum confidence for plate detections (spec default: 0.65).
    nms_iou_threshold : float
        NMS IoU overlap limit.
    multi_plate_policy : str
        How to resolve multiple plates.  Only ``"highest_confidence"``
        is currently supported.
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
        confidence_threshold: float = 0.65,
        nms_iou_threshold: float = 0.45,
        multi_plate_policy: str = "highest_confidence",
        device: str = "cuda",
        device_index: int = 0,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._multi_plate_policy = multi_plate_policy

        self._engine = InferenceEngine(
            EngineConfig(
                model_path=model_path,
                model_format=model_format,
                input_size=input_size,
                confidence_threshold=confidence_threshold,
                nms_iou_threshold=nms_iou_threshold,
                classes=_PLATE_CLASSES,
                device=device,
                device_index=device_index,
            ),
            stage_label="S3_plate_detect",
        )

    # ── Detector Protocol ─────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return f"plate_detector:{self._model_path}"

    @property
    def device(self) -> str:
        return self._device

    def warmup(self) -> None:
        """Load model weights and run a dummy forward pass."""
        self._engine.load()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._engine.predict(dummy)
        log.info("plate_detector_warmed_up", device=self._device)

    def detect(self, image: NDArray[np.uint8]) -> list[BoundingBox]:
        """Detect plates in *image* (full frame or pre-cropped ROI).

        When called directly the detector treats the entire image as the
        search area.  For the two-tiered pipeline use
        ``detect_in_roi()`` instead, which crops to the vehicle bounding
        box and remaps coordinates.

        Returns
        -------
        List of ``BoundingBox`` sorted by descending confidence (after
        multi-plate policy is applied).
        """
        t0 = time.perf_counter()
        detections = self._engine.predict(image)
        elapsed = time.perf_counter() - t0
        STAGE_LATENCY.labels(stage="S3_plate_detect").observe(elapsed)
        DETECTIONS_TOTAL.labels(stage="S3", cls="plate").inc(len(detections))

        return self._apply_multi_plate_policy(detections)

    def detect_in_roi(
        self,
        frame: NDArray[np.uint8],
        vehicle_bbox: BoundingBox,
    ) -> list[BoundingBox]:
        """Detect plates strictly within *vehicle_bbox* of *frame*.

        Steps:
        1. Crop ROI from the full frame using vehicle_bbox coordinates.
        2. Run plate detection on the cropped region.
        3. Remap resulting plate coordinates back to full-frame space.
        4. Apply multi-plate policy.

        Parameters
        ----------
        frame : (H, W, 3) uint8 BGR — the original camera frame.
        vehicle_bbox : S2 vehicle detection bounding box.

        Returns
        -------
        Plate ``BoundingBox`` list in full-frame coordinates, after
        policy filtering (typically a single entry).
        """
        t0 = time.perf_counter()

        # 1. Crop ROI (NumPy view — no copy)
        h, w = frame.shape[:2]
        x1 = max(0, vehicle_bbox.x1)
        y1 = max(0, vehicle_bbox.y1)
        x2 = min(w, vehicle_bbox.x2)
        y2 = min(h, vehicle_bbox.y2)

        if x2 <= x1 or y2 <= y1:
            log.warning(
                "plate_detect_empty_roi",
                bbox=(vehicle_bbox.x1, vehicle_bbox.y1, vehicle_bbox.x2, vehicle_bbox.y2),
            )
            return []

        roi = frame[y1:y2, x1:x2]

        # Ensure ROI is contiguous for the inference backend
        if not roi.flags["C_CONTIGUOUS"]:
            roi = np.ascontiguousarray(roi)

        # 2. Detect plates inside the crop
        detections = self._engine.predict(roi)

        # 3. Remap coordinates back to full-frame space
        remapped: list[BoundingBox] = []
        for det in detections:
            remapped.append(
                BoundingBox(
                    x1=det.x1 + x1,
                    y1=det.y1 + y1,
                    x2=det.x2 + x1,
                    y2=det.y2 + y1,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name,
                )
            )

        # 4. Apply policy
        result = self._apply_multi_plate_policy(remapped)

        elapsed = time.perf_counter() - t0
        STAGE_LATENCY.labels(stage="S3_plate_detect").observe(elapsed)
        DETECTIONS_TOTAL.labels(stage="S3", cls="plate").inc(len(result))

        log.debug(
            "plate_detect_roi_done",
            count=len(result),
            elapsed_ms=round(elapsed * 1000, 1),
        )
        return result

    def release(self) -> None:
        """Unload model weights and free all device memory."""
        self._engine.unload()
        log.info("plate_detector_released")

    # ── Multi-plate policy ────────────────────────────────────────

    def _apply_multi_plate_policy(self, detections: list[BoundingBox]) -> list[BoundingBox]:
        """Reduce detections according to the configured policy."""
        if not detections:
            return detections

        if self._multi_plate_policy == "highest_confidence":
            best = max(detections, key=lambda d: d.confidence)
            return [best]

        # Fallback: return all sorted by confidence
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    # ── Convenience ───────────────────────────────────────────────

    def switch_device(self, new_device: str) -> None:
        """Re-initialise engine on a different device (for offloading)."""
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
                classes=_PLATE_CLASSES,
                device=new_device,
                device_index=self._engine._cfg.device_index,
            ),
            stage_label="S3_plate_detect",
        )
        if was_loaded:
            self._engine.load()
        log.info("plate_detector_device_switched", new_device=new_device)
