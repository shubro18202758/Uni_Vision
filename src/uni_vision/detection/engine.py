"""Unified TensorRT / ONNX Runtime inference engine — spec §S2, §S3.

Provides a single ``InferenceEngine`` class that abstracts over two
execution backends:

* **TensorRT** (``*.engine`` files) — fastest path, INT8/FP16 quantised
  engines pre-compiled for the target GPU (RTX 4070).
* **ONNX Runtime** (``*.onnx`` files) — portable fallback with CUDA EP
  or CPU EP depending on hardware availability.

The engine handles:

1. **Letterbox preprocessing** — resize + pad to ``(input_h, input_w)``
   preserving aspect ratio, BGR→RGB, HWC→NCHW, float32 normalisation.
2. **GPU memory lifecycle** — explicit load (``warmup``) and unload
   (``release``) so the pipeline can time-slice Region C VRAM.
3. **NMS post-processing** — decode raw YOLO output tensors into
   ``BoundingBox`` DTOs with class filtering, confidence gating,
   and IoU-based non-maximum suppression.

Design notes
~~~~~~~~~~~~
* This module imports ``torch`` / ``tensorrt`` / ``onnxruntime`` **lazily**
  so that the package is importable on CPU-only hosts and in tests
  without the inference optional dependency group.
* All public methods are synchronous — they are called from the
  ``Pipeline`` inference consumer which executes stages sequentially
  on the GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

import structlog

from uni_vision.contracts.dtos import BoundingBox
from uni_vision.common.exceptions import VRAMError
from uni_vision.monitoring.metrics import STAGE_LATENCY, VRAM_USAGE

log = structlog.get_logger()

# Lazy-load cv2 once (not per-frame).
_cv2 = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


# ── Configuration dataclass ───────────────────────────────────────


@dataclass(frozen=True)
class EngineConfig:
    """Immutable configuration for a single inference engine instance."""

    model_path: str
    model_format: str                       # "tensorrt" | "onnx"
    input_size: Tuple[int, int] = (640, 640)  # (H, W)
    confidence_threshold: float = 0.60
    nms_iou_threshold: float = 0.45
    classes: dict[int, str] | None = None   # class_id → name mapping
    device: str = "cuda"
    device_index: int = 0


# ── Letterbox preprocessing ───────────────────────────────────────


def letterbox(
    image: NDArray[np.uint8],
    target_size: Tuple[int, int],
) -> Tuple[NDArray[np.float32], float, Tuple[int, int]]:
    """Resize *image* with letterbox padding for YOLO input.

    Returns
    -------
    blob : (1, 3, H, W) float32, normalised to [0, 1]
    ratio : scale factor applied to the original image
    pad : (pad_w, pad_h) pixel padding added
    """
    ih, iw = image.shape[:2]
    th, tw = target_size
    ratio = min(tw / iw, th / ih)
    new_w, new_h = int(round(iw * ratio)), int(round(ih * ratio))
    pad_w, pad_h = (tw - new_w) // 2, (th - new_h) // 2

    cv2 = _get_cv2()
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad with neutral grey (114)
    padded = np.full((th, tw, 3), 114, dtype=np.uint8)
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    # BGR → RGB, HWC → CHW, float32 / 255.0
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob[np.newaxis])  # (1, 3, H, W)
    return blob, ratio, (pad_w, pad_h)


# ── NMS (pure-NumPy, no torchvision dependency) ──────────────────


def _nms_numpy(
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    iou_threshold: float,
) -> NDArray[np.intp]:
    """Greedy NMS — returns indices of surviving boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.intp)


# ── Backend implementations ───────────────────────────────────────


class _TensorRTBackend:
    """TensorRT execution backend using the ``tensorrt`` Python API.

    Allocates device memory for I/O bindings at ``load()`` and frees
    it at ``unload()``.  The engine file (``*.engine``) must have been
    compiled on the same GPU architecture (sm_89 for RTX 4070).
    """

    def __init__(self, cfg: EngineConfig) -> None:
        self._cfg = cfg
        self._context = None
        self._engine = None
        self._runtime = None
        self._bindings: list = []
        self._d_inputs: list = []
        self._d_outputs: list = []
        self._h_outputs: list[NDArray] = []
        self._stream = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Deserialise TensorRT engine and allocate I/O device buffers."""
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — initialises CUDA context

        trt_logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(trt_logger)

        engine_path = Path(self._cfg.model_path)
        with open(engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        # Allocate I/O buffers
        self._bindings = []
        self._d_inputs = []
        self._d_outputs = []
        self._h_outputs = []

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = self._engine.get_tensor_shape(name)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            device_mem = cuda.mem_alloc(size)
            self._bindings.append(int(device_mem))

            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._d_inputs.append(device_mem)
            else:
                host_buf = np.empty(
                    [d if d > 0 else 1 for d in shape], dtype=dtype
                )
                self._d_outputs.append(device_mem)
                self._h_outputs.append(host_buf)

        self._loaded = True
        log.info(
            "tensorrt_engine_loaded",
            model=self._cfg.model_path,
            device=f"cuda:{self._cfg.device_index}",
        )

    def infer(self, blob: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run synchronous inference.  Returns raw output tensor."""
        import pycuda.driver as cuda

        cuda.memcpy_htod_async(self._d_inputs[0], blob, self._stream)

        for i, binding in enumerate(self._bindings):
            name = self._engine.get_tensor_name(i)
            self._context.set_tensor_address(name, binding)

        self._context.execute_async_v3(stream_handle=self._stream.handle)

        for d_out, h_out in zip(self._d_outputs, self._h_outputs):
            cuda.memcpy_dtoh_async(h_out, d_out, self._stream)

        self._stream.synchronize()
        return self._h_outputs[0].copy()

    def unload(self) -> None:
        """Free all device memory and destroy the execution context."""
        if not self._loaded:
            return

        for mem in self._d_inputs + self._d_outputs:
            try:
                mem.free()
            except Exception:
                pass

        self._d_inputs.clear()
        self._d_outputs.clear()
        self._h_outputs.clear()
        self._bindings.clear()
        self._context = None
        self._engine = None
        self._runtime = None
        self._stream = None
        self._loaded = False
        log.info("tensorrt_engine_unloaded", model=self._cfg.model_path)


class _ONNXBackend:
    """ONNX Runtime execution backend with CUDA EP → CPU EP fallback.

    Uses ``onnxruntime.InferenceSession`` with the ``CUDAExecutionProvider``
    as primary and ``CPUExecutionProvider`` as fallback.
    """

    def __init__(self, cfg: EngineConfig) -> None:
        self._cfg = cfg
        self._session = None
        self._input_name: str = ""
        self._loaded = False
        self._active_ep: str = "unknown"

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Create an inference session with GPU-first provider selection."""
        import onnxruntime as ort

        providers: list[str | tuple] = []
        if self._cfg.device == "cuda":
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {"device_id": self._cfg.device_index},
                )
            )
        providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.enable_mem_pattern = True

        self._session = ort.InferenceSession(
            self._cfg.model_path,
            sess_options=sess_opts,
            providers=providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        active = self._session.get_providers()
        self._active_ep = active[0] if active else "unknown"
        self._loaded = True
        log.info(
            "onnx_session_loaded",
            model=self._cfg.model_path,
            ep=self._active_ep,
        )

    def infer(self, blob: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run synchronous inference.  Returns raw output tensor."""
        outputs = self._session.run(None, {self._input_name: blob})
        return outputs[0]

    def unload(self) -> None:
        """Release the ONNX session and its associated device memory."""
        if not self._loaded:
            return
        del self._session
        self._session = None
        self._loaded = False
        log.info("onnx_session_unloaded", model=self._cfg.model_path)


# ── Public inference engine ───────────────────────────────────────


class InferenceEngine:
    """High-level inference engine that wraps a TensorRT or ONNX backend.

    Implements the full YOLO inference pipeline:

    1. Letterbox preprocessing (CPU, NumPy)
    2. Forward pass (GPU via backend)
    3. Post-processing: transpose → confidence gate → NMS → BoundingBox DTOs

    The engine exposes explicit ``warmup()`` / ``release()`` for VRAM
    lifecycle control.  Between events the orchestrator may call
    ``release()`` to yield Region C memory, then ``warmup()`` again
    before the next detection batch.

    Parameters
    ----------
    config : EngineConfig
        Model path, format, input size, thresholds, class mapping.
    stage_label : str
        Prometheus label for latency recording (e.g. ``"S2"``, ``"S3"``).
    """

    def __init__(self, config: EngineConfig, *, stage_label: str = "") -> None:
        self._cfg = config
        self._stage_label = stage_label
        self._classes = config.classes or {}

        if config.model_format == "tensorrt":
            self._backend: _TensorRTBackend | _ONNXBackend = _TensorRTBackend(config)
        elif config.model_format == "onnx":
            self._backend = _ONNXBackend(config)
        else:
            raise ValueError(f"Unsupported model format: {config.model_format!r}")

    # ── Properties ────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        """``True`` if backend weights are resident on the device."""
        return self._backend.loaded

    @property
    def model_format(self) -> str:
        return self._cfg.model_format

    # ── Lifecycle ─────────────────────────────────────────────────

    def load(self) -> None:
        """Load model weights onto the device (allocates VRAM)."""
        if self._backend.loaded:
            return
        self._backend.load()

    def unload(self) -> None:
        """Unload model weights and free all device memory."""
        if not self._backend.loaded:
            return
        self._backend.unload()
        _try_empty_cuda_cache()

    # ── Inference ─────────────────────────────────────────────────

    def predict(
        self,
        image: NDArray[np.uint8],
        *,
        confidence_threshold: float | None = None,
        nms_iou_threshold: float | None = None,
        class_filter: Sequence[int] | None = None,
    ) -> List[BoundingBox]:
        """Run the full detect pipeline: preprocess → infer → postprocess.

        Parameters
        ----------
        image : (H, W, 3) uint8 BGR array on CPU.
        confidence_threshold : Override engine-level threshold.
        nms_iou_threshold : Override engine-level NMS IoU.
        class_filter : If given, keep only these class IDs.

        Returns
        -------
        List of ``BoundingBox`` ordered by descending confidence.

        Raises
        ------
        VRAMError
            If the backend is not loaded (weights were released).
        """
        if not self._backend.loaded:
            raise VRAMError("Inference engine not loaded — call warmup() first")

        conf_thr = confidence_threshold or self._cfg.confidence_threshold
        iou_thr = nms_iou_threshold or self._cfg.nms_iou_threshold
        th, tw = self._cfg.input_size

        # 1. Letterbox preprocess (CPU)
        blob, ratio, (pad_w, pad_h) = letterbox(image, (th, tw))

        # 2. Forward pass (GPU or CPU depending on backend)
        t0 = time.perf_counter()
        raw = self._backend.infer(blob)
        infer_ms = (time.perf_counter() - t0) * 1000
        if self._stage_label:
            STAGE_LATENCY.labels(stage=f"{self._stage_label}_infer").observe(
                infer_ms / 1000
            )

        # 3. Post-process: decode YOLO output → BoundingBox DTOs
        return self._postprocess(
            raw,
            ratio=ratio,
            pad=(pad_w, pad_h),
            orig_shape=image.shape[:2],
            conf_thr=conf_thr,
            iou_thr=iou_thr,
            class_filter=class_filter,
        )

    # ── Post-processing ───────────────────────────────────────────

    def _postprocess(
        self,
        raw: NDArray[np.float32],
        *,
        ratio: float,
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int],
        conf_thr: float,
        iou_thr: float,
        class_filter: Sequence[int] | None,
    ) -> List[BoundingBox]:
        """Decode raw YOLO output into ``BoundingBox`` DTOs.

        Handles both YOLOv8 output formats:
          * ``(1, 84, N)`` — standard ultralytics export (needs transpose)
          * ``(1, N, 84)`` — some ONNX exports
        where 84 = 4 (xywh) + 80 (or fewer) class scores.
        """
        # Ensure shape is (N, 4+num_classes)
        if raw.ndim == 3:
            raw = raw[0]  # remove batch dim → (84, N) or (N, 84)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T  # (84, N) → (N, 84)

        # Split xywh and class scores
        xywh = raw[:, :4]
        class_scores = raw[:, 4:]
        num_classes = class_scores.shape[1]

        # Best class per detection
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Confidence filter
        mask = confidences >= conf_thr
        if class_filter is not None:
            class_mask = np.isin(class_ids, class_filter)
            mask &= class_mask
        if not np.any(mask):
            return []

        xywh = xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert xywh (centre) → xyxy (corners)
        cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # Reverse letterbox: remove padding, then un-scale
        pad_w, pad_h = pad
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= ratio

        # Clip to original image dimensions
        oh, ow = orig_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh)

        # NMS
        keep = _nms_numpy(boxes, confidences, iou_thr)
        boxes = boxes[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        # Sort by descending confidence
        order = confidences.argsort()[::-1]

        results: List[BoundingBox] = []
        for idx in order:
            cid = int(class_ids[idx])
            results.append(
                BoundingBox(
                    x1=int(round(boxes[idx, 0])),
                    y1=int(round(boxes[idx, 1])),
                    x2=int(round(boxes[idx, 2])),
                    y2=int(round(boxes[idx, 3])),
                    confidence=float(confidences[idx]),
                    class_id=cid,
                    class_name=self._classes.get(cid, str(cid)),
                )
            )

        return results


# ── Utilities ─────────────────────────────────────────────────────


def _try_empty_cuda_cache() -> None:
    """Best-effort call to ``torch.cuda.empty_cache()`` if torch is available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
