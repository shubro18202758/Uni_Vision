"""Concrete component wrappers for built-in and external modules.

These wrappers adapt existing Uni_Vision modules (YOLO detectors,
EasyOCR, Hough straightener, etc.) into the ``CVComponent`` interface
so the Manager Agent can treat them identically to externally-fetched
HuggingFace models or PyPI packages.

External / dynamic components are wrapped by ``HuggingFaceModelComponent``,
``PipPackageComponent``, and ``TorchHubComponent`` which download and
load at runtime with timeout and health-check support.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from uni_vision.components.base import (
    CVComponent,
    ComponentCapability,
    ComponentMetadata,
    ComponentState,
    ComponentType,
    ResourceEstimate,
)

log = logging.getLogger(__name__)

# Default timeout for model loading operations (seconds)
_LOAD_TIMEOUT_SECONDS = 120


# ═══════════════════════════════════════════════════════════════════
# Built-in wrappers — adapt existing Uni_Vision modules
# ═══════════════════════════════════════════════════════════════════


class BuiltinDetectorComponent(CVComponent):
    """Wraps an existing Uni_Vision Detector (vehicle or plate)."""

    def __init__(
        self,
        *,
        component_id: str,
        name: str,
        detector_instance: Any,
        capabilities: Set[ComponentCapability],
        vram_mb: int = 300,
    ) -> None:
        super().__init__()
        self._detector = detector_instance
        self._meta = ComponentMetadata(
            component_id=component_id,
            name=name,
            version="1.0.0",
            component_type=ComponentType.MODEL,
            capabilities=capabilities,
            source="builtin",
            description=f"Built-in {name}",
            resource_estimate=ResourceEstimate(vram_mb=vram_mb),
            trusted=True,
        )

    @property
    def metadata(self) -> ComponentMetadata:
        return self._meta

    async def load(self, *, device: str = "cuda") -> None:
        self._set_state(ComponentState.LOADING)
        try:
            self._detector.warmup()
            self._set_state(ComponentState.READY)
        except Exception as exc:
            self._load_error = str(exc)
            self._set_state(ComponentState.FAILED)
            raise

    async def unload(self) -> None:
        self._set_state(ComponentState.UNLOADING)
        try:
            self._detector.release()
        finally:
            self._set_state(ComponentState.REGISTERED)

    async def execute(
        self,
        data: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        pipeline_data = (context or {}).get("_pipeline_data", {})

        # ANPR plate-detection mode: when the stage receives the frame
        # but vehicle detections are available, detect plates in the full
        # frame and return plate crops for downstream OCR.
        if (
            ComponentCapability.PLATE_DETECTION in self._meta.capabilities
            and isinstance(data, np.ndarray)
        ):
            plate_bboxes = self._detector.detect(data)
            if not plate_bboxes:
                return []
            # Return structured plate crops so downstream stages
            # (preprocessors / OCR) can iterate over them.
            vehicle_detections = pipeline_data.get("detections", [])
            crops: list[Dict[str, Any]] = []
            for pbbox in plate_bboxes:
                crop = self._extract_crop(data, pbbox)
                if crop is not None:
                    # Try to associate with nearest vehicle detection
                    v_class = self._match_vehicle(pbbox, vehicle_detections)
                    crops.append({
                        "crop": crop,
                        "plate_bbox": pbbox,
                        "vehicle_class": v_class,
                    })
            return crops

        return self._detector.detect(data)

    # ── Helpers for plate detection mode ──────────────────────────

    @staticmethod
    def _extract_crop(
        frame: np.ndarray,
        bbox: Any,
        padding_px: int = 4,
    ) -> Optional[np.ndarray]:
        """Crop a bounding box region from *frame* with padding."""
        h, w = frame.shape[:2]
        try:
            x1 = max(0, int(bbox.x1) - padding_px)
            y1 = max(0, int(bbox.y1) - padding_px)
            x2 = min(w, int(bbox.x2) + padding_px)
            y2 = min(h, int(bbox.y2) + padding_px)
        except AttributeError:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def _match_vehicle(plate_bbox: Any, vehicle_detections: Any) -> str:
        """Best-effort match a plate bbox to the nearest vehicle."""
        if not vehicle_detections or not isinstance(vehicle_detections, list):
            return "unknown"
        try:
            px = (plate_bbox.x1 + plate_bbox.x2) / 2
            py = (plate_bbox.y1 + plate_bbox.y2) / 2
            best, best_cls = float("inf"), "unknown"
            for vdet in vehicle_detections:
                vx = (vdet.x1 + vdet.x2) / 2
                vy = (vdet.y1 + vdet.y2) / 2
                dist = (px - vx) ** 2 + (py - vy) ** 2
                if dist < best:
                    best = dist
                    best_cls = getattr(vdet, "class_name", "unknown")
            return best_cls
        except Exception:
            return "unknown"


class BuiltinPreprocessorComponent(CVComponent):
    """Wraps an existing Uni_Vision Preprocessor (deskew, enhance)."""

    def __init__(
        self,
        *,
        component_id: str,
        name: str,
        preprocessor_instance: Any,
        capabilities: Set[ComponentCapability],
    ) -> None:
        super().__init__()
        self._preprocessor = preprocessor_instance
        self._meta = ComponentMetadata(
            component_id=component_id,
            name=name,
            version="1.0.0",
            component_type=ComponentType.PREPROCESSOR,
            capabilities=capabilities,
            source="builtin",
            description=f"Built-in {name}",
            resource_estimate=ResourceEstimate(vram_mb=0, supports_gpu=False),
            trusted=True,
        )

    @property
    def metadata(self) -> ComponentMetadata:
        return self._meta

    async def load(self, *, device: str = "cuda") -> None:
        self._set_state(ComponentState.READY)

    async def unload(self) -> None:
        self._set_state(ComponentState.REGISTERED)

    async def execute(
        self,
        data: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        # Polymorphic: if data is a list of plate-crop dicts from
        # BuiltinDetectorComponent, process each crop individually.
        if isinstance(data, list) and data and isinstance(data[0], dict) and "crop" in data[0]:
            for item in data:
                item["crop"] = self._preprocessor.process(item["crop"])
            return data
        return self._preprocessor.process(data)


class BuiltinOCRComponent(CVComponent):
    """Wraps the OCR strategy (multi-engine, Manager provisions more at runtime)."""

    def __init__(
        self,
        *,
        ocr_strategy_instance: Any,
    ) -> None:
        super().__init__()
        self._ocr = ocr_strategy_instance
        self._meta = ComponentMetadata(
            component_id="builtin.ocr_strategy",
            name="OCR Strategy (EasyOCR + dynamic engines)",
            version="1.0.0",
            component_type=ComponentType.LIBRARY,
            capabilities={ComponentCapability.PLATE_OCR, ComponentCapability.SCENE_TEXT_OCR},
            source="builtin",
            description="Multi-engine OCR strategy — EasyOCR default, Manager provisions additional engines at runtime",
            resource_estimate=ResourceEstimate(vram_mb=0),
            trusted=True,
        )

    @property
    def metadata(self) -> ComponentMetadata:
        return self._meta

    async def load(self, *, device: str = "cuda") -> None:
        self._set_state(ComponentState.READY)

    async def unload(self) -> None:
        self._set_state(ComponentState.REGISTERED)

    async def execute(
        self,
        data: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        # Polymorphic: if data is a list of plate-crop dicts,
        # run OCR on each crop and annotate results.
        if isinstance(data, list) and data and isinstance(data[0], dict) and "crop" in data[0]:
            results: list[Dict[str, Any]] = []
            for item in data:
                ocr_result = await self._ocr.extract(item["crop"], context)
                results.append({
                    "plate_text": getattr(ocr_result, "text", str(ocr_result)),
                    "raw_ocr_text": getattr(ocr_result, "raw_text", getattr(ocr_result, "text", str(ocr_result))),
                    "confidence": getattr(ocr_result, "confidence", 0.0),
                    "engine": getattr(ocr_result, "engine", "unknown"),
                    "plate_bbox": item.get("plate_bbox"),
                    "vehicle_class": item.get("vehicle_class", "unknown"),
                })
            return results
        return await self._ocr.extract(data, context)


class BuiltinPostprocessorComponent(CVComponent):
    """Wraps the CognitiveOrchestrator (validator + adjudicator)."""

    def __init__(
        self,
        *,
        component_id: str,
        name: str,
        postprocessor_instance: Any,
        capabilities: Set[ComponentCapability],
    ) -> None:
        super().__init__()
        self._postprocessor = postprocessor_instance
        self._meta = ComponentMetadata(
            component_id=component_id,
            name=name,
            version="1.0.0",
            component_type=ComponentType.POSTPROCESSOR,
            capabilities=capabilities,
            source="builtin",
            description=f"Built-in {name}",
            resource_estimate=ResourceEstimate(vram_mb=0),
            trusted=True,
        )

    @property
    def metadata(self) -> ComponentMetadata:
        return self._meta

    async def load(self, *, device: str = "cuda") -> None:
        self._set_state(ComponentState.READY)

    async def unload(self) -> None:
        self._set_state(ComponentState.REGISTERED)

    async def execute(
        self,
        data: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        # For postprocessors, data is (ocr_result, plate_image)
        if isinstance(data, tuple) and len(data) == 2:
            return await self._postprocessor.validate(data[0], data[1])
        return await self._postprocessor.validate(data, None)


# ═══════════════════════════════════════════════════════════════════
# Dynamic wrappers — for externally-discovered components
# ═══════════════════════════════════════════════════════════════════


class HuggingFaceModelComponent(CVComponent):
    """Dynamically loads a model from HuggingFace Hub at runtime.

    The Manager Agent creates instances of this class when it decides
    a new model is needed for the current frame context.  The model
    is downloaded, cached locally, and loaded onto the appropriate
    device respecting the VRAM budget.

    Supports four load patterns (set via ``load_pattern``):
    - ``from_pretrained`` (default) — ``Cls.from_pretrained(repo_id)``
    - ``constructor``              — ``Cls(repo_id)``  (e.g. YOLO)
    - ``hub_load``                 — ``torch.hub.load(hub_repo, model_name)``
    - ``create_model``             — ``timm.create_model(name, pretrained=True)``
    """

    # Load patterns supported by this wrapper
    PATTERNS = frozenset({"from_pretrained", "constructor", "hub_load", "create_model"})

    def __init__(
        self,
        *,
        component_id: str,
        name: str,
        repo_id: str,
        model_class: str,
        capabilities: Set[ComponentCapability],
        vram_mb: int = 500,
        python_requirements: Optional[List[str]] = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        trusted: bool = False,
        load_pattern: str = "from_pretrained",
        hub_repo: Optional[str] = None,
        load_timeout: int = _LOAD_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__()
        self._repo_id = repo_id
        self._model_class = model_class
        self._load_kwargs = load_kwargs or {}
        self._model: Any = None
        self._device: str = "cpu"
        self._load_pattern = load_pattern if load_pattern in self.PATTERNS else "from_pretrained"
        self._hub_repo = hub_repo  # e.g. "ultralytics/yolov5" for hub_load
        self._load_timeout = load_timeout
        self._meta = ComponentMetadata(
            component_id=component_id,
            name=name,
            component_type=ComponentType.MODEL,
            capabilities=capabilities,
            source="huggingface",
            source_id=repo_id,
            python_requirements=python_requirements or [],
            resource_estimate=ResourceEstimate(vram_mb=vram_mb),
            trusted=trusted,
        )

    @property
    def metadata(self) -> ComponentMetadata:
        return self._meta

    async def load(self, *, device: str = "cuda") -> None:
        self._set_state(ComponentState.LOADING)
        self._device = device
        try:
            loop = asyncio.get_event_loop()
            self._model = await asyncio.wait_for(
                loop.run_in_executor(None, self._load_model_sync, device),
                timeout=self._load_timeout,
            )
            # Post-load health check: ensure model can accept a dummy input
            await loop.run_in_executor(None, self._health_check)
            self._set_state(ComponentState.READY)
            log.info("hf_model_loaded repo_id=%s device=%s pattern=%s",
                     self._repo_id, device, self._load_pattern)
        except asyncio.TimeoutError:
            self._load_error = f"Load timed out after {self._load_timeout}s"
            self._set_state(ComponentState.FAILED)
            raise RuntimeError(self._load_error)
        except Exception as exc:
            self._load_error = str(exc)
            self._set_state(ComponentState.FAILED)
            raise

    def _load_model_sync(self, device: str) -> Any:
        """Synchronous model loading (run in executor).

        Dispatches to the correct loading strategy based on
        ``self._load_pattern``.
        """
        if self._load_pattern == "hub_load":
            return self._load_via_hub(device)
        if self._load_pattern == "create_model":
            return self._load_via_create_model(device)
        if self._load_pattern == "constructor":
            return self._load_via_constructor(device)
        # Default: from_pretrained
        return self._load_via_from_pretrained(device)

    # ── Loading strategies ──────────────────────────────────────

    def _load_via_from_pretrained(self, device: str) -> Any:
        module_path, class_name = self._model_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        model = cls.from_pretrained(self._repo_id, **self._load_kwargs)
        return self._to_device(model, device)

    def _load_via_constructor(self, device: str) -> Any:
        """e.g. ``ultralytics.YOLO("yolov8n.pt")``."""
        module_path, class_name = self._model_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        model = cls(self._repo_id, **self._load_kwargs)
        return self._to_device(model, device)

    def _load_via_hub(self, device: str) -> Any:
        """``torch.hub.load("ultralytics/yolov5", "yolov5s")``."""
        import torch  # noqa: F811
        hub_repo = self._hub_repo or self._repo_id
        # The model_class here doubles as the model name for torch.hub
        _, model_name = (self._model_class.rsplit(".", 1)
                         if "." in self._model_class
                         else ("", self._model_class))
        model = torch.hub.load(hub_repo, model_name, **self._load_kwargs)
        return self._to_device(model, device)

    def _load_via_create_model(self, device: str) -> Any:
        """``timm.create_model("resnet50", pretrained=True)``."""
        import timm  # noqa: F811
        model = timm.create_model(self._repo_id, pretrained=True, **self._load_kwargs)
        return self._to_device(model, device)

    @staticmethod
    def _to_device(model: Any, device: str) -> Any:
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    # ── Health check ────────────────────────────────────────────

    def _health_check(self) -> None:
        """Verify the loaded model can accept a dummy input without crashing."""
        if self._model is None:
            return
        try:
            dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
            if hasattr(self._model, "predict"):
                self._model.predict(dummy, verbose=False)
            elif hasattr(self._model, "__call__"):
                import torch
                with torch.no_grad():
                    self._model(torch.from_numpy(dummy).to(self._device))
        except Exception:
            # Health check is best-effort — some models need specific
            # input shapes or types.  Log but don't fail.
            log.debug("health_check skipped (non-standard input): %s", self._repo_id)

    async def unload(self) -> None:
        self._set_state(ComponentState.UNLOADING)
        if self._model is not None:
            if hasattr(self._model, "cpu"):
                self._model.cpu()
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        self._set_state(ComponentState.REGISTERED)
        log.info("hf_model_unloaded repo_id=%s", self._repo_id)

    async def execute(
        self,
        data: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if self._model is None:
            raise RuntimeError(f"Component {self.metadata.component_id} not loaded")
        return await asyncio.get_event_loop().run_in_executor(
            None, self._infer_sync, data, context,
        )

    def _infer_sync(self, data: Any, context: Optional[Dict[str, Any]]) -> Any:
        """Synchronous inference (run in executor)."""
        if hasattr(self._model, "predict"):
            return self._model.predict(data)
        elif hasattr(self._model, "__call__"):
            return self._model(data)
        raise RuntimeError(f"Model {self._repo_id} has no predict or __call__ method")


class PipPackageComponent(CVComponent):
    """Wraps a pip-installable package as a pipeline component.

    Used for libraries like PaddleOCR, mmdetection, etc. that the
    Manager Agent discovers and decides to integrate.  The package
    is installed into the current environment and loaded dynamically.

    ``entry_class`` may be a fully-qualified dotted path such as
    ``"paddleocr.PaddleOCR"`` — in that case ``entry_module`` is
    inferred automatically (the part before the last dot).
    """

    def __init__(
        self,
        *,
        component_id: str,
        name: str,
        package_name: str,
        entry_module: str = "",
        entry_class: str,
        capabilities: Set[ComponentCapability],
        init_kwargs: Optional[Dict[str, Any]] = None,
        vram_mb: int = 0,
        trusted: bool = False,
        load_timeout: int = _LOAD_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__()
        self._package_name = package_name

        # Auto-split fully-qualified class path if entry_module is empty
        if not entry_module and "." in entry_class:
            self._entry_module, self._entry_class = entry_class.rsplit(".", 1)
        else:
            self._entry_module = entry_module
            self._entry_class = entry_class

        self._init_kwargs = init_kwargs or {}
        self._instance: Any = None
        self._load_timeout = load_timeout
        self._meta = ComponentMetadata(
            component_id=component_id,
            name=name,
            component_type=ComponentType.LIBRARY,
            capabilities=capabilities,
            source="pypi",
            source_id=package_name,
            python_requirements=[package_name],
            resource_estimate=ResourceEstimate(vram_mb=vram_mb),
            trusted=trusted,
        )

    @property
    def metadata(self) -> ComponentMetadata:
        return self._meta

    async def load(self, *, device: str = "cuda") -> None:
        self._set_state(ComponentState.LOADING)
        try:
            loop = asyncio.get_event_loop()
            self._instance = await asyncio.wait_for(
                loop.run_in_executor(None, self._init_sync, device),
                timeout=self._load_timeout,
            )
            self._set_state(ComponentState.READY)
            log.info("pip_component_loaded package=%s class=%s.%s",
                     self._package_name, self._entry_module, self._entry_class)
        except asyncio.TimeoutError:
            self._load_error = f"Load timed out after {self._load_timeout}s"
            self._set_state(ComponentState.FAILED)
            raise RuntimeError(self._load_error)
        except Exception as exc:
            self._load_error = str(exc)
            self._set_state(ComponentState.FAILED)
            raise

    def _init_sync(self, device: str) -> Any:
        module = importlib.import_module(self._entry_module)
        cls = getattr(module, self._entry_class)
        kwargs = {**self._init_kwargs}
        # Try to pass device kwarg if the class accepts it
        try:
            if "device" in cls.__init__.__code__.co_varnames:
                kwargs["device"] = device
        except (AttributeError, TypeError):
            pass
        return cls(**kwargs)

    async def unload(self) -> None:
        self._set_state(ComponentState.UNLOADING)
        del self._instance
        self._instance = None
        self._set_state(ComponentState.REGISTERED)

    async def execute(
        self,
        data: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if self._instance is None:
            raise RuntimeError(f"Component {self.metadata.component_id} not loaded")
        return await asyncio.get_event_loop().run_in_executor(
            None, self._run_sync, data, context,
        )

    def _run_sync(self, data: Any, context: Optional[Dict[str, Any]]) -> Any:
        for method_name in ("predict", "process", "detect", "ocr", "__call__"):
            fn = getattr(self._instance, method_name, None)
            if callable(fn):
                return fn(data)
        raise RuntimeError(
            f"Package {self._package_name} instance has no usable inference method"
        )


class TorchHubComponent(CVComponent):
    """Wraps a model loaded via ``torch.hub.load()`` at runtime.

    Used for models hosted on GitHub repos that follow the TorchHub
    protocol (e.g. ``ultralytics/yolov5``, ``facebookresearch/detr``).
    """

    def __init__(
        self,
        *,
        component_id: str,
        name: str,
        hub_repo: str,
        model_name: str,
        capabilities: Set[ComponentCapability],
        vram_mb: int = 500,
        python_requirements: Optional[List[str]] = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        trusted: bool = False,
        load_timeout: int = _LOAD_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__()
        self._hub_repo = hub_repo
        self._model_name = model_name
        self._load_kwargs = load_kwargs or {}
        self._model: Any = None
        self._device: str = "cpu"
        self._load_timeout = load_timeout
        self._meta = ComponentMetadata(
            component_id=component_id,
            name=name,
            component_type=ComponentType.MODEL,
            capabilities=capabilities,
            source="torchhub",
            source_id=f"{hub_repo}:{model_name}",
            python_requirements=python_requirements or [],
            resource_estimate=ResourceEstimate(vram_mb=vram_mb),
            trusted=trusted,
        )

    @property
    def metadata(self) -> ComponentMetadata:
        return self._meta

    async def load(self, *, device: str = "cuda") -> None:
        self._set_state(ComponentState.LOADING)
        self._device = device
        try:
            loop = asyncio.get_event_loop()
            self._model = await asyncio.wait_for(
                loop.run_in_executor(None, self._load_sync, device),
                timeout=self._load_timeout,
            )
            self._set_state(ComponentState.READY)
            log.info("torchhub_model_loaded repo=%s model=%s device=%s",
                     self._hub_repo, self._model_name, device)
        except asyncio.TimeoutError:
            self._load_error = f"TorchHub load timed out after {self._load_timeout}s"
            self._set_state(ComponentState.FAILED)
            raise RuntimeError(self._load_error)
        except Exception as exc:
            self._load_error = str(exc)
            self._set_state(ComponentState.FAILED)
            raise

    def _load_sync(self, device: str) -> Any:
        import torch
        model = torch.hub.load(self._hub_repo, self._model_name, **self._load_kwargs)
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    async def unload(self) -> None:
        self._set_state(ComponentState.UNLOADING)
        if self._model is not None:
            if hasattr(self._model, "cpu"):
                self._model.cpu()
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        self._set_state(ComponentState.REGISTERED)
        log.info("torchhub_model_unloaded repo=%s model=%s",
                 self._hub_repo, self._model_name)

    async def execute(
        self,
        data: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if self._model is None:
            raise RuntimeError(f"Component {self.metadata.component_id} not loaded")
        return await asyncio.get_event_loop().run_in_executor(
            None, self._infer_sync, data, context,
        )

    def _infer_sync(self, data: Any, context: Optional[Dict[str, Any]]) -> Any:
        if hasattr(self._model, "predict"):
            return self._model.predict(data)
        elif hasattr(self._model, "__call__"):
            return self._model(data)
        raise RuntimeError(
            f"TorchHub model {self._hub_repo}:{self._model_name} has no inference method"
        )
