"""Abstract base classes for all CV pipeline components.

Every model, library, or algorithmic package — whether a HuggingFace
transformer, a YOLO detector, or a hand-crafted preprocessing
algorithm — implements the ``CVComponent`` protocol.  This gives the
Manager Agent a uniform interface to load, execute, and unload any
component regardless of its origin.

Design principles:
  * Stateless ``execute()`` — all context flows through parameters.
  * Explicit ``load()`` / ``unload()`` for VRAM-aware lifecycle.
  * Capability tags drive automatic pipeline composition.
  * Resource estimates let the lifecycle manager respect VRAM budget.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any

# ── Enumerations ──────────────────────────────────────────────────


class ComponentType(str, enum.Enum):
    """Broad classification of a pipeline component."""

    MODEL = "model"  # ML model (YOLO, transformer, etc.)
    LIBRARY = "library"  # Algorithmic library (EasyOCR, PaddleOCR)
    ALGORITHM = "algorithm"  # Pure-code algorithm (NMS, deskew, regex)
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"


class ComponentState(str, enum.Enum):
    """Runtime lifecycle state."""

    UNREGISTERED = "unregistered"
    REGISTERED = "registered"  # Known to registry but not loaded
    DOWNLOADING = "downloading"
    LOADING = "loading"
    READY = "ready"  # Loaded and executable
    UNLOADING = "unloading"
    FAILED = "failed"
    SUSPENDED = "suspended"  # Offloaded to CPU / disk to free VRAM


class ComponentCapability(str, enum.Enum):
    """Semantic capability tags used for pipeline composition.

    The Manager Agent matches frame context requirements against these
    capabilities to decide which components to load.
    """

    # Detection
    VEHICLE_DETECTION = "vehicle_detection"
    PLATE_DETECTION = "plate_detection"
    PERSON_DETECTION = "person_detection"
    OBJECT_DETECTION = "object_detection"
    FACE_DETECTION = "face_detection"
    FIRE_DETECTION = "fire_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    ZERO_SHOT_DETECTION = "zero_shot_detection"

    # OCR / text
    PLATE_OCR = "plate_ocr"
    SCENE_TEXT_OCR = "scene_text_ocr"
    DOCUMENT_OCR = "document_ocr"

    # Classification
    VEHICLE_CLASSIFICATION = "vehicle_classification"
    SCENE_CLASSIFICATION = "scene_classification"
    ACTION_RECOGNITION = "action_recognition"

    # Preprocessing
    IMAGE_DESKEW = "image_deskew"
    IMAGE_ENHANCE = "image_enhance"
    IMAGE_DENOISE = "image_denoise"
    IMAGE_DENOISING = "image_denoising"  # Alias for pipeline_composer
    GEOMETRIC_CORRECTION = "geometric_correction"
    ROI_EXTRACTION = "roi_extraction"
    SUPER_RESOLUTION = "super_resolution"

    # Postprocessing
    TEXT_VALIDATION = "text_validation"
    TEXT_CORRECTION = "text_correction"
    PLATE_LOCALIZATION = "plate_localization"
    PLATE_VALIDATION = "plate_validation"
    DEDUPLICATION = "deduplication"
    TRACKING = "tracking"

    # Segmentation
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"

    # Advanced 3D / spatial
    DEPTH_ESTIMATION = "depth_estimation"
    POSE_ESTIMATION = "pose_estimation"


# ── Resource estimate ─────────────────────────────────────────────


@dataclass(frozen=True)
class ResourceEstimate:
    """Expected GPU / CPU resources required to load a component."""

    vram_mb: int = 0
    ram_mb: int = 0
    supports_gpu: bool = True
    supports_cpu: bool = True
    supports_half: bool = False  # FP16 capable
    supports_int8: bool = False  # INT8 quantised variant available


# ── Component metadata ────────────────────────────────────────────


@dataclass
class ComponentMetadata:
    """Descriptive metadata attached to every component."""

    component_id: str  # Unique identifier
    name: str  # Human-readable name
    version: str = "0.0.0"
    component_type: ComponentType = ComponentType.MODEL
    capabilities: set[ComponentCapability] = field(default_factory=set)
    source: str = "builtin"  # builtin | huggingface | pypi | github
    source_id: str = ""  # e.g. "ultralytics/yolov8n"
    description: str = ""
    license: str = ""
    python_requirements: list[str] = field(default_factory=list)
    resource_estimate: ResourceEstimate = field(default_factory=ResourceEstimate)
    tags: dict[str, str] = field(default_factory=dict)
    trusted: bool = False  # Only trusted sources are auto-loaded


# ── Abstract Component ────────────────────────────────────────────


class CVComponent(abc.ABC):
    """Abstract base for every component in the dynamic pipeline.

    Subclasses MUST implement:
      * ``metadata`` — static component description
      * ``load()``   — allocate resources (GPU memory, model weights)
      * ``unload()`` — free all resources
      * ``execute()`` — run inference / processing on input data

    Optionally override:
      * ``warmup()``     — pre-inference warm-up pass
      * ``health_check()`` — periodic health probe
      * ``get_runtime_info()`` — live telemetry
    """

    @property
    @abc.abstractmethod
    def metadata(self) -> ComponentMetadata:
        """Return immutable component metadata."""

    @property
    def state(self) -> ComponentState:
        """Current lifecycle state."""
        return self._state

    def __init__(self) -> None:
        self._state: ComponentState = ComponentState.REGISTERED
        self._load_error: str | None = None

    # ── Lifecycle ─────────────────────────────────────────────────

    @abc.abstractmethod
    async def load(self, *, device: str = "cuda") -> None:
        """Allocate resources and become READY.

        Parameters
        ----------
        device:
            Target device — ``"cuda"`` or ``"cpu"``.  Components that
            don't support the requested device should fall back or raise.
        """

    @abc.abstractmethod
    async def unload(self) -> None:
        """Release all resources (VRAM, RAM, file handles)."""

    @abc.abstractmethod
    async def execute(
        self,
        data: Any,
        *,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Run the component on input data.

        Parameters
        ----------
        data:
            Input — typically a numpy ndarray (image) but can be any
            type depending on component capabilities.
        context:
            Optional dict of runtime hints (camera_id, timestamp, etc.)

        Returns
        -------
        Any
            Output appropriate to the component's capability — bounding
            boxes, OCR text, enhanced image, classification labels, etc.
        """

    async def warmup(self) -> None:
        """Optional warm-up pass (e.g. dummy inference to JIT-compile)."""

    async def health_check(self) -> bool:
        """Return True if the component is healthy and operational."""
        return self._state == ComponentState.READY

    def get_runtime_info(self) -> dict[str, Any]:
        """Return live telemetry / debug information."""
        return {
            "component_id": self.metadata.component_id,
            "state": self._state.value,
            "type": self.metadata.component_type.value,
            "capabilities": [c.value for c in self.metadata.capabilities],
        }

    # ── State transitions ─────────────────────────────────────────

    def _set_state(self, new_state: ComponentState) -> None:
        self._state = new_state

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.metadata.component_id!r} state={self._state.value}>"
