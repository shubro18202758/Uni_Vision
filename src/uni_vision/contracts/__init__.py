"""Public contract surface — import Protocols and DTOs from here."""

from uni_vision.contracts.detector import Detector
from uni_vision.contracts.dispatcher import Dispatcher
from uni_vision.contracts.dtos import (
    BoundingBox,
    CameraSource,
    CircuitState,
    DetectionContext,
    DetectionRecord,
    FramePacket,
    GPUTelemetry,
    HealthStatus,
    OCRResult,
    OffloadMode,
    ProcessedResult,
    ValidationStatus,
    VehicleClass,
    VRAMRegionSnapshot,
)
from uni_vision.contracts.frame_source import FrameSource
from uni_vision.contracts.ocr_engine import OCREngine
from uni_vision.contracts.post_processor import PostProcessor
from uni_vision.contracts.preprocessor import Preprocessor

__all__ = [
    # DTOs & Enums
    "BoundingBox",
    "CameraSource",
    "CircuitState",
    "DetectionContext",
    "DetectionRecord",
    # Protocols
    "Detector",
    "Dispatcher",
    "FramePacket",
    "FrameSource",
    "GPUTelemetry",
    "HealthStatus",
    "OCREngine",
    "OCRResult",
    "OffloadMode",
    "PostProcessor",
    "Preprocessor",
    "ProcessedResult",
    "VRAMRegionSnapshot",
    "ValidationStatus",
    "VehicleClass",
]
