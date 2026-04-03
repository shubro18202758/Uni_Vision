"""Detection package — S2 vehicle detection + S3 plate sub-detection.

Public API:
    InferenceEngine, EngineConfig - low-level TensorRT / ONNX backend
    VehicleDetector              - S2 vehicle localisation
    PlateDetector                - S3 plate localisation within ROI
    GPUMemoryManager             - VRAM offload controller + memory fence
"""

from uni_vision.detection.engine import EngineConfig, InferenceEngine
from uni_vision.detection.gpu_memory import GPUMemoryManager
from uni_vision.detection.plate_detector import PlateDetector
from uni_vision.detection.vehicle_detector import VehicleDetector

__all__ = [
    "EngineConfig",
    "GPUMemoryManager",
    "InferenceEngine",
    "PlateDetector",
    "VehicleDetector",
]
