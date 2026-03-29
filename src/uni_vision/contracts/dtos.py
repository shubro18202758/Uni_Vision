"""Immutable data transfer objects for all inter-stage communication.

Every DTO is a frozen dataclass. No stage mutates data produced by a
previous stage — it produces a new DTO. numpy arrays are exempt from
true immutability (frozen stops attribute reassignment, not buffer
mutation), but the convention is enforced by code review.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


class VehicleClass(str, Enum):
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"


class ValidationStatus(str, Enum):
    VALID = "valid"
    LOW_CONFIDENCE = "low_confidence"
    REGEX_FAIL = "regex_fail"
    LLM_ERROR = "llm_error"
    FALLBACK = "fallback"
    PARSE_FAIL = "parse_fail"
    UNREADABLE = "unreadable"


class OffloadMode(str, Enum):
    """Hardware offloading mode — see spec §7.2."""

    GPU_PRIMARY = "gpu_primary"
    PARTIAL_OFFLOAD = "partial_offload"
    FULL_CPU = "full_cpu"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ── Core pipeline DTOs ────────────────────────────────────────────


@dataclass(frozen=True)
class FramePacket:
    """Output of S0/S1 — a single ingested, sampled frame."""

    camera_id: str
    timestamp_utc: float
    frame_index: int
    image: NDArray[np.uint8]  # (H, W, 3) BGR


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box with detection metadata."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class DetectionContext:
    """Accumulated context passed to the OCR stage."""

    camera_id: str
    timestamp_utc: float
    vehicle_bbox: BoundingBox
    plate_bbox: BoundingBox
    vehicle_class: str


@dataclass(frozen=True)
class OCRResult:
    """Output of S7 — structured OCR extraction."""

    plate_text: str
    raw_text: str
    confidence: float
    reasoning: str
    engine: str
    status: ValidationStatus


@dataclass(frozen=True)
class ProcessedResult:
    """Output of post-processing validation."""

    plate_text: str
    raw_ocr_text: str
    confidence: float
    validation_status: ValidationStatus
    char_corrections: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionRecord:
    """Final record persisted to storage and dispatched to consumers."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    camera_id: str = ""
    plate_number: str = ""
    raw_ocr_text: str = ""
    ocr_confidence: float = 0.0
    ocr_engine: str = ""
    vehicle_class: str = ""
    vehicle_image_path: str = ""
    plate_image_path: str = ""
    detected_at_utc: str = ""
    validation_status: str = ""
    location_tag: str = ""


@dataclass(frozen=True)
class AnalysisResult:
    """Structured anomaly analysis result from LLM vision analysis."""

    frame_id: str = ""
    camera_id: str = ""
    timestamp_utc: str = ""
    scene_description: str = ""
    objects_detected: List[Dict[str, str]] = field(default_factory=list)
    anomaly_detected: bool = False
    anomalies: List[Dict[str, str]] = field(default_factory=list)
    chain_of_thought: str = ""
    risk_level: str = "low"          # low | medium | high | critical
    risk_analysis: str = ""
    impact_analysis: str = ""
    confidence: float = 0.0
    recommendations: List[str] = field(default_factory=list)


# ── Hardware telemetry DTOs ───────────────────────────────────────


@dataclass(frozen=True)
class VRAMRegionSnapshot:
    """Point-in-time snapshot of a single VRAM budget region."""

    region_name: str
    budget_mb: float
    used_mb: float
    utilisation_pct: float


@dataclass(frozen=True)
class GPUTelemetry:
    """Aggregated GPU telemetry sample — VRAM usage, PCIe throughput, temp."""

    timestamp_utc: float
    device_index: int
    device_name: str
    vram_total_mb: float
    vram_used_mb: float
    vram_free_mb: float
    vram_utilisation_pct: float
    gpu_utilisation_pct: float
    temperature_c: int
    pcie_tx_kbps: int
    pcie_rx_kbps: int
    regions: List[VRAMRegionSnapshot] = field(default_factory=list)


@dataclass(frozen=True)
class HealthStatus:
    """System-wide health check result."""

    healthy: bool
    gpu_available: bool
    ollama_reachable: bool
    database_connected: bool
    streams_connected: int
    streams_total: int
    offload_mode: OffloadMode
    mode: str = "pipeline"
    details: Dict[str, str] = field(default_factory=dict)


# ── Camera source DTO ─────────────────────────────────────────────


@dataclass(frozen=True)
class CameraSource:
    """Parsed camera definition from cameras.yaml."""

    camera_id: str
    source_url: str
    location_tag: str
    fps_target: int
    enabled: bool
