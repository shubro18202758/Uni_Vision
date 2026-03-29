"""Tests for DTO immutability, enum values, and field defaults."""

from __future__ import annotations

import uuid

import numpy as np
import pytest


class TestEnums:
    """Verify all enum members match the spec."""

    def test_vehicle_class_members(self):
        from uni_vision.contracts.dtos import VehicleClass

        assert set(VehicleClass) == {
            VehicleClass.CAR,
            VehicleClass.TRUCK,
            VehicleClass.BUS,
            VehicleClass.MOTORCYCLE,
        }
        assert VehicleClass.CAR.value == "car"

    def test_validation_status_members(self):
        from uni_vision.contracts.dtos import ValidationStatus

        expected = {"valid", "low_confidence", "regex_fail", "llm_error",
                    "fallback", "parse_fail", "unreadable"}
        assert {v.value for v in ValidationStatus} == expected

    def test_offload_mode_members(self):
        from uni_vision.contracts.dtos import OffloadMode

        expected = {"gpu_primary", "partial_offload", "full_cpu"}
        assert {v.value for v in OffloadMode} == expected

    def test_circuit_state_members(self):
        from uni_vision.contracts.dtos import CircuitState

        expected = {"closed", "open", "half_open"}
        assert {v.value for v in CircuitState} == expected


class TestFramePacket:
    def test_creation(self, sample_bgr_image):
        from uni_vision.contracts.dtos import FramePacket

        pkt = FramePacket(
            camera_id="cam_01",
            timestamp_utc=1234567890.0,
            frame_index=42,
            image=sample_bgr_image,
        )
        assert pkt.camera_id == "cam_01"
        assert pkt.frame_index == 42
        assert pkt.image.shape == (64, 64, 3)

    def test_immutability(self, sample_bgr_image):
        from uni_vision.contracts.dtos import FramePacket

        pkt = FramePacket(
            camera_id="cam_01",
            timestamp_utc=0.0,
            frame_index=0,
            image=sample_bgr_image,
        )
        with pytest.raises(AttributeError):
            pkt.camera_id = "cam_02"  # type: ignore[misc]


class TestBoundingBox:
    def test_creation(self):
        from uni_vision.contracts.dtos import BoundingBox

        bb = BoundingBox(
            x1=10, y1=20, x2=100, y2=200,
            confidence=0.95, class_id=0, class_name="car",
        )
        assert bb.confidence == 0.95
        assert bb.class_name == "car"

    def test_frozen(self):
        from uni_vision.contracts.dtos import BoundingBox

        bb = BoundingBox(x1=0, y1=0, x2=1, y2=1, confidence=0.5,
                         class_id=0, class_name="truck")
        with pytest.raises(AttributeError):
            bb.x1 = 99  # type: ignore[misc]


class TestDetectionRecord:
    def test_default_id_is_uuid(self):
        from uni_vision.contracts.dtos import DetectionRecord

        rec = DetectionRecord()
        # Should be a valid UUID string
        uuid.UUID(rec.id)  # raises ValueError if invalid

    def test_field_defaults(self):
        from uni_vision.contracts.dtos import DetectionRecord

        rec = DetectionRecord()
        assert rec.camera_id == ""
        assert rec.plate_number == ""
        assert rec.ocr_confidence == 0.0

    def test_explicit_fields(self, sample_detection_record):
        assert sample_detection_record.plate_number == "MH12AB1234"
        assert sample_detection_record.ocr_confidence == 0.92


class TestOCRResult:
    def test_creation(self):
        from uni_vision.contracts.dtos import OCRResult, ValidationStatus

        r = OCRResult(
            plate_text="MH12AB1234",
            raw_text="mh12ab1234",
            confidence=0.88,
            reasoning="Clear image",
            engine="ollama_llm",
            status=ValidationStatus.VALID,
        )
        assert r.plate_text == "MH12AB1234"
        assert r.status == ValidationStatus.VALID


class TestProcessedResult:
    def test_default_corrections(self):
        from uni_vision.contracts.dtos import ProcessedResult, ValidationStatus

        pr = ProcessedResult(
            plate_text="MH12AB1234",
            raw_ocr_text="MH12AB1234",
            confidence=0.9,
            validation_status=ValidationStatus.VALID,
        )
        assert pr.char_corrections == {}


class TestHealthStatus:
    def test_creation(self):
        from uni_vision.contracts.dtos import HealthStatus, OffloadMode

        hs = HealthStatus(
            healthy=True,
            gpu_available=True,
            ollama_reachable=True,
            database_connected=True,
            streams_connected=2,
            streams_total=3,
            offload_mode=OffloadMode.GPU_PRIMARY,
        )
        assert hs.healthy is True
        assert hs.details == {}
