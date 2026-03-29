"""Integration tests: Dispatcher flow — dedup → upload → DB write.

Wires a real MultiDispatcher with fake DB and Store to verify
the consumer loop processes items correctly, deduplicates, and
routes audit-worthy records.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from uni_vision.common.config import (
    DatabaseConfig,
    DeduplicationConfig,
    DispatchConfig,
    StorageConfig,
)
from uni_vision.contracts.dtos import DetectionRecord, ValidationStatus
from uni_vision.postprocessing.dispatcher import MultiDispatcher


# ── Fake storage backends ─────────────────────────────────────────


class _FakeDB:
    """Lightweight in-memory stand-in for PostgresClient."""

    def __init__(self):
        self.detections: list[DetectionRecord] = []
        self.audit_logs: list[dict] = []

    async def connect(self):
        pass

    async def insert_detection(self, record: DetectionRecord):
        self.detections.append(record)

    async def insert_audit_log(self, **kwargs):
        self.audit_logs.append(kwargs)

    async def close(self):
        pass


class _FakeStore:
    """Lightweight in-memory stand-in for ObjectStoreArchiver."""

    def __init__(self):
        self.uploads: list[tuple[str, str]] = []

    async def ensure_bucket(self):
        pass

    async def upload_plate_image(self, record_id, camera_id, plate_image):
        self.uploads.append((record_id, camera_id))
        return f"s3://bucket/{record_id}.png"

    async def close(self):
        pass


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def fake_db():
    return _FakeDB()


@pytest.fixture
def fake_store():
    return _FakeStore()


@pytest.fixture
def dispatcher(fake_db, fake_store):
    d = MultiDispatcher(
        db_config=DatabaseConfig(),
        storage_config=StorageConfig(),
        dispatch_config=DispatchConfig(),
        dedup_config=DeduplicationConfig(),
    )
    # Replace real backends with fakes
    d._db = fake_db
    d._store = fake_store
    return d


@pytest.fixture
def valid_record():
    return DetectionRecord(
        id="rec-001",
        camera_id="cam-01",
        plate_number="MH12AB1234",
        raw_ocr_text="MH12AB1234",
        ocr_confidence=0.95,
        ocr_engine="ollama",
        vehicle_class="car",
        detected_at_utc="2024-01-01T00:00:00",
        validation_status=ValidationStatus.VALID.value,
    )


@pytest.fixture
def plate_image():
    return np.zeros((60, 200, 3), dtype=np.uint8)


# ── Tests ─────────────────────────────────────────────────────────


class TestDispatcherConsumerFlow:
    """Full dispatch pipeline: dedup → upload → DB write."""

    def test_single_record_written(
        self, dispatcher, fake_db, fake_store, valid_record, plate_image
    ):
        """A valid record should be uploaded and written to DB."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(dispatcher.start())
        loop.run_until_complete(dispatcher.dispatch(valid_record, plate_image))
        # Give consumer time to process
        loop.run_until_complete(asyncio.sleep(0.3))
        loop.run_until_complete(dispatcher.shutdown())

        assert len(fake_db.detections) == 1
        assert fake_db.detections[0].camera_id == "cam-01"
        assert len(fake_store.uploads) == 1

    def test_dispatch_without_plate_image(
        self, dispatcher, fake_db, fake_store, valid_record
    ):
        """Dispatch without plate_image should skip upload but still write DB."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(dispatcher.start())
        loop.run_until_complete(dispatcher.dispatch(valid_record))
        loop.run_until_complete(asyncio.sleep(0.3))
        loop.run_until_complete(dispatcher.shutdown())

        assert len(fake_db.detections) == 1
        assert len(fake_store.uploads) == 0


class TestDispatcherDeduplication:
    """Verify temporal deduplication suppresses repeated detections."""

    def test_duplicate_suppressed(
        self, dispatcher, fake_db, fake_store, valid_record, plate_image
    ):
        """Same plate+camera within the dedup window should be suppressed."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(dispatcher.start())

        # Dispatch the same record twice
        loop.run_until_complete(dispatcher.dispatch(valid_record, plate_image))
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.run_until_complete(dispatcher.dispatch(valid_record, plate_image))
        loop.run_until_complete(asyncio.sleep(0.3))
        loop.run_until_complete(dispatcher.shutdown())

        # Only one should have been written
        assert len(fake_db.detections) == 1

    def test_different_plates_not_deduplicated(
        self, dispatcher, fake_db, fake_store, valid_record, plate_image
    ):
        """Different plate numbers should not be suppressed."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(dispatcher.start())

        record_a = valid_record
        record_b = replace(
            valid_record,
            id="rec-002",
            plate_number="KA01CD5678",
        )
        loop.run_until_complete(dispatcher.dispatch(record_a, plate_image))
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.run_until_complete(dispatcher.dispatch(record_b, plate_image))
        loop.run_until_complete(asyncio.sleep(0.3))
        loop.run_until_complete(dispatcher.shutdown())

        assert len(fake_db.detections) == 2


class TestDispatcherAuditRouting:
    """Verify low-confidence / failed records go to audit log."""

    def test_low_confidence_goes_to_audit(
        self, dispatcher, fake_db, fake_store, valid_record, plate_image
    ):
        """Record with low_confidence status should hit audit log, not detections."""
        audit_record = replace(
            valid_record,
            validation_status=ValidationStatus.LOW_CONFIDENCE.value,
        )

        loop = asyncio.get_event_loop()
        loop.run_until_complete(dispatcher.start())
        loop.run_until_complete(dispatcher.dispatch(audit_record, plate_image))
        loop.run_until_complete(asyncio.sleep(0.3))
        loop.run_until_complete(dispatcher.shutdown())

        assert len(fake_db.detections) == 0
        assert len(fake_db.audit_logs) == 1
        assert fake_db.audit_logs[0]["record_id"] == "rec-001"

    def test_regex_fail_goes_to_audit(
        self, dispatcher, fake_db, valid_record, plate_image
    ):
        """Record with regex_fail status goes to audit log."""
        bad_record = replace(
            valid_record,
            validation_status=ValidationStatus.REGEX_FAIL.value,
        )

        loop = asyncio.get_event_loop()
        loop.run_until_complete(dispatcher.start())
        loop.run_until_complete(dispatcher.dispatch(bad_record, plate_image))
        loop.run_until_complete(asyncio.sleep(0.3))
        loop.run_until_complete(dispatcher.shutdown())

        assert len(fake_db.detections) == 0
        assert len(fake_db.audit_logs) == 1
