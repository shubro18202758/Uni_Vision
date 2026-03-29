"""Tests for the sliding-window temporal deduplicator."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest


class TestSlidingWindowDeduplicator:
    """Test dedup logic without asyncio event loop."""

    def _make_record(self, plate="MH12AB1234", camera="cam_01", conf=0.9):
        from uni_vision.contracts.dtos import DetectionRecord

        return DetectionRecord(
            camera_id=camera,
            plate_number=plate,
            ocr_confidence=conf,
            validation_status="valid",
        )

    def test_first_detection_not_duplicate(self, dedup_config):
        from uni_vision.postprocessing.deduplicator import SlidingWindowDeduplicator

        dedup = SlidingWindowDeduplicator(dedup_config)
        record = self._make_record()
        assert dedup.is_duplicate(record) is False

    def test_second_detection_within_window_is_duplicate(self, dedup_config):
        from uni_vision.postprocessing.deduplicator import SlidingWindowDeduplicator

        dedup = SlidingWindowDeduplicator(dedup_config)
        rec1 = self._make_record()
        rec2 = self._make_record()

        assert dedup.is_duplicate(rec1) is False
        assert dedup.is_duplicate(rec2) is True

    def test_different_plates_not_duplicates(self, dedup_config):
        from uni_vision.postprocessing.deduplicator import SlidingWindowDeduplicator

        dedup = SlidingWindowDeduplicator(dedup_config)
        rec1 = self._make_record(plate="MH12AB1234")
        rec2 = self._make_record(plate="KA01HG5678")

        assert dedup.is_duplicate(rec1) is False
        assert dedup.is_duplicate(rec2) is False

    def test_different_cameras_not_duplicates(self, dedup_config):
        from uni_vision.postprocessing.deduplicator import SlidingWindowDeduplicator

        dedup = SlidingWindowDeduplicator(dedup_config)
        rec1 = self._make_record(camera="cam_01")
        rec2 = self._make_record(camera="cam_02")

        assert dedup.is_duplicate(rec1) is False
        assert dedup.is_duplicate(rec2) is False

    def test_higher_confidence_updates_entry(self, dedup_config):
        from uni_vision.postprocessing.deduplicator import SlidingWindowDeduplicator

        dedup = SlidingWindowDeduplicator(dedup_config)
        rec_low = self._make_record(conf=0.7)
        rec_high = self._make_record(conf=0.95)

        assert dedup.is_duplicate(rec_low) is False
        assert dedup.is_duplicate(rec_high) is True

        # The entry should have the higher confidence now
        key = ("cam_01", "MH12AB1234")
        assert dedup._entries[key].confidence == 0.95

    def test_case_insensitive_plate_matching(self, dedup_config):
        from uni_vision.postprocessing.deduplicator import SlidingWindowDeduplicator

        dedup = SlidingWindowDeduplicator(dedup_config)
        rec1 = self._make_record(plate="mh12ab1234")
        rec2 = self._make_record(plate="MH12AB1234")

        assert dedup.is_duplicate(rec1) is False
        assert dedup.is_duplicate(rec2) is True  # normalised to same key

    def test_expired_window_allows_redetection(self, dedup_config):
        from uni_vision.postprocessing.deduplicator import SlidingWindowDeduplicator

        # Window is 2s in our test config
        dedup = SlidingWindowDeduplicator(dedup_config)
        rec = self._make_record()

        assert dedup.is_duplicate(rec) is False

        # Fast-forward time beyond window
        key = ("cam_01", "MH12AB1234")
        dedup._entries[key].first_seen = time.monotonic() - 3.0

        assert dedup.is_duplicate(self._make_record()) is False

    def test_count_increments(self, dedup_config):
        from uni_vision.postprocessing.deduplicator import SlidingWindowDeduplicator

        dedup = SlidingWindowDeduplicator(dedup_config)

        dedup.is_duplicate(self._make_record())
        dedup.is_duplicate(self._make_record())
        dedup.is_duplicate(self._make_record())

        key = ("cam_01", "MH12AB1234")
        assert dedup._entries[key].count == 3
