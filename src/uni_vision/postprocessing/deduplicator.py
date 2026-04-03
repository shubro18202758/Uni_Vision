"""Sliding-window temporal deduplication — spec §4 S8.

Tracks recently seen plate numbers per camera and suppresses redundant
detections originating from the same physical event across consecutive
frames.  Only the **highest-confidence** detection within each window
is forwarded to dispatch.

Algorithm
---------
For each incoming ``(camera_id, plate_text)`` pair, check if an entry
already exists in the window map with ``timestamp`` < ``window_seconds``
ago.  If so:

  * If the new detection has **higher confidence**, *replace* the
    stored entry (so the eventual dispatch carries the best reading).
  * In either case, return ``is_duplicate = True`` so the caller
    suppresses dispatch.

A periodic purge task evicts stale entries to bound memory growth.

Thread safety: all mutations happen on the asyncio event loop — no
locks required.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from uni_vision.monitoring.metrics import DETECTIONS_DEDUPLICATED

if TYPE_CHECKING:
    from uni_vision.common.config import DeduplicationConfig
    from uni_vision.contracts.dtos import DetectionRecord

logger = logging.getLogger(__name__)


class _WindowEntry:
    """Mutable entry tracking the best detection in the current window."""

    __slots__ = ("confidence", "count", "first_seen", "last_seen", "plate_text")

    def __init__(
        self,
        plate_text: str,
        confidence: float,
        first_seen: float,
        last_seen: float,
        count: int = 1,
    ) -> None:
        self.plate_text = plate_text
        self.confidence = confidence
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.count = count


# Key type: (camera_id, normalised_plate_text)
_Key = tuple[str, str]


class SlidingWindowDeduplicator:
    """Suppress repeated detections of the same plate within a time window.

    Parameters
    ----------
    config : DeduplicationConfig
        ``window_seconds`` and ``purge_interval_seconds``.
    """

    def __init__(self, config: DeduplicationConfig) -> None:
        self._window_s = config.window_seconds
        self._purge_interval = config.purge_interval_seconds
        self._entries: dict[_Key, _WindowEntry] = {}
        self._purge_task: asyncio.Task[None] | None = None

    # ── Public API ────────────────────────────────────────────────

    def is_duplicate(self, record: DetectionRecord) -> bool:
        """Return ``True`` if this detection should be suppressed.

        Side-effects:
          * Updates the window entry if a higher-confidence detection
            arrives for the same plate+camera.
          * Increments the ``DETECTIONS_DEDUPLICATED`` counter on
            suppression.
        """
        now = time.monotonic()
        key: _Key = (record.camera_id, record.plate_number.strip().upper())

        entry = self._entries.get(key)

        if entry is not None and (now - entry.first_seen) < self._window_s:
            # Within the active window — this is a duplicate
            entry.last_seen = now
            entry.count += 1

            # Keep the highest-confidence reading
            if record.ocr_confidence > entry.confidence:
                entry.confidence = record.ocr_confidence
                entry.plate_text = record.plate_number

            DETECTIONS_DEDUPLICATED.inc()
            logger.debug(
                "dedup_suppressed plate=%s camera=%s count=%d",
                key[1],
                key[0],
                entry.count,
            )
            return True

        # New plate or window expired — create/replace entry
        self._entries[key] = _WindowEntry(
            plate_text=record.plate_number,
            confidence=record.ocr_confidence,
            first_seen=now,
            last_seen=now,
        )
        return False

    # ── Background purge ──────────────────────────────────────────

    def start_purge_task(self) -> None:
        """Launch the background purge coroutine (call once at startup)."""
        if self._purge_task is None or self._purge_task.done():
            self._purge_task = asyncio.create_task(self._purge_loop())

    async def _purge_loop(self) -> None:
        """Periodically evict expired entries to bound memory."""
        while True:
            await asyncio.sleep(self._purge_interval)
            self._purge_expired()

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, v in self._entries.items() if (now - v.first_seen) >= self._window_s]
        for k in expired:
            del self._entries[k]

        if expired:
            logger.debug("dedup_purged count=%d remaining=%d", len(expired), len(self._entries))

    def stop(self) -> None:
        """Cancel the background purge task."""
        if self._purge_task is not None and not self._purge_task.done():
            self._purge_task.cancel()
