"""Asynchronous multi-target dispatcher — spec §4 S8 dispatch layer.

Implements the ``Dispatcher`` protocol.  When ``dispatch()`` is called
by the pipeline, the record is placed onto a bounded ``asyncio.Queue``
and control returns **immediately** — the caller (inference loop) is
never blocked by database or object-store I/O.

A background consumer task drains the queue and performs:

  1. **Temporal deduplication** — ``SlidingWindowDeduplicator`` check.
     Duplicate records are silently suppressed.
  2. **Image archival** — plate crop uploaded to S3/MinIO via
     ``ObjectStoreArchiver``.  The returned object key is stamped onto
     the record.
  3. **Database commit** — structured metadata inserted into PostgreSQL
     via ``PostgresClient``.

All three targets are resilient: each has its own retry budget, and a
failure in one does **not** block the others.  The 2-second dispatch
latency SLA (spec §4 S8) is enforced by the bounded queue and async
task design — no synchronous blocking ever occurs on the inference
path.

Spec references: §4 S8 dispatch, §9.2 Dispatcher protocol, §13 F09/F10.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from uni_vision.common.config import (
    DatabaseConfig,
    DeduplicationConfig,
    DispatchConfig,
    StorageConfig,
)
from uni_vision.common.exceptions import DatabaseWriteError, ObjectStoreError
from uni_vision.contracts.dtos import DetectionRecord, ValidationStatus
from uni_vision.monitoring.metrics import DISPATCH_ERRORS, DISPATCH_SUCCESS, STAGE_LATENCY
from uni_vision.postprocessing.deduplicator import SlidingWindowDeduplicator
from uni_vision.storage.object_store import ObjectStoreArchiver
from uni_vision.storage.postgres import PostgresClient

logger = logging.getLogger(__name__)

# Module-level frozenset — avoids rebuilding per dispatch item.
_AUDIT_STATUSES: frozenset[str] = frozenset({
    ValidationStatus.LOW_CONFIDENCE.value,
    ValidationStatus.REGEX_FAIL.value,
    ValidationStatus.LLM_ERROR.value,
    ValidationStatus.PARSE_FAIL.value,
    ValidationStatus.UNREADABLE.value,
})


class MultiDispatcher:
    """Decoupled, async multi-target record dispatcher.

    Implements the ``Dispatcher`` protocol (``async dispatch(record)``).
    Internally, the actual I/O happens on a background consumer task
    so the inference loop is never stalled.

    Parameters
    ----------
    db_config : DatabaseConfig
    storage_config : StorageConfig
    dispatch_config : DispatchConfig
    dedup_config : DeduplicationConfig
    """

    def __init__(
        self,
        db_config: DatabaseConfig,
        storage_config: StorageConfig,
        dispatch_config: DispatchConfig,
        dedup_config: DeduplicationConfig,
    ) -> None:
        self._dispatch_cfg = dispatch_config

        # Sub-components
        self._dedup = SlidingWindowDeduplicator(dedup_config)
        self._db = PostgresClient(db_config, dispatch_config)
        self._store = ObjectStoreArchiver(storage_config, dispatch_config)

        # Optional Databricks Delta Lake dual-write (set externally)
        self._delta_store = None

        # Bounded async queue — decouples inference from I/O
        self._queue: asyncio.Queue[_DispatchItem] = asyncio.Queue(
            maxsize=dispatch_config.queue_maxsize,
        )
        self._consumer_task: Optional[asyncio.Task[None]] = None

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise storage backends and start the background consumer."""
        await self._db.connect()
        await self._store.ensure_bucket()
        self._dedup.start_purge_task()
        self._consumer_task = asyncio.create_task(self._consumer_loop())
        logger.info("dispatcher_started queue_max=%d", self._dispatch_cfg.queue_maxsize)

    async def shutdown(self) -> None:
        """Drain the queue (with timeout) and release resources."""
        # Signal consumer to stop after draining
        if self._consumer_task is not None:
            # Push a sentinel to unblock the consumer
            try:
                self._queue.put_nowait(_SENTINEL)
            except asyncio.QueueFull:
                pass

            try:
                await asyncio.wait_for(self._consumer_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._consumer_task.cancel()
                logger.warning("dispatcher_drain_timeout")

        self._dedup.stop()
        await self._db.close()
        await self._store.close()
        logger.info("dispatcher_stopped")

    # ── Dispatcher protocol ───────────────────────────────────────

    async def dispatch(
        self,
        record: DetectionRecord,
        plate_image: Optional[NDArray[np.uint8]] = None,
    ) -> None:
        """Enqueue a record for async persistence.

        This method returns **immediately** (or near-immediately).
        If the queue is full, the oldest pending item is logged and
        dropped to prevent inference stalls.
        """
        item = _DispatchItem(record=record, plate_image=plate_image)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            # Drop oldest to make room (ring-buffer semantics)
            try:
                dropped = self._queue.get_nowait()
                self._queue.task_done()
                logger.warning(
                    "dispatch_queue_full dropped_id=%s", dropped.record.id
                )
                DISPATCH_ERRORS.labels(target="queue_overflow").inc()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(item)

    # ── Background consumer ───────────────────────────────────────

    async def _consumer_loop(self) -> None:
        """Drain the dispatch queue and persist each item."""
        import time

        while True:
            item: _DispatchItem = await self._queue.get()

            # Check for shutdown sentinel
            if item is _SENTINEL:
                self._queue.task_done()
                break

            t0 = time.perf_counter()
            try:
                await self._process_item(item)
            except Exception as exc:
                logger.error(
                    "dispatch_item_error id=%s err=%s",
                    item.record.id,
                    exc,
                    exc_info=True,
                )
            finally:
                STAGE_LATENCY.labels(stage="S8_dispatch_io").observe(
                    time.perf_counter() - t0
                )
                self._queue.task_done()

    async def _process_item(self, item: _DispatchItem) -> None:
        """Dedup → image upload → DB write for a single record."""
        record = item.record

        # ── Step 1: Deduplication ─────────────────────────────────
        if self._dedup.is_duplicate(record):
            return

        # ── Step 1b: Audit log routing (FR-11) ───────────────────
        # Low-confidence / failed reads go to the audit log, not
        # the primary detection_events table.
        if record.validation_status in _AUDIT_STATUSES:
            await self._db.insert_audit_log(
                record_id=record.id,
                camera_id=record.camera_id,
                raw_ocr_text=record.raw_ocr_text,
                ocr_confidence=record.ocr_confidence,
                failure_reason=record.validation_status,
            )
            # Delta Lake audit dual-write
            if self._delta_store is not None:
                try:
                    self._delta_store.append_audit(
                        record_id=record.id,
                        camera_id=record.camera_id,
                        raw_ocr_text=record.raw_ocr_text,
                        ocr_confidence=record.ocr_confidence,
                        failure_reason=record.validation_status,
                    )
                except Exception as exc:
                    logger.warning("delta_audit_failed id=%s err=%s", record.id, exc)
            return

        # ── Step 2: Image archival (if plate crop provided) ───────
        if item.plate_image is not None:
            try:
                image_key = await self._store.upload_plate_image(
                    record_id=record.id,
                    camera_id=record.camera_id,
                    plate_image=item.plate_image,
                )
                # Stamp the object key onto the record
                record = replace(record, plate_image_path=image_key)
            except ObjectStoreError as exc:
                logger.error("image_upload_failed id=%s err=%s", record.id, exc)
                # Continue with DB write even if image upload fails
                record = replace(record, plate_image_path="upload_pending")

        # ── Step 3: PostgreSQL insert ─────────────────────────────
        try:
            await self._db.insert_detection(record)
            DISPATCH_SUCCESS.inc()
        except DatabaseWriteError as exc:
            logger.error("db_write_failed id=%s err=%s", record.id, exc)
            # Record is lost; metrics track the failure
        # ── Step 4: Delta Lake dual-write (Databricks add-on) ─────
        if self._delta_store is not None:
            try:
                self._delta_store.append_detection(record)
            except Exception as exc:
                logger.warning("delta_write_failed id=%s err=%s", record.id, exc)

# ── Internal data types ───────────────────────────────────────────


class _DispatchItem:
    """Carrier for a record + optional plate image through the queue."""

    __slots__ = ("record", "plate_image")

    def __init__(
        self,
        record: DetectionRecord,
        plate_image: Optional[NDArray[np.uint8]],
    ) -> None:
        self.record = record
        self.plate_image = plate_image


# Shutdown sentinel (identity-compared, never dispatched)
_SENTINEL = _DispatchItem(
    record=DetectionRecord(),  # empty placeholder
    plate_image=None,
)
