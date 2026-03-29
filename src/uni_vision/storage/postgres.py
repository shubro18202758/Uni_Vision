"""Async PostgreSQL client — spec §4 S8 storage layer.

Wraps ``asyncpg`` to provide:

  * Lazy connection-pool initialisation.
  * Automatic schema creation (detection_events, camera_sources, ocr_audit_log).
  * ``insert_detection`` with parameterised query (SQL-injection safe).
  * ``insert_audit_log`` for low-confidence / failed OCR results (FR-11).
  * Exponential-backoff retry on transient failures.

Spec reference: §4 S8, PRD §12.1–§12.3.
Failure taxonomy: F09 — DatabaseWriteError.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import asyncpg  # type: ignore[import-untyped]

from uni_vision.common.config import DatabaseConfig, DispatchConfig
from uni_vision.common.exceptions import DatabaseConnectionError, DatabaseWriteError
from uni_vision.contracts.dtos import DetectionRecord
from uni_vision.monitoring.metrics import DISPATCH_ERRORS
from uni_vision.storage.models import (
    CREATE_AUDIT_LOG_SQL,
    CREATE_CAMERA_SOURCES_SQL,
    CREATE_TABLE_SQL,
    INSERT_AUDIT_LOG_SQL,
    INSERT_SQL,
)

logger = logging.getLogger(__name__)


class PostgresClient:
    """Async PostgreSQL writer backed by an ``asyncpg`` connection pool.

    Parameters
    ----------
    db_config : DatabaseConfig
        DSN and pool sizing.
    dispatch_config : DispatchConfig
        Timeout and retry parameters.
    """

    def __init__(
        self,
        db_config: DatabaseConfig,
        dispatch_config: DispatchConfig,
    ) -> None:
        self._dsn = db_config.dsn
        self._pool_min = db_config.pool_min
        self._pool_max = db_config.pool_max
        self._timeout = dispatch_config.db_write_timeout_s
        self._max_retries = dispatch_config.max_retries
        self._retry_delay = dispatch_config.retry_base_delay_s
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Create the connection pool and ensure the schema exists."""
        if self._pool is not None:
            return

        try:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=self._pool_min,
                max_size=self._pool_max,
                command_timeout=self._timeout,
            )
        except Exception as exc:
            raise DatabaseConnectionError(
                f"Failed to create PostgreSQL pool: {exc}"
            ) from exc

        await self._ensure_schema()
        logger.info("postgres_connected pool_min=%d pool_max=%d", self._pool_min, self._pool_max)

    async def _ensure_schema(self) -> None:
        """Run ``CREATE TABLE IF NOT EXISTS`` DDL for all tables."""
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)
            await conn.execute(CREATE_CAMERA_SOURCES_SQL)
            await conn.execute(CREATE_AUDIT_LOG_SQL)
        logger.info("postgres_schema_ensured")

    async def insert_detection(self, record: DetectionRecord) -> None:
        """Insert a detection record with retry on transient failure."""
        if self._pool is None:
            await self.connect()

        assert self._pool is not None

        delay = self._retry_delay
        last_exc: Optional[Exception] = None

        for attempt in range(1 + self._max_retries):
            try:
                async with self._pool.acquire(timeout=self._timeout) as conn:
                    await conn.execute(
                        INSERT_SQL,
                        record.id,
                        record.camera_id,
                        record.plate_number,
                        record.raw_ocr_text,
                        record.ocr_confidence,
                        record.ocr_engine,
                        record.vehicle_class,
                        record.vehicle_image_path,
                        record.plate_image_path,
                        record.detected_at_utc,
                        record.validation_status,
                        record.location_tag,
                    )
                return  # success

            except (asyncpg.PostgresError, asyncio.TimeoutError, OSError) as exc:
                last_exc = exc
                logger.warning(
                    "postgres_write_retry attempt=%d/%d err=%s",
                    attempt + 1,
                    1 + self._max_retries,
                    exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2  # exponential backoff

        DISPATCH_ERRORS.labels(target="postgres").inc()
        raise DatabaseWriteError(
            f"Insert failed after {1 + self._max_retries} attempts: {last_exc}"
        ) from last_exc

    async def insert_audit_log(
        self,
        *,
        record_id: str,
        camera_id: str,
        raw_ocr_text: str,
        ocr_confidence: float,
        failure_reason: str,
        frame_path: str = "",
    ) -> None:
        """Insert a low-confidence / failed OCR result into the audit log (FR-11)."""
        if self._pool is None:
            await self.connect()

        assert self._pool is not None

        try:
            async with self._pool.acquire(timeout=self._timeout) as conn:
                await conn.execute(
                    INSERT_AUDIT_LOG_SQL,
                    record_id,
                    camera_id,
                    raw_ocr_text,
                    ocr_confidence,
                    failure_reason,
                    frame_path,
                )
        except (asyncpg.PostgresError, asyncio.TimeoutError, OSError) as exc:
            logger.warning("audit_log_insert_failed err=%s", exc)

    async def close(self) -> None:
        """Drain and close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("postgres_disconnected")
