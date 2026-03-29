"""Periodic data retention cleanup task.

When enabled, runs on a configurable interval to delete detection records
and audit logs older than the configured retention period. Operates in
batches to avoid holding long-lived DB locks.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from uni_vision.common.config import RetentionConfig
from uni_vision.storage.models import AUDIT_LOG_TABLE, TABLE_NAME

logger = logging.getLogger(__name__)

PURGE_DETECTIONS_SQL = f"""\
DELETE FROM {TABLE_NAME}
WHERE id IN (
    SELECT id FROM {TABLE_NAME}
    WHERE created_at < NOW() - INTERVAL '1 day' * $1
    ORDER BY created_at ASC
    LIMIT $2
)
"""

PURGE_AUDIT_SQL = f"""\
DELETE FROM {AUDIT_LOG_TABLE}
WHERE id IN (
    SELECT id FROM {AUDIT_LOG_TABLE}
    WHERE logged_at < NOW() - INTERVAL '1 day' * $1
    ORDER BY logged_at ASC
    LIMIT $2
)
"""


class RetentionTask:
    """Background async task that purges stale records on a schedule.

    Parameters
    ----------
    config : RetentionConfig
        Retention policy parameters.
    pg_client : object
        A ``PostgresClient`` (or compatible) with a ``_pool`` attribute.
    """

    def __init__(self, config: RetentionConfig, pg_client: object) -> None:
        self._config = config
        self._pg = pg_client
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        if not self._config.enabled:
            logger.info("retention_disabled")
            return
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "retention_started max_age_days=%d interval_h=%.1f",
            self._config.max_age_days,
            self._config.check_interval_hours,
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            logger.info("retention_stopped")

    async def _loop(self) -> None:
        interval = self._config.check_interval_hours * 3600
        while True:
            try:
                await self._purge_cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("retention_purge_error")
            await asyncio.sleep(interval)

    async def _purge_cycle(self) -> None:
        pool = getattr(self._pg, "_pool", None)
        if pool is None:
            return

        # Purge old detections
        total_detections = 0
        while True:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    PURGE_DETECTIONS_SQL,
                    self._config.max_age_days,
                    self._config.batch_size,
                )
            count = int(result.split()[-1]) if result else 0
            total_detections += count
            if count < self._config.batch_size:
                break

        # Purge old audit logs
        total_audit = 0
        while True:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    PURGE_AUDIT_SQL,
                    self._config.audit_max_age_days,
                    self._config.batch_size,
                )
            count = int(result.split()[-1]) if result else 0
            total_audit += count
            if count < self._config.batch_size:
                break

        if total_detections or total_audit:
            logger.info(
                "retention_purged detections=%d audit_logs=%d",
                total_detections,
                total_audit,
            )
