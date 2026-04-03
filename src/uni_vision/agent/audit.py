"""Agent audit trail — persistent record of all agent actions.

Stores every tool invocation and agent response in PostgreSQL
for compliance, debugging, and learning-loop analysis.

Table: ``agent_audit_log``
    id          BIGSERIAL PRIMARY KEY
    timestamp   TIMESTAMPTZ
    session_id  TEXT
    intent      TEXT
    agent_role  TEXT
    action      TEXT (tool name or 'answer')
    arguments   JSONB
    result      TEXT (truncated observation / answer)
    success     BOOLEAN
    elapsed_ms  REAL
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_audit_log (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id  TEXT,
    intent      TEXT,
    agent_role  TEXT,
    action      TEXT NOT NULL,
    arguments   JSONB DEFAULT '{}',
    result      TEXT DEFAULT '',
    success     BOOLEAN DEFAULT TRUE,
    elapsed_ms  REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp
    ON agent_audit_log (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_session
    ON agent_audit_log (session_id)
    WHERE session_id IS NOT NULL;
"""


@dataclass
class AuditEntry:
    """Single audit log entry."""

    session_id: str | None = None
    intent: str | None = None
    agent_role: str | None = None
    action: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    success: bool = True
    elapsed_ms: float = 0.0


class AuditTrail:
    """Write-behind audit logger for agent actions.

    Buffers entries and flushes them in batches to PostgreSQL.
    """

    def __init__(self, *, buffer_limit: int = 50) -> None:
        self._buffer: list[AuditEntry] = []
        self._buffer_limit = buffer_limit

    def record(self, entry: AuditEntry) -> None:
        """Buffer an audit entry for later flush."""
        self._buffer.append(entry)
        if len(self._buffer) >= self._buffer_limit:
            logger.debug("audit_buffer_full count=%d", len(self._buffer))

    async def flush(self, pg_client: Any) -> int:
        """Write all buffered entries to PostgreSQL.

        Returns the number of entries flushed.
        """
        if not self._buffer or pg_client is None:
            return 0

        entries = self._buffer[:]
        self._buffer.clear()

        try:
            async with pg_client.acquire() as conn:
                for entry in entries:
                    await conn.execute(
                        """
                        INSERT INTO agent_audit_log
                            (session_id, intent, agent_role, action,
                             arguments, result, success, elapsed_ms)
                        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
                        """,
                        entry.session_id,
                        entry.intent,
                        entry.agent_role,
                        entry.action,
                        json.dumps(entry.arguments, default=str),
                        entry.result[:5000],  # truncate large results
                        entry.success,
                        entry.elapsed_ms,
                    )
            logger.info("audit_flushed count=%d", len(entries))
            return len(entries)
        except Exception:
            # Put entries back on failure
            self._buffer = entries + self._buffer
            logger.warning("audit_flush_failed count=%d", len(entries), exc_info=True)
            return 0

    async def ensure_table(self, pg_client: Any) -> None:
        """Create the audit log table if it doesn't exist."""
        if pg_client is None:
            return
        try:
            async with pg_client.acquire() as conn:
                await conn.execute(CREATE_TABLE_SQL)
            logger.info("audit_table_ensured")
        except Exception:
            logger.warning("audit_table_create_failed", exc_info=True)

    @property
    def pending_count(self) -> int:
        return len(self._buffer)

    def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent buffered entries (not yet flushed)."""
        recent = self._buffer[-limit:]
        return [
            {
                "action": e.action,
                "intent": e.intent,
                "agent_role": e.agent_role,
                "success": e.success,
                "elapsed_ms": e.elapsed_ms,
            }
            for e in reversed(recent)
        ]
