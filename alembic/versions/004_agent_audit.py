"""add agent_audit_log table for action audit trail

Revision ID: 004_agent_audit
Revises: 003_agent_knowledge
Create Date: 2025-01-01 00:00:03.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# Revision identifiers
revision: str = "004_agent_audit"
down_revision: Union[str, None] = "003_agent_knowledge"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
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
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp
            ON agent_audit_log (timestamp DESC);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_session
            ON agent_audit_log (session_id)
            WHERE session_id IS NOT NULL;
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_audit_session;")
    op.execute("DROP INDEX IF EXISTS idx_audit_timestamp;")
    op.execute("DROP TABLE IF EXISTS agent_audit_log;")
