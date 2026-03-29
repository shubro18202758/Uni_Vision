"""add schema_migrations tracking and performance indexes

Revision ID: 002_perf_indexes
Revises: 001_initial
Create Date: 2025-01-01 00:00:01.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "002_perf_indexes"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # GIN trigram index for fuzzy plate search (requires pg_trgm extension)
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("""\
        CREATE INDEX IF NOT EXISTS idx_detection_events_plate_trgm
            ON detection_events USING gin (plate_number gin_trgm_ops)
    """)

    # Partial index for recent unvalidated detections (query hot path)
    op.execute("""\
        CREATE INDEX IF NOT EXISTS idx_detection_events_pending
            ON detection_events (detected_at_utc DESC)
            WHERE validation_status = ''
    """)

    # Covering index for audit log failure analysis
    op.execute("""\
        CREATE INDEX IF NOT EXISTS idx_audit_log_failure
            ON ocr_audit_log (failure_reason, logged_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_audit_log_failure")
    op.execute("DROP INDEX IF EXISTS idx_detection_events_pending")
    op.execute("DROP INDEX IF EXISTS idx_detection_events_plate_trgm")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
