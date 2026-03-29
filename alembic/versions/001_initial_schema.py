"""initial schema — detection_events, camera_sources, ocr_audit_log

Revision ID: 001_initial
Revises: None
Create Date: 2025-01-01 00:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── detection_events (PRD §12.1) ──
    op.execute("""\
        CREATE TABLE IF NOT EXISTS detection_events (
            id                TEXT PRIMARY KEY,
            camera_id         TEXT        NOT NULL,
            plate_number      TEXT        NOT NULL,
            raw_ocr_text      TEXT        NOT NULL DEFAULT '',
            ocr_confidence    REAL        NOT NULL DEFAULT 0.0,
            ocr_engine        TEXT        NOT NULL DEFAULT '',
            vehicle_class     TEXT        NOT NULL DEFAULT '',
            vehicle_image_path TEXT       NOT NULL DEFAULT '',
            plate_image_path  TEXT        NOT NULL DEFAULT '',
            detected_at_utc   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            validation_status TEXT        NOT NULL DEFAULT '',
            location_tag      TEXT        NOT NULL DEFAULT '',
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""\
        CREATE INDEX IF NOT EXISTS idx_detection_events_camera
            ON detection_events (camera_id, detected_at_utc DESC)
    """)
    op.execute("""\
        CREATE INDEX IF NOT EXISTS idx_detection_events_plate
            ON detection_events (plate_number, detected_at_utc DESC)
    """)

    # ── camera_sources (PRD §12.2) ──
    op.execute("""\
        CREATE TABLE IF NOT EXISTS camera_sources (
            camera_id    TEXT PRIMARY KEY,
            source_url   TEXT        NOT NULL,
            location_tag TEXT        NOT NULL DEFAULT '',
            fps_target   SMALLINT    NOT NULL DEFAULT 3,
            enabled      BOOLEAN     NOT NULL DEFAULT TRUE,
            added_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # ── ocr_audit_log (PRD §12.3 / FR-11) ──
    op.execute("""\
        CREATE TABLE IF NOT EXISTS ocr_audit_log (
            id             TEXT PRIMARY KEY,
            camera_id      TEXT        NOT NULL DEFAULT '',
            raw_ocr_text   TEXT        NOT NULL DEFAULT '',
            ocr_confidence REAL        NOT NULL DEFAULT 0.0,
            failure_reason TEXT        NOT NULL DEFAULT '',
            frame_path     TEXT        NOT NULL DEFAULT '',
            logged_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""\
        CREATE INDEX IF NOT EXISTS idx_audit_log_camera
            ON ocr_audit_log (camera_id, logged_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ocr_audit_log CASCADE")
    op.execute("DROP TABLE IF EXISTS camera_sources CASCADE")
    op.execute("DROP TABLE IF EXISTS detection_events CASCADE")
