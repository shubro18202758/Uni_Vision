"""SQL table definitions for storage tables.

Provides ``CREATE TABLE`` DDL and helper constants.  Used by
``postgres.py`` to initialise the schema on first connection.

Tables defined (PRD §12):
  * ``detection_events`` — validated detection records (§12.1)
  * ``camera_sources``   — registered camera feeds   (§12.2)
  * ``ocr_audit_log``    — low-confidence / failed reads (§12.3)
"""

from __future__ import annotations

# ── Table name constants ──────────────────────────────────────────

TABLE_NAME = "detection_events"
CAMERA_SOURCES_TABLE = "camera_sources"
AUDIT_LOG_TABLE = "ocr_audit_log"

# ── detection_events (PRD §12.1) ──────────────────────────────────

CREATE_TABLE_SQL: str = f"""\
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
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
);

CREATE INDEX IF NOT EXISTS idx_detection_events_camera
    ON {TABLE_NAME} (camera_id, detected_at_utc DESC);

CREATE INDEX IF NOT EXISTS idx_detection_events_plate
    ON {TABLE_NAME} (plate_number, detected_at_utc DESC);
"""

INSERT_SQL: str = f"""\
INSERT INTO {TABLE_NAME} (
    id, camera_id, plate_number, raw_ocr_text,
    ocr_confidence, ocr_engine, vehicle_class,
    vehicle_image_path, plate_image_path,
    detected_at_utc, validation_status, location_tag
) VALUES (
    $1, $2, $3, $4,
    $5, $6, $7,
    $8, $9,
    $10, $11, $12
)
ON CONFLICT (id) DO NOTHING;
"""

# ── camera_sources (PRD §12.2) ────────────────────────────────────

CREATE_CAMERA_SOURCES_SQL: str = f"""\
CREATE TABLE IF NOT EXISTS {CAMERA_SOURCES_TABLE} (
    camera_id    TEXT PRIMARY KEY,
    source_url   TEXT        NOT NULL,
    location_tag TEXT        NOT NULL DEFAULT '',
    fps_target   SMALLINT    NOT NULL DEFAULT 3,
    enabled      BOOLEAN     NOT NULL DEFAULT TRUE,
    added_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

INSERT_CAMERA_SOURCE_SQL: str = f"""\
INSERT INTO {CAMERA_SOURCES_TABLE} (
    camera_id, source_url, location_tag, fps_target, enabled
) VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (camera_id) DO UPDATE SET
    source_url   = EXCLUDED.source_url,
    location_tag = EXCLUDED.location_tag,
    fps_target   = EXCLUDED.fps_target,
    enabled      = EXCLUDED.enabled;
"""

DELETE_CAMERA_SOURCE_SQL: str = f"""\
DELETE FROM {CAMERA_SOURCES_TABLE} WHERE camera_id = $1;
"""

SELECT_CAMERA_SOURCES_SQL: str = f"""\
SELECT camera_id, source_url, location_tag, fps_target, enabled, added_at
FROM {CAMERA_SOURCES_TABLE}
WHERE enabled = TRUE
ORDER BY added_at;
"""

# ── ocr_audit_log (PRD §12.3 / FR-11) ────────────────────────────

CREATE_AUDIT_LOG_SQL: str = f"""\
CREATE TABLE IF NOT EXISTS {AUDIT_LOG_TABLE} (
    id             TEXT PRIMARY KEY,
    camera_id      TEXT        NOT NULL DEFAULT '',
    raw_ocr_text   TEXT        NOT NULL DEFAULT '',
    ocr_confidence REAL        NOT NULL DEFAULT 0.0,
    failure_reason TEXT        NOT NULL DEFAULT '',
    frame_path     TEXT        NOT NULL DEFAULT '',
    logged_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_camera
    ON {AUDIT_LOG_TABLE} (camera_id, logged_at DESC);
"""

INSERT_AUDIT_LOG_SQL: str = f"""\
INSERT INTO {AUDIT_LOG_TABLE} (
    id, camera_id, raw_ocr_text, ocr_confidence, failure_reason, frame_path
) VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (id) DO NOTHING;
"""
