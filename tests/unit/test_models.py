"""Tests for SQL model definitions."""

from __future__ import annotations


class TestTableNames:
    """Verify table name constants."""

    def test_detection_events_table(self):
        from uni_vision.storage.models import TABLE_NAME

        assert TABLE_NAME == "detection_events"

    def test_camera_sources_table(self):
        from uni_vision.storage.models import CAMERA_SOURCES_TABLE

        assert CAMERA_SOURCES_TABLE == "camera_sources"

    def test_audit_log_table(self):
        from uni_vision.storage.models import AUDIT_LOG_TABLE

        assert AUDIT_LOG_TABLE == "ocr_audit_log"


class TestDetectionEventsSQL:
    """Verify detection_events DDL and DML."""

    def test_create_sql_contains_table_name(self):
        from uni_vision.storage.models import CREATE_TABLE_SQL, TABLE_NAME

        assert f"CREATE TABLE IF NOT EXISTS {TABLE_NAME}" in CREATE_TABLE_SQL

    def test_create_sql_has_all_columns(self):
        from uni_vision.storage.models import CREATE_TABLE_SQL

        required_columns = [
            "id", "camera_id", "plate_number", "raw_ocr_text",
            "ocr_confidence", "ocr_engine", "vehicle_class",
            "vehicle_image_path", "plate_image_path",
            "detected_at_utc", "validation_status", "location_tag",
            "created_at",
        ]
        for col in required_columns:
            assert col in CREATE_TABLE_SQL, f"Missing column: {col}"

    def test_create_sql_has_indexes(self):
        from uni_vision.storage.models import CREATE_TABLE_SQL

        assert "idx_detection_events_camera" in CREATE_TABLE_SQL
        assert "idx_detection_events_plate" in CREATE_TABLE_SQL

    def test_insert_sql_has_upsert(self):
        from uni_vision.storage.models import INSERT_SQL

        assert "ON CONFLICT (id) DO NOTHING" in INSERT_SQL

    def test_insert_sql_has_12_params(self):
        from uni_vision.storage.models import INSERT_SQL

        assert "$12" in INSERT_SQL


class TestCameraSourcesSQL:
    """Verify camera_sources DDL and DML (PRD §12.2)."""

    def test_create_camera_sources(self):
        from uni_vision.storage.models import CREATE_CAMERA_SOURCES_SQL

        assert "CREATE TABLE IF NOT EXISTS camera_sources" in CREATE_CAMERA_SOURCES_SQL
        assert "camera_id" in CREATE_CAMERA_SOURCES_SQL
        assert "source_url" in CREATE_CAMERA_SOURCES_SQL
        assert "location_tag" in CREATE_CAMERA_SOURCES_SQL
        assert "fps_target" in CREATE_CAMERA_SOURCES_SQL
        assert "enabled" in CREATE_CAMERA_SOURCES_SQL

    def test_insert_camera_source_upsert(self):
        from uni_vision.storage.models import INSERT_CAMERA_SOURCE_SQL

        assert "ON CONFLICT (camera_id) DO UPDATE" in INSERT_CAMERA_SOURCE_SQL

    def test_delete_camera_source(self):
        from uni_vision.storage.models import DELETE_CAMERA_SOURCE_SQL

        assert "DELETE FROM camera_sources" in DELETE_CAMERA_SOURCE_SQL
        assert "$1" in DELETE_CAMERA_SOURCE_SQL

    def test_select_camera_sources(self):
        from uni_vision.storage.models import SELECT_CAMERA_SOURCES_SQL

        assert "WHERE enabled = TRUE" in SELECT_CAMERA_SOURCES_SQL
        assert "ORDER BY added_at" in SELECT_CAMERA_SOURCES_SQL


class TestAuditLogSQL:
    """Verify ocr_audit_log DDL and DML (PRD §12.3 / FR-11)."""

    def test_create_audit_log(self):
        from uni_vision.storage.models import CREATE_AUDIT_LOG_SQL

        assert "CREATE TABLE IF NOT EXISTS ocr_audit_log" in CREATE_AUDIT_LOG_SQL
        assert "failure_reason" in CREATE_AUDIT_LOG_SQL
        assert "raw_ocr_text" in CREATE_AUDIT_LOG_SQL
        assert "ocr_confidence" in CREATE_AUDIT_LOG_SQL
        assert "frame_path" in CREATE_AUDIT_LOG_SQL

    def test_audit_log_has_index(self):
        from uni_vision.storage.models import CREATE_AUDIT_LOG_SQL

        assert "idx_audit_log_camera" in CREATE_AUDIT_LOG_SQL

    def test_insert_audit_log(self):
        from uni_vision.storage.models import INSERT_AUDIT_LOG_SQL

        assert "INSERT INTO ocr_audit_log" in INSERT_AUDIT_LOG_SQL
        assert "ON CONFLICT (id) DO NOTHING" in INSERT_AUDIT_LOG_SQL
        assert "$6" in INSERT_AUDIT_LOG_SQL
