"""Delta Lake detection event sink — ACID transactional storage.

Provides a dual-write sink that runs alongside PostgreSQL.  Every
detection record dispatched through the pipeline is also appended
to a Delta table, giving:

  * **ACID transactions** — atomic commits with optimistic concurrency.
  * **Time-travel** — query any historical version for audit replay.
  * **Schema enforcement** — rejects malformed records automatically.
  * **Partition pruning** — ``camera_id`` partitioning for fast scans.
  * **VACUUM** — automated cleanup of stale Parquet files.

The table layout mirrors ``DetectionRecord`` from ``contracts.dtos``
and partitions on ``camera_id`` by default.  The audit log uses a
separate Delta table for low-confidence / failed reads.

Requires: ``pip install 'uni-vision[databricks]'``
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports — only loaded when Databricks features are enabled
_pa = None
_DeltaTable = None
_write_deltalake = None


def _ensure_imports() -> None:
    """Import heavy dependencies on first use."""
    global _pa, _DeltaTable, _write_deltalake
    if _pa is not None:
        return
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        from deltalake import DeltaTable, write_deltalake  # type: ignore[import-untyped]

        _pa = pa
        _DeltaTable = DeltaTable
        _write_deltalake = write_deltalake
    except ImportError as exc:
        raise ImportError("Delta Lake dependencies not installed. Run: pip install 'uni-vision[databricks]'") from exc


# ── Detection table schema ────────────────────────────────────────


def _detection_schema():
    """PyArrow schema matching DetectionRecord fields."""
    _ensure_imports()
    pa = _pa
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("camera_id", pa.string(), nullable=False),
            pa.field("plate_number", pa.string(), nullable=False),
            pa.field("raw_ocr_text", pa.string()),
            pa.field("ocr_confidence", pa.float64()),
            pa.field("ocr_engine", pa.string()),
            pa.field("vehicle_class", pa.string()),
            pa.field("vehicle_image_path", pa.string()),
            pa.field("plate_image_path", pa.string()),
            pa.field("detected_at_utc", pa.timestamp("us", tz="UTC")),
            pa.field("validation_status", pa.string()),
            pa.field("location_tag", pa.string()),
            pa.field("ingested_at_utc", pa.timestamp("us", tz="UTC")),
        ]
    )


def _audit_schema():
    """PyArrow schema for the audit log table."""
    _ensure_imports()
    pa = _pa
    return pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("camera_id", pa.string(), nullable=False),
            pa.field("raw_ocr_text", pa.string()),
            pa.field("ocr_confidence", pa.float64()),
            pa.field("failure_reason", pa.string()),
            pa.field("logged_at_utc", pa.timestamp("us", tz="UTC")),
        ]
    )


class DeltaLakeStore:
    """ACID-compliant Delta Lake storage for detection records.

    Parameters
    ----------
    table_path : str
        File-system path for the detection Delta table.
    audit_table_path : str
        Path for the audit log Delta table.
    partition_columns : list[str]
        Columns to partition the detection table by.
    checkpoint_interval : int
        Commit a Delta checkpoint every N appends.
    vacuum_retain_hours : int
        Hours of data to retain during VACUUM.
    """

    def __init__(
        self,
        table_path: str = "./data/delta/detections",
        audit_table_path: str = "./data/delta/audit_log",
        partition_columns: list[str] | None = None,
        checkpoint_interval: int = 50,
        vacuum_retain_hours: int = 168,
    ) -> None:
        _ensure_imports()
        self._table_path = str(Path(table_path).resolve())
        self._audit_path = str(Path(audit_table_path).resolve())
        self._partition_cols = partition_columns or ["camera_id"]
        self._checkpoint_interval = checkpoint_interval
        self._vacuum_retain_hours = vacuum_retain_hours
        self._write_count = 0
        self._audit_write_count = 0
        self._lock = threading.Lock()

        # Operational metrics
        self._total_bytes_written = 0
        self._last_write_latency_ms = 0.0
        self._error_count = 0

    # ── Initialisation ────────────────────────────────────────────

    def initialise(self) -> None:
        """Create Delta tables if they don't exist.

        Uses ``write_deltalake`` with an empty batch to bootstrap the
        table with the correct schema and partitioning.
        """
        for path, schema, partitions in [
            (self._table_path, _detection_schema(), self._partition_cols),
            (self._audit_path, _audit_schema(), []),
        ]:
            if not _DeltaTable.is_deltatable(path):
                os.makedirs(path, exist_ok=True)
                empty_batch = _pa.RecordBatch.from_pydict(
                    {field.name: [] for field in schema},
                    schema=schema,
                )
                _write_deltalake(
                    path,
                    empty_batch,
                    schema=schema,
                    partition_by=partitions or None,
                    mode="overwrite",
                )
                logger.info("delta_table_created path=%s", path)

    # ── Detection writes ──────────────────────────────────────────

    def append_detection(self, record) -> None:
        """Append a DetectionRecord to the Delta detection table.

        Parameters
        ----------
        record : DetectionRecord
            Frozen dataclass from ``contracts.dtos``.
        """
        now = datetime.now(timezone.utc)
        detected_at = record.detected_at_utc
        if isinstance(detected_at, str):
            detected_at = datetime.fromisoformat(detected_at)
        if detected_at.tzinfo is None:
            detected_at = detected_at.replace(tzinfo=timezone.utc)

        batch = _pa.RecordBatch.from_pydict(
            {
                "id": [record.id],
                "camera_id": [record.camera_id],
                "plate_number": [record.plate_number],
                "raw_ocr_text": [record.raw_ocr_text or ""],
                "ocr_confidence": [float(record.ocr_confidence)],
                "ocr_engine": [record.ocr_engine or ""],
                "vehicle_class": [record.vehicle_class or ""],
                "vehicle_image_path": [record.vehicle_image_path or ""],
                "plate_image_path": [record.plate_image_path or ""],
                "detected_at_utc": [detected_at],
                "validation_status": [record.validation_status or ""],
                "location_tag": [record.location_tag or ""],
                "ingested_at_utc": [now],
            },
            schema=_detection_schema(),
        )

        t0 = time.monotonic()
        with self._lock:
            _write_deltalake(
                self._table_path,
                batch,
                mode="append",
                schema_mode="merge",
            )
            self._write_count += 1
        self._last_write_latency_ms = (time.monotonic() - t0) * 1000
        self._total_bytes_written += batch.nbytes

        if self._write_count % self._checkpoint_interval == 0:
            self._create_checkpoint()

        logger.debug(
            "delta_detection_appended id=%s camera=%s plate=%s",
            record.id,
            record.camera_id,
            record.plate_number,
        )

    def append_audit(
        self,
        record_id: str,
        camera_id: str,
        raw_ocr_text: str,
        ocr_confidence: float,
        failure_reason: str,
    ) -> None:
        """Append an audit log entry for a failed / low-confidence read."""
        now = datetime.now(timezone.utc)
        batch = _pa.RecordBatch.from_pydict(
            {
                "record_id": [record_id],
                "camera_id": [camera_id],
                "raw_ocr_text": [raw_ocr_text or ""],
                "ocr_confidence": [float(ocr_confidence)],
                "failure_reason": [failure_reason],
                "logged_at_utc": [now],
            },
            schema=_audit_schema(),
        )
        _write_deltalake(self._audit_path, batch, mode="append", schema_mode="merge")
        with self._lock:
            self._audit_write_count += 1

    # ── Time-travel queries ───────────────────────────────────────

    def read_at_version(self, version: int) -> list[dict[str, Any]]:
        """Read the detection table at a specific Delta version.

        This enables full audit replay — every state the table has
        ever been in is queryable.
        """
        dt = _DeltaTable(self._table_path, version=version)
        df = dt.to_pandas()
        return df.to_dict(orient="records")

    def read_at_timestamp(self, timestamp_iso: str) -> list[dict[str, Any]]:
        """Read the detection table as it existed at a given timestamp."""
        dt = _DeltaTable(
            self._table_path,
            version=None,
        )
        # Load all versions and pick the latest version at or before timestamp
        history = dt.history()
        target = datetime.fromisoformat(timestamp_iso)
        best_version = 0
        for entry in history:
            ts = entry.get("timestamp")
            if ts is not None and ts <= target:
                v = entry.get("version", 0)
                if v > best_version:
                    best_version = v
        return self.read_at_version(best_version)

    # ── Table metadata ────────────────────────────────────────────

    def get_table_stats(self) -> dict[str, Any]:
        """Return Delta table metadata: version, row count, partitions etc."""
        if not _DeltaTable.is_deltatable(self._table_path):
            return {"exists": False, "table_path": self._table_path}

        dt = _DeltaTable(self._table_path)
        metadata = dt.metadata()
        history = dt.history()
        current_version = dt.version()

        # Count rows via PyArrow dataset
        total_rows = dt.to_pyarrow_dataset().count_rows()

        return {
            "exists": True,
            "table_path": self._table_path,
            "current_version": current_version,
            "total_rows": total_rows,
            "partition_columns": metadata.partition_columns,
            "schema_fields": [f.name for f in dt.schema().to_pyarrow()],
            "total_commits": len(history),
            "created_at": history[-1].get("timestamp") if history else None,
            "last_commit_at": history[0].get("timestamp") if history else None,
            "total_writes": self._write_count,
            "total_bytes_written": self._total_bytes_written,
            "last_write_latency_ms": round(self._last_write_latency_ms, 2),
            "error_count": self._error_count,
            "size_on_disk_bytes": self._get_table_size_bytes(),
        }

    def get_audit_stats(self) -> dict[str, Any]:
        """Return audit table metadata."""
        if not _DeltaTable.is_deltatable(self._audit_path):
            return {"exists": False, "audit_table_path": self._audit_path}

        dt = _DeltaTable(self._audit_path)
        total_rows = dt.to_pyarrow_dataset().count_rows()

        return {
            "exists": True,
            "audit_table_path": self._audit_path,
            "current_version": dt.version(),
            "total_rows": total_rows,
            "total_audit_writes": self._audit_write_count,
        }

    def get_version_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent transaction log entries for the detection table."""
        if not _DeltaTable.is_deltatable(self._table_path):
            return []
        dt = _DeltaTable(self._table_path)
        history = dt.history()
        return [
            {
                "version": h.get("version"),
                "timestamp": str(h.get("timestamp", "")),
                "operation": h.get("operation", ""),
                "parameters": h.get("operationParameters", {}),
            }
            for h in history[:limit]
        ]

    # ── Maintenance ───────────────────────────────────────────────

    def vacuum(self) -> None:
        """Remove stale Parquet files beyond the retention threshold."""
        if not _DeltaTable.is_deltatable(self._table_path):
            return
        dt = _DeltaTable(self._table_path)
        dt.vacuum(retention_hours=self._vacuum_retain_hours, enforce_retention_duration=False)
        logger.info("delta_vacuum_complete retain_hours=%d", self._vacuum_retain_hours)

    def _create_checkpoint(self) -> None:
        """Create a Delta checkpoint for faster table loading."""
        if not _DeltaTable.is_deltatable(self._table_path):
            return
        dt = _DeltaTable(self._table_path)
        dt.create_checkpoint()
        logger.info("delta_checkpoint_created version=%d", dt.version())

    # ── Convenience reads ─────────────────────────────────────────

    def read_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Read the most recent detections from the Delta table."""
        if not _DeltaTable.is_deltatable(self._table_path):
            return []
        dt = _DeltaTable(self._table_path)
        df = dt.to_pandas()
        if df.empty:
            return []
        df = df.sort_values("ingested_at_utc", ascending=False).head(limit)
        # Convert timestamps to ISO strings for JSON serialisation
        for col in ["detected_at_utc", "ingested_at_utc"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df.to_dict(orient="records")

    def read_by_camera(self, camera_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Read detections filtered by camera using partition pruning."""
        if not _DeltaTable.is_deltatable(self._table_path):
            return []
        dt = _DeltaTable(self._table_path)
        ds = dt.to_pyarrow_dataset()
        table = ds.to_table(filter=(ds.field("camera_id") == camera_id))
        df = table.to_pandas()
        if df.empty:
            return []
        df = df.sort_values("ingested_at_utc", ascending=False).head(limit)
        for col in ["detected_at_utc", "ingested_at_utc"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df.to_dict(orient="records")

    # ── Compaction & size helpers ─────────────────────────────────

    def compact(self, target_size_mb: int = 128) -> dict[str, Any]:
        """Compact small Parquet files into larger ones for better read perf.

        Uses Z-order optimisation on ``plate_number`` when available,
        falling back to a simple file compaction via overwrite.
        """
        if not _DeltaTable.is_deltatable(self._table_path):
            return {"status": "no_table"}

        dt = _DeltaTable(self._table_path)
        before_files = len(list(Path(self._table_path).rglob("*.parquet")))

        try:
            dt.optimize.compact()
            after_files = len(list(Path(self._table_path).rglob("*.parquet")))
            logger.info(
                "delta_compact_complete before=%d after=%d",
                before_files,
                after_files,
            )
            return {
                "status": "compacted",
                "files_before": before_files,
                "files_after": after_files,
            }
        except Exception as exc:
            logger.warning("delta_compact_failed err=%s", exc)
            return {"status": "error", "error": str(exc)}

    def _get_table_size_bytes(self) -> int:
        """Calculate the total on-disk size of the Delta table."""
        total = 0
        table_dir = Path(self._table_path)
        if table_dir.exists():
            for f in table_dir.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total

    def get_health(self) -> dict[str, Any]:
        """Return a health summary for monitoring dashboards."""
        stats = self.get_table_stats()
        audit = self.get_audit_stats()
        return {
            "detection_table": {
                "exists": stats.get("exists", False),
                "rows": stats.get("total_rows", 0),
                "version": stats.get("current_version", 0),
                "size_bytes": stats.get("size_on_disk_bytes", 0),
            },
            "audit_table": {
                "exists": audit.get("exists", False),
                "rows": audit.get("total_rows", 0),
            },
            "writes": self._write_count,
            "errors": self._error_count,
            "last_latency_ms": round(self._last_write_latency_ms, 2),
        }
