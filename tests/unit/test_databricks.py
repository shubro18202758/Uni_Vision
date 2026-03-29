"""Comprehensive unit tests for the Databricks integration modules.

Tests cover DeltaLakeStore, InferenceTracker, SparkAnalyticsEngine, and
VectorSearchEngine — all with mocked heavy dependencies so the tests
run fast without installing PySpark / FAISS / MLflow / deltalake.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest

# ────────────────────────────────────────────────────────────────
# Helper: lightweight numpy-like array stub for FAISS tests
# ────────────────────────────────────────────────────────────────

class _FakeNP:
    """Minimal numpy stub."""
    float32 = "float32"

    @staticmethod
    def array(data, dtype=None):
        return data

    @staticmethod
    def zeros(shape, dtype=None):
        return [[0.0] * shape[1] for _ in range(shape[0])]

    @staticmethod
    def where(arr):
        return ([i for i, v in enumerate(arr) if v],)

    @staticmethod
    def dot(a, b):
        # Just return ones for coherence test
        return [[1.0] * len(a)]

    @staticmethod
    def mean(arr):
        if not len(arr):
            return 0.0
        return sum(arr) / len(arr) if isinstance(arr, list) else 1.0


# ════════════════════════════════════════════════════════════════
# DeltaLakeStore tests
# ════════════════════════════════════════════════════════════════


class TestDeltaLakeStore:
    """Tests for src/uni_vision/databricks/delta_store.py."""

    @pytest.fixture(autouse=True)
    def _patch_deltalake(self, tmp_path):
        """Patch the lazy imports so we never need real deltalake/pyarrow."""
        self.det_path = str(tmp_path / "detections")
        self.aud_path = str(tmp_path / "audits")

        # Fake DeltaTable
        fake_dt = MagicMock()
        fake_dt.version.return_value = 3
        fake_dt.history.return_value = [
            {"version": 3, "timestamp": "2025-01-01"},
            {"version": 2, "timestamp": "2024-12-31"},
        ]
        fake_dt.to_pyarrow_table.return_value = MagicMock(
            num_rows=42,
            schema=MagicMock(names=["plate_text", "camera_id"]),
        )
        fake_dt.metadata.return_value = MagicMock(partition_columns=["camera_id"])
        fake_dt.optimize = MagicMock()
        fake_dt.optimize.compact.return_value = None

        self.fake_dt_cls = MagicMock(return_value=fake_dt)

        # Fake write_deltalake
        self.fake_write = MagicMock()

        fake_pa = MagicMock()
        fake_pa.Table.from_pydict.return_value = MagicMock(nbytes=1024)

        with (
            patch.dict("sys.modules", {
                "deltalake": MagicMock(DeltaTable=self.fake_dt_cls, write_deltalake=self.fake_write),
                "pyarrow": fake_pa,
            }),
            patch("uni_vision.databricks.delta_store._DeltaTable", self.fake_dt_cls),
            patch("uni_vision.databricks.delta_store._write_deltalake", self.fake_write),
            patch("uni_vision.databricks.delta_store._pa", fake_pa),
            patch("uni_vision.databricks.delta_store._ensure_imports", lambda: None),
        ):
            from uni_vision.databricks.delta_store import DeltaLakeStore
            self.store = DeltaLakeStore(
                table_path=self.det_path,
                audit_table_path=self.aud_path,
            )
            self.store._det_table = fake_dt
            self.store._aud_table = fake_dt
            yield

    def test_thread_safety_fields_exist(self):
        assert hasattr(self.store, "_lock")
        assert isinstance(self.store._lock, type(threading.Lock()))

    def test_get_table_stats_includes_metrics(self):
        stats = self.store.get_table_stats()
        assert "total_bytes_written" in stats
        assert "last_write_latency_ms" in stats
        assert "error_count" in stats

    def test_get_health_returns_dict(self):
        health = self.store.get_health()
        assert isinstance(health, dict)
        assert "detection_table" in health
        assert "audit_table" in health
        assert "errors" in health


# ════════════════════════════════════════════════════════════════
# InferenceTracker (MLflow) tests
# ════════════════════════════════════════════════════════════════


class TestInferenceTracker:
    """Tests for src/uni_vision/databricks/mlflow_tracker.py."""

    @pytest.fixture(autouse=True)
    def _patch_mlflow(self, tmp_path):
        fake_mlflow = MagicMock()
        fake_mlflow.set_tracking_uri = MagicMock()
        fake_mlflow.set_experiment = MagicMock()
        fake_run = MagicMock()
        fake_run.info.run_id = "test-run-123"
        fake_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=fake_run)
        fake_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        fake_mlflow.active_run.return_value = fake_run
        fake_mlflow.log_metrics = MagicMock()
        fake_mlflow.set_tags = MagicMock()
        fake_mlflow.search_experiments.return_value = []
        fake_runs_df = MagicMock()
        fake_runs_df.empty = True
        fake_runs_df.__len__ = lambda self: 0
        fake_mlflow.search_runs.return_value = fake_runs_df

        with (
            patch.dict("sys.modules", {"mlflow": fake_mlflow}),
            patch("uni_vision.databricks.mlflow_tracker._mlflow", fake_mlflow),
            patch("uni_vision.databricks.mlflow_tracker._ensure_imports", lambda: None),
        ):
            from uni_vision.databricks.mlflow_tracker import InferenceTracker
            self.tracker = InferenceTracker(
                tracking_uri=str(tmp_path / "mlruns"),
                experiment_name="test-exp",
            )
            self.tracker._active_run = fake_run
            self.tracker._run_id = "test-run-123"
            self.fake_mlflow = fake_mlflow
            yield

    def test_log_stage_metrics_increments_counter(self):
        before = self.tracker._frame_count
        self.tracker.log_stage_metrics(
            stage="S7_ocr",
            latency_ms=42.0,
            confidence=0.85,
        )
        assert self.tracker._frame_count == before + 1

    def test_log_stage_metrics_adds_to_buffer(self):
        self.tracker.log_stage_metrics(
            stage="S2_detect",
            latency_ms=10.0,
            confidence=0.9,
        )
        assert len(self.tracker._buffer) > 0

    def test_tag_run_calls_mlflow(self):
        self.tracker.tag_run({"env": "test", "version": "1.0"})
        self.fake_mlflow.set_tags.assert_called_once_with({"env": "test", "version": "1.0"})

    def test_get_health_returns_expected_keys(self):
        health = self.tracker.get_health()
        assert "active" in health
        assert "experiment" in health
        assert "frames_logged" in health
        assert "buffer_pending" in health

    def test_get_experiment_summary_no_init(self):
        # Without initialise(), experiment_id is None → returns {active: False}
        summary = self.tracker.get_experiment_summary()
        assert isinstance(summary, dict)
        assert summary["active"] is False

    def test_get_experiment_summary_with_id(self):
        self.tracker._experiment_id = "exp-001"
        summary = self.tracker.get_experiment_summary()
        assert isinstance(summary, dict)
        assert "experiment_name" in summary


# ════════════════════════════════════════════════════════════════
# SparkAnalyticsEngine tests
# ════════════════════════════════════════════════════════════════


class TestSparkAnalyticsEngine:
    """Tests for src/uni_vision/databricks/spark_analytics.py."""

    @pytest.fixture(autouse=True)
    def _patch_pyspark(self, tmp_path):
        fake_spark_builder = MagicMock()
        fake_spark = MagicMock()
        fake_spark_builder.getOrCreate.return_value = fake_spark
        fake_spark.read.format.return_value.load.return_value = MagicMock()
        fake_spark.catalog.clearCache = MagicMock()

        with (
            patch.dict("sys.modules", {
                "pyspark": MagicMock(),
                "pyspark.sql": MagicMock(),
                "pyspark.sql.functions": MagicMock(),
                "pyspark.sql.window": MagicMock(),
            }),
            patch("uni_vision.databricks.spark_analytics._SparkSession", MagicMock(
                builder=MagicMock(
                    appName=MagicMock(return_value=MagicMock(
                        master=MagicMock(return_value=MagicMock(
                            config=MagicMock(return_value=MagicMock(
                                config=MagicMock(return_value=fake_spark_builder)
                            ))
                        ))
                    ))
                )
            )),
            patch("uni_vision.databricks.spark_analytics._F", MagicMock()),
            patch("uni_vision.databricks.spark_analytics._ensure_imports", lambda: None),
        ):
            from uni_vision.databricks.spark_analytics import SparkAnalyticsEngine
            self.engine = SparkAnalyticsEngine(
                delta_table_path=str(tmp_path / "delta"),
                app_name="test-spark",
            )
            self.engine._spark = fake_spark
            yield

    def test_invalidate_cache(self):
        self.engine.invalidate_cache()
        self.engine._spark.catalog.clearCache.assert_called_once()

    def test_get_analytics_overview_returns_dict(self):
        # Mock the internal _read_delta to return a fake dataframe
        fake_df = MagicMock()
        fake_df.count.return_value = 100
        fake_df.select.return_value.distinct.return_value.count.return_value = 5
        fake_agg = MagicMock()
        fake_agg.collect.return_value = [MagicMock(asDict=MagicMock(
            return_value={"avg_conf": 0.85}
        ))]
        fake_df.agg.return_value = fake_agg
        self.engine._read_delta = MagicMock(return_value=fake_df)

        overview = self.engine.get_analytics_overview()
        assert isinstance(overview, dict)


# ════════════════════════════════════════════════════════════════
# VectorSearchEngine tests
# ════════════════════════════════════════════════════════════════


class TestVectorSearchEngine:
    """Tests for src/uni_vision/databricks/vector_search.py."""

    @pytest.fixture(autouse=True)
    def _patch_faiss(self, tmp_path):
        # Create fake FAISS index
        fake_index = MagicMock()
        fake_index.ntotal = 0
        fake_index.add = MagicMock(side_effect=lambda v: setattr(
            fake_index, "ntotal", fake_index.ntotal + (len(v) if hasattr(v, "__len__") else 1)
        ))
        fake_index.search = MagicMock(return_value=(
            [[0.95, 0.88, 0.42]],  # scores
            [[0, 1, 2]],            # indices
        ))
        fake_index.reconstruct = MagicMock(return_value=[0.1] * 384)

        fake_faiss = MagicMock()
        fake_faiss.IndexFlatIP = MagicMock(return_value=fake_index)
        fake_faiss.read_index = MagicMock(return_value=fake_index)
        fake_faiss.write_index = MagicMock()
        fake_kmeans = MagicMock()
        fake_kmeans.train = MagicMock()
        fake_kmeans.index.search = MagicMock(return_value=(
            [[0.9]], [[0]]
        ))
        fake_kmeans.centroids = [[0.1] * 384]
        fake_faiss.Kmeans = MagicMock(return_value=fake_kmeans)

        fake_model = MagicMock()
        fake_model.encode = MagicMock(return_value=[[0.1] * 384])

        fake_np = _FakeNP()

        with (
            patch("uni_vision.databricks.vector_search._faiss", fake_faiss),
            patch("uni_vision.databricks.vector_search._SentenceTransformer", MagicMock(return_value=fake_model)),
            patch("uni_vision.databricks.vector_search._np", fake_np),
            patch("uni_vision.databricks.vector_search._ensure_imports", lambda: None),
        ):
            from uni_vision.databricks.vector_search import VectorSearchEngine
            self.engine = VectorSearchEngine(
                index_path=str(tmp_path / "index.bin"),
                metadata_path=str(tmp_path / "meta.json"),
            )
            self.engine._index = fake_index
            self.engine._model = fake_model
            self.fake_index = fake_index
            self.fake_faiss = fake_faiss
            yield

    def test_add_plate_observation(self):
        vid = self.engine.add_plate_observation(
            plate_text="ABC123",
            camera_id="cam1",
            confidence=0.92,
            engine="easyocr",
            validation_status="valid",
        )
        assert vid == 0
        assert len(self.engine._metadata) == 1
        assert self.engine._metadata[0]["plate_text"] == "ABC123"

    def test_add_batch(self):
        obs = [
            {"plate_text": "XYZ789", "camera_id": "cam1", "confidence": 0.8,
             "engine": "paddleocr", "validation_status": "valid"},
            {"plate_text": "DEF456", "camera_id": "cam2", "confidence": 0.75,
             "engine": "easyocr", "validation_status": "partial"},
        ]
        count = self.engine.add_batch(obs)
        assert count == 2
        assert len(self.engine._metadata) == 2

    def test_search_similar_plates(self):
        # Pre-populate metadata so search has something to match
        self.engine._metadata = [
            {"plate_text": "ABC123", "camera_id": "cam1", "confidence": 0.9,
             "engine": "easyocr", "validation_status": "valid", "timestamp": 1000.0},
            {"plate_text": "ABC124", "camera_id": "cam2", "confidence": 0.85,
             "engine": "paddleocr", "validation_status": "valid", "timestamp": 1001.0},
            {"plate_text": "XYZ999", "camera_id": "cam1", "confidence": 0.7,
             "engine": "easyocr", "validation_status": "partial", "timestamp": 1002.0},
        ]
        self.fake_index.ntotal = 3
        results = self.engine.search_similar_plates("ABC123", top_k=3, threshold=0.4)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "plate_text" in results[0]
        assert "similarity" in results[0]

    def test_search_by_time_range(self):
        self.engine._metadata = [
            {"plate_text": "T1", "camera_id": "c1", "confidence": 0.9,
             "engine": "e", "validation_status": "v", "timestamp": 1000.0},
            {"plate_text": "T2", "camera_id": "c1", "confidence": 0.8,
             "engine": "e", "validation_status": "v", "timestamp": 2000.0},
            {"plate_text": "T3", "camera_id": "c1", "confidence": 0.7,
             "engine": "e", "validation_status": "v", "timestamp": 3000.0},
        ]
        self.fake_index.ntotal = 3
        results = self.engine.search_by_time_range("T1", start_ts=900, end_ts=1100, top_k=10)
        # All returned results should be within the time window
        for r in results:
            assert 900 <= r["timestamp"] <= 1100

    def test_find_potential_duplicates_empty(self):
        self.fake_index.ntotal = 0
        dupes = self.engine.find_potential_duplicates()
        assert dupes == []

    def test_get_stats(self):
        stats = self.engine.get_stats()
        assert "total_vectors" in stats
        assert "embedding_dim" in stats
        assert stats["embedding_dim"] == 384

    def test_get_health(self):
        self.engine._metadata = [
            {"plate_text": "A1", "camera_id": "c1"},
            {"plate_text": "A2", "camera_id": "c2"},
        ]
        health = self.engine.get_health()
        assert health["unique_plates"] == 2
        assert health["unique_cameras"] == 2
        assert "index_type" in health

    def test_rebuild_index_empty(self):
        self.engine._metadata = []
        result = self.engine.rebuild_index()
        assert result["status"] == "empty"


# ════════════════════════════════════════════════════════════════
# API Routes tests
# ════════════════════════════════════════════════════════════════


class TestDatabricksRoutes:
    """Tests for the Databricks API route handlers."""

    @pytest.fixture
    def mock_services(self):
        """Create mock Databricks services."""
        delta = MagicMock()
        delta.get_table_stats.return_value = {"num_rows": 100, "version": 5}
        delta.get_audit_stats.return_value = {"num_rows": 10}
        delta.get_version_history.return_value = [{"version": 5}]
        delta.read_recent.return_value = [{"plate": "ABC"}]
        delta.compact.return_value = {"status": "compacted"}
        delta.get_health.return_value = {"status": "active"}

        mlflow = MagicMock()
        mlflow.get_experiment_summary.return_value = {"experiment_name": "test", "total_runs": 5}
        mlflow.get_metric_history.return_value = [{"value": 0.9}]
        mlflow.get_health.return_value = {"status": "active"}

        spark = MagicMock()
        spark.get_analytics_overview.return_value = {"status": "ok", "total_detections": 500}
        spark.hourly_rollup.return_value = [{"hour": 1, "count": 10}]
        spark.plate_frequency.return_value = [{"plate": "ABC", "count": 5}]
        spark.anomaly_detection.return_value = [{"plate_text": "X", "z_score": 3.1}]
        spark.detection_rate.return_value = [{"bucket": "12:00", "plates_per_min": 2.5}]

        vector = MagicMock()
        vector.get_stats.return_value = {"total_vectors": 200}
        vector.search_similar_plates.return_value = [{"plate_text": "ABC", "similarity": 0.95}]
        vector.search_by_time_range.return_value = [{"plate_text": "ABC", "similarity": 0.9}]
        vector.find_potential_duplicates.return_value = []
        vector.get_cluster_analysis.return_value = {"n_clusters": 5, "clusters": []}
        vector.get_health.return_value = {"status": "active"}

        return {"delta": delta, "mlflow": mlflow, "spark": spark, "vector": vector}

    @pytest.fixture
    def app(self, mock_services):
        """Create a test FastAPI app with mocked Databricks services."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from uni_vision.api.routes.databricks_routes import router

        app = FastAPI()
        app.include_router(router)

        # Attach mock services to app state
        app.state.databricks_delta = mock_services["delta"]
        app.state.databricks_mlflow = mock_services["mlflow"]
        app.state.databricks_spark = mock_services["spark"]
        app.state.databricks_vector = mock_services["vector"]

        return TestClient(app)

    def test_overview(self, app, mock_services):
        resp = app.get("/api/databricks/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True

    def test_delta_stats(self, app, mock_services):
        resp = app.get("/api/databricks/delta/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        assert "audits" in data

    def test_delta_history(self, app):
        resp = app.get("/api/databricks/delta/history?limit=5")
        assert resp.status_code == 200
        assert "versions" in resp.json()

    def test_delta_compact(self, app, mock_services):
        resp = app.post("/api/databricks/delta/compact")
        assert resp.status_code == 200
        mock_services["delta"].compact.assert_called_once()

    def test_mlflow_summary(self, app):
        resp = app.get("/api/databricks/mlflow/summary")
        assert resp.status_code == 200

    def test_spark_analytics_anomaly(self, app, mock_services):
        resp = app.post(
            "/api/databricks/spark/analytics",
            json={"query_type": "anomaly_detection", "z_threshold": 2.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query_type"] == "anomaly_detection"
        mock_services["spark"].anomaly_detection.assert_called_once()

    def test_spark_analytics_detection_rate(self, app, mock_services):
        resp = app.post(
            "/api/databricks/spark/analytics",
            json={"query_type": "detection_rate", "bucket_minutes": 5},
        )
        assert resp.status_code == 200
        mock_services["spark"].detection_rate.assert_called_once()

    def test_spark_analytics_invalid_type(self, app):
        resp = app.post(
            "/api/databricks/spark/analytics",
            json={"query_type": "nonexistent_query"},
        )
        assert resp.status_code == 400

    def test_vector_search(self, app, mock_services):
        resp = app.post(
            "/api/databricks/vector/search",
            json={"query": "ABC123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "ABC123"
        assert "results" in data

    def test_vector_time_range_search(self, app, mock_services):
        resp = app.post(
            "/api/databricks/vector/search/time-range",
            json={"query": "ABC", "start_ts": 1000.0, "end_ts": 2000.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "time_range" in data

    def test_vector_clusters(self, app, mock_services):
        resp = app.get("/api/databricks/vector/clusters?n_clusters=5")
        assert resp.status_code == 200
        mock_services["vector"].get_cluster_analysis.assert_called_once_with(n_clusters=5)

    def test_vector_duplicates(self, app):
        resp = app.get("/api/databricks/vector/duplicates?threshold=0.9")
        assert resp.status_code == 200
        data = resp.json()
        assert "duplicates" in data

    def test_health_endpoint(self, app):
        resp = app.get("/api/databricks/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall"] == "ok"
        assert "delta" in data
        assert "mlflow" in data
        assert "spark" in data
        assert "vector" in data

    def test_503_when_service_missing(self):
        """Endpoints return 503 when Databricks is not enabled."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from uni_vision.api.routes.databricks_routes import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/api/databricks/delta/stats")
        assert resp.status_code == 503
