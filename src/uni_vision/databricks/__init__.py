"""Databricks technology integrations for Uni_Vision.

Add-on package providing Delta Lake, MLflow, PySpark analytics,
and FAISS vector search capabilities.  All components are optional
and gated behind ``databricks.enabled`` in the application config.
Nothing in this package replaces existing PostgreSQL, Redis or S3
infrastructure — it runs alongside them as a complementary
analytics and ML-ops layer.

Components
----------
- **DeltaLakeStore** — Detection event sink with ACID transactions,
  schema enforcement, time-travel queries, and partition pruning.
- **InferenceTracker** — MLflow experiment tracking for pipeline
  stage metrics (latency, confidence, VRAM usage).
- **SparkAnalyticsEngine** — PySpark batch analytics over Delta
  tables: hourly rollups, plate frequency, cross-camera correlation.
- **VectorSearchEngine** — FAISS index for plate-text similarity
  search and agent knowledge RAG enhancement.
"""

from __future__ import annotations

__all__ = [
    "DeltaLakeStore",
    "InferenceTracker",
    "SparkAnalyticsEngine",
    "VectorSearchEngine",
]


def __getattr__(name: str):
    """Lazy-import heavy classes so the package is importable without
    the ``databricks`` optional dependency group installed."""

    if name == "DeltaLakeStore":
        from uni_vision.databricks.delta_store import DeltaLakeStore
        return DeltaLakeStore

    if name == "InferenceTracker":
        from uni_vision.databricks.mlflow_tracker import InferenceTracker
        return InferenceTracker

    if name == "SparkAnalyticsEngine":
        from uni_vision.databricks.spark_analytics import SparkAnalyticsEngine
        return SparkAnalyticsEngine

    if name == "VectorSearchEngine":
        from uni_vision.databricks.vector_search import VectorSearchEngine
        return VectorSearchEngine

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
