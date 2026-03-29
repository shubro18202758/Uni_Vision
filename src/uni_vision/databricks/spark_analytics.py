"""PySpark batch analytics engine over Delta Lake tables.

Provides aggregation queries that leverage Apache Spark's distributed
compute engine to process ANPR detection data stored in Delta tables:

  * **Hourly rollups** — detection volume, avg confidence by hour.
  * **Plate frequency analysis** — most/least frequently seen plates.
  * **Cross-camera correlation** — plates appearing across cameras.
  * **Confidence trend analysis** — OCR accuracy trends over time.
  * **Camera performance comparison** — throughput & error rates.
  * **Temporal traffic patterns** — peak hours, quiet periods.

Runs locally with ``local[*]`` master by default but scales to
a Databricks cluster by simply changing the ``master`` URI.

Requires: ``pip install 'uni-vision[databricks]'``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports
_SparkSession = None
_F = None


def _ensure_imports() -> None:
    global _SparkSession, _F
    if _SparkSession is not None:
        return
    try:
        from pyspark.sql import SparkSession  # type: ignore[import-untyped]
        from pyspark.sql import functions as F  # type: ignore[import-untyped]
        _SparkSession = SparkSession
        _F = F
    except ImportError as exc:
        raise ImportError(
            "PySpark not installed. Run: pip install 'uni-vision[databricks]'"
        ) from exc


class SparkAnalyticsEngine:
    """PySpark-based batch analytics over Delta Lake detection data.

    Parameters
    ----------
    master : str
        Spark master URL (``local[*]`` for single-machine).
    app_name : str
        Spark application name visible in the Spark UI.
    driver_memory : str
        JVM heap for the Spark driver.
    shuffle_partitions : int
        Number of shuffle partitions (tune down for small data).
    delta_table_path : str
        Path to the detection Delta table.
    """

    def __init__(
        self,
        master: str = "local[*]",
        app_name: str = "UniVisionAnalytics",
        driver_memory: str = "2g",
        executor_memory: str = "1g",
        shuffle_partitions: int = 4,
        delta_table_path: str = "./data/delta/detections",
    ) -> None:
        _ensure_imports()
        self._master = master
        self._app_name = app_name
        self._driver_memory = driver_memory
        self._executor_memory = executor_memory
        self._shuffle_partitions = shuffle_partitions
        self._delta_path = str(Path(delta_table_path).resolve())
        self._spark = None

    # ── Lifecycle ─────────────────────────────────────────────────

    def initialise(self) -> None:
        """Create or get the SparkSession with Delta Lake support."""
        self._spark = (
            _SparkSession.builder
            .master(self._master)
            .appName(self._app_name)
            .config("spark.driver.memory", self._driver_memory)
            .config("spark.executor.memory", self._executor_memory)
            .config("spark.sql.shuffle.partitions", str(self._shuffle_partitions))
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0")
            .getOrCreate()
        )
        self._spark.sparkContext.setLogLevel("WARN")
        logger.info("spark_session_initialised master=%s", self._master)

    def shutdown(self) -> None:
        """Stop the SparkSession."""
        if self._spark is not None:
            self._spark.stop()
            self._spark = None
            logger.info("spark_session_stopped")

    def _read_delta(self):
        """Read the detection Delta table as a Spark DataFrame.

        Results are cached in memory for the lifetime of the session to
        avoid re-reading Parquet files on every query.  Call
        ``invalidate_cache()`` after new writes land.
        """
        if self._spark is None:
            self.initialise()
        df = self._spark.read.format("delta").load(self._delta_path)
        df.cache()
        return df

    def invalidate_cache(self) -> None:
        """Clear Spark's in-memory cache after new Delta writes."""
        if self._spark is not None:
            self._spark.catalog.clearCache()

    # ── Analytics queries ─────────────────────────────────────────

    def hourly_rollup(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Aggregate detection metrics by hour.

        Returns
        -------
        list[dict]
            Each entry: {hour, detection_count, avg_confidence,
            unique_plates, unique_cameras}.
        """
        F = _F
        df = self._read_delta()

        result = (
            df.withColumn("hour", F.date_trunc("hour", "detected_at_utc"))
            .groupBy("hour")
            .agg(
                F.count("*").alias("detection_count"),
                F.avg("ocr_confidence").alias("avg_confidence"),
                F.countDistinct("plate_number").alias("unique_plates"),
                F.countDistinct("camera_id").alias("unique_cameras"),
            )
            .orderBy(F.desc("hour"))
            .limit(hours_back)
        )
        return [row.asDict() for row in result.collect()]

    def plate_frequency(self, top_n: int = 30) -> List[Dict[str, Any]]:
        """Top-N most frequently detected plates.

        Returns
        -------
        list[dict]
            Each entry: {plate_number, count, avg_confidence,
            cameras_seen, first_seen, last_seen}.
        """
        F = _F
        df = self._read_delta()

        result = (
            df.groupBy("plate_number")
            .agg(
                F.count("*").alias("count"),
                F.avg("ocr_confidence").alias("avg_confidence"),
                F.countDistinct("camera_id").alias("cameras_seen"),
                F.min("detected_at_utc").alias("first_seen"),
                F.max("detected_at_utc").alias("last_seen"),
            )
            .orderBy(F.desc("count"))
            .limit(top_n)
        )
        rows = result.collect()
        return [
            {
                "plate_number": r["plate_number"],
                "count": r["count"],
                "avg_confidence": round(float(r["avg_confidence"] or 0), 4),
                "cameras_seen": r["cameras_seen"],
                "first_seen": str(r["first_seen"]),
                "last_seen": str(r["last_seen"]),
            }
            for r in rows
        ]

    def cross_camera_correlation(
        self, min_cameras: int = 2
    ) -> List[Dict[str, Any]]:
        """Find plates detected across multiple cameras.

        Useful for tracking vehicle movement patterns and
        identifying high-traffic / transit vehicles.
        """
        F = _F
        df = self._read_delta()

        correlated = (
            df.groupBy("plate_number")
            .agg(
                F.collect_set("camera_id").alias("cameras"),
                F.count("*").alias("total_detections"),
                F.avg("ocr_confidence").alias("avg_confidence"),
            )
            .withColumn("camera_count", F.size("cameras"))
            .filter(F.col("camera_count") >= min_cameras)
            .orderBy(F.desc("camera_count"), F.desc("total_detections"))
            .limit(50)
        )
        rows = correlated.collect()
        return [
            {
                "plate_number": r["plate_number"],
                "cameras": list(r["cameras"]),
                "camera_count": r["camera_count"],
                "total_detections": r["total_detections"],
                "avg_confidence": round(float(r["avg_confidence"] or 0), 4),
            }
            for r in rows
        ]

    def confidence_trend(self, bucket_hours: int = 1) -> List[Dict[str, Any]]:
        """Track OCR confidence trends over time.

        Useful for detecting model drift or environmental changes
        (lighting, weather) affecting recognition accuracy.
        """
        F = _F
        df = self._read_delta()

        result = (
            df.withColumn("bucket", F.date_trunc("hour", "detected_at_utc"))
            .groupBy("bucket")
            .agg(
                F.avg("ocr_confidence").alias("avg_confidence"),
                F.min("ocr_confidence").alias("min_confidence"),
                F.max("ocr_confidence").alias("max_confidence"),
                F.stddev("ocr_confidence").alias("stddev_confidence"),
                F.count("*").alias("count"),
            )
            .orderBy("bucket")
        )
        return [
            {
                "time_bucket": str(r["bucket"]),
                "avg_confidence": round(float(r["avg_confidence"] or 0), 4),
                "min_confidence": round(float(r["min_confidence"] or 0), 4),
                "max_confidence": round(float(r["max_confidence"] or 0), 4),
                "stddev_confidence": round(float(r["stddev_confidence"] or 0), 4),
                "count": r["count"],
            }
            for r in result.collect()
        ]

    def camera_performance(self) -> List[Dict[str, Any]]:
        """Compare per-camera detection volume and quality.

        Identifies cameras with low confidence or high error rates,
        feeding back into the agent's error profiling tools.
        """
        F = _F
        df = self._read_delta()

        result = (
            df.groupBy("camera_id")
            .agg(
                F.count("*").alias("total_detections"),
                F.avg("ocr_confidence").alias("avg_confidence"),
                F.countDistinct("plate_number").alias("unique_plates"),
                F.sum(
                    F.when(F.col("validation_status") != "valid", 1).otherwise(0)
                ).alias("error_count"),
                F.min("detected_at_utc").alias("first_detection"),
                F.max("detected_at_utc").alias("last_detection"),
            )
            .withColumn(
                "error_rate",
                F.round(F.col("error_count") / F.col("total_detections"), 4),
            )
            .orderBy(F.desc("total_detections"))
        )
        return [
            {
                "camera_id": r["camera_id"],
                "total_detections": r["total_detections"],
                "avg_confidence": round(float(r["avg_confidence"] or 0), 4),
                "unique_plates": r["unique_plates"],
                "error_count": r["error_count"],
                "error_rate": float(r["error_rate"] or 0),
                "first_detection": str(r["first_detection"]),
                "last_detection": str(r["last_detection"]),
            }
            for r in result.collect()
        ]

    def temporal_pattern(self) -> Dict[str, Any]:
        """Analyse traffic patterns by hour of day and day of week.

        Returns peak/quiet hours and day-of-week distributions
        useful for operational planning and anomaly detection.
        """
        F = _F
        df = self._read_delta()

        by_hour = (
            df.withColumn("hour_of_day", F.hour("detected_at_utc"))
            .groupBy("hour_of_day")
            .agg(F.count("*").alias("count"))
            .orderBy("hour_of_day")
            .collect()
        )

        by_dow = (
            df.withColumn("day_of_week", F.dayofweek("detected_at_utc"))
            .groupBy("day_of_week")
            .agg(F.count("*").alias("count"))
            .orderBy("day_of_week")
            .collect()
        )

        hourly = {r["hour_of_day"]: r["count"] for r in by_hour}
        daily = {r["day_of_week"]: r["count"] for r in by_dow}

        peak_hour = max(hourly, key=hourly.get) if hourly else None
        quiet_hour = min(hourly, key=hourly.get) if hourly else None

        return {
            "hourly_distribution": hourly,
            "daily_distribution": daily,
            "peak_hour": peak_hour,
            "quiet_hour": quiet_hour,
            "total_detections": sum(hourly.values()),
        }

    # ── Overview ──────────────────────────────────────────────────

    def anomaly_detection(self, z_threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect plates with anomalous confidence patterns via Z-score.

        Flags plates whose average confidence deviates significantly from
        the global mean, indicating possible OCR model drift or adversarial
        plate modifications.
        """
        F = _F
        df = self._read_delta()

        global_stats = df.agg(
            F.avg("ocr_confidence").alias("global_avg"),
            F.stddev("ocr_confidence").alias("global_std"),
        ).collect()[0]

        g_avg = float(global_stats["global_avg"] or 0)
        g_std = float(global_stats["global_std"] or 1)
        if g_std == 0:
            g_std = 1.0

        per_plate = (
            df.groupBy("plate_number")
            .agg(
                F.avg("ocr_confidence").alias("avg_confidence"),
                F.count("*").alias("count"),
                F.stddev("ocr_confidence").alias("plate_std"),
            )
            .filter(F.col("count") >= 3)
            .withColumn("z_score", (F.col("avg_confidence") - F.lit(g_avg)) / F.lit(g_std))
            .filter(F.abs(F.col("z_score")) >= z_threshold)
            .orderBy(F.desc(F.abs(F.col("z_score"))))
            .limit(50)
        )

        return [
            {
                "plate_number": r["plate_number"],
                "avg_confidence": round(float(r["avg_confidence"] or 0), 4),
                "count": r["count"],
                "z_score": round(float(r["z_score"] or 0), 3),
                "anomaly_type": "low_confidence" if r["z_score"] < 0 else "unusually_high",
            }
            for r in per_plate.collect()
        ]

    def detection_rate(self, bucket_minutes: int = 5) -> List[Dict[str, Any]]:
        """Calculate detection rate (plates per minute) in time buckets.

        Useful for capacity planning and identifying traffic surges.
        """
        F = _F
        df = self._read_delta()

        result = (
            df.withColumn(
                "bucket",
                F.window("detected_at_utc", f"{bucket_minutes} minutes"),
            )
            .groupBy("bucket")
            .agg(
                F.count("*").alias("detections"),
                F.countDistinct("plate_number").alias("unique_plates"),
                F.countDistinct("camera_id").alias("active_cameras"),
            )
            .withColumn("plates_per_min", F.col("detections") / F.lit(bucket_minutes))
            .orderBy(F.desc("bucket"))
            .limit(100)
        )

        return [
            {
                "window_start": str(r["bucket"]["start"]),
                "window_end": str(r["bucket"]["end"]),
                "detections": r["detections"],
                "unique_plates": r["unique_plates"],
                "active_cameras": r["active_cameras"],
                "plates_per_min": round(float(r["plates_per_min"]), 2),
            }
            for r in result.collect()
        ]

    def get_analytics_overview(self) -> Dict[str, Any]:
        """Return a high-level analytics summary."""
        F = _F
        try:
            df = self._read_delta()
            total = df.count()
            if total == 0:
                return {"total_detections": 0, "status": "empty"}

            stats = df.agg(
                F.count("*").alias("total"),
                F.avg("ocr_confidence").alias("avg_confidence"),
                F.countDistinct("plate_number").alias("unique_plates"),
                F.countDistinct("camera_id").alias("unique_cameras"),
                F.min("detected_at_utc").alias("earliest"),
                F.max("detected_at_utc").alias("latest"),
            ).collect()[0]

            return {
                "status": "active",
                "total_detections": stats["total"],
                "avg_confidence": round(float(stats["avg_confidence"] or 0), 4),
                "unique_plates": stats["unique_plates"],
                "unique_cameras": stats["unique_cameras"],
                "time_range": {
                    "earliest": str(stats["earliest"]),
                    "latest": str(stats["latest"]),
                },
                "spark_master": self._master,
                "delta_path": self._delta_path,
            }
        except Exception as exc:
            logger.warning("spark_analytics_overview_failed err=%s", exc)
            return {"status": "error", "error": str(exc)}
