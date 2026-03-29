# Databricks notebook source
# MAGIC %md
# MAGIC # ⚡ Uni Vision — PySpark Analytics Engine Demo
# MAGIC
# MAGIC **Distributed batch analytics** over Delta Lake detection data using Apache Spark.
# MAGIC
# MAGIC This notebook demonstrates:
# MAGIC - Reading Delta tables into Spark DataFrames
# MAGIC - Hourly detection rollups
# MAGIC - Anomaly frequency analysis
# MAGIC - Cross-camera correlation
# MAGIC - Confidence trend analysis
# MAGIC - Z-score anomaly detection
# MAGIC - Temporal traffic patterns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1️⃣ Initialise Spark Session with Delta Lake

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os

# Create Spark session with Delta Lake support
spark = (
    SparkSession.builder
    .appName("UniVisionAnalytics")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

print(f"✅ Spark session active: {spark.sparkContext.appName}")
print(f"   Spark version:  {spark.version}")
print(f"   Master:         {spark.sparkContext.master}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2️⃣ Generate Comprehensive Detection Dataset
# MAGIC
# MAGIC Creating a rich synthetic dataset with realistic anomaly patterns.

# COMMAND ----------

import random
from datetime import datetime, timedelta

CAMERAS = ["cam-north-01", "cam-south-02", "cam-east-03", "cam-west-04", "cam-gate-05"]
ANOMALY_TYPES = [
    "person_in_restricted_zone", "unattended_bag", "crowd_formation",
    "vehicle_wrong_way", "fire_smoke_detected", "unusual_movement",
    "perimeter_breach", "loitering_detected", "tailgating",
    "abandoned_vehicle", "running_in_corridor", "unauthorized_access",
]
ZONES = ["zone-A", "zone-B", "zone-C", "zone-D"]

# Generate 1000 records spanning 7 days
now = datetime.now()
records = []
for i in range(1000):
    detected = now - timedelta(hours=random.randint(0, 168))
    camera = random.choice(CAMERAS)
    anomaly = random.choice(ANOMALY_TYPES)
    # Cameras have different confidence profiles
    base_conf = {"cam-north-01": 0.85, "cam-south-02": 0.78, "cam-east-03": 0.90,
                 "cam-west-04": 0.72, "cam-gate-05": 0.88}
    conf = min(0.99, max(0.3, random.gauss(base_conf[camera], 0.08)))
    
    records.append((
        f"det-{i:05d}",
        camera,
        anomaly,
        f"Scene: {anomaly.replace('_', ' ')} detected",
        round(conf, 4),
        "qwen3.5-9b-vision",
        random.choice(["confirmed", "pending", "rejected"]),
        random.choice(ZONES),
        detected.strftime("%Y-%m-%d %H:%M:%S"),
    ))

# Create Spark DataFrame
columns = ["id", "camera_id", "anomaly_type", "description", "confidence",
           "model", "status", "zone", "detected_at"]

df = spark.createDataFrame(records, columns)
df = df.withColumn("detected_at", F.to_timestamp("detected_at"))
df = df.withColumn("hour", F.hour("detected_at"))
df = df.withColumn("day_of_week", F.dayofweek("detected_at"))
df = df.withColumn("date", F.to_date("detected_at"))

df.cache()
print(f"✅ Created DataFrame with {df.count()} detection records")
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3️⃣ Hourly Detection Rollup

# COMMAND ----------

hourly = (
    df.groupBy("date", "hour")
    .agg(
        F.count("*").alias("detection_count"),
        F.avg("confidence").alias("avg_confidence"),
        F.min("confidence").alias("min_confidence"),
        F.max("confidence").alias("max_confidence"),
        F.countDistinct("camera_id").alias("active_cameras"),
        F.countDistinct("anomaly_type").alias("unique_anomalies"),
    )
    .orderBy("date", "hour")
)

print("📊 Hourly Detection Rollup (last 10 hours):")
hourly.show(10, truncate=False)

# Busiest hours
print("\n🔥 Top 5 Busiest Hours:")
hourly.orderBy(F.desc("detection_count")).show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4️⃣ Anomaly Type Frequency Analysis

# COMMAND ----------

anomaly_freq = (
    df.groupBy("anomaly_type")
    .agg(
        F.count("*").alias("total_count"),
        F.avg("confidence").alias("avg_confidence"),
        F.countDistinct("camera_id").alias("cameras_seen"),
        F.sum(F.when(F.col("status") == "confirmed", 1).otherwise(0)).alias("confirmed"),
        F.sum(F.when(F.col("status") == "rejected", 1).otherwise(0)).alias("rejected"),
    )
    .withColumn("confirmation_rate", F.round(F.col("confirmed") / F.col("total_count") * 100, 1))
    .orderBy(F.desc("total_count"))
)

print("🚨 Anomaly Type Frequency Analysis:")
anomaly_freq.show(15, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5️⃣ Cross-Camera Correlation
# MAGIC
# MAGIC Find anomaly types that appear across multiple cameras — potential coordinated events.

# COMMAND ----------

cross_camera = (
    df.groupBy("anomaly_type", "camera_id")
    .agg(F.count("*").alias("count"))
    .groupBy("anomaly_type")
    .agg(
        F.count("camera_id").alias("cameras_observed"),
        F.sum("count").alias("total_events"),
        F.collect_list(
            F.struct("camera_id", "count")
        ).alias("camera_breakdown"),
    )
    .filter(F.col("cameras_observed") >= 3)
    .orderBy(F.desc("cameras_observed"), F.desc("total_events"))
)

print("🔗 Cross-Camera Correlation (anomalies appearing in 3+ cameras):")
cross_camera.select("anomaly_type", "cameras_observed", "total_events").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6️⃣ Camera Performance Comparison

# COMMAND ----------

camera_perf = (
    df.groupBy("camera_id")
    .agg(
        F.count("*").alias("total_detections"),
        F.avg("confidence").alias("avg_confidence"),
        F.stddev("confidence").alias("stddev_confidence"),
        F.countDistinct("anomaly_type").alias("anomaly_diversity"),
        F.sum(F.when(F.col("confidence") < 0.6, 1).otherwise(0)).alias("low_conf_count"),
        F.sum(F.when(F.col("status") == "confirmed", 1).otherwise(0)).alias("confirmed_count"),
    )
    .withColumn("low_conf_pct", F.round(F.col("low_conf_count") / F.col("total_detections") * 100, 1))
    .withColumn("confirmation_rate", F.round(F.col("confirmed_count") / F.col("total_detections") * 100, 1))
    .orderBy(F.desc("avg_confidence"))
)

print("📹 Camera Performance Comparison:")
camera_perf.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7️⃣ Confidence Trend Analysis (Moving Average)

# COMMAND ----------

# Window for moving average
window_spec = Window.orderBy("date").rowsBetween(-2, 0)

daily_conf = (
    df.groupBy("date")
    .agg(
        F.avg("confidence").alias("daily_avg_confidence"),
        F.count("*").alias("daily_count"),
    )
    .withColumn("moving_avg_3day", F.avg("daily_avg_confidence").over(window_spec))
    .orderBy("date")
)

print("📈 Daily Confidence Trend with 3-Day Moving Average:")
daily_conf.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8️⃣ Z-Score Anomaly Detection
# MAGIC
# MAGIC Flag cameras with statistically anomalous detection counts using Z-scores.

# COMMAND ----------

# Calculate per-camera hourly counts
camera_hourly = (
    df.groupBy("camera_id", "date", "hour")
    .agg(F.count("*").alias("hourly_count"))
)

# Global statistics
global_stats = camera_hourly.agg(
    F.avg("hourly_count").alias("global_mean"),
    F.stddev("hourly_count").alias("global_std"),
).collect()[0]

global_mean = global_stats["global_mean"]
global_std = global_stats["global_std"] or 1.0

# Calculate Z-scores
z_scores = (
    camera_hourly
    .withColumn("z_score", (F.col("hourly_count") - F.lit(global_mean)) / F.lit(global_std))
    .filter(F.abs(F.col("z_score")) > 2.0)
    .orderBy(F.desc(F.abs(F.col("z_score"))))
)

print(f"📊 Statistical Anomaly Detection")
print(f"   Global mean (detections/hour/camera): {global_mean:.2f}")
print(f"   Global std deviation:                  {global_std:.2f}")
print(f"   Z-score threshold:                     ±2.0")
print(f"\n🚨 Anomalous Camera-Hours (|Z| > 2.0):")
z_scores.show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9️⃣ Temporal Traffic Patterns

# COMMAND ----------

temporal = (
    df.groupBy("hour")
    .agg(
        F.count("*").alias("total_detections"),
        F.avg("confidence").alias("avg_confidence"),
    )
    .orderBy("hour")
)

print("🕐 Hourly Detection Pattern (24h):")
temporal_data = temporal.collect()
print(f"{'Hour':<6} {'Detections':<12} {'Confidence':<12} {'Bar'}")
print("-" * 60)
max_count = max(r["total_detections"] for r in temporal_data) if temporal_data else 1
for r in temporal_data:
    bar_len = int(r["total_detections"] / max_count * 30)
    bar = "█" * bar_len
    print(f"  {r['hour']:02d}   {r['total_detections']:<12} {r['avg_confidence']:<12.4f} {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔟 Zone-Level Heatmap

# COMMAND ----------

zone_camera = (
    df.groupBy("zone", "camera_id")
    .agg(
        F.count("*").alias("events"),
        F.avg("confidence").alias("avg_conf"),
    )
    .orderBy("zone", "camera_id")
)

print("🗺️ Zone × Camera Detection Heatmap:")
zone_camera.show(20, truncate=False)

# Summary by zone
zone_summary = (
    df.groupBy("zone")
    .agg(
        F.count("*").alias("total"),
        F.avg("confidence").alias("avg_confidence"),
        F.countDistinct("anomaly_type").alias("anomaly_types"),
    )
    .orderBy(F.desc("total"))
)
print("\n📍 Zone Summary:")
zone_summary.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Summary
# MAGIC
# MAGIC | Analysis | Status |
# MAGIC |----------|--------|
# MAGIC | Hourly rollup | ✅ Detection volume by hour |
# MAGIC | Anomaly frequency | ✅ Type distribution + confirmation rates |
# MAGIC | Cross-camera correlation | ✅ Multi-camera anomaly patterns |
# MAGIC | Camera performance | ✅ Confidence, diversity, error rates |
# MAGIC | Confidence trends | ✅ Daily moving average |
# MAGIC | Z-score anomaly detection | ✅ Statistical outlier flagging |
# MAGIC | Temporal patterns | ✅ 24-hour heatmap |
# MAGIC | Zone heatmap | ✅ Spatial distribution |
