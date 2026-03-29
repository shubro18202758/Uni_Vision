# Databricks notebook source
# MAGIC %md
# MAGIC # 🗄️ Uni Vision — Delta Lake Store Demo
# MAGIC
# MAGIC **ACID-transactional event storage** for anomaly detections.
# MAGIC
# MAGIC This notebook demonstrates:
# MAGIC - Schema enforcement with PyArrow
# MAGIC - ACID writes to Delta tables
# MAGIC - Time-travel queries (read historical versions)
# MAGIC - Partition pruning by `camera_id`
# MAGIC - Table statistics and version history
# MAGIC - Automated VACUUM for stale Parquet cleanup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1️⃣ Initialise the Delta Lake Store

# COMMAND ----------

import os
import sys
import uuid
from datetime import datetime, timezone, timedelta

# Delta Lake imports
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

# Paths (use DBFS for Databricks, local for Community Edition)
DELTA_TABLE_PATH = "/dbfs/uni_vision/delta/detections"
AUDIT_TABLE_PATH = "/dbfs/uni_vision/delta/audit_log"

# Fallback to local paths if /dbfs doesn't exist
if not os.path.exists("/dbfs"):
    DELTA_TABLE_PATH = "/tmp/uni_vision/delta/detections"
    AUDIT_TABLE_PATH = "/tmp/uni_vision/delta/audit_log"
    os.makedirs(DELTA_TABLE_PATH, exist_ok=True)
    os.makedirs(AUDIT_TABLE_PATH, exist_ok=True)

print(f"📂 Detection table: {DELTA_TABLE_PATH}")
print(f"📂 Audit table:     {AUDIT_TABLE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2️⃣ Define Detection & Audit Schemas

# COMMAND ----------

detection_schema = pa.schema([
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
])

audit_schema = pa.schema([
    pa.field("record_id", pa.string(), nullable=False),
    pa.field("camera_id", pa.string(), nullable=False),
    pa.field("raw_ocr_text", pa.string()),
    pa.field("ocr_confidence", pa.float64()),
    pa.field("failure_reason", pa.string()),
    pa.field("logged_at_utc", pa.timestamp("us", tz="UTC")),
])

print("✅ Schemas defined:")
print(f"   Detection: {len(detection_schema)} fields")
print(f"   Audit:     {len(audit_schema)} fields")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3️⃣ Generate Synthetic Detection Data
# MAGIC
# MAGIC Simulating real-time anomaly detection events from multiple cameras.

# COMMAND ----------

import random

CAMERAS = ["cam-north-01", "cam-south-02", "cam-east-03", "cam-west-04", "cam-gate-05"]
ANOMALY_TYPES = ["person_in_restricted_zone", "unattended_bag", "crowd_formation",
                 "vehicle_wrong_way", "fire_smoke_detected", "unusual_movement",
                 "perimeter_breach", "loitering_detected"]
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
VEHICLES = ["sedan", "suv", "truck", "motorcycle", "bus", "van", "auto-rickshaw"]

def generate_detection(i: int) -> dict:
    """Generate a single synthetic detection record."""
    now = datetime.now(timezone.utc)
    detected = now - timedelta(minutes=random.randint(0, 1440))  # last 24h
    anomaly = random.choice(ANOMALY_TYPES)
    camera = random.choice(CAMERAS)
    confidence = round(random.uniform(0.45, 0.99), 4)
    
    return {
        "id": str(uuid.uuid4()),
        "camera_id": camera,
        "plate_number": f"ANOMALY-{anomaly[:3].upper()}-{i:04d}",
        "raw_ocr_text": f"Scene analysis: {anomaly.replace('_', ' ')}",
        "ocr_confidence": confidence,
        "ocr_engine": "qwen3.5-9b-vision",
        "vehicle_class": random.choice(VEHICLES) if "vehicle" in anomaly else "N/A",
        "vehicle_image_path": f"s3://uni-vision/frames/{camera}/{detected.strftime('%Y%m%d_%H%M%S')}.jpg",
        "plate_image_path": f"s3://uni-vision/crops/{camera}/{detected.strftime('%Y%m%d_%H%M%S')}_crop.jpg",
        "detected_at_utc": detected,
        "validation_status": random.choice(["confirmed", "pending", "rejected"]),
        "location_tag": f"zone-{random.choice(['A', 'B', 'C', 'D'])}",
        "ingested_at_utc": now,
    }

# Generate 200 synthetic detections
detections = [generate_detection(i) for i in range(200)]
print(f"✅ Generated {len(detections)} synthetic detection events")
print(f"   Cameras: {len(CAMERAS)}")
print(f"   Anomaly types: {len(ANOMALY_TYPES)}")
print(f"\n📊 Sample record:")
for k, v in detections[0].items():
    print(f"   {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4️⃣ Write Detections to Delta Lake (ACID Append)

# COMMAND ----------

# Convert to PyArrow table
arrays = {field.name: [d[field.name] for d in detections] for field in detection_schema}
table = pa.table(arrays, schema=detection_schema)

# Write to Delta Lake with partitioning
write_deltalake(
    DELTA_TABLE_PATH,
    table,
    mode="overwrite",   # First write — create table
    schema=detection_schema,
    partition_by=["camera_id"],
)

print(f"✅ Wrote {len(detections)} records to Delta Lake")
print(f"   Path: {DELTA_TABLE_PATH}")
print(f"   Partitioned by: camera_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5️⃣ Read & Query the Delta Table

# COMMAND ----------

dt = DeltaTable(DELTA_TABLE_PATH)

# Basic stats
print("📊 Delta Table Statistics:")
print(f"   Version:       {dt.version()}")
print(f"   Files:          {len(dt.files())}")
print(f"   Partitions:     {dt.metadata().partition_columns}")
print(f"   Schema fields:  {len(dt.schema().to_pyarrow())}")

# Read all data
df = dt.to_pandas()
print(f"\n📋 Total records: {len(df)}")
print(f"\n📊 Records per camera:")
print(df["camera_id"].value_counts().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6️⃣ Append More Data (ACID Transaction)

# COMMAND ----------

# Generate 50 more detections (simulating new events)
new_detections = [generate_detection(200 + i) for i in range(50)]
new_arrays = {field.name: [d[field.name] for d in new_detections] for field in detection_schema}
new_table = pa.table(new_arrays, schema=detection_schema)

write_deltalake(DELTA_TABLE_PATH, new_table, mode="append")

dt = DeltaTable(DELTA_TABLE_PATH)
print(f"✅ Appended 50 new records")
print(f"   New version: {dt.version()}")
print(f"   Total files: {len(dt.files())}")
print(f"   Total rows:  {len(dt.to_pandas())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7️⃣ Time-Travel — Read Historical Version

# COMMAND ----------

# Read version 0 (the original 200 records)
dt_v0 = DeltaTable(DELTA_TABLE_PATH, version=0)
df_v0 = dt_v0.to_pandas()

# Read current version (250 records)
dt_latest = DeltaTable(DELTA_TABLE_PATH)
df_latest = dt_latest.to_pandas()

print("⏰ Time-Travel Query Results:")
print(f"   Version 0: {len(df_v0)} records")
print(f"   Version {dt_latest.version()}: {len(df_latest)} records")
print(f"   New records added: {len(df_latest) - len(df_v0)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8️⃣ Version History

# COMMAND ----------

history = dt_latest.history()
print("📜 Version History:")
print("-" * 70)
for entry in history:
    ts = entry.get("timestamp", "?")
    op = entry.get("operation", "?")
    ver = entry.get("version", "?")
    params = entry.get("operationParameters", {})
    print(f"  v{ver} | {ts} | {op} | {params}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9️⃣ Write Audit Log (Low-Confidence Detections)

# COMMAND ----------

os.makedirs(AUDIT_TABLE_PATH, exist_ok=True)

audit_records = []
for d in detections:
    if d["ocr_confidence"] < 0.6:
        audit_records.append({
            "record_id": d["id"],
            "camera_id": d["camera_id"],
            "raw_ocr_text": d["raw_ocr_text"],
            "ocr_confidence": d["ocr_confidence"],
            "failure_reason": "low_confidence" if d["ocr_confidence"] < 0.5 else "marginal_confidence",
            "logged_at_utc": datetime.now(timezone.utc),
        })

if audit_records:
    audit_arrays = {field.name: [r[field.name] for r in audit_records] for field in audit_schema}
    audit_table = pa.table(audit_arrays, schema=audit_schema)
    write_deltalake(AUDIT_TABLE_PATH, audit_table, mode="overwrite", partition_by=["camera_id"])
    
    print(f"⚠️ Audit Log: {len(audit_records)} low-confidence detections logged")
    print(f"   Path: {AUDIT_TABLE_PATH}")
else:
    print("✅ No low-confidence detections to audit")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Summary
# MAGIC
# MAGIC | Feature | Status |
# MAGIC |---------|--------|
# MAGIC | Schema enforcement | ✅ PyArrow typed schema |
# MAGIC | ACID writes | ✅ Atomic append + overwrite |
# MAGIC | Partitioning | ✅ By `camera_id` |
# MAGIC | Time-travel | ✅ Version 0 vs latest |
# MAGIC | Audit log | ✅ Separate Delta table |
# MAGIC | Version history | ✅ Full operation log |
