# Databricks notebook source
# MAGIC %md
# MAGIC # 📊 Uni Vision — MLflow Inference Tracking Demo
# MAGIC
# MAGIC **Experiment tracking for the anomaly detection pipeline.**
# MAGIC
# MAGIC This notebook demonstrates:
# MAGIC - MLflow experiment creation and management
# MAGIC - Per-stage metric logging (latency, confidence, VRAM)
# MAGIC - Model parameter tracking
# MAGIC - Run comparison and metric history
# MAGIC - Experiment summaries and visualisation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1️⃣ Initialise MLflow Tracking

# COMMAND ----------

import os
import time
import random
import mlflow

# Set tracking URI
TRACKING_URI = "/dbfs/uni_vision/mlflow"
if not os.path.exists("/dbfs"):
    TRACKING_URI = "/tmp/uni_vision/mlflow"

os.makedirs(TRACKING_URI, exist_ok=True)
mlflow.set_tracking_uri(TRACKING_URI)

EXPERIMENT_NAME = "uni-vision-inference"

# Create or get experiment
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(
        EXPERIMENT_NAME,
        artifact_location=os.path.join(TRACKING_URI, "artifacts"),
    )
    print(f"✅ Created new experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
else:
    experiment_id = experiment.experiment_id
    print(f"✅ Using existing experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")

mlflow.set_experiment(EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2️⃣ Simulate Pipeline Inference Runs
# MAGIC
# MAGIC Each run represents processing a batch of video frames through the multi-stage pipeline:
# MAGIC - **Stage S2**: Object Detection (YOLOv8)
# MAGIC - **Stage S3**: Region of Interest Extraction
# MAGIC - **Stage S7**: LLM Vision Analysis (Gemma 4 E2B)
# MAGIC - **Stage S8**: Anomaly Validation & Scoring

# COMMAND ----------

PIPELINE_STAGES = ["S2_object_detect", "S3_roi_extract", "S7_llm_analysis", "S8_validation"]

MODEL_CONFIGS = [
    {
        "name": "gemma4-e2b-q4km",
        "architecture": "Gemma 4 E2B MoE",
        "quantization": "Q4_K_M",
        "vram_mb": 7200,
        "context_length": 131072,
    },
    {
        "name": "navarasa-2.0-7b-q4km",
        "architecture": "Gemma 7B (Telugu-LLM-Labs fine-tune)",
        "quantization": "Q4_K_M",
        "vram_mb": 5600,
        "context_length": 4096,
    },
]

def simulate_stage_metrics(stage: str, batch_size: int) -> dict:
    """Generate realistic metrics for a pipeline stage."""
    base_latency = {
        "S2_object_detect": 25.0,
        "S3_roi_extract": 8.0,
        "S7_llm_analysis": 180.0,
        "S8_validation": 15.0,
    }
    
    latency = base_latency.get(stage, 50.0) + random.gauss(0, base_latency.get(stage, 50.0) * 0.15)
    confidence = min(0.99, max(0.3, random.gauss(0.82, 0.12)))
    vram_delta = random.uniform(50, 300) if "llm" in stage else random.uniform(10, 80)
    
    return {
        f"{stage}_latency_ms": max(1.0, latency),
        f"{stage}_confidence": confidence,
        f"{stage}_vram_delta_mb": vram_delta,
        f"{stage}_batch_size": batch_size,
        f"{stage}_throughput_fps": batch_size / (latency / 1000.0),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3️⃣ Run Multiple Inference Experiments

# COMMAND ----------

NUM_RUNS = 5
FRAMES_PER_RUN = 50

all_run_ids = []

for run_idx in range(NUM_RUNS):
    with mlflow.start_run(run_name=f"uv-pipeline-run-{run_idx + 1}") as run:
        run_id = run.info.run_id
        all_run_ids.append(run_id)
        
        # Log model configuration
        model = random.choice(MODEL_CONFIGS)
        mlflow.log_params({
            "model_name": model["name"],
            "model_architecture": model["architecture"],
            "quantization": model["quantization"],
            "vram_budget_mb": model["vram_mb"],
            "context_length": model["context_length"],
            "gpu_device": "RTX 4070 8GB",
            "cuda_version": "12.4",
            "pipeline_stages": ",".join(PIPELINE_STAGES),
            "batch_size": FRAMES_PER_RUN,
        })
        
        # Log stage metrics for each batch of frames
        total_latency = 0
        all_confidences = []
        
        for frame_batch in range(0, FRAMES_PER_RUN, 10):
            batch_size = min(10, FRAMES_PER_RUN - frame_batch)
            step = frame_batch // 10
            
            for stage in PIPELINE_STAGES:
                metrics = simulate_stage_metrics(stage, batch_size)
                mlflow.log_metrics(metrics, step=step)
                total_latency += metrics[f"{stage}_latency_ms"]
                all_confidences.append(metrics[f"{stage}_confidence"])
        
        # Log summary metrics
        avg_conf = sum(all_confidences) / len(all_confidences)
        mlflow.log_metrics({
            "total_pipeline_latency_ms": total_latency,
            "avg_confidence": avg_conf,
            "frames_processed": FRAMES_PER_RUN,
            "anomalies_detected": random.randint(3, 15),
            "gpu_utilization_pct": random.uniform(65, 95),
            "vram_peak_mb": model["vram_mb"] + random.uniform(200, 800),
        })
        
        # Tag the run
        mlflow.set_tags({
            "pipeline_version": "0.1.0",
            "environment": "databricks-demo",
            "model_family": model["architecture"].split()[0],
        })
        
        print(f"  ✅ Run {run_idx + 1}/{NUM_RUNS}: {model['name']} | "
              f"Latency: {total_latency:.0f}ms | Confidence: {avg_conf:.3f} | "
              f"Run ID: {run_id[:8]}...")

print(f"\n🎉 All {NUM_RUNS} runs logged to MLflow!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4️⃣ Query Experiment Results

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient(TRACKING_URI)

# Get all runs
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.avg_confidence DESC"],
)

print("📊 Experiment Summary — Sorted by Confidence")
print("=" * 90)
print(f"{'Run':<8} {'Model':<25} {'Latency (ms)':<15} {'Confidence':<12} {'Anomalies':<10} {'GPU %':<8}")
print("-" * 90)

for r in runs:
    name = r.data.params.get("model_name", "?")
    lat = r.data.metrics.get("total_pipeline_latency_ms", 0)
    conf = r.data.metrics.get("avg_confidence", 0)
    anom = r.data.metrics.get("anomalies_detected", 0)
    gpu = r.data.metrics.get("gpu_utilization_pct", 0)
    print(f"{r.info.run_id[:8]:<8} {name:<25} {lat:<15.1f} {conf:<12.4f} {int(anom):<10} {gpu:<8.1f}")

print("=" * 90)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5️⃣ Metric History — Latency Across Steps

# COMMAND ----------

# Get metric history for the first run
if all_run_ids:
    best_run_id = all_run_ids[0]
    history = client.get_metric_history(best_run_id, "S7_llm_analysis_latency_ms")
    
    print(f"📈 LLM Analysis Latency History (Run: {best_run_id[:8]}...)")
    print("-" * 40)
    for point in history:
        bar = "█" * int(point.value / 10)
        print(f"  Step {point.step}: {point.value:7.1f} ms {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6️⃣ Compare Runs by Model

# COMMAND ----------

print("\n📊 Performance by Model Architecture:")
print("=" * 70)

model_metrics = {}
for r in runs:
    model = r.data.params.get("model_name", "unknown")
    if model not in model_metrics:
        model_metrics[model] = {"latencies": [], "confidences": [], "count": 0}
    model_metrics[model]["latencies"].append(r.data.metrics.get("total_pipeline_latency_ms", 0))
    model_metrics[model]["confidences"].append(r.data.metrics.get("avg_confidence", 0))
    model_metrics[model]["count"] += 1

for model, data in model_metrics.items():
    avg_lat = sum(data["latencies"]) / len(data["latencies"])
    avg_conf = sum(data["confidences"]) / len(data["confidences"])
    print(f"\n  🧠 {model}")
    print(f"     Runs: {data['count']}")
    print(f"     Avg Latency:    {avg_lat:.1f} ms")
    print(f"     Avg Confidence: {avg_conf:.4f}")
    print(f"     Min Latency:    {min(data['latencies']):.1f} ms")
    print(f"     Max Confidence: {max(data['confidences']):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Summary
# MAGIC
# MAGIC | Feature | Status |
# MAGIC |---------|--------|
# MAGIC | Experiment creation | ✅ Auto-create or reuse |
# MAGIC | Run logging | ✅ 5 full pipeline runs |
# MAGIC | Stage metrics | ✅ 4 stages × 5 batches each |
# MAGIC | Model params | ✅ Architecture, quantization, VRAM |
# MAGIC | Run comparison | ✅ Sorted by confidence |
# MAGIC | Metric history | ✅ Per-step latency trace |
# MAGIC | Tag management | ✅ Pipeline version, environment |
