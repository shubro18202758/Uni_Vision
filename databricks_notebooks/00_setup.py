# Databricks notebook source
# MAGIC %md
# MAGIC # 🔮 Uni Vision — Environment Setup
# MAGIC
# MAGIC **Installs all required Python packages for the Uni Vision Databricks analytics stack.**
# MAGIC
# MAGIC Run this notebook once before executing any other notebook in this workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Install Dependencies

# COMMAND ----------

# MAGIC %pip install deltalake>=0.14.0 pyarrow>=14.0.0 mlflow>=2.10.0 faiss-cpu>=1.7.4 sentence-transformers>=2.2.0 pydantic>=2.10.0 structlog>=24.4.0

# COMMAND ----------

# Restart Python to pick up newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Verify Installations

# COMMAND ----------

import importlib

packages = [
    ("deltalake", "Delta Lake"),
    ("pyarrow", "PyArrow"),
    ("mlflow", "MLflow"),
    ("faiss", "FAISS"),
    ("sentence_transformers", "Sentence Transformers"),
    ("pyspark", "PySpark (built-in)"),
]

print("=" * 55)
print("  🔮 Uni Vision — Dependency Check")
print("=" * 55)
for mod_name, display_name in packages:
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", "✓")
        print(f"  ✅ {display_name:<25s} {ver}")
    except ImportError:
        print(f"  ❌ {display_name:<25s} NOT FOUND")
print("=" * 55)
print("\n🎉 Setup complete! Proceed to the next notebook.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📁 Create Data Directories

# COMMAND ----------

import os

dirs = [
    "/dbfs/uni_vision/delta/detections",
    "/dbfs/uni_vision/delta/audit_log",
    "/dbfs/uni_vision/mlflow",
    "/dbfs/uni_vision/faiss",
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"  📂 {d}")

print("\n✅ All data directories created.")
