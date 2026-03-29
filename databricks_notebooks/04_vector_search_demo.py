# Databricks notebook source
# MAGIC %md
# MAGIC # 🔍 Uni Vision — FAISS Vector Search Demo
# MAGIC
# MAGIC **Semantic similarity search** using FAISS indexes and sentence-transformer embeddings.
# MAGIC
# MAGIC This notebook demonstrates:
# MAGIC - Generating dense embeddings from anomaly descriptions
# MAGIC - Building a FAISS IVF index for fast retrieval
# MAGIC - K-nearest-neighbour similarity search
# MAGIC - Fuzzy deduplication of similar events
# MAGIC - Cluster analysis of detection embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1️⃣ Install & Import Dependencies

# COMMAND ----------

# %pip install faiss-cpu>=1.7.4 sentence-transformers>=2.2.0 numpy
# dbutils.library.restartPython()

import numpy as np
import time

try:
    import faiss
    print(f"✅ FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'available'}")
except ImportError:
    print("⚠️  faiss-cpu not installed. Run the %pip install cell above first.")
    raise

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers available")
except ImportError:
    print("⚠️  sentence-transformers not installed. Run the %pip install cell above first.")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2️⃣ Load Embedding Model
# MAGIC
# MAGIC Using `all-MiniLM-L6-v2` — a lightweight 384-dimensional encoder
# MAGIC optimised for semantic similarity (same model as production config).

# COMMAND ----------

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

print(f"📥 Loading model: {MODEL_NAME} ...")
start = time.time()
model = SentenceTransformer(MODEL_NAME)
elapsed = time.time() - start
print(f"✅ Model loaded in {elapsed:.1f}s — output dim: {model.get_sentence_embedding_dimension()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3️⃣ Generate Synthetic Detection Descriptions
# MAGIC
# MAGIC Creating realistic anomaly event descriptions that mimic production output.

# COMMAND ----------

import random
from datetime import datetime, timedelta

CAMERAS = ["cam-north-01", "cam-south-02", "cam-east-03", "cam-west-04", "cam-gate-05"]
ZONES = ["zone-A", "zone-B", "zone-C", "zone-D"]

# Templates for rich anomaly descriptions
TEMPLATES = [
    "Person detected in restricted zone {zone} near {camera}. Subject wearing dark clothing, moving towards secure area. Confidence: {conf:.0%}.",
    "Unattended bag spotted in {zone} by {camera}. Object stationary for extended period. Medium-sized backpack on floor. Confidence: {conf:.0%}.",
    "Crowd formation detected in {zone}. {count} individuals gathering near entrance. Unusual for this time of day. Camera: {camera}. Confidence: {conf:.0%}.",
    "Vehicle traveling wrong direction in {zone}. Dark sedan moving against traffic flow. Captured by {camera}. Confidence: {conf:.0%}.",
    "Fire or smoke indicators in {zone}. Haze detected near ventilation system. Alert from {camera}. Confidence: {conf:.0%}.",
    "Unusual movement pattern in {zone}. Individual exhibiting erratic trajectory. Tracked across 3 frames by {camera}. Confidence: {conf:.0%}.",
    "Perimeter breach detected at {zone} boundary. Motion sensor and {camera} corroboration. Single individual scaling fence. Confidence: {conf:.0%}.",
    "Loitering alert in {zone}. Individual stationary for over 10 minutes near restricted door. Captured by {camera}. Confidence: {conf:.0%}.",
    "Tailgating attempt at {zone} access point. Two individuals entering on single badge swipe. {camera} alert. Confidence: {conf:.0%}.",
    "Abandoned vehicle in {zone} parking area. Vehicle stationary for 6+ hours in no-parking zone. {camera} monitoring. Confidence: {conf:.0%}.",
    "Running detected in {zone} corridor. Individual moving at high speed through secure hallway. {camera} tracking. Confidence: {conf:.0%}.",
    "Unauthorized access attempt at {zone} secure door. Badge rejection followed by physical entry attempt. {camera} recording. Confidence: {conf:.0%}.",
    "Suspicious package left near {zone} entrance by unknown individual who departed quickly. {camera} captured face. Confidence: {conf:.0%}.",
    "Group altercation in {zone}. Multiple individuals in physical confrontation. {camera} recording active. Confidence: {conf:.0%}.",
    "Individual photographing security infrastructure in {zone}. {camera} flagged reconnaissance behavior. Confidence: {conf:.0%}.",
]

# Generate 300 detection descriptions
detections = []
now = datetime.now()
for i in range(300):
    template = random.choice(TEMPLATES)
    desc = template.format(
        zone=random.choice(ZONES),
        camera=random.choice(CAMERAS),
        conf=random.uniform(0.6, 0.99),
        count=random.randint(5, 25),
    )
    detections.append({
        "id": f"det-{i:05d}",
        "description": desc,
        "camera_id": random.choice(CAMERAS),
        "timestamp": (now - timedelta(hours=random.randint(0, 168))).isoformat(),
    })

print(f"✅ Generated {len(detections)} detection descriptions")
print(f"\n📝 Sample descriptions:")
for d in detections[:3]:
    print(f"   [{d['id']}] {d['description'][:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4️⃣ Compute Embeddings

# COMMAND ----------

descriptions = [d["description"] for d in detections]

print(f"🧠 Encoding {len(descriptions)} descriptions with {MODEL_NAME}...")
start = time.time()
embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=64)
elapsed = time.time() - start

embeddings = np.array(embeddings, dtype=np.float32)
print(f"✅ Embeddings computed in {elapsed:.1f}s")
print(f"   Shape: {embeddings.shape}")
print(f"   Throughput: {len(descriptions)/elapsed:.0f} descriptions/sec")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5️⃣ Build FAISS Index
# MAGIC
# MAGIC Using IVF (Inverted File) index for scalable approximate nearest-neighbour search.

# COMMAND ----------

# Normalise embeddings for cosine similarity (inner product on L2-normalised vectors)
faiss.normalize_L2(embeddings)

# For 300 vectors, use a flat index. For production (>10K), use IVF.
n_vectors = embeddings.shape[0]
dim = embeddings.shape[1]

if n_vectors < 1000:
    # Flat index for small datasets — exact search
    index = faiss.IndexFlatIP(dim)
    index_type = "Flat (exact)"
else:
    # IVF index for larger datasets — approximate search
    n_clusters = int(np.sqrt(n_vectors))
    quantiser = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantiser, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index_type = f"IVF (nlist={n_clusters})"

index.add(embeddings)
print(f"✅ FAISS index built")
print(f"   Type:    {index_type}")
print(f"   Vectors: {index.ntotal}")
print(f"   Dim:     {dim}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6️⃣ Similarity Search
# MAGIC
# MAGIC Find the most similar past detections for a given query.

# COMMAND ----------

def search_similar(query: str, top_k: int = 5, threshold: float = 0.65):
    """Search for similar detections using cosine similarity."""
    query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and score >= threshold:
            results.append({
                "id": detections[idx]["id"],
                "score": float(score),
                "camera": detections[idx]["camera_id"],
                "description": detections[idx]["description"][:80] + "...",
            })
    return results

# --- Query 1: Person in restricted area ---
print("🔎 Query: 'person detected in restricted zone near entrance'\n")
results = search_similar("person detected in restricted zone near entrance", top_k=5)
for r in results:
    print(f"   [{r['id']}] score={r['score']:.4f}  cam={r['camera']}")
    print(f"            {r['description']}")

# COMMAND ----------

# --- Query 2: Fire detection ---
print("🔎 Query: 'smoke or fire detected near building'\n")
results = search_similar("smoke or fire detected near building", top_k=5)
for r in results:
    print(f"   [{r['id']}] score={r['score']:.4f}  cam={r['camera']}")
    print(f"            {r['description']}")

# COMMAND ----------

# --- Query 3: Vehicle anomaly ---
print("🔎 Query: 'car driving the wrong way in parking lot'\n")
results = search_similar("car driving the wrong way in parking lot", top_k=5)
for r in results:
    print(f"   [{r['id']}] score={r['score']:.4f}  cam={r['camera']}")
    print(f"            {r['description']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7️⃣ Fuzzy Deduplication
# MAGIC
# MAGIC Identify near-duplicate detections that describe the same event.

# COMMAND ----------

DEDUP_THRESHOLD = 0.90

print(f"🔄 Scanning for near-duplicates (similarity > {DEDUP_THRESHOLD})...\n")

# Search each vector against the index
top_k_dedup = 5
all_scores, all_indices = index.search(embeddings, top_k_dedup)

duplicate_pairs = set()
for i in range(len(detections)):
    for score, j in zip(all_scores[i], all_indices[i]):
        if j > i and score >= DEDUP_THRESHOLD:
            duplicate_pairs.add((i, int(j), float(score)))

print(f"Found {len(duplicate_pairs)} near-duplicate pairs\n")
print(f"{'Pair':<20} {'Score':<8} {'Description A':<45} {'Description B'}")
print("-" * 130)
for a, b, score in sorted(duplicate_pairs, key=lambda x: -x[2])[:10]:
    desc_a = detections[a]["description"][:42] + "..."
    desc_b = detections[b]["description"][:42] + "..."
    print(f"  {detections[a]['id']} ↔ {detections[b]['id']}  {score:.4f}   {desc_a}  {desc_b}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8️⃣ Embedding Cluster Analysis
# MAGIC
# MAGIC K-Means clustering of detection embeddings to discover natural groupings.

# COMMAND ----------

N_CLUSTERS = 8

print(f"📊 Running K-Means clustering with {N_CLUSTERS} clusters...\n")
kmeans = faiss.Kmeans(dim, N_CLUSTERS, niter=20, verbose=False)
kmeans.train(embeddings)

# Assign each detection to a cluster
_, cluster_ids = kmeans.index.search(embeddings, 1)
cluster_ids = cluster_ids.flatten()

# Analyse clusters
from collections import Counter
cluster_counts = Counter(cluster_ids.tolist())

print(f"{'Cluster':<10} {'Size':<8} {'Sample Description'}")
print("-" * 100)
for cluster_id in range(N_CLUSTERS):
    count = cluster_counts.get(cluster_id, 0)
    # Find a sample from this cluster
    sample_idx = next((i for i, c in enumerate(cluster_ids) if c == cluster_id), None)
    sample_desc = detections[sample_idx]["description"][:70] + "..." if sample_idx is not None else "N/A"
    print(f"  {cluster_id:<10} {count:<8} {sample_desc}")

print(f"\n✅ Cluster size distribution: min={min(cluster_counts.values())}, max={max(cluster_counts.values())}, mean={np.mean(list(cluster_counts.values())):.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9️⃣ Index Persistence
# MAGIC
# MAGIC Save and reload the FAISS index for production use.

# COMMAND ----------

import os
import tempfile

save_dir = "/dbfs/uni_vision/faiss" if os.path.exists("/dbfs") else tempfile.mkdtemp()
os.makedirs(save_dir, exist_ok=True)
index_path = os.path.join(save_dir, "detection_index.bin")

# Save
faiss.write_index(index, index_path)
file_size = os.path.getsize(index_path)
print(f"💾 Index saved to: {index_path}")
print(f"   File size: {file_size / 1024:.1f} KB")

# Reload
loaded_index = faiss.read_index(index_path)
print(f"✅ Index reloaded — {loaded_index.ntotal} vectors")

# Verify: search with reloaded index
query_vec = model.encode(["fire smoke detected"], normalize_embeddings=True).astype(np.float32)
scores, indices = loaded_index.search(query_vec, 3)
print(f"\n🔎 Verification search (reloaded index):")
for s, idx in zip(scores[0], indices[0]):
    if idx >= 0:
        print(f"   [{detections[idx]['id']}] score={s:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Summary
# MAGIC
# MAGIC | Feature | Status |
# MAGIC |---------|--------|
# MAGIC | Embedding model (all-MiniLM-L6-v2) | ✅ 384-dim dense vectors |
# MAGIC | FAISS index build | ✅ Flat/IVF adaptive |
# MAGIC | Similarity search | ✅ Top-K with threshold |
# MAGIC | Fuzzy deduplication | ✅ Near-duplicate detection |
# MAGIC | K-Means clustering | ✅ 8-cluster grouping |
# MAGIC | Index persistence | ✅ Save/load to DBFS |
