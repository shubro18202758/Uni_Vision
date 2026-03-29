# Uni Vision — Real-Time Visual Anomaly Detection with Databricks Analytics

Uni Vision is a GPU-accelerated, real-time computer vision pipeline that ingests video feeds, runs multi-stage anomaly detection using **YOLOv8** object detection and **Qwen 3.5 9B** vision-language reasoning (with **Navarasa 2.0 7B** for multilingual user interaction in 16 Indian languages + English), and stores every detection event in **Delta Lake** with **MLflow** experiment tracking, **PySpark** batch analytics, and **FAISS** vector search — orchestrated by an agentic AI manager. Both LLMs run locally via Ollama on a single 8 GB GPU.

---

## Architecture — How Databricks Components Connect

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      Uni Vision — System Architecture                      │
│                                                                            │
│  ┌──────────┐   ┌───────────┐   ┌───────────────────────────────┐          │
│  │  Video   │──▶│ Ingestion │──▶│   Inference Pipeline          │          │
│  │  Upload  │   │  Layer    │   │  (YOLOv8 + Qwen 3.5 9B +     │          │
│  └──────────┘   └───────────┘   │   Navarasa 2.0 7B via Ollama) │          │
│                                 └──────────┬────────────────────┘          │
│                                             ▼                              │
│                              ┌──────────────────────────┐                  │
│                              │    Manager Agent (LLM)    │                  │
│                              │ ContextAnalyzer →         │                  │
│                              │ ComponentResolver →       │                  │
│                              │ PipelineComposer →        │                  │
│                              │ LifecycleManager →        │                  │
│                              │ FeedbackLoop              │                  │
│                              └──────────┬───────────────┘                  │
│                                         │                                  │
│         ┌───────────────────────────────┼──────────────────────────┐        │
│         │        Databricks Integration Layer                     │        │
│         │                               │                         │        │
│         │  ┌────────────────┐  ┌────────┴───────┐  ┌───────────┐  │        │
│         │  │  Delta Lake    │  │   MLflow        │  │  PySpark  │  │        │
│         │  │  Store         │  │   Inference     │  │  Batch    │  │        │
│         │  │                │  │   Tracker       │  │  Analytics│  │        │
│         │  │ • ACID writes  │  │                 │  │           │  │        │
│         │  │ • Time-travel  │  │ • Stage latency │  │ • Hourly  │  │        │
│         │  │ • Schema       │  │ • Confidence    │  │   rollups │  │        │
│         │  │   enforcement  │  │ • VRAM deltas   │  │ • Cross-  │  │        │
│         │  │ • Partition    │  │ • Model params  │  │   camera  │  │        │
│         │  │   pruning      │  │ • Throughput    │  │   correl. │  │        │
│         │  │ • Audit log    │  │ • Batch flush   │  │ • Anomaly │  │        │
│         │  └───────┬────────┘  └────────┬───────┘  └─────┬─────┘  │        │
│         │          │                    │                 │        │        │
│         │          ▼                    ▼                 ▼        │        │
│         │  ┌────────────────────────────────────────────────────┐  │        │
│         │  │          FAISS Vector Search Engine                │  │        │
│         │  │  • Sentence-transformer embeddings                │  │        │
│         │  │  • Similarity search & fuzzy deduplication         │  │        │
│         │  │  • Agent knowledge RAG enhancement                │  │        │
│         │  │  • Cluster analysis & anomaly flagging             │  │        │
│         │  └────────────────────────────────────────────────────┘  │        │
│         └─────────────────────────────────────────────────────────┘        │
│                                         │                                  │
│                                         ▼                                  │
│  ┌───────────────┐   ┌──────────────────────┐   ┌──────────────────────┐   │
│  │ FastAPI REST   │   │ WebSocket Real-Time  │   │ React Dashboard      │   │
│  │ + /api/databricks│ │ Event Stream         │   │ + Databricks Insights│   │
│  └───────────────┘   └──────────────────────┘   └──────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

### Databricks Components

| Component | Module | Purpose |
|---|---|---|
| **Delta Lake Store** | `src/uni_vision/databricks/delta_store.py` | ACID-transactional event sink — every detection is written to a partitioned Delta table with schema enforcement, time-travel queries, and automated VACUUM |
| **MLflow Inference Tracker** | `src/uni_vision/databricks/mlflow_tracker.py` | Logs per-stage metrics (latency, confidence, VRAM delta) to MLflow experiments; batch-flushes for low overhead on the hot inference path |
| **PySpark Analytics Engine** | `src/uni_vision/databricks/spark_analytics.py` | Runs hourly rollups, frequency analysis, cross-camera correlation, confidence trends, and Z-score anomaly detection over the Delta tables |
| **FAISS Vector Search** | `src/uni_vision/databricks/vector_search.py` | Embeds detection text with `all-MiniLM-L6-v2`, indexes in FAISS for similarity search, fuzzy deduplication, and agent knowledge RAG |

All four are **optional add-ons** gated behind `databricks.enabled` in `config/default.yaml` and installed via `pip install '.[databricks]'`.

---

### LLM Models — Qwen 3.5 9B & Navarasa 2.0 7B

The entire inference brain runs on **two open-weight LLMs** served locally via [Ollama](https://ollama.com), both quantised to Q4_K_M to fit inside an **8 GB RTX 4070** VRAM budget:

| Model | Architecture | Role | VRAM |
|---|---|---|---|
| **Qwen 3.5 9B** (Q4_K_M) | Qwen 9B Vision | **Manager Agent brain** — performs all computer-vision reasoning: scene analysis, anomaly detection, OCR interpretation, chain-of-thought risk assessment, confidence scoring, and agentic pipeline decisions. Runs as three Ollama variants (`uni-vision-ocr`, `uni-vision-adjudicator`, and the base `qwen3.5:9b-q4_K_M`) that share the same underlying weights via hard-linking. | ~5.1 GB weights + 512 MB KV cache |
| **Navarasa 2.0 7B** (Q4_K_M) | Gemma 7B fine-tuned (Telugu-LLM-Labs) | **Multilingual conversational UI** — handles user-facing chat in **16 languages** (Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia, Urdu, Assamese, Konkani, Nepali, Sindhi, English). Translates user queries → English for Qwen, translates Qwen's English responses → user's language, and translates real-time WebSocket alerts into the user's preferred language. | ~5.3 GB weights + 300 MB KV cache |

Ollama swaps models on demand — both share the same GPU **sequentially** (never concurrently), so the system runs entirely on a single 8 GB GPU. No cloud API keys required.

The Modelfiles live in [`config/ollama/`](config/ollama/) and are built during setup:
```bash
ollama create uni-vision-ocr -f config/ollama/Modelfile.ocr
ollama create uni-vision-adjudicator -f config/ollama/Modelfile.adjudicator
ollama create uni-vision-navarasa -f config/ollama/Modelfile.navarasa
```

---

## How to Run

### Prerequisites

- **Python 3.10+** and [uv](https://docs.astral.sh/uv/) (or pip)
- **Node.js 18+** and npm
- **Ollama** running locally ([install](https://ollama.com/download))
- **NVIDIA GPU** with CUDA (RTX 4070 8 GB recommended)

### 1. Backend — Install & Start

```bash
# Clone the repo
git clone https://github.com/shubro18202758/Uni_Vision.git
cd Uni_Vision

# Install Python dependencies (core + dev)
pip install -e ".[dev]"

# (Optional) Install Databricks extras — enables Delta Lake, MLflow, Spark, FAISS
pip install -e ".[databricks]"

# Copy environment file
cp .env.example .env

# Initialise Ollama models (one-time)
# Windows:
.\scripts\init-ollama.ps1
# Linux/macOS:
# ./scripts/init-ollama.sh

# Start the FastAPI backend on port 8000
uvicorn uni_vision.api:create_app --factory --host 0.0.0.0 --port 8000
```

### 2. Frontend — Install & Start

```bash
cd univision
npm install
npm run dev
# Frontend opens at http://localhost:5176
```

### 3. (Optional) Enable Databricks Integration

Edit `config/default.yaml` and set `databricks.enabled: true`, then restart the backend.
This activates Delta Lake writes, MLflow tracking, Spark analytics, and FAISS vector search.

### 4. (Optional) Docker Compose — Full Stack

```bash
cp .env.example .env
docker compose up -d
```

This brings up: API (8000), Ollama (11434), PostgreSQL, MinIO, Redis, Prometheus (9090), Grafana (3000).

---

## Demo Steps

1. **Open the dashboard** — navigate to `http://localhost:5176` in your browser.

2. **Upload a video** — click **"Upload"** in the sidebar, select any video file (MP4/AVI), and click **"Start Processing"**.

3. **Watch real-time analysis** — switch to the **"Pipeline"** tab to see live stage-by-stage progress (frame decode → detection → LLM analysis → anomaly scoring). The WebSocket stream pushes events in real time.

4. **Browse detections** — open the **"Detections"** panel to see flagged anomalies with scene description, confidence score, risk level, and chain-of-thought reasoning from the Ollama LLM.

5. **Click any detection** — the detail modal shows the full breakdown: **Summary** (anomaly status, scene description), **Chain of Thought** (LLM reasoning steps), **Risk Assessment** (risk level + justification), and **Impact Analysis**.

6. **Databricks Insights** (if enabled) — open the **"Databricks"** tab in the right panel to see:
   - Delta Lake table stats (versions, row count, partitions)
   - MLflow experiment metrics (latency, confidence trends)
   - Spark analytics (hourly rollups, frequency, anomaly Z-scores)
   - FAISS vector search (similar observation lookup, cluster analysis)

---

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src/uni_vision --cov-report=term-missing

# Quick — stop on first failure
pytest -x
```

---

## Project Structure

```
Uni_Vision/
├── src/uni_vision/
│   ├── agent/              # Agentic AI manager (LLM-driven pipeline control)
│   ├── api/                # FastAPI REST + WebSocket + Databricks routes
│   ├── common/             # Shared config, logging, exceptions
│   ├── contracts/          # DTOs, validation schemas
│   ├── databricks/         # Delta Lake, MLflow, Spark, FAISS integrations
│   ├── detection/          # YOLO object detection
│   ├── ingestion/          # Video frame capture & sampling
│   ├── manager/            # Pipeline lifecycle management
│   ├── monitoring/         # Prometheus metrics
│   ├── ocr/                # Ollama LLM + EasyOCR engines
│   ├── orchestrator/       # Pipeline composition & event dispatch
│   ├── postprocessing/     # Deduplication, validation
│   ├── preprocessing/      # Deskew, enhance, crop
│   └── storage/            # S3/MinIO persistence
├── univision/              # React + TypeScript frontend (Vite)
├── config/                 # YAML config + Ollama Modelfiles + Grafana dashboards
├── tests/                  # Unit + integration tests
├── alembic/                # Database migrations
└── scripts/                # Ollama init, model download, smoke test
```

---

## License

MIT
