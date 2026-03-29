<div align="center">

# 🔮 Uni Vision

### *Real-Time Visual Anomaly Detection with Databricks Analytics*

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React_18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)](https://databricks.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br>

> **GPU-accelerated, real-time computer vision pipeline** that ingests video feeds, runs multi-stage anomaly detection using **YOLOv8** + **Qwen 3.5 9B** vision-language reasoning (with **Navarasa 2.0 7B** for multilingual chat in 16 Indian languages 🇮🇳), and stores every event in **Delta Lake** with **MLflow** tracking, **PySpark** analytics, and **FAISS** vector search — all orchestrated by an agentic AI manager on a **single 8 GB GPU**. No cloud API keys. No nonsense.

<br>

<img src="https://media1.tenor.com/m/eJEW4Mjno4IAAAAC/babu-rao-apte-paresh-rawal.gif" width="300">

*"25 din mein paisa double!"*
— Baburao, **Hera Pheri** (2000) 💰

*^^ Except here, it's "1 GPU mein sab kuch!"*

</div>

<img width="3199" height="1799" alt="Screenshot 2026-03-29 142217" src="https://github.com/user-attachments/assets/71a5c201-d654-4adc-aec8-6ff76d3a714d" />
---

## 🏗️ Architecture — How Databricks Components Connect

```
┌────────────────────────────────────────────────────────────────────────────┐
│                   🔮 Uni Vision — System Architecture                      │
│                                                                            │
│  ┌──────────┐   ┌───────────┐   ┌───────────────────────────────┐          │
│  │ 🎬 Video │──▶│ Ingestion │──▶│   Inference Pipeline          │          │
│  │  Upload  │   │  Layer    │   │  (YOLOv8 + Qwen 3.5 9B +     │          │
│  └──────────┘   └───────────┘   │   Navarasa 2.0 7B via Ollama) │          │
│                                 └──────────┬────────────────────┘          │
│                                             ▼                              │
│                              ┌──────────────────────────┐                  │
│                              │   🤖 Manager Agent (LLM)  │                  │
│                              │ ContextAnalyzer →         │                  │
│                              │ ComponentResolver →       │                  │
│                              │ PipelineComposer →        │                  │
│                              │ LifecycleManager →        │                  │
│                              │ FeedbackLoop              │                  │
│                              └──────────┬───────────────┘                  │
│                                         │                                  │
│         ┌───────────────────────────────┼──────────────────────────┐        │
│         │    🧱 Databricks Integration Layer                      │        │
│         │                               │                         │        │
│         │  ┌────────────────┐  ┌────────┴───────┐  ┌───────────┐  │        │
│         │  │ 🗄️ Delta Lake  │  │ 📊 MLflow      │  │ ⚡ PySpark│  │        │
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
│         │  │       🔍 FAISS Vector Search Engine               │  │        │
│         │  │  • Sentence-transformer embeddings                │  │        │
│         │  │  • Similarity search & fuzzy deduplication         │  │        │
│         │  │  • Agent knowledge RAG enhancement                │  │        │
│         │  │  • Cluster analysis & anomaly flagging             │  │        │
│         │  └────────────────────────────────────────────────────┘  │        │
│         └─────────────────────────────────────────────────────────┘        │
│                                         │                                  │
│                                         ▼                                  │
│  ┌───────────────┐   ┌──────────────────────┐   ┌──────────────────────┐   │
│  │ 🌐 FastAPI    │   │ ⚡ WebSocket         │   │ 💻 React Dashboard   │   │
│  │ + /api/databricks│ │ Event Stream         │   │ + Databricks Insights│   │
│  └───────────────┘   └──────────────────────┘   └──────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```
<img width="3198" height="1791" alt="Screenshot 2026-03-29 142420" src="https://github.com/user-attachments/assets/0734b9c7-8e3a-4860-b45d-3fe08641a7bd" />

---

## 🧱 Databricks Components

| Component | Module | Purpose |
|:---:|---|---|
| 🗄️ **Delta Lake Store** | `src/uni_vision/databricks/delta_store.py` | ACID-transactional event sink — every detection is written to a partitioned Delta table with schema enforcement, time-travel queries, and automated VACUUM |
| 📊 **MLflow Inference Tracker** | `src/uni_vision/databricks/mlflow_tracker.py` | Logs per-stage metrics (latency, confidence, VRAM delta) to MLflow experiments; batch-flushes for low overhead on the hot inference path |
| ⚡ **PySpark Analytics Engine** | `src/uni_vision/databricks/spark_analytics.py` | Runs hourly rollups, frequency analysis, cross-camera correlation, confidence trends, and Z-score anomaly detection over the Delta tables |
| 🔍 **FAISS Vector Search** | `src/uni_vision/databricks/vector_search.py` | Embeds detection text with `all-MiniLM-L6-v2`, indexes in FAISS for similarity search, fuzzy deduplication, and agent knowledge RAG |

> 💡 All four are **optional add-ons** gated behind `databricks.enabled` in `config/default.yaml` and installed via `pip install '.[databricks]'`.

<div align="center">

<img src="https://media1.tenor.com/m/5oRaZZg9uZ8AAAAC/nana-patekar-seh-lenge-thodasa.gif" width="300">

*"Seh lenge thoda sa!"*
— Uday Shetty, **Welcome** (2007) 🤵

*^^ The GPU handling Delta Lake + MLflow + Spark + FAISS simultaneously*

</div>

---

## 🧠 LLM Models — Qwen 3.5 9B & Navarasa 2.0 7B

The entire inference brain runs on **two open-weight LLMs** served locally via [Ollama](https://ollama.com), both quantised to Q4_K_M to fit inside an **8 GB RTX 4070** VRAM budget:

| Model | Architecture | Role | VRAM |
|:---:|:---:|---|:---:|
| 🧠 **Qwen 3.5 9B** (Q4_K_M) | Qwen 9B Vision | **Manager Agent brain** — performs all computer-vision reasoning: scene analysis, anomaly detection, OCR interpretation, chain-of-thought risk assessment, confidence scoring, and agentic pipeline decisions. Runs as three Ollama variants (`uni-vision-ocr`, `uni-vision-adjudicator`, and the base `qwen3.5:9b-q4_K_M`) that share the same underlying weights via hard-linking. | ~5.6 GB |
| 🇮🇳 **Navarasa 2.0 7B** (Q4_K_M) | Gemma 7B fine-tuned (Telugu-LLM-Labs) | **Multilingual conversational UI** — handles user-facing chat in **16 languages** (Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia, Urdu, Assamese, Konkani, Nepali, Sindhi, English). Translates user queries → English for Qwen, translates Qwen's English responses → user's language, and translates real-time WebSocket alerts into the user's preferred language. | ~5.6 GB |

<div align="center">

> 🔄 Ollama swaps models on demand — both share the same GPU **sequentially** (never concurrently), so the system runs entirely on a **single 8 GB GPU**. No cloud API keys required. 🚫☁️

<img src="https://media1.tenor.com/m/bYKbtJCbMF8AAAAC/3-idiots-aamir-khan.gif" width="300">

*"All izz well!"*
— Rancho, **3 Idiots** (2009) ✨

*^^ The GPU when both models share it politely*

</div>

<details>
<summary>📦 <b>Modelfile Build Commands</b> (click to expand)</summary>

```bash
# The Modelfiles live in config/ollama/
ollama create uni-vision-ocr -f config/ollama/Modelfile.ocr
ollama create uni-vision-adjudicator -f config/ollama/Modelfile.adjudicator
ollama create uni-vision-navarasa -f config/ollama/Modelfile.navarasa
```
</details>

---
<img width="3199" height="1793" alt="Screenshot 2026-03-29 142352" src="https://github.com/user-attachments/assets/03d34713-7410-4011-9af0-12be5297384d" />

## 🚀 How to Run

### ✅ Prerequisites

| Requirement | Version |
|:---:|:---:|
| 🐍 Python | 3.10+ |
| 📦 Node.js | 18+ |
| 🦙 Ollama | [latest](https://ollama.com/download) |
| 🟢 NVIDIA GPU | CUDA (RTX 4070 8 GB recommended) |

<div align="center">

<img src="https://media1.tenor.com/m/fxwLnM3W1LgAAAAC/awesome-aamir.gif" width="300">

*"Jhakaas!"*
— Amar, **Andaz Apna Apna** (1994) 🎩

*^^ You, after successfully installing all prerequisites*

</div>

### 1️⃣ Backend — Install & Start

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

### 2️⃣ Frontend — Install & Start

```bash
cd univision
npm install
npm run dev
# Frontend opens at http://localhost:5176
```

### 3️⃣ (Optional) Enable Databricks Integration

Edit `config/default.yaml` and set `databricks.enabled: true`, then restart the backend.
This activates Delta Lake writes, MLflow tracking, Spark analytics, and FAISS vector search.

### 4️⃣ (Optional) Docker Compose — Full Stack

```bash
cp .env.example .env
docker compose up -d
```

> 🐳 This brings up: API (8000), Ollama (11434), PostgreSQL, MinIO, Redis, Prometheus (9090), Grafana (3000).

<div align="center">

<img src="https://media1.tenor.com/m/12flE7vVhe4AAAAC/dhamaal-javed-jaffrey.gif" width="300">

*"Pata nahi aise dangerous situations mein mai automatically aage kaise aa jaata hoon!"*
— Manav, **Dhamaal** (2007) 🪂

*^^ Docker Compose spinning up 7 containers on your machine*

</div>

---
<img width="3199" height="1787" alt="Screenshot 2026-03-29 142356" src="https://github.com/user-attachments/assets/d5c0298e-bc75-4cb4-b1a8-2adfcc815bcd" />

## 🎬 Demo Steps

<div align="center">

<img src="https://media1.tenor.com/m/hb36bfMal5MAAAAC/munna-bhai-mbbs-bollywood.gif" width="300">

*"Picture abhi baaki hai mere dost!"*
— Munna Bhai 🎬

</div>

1. 🌐 **Open the dashboard** — navigate to `http://localhost:5176` in your browser.

2. 📤 **Upload a video** — click **"Upload"** in the sidebar, select any video file (MP4/AVI), and click **"Start Processing"**.

3. 👁️ **Watch real-time analysis** — switch to the **"Pipeline"** tab to see live stage-by-stage progress (frame decode → detection → LLM analysis → anomaly scoring). The WebSocket stream pushes events in real time.

4. 🚨 **Browse detections** — open the **"Detections"** panel to see flagged anomalies with scene description, confidence score, risk level, and chain-of-thought reasoning from the Ollama LLM.

5. 🔎 **Click any detection** — the detail modal shows the full breakdown: **Summary** (anomaly status, scene description), **Chain of Thought** (LLM reasoning steps), **Risk Assessment** (risk level + justification), and **Impact Analysis**.

6. 📊 **Databricks Insights** (if enabled) — open the **"Databricks"** tab in the right panel to see:
   - 🗄️ Delta Lake table stats (versions, row count, partitions)
   - 📈 MLflow experiment metrics (latency, confidence trends)
   - ⚡ Spark analytics (hourly rollups, frequency, anomaly Z-scores)
   - 🔍 FAISS vector search (similar observation lookup, cluster analysis)

---

## 🧪 Testing

<div align="center">

<img src="https://media1.tenor.com/m/0AVBKGsQ6YsAAAAC/phir-hera-pheri-akshay-kumar.gif" width="300">

*"Mogambo khush hua!"*
— when all tests pass ✅

</div>

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src/uni_vision --cov-report=term-missing

# Quick — stop on first failure
pytest -x
```
<img width="3199" height="1789" alt="Screenshot 2026-03-29 142406" src="https://github.com/user-attachments/assets/6102e9c0-548e-4505-bd3a-9d17a0340662" />
<img width="3196" height="1798" alt="Screenshot 2026-03-29 142402" src="https://github.com/user-attachments/assets/9dd83250-9e33-40df-a6a2-e2b2526de544" />

<div align="center">

<img src="https://media1.tenor.com/m/3Wol6b87J_4AAAAC/chup-chup-ke-rajpal-yadav-bandya.gif" width="300">

*"Hey bhagwan! Kya zulum hai!"*
— Bandya, **Chup Chup Ke** (2006) 😱

*^^ First time looking at the project structure below*

</div>

---

## 📂 Project Structure

```
Uni_Vision/
├── src/uni_vision/
│   ├── 🤖 agent/           # Agentic AI manager (LLM-driven pipeline control)
│   ├── 🌐 api/             # FastAPI REST + WebSocket + Databricks routes
│   ├── ⚙️ common/          # Shared config, logging, exceptions
│   ├── 📋 contracts/       # DTOs, validation schemas
│   ├── 🧱 databricks/      # Delta Lake, MLflow, Spark, FAISS integrations
│   ├── 🎯 detection/       # YOLO object detection
│   ├── 📥 ingestion/       # Video frame capture & sampling
│   ├── 🎛️ manager/         # Pipeline lifecycle management
│   ├── 📊 monitoring/      # Prometheus metrics
│   ├── 👁️ ocr/             # Ollama LLM + EasyOCR engines
│   ├── 🔀 orchestrator/    # Pipeline composition & event dispatch
│   ├── 🔧 postprocessing/  # Deduplication, validation
│   ├── 🖼️ preprocessing/   # Deskew, enhance, crop
│   └── 💾 storage/         # S3/MinIO persistence
├── 💻 univision/            # React + TypeScript frontend (Vite)
├── ⚙️ config/               # YAML config + Ollama Modelfiles + Grafana dashboards
├── 🧪 tests/                # Unit + integration tests
├── 🗃️ alembic/              # Database migrations
└── 📜 scripts/              # Ollama init, model download, smoke test
```

---

<div align="center">

### 🇮🇳 Made with ❤️ and chai ☕

[![forthebadge](https://img.shields.io/badge/Powered%20By-Jugaad-orange?style=for-the-badge)](https://github.com/shubro18202758/Uni_Vision)
[![forthebadge](https://img.shields.io/badge/Runs%20On-Single%20GPU-green?style=for-the-badge&logo=nvidia)](https://github.com/shubro18202758/Uni_Vision)
[![forthebadge](https://img.shields.io/badge/Speaks-16%20Languages-blue?style=for-the-badge&logo=googletranslate)](https://github.com/shubro18202758/Uni_Vision)

<br>

<img src="https://media1.tenor.com/m/PI_Pv8RRhT4AAAAC/golmaal-paresh-rawal.gif" width="300">

*"Bachpan mein dekha tha... lollypop khata tha..."*
— Babli, **Golmaal** (2006) 🍭

*^^ You, explaining this project in your viva*

<br>

**License: MIT** 📄

</div>
