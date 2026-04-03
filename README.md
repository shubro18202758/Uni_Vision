<div align="center">

# 🔮 Uni Vision

### *Autonomous Visual Intelligence — Design Pipelines with Words, Not Code*

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React_18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![Gemma 4](https://img.shields.io/badge/Gemma_4_E2B-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/gemma)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)](https://databricks.com)
[![Blocks](https://img.shields.io/badge/38_Pipeline_Blocks-8B5CF6?style=for-the-badge&logo=stackblitz&logoColor=white)](#-the-block-universe--38-pipeline-blocks)
[![Agent Tools](https://img.shields.io/badge/39_Agent_Tools-F59E0B?style=for-the-badge&logo=robot&logoColor=white)](#-agentic-ai-manager--39-tools)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br>

> **Tell the AI what you want to detect. It designs the pipeline. It processes the video. It analyses the results.**
>
> A **GPU-accelerated, agentic computer vision platform** where you describe a surveillance task in plain English (or 15 other Indian languages 🇮🇳) and the **Gemma 4 E2B** vision-language model autonomously designs a pipeline DAG from **38 specialized blocks**, processes your video feed through **YOLOv8** + multimodal LLM reasoning, and delivers real-time anomaly detection with chain-of-thought risk assessment — all backed by **Delta Lake** + **MLflow** + **PySpark** analytics + **FAISS** vector search, running entirely on a **single 8 GB GPU**. Zero cloud API keys. Zero compromises.

<br>

<img width="3198" height="1797" alt="Screenshot 2026-04-03 034418" src="https://github.com/user-attachments/assets/635e815b-0086-4032-8e38-4316649cee18" />
<img src="https://media1.tenor.com/m/eJEW4Mjno4IAAAAC/babu-rao-apte-paresh-rawal.gif" width="300">

</div>

<img width="3199" height="1799" alt="Screenshot 2026-03-29 142217" src="https://github.com/user-attachments/assets/71a5c201-d654-4adc-aec8-6ff76d3a714d" />

---

## ✨ What's New — Recent Changes

<div align="center">

<img src="https://media1.tenor.com/m/fxwLnM3W1LgAAAAC/awesome-aamir.gif" width="200">

*"Jhakaas!"* — Amar, **Andaz Apna Apna** (1994) 🎩

</div>

| Change | Details |
|---|---|
| 🧠 **Gemma 4 E2B (MoE)** | Migrated entire stack to Google's **Gemma 4 E2B** — a Mixture-of-Experts model (5.1B total / 2.3B effective params, 128K context, natively multimodal). Replaces all prior models (Qwen 3.5 9B → Gemma 3 4B → Gemma 3N E4B → Gemma 4 E4B → **Gemma 4 E2B**). |
| 🎨 **Autonomous Workflow Designer** | Describe your surveillance task in natural language → the LLM autonomously designs a complete pipeline DAG with block selection, wiring, and layout. 6-phase process: Translate → Analyse → Select Blocks → Wire Connections → Validate → Return Graph. |
| 🧧 **38 Pipeline Blocks** | Expanded block registry across 8 categories: Input, Ingestion, Preprocessing, Detection, OCR, PostProcessing, Output, and Utility. Every block has an "instruction" field for user-specified LLM guidance. |
| 🤖 **39 Agent Tools** | Full agentic toolkit across 4 modules: graph & design, pipeline & system, knowledge & feedback, and control & self-healing. |
| 🎬 **Pipeline Vision Theater** | Real-time visual processing preview with per-stage streaming — watch your pipeline execute frame by frame with live stage-by-stage progress, latency metrics, and anomaly overlays. |
| 📊 **Technical Metrics Dashboard** | Per-detection performance breakdown: stage latencies, VRAM usage, model confidence, bottleneck identification, and throughput analysis. |
| ⚡ **Performance Optimizations** | Avg overhead: **27.3s → 8.2s** (70% reduction). Pipeline latency: **60.7s → 42.2s** (30% reduction). Compact LLM prompts: **16K → 3.6K chars** with JSON structured output. |
| 🔤 **JSON Structured Output** | Workflow design now uses Ollama's `format: "json"` mode + `num_ctx: 8192` for reliable structured generation. No more JSON parse failures. |
| 📈 **Stats & Analytics Dashboard** | Camera-specific analytics, anomaly trends, confidence distributions, and processing throughput visualizations. |

---

## 🏗️ Architecture — How Everything Connects

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    🔮 Uni Vision — System Architecture                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐      │
│  │  💬 Natural Language Input                                          │      │
│  │  "Detect people loitering in the parking lot after midnight"        │      │
│  └───────────────────────────┬─────────────────────────────────────────┘      │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐      │
│  │  🎨 Autonomous Workflow Designer (6-Phase)                          │      │
│  │  Translate → Analyse → Select Blocks → Wire → Validate → Graph     │      │
│  │  (Compact prompt: 3.6K chars · format:json · num_ctx: 8192)        │      │
│  └───────────────────────────┬─────────────────────────────────────────┘      │
│                              ▼                                               │
│  ┌──────────┐   ┌───────────┐   ┌───────────────────────────────┐            │
│  │ 🎬 Video │──▶│ Ingestion │──▶│   Inference Pipeline          │            │
│  │  Upload  │   │  Layer    │   │  (YOLOv8 + Gemma 4 E2B +     │            │
│  └──────────┘   └───────────┘   │   Navarasa 2.0 7B via Ollama) │            │
│                                 └──────────┬────────────────────┘            │
│                                             ▼                                │
│                  ┌──────────────────────────────────────┐                     │
│                  │   🤖 Manager Agent (39 Tools)        │                     │
│                  │  ContextAnalyzer → ComponentResolver  │                     │
│                  │  → PipelineComposer → LifecycleManager│                     │
│                  │  → FeedbackLoop → DependencyResolver  │                     │
│                  └──────────┬───────────────────────────┘                     │
│                             │                                                │
│    ┌────────────────────────┼──────────────────────────────┐                  │
│    │  🧧 38-Block Pipeline Engine (8 Categories)          │                  │
│    │  Input · Ingestion · Preprocessing · Detection        │                  │
│    │  OCR · PostProcessing · Output · Utility              │                  │
│    └────────────────────────┬──────────────────────────────┘                  │
│                             │                                                │
│    ┌────────────────────────┼──────────────────────────────┐                  │
│    │  🧱 Databricks Integration Layer                      │                  │
│    │                        │                              │                  │
│    │  ┌──────────────┐ ┌────┴──────┐ ┌──────────┐          │                  │
│    │  │ 🗄️ Delta Lake│ │ 📊 MLflow │ │ ⚡ Spark │          │                  │
│    │  │  ACID Store  │ │  Tracker  │ │ Analytics│          │                  │
│    │  └──────┬───────┘ └─────┬─────┘ └────┬─────┘          │                  │
│    │         ▼               ▼             ▼               │                  │
│    │  ┌────────────────────────────────────────────────┐   │                  │
│    │  │       🔍 FAISS Vector Search Engine            │   │                  │
│    │  │  Embeddings · Similarity · RAG · Clustering    │   │                  │
│    │  └────────────────────────────────────────────────┘   │                  │
│    └───────────────────────────────────────────────────────┘                  │
│                             │                                                │
│                             ▼                                                │
│  ┌──────────────┐ ┌──────────────────┐ ┌─────────────────────────────────┐   │
│  │ 🌐 FastAPI   │ │ ⚡ WebSocket     │ │ 💻 React Dashboard              │   │
│  │ REST + Routes│ │ Real-time Stream │ │ Canvas + Vision Theater +       │   │
│  │ + Databricks │ │ + Per-Stage Push │ │ Technical Metrics + Analytics   │   │
│  └──────────────┘ └──────────────────┘ └─────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

<img width="3198" height="1791" alt="Screenshot 2026-03-29 142420" src="https://github.com/user-attachments/assets/0734b9c7-8e3a-4860-b45d-3fe08641a7bd" />

---

## 🧱 The Block Universe — 38 Pipeline Blocks

<div align="center">

Every vision pipeline is assembled from **38 specialized blocks** across **8 categories**. Drag them onto the canvas, wire them together, or let the **Workflow Designer** do it for you.

</div>

| Category | Blocks | Description |
|:---:|---|---|
| 📥 **Input** (3) | `image-input` · `rtsp-stream` · `video-file` | Feed frames into the pipeline from cameras, files, or streams |
| 🎞️ **Ingestion** (2) | `frame-sampler` · `roi-crop` | Down-sample rates and focus on regions of interest |
| 🖼️ **Preprocessing** (5) | `grayscale` · `contrast-enhance` · `denoise` · `resize` · `plate-preprocessor` | Clean, enhance, and prepare frames for detection |
| 🎯 **Detection** (12) | `yolo-detector` · `motion-detector` · `fire-smoke-detector` · `crowd-density` · `vehicle-detector` · `plate-detector` · `scene-classifier` · `anomaly-scorer` · `pose-estimator` · `ppe-detector` · `zone-intrusion` · `llm-vision` | Core detection engines — YOLO, motion, fire/smoke, crowds, vehicles, plates, scene classification, anomaly scoring, pose estimation, PPE compliance, zone intrusion monitoring, and multimodal Gemma 4 vision analysis |
| 👁️ **OCR** (3) | `easy-ocr` · `paddleocr` · `text-reader` | EasyOCR, PaddleOCR, and general-purpose text extraction |
| 🔧 **PostProcessing** (7) | `object-tracker` · `optical-flow` · `regex-validator` · `deduplicator` · `threshold-gate` · `face-anonymizer` · `adjudicator` | Multi-object tracking (DeepSORT), optical flow, regex validation, perceptual-hash deduplication, confidence gating, face anonymization for privacy, and multi-engine OCR adjudication |
| 📤 **Output** (6) | `heatmap-generator` · `annotator` · `dispatcher` · `alert-trigger` · `console-logger` · `video-recorder` | Activity heatmaps, bounding-box annotations, event dispatch via WebSocket, webhook/email alerts, console logging, and video recording |

> 💡 Every block exposes an **"instruction" field** — write plain-English guidance for the LLM (e.g., *"Focus on people near the emergency exit after 10 PM"*), and the model customizes its behaviour per-block.

<div align="center">

<img src="https://media1.tenor.com/m/5oRaZZg9uZ8AAAAC/nana-patekar-seh-lenge-thodasa.gif" width="300">

*"Seh lenge thoda sa!"*
— Uday Shetty, **Welcome** (2007) 🤵

*^^ The pipeline engine juggling 38 blocks on one GPU*

</div>

---

## 💻 Frontend Architecture — React + TypeScript + Vite

The React 18 frontend is a full-featured IDE-like dashboard built with **TypeScript**, **Vite 6**, **Zustand** for state, **ReactFlow** for the node canvas, **Recharts** for charts, **Monaco Editor** for code views, and **Framer Motion** for animations.

### 📦 14 Component Modules

| Module | Purpose |
|:---:|---|
| 🎨 **canvas/** | Workbench — drag-and-drop block canvas (ReactFlow-powered) |
| 💬 **chat/** | AI Chat overlay + Navarasa multilingual conversation |
| 🎬 **pipeline/** | Pipeline Monitor + Pipeline Vision Theater (real-time stage streaming) |
| 🚨 **detections/** | Detection modals + Risk/Impact analysis + Technical Metrics Dashboard |
| 📊 **analytics/** | Stats dashboard + Databricks insights + anomaly trends |
| 🧱 **blocks/** | Block configuration panels and block type renderers |
| 🎨 **palette/** | Block palette sidebar — drag blocks onto canvas by category |
| 📷 **cameras/** | Camera management, stream preview, camera stats |
| 🔍 **inspector/** | Node/edge inspector — configure selected block properties |
| 💻 **code/** | Monaco-based code viewer for pipeline serialization |
| 🔗 **edges/** | Custom edge renderers for pipeline wiring |
| 💬 **feedback/** | User feedback widgets for detection corrections |
| 📊 **status/** | System status indicators and health monitors |
| 🏗️ **layout/** | StatusBar, Topbar, Sidebars, AppShell |

### 🗂️ 12 Zustand Stores

| Store | Manages |
|:---:|---|
| `graphStore` | Pipeline DAG state — nodes, edges, selection |
| `pipelineStore` | Pipeline execution lifecycle + results |
| `pipelineMonitorStore` | Real-time processing metrics and stage progress |
| `detectionStore` | Detection results, filtering, detail views |
| `chatStore` | AI chat messages, streaming state, Navarasa lang |
| `modelStore` | Ollama model state (primary_loaded, VRAM) |
| `cameraStore` | Camera configs, stream URLs, health |
| `agenticStore` | Manager Agent state, tool invocations |
| `codeStore` | Code view state (serialized pipeline JSON) |
| `historyStore` | Undo/redo history for canvas operations |
| `uiStore` | Sidebar visibility, panel layout, theme |
| `toastStore` | Notification toasts and alerts |

### 🔧 Utilities & Type System

| Layer | Files | Purpose |
|:---:|:---:|---|
| `lib/` | 8 modules | `autoLayout` · `blockRegistry` · `blockRegistryDefaults` · `cycleDetection` · `graphSerializer` · `graphValidator` · `mockCodeGenerator` · `pipelineIO` |
| `types/` | 7 files | `api` · `block` · `configSchema` · `connection` · `graph` · `port` · `dagre.d` |
| `constants/` | 4 files | `categories` · `keyboardShortcuts` · `portTypes` · `templates` |
| `hooks/` | 1 hook | `useKeyboardShortcuts` — global keyboard shortcut system |
| `services/` | 2 clients | `api.ts` (REST) · `websocket.ts` (WebSocket real-time stream) |

---

## 🤖 Agentic AI Manager — 39 Tools

The **Manager Agent** is an autonomous LLM-driven orchestrator that controls the entire pipeline lifecycle with **39 registered tools** across **4 modules**:

| Tool Group | Tools | Capabilities |
|:---:|:---:|---|
| 🎨 **Graph & Design** | 9 | `describe_pipeline_graph` · `list_available_blocks` · `get_block_details` · `get_deployed_graph` · `validate_pipeline_graph` · `deploy_pipeline_graph` · `clear_pipeline_graph` · `register_custom_block` · `design_workflow_from_nl` |
| ▶️ **Pipeline & System** | 13 | `query_detections` · `get_detection_summary` · `get_pipeline_stats` · `get_system_health` · `list_cameras` · `manage_camera` · `adjust_threshold` · `get_current_config` · `search_audit_log` · `analyze_detection_patterns` · `diagnose_camera` · `reset_circuit_breaker` · `run_analytics_query` |
| 🧠 **Knowledge & Feedback** | 11 | `get_knowledge_stats` · `get_frequent_detections` · `get_camera_error_profile` · `get_all_camera_profiles` · `get_cross_camera_detections` · `get_ocr_error_patterns` · `detect_detection_anomalies` · `record_detection_feedback` · `get_recent_feedback` · `get_camera_hints` · `save_knowledge` |
| ⚙️ **Control & Self-Healing** | 6 | `auto_tune_confidence` · `get_stage_analytics` · `get_ocr_strategy_stats` · `self_heal_pipeline` · `get_queue_pressure` · `flush_inference_queue` |

### Manager Agent Pipeline

```
  User Request ──▶ ContextAnalyzer ──▶ ComponentResolver ──▶ PipelineComposer
                                                                    │
                    FeedbackLoop ◀── LifecycleManager ◀── DependencyResolver ◀┘
```

---

## 🧱 Databricks Components

| Component | Module | Purpose |
|:---:|---|---|
| 🗄️ **Delta Lake Store** | `src/uni_vision/databricks/delta_store.py` | ACID-transactional event sink — every detection is written to a partitioned Delta table with schema enforcement, time-travel queries, and automated VACUUM |
| 📊 **MLflow Inference Tracker** | `src/uni_vision/databricks/mlflow_tracker.py` | Logs per-stage metrics (latency, confidence, VRAM delta) to MLflow experiments; batch-flushes for low overhead on the hot inference path |
| ⚡ **PySpark Analytics Engine** | `src/uni_vision/databricks/spark_analytics.py` | Runs hourly rollups, frequency analysis, cross-camera correlation, confidence trends, and Z-score anomaly detection over the Delta tables |
| 🔍 **FAISS Vector Search** | `src/uni_vision/databricks/vector_search.py` | Embeds detection text with `all-MiniLM-L6-v2`, indexes in FAISS for similarity search, fuzzy deduplication, and agent knowledge RAG |

> 💡 All four are **optional add-ons** gated behind `databricks.enabled` in `config/default.yaml` and installed via `pip install '.[databricks]'`.

---

## 🧠 LLM Models — Gemma 4 E2B & Navarasa 2.0 7B

The entire inference brain runs on **two open-weight LLMs** served locally via [Ollama](https://ollama.com), both quantised to Q4_K_M to fit inside an **8 GB RTX 4070** VRAM budget:

| Model | Architecture | Role | VRAM |
|:---:|:---:|---|:---:|
| 🧠 **Gemma 4 E2B** (Q4_K_M) | Mixture-of-Experts (Google) — 5.1B total / 2.3B effective params, 128K context window, natively multimodal (text + image + audio) | **The Brain** — powers the Workflow Designer (NL → pipeline DAG), scene analysis, anomaly detection, OCR interpretation, chain-of-thought risk assessment, confidence scoring, and all agentic pipeline decisions. Runs as three Ollama variants (`uni-vision-ocr`, `uni-vision-adjudicator`, `gemma4:e2b`) sharing the same weights via hard-linking. | ~7.2 GB |
| 🇮🇳 **Navarasa 2.0 7B** (Q4_K_M) | Gemma 7B fine-tuned (Telugu-LLM-Labs) | **Multilingual conversational UI** — handles user-facing chat in **16 languages** (Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia, Urdu, Assamese, Konkani, Nepali, Sindhi, English). Translates user queries → English for Gemma 4 E2B and back to the user's language. Translates real-time WebSocket alerts. | ~5.6 GB |

### Model Evolution

```
Qwen 3.5 9B → Gemma 3 4B → Gemma 3N E4B → Gemma 4 E4B → ✅ Gemma 4 E2B
```

> The journey from a 9B generic model to a 2.3B-effective MoE powerhouse — smaller, faster, smarter. Every migration consolidated the codebase further. The final E2B runs 128K context with native multimodality.

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
| 🦙 Ollama | 0.20+ ([download](https://ollama.com/download)) |
| 🟢 NVIDIA GPU | CUDA (RTX 4070 8 GB recommended) |
| 🧠 Gemma 4 E2B | `ollama pull gemma4:e2b` (~7.2 GB) |

<div align="center">

<img src="https://media1.tenor.com/m/12flE7vVhe4AAAAC/dhamaal-javed-jaffrey.gif" width="300">

*"Pata nahi aise dangerous situations mein mai automatically aage kaise aa jaata hoon!"*
— Manav, **Dhamaal** (2007) 🪂

*^^ You, diving into the prerequisites list*

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

<img src="https://media1.tenor.com/m/fxwLnM3W1LgAAAAC/awesome-aamir.gif" width="300">

*"Jhakaas!"*
— Amar, **Andaz Apna Apna** (1994) 🎩

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

2. � **Design a pipeline with words** — open the **AI Chat**, type something like *"detect people loitering in the parking lot"* or *"fire detection in warehouse"*, and watch the **Workflow Designer** autonomously build a pipeline DAG from the 38-block catalog. The 6-phase design streams progress in real time.

3. 🧱 **Or build manually** — drag blocks from the **Block Palette** (organized in 8 categories) onto the **Workbench Canvas**, wire ports together, and configure each block's instruction field.

4. 📤 **Upload a video** — click **"Upload"** in the sidebar, select any video file (MP4/AVI), and click **"Start Processing"**.

5. 🎬 **Pipeline Vision Theater** — switch to the **"Theater"** view to watch live stage-by-stage visual processing with frame previews, latency badges, and anomaly overlays streaming in real time.

6. 🚨 **Browse detections** — open the **"Detections"** panel to see flagged anomalies with scene descriptions, confidence scores, risk levels, and chain-of-thought reasoning from Gemma 4 E2B.

7. 🔎 **Click any detection** — the detail modal shows: **Summary** (anomaly status, scene description), **Chain of Thought** (LLM reasoning steps), **Risk Assessment** (risk level + justification), **Impact Analysis**, and **Technical Metrics** (per-stage latency, VRAM, bottleneck analysis).

8. 📊 **Stats & Analytics** — explore the **Analytics Dashboard** for camera-specific stats, anomaly trends, confidence distributions, and throughput metrics.

9. 📊 **Databricks Insights** (if enabled) — open the **"Databricks"** tab to see:
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
<img width="3199" height="1999" alt="Screenshot 2026-04-03 132521" src="https://github.com/user-attachments/assets/2a12c070-fea9-4861-b5eb-dc265e193200" />
<img width="3199" height="1999" alt="Screenshot 2026-04-03 132518" src="https://github.com/user-attachments/assets/7a4c65a3-dc97-4079-840f-72a992c6abaa" />


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
│   ├── 🤖 agent/           # 39 tools across 4 modules + workflow designer, model router,
│   │   ├── graph_tools.py       # 9 tools — design, graph ops, block catalog
│   │   ├── pipeline_tools.py    # 13 tools — detections, cameras, diagnostics, analytics
│   │   ├── knowledge_tools.py   # 11 tools — knowledge base, feedback, camera profiling
│   │   ├── control_tools.py     # 6 tools — self-healing, tuning, queue management
│   │   ├── workflow_designer.py # NL → Pipeline DAG (6-phase autonomous design)
│   │   ├── model_router.py      # VRAM-exclusive model activation
│   │   ├── coordinator.py       # Agent orchestration loop
│   │   ├── knowledge.py         # FAISS-backed knowledge store
│   │   ├── navarasa_client.py   # 16-language multilingual chat bridge
│   │   ├── audit.py             # Tool invocation audit logging
│   │   └── ...                  # intent, memory, monitor, prompts, sessions, sub_agents
│   ├── 🌐 api/             # FastAPI REST + WebSocket + pipeline graph + stats routes
│   ├── ⚙️ common/          # Shared config, logging, exceptions
│   ├── 🔌 components/      # Component registry + DI container
│   ├── 📋 contracts/       # DTOs, validation schemas
│   ├── 🧱 databricks/      # Delta Lake, MLflow, Spark, FAISS integrations
│   ├── 🎯 detection/       # YOLO object detection + vision analyzer
│   ├── 📥 ingestion/       # Video frame capture & sampling
│   ├── 🎛️ manager/         # Pipeline lifecycle — context analyzer, component resolver,
│   │                        #   pipeline composer, dependency resolver, job lifecycle
│   ├── 📊 monitoring/      # Prometheus metrics + VRAM budget tracking
│   ├── 👁️ ocr/             # Ollama LLM + EasyOCR engines
│   ├── 🔀 orchestrator/    # 38-block registry, pipeline composition, graph engine,
│   │                        #   container DI, event dispatch
│   ├── 🔧 postprocessing/  # Deduplication, validation, risk & impact analysis, adjudicator
│   ├── 🖼️ preprocessing/   # Deskew, enhance, crop
│   ├── 💾 storage/         # S3/MinIO persistence
│   └── 📈 visualizer/      # OCR output visualization
├── 💻 univision/            # React 18 + TypeScript + Vite 6 frontend
│   └── src/
│       ├── app/             # AppShell — main layout container
│       ├── components/      # 14 modules: canvas, chat, pipeline, detections, analytics,
│       │                    #   blocks, palette, cameras, inspector, code, edges,
│       │                    #   feedback, status, layout
│       ├── store/           # 12 Zustand stores (graph, pipeline, detection, chat, model,
│       │                    #   camera, agentic, code, history, pipelineMonitor, ui, toast)
│       ├── services/        # REST API client + WebSocket real-time stream
│       ├── lib/             # 8 utility modules (autoLayout, blockRegistry, cycleDetection,
│       │                    #   graphSerializer, graphValidator, mockCodeGenerator, pipelineIO)
│       ├── types/           # 7 type definitions (api, block, configSchema, connection, graph, port)
│       ├── constants/       # 4 files (categories, keyboardShortcuts, portTypes, templates)
│       └── hooks/           # useKeyboardShortcuts — global keyboard shortcuts
├── ⚙️ config/               # YAML configs + Ollama Modelfiles + Prometheus + Grafana
│   ├── default.yaml         # Master config (Ollama, cameras, pipeline, Databricks toggle)
│   ├── cameras.yaml         # Camera definitions + stream URLs
│   ├── models.yaml          # Model registry + VRAM budgets
│   ├── prometheus.yml       # Prometheus scrape config
│   ├── prometheus_alerts.yml # Alert rules (latency, VRAM, error rates)
│   ├── grafana/             # Dashboard JSON + provisioning (datasources, dashboards)
│   └── ollama/              # Modelfiles: ocr, adjudicator, navarasa
├── 🧪 tests/                # 376 tests — 320 unit (20 files) + 56 integration (5 files)
├── 🗃️ alembic/              # Database migrations (4 versions)
├── 📓 databricks_notebooks/ # 6 notebooks: Setup, Delta Lake, MLflow, Spark, Vector Search, Full Pipeline
├── 📜 scripts/              # 6 scripts: init-ollama (PS1/SH), download_models, smoke_test_agent,
│                            #   deploy_databricks, test_ws
├── 📄 docs/                 # 10 markdown docs (see below)
├── 🐳 docker-compose.yml   # 7 services: app, postgres, ollama, minio, redis, prometheus, grafana
├── 🐳 Dockerfile           # Multi-stage CUDA 12.4 build (non-root, health checks)
├── 📦 pyproject.toml       # 4 extras: [dev], [inference], [databricks], [visualizer]
└── 🔧 Makefile             # 14 targets: install, lint, test, docker-build/up/down, etc.
```

---

## 📚 Documentation

Beyond this README, the repository includes **10 detailed documentation files**:

| Document | Purpose |
|:---:|---|
| 📐 [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture deep-dive — modules, data flow, dependency graph |
| 🚀 [DEPLOYMENT.md](DEPLOYMENT.md) | Production deployment guide — Docker, cloud, Ollama GPU setup |
| 🔎 [SYSTEM_WALKTHROUGH.md](SYSTEM_WALKTHROUGH.md) | End-to-end system walkthrough — from video upload to detection output |
| 📘 [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) | Full technical reference — APIs, schemas, protocols, config options |
| 📋 [spec.md](spec.md) | System specification — requirements, constraints, design decisions |
| 📝 [prd.md](prd.md) | Product requirements document — features, user stories, acceptance criteria |
| 📄 [abstract.md](abstract.md) | Project abstract — concise overview for academic/conference submission |
| 📖 [literatureabstract.md](literatureabstract.md) | Literature review abstract — related work and theoretical foundations |
| 🎤 [PPT.MD](PPT.MD) | Presentation slide structure and talking points |
| 🧠 [PRESENTATION_KNOWLEDGE_BASE.md](PRESENTATION_KNOWLEDGE_BASE.md) | Deep knowledge base for presentation Q&A preparation |

---

## ⚡ Performance

| Metric | Before | After | Improvement |
|:---:|:---:|:---:|:---:|
| Pipeline overhead | 27.3s | 8.2s | **70% reduction** |
| End-to-end latency | 60.7s | 42.2s | **30% reduction** |
| Workflow design prompt | 16,000 chars | 3,600 chars | **78% smaller** |
| Block registry | 16 types | 38 types | **138% more blocks** |
| Agent tools | — | 39 tools (4 modules) | Full autonomy |
| Test suite | — | 376 tests (320 unit + 56 integration) | Full coverage |
| Gemma 4 E2B context | 4,096 tokens | 128K tokens | **31× larger** |

> 📐 Measured on RTX 4070 (8 GB VRAM), Ollama 0.20, gemma4:e2b Q4_K_M.

---

<div align="center">

### 🇮🇳 Made with ❤️ and chai ☕

[![forthebadge](https://img.shields.io/badge/Powered%20By-Jugaad-orange?style=for-the-badge)](https://github.com/shubro18202758/Uni_Vision)
[![forthebadge](https://img.shields.io/badge/Runs%20On-Single%20GPU-green?style=for-the-badge&logo=nvidia)](https://github.com/shubro18202758/Uni_Vision)
[![forthebadge](https://img.shields.io/badge/38_Blocks-on_Canvas-8B5CF6?style=for-the-badge&logo=stackblitz)](https://github.com/shubro18202758/Uni_Vision)
[![forthebadge](https://img.shields.io/badge/Speaks-16%20Languages-blue?style=for-the-badge&logo=googletranslate)](https://github.com/shubro18202758/Uni_Vision)
[![forthebadge](https://img.shields.io/badge/Gemma%204-E2B%20MoE-4285F4?style=for-the-badge&logo=google)](https://github.com/shubro18202758/Uni_Vision)

<br>

<img src="https://media1.tenor.com/m/PI_Pv8RRhT4AAAAC/golmaal-paresh-rawal.gif" width="300">

*"Bachpan mein dekha tha... lollypop khata tha..."*
— Babli, **Golmaal** (2006) 🍭

*^^ You, explaining this project in your viva*

<br>

**License: MIT** 📄

</div>
