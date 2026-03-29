# Uni_Vision — Comprehensive Technical & Backend Logic Documentation

> **System Codename:** Uni_Vision  
> **Domain:** Computer Vision · Deep Learning · Intelligent Transportation Systems  
> **Version:** 0.1.0  
> **Last Updated:** 2026-03-21  
> **Classification:** Internal Technical Reference  
> **Hardware Target:** NVIDIA RTX 4070 — 8192 MB VRAM  
> **Primary Runtime:** Python 3.12 / CUDA 12.4+ / Ollama 0.17.7  

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [System Architecture](#2-system-architecture)
3. [Architectural Philosophy & Design Principles](#3-architectural-philosophy--design-principles)
4. [Technology Stack](#4-technology-stack)
5. [Project Structure](#5-project-structure)
6. [Configuration System](#6-configuration-system)
7. [8-Stage Processing Pipeline (S0–S8)](#7-8-stage-processing-pipeline-s0s8)
   - [S0: Stream Ingestion](#s0-stream-ingestion)
   - [S1: Frame Sampling & Deduplication](#s1-frame-sampling--deduplication)
   - [S2: Vehicle Detection](#s2-vehicle-detection)
   - [S3: License Plate Localisation](#s3-license-plate-localisation)
   - [S4: Plate Cropping & GPU→CPU Transfer](#s4-plate-cropping--gpucpu-transfer)
   - [S5: Geometric Correction](#s5-geometric-correction-deskew)
   - [S6: Photometric Enhancement](#s6-photometric-enhancement)
   - [S7: LLM OCR + Agentic Reasoning](#s7-llm-ocr--agentic-reasoning)
   - [S8: Post-Processing & Dispatch](#s8-post-processing--dispatch)
8. [GPU & VRAM Memory Management](#8-gpu--vram-memory-management)
9. [OCR Strategy Pattern](#9-ocr-strategy-pattern)
10. [Cognitive Orchestrator (Post-Processing)](#10-cognitive-orchestrator-post-processing)
11. [Agentic AI System](#11-agentic-ai-system)
    - [ReAct Pattern & Agent Loop](#111-react-pattern--agent-loop)
    - [Tool Registry & Definitions](#112-tool-registry--definitions)
    - [Multi-Agent Routing](#113-multi-agent-routing)
    - [Working Memory & Session Management](#114-working-memory--session-management)
    - [Intent Classification](#115-intent-classification)
    - [Knowledge Base & Learning Loop](#116-knowledge-base--learning-loop)
    - [Autonomous Monitoring Agent](#117-autonomous-monitoring-agent)
    - [Audit Trail](#118-audit-trail)
    - [Pipeline Control Tools](#119-pipeline-control-tools)
12. [REST API Layer](#12-rest-api-layer)
13. [WebSocket Streaming](#13-websocket-streaming)
14. [Middleware Stack](#14-middleware-stack)
15. [Storage Layer](#15-storage-layer)
16. [Database Schema & Migrations](#16-database-schema--migrations)
17. [Monitoring & Observability](#17-monitoring--observability)
18. [Streamlit Visualizer](#18-streamlit-visualizer)
19. [Error Handling & Failure Taxonomy](#19-error-handling--failure-taxonomy)
20. [Testing Strategy](#20-testing-strategy)
21. [CI/CD Pipeline](#21-cicd-pipeline)
22. [Docker & Containerisation](#22-docker--containerisation)
23. [Security Architecture](#23-security-architecture)
24. [Performance Targets & Benchmarks](#24-performance-targets--benchmarks)
25. [Data Flow Diagrams](#25-data-flow-diagrams)
26. [Deployment Guide Summary](#26-deployment-guide-summary)
27. [Appendix A — Environment Variables Reference](#appendix-a--environment-variables-reference)
28. [Appendix B — Complete Tool Registry](#appendix-b--complete-tool-registry)
29. [Appendix C — Tensor Format Specifications](#appendix-c--tensor-format-specifications)

---

## 1. Executive Overview

**Uni_Vision** is a production-grade **Automated Number Plate Recognition (ANPR)** system engineered for toll gate infrastructure. It ingests live RTSP video streams from static CCTV and IP cameras, processes each frame through an 8-stage asynchronous computer vision pipeline, and produces structured detection records containing validated license plate text, confidence scores, vehicle classification, and timestamped imagery.

### What Makes Uni_Vision Different

1. **LLM-Powered OCR** — Instead of traditional OCR engines alone, the primary OCR engine is **Qwen 3.5 9B** (a 9-billion-parameter multimodal language model) served via Ollama. The LLM performs multimodal reasoning directly on plate images, achieving higher accuracy than conventional OCR especially under adverse conditions.

2. **Fully Agentic Control Plane** — A ReAct-pattern AI agent backed by the same Qwen 3.5 9B model provides natural-language pipeline management, autonomous monitoring, self-healing capabilities, and intelligent analytics — all through conversational queries.

3. **Single-GPU Optimised** — The entire system (two YOLOv8 detection models + one 9B-parameter LLM + inference pipeline) operates within the strict 8192 MB VRAM ceiling of an NVIDIA RTX 4070 through a carefully engineered sequential-exclusivity memory protocol.

4. **Protocol-Driven Modularity** — Every pipeline stage adheres to Python `Protocol` interface contracts. Detection models, OCR engines, and post-processing logic are independently replaceable without modifying adjacent stages.

### Key Performance Targets

| Metric | Target |
|--------|--------|
| End-to-end latency (frame → dispatch) | ≤ 3 seconds |
| Vehicle detection precision | ≥ 92% |
| Plate detection recall | ≥ 90% |
| OCR exact-match accuracy (clean images) | ≥ 88% |
| OCR exact-match accuracy (night/adverse) | ≥ 75% |
| Concurrent camera streams | ≥ 4 at 5 FPS each |
| System uptime | ≥ 99.5% |

---

## 2. System Architecture

### High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Uni_Vision ANPR System                               │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │ Camera Layer │───▶│  Ingestion  │───▶│  Detection   │───▶│ Preprocess  │  │
│  │ CCTV / RTSP │    │  S0-S1      │    │  S2-S3       │    │ S4-S6       │  │
│  │ IP Camera   │    │  Frame      │    │  YOLOv8n     │    │ Crop/Deskew │  │
│  │ Video File  │    │  pHash Dedup│    │  Vehicle +   │    │ CLAHE/Filter│  │
│  └─────────────┘    └─────────────┘    │  Plate Det.  │    └──────┬──────┘  │
│                                         └──────────────┘           │         │
│                                                                    ▼         │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │                        OCR Layer (S7)                            │        │
│  │  ┌─────────────────────┐    ┌──────────────────────────────┐    │        │
│  │  │  Primary: OllamaLLM │◀──│ OCRStrategy (Strategy Pattern)│    │        │
│  │  │  Qwen 3.5 9B Q4_K_M│    │  Circuit Breaker + Retry     │    │        │
│  │  │  via HTTP API       │    │  Context-Aware Re-prompting  │    │        │
│  │  └─────────────────────┘    └──────────────────────────────┘    │        │
│  │  ┌─────────────────────┐                                        │        │
│  │  │ Fallback: EasyOCR   │◀── (activates on circuit break/timeout)│        │
│  │  │  CPU-only, threaded │                                        │        │
│  │  └─────────────────────┘                                        │        │
│  └──────────────────────────────────────────────────────────────────┘        │
│                          │                                                   │
│                          ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │              Cognitive Orchestrator (S8)                          │        │
│  │  Layer 1: DeterministicValidator                                 │        │
│  │    → Position-aware char correction → Regex validation           │        │
│  │  Layer 2: AgenticAdjudicator (LLM-based, only on failures)      │        │
│  │    → Multi-engine result adjudication via Qwen 3.5               │        │
│  └──────────────────────────────────────────────────────────────────┘        │
│                          │                                                   │
│         ┌────────────────┼──────────────────┐                               │
│         ▼                ▼                  ▼                                │
│  ┌────────────┐   ┌────────────┐   ┌──────────────┐                        │
│  │ PostgreSQL │   │ MinIO (S3) │   │ Redis PubSub │                        │
│  │ Detection  │   │ Plate/Veh  │   │ → WebSocket  │                        │
│  │ Records    │   │ Images     │   │   Broadcast  │                        │
│  └────────────┘   └────────────┘   └──────────────┘                        │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │              Agentic Control Plane                                │        │
│  │  AgentCoordinator → MultiAgentRouter → ReAct Loop               │        │
│  │  30 Tools │ 3 Sub-Agents │ KnowledgeBase │ AuditTrail           │        │
│  │  AutonomousMonitor → Self-Healing │ WebSocket Streaming         │        │
│  └──────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │              Observability Layer                                  │        │
│  │  Prometheus (14+ metrics) → Grafana (pre-built dashboards)       │        │
│  │  VRAM Monitor (pynvml) → Health Checks → Structured Logs        │        │
│  └──────────────────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Architectural Style

The system is a **bounded-memory sequential pipeline** with:

- **Synchronous data flow** within a single detection event (frame → vehicle → plate → enhance → OCR → dispatch)
- **Asynchronous decoupling** between stream ingestion and the detection pipeline (producer-consumer via bounded queues)
- **Externalized inference** for the LLM (Ollama process, HTTP API boundary)
- **Internalized inference** for vision models (in-process ONNX/TensorRT, shared CUDA context)

This is **not** a microservices architecture. It is a **monolithic pipeline with interface-segregated internal boundaries**, deployed as a single containerized process with an externalized LLM sidecar.

---

## 3. Architectural Philosophy & Design Principles

Five non-negotiable axioms govern every architectural decision:

### P1 — Memory is the Primary Constraint, Not Compute

Every decision is subordinate to the 8192 MB VRAM ceiling. A pipeline that is fast but exceeds VRAM by a single megabyte is a pipeline that crashes. Designs that trade latency for memory predictability are always preferred.

### P2 — Zero-Copy Unless Proven Impossible

Tensor data must never be duplicated across memory domains without explicit justification. Every CPU↔GPU transfer is an admission of architectural failure or a deliberate, profiled offloading decision. GPU↔CPU transitions per detection event: **exactly 2** (frame upload, plate download).

### P3 — Sequential Exclusivity of GPU-Bound Inference

No two neural network forward passes may occupy VRAM simultaneously unless their combined resident footprint has been statically verified. The vision models and the LLM orchestrator are **time-sliced, not concurrent**. The GPU is a single-tenant resource during inference.

### P4 — The LLM is the Orchestrator, Not Middleware

No framework-level orchestration layer (LangChain, LlamaIndex, AutoGen) exists. The Qwen 3.5 9B model is the reasoning engine. The application layer is a thin, deterministic Python harness that issues prompts, parses structured responses, catches failures, and re-prompts with error context.

### P5 — Every Stage is a Replaceable Unit

Each pipeline stage adheres to a strict interface contract via Python `Protocol` classes, not inheritance hierarchies. The vehicle detector, plate localizer, enhancement chain, and OCR engine are independently substitutable without modifying adjacent stages.

---

## 4. Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Language** | Python | 3.12+ | Runtime and orchestration |
| **API Framework** | FastAPI + Uvicorn | ≥0.115 | REST API + WebSocket endpoints |
| **Visualizer** | Streamlit | ≥1.41 | 8-page pipeline debugging dashboard |
| **Vehicle Detection** | YOLOv8n (Ultralytics) | ≥8.3 | INT8 quantized, TensorRT primary, ONNX fallback |
| **Plate Detection** | YOLOv8n | Same | Single-class "plate" detector |
| **Primary OCR** | Qwen 3.5 9B Q4_K_M | via Ollama 0.17.7 | Multimodal LLM OCR via HTTP API |
| **Fallback OCR** | EasyOCR | ≥1.7 | CPU-only fallback with ThreadPoolExecutor |
| **Image Processing** | OpenCV | ≥4.10 | Frame decode, preprocessing, geometric transforms |
| **Array Computing** | NumPy | ≥1.26 | Tensor manipulation, pHash computation |
| **Database** | PostgreSQL 16 | via asyncpg | Async connection pool, detection records |
| **Object Storage** | MinIO | S3-compatible | Plate/vehicle image archival (aioboto3) |
| **Message Broker** | Redis 7 | Pub/Sub | WebSocket bridge, task notification |
| **Monitoring** | Prometheus + Grafana | Latest | 14+ custom metrics, pre-built dashboards |
| **GPU Monitoring** | pynvml | ≥12.0 | VRAM usage, temperature, PCIe bandwidth |
| **Migrations** | Alembic + SQLAlchemy | ≥1.14 / ≥2.0 | Async schema migrations (4 versions) |
| **Configuration** | Pydantic Settings + YAML | ≥2.10 | Type-safe 3-layer config hierarchy |
| **Logging** | structlog | ≥24.4 | JSON-formatted structured logging |
| **HTTP Client** | httpx | ≥0.28 | Async Ollama API communication |
| **Containerization** | Docker + NVIDIA Toolkit | ≥24.0 | Multi-stage build, GPU passthrough |
| **Build System** | hatchling | ≥1.27 | PEP 517 wheel building |
| **Testing** | pytest + pytest-asyncio | ≥8.3 | 288 tests (288 passed, 21 skipped) |
| **Linting** | ruff | ≥0.9 | Fast Python linter/formatter |
| **Type Checking** | mypy | ≥1.14 | Static type analysis |

### Key Python Dependencies

```toml
dependencies = [
    "fastapi>=0.115.0", "uvicorn[standard]>=0.34.0", "httpx>=0.28.0",
    "opencv-python-headless>=4.10.0", "numpy>=1.26.0,<3.0",
    "pyyaml>=6.0.2", "pydantic>=2.10.0", "pydantic-settings>=2.7.0",
    "structlog>=24.4.0", "prometheus-client>=0.21.0",
    "asyncpg>=0.30.0", "alembic>=1.14.0", "sqlalchemy>=2.0.0",
    "boto3>=1.35.0", "aioboto3>=13.0.0", "redis>=5.2.0", "pynvml>=12.0.0",
]

[project.optional-dependencies]
inference = ["torch>=2.5.0", "ultralytics>=8.3.0", "onnxruntime-gpu>=1.20.0", "easyocr>=1.7.0"]
visualizer = ["streamlit>=1.41.0"]
dev = ["pytest>=8.3.0", "pytest-asyncio>=0.25.0", "pytest-cov>=6.0.0", "ruff>=0.9.0", "mypy>=1.14.0"]
```

---

## 5. Project Structure

```
Uni_Vision/
├── pyproject.toml                    # Build config, dependencies, entry points
├── alembic.ini                       # Async Alembic config (postgresql+asyncpg)
├── docker-compose.yml                # 7 services + ollama-init sidecar
├── Dockerfile                        # 2-stage build (python:3.12-slim → nvidia/cuda:12.4.1)
├── Makefile                          # install, lint, test, build, up/down, pipeline
├── .env.example                      # All UV_ environment variables
├── README.md                         # User-facing documentation
├── DEPLOYMENT.md                     # Production deployment guide
├── prd.md                            # Product Requirements Document
├── spec.md                           # System Architecture Specification
├── abstract.md                       # Technical Abstract
├── literatureabstract.md             # Academic Literature Abstract
│
├── alembic/
│   ├── env.py                        # Async migration runner (asyncpg + SQLAlchemy)
│   ├── script.py.mako                # Migration template
│   └── versions/
│       ├── 001_initial_schema.py     # detection_events, camera_sources, ocr_audit_log
│       ├── 002_perf_indexes.py       # B-tree + composite indexes for query performance
│       ├── 003_agent_knowledge.py    # agent_knowledge table (JSONB store)
│       └── 004_agent_audit.py        # agent_audit_log table
│
├── config/
│   ├── default.yaml                  # Complete pipeline defaults (100+ settings)
│   ├── cameras.yaml                  # Camera RTSP source definitions
│   ├── models.yaml                   # YOLOv8 detector config (vehicle + plate)
│   ├── prometheus.yml                # Prometheus scrape configuration
│   ├── prometheus_alerts.yml         # Alert rules (latency, errors, VRAM, pool)
│   ├── grafana/
│   │   ├── dashboards/               # Pre-built Grafana dashboard JSONs
│   │   └── provisioning/             # Auto-provisioning config
│   └── ollama/
│       ├── Modelfile.ocr             # OCR model (temperature=0.10, XML output schema)
│       └── Modelfile.adjudicator     # Adjudicator model (temperature=0.15)
│
├── models/                           # ONNX/TRT model weight files
│
├── scripts/
│   ├── download_models.py            # YOLOv8n weight download + ONNX export (opset 17)
│   ├── init-ollama.sh                # Linux: Pull qwen3.5 + create custom models
│   ├── init-ollama.ps1               # Windows: Same as above
│   └── smoke_test_agent.py           # End-to-end agent validation (3-part test)
│
├── src/uni_vision/
│   ├── __init__.py
│   │
│   ├── common/                       # Shared utilities
│   │   ├── config.py                 # 15+ nested Pydantic BaseSettings models
│   │   ├── exceptions.py             # Full exception hierarchy (13 error types)
│   │   └── logging.py                # structlog configuration
│   │
│   ├── contracts/                    # Interface contracts & DTOs
│   │   ├── protocols.py              # typing.Protocol for all pipeline stages
│   │   ├── dtos.py                   # FrameData, BoundingBox, PlateImage, OCRResult,
│   │   │                             # DetectionRecord, CameraSource DTOs
│   │   ├── pipeline_event.py         # PipelineEvent for inter-stage communication
│   │   ├── ocr_protocols.py          # OCREngine Protocol
│   │   ├── postprocessing_protocols.py  # PostProcessor Protocol
│   │   ├── preprocessing_protocols.py   # PreprocessingStage Protocol
│   │   └── events.py                 # Event types for pub/sub
│   │
│   ├── ingestion/                    # S0-S1: Stream capture & sampling
│   │   ├── rtsp_source.py            # Threaded RTSP reader, reconnection logic
│   │   ├── sampler.py                # Temporal FPS gating, round-robin polling
│   │   └── phash.py                  # 64-bit DCT perceptual hash + dedup
│   │
│   ├── detection/                    # S2-S3: YOLO-based detection
│   │   ├── vehicle_detector.py       # YOLOv8n vehicle detector (4 classes)
│   │   ├── plate_detector.py         # YOLOv8n plate detector (single-class)
│   │   ├── engine.py                 # TensorRT/ONNX Runtime inference engine
│   │   └── gpu_memory.py             # GPUMemoryManager, 3-mode offload controller
│   │
│   ├── preprocessing/                # S4-S6: Image processing chain
│   │   ├── roi_extractor.py          # S4: Plate cropping with symmetric padding
│   │   ├── deskew.py                 # S5: Canny → Hough → affine rotation
│   │   ├── enhance.py                # S6: Resize/CLAHE/Gaussian/bilateral
│   │   ├── chain.py                  # PreprocessingChain (sequential stage runner)
│   │   └── straighten.py             # Perspective correction utilities
│   │
│   ├── ocr/                          # S7: Optical Character Recognition
│   │   ├── strategy.py               # OCRStrategy (Strategy pattern dispatcher)
│   │   ├── llm_ocr.py                # OllamaLLMOCR (primary, circuit breaker)
│   │   ├── easyocr_engine.py         # EasyOCR fallback (CPU, ThreadPoolExecutor)
│   │   ├── response_parser.py        # XML/JSON response parsing from LLM output
│   │   ├── circuit_breaker.py        # CircuitBreaker (CLOSED→OPEN→HALF_OPEN)
│   │   └── prompts.py                # System/user prompt templates for OCR
│   │
│   ├── postprocessing/               # S8: Validation & dispatch
│   │   ├── orchestrator.py           # CognitiveOrchestrator (Layer 1 + Layer 2)
│   │   ├── validator.py              # DeterministicValidator (char correction + regex)
│   │   ├── adjudicator.py            # AgenticAdjudicator (LLM-based, on failures only)
│   │   ├── deduplicator.py           # Sliding-window duplicate suppression (10s)
│   │   ├── dispatcher.py             # AsyncDispatcher (DB + S3 + Redis publish)
│   │   └── char_map.py               # OCR confusion pair substitution maps
│   │
│   ├── orchestrator/                 # Pipeline controller & DI
│   │   ├── pipeline.py               # PipelineController (async consumer, GPU exclusivity)
│   │   ├── container.py              # DI Container (wires all 11 components)
│   │   └── bootstrap.py              # Application bootstrap & entry point
│   │
│   ├── storage/                      # Persistence layer
│   │   ├── database.py               # PostgreSQL async pool (asyncpg)
│   │   ├── object_store.py           # MinIO S3 archiver (aioboto3)
│   │   ├── queries.py                # SQL query templates
│   │   ├── retention.py              # RetentionTask (background data purge)
│   │   └── models.py                 # SQLAlchemy-compatible model definitions
│   │
│   ├── monitoring/                   # Observability
│   │   ├── metrics.py                # 14+ Prometheus metrics definitions
│   │   ├── vram_monitor.py           # VRAM polling (pynvml, 500ms interval)
│   │   ├── health.py                 # HealthService (GPU/Ollama/DB/stream checks)
│   │   ├── profiler.py               # @profile_stage decorator, ring buffer
│   │   └── vram_budget.py            # VRAMBudgetReport, validate_budget()
│   │
│   ├── agent/                        # Agentic AI control plane
│   │   ├── coordinator.py            # AgentCoordinator (top-level orchestrator)
│   │   ├── loop.py                   # AgentLoop (ReAct execution engine)
│   │   ├── tools.py                  # ToolRegistry, @tool decorator, auto-schema
│   │   ├── llm_client.py             # AgentLLMClient (httpx → Ollama)
│   │   ├── sub_agents.py             # MultiAgentRouter, AgentRole, SubAgentProfile
│   │   ├── memory.py                 # WorkingMemory (bounded FIFO, token budget)
│   │   ├── sessions.py               # SessionManager (TTL=1800s, LRU eviction)
│   │   ├── knowledge.py              # KnowledgeBase (PlateObservation, feedback)
│   │   ├── intent.py                 # IntentClassifier (regex, 8 categories)
│   │   ├── monitor.py                # AutonomousMonitor (bg health surveillance)
│   │   ├── audit.py                  # AuditTrail (write-behind → PostgreSQL)
│   │   ├── prompts.py                # Agent system/user prompt templates
│   │   ├── pipeline_tools.py         # 10 pipeline management tools
│   │   ├── control_tools.py          # 6 pipeline control tools
│   │   ├── knowledge_tools.py        # 7 knowledge & analytics tools
│   │   └── context.py                # ToolExecutionContext injection
│   │
│   ├── api/                          # HTTP/WS API layer
│   │   ├── __init__.py               # create_app() factory, lifespan, middleware wiring
│   │   ├── routes/
│   │   │   ├── health.py             # GET /health (deep + lightweight)
│   │   │   ├── detections.py         # GET /detections (paginated, filtered)
│   │   │   ├── sources.py            # POST/GET/DELETE /sources
│   │   │   ├── metrics.py            # GET /metrics (Prometheus text format)
│   │   │   ├── stats.py              # GET /stats (pipeline telemetry)
│   │   │   ├── agent_chat.py         # POST /api/agent/chat, GET /status, etc.
│   │   │   ├── ws_events.py          # WS /ws/events (Redis → WebSocket bridge)
│   │   │   └── ws_agent.py           # WS /ws/agent (streaming agent reasoning)
│   │   └── middleware/
│   │       ├── auth.py               # API key validation (SHA-256, timing-safe)
│   │       ├── rate_limit.py         # Sliding-window per-IP rate limiting
│   │       └── security_headers.py   # X-Content-Type-Options, X-Frame-Options, etc.
│   │
│   └── visualizer/                   # Streamlit debugging UI
│       ├── app.py                    # Main Streamlit app (8-page router)
│       └── pages/                    # Individual page modules (S1-S8 visualization)
│
└── tests/
    ├── conftest.py                   # Heavyweight dependency stubbing
    ├── unit/                         # 16 test files
    │   ├── test_agent.py             # 54 tests — ToolRegistry, Memory, Intent, etc.
    │   ├── test_api.py               # API endpoint tests
    │   ├── test_config.py            # Configuration loading tests
    │   ├── test_container.py         # DI container wiring tests
    │   ├── test_dtos.py              # DTO construction & validation
    │   ├── test_exceptions.py        # Exception hierarchy tests
    │   ├── test_deduplicator.py      # Sliding-window dedup logic
    │   ├── test_models.py            # Detection model interface tests
    │   ├── test_phash.py             # Perceptual hash computation & comparison
    │   ├── test_profiler.py          # Stage profiling tests
    │   ├── test_response_parser.py   # LLM output parsing tests
    │   ├── test_security.py          # Auth middleware, rate limiting
    │   ├── test_validator.py         # Character correction, regex validation
    │   ├── test_vram_budget.py       # VRAM budget enforcement tests
    │   ├── test_websocket.py         # WebSocket lifecycle tests
    │   └── test_visualizer.py        # Streamlit page rendering tests
    └── integration/                  # 5 integration test files
        ├── test_pipeline_smoke.py    # Full S2→S8 chain (mocked hardware)
        ├── test_api_lifecycle.py      # FastAPI TestClient with mocked DB
        ├── test_config_loading.py    # YAML + env var precedence
        ├── test_dispatcher_flow.py   # DB→S3→Redis dispatch chain
        └── test_websocket_broadcast.py # WS event propagation
```

---

## 6. Configuration System

Uni_Vision employs a **3-layer configuration hierarchy** (highest priority first):

1. **Environment variables** — Prefixed with `UV_` (e.g., `UV_OLLAMA_BASE_URL`)
2. **YAML overrides** — `config/default.yaml`, `config/cameras.yaml`, `config/models.yaml`
3. **Code defaults** — Pydantic `BaseSettings` model defaults in `config.py`

### Configuration Models (Pydantic BaseSettings)

```python
# 15+ nested configuration models, each with env_prefix

class HardwareConfig:
    device: str = "cuda"                    # UV_DEVICE
    cuda_device_index: int = 0              # UV_CUDA_DEVICE_INDEX

class VRAMBudgets:
    region_a_llm_mb: int = 5120            # LLM weights
    region_b_kv_cache_mb: int = 512        # KV cache
    region_c_vision_mb: int = 1024         # Vision models
    region_d_system_mb: int = 512          # System overhead
    ceiling_mb: int = 8192                 # Hard ceiling

class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:9b-q4_K_M"
    timeout_s: int = 5
    num_ctx: int = 4096
    temperature: float = 0.1
    num_predict: int = 256
    num_gpu: int = -1                      # -1 = all layers on GPU
    seed: int = 42

class CircuitBreakerConfig:
    failure_threshold: int = 3
    recovery_timeout_s: int = 30
    half_open_max_calls: int = 1

class DatabaseConfig:                      # UV_POSTGRES_ prefix
    dsn: str = "postgresql://..."
    pool_min: int = 2
    pool_max: int = 8

class StorageConfig:                       # UV_S3_ prefix
    endpoint: str, bucket: str, access_key: str, secret_key: str

class RedisConfig:                         # UV_REDIS_ prefix
    url: str = "redis://localhost:6379/0"

class APIConfig:                           # UV_API_ prefix
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: List[str] = []               # Empty = auth disabled
    rate_limit_rpm: int = 120

class AgentConfig:                         # UV_AGENT_ prefix
    max_iterations: int = 10
    memory_token_budget: int = 4000
    session_ttl_s: int = 1800
    max_sessions: int = 100
    monitor_interval_s: int = 60

class AppConfig:                           # Root config, nests all above
    hardware, vram, pipeline, ollama, circuit_breaker,
    database, storage, redis, api, agent, ...
```

### YAML Configuration Files

| File | Purpose | Key Settings |
|------|---------|--------------|
| `config/default.yaml` | Master defaults | VRAM regions, plate regex, char substitution maps, stage toggles |
| `config/cameras.yaml` | Camera sources | camera_id, source_url, fps_target, location_tag |
| `config/models.yaml` | Detection models | Model paths, input size (640×640), confidence thresholds, class lists |
| `config/ollama/Modelfile.ocr` | OCR model | System prompt, temperature=0.10, XML output schema, stop tokens |
| `config/ollama/Modelfile.adjudicator` | Adjudicator model | System prompt, temperature=0.15, adjudication output schema |

---

## 7. 8-Stage Processing Pipeline (S0–S8)

The pipeline processes frames sequentially through 8 stages, ensuring GPU single-tenancy (Principle P3). A single async consumer loop in `PipelineController` guarantees that no two GPU-bound inference operations overlap.

### Pipeline Topology

```
            CPU Domain                   GPU Domain (CUDA)              GPU Domain (Ollama)
            ──────────                   ────────────────               ──────────────────
  S0,S1 ──────┐
   (ingest,   │
    sample,   │
    dedup)    │
              │── UPLOAD (pinned mem) ──▶ S2 (YOLO vehicle detection)
              │                          │
              │                          │── in-GPU crop ──▶ S3 (LPD plate detection)
              │                          │
              │◀── DOWNLOAD (plate) ─────│── S4 (crop + transfer)
              │                          │   FREE Region C tensors
  S5 (deskew) │
  S6 (enhance)│
              │── HTTP POST (base64) ──────────────────────▶ S7 (Qwen 3.5 OCR)
              │                                              │
              │◀── HTTP RESPONSE ───────────────────────────│
  S8 (post-   │
   process,   │
   dispatch)  │
```

**GPU↔CPU transitions per detection event: exactly 2.**

---

### S0: Stream Ingestion

| Property | Specification |
|----------|---------------|
| **Module** | `ingestion/rtsp_source.py` |
| **Input** | RTSP URL / IP camera address / local video file |
| **Output** | Raw BGR frame as `numpy.ndarray` (uint8, shape: 1080×1920×3) |
| **Library** | OpenCV `cv2.VideoCapture` |
| **Concurrency** | One thread per camera source (I/O-bound, GIL-safe) |
| **Reconnection** | Exponential backoff: 1s → 2s → 4s → 8s → 16s (max). Alert after 3 failures |
| **Memory** | Ring buffer queue (`queue.Queue(maxsize=50)`) per stream |
| **Configuration** | Per-source: `camera_id`, `source_url`, `location_tag`, `fps_target`, `enabled` |
| **Sizing** | ~6.2 MB per frame at 1080p BGR uint8 |

**Backend Logic:**
1. Each camera source spawns a dedicated reader thread
2. Thread continuously calls `cv2.VideoCapture.read()` in a loop
3. On frame decode failure, triggers exponential backoff reconnection
4. Each successfully decoded frame is placed in a per-source bounded queue
5. Queue overflow → frames dropped with FRAMES_DROPPED counter increment
6. Frame carries metadata: `camera_id`, `timestamp_utc`, `frame_index`

---

### S1: Frame Sampling & Deduplication

| Property | Specification |
|----------|---------------|
| **Module** | `ingestion/sampler.py` + `ingestion/phash.py` |
| **FPS Gating** | Accept 1 frame per `1/fps_target` second interval |
| **Deduplication** | 64-bit DCT perceptual hash (pHash) on downscaled 32×32 grayscale |
| **Hash Distance** | Hamming distance ≤ 5 → discard as duplicate |
| **Polling** | Weighted round-robin across all camera source queues |
| **Target** | ≥ 20% reduction in downstream processing under low-traffic |

**pHash Algorithm:**
```
1. Convert frame BGR → grayscale
2. Resize to 32×32 pixels (cv2.INTER_AREA)
3. Compute 2D DCT (Discrete Cosine Transform)
4. Extract top-left 8×8 low-frequency coefficients
5. Compute median of 64 coefficients
6. Generate 64-bit hash: bit[i] = 1 if coeff[i] > median, else 0
7. Compare with previous frame hash via Hamming distance
8. Distance ≤ 5 → duplicate → discard
```

---

### S2: Vehicle Detection

| Property | Specification |
|----------|---------------|
| **Module** | `detection/vehicle_detector.py` |
| **Model** | YOLOv8n, INT8 quantized |
| **Export Format** | TensorRT engine (`.engine`) primary, ONNX Runtime (`.onnx`) fallback |
| **Classes** | car (0), truck (1), bus (2), motorcycle (3) |
| **Input Resolution** | 640×640 (letterbox-padded, aspect-preserved) |
| **Confidence Threshold** | ≥ 0.60 (configurable) |
| **NMS IoU Threshold** | 0.45 |
| **Output** | `List[VehicleBBox]`: each `(x1, y1, x2, y2, confidence, class_id)` |
| **Latency Target** | ≤ 80ms per frame (GPU, INT8) |
| **VRAM Footprint** | ~200–400 MB (weights + I/O tensors) — Region C |
| **Failure** | No vehicle detected → frame discarded, logged to audit trail |

**Inference Pipeline:**
```
Raw frame (1080×1920×3 uint8)
    → Letterbox resize to 640×640 (aspect-preserved, gray padding)
    → Normalize: float32, /255.0
    → Transpose: HWC → CHW → NCHW (batch=1)
    → TensorRT/ONNX forward pass
    → Decode: grid predictions → absolute bbox coordinates
    → NMS: suppress overlapping detections (IoU > 0.45)
    → Filter: confidence < 0.60 removed
    → Output: List[VehicleBBox]
```

---

### S3: License Plate Localisation

| Property | Specification |
|----------|---------------|
| **Module** | `detection/plate_detector.py` |
| **Model** | YOLO-variant LPD, INT8 quantized |
| **Input** | Cropped vehicle ROI tensor (GPU-resident, from S2 bbox) |
| **Confidence Threshold** | ≥ 0.65 (configurable) |
| **Multi-plate Policy** | Select highest confidence detection |
| **Output** | `PlateBBox`: `(x1, y1, x2, y2, confidence)` |
| **Recall Target** | ≥ 90% |
| **False Positive Target** | ≤ 5% |
| **VRAM** | Shared Region C with S2 (sequential execution) |

**Backend Logic:**
1. Receive vehicle ROI coordinates from S2
2. Crop the vehicle region from the full frame (GPU tensor slice — zero-copy)
3. Run plate detection model on the cropped ROI
4. If multiple plates detected, select highest confidence
5. Map plate coordinates back to full-frame coordinate space
6. No plate detected → log to audit trail, skip remaining stages

---

### S4: Plate Cropping & GPU→CPU Transfer

| Property | Specification |
|----------|---------------|
| **Module** | `preprocessing/roi_extractor.py` |
| **Operation** | GPU tensor slice → `cudaMemcpyDeviceToHost` → numpy array |
| **Padding** | Configurable margin (default: 5px) clamped to frame boundaries |
| **Output** | `PlateImage` as `numpy.ndarray` (uint8, BGR, ~200×600 px) on CPU |
| **Post-Transfer** | All GPU Region C tensors freed |
| **Latency** | < 1ms (small tensor: ~0.36 MB) |

---

### S5: Geometric Correction (Deskew)

| Property | Specification |
|----------|---------------|
| **Module** | `preprocessing/deskew.py` |
| **Algorithm** | Canny edge detection → Probabilistic Hough Line Transform → Median angle → Affine rotation |
| **Max Correctable Skew** | ±30° |
| **Skip Threshold** | |skew| ≤ 3° → no transformation applied |
| **Library** | OpenCV (`cv2.getRotationMatrix2D`, `cv2.warpAffine`) |

**Deskew Algorithm:**
```
1. Convert plate image → grayscale
2. Apply Canny edge detection (threshold: 50/150)
3. Run Probabilistic Hough Line Transform
4. Compute angles of all detected lines
5. Take median angle as the skew estimate
6. If |angle| ≤ 3°: skip (negligible skew)
7. If |angle| > 30°: skip (too extreme, likely wrong detection)
8. Apply affine rotation: cv2.getRotationMatrix2D(center, angle, 1.0)
9. cv2.warpAffine with border replication
```

---

### S6: Photometric Enhancement

| Property | Specification |
|----------|---------------|
| **Module** | `preprocessing/enhance.py` |
| **Sub-stages** | 4 sequential, each independently toggleable via config |

**Enhancement Chain:**
```
(a) RESIZE    — Upscale to minimum 200px height, preserve aspect ratio
                cv2.resize(..., interpolation=cv2.INTER_CUBIC)

(b) CLAHE     — Convert BGR → LAB color space
                Apply CLAHE on L-channel: clipLimit=2.0, tileGridSize=(8,8)
                Convert LAB → BGR

(c) GAUSSIAN  — Kernel (3,3), σ auto-computed
                Mild noise suppression

(d) BILATERAL — d=9, sigmaColor=75, sigmaSpace=75
                Edge-preserving noise smoothing
```

All sub-stages preserve the original image separately for audit/debug. The `PreprocessingChain` class executes S4→S5→S6 as a sequential pipeline, with each stage's output becoming the next stage's input.

---

### S7: LLM OCR + Agentic Reasoning

| Property | Specification |
|----------|---------------|
| **Module** | `ocr/llm_ocr.py` + `ocr/strategy.py` |
| **Model** | Qwen 3.5 9B (Q4_K_M GGUF, 6.6 GB) served by Ollama |
| **Invocation** | HTTP POST to `http://localhost:11434/api/chat` |
| **Input** | System prompt + base64-encoded enhanced plate image |
| **Context Window** | Hard-capped at 4096 tokens (`num_ctx: 4096`) |
| **Thinking Mode** | Explicitly disabled (`"think": false` in API payload) |
| **Output Schema** | XML: `<plate_text>`, `<confidence>`, `<char_alternatives>`, `<reasoning_trace>` |
| **Latency Target** | ≤ 2000ms per plate |
| **VRAM** | Region A (5120 MB weights) + Region B (512 MB KV cache) |
| **Fallback** | Ollama timeout/circuit open → EasyOCR (CPU-only) |

**Critical Discovery — Qwen 3.5 Thinking Mode:**
Qwen 3.5 enables chain-of-thought "thinking mode" by default. The API response has a separate `thinking` field that consumes `num_predict` tokens, leaving the `content` field empty. All three Ollama API call sites (`llm_client.py`, `llm_ocr.py`, `adjudicator.py`) include `"think": false` in the payload to ensure clean, direct output.

**OCR Strategy Pattern:**
```python
class OCRStrategy:
    """Strategy pattern: Primary → Fallback with circuit breaker"""
    
    def recognize(self, plate_image) -> OCRResult:
        if circuit_breaker.state != OPEN:
            try:
                result = ollama_llm_ocr.recognize(plate_image)
                circuit_breaker.record_success()
                return result
            except (timeout, error):
                circuit_breaker.record_failure()
        
        # Fallback to EasyOCR
        return easyocr_engine.recognize(plate_image)
```

**Circuit Breaker States:**
```
CLOSED ──[failure_count >= threshold]──▶ OPEN
   ▲                                       │
   │                                       │ [recovery_timeout expires]
   │                                       ▼
   └────────[success]──────────────── HALF_OPEN
                                       │
                                       │ [failure]
                                       ▼
                                      OPEN
```

- `failure_threshold`: 3 consecutive failures
- `recovery_timeout_s`: 30 seconds
- `half_open_max_calls`: 1 (single test request)

---

### S8: Post-Processing & Dispatch

| Property | Specification |
|----------|---------------|
| **Module** | `postprocessing/orchestrator.py` |
| **Architecture** | Two-layer Cognitive Orchestrator |

**Layer 1 — DeterministicValidator (`validator.py`):**
```
1. Position-aware character correction:
   - Positions expected to be letters: 0↔O, 1↔I, 5↔S, 8↔B, 2↔Z
   - Positions expected to be digits: O↔0, I↔1, S↔5, B↔8, Z↔2
   
2. Noise character removal:
   - Strip whitespace, special characters, non-alphanumeric glyphs

3. Regex validation:
   - Default Indian format: ^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$
   - Configurable per deployment locale

4. Validation status assignment:
   - "valid" → passes regex, confidence ≥ threshold
   - "low_confidence" → passes regex, confidence < threshold
   - "regex_fail" → fails regex validation
```

**Layer 2 — AgenticAdjudicator (`adjudicator.py`):**
```
- Activated ONLY on "regex_fail" or "low_confidence" results
- Sends original OCR result + plate image to Qwen 3.5 9B
- LLM provides adjudicated plate text with reasoning
- If adjudication succeeds → upgrade validation status
- If adjudication fails → retain original status, log to audit
- "think": false required in API payload
```

**Deduplication (`deduplicator.py`):**
```
- Sliding window: default 10 seconds (configurable)
- Same plate text from same camera within window → suppressed
- Keep the detection with highest OCR confidence
- Counter: FRAMES_DEDUPLICATED incremented
```

**Dispatch (`dispatcher.py`):**
```
Three parallel dispatch targets:
1. PostgreSQL → INSERT into detection_events table
2. MinIO S3 → Upload plate/vehicle images (deterministic key)
3. Redis PUBLISH → "anpr:detections" channel → WebSocket broadcast
All operations are async with error isolation.
```

---

## 8. GPU & VRAM Memory Management

### 4-Region VRAM Budget (Strictly Enforced)

```
┌─────────────────────────────────────────────────────────────────┐
│                    VRAM: 8192 MB (RTX 4070)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────┐                │
│  │  REGION A — LLM Weights                     │  5120 MB      │
│  │  Qwen 3.5 9B Q4_K_M (6.6 GB on disk)       │               │
│  │  Static. Loaded once at boot. Never evicted. │               │
│  └─────────────────────────────────────────────┘                │
│                                                                 │
│  ┌─────────────────────────────────────────────┐                │
│  │  REGION B — LLM KV Cache                    │  512 MB       │
│  │  Dynamic. Grows per token.                   │               │
│  │  Capped at num_ctx=4096 tokens               │               │
│  └─────────────────────────────────────────────┘                │
│                                                                 │
│  ┌─────────────────────────────────────────────┐                │
│  │  REGION C — Vision Model Workspace           │  1024 MB      │
│  │  Time-sliced: YOLOv8 vehicle + LPD plate     │               │
│  │  INT8 quantized (TensorRT/ONNX Runtime)      │               │
│  │  Includes: weights + input + output tensors   │               │
│  └─────────────────────────────────────────────┘                │
│                                                                 │
│  ┌─────────────────────────────────────────────┐                │
│  │  REGION D — System / CUDA Overhead           │  512 MB       │
│  │  Non-negotiable. OS GPU driver, CUDA context  │               │
│  └─────────────────────────────────────────────┘                │
│                                                                 │
│  Safety margin: 256 MB (configurable)                           │
└─────────────────────────────────────────────────────────────────┘
```

### GPUMemoryManager — 3-Mode Offload Controller

```
┌───────────────┐── free_vram ≥ 1024 MB ──▶ GPU_PRIMARY
│               │   All inference on GPU
│ GPUMemory     │
│ Manager       │── free_vram ≥ 512 MB ───▶ PARTIAL_OFFLOAD
│               │   Vehicle det on GPU, enhancement on CPU
│               │
│               │── free_vram < 512 MB ───▶ FULL_CPU
└───────────────┘   Emergency CPU-only mode
```

### VRAMMonitor

- Background `asyncio.Task` polling pynvml at 500ms intervals
- Tracks per-region VRAM utilisation
- Reports PCIe TX/RX bandwidth, GPU temperature
- Feeds `VRAM_USAGE` Prometheus gauge

### Budget Validation

`VRAMBudgetReport.validate_budget()` runs at startup:
- If `Region_A + Region_B + Region_C + Region_D > ceiling_mb` → raises `MemoryError`
- Prevents pipeline launch with overcommitted VRAM budget

---

## 9. OCR Strategy Pattern

The OCR layer implements the **Strategy design pattern** with automatic failover:

```
                    ┌──────────────────────┐
                    │    OCRStrategy       │
                    │  (strategy.py)       │
                    │                      │
                    │  recognize(image)    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Circuit Breaker     │
                    │  CLOSED? ────────────┼──▶ YES ──▶ OllamaLLMOCR (Primary)
                    │                      │            │
                    │  OPEN/HALF_OPEN? ───┼──▶ YES ──▶ EasyOCR (Fallback)
                    └──────────────────────┘

OllamaLLMOCR (Primary):
  - Encodes plate image to base64 PNG
  - Constructs system prompt with XML output schema
  - POST /api/chat with {model, messages, stream: false, think: false, options: {...}}
  - Parses XML response: <plate_text>, <confidence>, <char_alternatives>
  - On parse failure: appends error to context, re-prompts (max 2 retries)
  - On timeout (5s): records circuit breaker failure

EasyOCR (Fallback):
  - Runs on CPU via ThreadPoolExecutor (no VRAM usage)
  - Returns raw text + confidence score
  - Lower accuracy but guaranteed availability
```

---

## 10. Cognitive Orchestrator (Post-Processing)

The post-processing stage uses a **2-layer cognitive architecture**:

```
                    ┌──────────────────────────────┐
                    │  CognitiveOrchestrator       │
                    │  (orchestrator.py)            │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │  Layer 1: DeterministicValidator         │
              │  (validator.py)                          │
              │                                          │
              │  1. Position-aware char correction       │
              │  2. Noise character removal               │
              │  3. Regex validation                      │
              │  4. Validation status assignment          │
              │                                          │
              │  Result: valid? ─────────────────────────┼──▶ PASS → Dispatch
              │  Result: regex_fail / low_confidence ───┼──▶ Layer 2
              └────────────────────┬────────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │  Layer 2: AgenticAdjudicator             │
              │  (adjudicator.py)                        │
              │                                          │
              │  - LLM-based multi-engine adjudication   │
              │  - Sends plate image + OCR result to LLM │
              │  - Qwen 3.5 reasons about the correct    │
              │    plate text and provides confidence     │
              │  - Only activated on Layer 1 failures     │
              │                                          │
              │  Result: adjudicated → upgrade to valid   │
              │  Result: failed → retain original status  │
              └────────────────────────────────────────┘
```

---

## 11. Agentic AI System

The agentic control plane transforms Uni_Vision from a passive pipeline into an **intelligent, self-managing system**. An AI agent backed by Qwen 3.5 9B can monitor, diagnose, tune, and repair the pipeline through natural language interaction.

### 11.1 ReAct Pattern & Agent Loop

The agent uses the **ReAct** (Reasoning + Acting) pattern:

```
User Query: "Why is camera cam_01 producing low confidence detections?"

Iteration 1:
  Thought: "I need to check the camera's error profile and recent OCR stats"
  Action: get_camera_error_profile
  Arguments: {"camera_id": "cam_01"}
  
Observation: {"camera_id": "cam_01", "total_errors": 15, "ocr_failures": 8, ...}

Iteration 2:
  Thought: "The error rate is high. Let me check OCR error patterns for this camera"
  Action: get_ocr_error_patterns
  Arguments: {}
  
Observation: [{"pattern": "O↔0 confusion", "frequency": 23}, ...]

Iteration 3:
  Thought: "The camera has high O/0 confusion. This suggests poor contrast or glare.
            I should recommend adjusting the enhancement settings."
  Answer: "Camera cam_01 has an elevated error rate (15 errors, 8 OCR failures). 
           The most common issue is O/0 character confusion (23 occurrences), 
           which typically indicates poor contrast or glare conditions. 
           I recommend enabling CLAHE enhancement or adjusting the confidence threshold."
```

**Agent Loop Implementation (`loop.py`):**
```python
class AgentLoop:
    async def run(self, query: str, memory: WorkingMemory, 
                  tools: ToolRegistry, client: AgentLLMClient) -> str:
        for i in range(max_iterations):
            # Build messages from working memory
            messages = memory.get_messages()
            
            # Get LLM response
            response = await client.chat(messages)
            
            # Parse response as JSON
            parsed = json.loads(response.content)
            
            if "answer" in parsed:
                return parsed["answer"]  # Final answer
            
            if "action" in parsed:
                # Execute tool
                result = await tools.invoke(
                    parsed["action"], 
                    parsed["arguments"],
                    context=execution_context
                )
                # Add observation to memory
                memory.add_message("tool", str(result))
        
        return "Maximum iterations reached."
```

**LLM Output Format:**
```json
// Tool call
{"thought": "I need to check system health", "action": "get_system_health", "arguments": {}}

// Final answer
{"thought": "Based on the data, the system is healthy", "answer": "The system is operating normally."}
```

### 11.2 Tool Registry & Definitions

The `@tool` decorator automatically generates JSON schemas from type hints:

```python
@tool(description="Get current pipeline health status")
async def get_system_health(ctx: ToolExecutionContext) -> dict:
    """Returns pipeline health including running status, stream count, etc."""
    return {
        "pipeline_running": True,
        "active_streams": 4,
        "queue_depth": 3,
        "vram_usage_mb": 6800,
        ...
    }
```

**ToolRegistry internals:**
- `register(fn)` — Extracts `_tool_definition` from decorated function
- `list_tools()` — Returns all `ToolDefinition` objects
- `tool_names()` — Returns list of registered tool names
- `invoke(name, args, context)` — Calls tool function, tracks invocation/error counts
- Auto-schema: Inspects function signature, generates JSON Schema for parameters, skips `self` and `ctx` parameters

### 11.3 Multi-Agent Routing

The `MultiAgentRouter` classifies queries and routes them to specialized sub-agents:

```
User Query → IntentClassifier → AgentRole → SubAgentProfile → Filtered ToolRegistry

                    ┌──────────────┐
                    │ IntentClass. │
                    │              │
                    │ 8 intents:   │
                    │ DETECTION    │──▶ OCR Quality Agent (12 tools)
                    │ ANALYTICS    │──▶ Analytics Agent (10 tools)
                    │ OPERATIONS   │──▶ Operations Agent (13 tools)
                    │ KNOWLEDGE    │──▶ OCR Quality Agent
                    │ MONITORING   │──▶ Operations Agent
                    │ CONFIG       │──▶ Operations Agent
                    │ DIAGNOSTIC   │──▶ OCR Quality Agent
                    │ GENERAL      │──▶ Full registry (all 30 tools)
                    └──────────────┘
```

**3 Sub-Agent Profiles:**

| Agent Role | Tool Count | Capabilities |
|-----------|-----------|--------------|
| **OCR Quality** | 12 | Camera diagnostics, error profiles, OCR error patterns, plate feedback, audit log search, anomaly detection |
| **Analytics** | 10 | Query detections, analytics queries, plate patterns, plate frequency, cross-camera analysis, knowledge stats |
| **Operations** | 13 | System health, pipeline stats, queue pressure, self-heal, auto-tune confidence, config, camera management, circuit breaker reset, flush queue |

Each sub-agent receives a **filtered `ToolRegistry`** containing only the tools relevant to its role, preventing tool confusion and improving LLM reasoning quality.

### 11.4 Working Memory & Session Management

**WorkingMemory (`memory.py`):**
- Bounded FIFO message buffer with configurable token budget (default: 4000 tokens)
- System prompt is **pinned** — never evicted
- When budget exceeded → oldest non-system messages evicted first
- Message types: system, user, assistant, tool (observation)

**SessionManager (`sessions.py`):**
- Multi-turn conversation sessions
- TTL: 1800 seconds (30 minutes) per session
- Max concurrent sessions: 100
- LRU eviction when session limit reached
- Each session stores: `session_id`, `created_at`, `last_accessed`, `WorkingMemory`

### 11.5 Intent Classification

Zero-cost keyword/regex classifier (`intent.py`) — no LLM calls required:

```python
class QueryIntent(Enum):
    DETECTION   = "detection"      # plate lookups, searches
    ANALYTICS   = "analytics"      # trends, summaries, statistics
    OPERATIONS  = "operations"     # health, config, pipeline control
    KNOWLEDGE   = "knowledge"      # error patterns, learning
    MONITORING  = "monitoring"     # alerts, status checks
    CONFIG      = "config"         # threshold adjustments, camera setup
    DIAGNOSTIC  = "diagnostic"     # camera diagnostics, troubleshooting
    GENERAL     = "general"        # fallback for unclassified queries
```

Each intent has a list of regex patterns. The classifier scores queries against all patterns, selecting the intent with the highest pattern hit ratio.

### 11.6 Knowledge Base & Learning Loop

The `KnowledgeBase` (`knowledge.py`) provides persistent learning:

**In-Memory Stores:**
```python
class KnowledgeBase:
    plate_observations: Dict[str, List[PlateObservation]]   # per-plate history
    camera_error_profiles: Dict[str, CameraErrorProfile]    # per-camera error stats
    feedback_entries: List[FeedbackEntry]                    # human corrections
    
    # FIFO capped stores (prevent unbounded growth)
    max_observations_per_plate: int = 100
    max_feedback_entries: int = 500
```

**PlateObservation:**
```python
@dataclass
class PlateObservation:
    plate_text: str
    camera_id: str
    confidence: float
    engine: str          # "ollama_llm" or "easyocr"
    validation_status: str  # "valid", "low_confidence", "regex_fail"
    timestamp: datetime
```

**Knowledge Tools:**
- `get_frequent_plates()` → Top-N plates by detection frequency
- `get_camera_error_profile(camera_id)` → Error breakdown per camera
- `get_cross_camera_plates()` → Plates seen across multiple cameras
- `get_ocr_error_patterns()` → Common OCR confusion patterns
- `detect_plate_anomalies()` → Statistical anomaly detection

**Persistence:** Knowledge state serialized to PostgreSQL `agent_knowledge` table (JSONB) on shutdown, restored on startup.

### 11.7 Autonomous Monitoring Agent

The `AutonomousMonitor` (`monitor.py`) is a background `asyncio.Task`:

```
Every 60 seconds:
  1. Check circuit breaker state
     → If OPEN: attempt auto-remediation (reset if recovery timeout passed)
  
  2. Check inference queue pressure
     → If depth > high_water mark: alert + throttle ingestion
  
  3. Check error rate (recent detection failures / total)
     → If spike detected: diagnose + auto-tune confidence thresholds
  
  4. Check VRAM usage
     → If near ceiling: alert + switch to PARTIAL_OFFLOAD mode
  
  5. Emit health status via WebSocket (/ws/agent)
```

**Auto-Remediation Actions:**
- Reset circuit breaker after recovery timeout
- Flush inference queue on persistent backlog
- Auto-tune confidence thresholds based on error analysis
- Switch GPU memory mode on VRAM pressure

### 11.8 Audit Trail

The `AuditTrail` (`audit.py`) provides compliance-grade logging:

```python
class AuditEntry:
    session_id: str
    intent: str
    agent_role: str
    action: str           # tool name or "answer"
    arguments: dict       # JSONB
    result: str
    success: bool
    elapsed_ms: float
    timestamp: datetime
```

**Write-Behind Buffer:**
- Entries buffered in memory (max 50)
- Flushed to PostgreSQL `agent_audit_log` table in batch
- Triggers: buffer full, periodic flush (every 50 operations), shutdown
- Zero pipeline latency impact

### 11.9 Pipeline Control Tools

**30 registered tools organized into 3 categories:**

**Pipeline Management (10 tools — `pipeline_tools.py`):**

| Tool | Description |
|------|-------------|
| `query_detections` | Search detection records with filters (camera, plate, time range) |
| `get_detection_summary` | Aggregate statistics (total, by camera, by status) |
| `get_pipeline_stats` | Raw Prometheus metric values |
| `get_system_health` | Pipeline running status, stream count, queue depth, VRAM |
| `list_cameras` | All registered camera sources |
| `manage_camera` | Add/remove/enable/disable cameras at runtime |
| `adjust_threshold` | Modify confidence thresholds (vehicle/plate detection, OCR) |
| `get_current_config` | Current runtime configuration snapshot |
| `search_audit_log` | Query OCR audit log entries |
| `analyze_plate_patterns` | Pattern analysis on detected plates |

**Pipeline Control (6 tools — `control_tools.py`):**

| Tool | Description |
|------|-------------|
| `auto_tune_confidence` | AI-driven confidence threshold optimization |
| `get_stage_analytics` | Per-stage latency and throughput metrics |
| `get_ocr_strategy_stats` | Primary vs fallback OCR usage statistics |
| `self_heal_pipeline` | Diagnose and repair pipeline issues |
| `get_queue_pressure` | Current inference queue depth and flow control status |
| `flush_inference_queue` | Clear backed-up frames from inference queue |

**Knowledge & Analytics (7 tools — `knowledge_tools.py`):**

| Tool | Description |
|------|-------------|
| `get_knowledge_stats` | Knowledge base size and coverage statistics |
| `get_frequent_plates` | Top-N most frequently detected plates |
| `get_camera_error_profile` | Error breakdown for a specific camera |
| `get_all_camera_profiles` | Error profiles for all cameras |
| `get_cross_camera_plates` | Plates detected across multiple cameras |
| `get_ocr_error_patterns` | Common OCR character confusion patterns |
| `detect_plate_anomalies` | Statistical anomaly detection in plate data |

**Additional tools (from coordinator.py):**

| Tool | Description |
|------|-------------|
| `record_plate_feedback` | Record human correction/confirmation for a plate |
| `get_recent_feedback` | Retrieve recent feedback entries |
| `get_camera_hints` | Get ML-derived hints for camera positioning |
| `save_knowledge` | Persist knowledge base to database |
| `diagnose_camera` | Deep diagnostic analysis for a specific camera |
| `reset_circuit_breaker` | Force-reset the OCR circuit breaker |
| `run_analytics_query` | Execute custom analytics queries |

---

## 12. REST API Layer

### Application Factory

The `create_app()` factory in `api/__init__.py` wires everything together:

```python
def create_app(config: AppConfig, start_pipeline: bool = True) -> FastAPI:
    app = FastAPI(title="Uni_Vision ANPR", version="0.1.0")
    
    # Middleware stack (applied in reverse order)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, rpm=config.api.rate_limit_rpm)
    app.add_middleware(APIKeyMiddleware, api_keys=config.api.api_keys)
    app.add_middleware(CORSMiddleware, allow_origins=config.api.cors_origins)
    
    # Route registration
    app.include_router(health_router)
    app.include_router(detections_router)
    app.include_router(sources_router)
    app.include_router(metrics_router)
    app.include_router(stats_router)
    app.include_router(agent_chat_router)
    app.include_router(ws_events_router)
    app.include_router(ws_agent_router)
    
    # Lifespan: build pipeline via DI, start background tasks
    # (Redis subscriber, RetentionTask, AgentCoordinator)
```

### Endpoint Reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | Public | Liveness/readiness probe with deep checks |
| GET | `/detections` | API Key | Paginated detection records (camera, plate, time, status filters) |
| POST | `/sources` | API Key | Register/upsert a camera source |
| GET | `/sources` | API Key | List all registered camera sources |
| DELETE | `/sources/{camera_id}` | API Key | Remove a camera source |
| GET | `/metrics` | Public | Prometheus text-format scrape endpoint |
| GET | `/stats` | API Key | Pipeline telemetry summary |
| POST | `/api/agent/chat` | API Key | Natural language agent query (sync ReAct response) |
| GET | `/api/agent/status` | API Key | Agent health + available tool list |
| POST | `/api/agent/feedback` | API Key | Submit human feedback (confirm/correct/reject) |
| GET | `/api/agent/sessions` | API Key | List active agent sessions |
| DELETE | `/api/agent/sessions/{id}` | API Key | Delete a specific session |
| GET | `/api/agent/monitor` | API Key | Autonomous monitor status |
| GET | `/api/agent/agents` | API Key | List sub-agent profiles and capabilities |
| GET | `/api/agent/audit` | API Key | Query agent audit trail |

### Request/Response Examples

**Detection Query:**
```bash
GET /detections?camera_id=cam_01&from=2026-03-20T00:00:00Z&limit=10
```
```json
{
  "total": 1240,
  "page": 1,
  "results": [
    {
      "id": "uuid",
      "camera_id": "cam_01",
      "plate_number": "MH12AB1234",
      "ocr_confidence": 0.94,
      "vehicle_class": "car",
      "detected_at_utc": "2026-03-20T08:32:11Z",
      "validation_status": "valid",
      "vehicle_image_url": "https://storage/images/uuid.jpg"
    }
  ]
}
```

**Agent Chat:**
```bash
POST /api/agent/chat
{
  "message": "What's the system health status?",
  "session_id": "optional-session-uuid"
}
```
```json
{
  "response": "The system is operating normally. Pipeline is running with 4 active streams...",
  "session_id": "uuid",
  "agent_role": "operations",
  "tools_used": ["get_system_health"],
  "elapsed_ms": 1250
}
```

---

## 13. WebSocket Streaming

### `/ws/events` — Real-Time Detection Stream

Bridges Redis Pub/Sub to WebSocket clients:

```
Pipeline Dispatch → Redis PUBLISH "anpr:detections" → WS Bridge → Connected Clients

Frame format:
{
  "type": "detection",
  "data": {
    "plate_number": "MH12AB1234",
    "camera_id": "cam_01",
    "confidence": 0.94,
    "vehicle_class": "car",
    "timestamp": "2026-03-20T08:32:11Z"
  }
}
```

### `/ws/agent` — Streaming Agent Reasoning

Streams agent ReAct loop steps as they execute:

```
Client connects → sends query → receives streaming frames:

{"type": "intent",     "data": {"intent": "operations", "role": "operations"}}
{"type": "thought",    "data": {"thought": "I need to check system health"}}
{"type": "tool_call",  "data": {"tool": "get_system_health", "arguments": {}}}
{"type": "observation", "data": {"result": {"pipeline_running": true, ...}}}
{"type": "thought",    "data": {"thought": "The system is healthy"}}
{"type": "answer",     "data": {"answer": "The system is operating normally..."}}
{"type": "done",       "data": {"elapsed_ms": 1250}}
```

---

## 14. Middleware Stack

Middleware is applied in this order (outermost first):

```
Request → CORS → API Key Auth → Rate Limit → Security Headers → Route Handler
```

### API Key Authentication (`auth.py`)
- Keys provided via `UV_API_API_KEYS` (comma-separated)
- Keys stored as SHA-256 hashes in memory
- Client sends key in `X-API-Key` header
- Comparison uses `hmac.compare_digest()` for timing-safe validation
- Public paths exempted: `/health`, `/metrics`, `/docs`, `/openapi.json`
- Empty key list = auth disabled (development mode)

### Rate Limiting (`rate_limit.py`)
- Sliding-window per-IP counter
- Default: 120 requests/minute per IP
- Supports `X-Forwarded-For` for reverse proxy setups
- Returns `429 Too Many Requests` with `Retry-After` header
- Configurable via `UV_API_RATE_LIMIT_RPM` (0 = disabled)

### Security Headers (`security_headers.py`)
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Cache-Control: no-store
Content-Security-Policy: default-src 'self'
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

---

## 15. Storage Layer

### PostgreSQL (asyncpg)

**Connection Management:**
- Async connection pool via `asyncpg.create_pool()`
- Pool bounds: min=2, max=8 (configurable)
- Connection string: `UV_POSTGRES_DSN`
- All queries use parameterized statements (SQL injection safe)

**Core Operations:**
```python
class DatabaseClient:
    async def insert_detection(record: DetectionRecord) -> str:
        """INSERT into detection_events, returns UUID"""
    
    async def query_detections(filters) -> List[DetectionRecord]:
        """Paginated SELECT with camera/plate/time/status filters"""
    
    async def upsert_camera(source: CameraSource) -> None:
        """INSERT ... ON CONFLICT UPDATE for camera sources"""
    
    async def insert_audit_entry(entry: AuditEntry) -> None:
        """INSERT into ocr_audit_log (low-confidence/failure records)"""
```

### MinIO Object Storage (aioboto3)

```python
class ObjectStoreArchiver:
    async def upload_image(camera_id, record_id, image_bytes, ext="png") -> str:
        """Upload to plates/{camera_id}/{record_id}.{ext}"""
        # Deterministic key path
        # Exponential-backoff retry on failure
        # Returns the S3 object key
```

### Data Retention

`RetentionTask` — Background async purge:
- Configurable via `UV_RETENTION_*` environment variables
- Detection events: purge older than `max_age_days` (default: 90)
- Audit logs: purge older than `audit_max_age_days` (default: 180)
- Batch deletion: `batch_size` rows per DELETE (default: 1000) to limit lock time
- Check interval: every `check_interval_hours` (default: 24)

---

## 16. Database Schema & Migrations

### Schema (4 tables)

**`detection_events`** — Primary detection records:
```sql
CREATE TABLE detection_events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id           VARCHAR(64) NOT NULL,
    plate_number        VARCHAR(20) NOT NULL,
    raw_ocr_text        VARCHAR(50),
    ocr_confidence      FLOAT NOT NULL,
    ocr_engine          VARCHAR(20) NOT NULL,       -- 'ollama_llm' | 'easyocr'
    vehicle_class       VARCHAR(20),                 -- 'car' | 'truck' | 'bus' | 'motorcycle'
    vehicle_image_path  TEXT,
    plate_image_path    TEXT,
    detected_at_utc     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_validated        BOOLEAN DEFAULT FALSE,
    validation_status   VARCHAR(20),                 -- 'valid' | 'low_confidence' | 'regex_fail'
    location_tag        VARCHAR(100),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
```

**`camera_sources`** — Camera registry:
```sql
CREATE TABLE camera_sources (
    camera_id       VARCHAR(64) PRIMARY KEY,
    source_url      TEXT NOT NULL,
    location_tag    VARCHAR(100),
    fps_target      SMALLINT DEFAULT 3,
    enabled         BOOLEAN DEFAULT TRUE,
    added_at        TIMESTAMPTZ DEFAULT NOW()
);
```

**`ocr_audit_log`** — Low-confidence/failure audit:
```sql
CREATE TABLE ocr_audit_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id       VARCHAR(64),
    raw_ocr_text    VARCHAR(100),
    ocr_confidence  FLOAT,
    failure_reason  VARCHAR(50),
    frame_path      TEXT,
    logged_at       TIMESTAMPTZ DEFAULT NOW()
);
```

**`agent_audit_log`** — Agent action audit:
```sql
CREATE TABLE agent_audit_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      VARCHAR(64),
    intent          VARCHAR(32),
    agent_role      VARCHAR(32),
    action          VARCHAR(64),
    arguments       JSONB,
    result          TEXT,
    success         BOOLEAN DEFAULT TRUE,
    elapsed_ms      FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### Performance Indexes (Migration 002)

```sql
CREATE INDEX idx_detection_camera_time ON detection_events (camera_id, detected_at_utc DESC);
CREATE INDEX idx_detection_plate ON detection_events (plate_number);
CREATE INDEX idx_detection_status ON detection_events (validation_status);
CREATE INDEX idx_audit_camera_time ON ocr_audit_log (camera_id, logged_at DESC);
```

### Migration History

| Version | Name | Description |
|---------|------|-------------|
| 001 | `initial_schema` | detection_events, camera_sources, ocr_audit_log tables |
| 002 | `perf_indexes` | B-tree + composite indexes for query performance |
| 003 | `agent_knowledge` | agent_knowledge table (JSONB store for KB persistence) |
| 004 | `agent_audit` | agent_audit_log table for agent action compliance |

---

## 17. Monitoring & Observability

### Prometheus Metrics (14+)

**Counters:**
| Metric | Description |
|--------|-------------|
| `uv_frames_ingested_total` | Total frames decoded from camera streams |
| `uv_frames_deduplicated_total` | Frames discarded by pHash deduplication |
| `uv_frames_dropped_total` | Frames dropped due to queue overflow |
| `uv_detections_total` | Total successful detections (vehicle + plate + OCR) |
| `uv_ocr_requests_total` | Total OCR engine invocations |
| `uv_ocr_success_total` | Successful OCR completions |
| `uv_ocr_fallback_total` | Fallback OCR invocations (EasyOCR) |
| `uv_dispatch_success_total` | Successful DB+S3+Redis dispatches |
| `uv_dispatch_errors_total` | Dispatch failures |
| `uv_agent_requests_total` | Agent chat queries received |
| `uv_agent_tool_calls_total` | Agent tool invocations |

**Histograms:**
| Metric | Description |
|--------|-------------|
| `uv_pipeline_latency_seconds` | End-to-end pipeline latency (S0→S8) |
| `uv_stage_latency_seconds` | Per-stage latency (labels: stage_name) |
| `uv_ocr_confidence` | OCR confidence score distribution |
| `uv_agent_latency_seconds` | Agent chat response time |
| `uv_agent_steps` | Number of ReAct steps per query |

**Gauges:**
| Metric | Description |
|--------|-------------|
| `uv_vram_usage_bytes` | Current GPU VRAM utilisation |
| `uv_inference_queue_depth` | Current inference queue size |
| `uv_stream_status` | Per-camera stream health (0=down, 1=up) |

### Prometheus Alert Rules (`prometheus_alerts.yml`)

```yaml
- High request latency:    p95 > 2 seconds for 5 minutes
- Elevated error rate:     > 5% 5xx responses for 5 minutes
- Queue depth saturation:  > high_water mark for 5 minutes
- VRAM budget exceeded:    usage > ceiling - safety_margin
- Connection pool exhaust: available connections < 1 for 1 minute
```

### Grafana Dashboards

Pre-built dashboards auto-provisioned:
- **Pipeline Overview** — Throughput, latency, queue depth, detection rate
- **OCR Performance** — Confidence distribution, engine comparison, fallback rate
- **GPU Utilisation** — VRAM per region, temperature, PCIe bandwidth
- **API Performance** — Request rate, latency percentiles, error rate

### Profiling Infrastructure

```python
@profile_stage("vehicle_detection")
async def detect_vehicles(frame):
    # Automatically measures:
    # - Execution time (wall clock + CPU time)
    # - VRAM delta (before/after via pynvml)
    # - Stored in ring buffer (512 entries)
    ...
```

`VRAMSampler` context manager for fine-grained VRAM tracking:
```python
async with VRAMSampler() as sampler:
    result = await model.predict(tensor)
    # sampler.delta_mb → VRAM change during block
```

---

## 18. Streamlit Visualizer

An 8-page debugging dashboard for inspecting each pipeline stage:

| Page | Stage | Visualization |
|------|-------|---------------|
| 1: Pipeline Stats | Overview | Real-time throughput, latency, queue depth |
| 2: Frame Sampler | S1 | pHash similarity scores, dedup decisions |
| 3: Vehicle Detection | S2 | Bounding box overlay on frame, confidence bars |
| 4: Plate Detection | S3 | Plate localization overlay, ROI highlighting |
| 5: Crop & Straighten | S4-S5 | Side-by-side: original crop vs deskewed result |
| 6: Enhancement | S6 | Before/after: CLAHE, bilateral filter effects |
| 7: OCR Output | S7 | LLM raw response, parsed result, fallback comparison |
| 8: Post-Processing | S8 | Validation decisions, adjudication traces, dedup log |

**Launch:** `streamlit run src/uni_vision/visualizer/app.py`

---

## 19. Error Handling & Failure Taxonomy

Complete exception hierarchy with failure codes:

```
UniVisionError (base)
├── StreamError (F01)               — Camera stream failure
├── QueueOverflowError (F02)        — Frame queue exceeded maxsize
├── DetectionError
│   ├── NoVehicleDetected (F03)     — No vehicle in frame (expected, frequent)
│   └── NoPlateDetected (F04)       — Vehicle found but no plate
├── VRAMError
│   └── VRAMBudgetExceeded (F05)    — VRAM usage exceeds region budget
├── OllamaError
│   ├── OllamaTimeoutError (F06)    — Ollama HTTP timeout (>5s)
│   └── OllamaCircuitOpen (F06)     — Circuit breaker in OPEN state
├── LLMParseError (F07)             — Failed to parse LLM structured output
├── LLMRepetitionError (F08)        — LLM produced repetitive/looping output
├── StorageError
│   ├── DatabaseWriteError (F09)    — PostgreSQL INSERT failure
│   └── ObjectStoreError (F10)      — MinIO S3 upload failure
├── ConfigurationError              — Invalid config values
├── PipelineShutdownError           — Graceful shutdown failure
└── AgentError (F11)
    ├── ToolExecutionError (F12)    — Tool function raised exception
    └── AgentTimeoutError (F13)     — Agent loop exceeded max iterations
```

**Recovery Strategy per Failure:**

| Failure | Recovery |
|---------|----------|
| F01 StreamError | Exponential backoff reconnection (1s→16s) |
| F02 QueueOverflow | Drop oldest frames, increment FRAMES_DROPPED counter |
| F03 NoVehicle | Skip frame, log to audit (expected behavior) |
| F04 NoPlate | Skip frame, log to audit |
| F05 VRAMExceeded | Switch to PARTIAL_OFFLOAD or FULL_CPU mode |
| F06 OllamaTimeout | Record circuit breaker failure → fallback to EasyOCR |
| F07 LLMParseError | Re-prompt with error context (max 2 retries) → EasyOCR fallback |
| F08 LLMRepetition | Abort OCR, emit low_confidence record |
| F09 DatabaseWrite | Log error, retry with backoff |
| F10 ObjectStore | Log error, skip image upload, continue pipeline |
| F12 ToolExecution | Return error message to agent, agent decides next action |

---

## 20. Testing Strategy

### Test Suite Summary

- **Total Tests:** 288 passed, 21 skipped
- **Framework:** pytest + pytest-asyncio + pytest-cov
- **Command:** `python -m pytest tests/ -q --tb=short --timeout=10`

### Test Infrastructure (`conftest.py`)

The conftest stubs **15+ heavyweight dependencies** that require CUDA, system libraries, or network:

```python
# Stubbed modules (replaced with MagicMock or minimal implementations):
cv2              # Minimal: cvtColor, resize, INTER_CUBIC, etc.
pynvml           # Full mock (no GPU required for tests)
torch            # Full mock
tensorrt         # Full mock
asyncpg          # Full mock (no PostgreSQL required)
easyocr          # Full mock
httpx            # Full mock (no Ollama required)
aioboto3         # Full mock (no MinIO required)
redis            # Full mock
prometheus_client # Minimal: Counter, Histogram, Gauge with _FakeMetric
structlog        # Minimal: get_logger() returning Mock
```

This enables the **full test suite to run without GPU, database, or any external service**.

### Unit Tests (16 files, ~240 tests)

| Test File | Coverage Area | Key Tests |
|-----------|--------------|-----------|
| `test_agent.py` | Agent system (54 tests, 12 classes) | ToolRegistry, WorkingMemory, IntentClassifier, SessionManager, MultiAgentRouter, AuditTrail, Monitor, KnowledgeBase, LLMClient, Prompts, CoordinatorUnit, APIEndpoints |
| `test_api.py` | API endpoints | Health, detections, sources CRUD, stats |
| `test_config.py` | Configuration | Default values, env var overrides, YAML loading |
| `test_container.py` | DI container | Component wiring, lazy initialization |
| `test_dtos.py` | Data transfer objects | FrameData, BoundingBox, OCRResult construction |
| `test_exceptions.py` | Exception hierarchy | Custom exception attributes, inheritance |
| `test_deduplicator.py` | Sliding-window dedup | Time window, confidence preference, multi-camera |
| `test_models.py` | Detection models | Interface compliance, output format |
| `test_phash.py` | Perceptual hashing | Hash computation, hamming distance, dedup decisions |
| `test_profiler.py` | Stage profiling | Timing accuracy, ring buffer, VRAM sampling |
| `test_response_parser.py` | LLM output parsing | XML parsing, JSON parsing, error handling |
| `test_security.py` | Auth + rate limiting | API key validation, timing-safe comparison, rate limit enforcement |
| `test_validator.py` | Character correction | O↔0 substitution, regex validation, position-aware correction |
| `test_vram_budget.py` | VRAM budget | Budget calculation, overcommit detection, region allocation |
| `test_websocket.py` | WebSocket lifecycle | Connection, message format, disconnect handling |
| `test_visualizer.py` | Streamlit pages | Page rendering, data formatting |

### Integration Tests (5 files, ~48 tests)

| Test File | Coverage Area |
|-----------|--------------|
| `test_pipeline_smoke.py` | Full S2→S8 chain with mock stages (no GPU/Ollama/DB) |
| `test_api_lifecycle.py` | FastAPI TestClient with full middleware + mocked DB |
| `test_config_loading.py` | YAML + env var precedence, 3-layer override logic |
| `test_dispatcher_flow.py` | DB→S3→Redis dispatch chain (mocked backends) |
| `test_websocket_broadcast.py` | Redis→WebSocket event propagation |

### End-to-End Smoke Test (`scripts/smoke_test_agent.py`)

3-part validation against live Qwen 3.5 9B model:

```
Test 1: Simple Q&A    → Clean text answer about ANPR
Test 2: Tool Call      → Valid JSON: {"thought": "...", "action": "get_system_health", "arguments": {}}
Test 3: Multi-Turn     → observation input → {"thought": "...", "answer": "Yes, the system is healthy."}

Result: ALL 3 TESTS PASSED ✅
```

---

## 21. CI/CD Pipeline

### GitHub Actions (`.github/workflows/`)

```yaml
# ci.yml — Triggered on push/PR to main
jobs:
  lint:
    - ruff check src/ tests/
    
  typecheck:
    - mypy src/uni_vision/
    
  test:
    - pytest tests/ -v --tb=short --timeout=10
    - Coverage report (target: ≥80%)
    
  build:
    - docker build -t uni-vision:latest .
    - Verify image builds successfully
```

### Makefile Targets

```makefile
make install        # Install editable with dev + inference extras
make install-dev    # Install dev-only (no CUDA dependencies)
make lint           # Run ruff linter
make lint-fix       # Auto-fix lint issues
make format         # Format code with ruff
make typecheck      # Run mypy type checking
make test           # Run all tests (verbose)
make test-quick     # Run tests (quiet mode)
make test-cov       # Run tests with coverage report
make build          # Build Docker image
make up             # docker compose up -d
make down           # docker compose down
make down-clean     # docker compose down -v (destroy volumes)
make serve          # uvicorn with factory pattern
make pipeline       # Run pipeline directly
make ollama-init    # Pull models + create custom variants
make db-upgrade     # Alembic upgrade head
make db-downgrade   # Alembic downgrade -1
make db-revision    # Generate new migration
```

---

## 22. Docker & Containerisation

### Multi-Stage Dockerfile

```dockerfile
# Stage 1: Builder (python:3.12-slim)
# - Installs system deps (build-essential, libpq-dev, libgl1)
# - Creates isolated venv at /opt/venv
# - Installs all Python dependencies including inference extras

# Stage 2: Runtime (nvidia/cuda:12.4.1-runtime-ubuntu22.04)
# - Copies venv from builder
# - Copies application source code
# - Runs Alembic migrations on startup
# - Launches uvicorn with factory app
```

### Docker Compose Stack (7 Services + 1 Init Sidecar)

```
┌──────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                        │
│                                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   app    │  │ postgres │  │  ollama  │  │  minio   │      │
│  │ :8000    │  │ :5432    │  │ :11434   │  │ :9000/01 │      │
│  │ GPU ✓   │  │ Vol: pg  │  │ GPU ✓   │  │ Vol: min │      │
│  └────┬─────┘  └──────────┘  └──────────┘  └──────────┘      │
│       │                                                        │
│  ┌────┴─────┐  ┌──────────┐  ┌──────────┐                    │
│  │  redis   │  │prometheus│  │ grafana  │                    │
│  │ :6379    │  │ :9090    │  │ :3000    │                    │
│  │ Vol: red │  │ Vol: pro │  │ Vol: gra │                    │
│  └──────────┘  └──────────┘  └──────────┘                    │
│                                                                │
│  ┌─────────────┐                                               │
│  │ ollama-init │ ← One-shot: pulls model + creates variants    │
│  │ (restart:no)│                                               │
│  └─────────────┘                                               │
└──────────────────────────────────────────────────────────────┘
```

**Service Details:**

| Service | Image | Port | GPU | Health Check | Volume |
|---------|-------|------|-----|-------------|--------|
| app | Custom (Dockerfile) | 8000 | ✓ (1 GPU) | curl /health | config (ro) |
| postgres | postgres:16-alpine | 5432 | — | pg_isready | pgdata |
| ollama | ollama/ollama:latest | 11434 | ✓ (1 GPU) | curl /api/tags | ollama_data |
| ollama-init | curlimages/curl:latest | — | — | — | scripts (ro) |
| minio | minio/minio:latest | 9000, 9001 | — | curl /health/live | minio_data |
| redis | redis:7-alpine | 6379 | — | redis-cli ping | redis_data |
| prometheus | prom/prometheus:latest | 9090 | — | — | prometheus_data |
| grafana | grafana/grafana:latest | 3000 | — | — | grafana_data |

---

## 23. Security Architecture

### Authentication
- **API Key Authentication** — SHA-256 hashed keys, timing-safe comparison
- Keys configured via `UV_API_API_KEYS` (comma-separated)
- Empty list = auth disabled (development mode only)
- `X-API-Key` header required on all non-public endpoints

### Authorization
- Public endpoints: `/health`, `/metrics`, `/docs`, `/openapi.json`
- All other endpoints require valid API key
- No role-based access control (single-tier auth)

### Transport Security
- Production deployment behind reverse proxy with TLS termination
- `Strict-Transport-Security` header enforced
- WebSocket connections upgraded from wss:// in production

### Data Security
- Parameterized SQL queries (asyncpg) — SQL injection safe
- Vehicle images stored with access-controlled S3 keys
- No PII logging in structured logs
- Sensitive config values (API keys, DB passwords) via environment variables only

### Rate Limiting
- Per-IP sliding window (default: 120 RPM)
- `X-Forwarded-For` support for reverse proxy deployments
- Returns `429 Too Many Requests` with `Retry-After` header

### Security Headers
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Cache-Control: no-store`
- `Content-Security-Policy: default-src 'self'`

---

## 24. Performance Targets & Benchmarks

### Pipeline Latency Targets

| Stage | Target Latency | Hardware |
|-------|---------------|----------|
| S0-S1: Ingest + Dedup | ~5ms | CPU |
| S2: Vehicle Detection | ≤ 80ms | GPU (INT8) |
| S3: Plate Localisation | ≤ 50ms | GPU (INT8) |
| S4: Crop + Transfer | < 1ms | GPU→CPU |
| S5: Deskew | ~2ms | CPU |
| S6: Enhancement | ~5ms | CPU |
| S7: LLM OCR | ≤ 2000ms | GPU (Ollama) |
| S8: Post-Processing | ~10ms | CPU |
| **Total E2E** | **≤ 3 seconds** | |

### Throughput

| Metric | Target |
|--------|--------|
| Concurrent camera streams | ≥ 4 at 5 FPS each |
| Frames processed per second | ≥ 20 (across all streams) |
| Frame queue overflow | 0 drops under normal load |

### Accuracy

| Metric | Target |
|--------|--------|
| Vehicle detection precision | ≥ 92% |
| Plate detection recall | ≥ 90% |
| Plate detection false positive rate | ≤ 5% |
| OCR exact match (clean, daytime) | ≥ 88% |
| OCR exact match (night/rain/noise) | ≥ 75% |
| Post-processing CER reduction | ≥ 15% vs raw OCR |
| Deduplication effectiveness | ≥ 20% reduction |

### Qwen 3.5 9B Validated Performance

Measured on RTX 4070 with `think: false`:
- Simple Q&A: ~1 second total, ~11.5 tok/s evaluation rate
- ReAct tool-call JSON: Clean structured output, correct format
- Multi-turn reasoning: Correct observation→answer chains

---

## 25. Data Flow Diagrams

### Detection Event Lifecycle

```
Camera RTSP Stream
       │
       ▼
  ┌─────────────────┐
  │ Frame Decoded    │ → 6.2 MB BGR uint8 (CPU)
  │ (per camera     │
  │  thread)        │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ pHash Dedup     │ → 64-bit hash comparison
  │ (hamming ≤ 5    │ → 20%+ frame reduction
  │  = duplicate)   │
  └────────┬────────┘
           │ (accepted frame)
           ▼
  ┌─────────────────┐
  │ GPU Upload      │ → Pinned memory → CUDA memcpy async
  │ (4.9 MB         │ → 640×640 float32 (Region C)
  │  letterboxed)   │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ YOLOv8 Vehicle  │ → N × (x1,y1,x2,y2,conf,cls)
  │ Detection       │ → NMS (IoU 0.45)
  │ (≤80ms)         │ → conf ≥ 0.60
  └────────┬────────┘
           │ (vehicle ROIs, GPU-resident)
           ▼
  ┌─────────────────┐
  │ YOLOv8 Plate    │ → Plate bbox (highest confidence)
  │ Detection       │ → Zero-copy tensor slice from vehicle ROI
  │ (per vehicle)   │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Crop + Download │ → ~0.36 MB plate image → CPU
  │ FREE Region C   │ → torch.cuda.empty_cache()
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ CPU Preprocess  │ → Deskew (Hough + affine)
  │ (S5 + S6)       │ → CLAHE + bilateral filter
  │ (~7ms total)    │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Ollama HTTP     │ → base64 PNG in prompt
  │ POST /api/chat  │ → think: false
  │ (≤2000ms)       │ → XML output: plate_text + confidence
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Cognitive       │ → Layer 1: Deterministic validation
  │ Orchestrator    │ → Layer 2: LLM adjudication (on failure)
  │ (S8)            │ → Deduplication (10s window)
  └────────┬────────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
┌──────┐ ┌────┐ ┌─────┐
│Postgr│ │S3  │ │Redis│
│ SQL  │ │img │ │pub/ │
│INSERT│ │PUT │ │sub  │
└──────┘ └────┘ └──┬──┘
                   │
                   ▼
              ┌─────────┐
              │WebSocket│ → Connected clients
              │broadcast│
              └─────────┘
```

### Agent Interaction Flow

```
User (HTTP POST /api/agent/chat or WS /ws/agent)
       │
       ▼
  ┌─────────────────┐
  │ IntentClassifier │ → 8 QueryIntent categories
  │ (zero-cost regex)│ → No LLM invocation
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ MultiAgentRouter │ → Intent → AgentRole mapping
  │                  │ → Filtered ToolRegistry
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ AgentLoop       │ → ReAct: Think → Act → Observe → Repeat
  │ (max 10 iter)   │ → LLM: Qwen 3.5 9B via Ollama
  └────────┬────────┘
           │
     ┌─────┤ (per iteration)
     │     │
     │     ▼
     │  ┌──────────────┐
     │  │ LLM Response │ → JSON: {thought, action, arguments}
     │  │              │    or:  {thought, answer}
     │  └──────┬───────┘
     │         │
     │    ┌────┴────┐
     │    │ action? │──▶ ToolRegistry.invoke(name, args, ctx)
     │    │ answer? │──▶ Return final answer to user
     │    └─────────┘
     │         │
     │         ▼ (observation)
     │    ┌──────────────┐
     │    │WorkingMemory │ → Append observation, check token budget
     │    │ (FIFO evict) │
     │    └──────────────┘
     │         │
     └─────────┘ (next iteration)
           │
           ▼
  ┌─────────────────┐
  │ AuditTrail      │ → Write-behind buffer → PostgreSQL
  │ (compliance)    │ → session_id, intent, role, action, result
  └─────────────────┘
```

---

## 26. Deployment Guide Summary

### Prerequisites

| Component | Minimum Version |
|-----------|----------------|
| Docker Engine | 24.0+ |
| Docker Compose | V2 |
| NVIDIA Driver | 535+ |
| NVIDIA Container Toolkit | 1.14+ |
| Disk | 40 GB free |
| RAM | 16 GB |
| GPU | NVIDIA with ≥ 8 GB VRAM |

### Quick Start (Docker)

```bash
git clone <repo-url> && cd Uni_Vision
cp .env.example .env               # Edit: set strong API keys + DB password
docker compose up -d --build       # Start all 7 services
docker compose exec app alembic upgrade head  # Run migrations
curl -s http://localhost:8000/health | jq .    # Verify
```

### Quick Start (Local Development)

```bash
make install-dev                   # Install dev dependencies
cp .env.example .env               # Configure local services
.\scripts\init-ollama.ps1          # Download + create Ollama models
make serve                         # Start API server
make pipeline                     # Start processing pipeline
```

### Services & Ports

| Service | URL |
|---------|-----|
| Uni_Vision API | http://localhost:8000 |
| MinIO Console | http://localhost:9001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Ollama | http://localhost:11434 |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |

### Production Hardening

1. **TLS**: Deploy reverse proxy (Caddy/NGINX) with auto-provisioned certificates
2. **Auth**: Set strong API keys via `UV_API_API_KEYS`
3. **Rate Limiting**: Configure `UV_API_RATE_LIMIT_RPM` (default: 120)
4. **Data Retention**: Enable `UV_RETENTION_ENABLED=true` with appropriate age limits
5. **Backups**: Schedule `pg_dump` via cron for PostgreSQL
6. **Monitoring**: Configure Prometheus alerting with the pre-built alert rules
7. **GPU**: Ensure NVIDIA Container Toolkit is properly configured

---

## Appendix A — Environment Variables Reference

### Hardware

| Variable | Default | Description |
|----------|---------|-------------|
| `UV_DEVICE` | `cuda` | Device mode: `cuda` or `cpu` |
| `UV_CUDA_DEVICE_INDEX` | `0` | CUDA device ordinal |
| `UV_VRAM_CEILING_MB` | `8192` | Max VRAM budget |
| `UV_VRAM_SAFETY_MARGIN_MB` | `256` | VRAM headroom |
| `UV_VRAM_POLL_INTERVAL_MS` | `500` | VRAM monitor polling interval |

### Ollama (LLM Runtime)

| Variable | Default | Description |
|----------|---------|-------------|
| `UV_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `UV_OLLAMA_MODEL` | `qwen3.5:9b-q4_K_M` | Model name |
| `UV_OLLAMA_TIMEOUT_S` | `5` | HTTP timeout |
| `UV_OLLAMA_NUM_CTX` | `4096` | Context window (tokens) |
| `UV_OLLAMA_TEMPERATURE` | `0.1` | Generation temperature |
| `UV_OLLAMA_TOP_P` | `0.9` | Top-p sampling |
| `UV_OLLAMA_TOP_K` | `20` | Top-k sampling |
| `UV_OLLAMA_REPEAT_PENALTY` | `1.15` | Repetition penalty |
| `UV_OLLAMA_NUM_PREDICT` | `256` | Max tokens to generate |
| `UV_OLLAMA_NUM_BATCH` | `256` | Batch size |
| `UV_OLLAMA_NUM_GPU` | `-1` | GPU layers (-1 = all) |
| `UV_OLLAMA_SEED` | `42` | Random seed |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `UV_POSTGRES_DSN` | `postgresql://uni_vision:changeme@localhost:5432/uni_vision` | Connection string |
| `UV_POSTGRES_POOL_MIN` | `2` | Min pool connections |
| `UV_POSTGRES_POOL_MAX` | `8` | Max pool connections |

### Object Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `UV_S3_ENDPOINT` | `http://localhost:9000` | MinIO endpoint |
| `UV_S3_BUCKET` | `uni-vision-images` | Image bucket name |
| `UV_S3_ACCESS_KEY` | `minioadmin` | S3 access key |
| `UV_S3_SECRET_KEY` | `minioadmin` | S3 secret key |
| `UV_S3_REGION` | `us-east-1` | S3 region |

### API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `UV_API_HOST` | `0.0.0.0` | Bind address |
| `UV_API_PORT` | `8000` | Listen port |
| `UV_API_API_KEYS` | *(empty)* | Comma-separated API keys (empty = no auth) |
| `UV_API_RATE_LIMIT_RPM` | `120` | Requests/minute per IP |
| `UV_API_CORS_ORIGINS` | *(empty)* | Allowed CORS origins |

### Data Retention

| Variable | Default | Description |
|----------|---------|-------------|
| `UV_RETENTION_ENABLED` | `false` | Enable automated cleanup |
| `UV_RETENTION_MAX_AGE_DAYS` | `90` | Max age for detection records |
| `UV_RETENTION_AUDIT_MAX_AGE_DAYS` | `180` | Max age for audit logs |
| `UV_RETENTION_CHECK_INTERVAL_HOURS` | `24` | Cleanup frequency |
| `UV_RETENTION_BATCH_SIZE` | `1000` | Rows deleted per batch |

### Pipeline Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `UV_INFERENCE_QUEUE_MAXSIZE` | `10` | Max frames queued |
| `UV_INFERENCE_QUEUE_HIGH_WATER` | `8` | Throttle threshold |
| `UV_INFERENCE_QUEUE_LOW_WATER` | `3` | Resume threshold |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `UV_LOG_LEVEL` | `INFO` | Log level |
| `UV_LOG_FORMAT` | `json` | Log format (`json` or `console`) |

---

## Appendix B — Complete Tool Registry

### Pipeline Management Tools (10)

| # | Tool | Parameters | Returns |
|---|------|-----------|---------|
| 1 | `query_detections` | camera_id?, plate_text?, from_time?, to_time?, limit? | List[DetectionRecord] |
| 2 | `get_detection_summary` | time_range? | {total, by_camera, by_status, by_engine} |
| 3 | `get_pipeline_stats` | — | Raw Prometheus metric snapshot |
| 4 | `get_system_health` | — | {pipeline_running, active_streams, queue_depth, vram_usage_mb} |
| 5 | `list_cameras` | — | List[CameraSource] |
| 6 | `manage_camera` | action, camera_id, source_url?, fps_target?, enabled? | Success/failure message |
| 7 | `adjust_threshold` | target, value | Confirmation with old/new values |
| 8 | `get_current_config` | — | Runtime config snapshot |
| 9 | `search_audit_log` | camera_id?, failure_reason?, limit? | List[AuditEntry] |
| 10 | `analyze_plate_patterns` | — | Pattern frequency analysis |

### Pipeline Control Tools (6)

| # | Tool | Parameters | Returns |
|---|------|-----------|---------|
| 11 | `auto_tune_confidence` | — | AI-optimized threshold recommendations |
| 12 | `get_stage_analytics` | — | Per-stage latency/throughput metrics |
| 13 | `get_ocr_strategy_stats` | — | Primary vs fallback usage breakdown |
| 14 | `self_heal_pipeline` | — | Diagnostic report + remediation actions taken |
| 15 | `get_queue_pressure` | — | {current_depth, high_water, low_water, is_throttled} |
| 16 | `flush_inference_queue` | — | Frames flushed count |

### Knowledge & Analytics Tools (7)

| # | Tool | Parameters | Returns |
|---|------|-----------|---------|
| 17 | `get_knowledge_stats` | — | {plates_tracked, cameras_profiled, feedback_count} |
| 18 | `get_frequent_plates` | top_n? | List[(plate_text, count)] |
| 19 | `get_camera_error_profile` | camera_id | {total_errors, ocr_failures, detection_failures, ...} |
| 20 | `get_all_camera_profiles` | — | Dict[camera_id → CameraErrorProfile] |
| 21 | `get_cross_camera_plates` | — | List[(plate_text, [camera_ids])] |
| 22 | `get_ocr_error_patterns` | — | List[(pattern, frequency)] |
| 23 | `detect_plate_anomalies` | — | List[AnomalyReport] |

### Additional Tools (7)

| # | Tool | Parameters | Returns |
|---|------|-----------|---------|
| 24 | `record_plate_feedback` | plate_text, feedback_type, corrected_text? | Confirmation |
| 25 | `get_recent_feedback` | limit? | List[FeedbackEntry] |
| 26 | `get_camera_hints` | camera_id | ML-derived camera positioning hints |
| 27 | `save_knowledge` | — | Knowledge base snapshot saved to DB |
| 28 | `diagnose_camera` | camera_id | Deep diagnostic report |
| 29 | `reset_circuit_breaker` | — | Circuit breaker state reset confirmation |
| 30 | `run_analytics_query` | query | Custom analytics query results |

---

## Appendix C — Tensor Format Specifications

| Stage | Tensor Shape | dtype | Domain | Size (typical) |
|-------|-------------|-------|--------|---------------|
| S0 output (raw frame) | (1080, 1920, 3) | uint8 | CPU | 6.2 MB |
| S2 input (YOLO) | (1, 3, 640, 640) | float32 | GPU | 4.9 MB |
| S2 output (bboxes) | (N, 6) | float32 | GPU | < 1 KB |
| S3 input (vehicle ROI) | (H, W, 3) variable | float32 | GPU | 0.5–2 MB |
| S3 output (plate bbox) | (M, 6) | float32 | GPU | < 1 KB |
| S4 output (plate crop) | (~200, ~600, 3) | uint8 | CPU | 0.36 MB |
| S5 output (deskewed) | (~200, ~600, 3) | uint8 | CPU | 0.36 MB |
| S6 output (enhanced) | (~200, ~600, 3) | uint8 | CPU | 0.36 MB |
| S7 input (base64 PNG) | string | — | HTTP | 5–50 KB |
| S7 output (OCR result) | JSON | — | HTTP | < 2 KB |

### System Memory (RAM) Budget

| Component | Estimated RAM |
|-----------|--------------|
| Python process + libraries | 400–600 MB |
| Frame buffer queue (50 frames) | 200–400 MB |
| Image preprocessing tensors | 100–200 MB |
| PostgreSQL connection pool | 50–100 MB |
| Redis client buffers | 50 MB |
| Ollama process (CPU-side) | 200–400 MB |
| **Total** | **1.0–1.8 GB** |

---

*Document generated from codebase analysis. Uni_Vision v0.1.0 — 288 tests passed, 21 skipped. All 28 development phases complete.*
