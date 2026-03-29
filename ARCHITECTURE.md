# Uni_Vision — System Architecture

> **Date:** 2026-03-26 · **Hardware Target:** NVIDIA RTX 4070 (8 GB VRAM) · **LLM:** Qwen 3.5 9B Q4_K_M via Ollama

---

## 1  High-Level System Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│                             RTSP Camera Network                          │
│            cam-1 ─────┐   cam-2 ─────┐   cam-N ─────┐                   │
└────────────────────────┼──────────────┼──────────────┼───────────────────┘
                         │              │              │
                         ▼              ▼              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         INGESTION LAYER  (CPU only)                       │
│                                                                           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                  │
│  │RTSPFrameSource│  │RTSPFrameSource│  │RTSPFrameSource│  ← bg threads   │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                  │
│         │ ring buf (50)    │ ring buf (50)    │ ring buf (50)             │
│         └─────────────┬────┴──────────────────┘                          │
│                       ▼                                                   │
│            ┌─────────────────────┐                                        │
│            │  TemporalSampler    │  FPS gate + 64-bit pHash dedup        │
│            │  (<0.3 ms / frame)  │  Discards 60-90% of static frames     │
│            └──────────┬──────────┘                                        │
└───────────────────────┼──────────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE QUEUE  (bounded: 10)                      │
│              Adaptive throttle: high_water=8, low_water=3                │
│              Backpressure → FPS × 0.5 on all sources                    │
└───────────────────────┬──────────────────────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
┌──────────────────────┐   ┌─────────────────────────────────────────────┐
│   LEGACY PIPELINE    │   │         MANAGER AGENT PIPELINE              │
│   (Fixed S2 → S8)    │   │     (Dynamic, Self-Assembling)             │
│                      │   │                                             │
│ S2  VehicleDetector  │   │  ContextAnalyzer → SceneTransition         │
│ S3  PlateDetector    │   │         ↓                                   │
│ S4  ROI Extraction   │   │  ComponentResolver → HubClient             │
│ ── VRAM Fence ──     │   │         ↓                                   │
│ S5  HoughStraightener│   │  PipelineComposer → ConflictResolver       │
│ S6  PhotometricEnhance│  │         ↓                                   │
│ S7  OCRStrategy      │   │  LifecycleManager → Execute Blueprint      │
│ S8  CognitiveOrch    │   │         ↓                                   │
│                      │   │  FeedbackLoop → AdaptationEngine           │
└──────────┬───────────┘   └─────────────────────┬───────────────────────┘
           │                                     │
           └──────────────┬──────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                       DISPATCH LAYER  (async)                            │
│                                                                          │
│  ┌──────────────────┐                                                    │
│  │ MultiDispatcher   │                                                   │
│  │ ├─ SlidingWindowDeduplicator (10 s per camera×plate)                 │
│  │ ├─ ObjectStoreArchiver ──────► MinIO / S3  (plate images)            │
│  │ ├─ PostgresClient ──────────► PostgreSQL   (detection records)       │
│  │ ├─ Redis PUBLISH ───────────► uv:events    (WebSocket broadcast)     │
│  │ └─ AuditLog ────────────────► ocr_audit_log (low-conf / failures)   │
│  └──────────────────┘                                                    │
└───────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        API & PRESENTATION                                │
│                                                                          │
│  FastAPI  ──  REST (12 endpoints) + WebSocket (2 endpoints)              │
│  Middleware: RequestLogging → APIKey → RateLimit → SecurityHeaders       │
│  Streamlit Visualiser (8 debug pages)                                    │
│                                                                          │
│  ┌────────────┐  ┌───────────────────┐  ┌──────────────────────────┐    │
│  │ /health    │  │ /ws/events        │  │ /api/agent/chat          │    │
│  │ /detections│  │   (plate stream)  │  │   (ReAct conversations)  │    │
│  │ /sources   │  │ /ws/agent         │  │ /api/agent/status        │    │
│  │ /stats     │  │   (reasoning)     │  │ /api/agent/sessions      │    │
│  │ /metrics   │  │                   │  │ /api/agent/feedback      │    │
│  └────────────┘  └───────────────────┘  └──────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2  VRAM Region Layout

The entire system operates within a hard **8192 MB ceiling**. Memory is partitioned into four named regions to prevent cross-subsystem contention:

```
 0 MB                                                          8192 MB
  ├──────────── Region A ──────────────┤─ B ─┤─── C ───┤─ D ─┤ Headroom ┤
  │      LLM Weights (5120 MB)        │512 MB│1024 MB  │512MB│ 1024 MB  │
  │      Qwen 3.5 9B Q4_K_M          │ KV   │ CV      │CUDA │ Safety   │
  │                                    │cache │ models  │ drvr│ margin   │
  └────────────────────────────────────┴──────┴─────────┴─────┴──────────┘
```

| Region | Size | Contents | Managed By |
|--------|------|----------|-----------|
| **A — LLM Weights** | 5120 MB | Qwen 3.5 9B Q4_K_M parameters | Ollama (never evicted) |
| **B — KV Cache** | 512 MB | 4096-token sliding window | Ollama (grows/shrinks per request) |
| **C — Vision** | 1024 MB | YOLO, OCR, downloaded models | `LifecycleManager` + `GPUMemoryManager` |
| **D — System** | 512 MB | CUDA runtime, driver allocations | OS/driver (uncontrollable) |
| **Headroom** | 1024 MB | Fragmentation buffer (12.5%) | Untouched; alert if encroached |

**Memory Fence:** A hard `torch.cuda.synchronize()` + `empty_cache()` barrier separates every GPU transition between vision inference and LLM calls, preventing asynchronous CUDA kernels from colliding.

**Dynamic Offload Tiers** (managed by `GPUMemoryManager`):

| Free VRAM | Mode | Behaviour |
|-----------|------|-----------|
| ≥ 1024 MB | `GPU_PRIMARY` | All CV models on GPU |
| ≥ 512 MB | `PARTIAL_OFFLOAD` | Vehicle detector on GPU, plate detector on CPU |
| < 512 MB | `FULL_CPU` | All CV inference on CPU (emergency) |

---

## 3  Package Architecture

### 3.1  Package Dependency DAG

```
                            ┌───────────┐
                            │  common/  │  config, logging, exceptions
                            └─────┬─────┘
                                  │ imported by everything
                    ┌─────────────┼─────────────┐
                    │             │             │
               ┌────┴────┐  ┌────┴────┐  ┌────┴──────┐
               │contracts│  │components│  │monitoring │
               │ (DTOs)  │  │ (ABC)   │  │ (metrics) │
               └────┬────┘  └────┬────┘  └────┬──────┘
                    │            │             │
      ┌─────────┬──┴──┬────────┬┴─────┬───────┘
      │         │     │        │      │
 ┌────┴───┐ ┌──┴──┐ ┌┴─────┐ ┌┴────┐ │
 │ingestion│ │detec│ │preproc│ │ ocr │ │
 └────┬────┘ └──┬──┘ └──┬───┘ └──┬──┘ │
      │         │       │        │    │
      │    ┌────┴───────┴────────┴────┘
      │    │
      │ ┌──┴──────────┐
      │ │postprocessing│  validator, adjudicator, dedup, dispatcher, orchestrator
      │ └──────┬───────┘
      │        │
      │ ┌──────┴──┐
      │ │ storage │  PostgreSQL, S3, retention
      │ └──────┬──┘
      │        │
      ├────────┘
      │
 ┌────┴──────────────────────────────────────────────┐
 │                orchestrator/                       │
 │   pipeline.py  (frame loop, queue, GPU exclusion)  │
 │   container.py (DI wiring for 17+ subsystems)      │
 └────┬──────────────────────────────────────────────┘
      │ optional wiring
      │
 ┌────┴──────────────────────────────────────────────┐
 │                manager/  (18 subsystems)           │
 │   Self-assembling pipeline orchestrator            │
 │   Internet-aware component provisioning            │
 └────┬──────────────────────────────────────────────┘
      │
 ┌────┴──────────────────────────────────────────────┐
 │                agent/  (16 files)                  │
 │   ReAct loop, 37 tools, sessions, knowledge        │
 └────┬──────────────────────────────────────────────┘
      │
 ┌────┴──────────────────────────────────────────────┐
 │                api/  (routes + middleware)          │
 │   FastAPI app, REST + WebSocket + middleware stack  │
 └───────────────────────────────────────────────────┘
```

### 3.2  Package Inventory

| Package | Files | Purpose | GPU? |
|---------|-------|---------|------|
| `common/` | 4 | Config (27+ Pydantic models), logging, exceptions | No |
| `contracts/` | 6 | DTOs (`FramePacket` → `DetectionRecord`), protocols, enums | No |
| `components/` | 3 | `CVComponent` ABC, `ComponentMetadata`, 4 wrapper adapters | Varies |
| `ingestion/` | 3 | RTSP source, temporal sampler, pHash dedup | No |
| `detection/` | 4 | YOLO detectors, TensorRT/ONNX engine, GPU memory manager | Yes |
| `preprocessing/` | 4 | Deskew (Hough), enhance (CLAHE), chain, ROI extract | No |
| `ocr/` | 4 | Multi-engine OCR strategy, EasyOCR fallback, adapter, parser | Mixed |
| `postprocessing/` | 5 | Validator, adjudicator, dedup, dispatcher, orchestrator | No |
| `storage/` | 4 | asyncpg client, aioboto3 archiver, DDL models, retention | No |
| `monitoring/` | 5 | VRAM monitor, Prometheus metrics, health, profiler, budget | No |
| `orchestrator/` | 2 | Pipeline runner, DI container | Orchestrates |
| `manager/` | 18 | Self-assembling pipeline: all 18 subsystems | Orchestrates |
| `agent/` | 16 | ReAct loop, 37 tools, sessions, knowledge, audit | No |
| `api/` | 12 | FastAPI routes (8 files), middleware (4 files) | No |
| `visualizer/` | 8 | Streamlit debug pages | No |

---

## 4  Data Model Lifecycle

Every frame traverses a well-defined DTO chain:

```
                    Camera RTSP stream
                          │
                          ▼
                  ┌───────────────┐
                  │  FramePacket  │  camera_id, timestamp_utc, frame_index, image (BGR ndarray)
                  └───────┬───────┘
                          │ Detection
                          ▼
                  ┌───────────────┐
                  │  BoundingBox  │  x1, y1, x2, y2, confidence, class_id, class_name
                  │  (vehicles)   │  ← List per frame
                  └───────┬───────┘
                          │ Plate localisation
                          ▼
                  ┌───────────────┐
                  │  BoundingBox  │  Same struct, within vehicle ROI
                  │  (plates)     │
                  └───────┬───────┘
                          │ Context assembly
                          ▼
                  ┌──────────────────┐
                  │ DetectionContext  │  camera_id, timestamp, vehicle_bbox, plate_bbox, vehicle_class
                  └───────┬──────────┘
                          │ OCR
                          ▼
                  ┌───────────────┐
                  │   OCRResult   │  plate_text, raw_text, confidence, engine, status
                  └───────┬───────┘
                          │ Validation + Adjudication
                          ▼
                  ┌──────────────────┐
                  │ ProcessedResult  │  plate_text, validation_status, char_corrections
                  └───────┬──────────┘
                          │ Persistence
                          ▼
                  ┌──────────────────┐
                  │ DetectionRecord  │  UUID, all fields, image paths, location_tag
                  └──────────────────┘  → PostgreSQL + S3 + Redis pub/sub
```

### Enum Transitions

```
ValidationStatus flow:
  OCR output → "valid"           (regex passes, confidence ≥ threshold)
             → "low_confidence"  (regex passes, confidence < threshold)
             → "regex_fail"      (char-corrected text still fails regex)
             → "fallback"        (primary failed, EasyOCR succeeded)
             → "unreadable"      (all engines failed)

ComponentState flow:
  UNREGISTERED → REGISTERED → DOWNLOADING → LOADING → READY
                                                           ↓
                                              UNLOADING ← ─┘
                                                  ↓
                                              REGISTERED (cycle)
                                              or FAILED / SUSPENDED
```

---

## 5  Manager Agent — Internal Architecture

### 5.1  Subsystem Interaction Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ManagerAgent                                    │
│                   process_frame(frame) → PipelineExecutionResult        │
│                                                                         │
│  ┌─────────────────┐    ┌───────────────────┐    ┌──────────────────┐  │
│  │ ContextAnalyzer  │───►│ SceneTransition    │    │ TemporalTracker  │  │
│  │ (heuristic/LLM)  │   │ Detector           │◄──►│ (object tracks,  │  │
│  │ → FrameContext    │   │ (hysteresis FSM)   │    │  env trends)     │  │
│  └────────┬─────────┘   └─────────┬──────────┘    └───────┬──────────┘  │
│           │                       │   scene change?        │ hints      │
│           ▼                       ▼                        ▼            │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Blueprint Cache (LRU)                           │ │
│  │              keyed by frozenset(capabilities) hash                 │ │
│  └────────────────────────────┬───────────────────────────────────────┘ │
│                               │ miss                                    │
│                               ▼                                         │
│  ┌──────────────────┐    ┌───────────────┐    ┌───────────────────┐    │
│  │ ComponentResolver │───►│   HubClient   │    │ ComponentRegistry │    │
│  │ (resolve caps)    │   │ (HuggingFace, │◄──►│ (thread-safe,     │    │
│  │                   │◄──│  PyPI, Torch)  │    │  capability-idx)  │    │
│  └────────┬──────────┘   └───────────────┘    └───────────────────┘    │
│           │ resolved components                                         │
│           ▼                                                             │
│  ┌──────────────────┐    ┌──────────────────┐                          │
│  │ PipelineComposer  │───►│ ConflictResolver │                          │
│  │ (stage ordering,  │   │ (VRAM budget,    │                          │
│  │  IO key wiring)   │   │  dep clashes,    │                          │
│  │ → Blueprint       │   │  cap overlaps)   │                          │
│  └────────┬──────────┘   └──────────────────┘                          │
│           │ validated blueprint                                         │
│           ▼                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │ LifecycleManager  │   │ GPUProfiler      │    │ Compatibility    │  │
│  │ (load/unload/swap │◄─►│ (measure VRAM,   │    │ Matrix           │  │
│  │  LRU eviction,    │   │  detect leaks)   │    │ (pairwise check, │  │
│  │  CPU fallback)    │   └──────────────────┘    │  empirical learn)│  │
│  └────────┬──────────┘                           └──────────────────┘  │
│           │ all components READY                                        │
│           ▼                                                             │
│  ┌──────────────────────────────────────────────┐                      │
│  │ Execute Blueprint (sequential stage runner)   │                      │
│  │ Stage₁.execute() → Stage₂.execute() → ...    │                      │
│  │ IO keys: frame → detections → plate_crops →   │                      │
│  │          ocr_results → validated_results       │                      │
│  └────────┬─────────────────────────────────────┘                      │
│           │ PipelineExecutionResult                                     │
│           ▼                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  FeedbackLoop     │───►│ AdaptationEngine │    │ QualityScorer    │  │
│  │  (EWMA per-comp:  │   │ (8 signal types, │◄──►│ (Beta-Binomial,  │  │
│  │   latency, conf,  │   │  4 internal checks│   │  4 sub-scores)   │  │
│  │   reliability)    │   │  → swap/add/remove│    └──────────────────┘  │
│  └──────────────────┘   │    /recompose)     │    ┌──────────────────┐  │
│                          └──────────────────┘    │ FallbackChain    │  │
│                                                   │ Manager          │  │
│                                                   │ (4 tiers per cap,│  │
│                                                   │  auto-recovery)  │  │
│                                                   └──────────────────┘  │
│  ┌──────────────────┐                                                   │
│  │ PipelineValidator │  Pre-execution: components exist? IO chain?      │
│  │                   │  VRAM fits? Non-empty? Optional dry_run().       │
│  └──────────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2  Internet Component Discovery Flow

```
missing capability (e.g. SCENE_TEXT_OCR)
    │
    ▼
ComponentResolver.resolve_capabilities()
    │
    ├──► Check ComponentRegistry (local)
    │    └── not found
    │
    ├──► HubClient.search_huggingface(
    │        task="image-to-text",
    │        sort="downloads",
    │        limit=10)
    │    │
    │    ├── Filter by TRUSTED_REPOS (15 orgs):
    │    │   ultralytics, google, microsoft,
    │    │   IDEA-Research, PaddlePaddle, facebook,
    │    │   hustvl, nvidia, intel, openai, ...
    │    │
    │    ├── Filter by _LIBRARY_LOAD_PATTERNS (11 libs):
    │    │   transformers, ultralytics, timm, diffusers,
    │    │   open_clip, sentence-transformers, paddlepaddle,
    │    │   detectron2, yolov5, keras, pytorch
    │    │
    │    └── Score: popularity × vram_fit × trust_bonus
    │
    ├──► HubClient.search_pypi()
    │    └── _KNOWN_PIP_PACKAGES: PaddleOCR, DeepSort,
    │        timm, Kornia, Albumentations, OpenCLIP
    │
    └──► Select best candidate
         │
         ▼
    provision_candidate()
    ├── pip install --no-deps (if needed)
    ├── Download model weights (httpx, 5 min timeout, 3 retries)
    ├── Wrap as CVComponent:
    │   ├── HuggingFaceModelComponent (from_pretrained)
    │   ├── PipPackageComponent (library init)
    │   └── TorchHubComponent (torch.hub.load)
    ├── Register in ComponentRegistry
    └── Return ResolutionResult{status=INSTALLED}
```

### 5.3  Adaptation & Self-Healing Cycle

```
                    PipelineExecutionResult
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      FeedbackLoop    QualityScorer    TemporalTracker
      (record EWMA)  (update scores)  (update tracks)
            │               │               │
            └───────────────┼───────────────┘
                            ▼
                    AdaptationEngine
                    .ingest_result()
                            │
                    ┌───────┴───────┐
                    │ 4 Checks:     │
                    │ 1. Scene drift│  dominant scene changed → RECOMPOSE
                    │ 2. Quality ↓  │  avg confidence < 0.4  → SWAP component
                    │ 3. Latency ↑  │  P95 > latency budget  → DOWNGRADE model
                    │ 4. Error rate │  error rate > 0.3      → REMOVE + fallback
                    └───────┬───────┘
                            │ AdaptationAction
                            ▼
                    ┌───────────────────┐
                    │ Cooldown Guard     │  Prevent action storms
                    │ (per-action timer) │  (60 s default)
                    └───────┬───────────┘
                            │ if not cooled
                            ▼
                    ManagerAgent applies:
                    ├── SWAP: unload old → load new via LifecycleManager
                    ├── ADD: resolve + load additional component
                    ├── REMOVE: unload poor-performing component
                    ├── DOWNGRADE: replace with lighter model
                    └── RECOMPOSE: full blueprint rebuild
```

---

## 6  ReAct Agent Architecture

### 6.1  Message Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     AgentCoordinator                         │
│                                                              │
│   ┌──────────────────┐                                       │
│   │ IntentClassifier  │  8 intents via regex scoring         │
│   │ (zero LLM calls)  │  STATUS, DETECTION, ANALYTICS,      │
│   └────────┬─────────┘  CAMERA, CONFIG, KNOWLEDGE,          │
│            │             DIAGNOSTICS, GENERAL                 │
│            ▼                                                  │
│   ┌──────────────────┐                                       │
│   │ MultiAgentRouter  │  Routes to specialised role:         │
│   └────────┬─────────┘                                       │
│            │                                                  │
│   ┌────────┼─────────┬──────────────┬──────────────┐        │
│   ▼        ▼         ▼              ▼              ▼        │
│ OCR_QUALITY  ANALYTICS  OPERATIONS   GENERAL                │
│ (12 tools)  (10 tools) (13 tools)   (37 tools)              │
│                                                              │
│   Each sub-agent gets:                                       │
│   ├── Curated ToolRegistry subset                            │
│   ├── Role-specific system prompt                            │
│   └── Shared WorkingMemory + SessionManager                 │
└────────────────────┬─────────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                       AgentLoop                              │
│                    (ReAct Engine)                             │
│                                                              │
│   Iteration 1:                                               │
│   ├─ LLM (Ollama /api/chat, think=false)                    │
│   ├─ Parse JSON: {action, action_input} or {answer}          │
│   ├─ If action: ToolRegistry.invoke(action, action_input)    │
│   ├─ Format Observation                                      │
│   └─ Append to conversation                                  │
│                                                              │
│   Iteration 2-10:                                            │
│   ├─ LLM sees: thought + action + observation history        │
│   ├─ Either: deeper tool call, or final {answer}             │
│   └─ Guard: max_iterations = 10                              │
│                                                              │
│   Output: ChatResponse (answer text, steps, elapsed_ms)      │
└──────────────────────────────────────────────────────────────┘
```

### 6.2  Tool Categories

```
┌─────────────────────────────────────────────────────┐
│                   37 Registered Tools                │
│                                                     │
│  ┌─ Pipeline Tools (11) ──────────────────────────┐ │
│  │ query_detections      get_detection_summary     │ │
│  │ get_pipeline_stats    get_system_health         │ │
│  │ list_cameras          manage_camera             │ │
│  │ adjust_threshold      get_current_config        │ │
│  │ search_audit_log      analyze_plate_patterns    │ │
│  │ diagnose_camera       run_analytics_query       │ │
│  └─────────────────────────────────────────────────┘ │
│                                                     │
│  ┌─ Knowledge Tools (10) ─────────────────────────┐ │
│  │ get_knowledge_stats      get_frequent_plates    │ │
│  │ get_camera_error_profile get_all_camera_profiles│ │
│  │ get_cross_camera_plates  get_ocr_error_patterns │ │
│  │ detect_plate_anomalies   record_plate_feedback  │ │
│  │ get_recent_feedback      save_knowledge         │ │
│  └─────────────────────────────────────────────────┘ │
│                                                     │
│  ┌─ Control Tools (6) ────────────────────────────┐ │
│  │ auto_tune_confidence  get_stage_analytics       │ │
│  │ get_ocr_strategy_stats self_heal_pipeline       │ │
│  │ get_queue_pressure    flush_inference_queue      │ │
│  └─────────────────────────────────────────────────┘ │
│                                                     │
│  ┌─ Session/Feedback Tools (via coordinator) ─────┐ │
│  │ Session CRUD, feedback recording, KB persist    │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 6.3  Supporting Subsystems

```
┌─────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐
│   WorkingMemory      │  │  SessionManager    │  │   KnowledgeBase      │
│  3072-token FIFO     │  │  TTL: 30 min       │  │  Plate frequency     │
│  System prompt pinned│  │  Max: 100 sessions │  │  Camera error profs  │
│  Scratchpad KV store │  │  LRU eviction      │  │  Cross-camera correl │
└─────────────────────┘  └────────────────────┘  │  Feedback store       │
                                                  │  → PostgreSQL persist │
┌─────────────────────┐  ┌────────────────────┐  └──────────────────────┘
│   AuditTrail         │  │AutonomousMonitor   │
│  Write-behind buffer │  │  60 s health loop  │
│  Batch flush (50)    │  │  VRAM > 90%? warn  │
│  → agent_audit_log   │  │  Queue > 80%? flush│
└─────────────────────┘  │  OCR < 50%? alert   │
                          │  Auto-remediation   │
                          └────────────────────┘
```

---

## 7  Postprocessing Decision Tree

```
                        OCRResult (raw)
                            │
                            ▼
                ┌───────────────────────┐
                │ DeterministicValidator │
                │ (sub-millisecond)      │
                └───────────┬───────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
          ACCEPTED      CORRECTED     REGEX_FAIL / LOW_CONFIDENCE
          (done)    (char-corrected,  │
                     regex now passes) │
                                       ▼
                           ┌───────────────────────┐
                           │ ConsensusAdjudicator   │
                           │ (multi-engine vote)    │
                           └───────────┬───────────┘
                                       │
                          ┌────────────┼────────────┐
                          │            │            │
                          ▼            ▼            ▼
                     CONSENSUS     NO CONSENSUS  < 2 ENGINES
                     (≥2 agree)    (fallback)    (skip adjud.)
                          │            │            │
                          ▼            ▼            ▼
                     ProcessedResult with final validation_status:
                     valid | low_confidence | regex_fail | fallback | unreadable

    Character Correction Matrix (positional):
    ┌────────┬────────┬─────────────────────────────┐
    │ From   │ To     │ When                        │
    ├────────┼────────┼─────────────────────────────┤
    │ O      │ 0      │ In digit slot               │
    │ 0      │ O      │ In alpha slot               │
    │ I      │ 1      │ In digit slot               │
    │ 1      │ I      │ In alpha slot               │
    │ S      │ 5      │ In digit slot               │
    │ 5      │ S      │ In alpha slot               │
    │ B      │ 8      │ In digit slot               │
    │ 8      │ B      │ In alpha slot               │
    │ D      │ 0      │ In digit slot               │
    │ Z      │ 2      │ In digit slot               │
    │ G      │ 6      │ In digit slot               │
    │ T      │ 7      │ In digit slot               │
    │ A      │ 4      │ In digit slot               │
    └────────┴────────┴─────────────────────────────┘
```

---

## 8  Infrastructure Architecture

### 8.1  Docker Compose Service Graph

```
                          ┌──────────────┐
                          │  ollama-init │
                          │  (curl)      │
                          │ Pull Qwen,   │
                          │ create       │
                          │ Modelfiles   │
                          └──────┬───────┘
                                 │ depends_on
                                 ▼
┌───────────┐  depends   ┌──────────────┐
│  grafana  │◄───────────│   ollama     │  GPU reservation
│  :3000    │            │  :11434      │  Volume: ollama_data
│ dashboard │            │  Qwen 3.5   │
│ provision │            └──────────────┘
└─────┬─────┘                    │
      │ datasource               │
      ▼                          │
┌───────────┐                    │
│prometheus │  15s scrape        │
│  :9090    │  10 alert rules    │
└─────┬─────┘                    │
      │ scrapes /metrics         │
      ▼                          ▼
┌──────────────────────────────────────────┐
│                  app                      │
│   nvidia/cuda:12.4.1-runtime-ubuntu22.04 │
│   FastAPI + Uvicorn                      │
│   GPU reservation (all)                  │
│   Healthcheck: curl /health every 30s    │
│   Non-root user (uid 1000)               │
└──────┬───────────┬────────────┬──────────┘
       │           │            │
       ▼           ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ postgres │ │  minio   │ │  redis   │
│  :5432   │ │  :9000   │ │  :6379   │
│ pgdata   │ │minio_data│ │redis_data│
│ 4 tables │ │ plates/  │ │ pub/sub  │
│ pg_trgm  │ │ bucket   │ │uv:events │
└──────────┘ └──────────┘ └──────────┘
```

### 8.2  Database Schema

```
┌──────────────────────────────────────────────────┐
│               detection_events                    │
├──────────────────────────────────────────────────┤
│ id               UUID PK                          │
│ camera_id        VARCHAR    ┐                     │
│ plate_number     VARCHAR    │ idx: (camera, ts)   │
│ raw_ocr_text     VARCHAR    │ idx: (plate, ts)    │
│ ocr_confidence   FLOAT      │ idx: trgm(plate)    │
│ ocr_engine       VARCHAR    │                     │
│ vehicle_class    VARCHAR    │                     │
│ vehicle_image_path VARCHAR  │                     │
│ plate_image_path VARCHAR    │                     │
│ detected_at_utc  TIMESTAMPTZ┘                     │
│ validation_status VARCHAR                         │
│ location_tag     VARCHAR                          │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│                camera_sources                     │
├──────────────────────────────────────────────────┤
│ camera_id        VARCHAR PK                       │
│ url              VARCHAR                          │
│ label            VARCHAR                          │
│ fps_target       FLOAT                            │
│ enabled          BOOLEAN                          │
│ created_at       TIMESTAMPTZ                      │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│                ocr_audit_log                      │
├──────────────────────────────────────────────────┤
│ id               SERIAL PK                        │
│ record_id        UUID FK → detection_events       │
│ camera_id        VARCHAR                          │
│ raw_ocr_text     VARCHAR                          │
│ ocr_confidence   FLOAT                            │
│ failure_reason   VARCHAR                          │
│ frame_path       VARCHAR                          │
│ created_at       TIMESTAMPTZ                      │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│              agent_knowledge                      │
├──────────────────────────────────────────────────┤
│ key              VARCHAR PK                       │
│ value            JSONB                            │
│ updated_at       TIMESTAMPTZ                      │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│              agent_audit_log                      │
├──────────────────────────────────────────────────┤
│ id               SERIAL PK                        │
│ session_id       VARCHAR                          │
│ intent           VARCHAR                          │
│ agent_role       VARCHAR                          │
│ action           VARCHAR                          │
│ arguments        JSONB                            │
│ result           TEXT                             │
│ success          BOOLEAN                          │
│ elapsed_ms       FLOAT                            │
│ created_at       TIMESTAMPTZ                      │
└──────────────────────────────────────────────────┘
```

### 8.3  Observability Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        Prometheus Metrics (14)                  │
│                                                                 │
│  Counters:                        Histograms:                   │
│  ├─ uv_frames_ingested_total      ├─ uv_pipeline_latency_sec   │
│  ├─ uv_frames_deduplicated_total  ├─ uv_stage_latency_sec      │
│  ├─ uv_frames_dropped_total       ├─ uv_ocr_confidence         │
│  ├─ uv_detections_total           ├─ uv_agent_latency_sec      │
│  ├─ uv_ocr_requests_total         └─ uv_agent_steps            │
│  ├─ uv_ocr_success_total                                       │
│  ├─ uv_ocr_fallback_total         Gauges:                      │
│  ├─ uv_detections_dedup_total     ├─ uv_vram_usage_bytes       │
│  ├─ uv_dispatch_success_total     ├─ uv_inference_queue_depth  │
│  ├─ uv_dispatch_errors_total      └─ uv_stream_status          │
│  ├─ uv_agent_requests_total                                    │
│  └─ uv_agent_tool_calls_total                                  │
│                                                                 │
│  10 Alert Rules:                                                │
│  VRAMCritical(>90%) | FrameDropHigh(>10%) | OCRLow(<50%)       │
│  LatencyHigh(P95>5s) | QueueBacklog(>8) | DispatchErrors(>0)    │
│  StreamDown(1m) | DBFailed(1m) | OllamaDown(2m) | Fallback(>40%)│
└─────────────────────────────────────────────────────────────────┘
```

---

## 9  API Architecture

### 9.1  Middleware Pipeline

```
Incoming HTTP Request
    │
    ▼
┌────────────────────────┐
│ SecurityHeadersMiddleware│  X-Content-Type-Options, X-Frame-Options,
│                          │  X-XSS-Protection, Referrer-Policy, Cache-Control
└────────────┬─────────────┘
             ▼
┌────────────────────────┐
│ RateLimitMiddleware     │  Sliding window per IP (default: 60 req/min)
│                          │  Exempt: /health, /metrics
│                          │  Returns 429 when exceeded
└────────────┬─────────────┘
             ▼
┌────────────────────────┐
│ APIKeyMiddleware        │  SHA-256 hashed key validation
│                          │  Header: X-API-Key
│                          │  Exempt: /health, /metrics, /ws/
└────────────┬─────────────┘
             ▼
┌────────────────────────┐
│ RequestLoggingMiddleware│  Logs: method, path, status, latency_ms
└────────────┬─────────────┘
             ▼
        Route Handler
```

### 9.2  Endpoint Map

```
REST Endpoints:
    GET  /health ─────────────────── Deep health (GPU, Ollama, DB, streams)
    GET  /detections ─────────────── Paginated query (camera, plate, time, status)
    GET  /stats ──────────────────── Prometheus metrics as JSON
    GET  /metrics ────────────────── Prometheus text format (scraped by Prometheus)
    POST /sources ────────────────── Add camera source
    GET  /sources ────────────────── List camera sources
    DEL  /sources/{camera_id} ────── Remove camera source

Agent Endpoints:
    POST /api/agent/chat ─────────── Send message → ReAct → response
    GET  /api/agent/status ───────── Uptime, active sessions, tool count
    POST /api/agent/feedback ─────── Submit operator plate correction
    GET  /api/agent/sessions ─────── List active sessions
    DEL  /api/agent/sessions/{id} ── Delete session

WebSocket Endpoints:
    WS  /ws/events ───────────────── Plate events (Redis pub/sub → client)
    WS  /ws/agent ────────────────── Streaming reasoning steps:
                                      thought → tool_call → observation →
                                      answer → done
```

---

## 10  Concurrency Model

```
┌──────────────────────────────────────────────────────────────┐
│                      asyncio Event Loop                      │
│                      (single-threaded)                       │
│                                                              │
│  Async Tasks:                                                │
│  ├─ Pipeline._run_loop()        ← inference consumer         │
│  ├─ VRAMMonitor._poll_loop()    ← 500 ms GPU telemetry      │
│  ├─ AutonomousMonitor._tick()   ← 60 s health check         │
│  ├─ RetentionTask._purge()      ← periodic data cleanup     │
│  ├─ SlidingWindowDedup._purge() ← 30 s expired entry sweep  │
│  └─ MultiDispatcher._consumer() ← async multi-target write  │
│                                                              │
│  Thread Pool:                                                │
│  ├─ RTSPFrameSource._reader()   ← N camera bg threads       │
│  ├─ EasyOCRFallback._run()      ← single-thread executor    │
│  └─ ComponentRegistry (Lock)    ← thread-safe mutations      │
│                                                              │
│  GPU Exclusivity:                                            │
│  ├─ Single inference consumer   ← no parallel GPU calls      │
│  ├─ Memory fence between V/L    ← cuda.synchronize()        │
│  └─ Sequential blueprint exec   ← stage₁ → stage₂ → ...    │
└──────────────────────────────────────────────────────────────┘
```

---

## 11  Security Architecture

```
┌─────────── Network Boundary ──────────┐
│                                       │
│  External clients ──► API Gateway     │
│  ├─ API Key (SHA-256) authentication  │
│  ├─ Rate limiting (60 req/min/IP)     │
│  ├─ Security headers (XSS, CORS, etc)│
│  └─ Request logging (audit trail)     │
│                                       │
│  Internal services (Docker network):  │
│  ├─ PostgreSQL ── password auth       │
│  ├─ MinIO ── access key + secret key  │
│  ├─ Redis ── no auth (internal only)  │
│  ├─ Ollama ── no auth (internal only) │
│  └─ Prometheus/Grafana ── internal    │
│                                       │
│  Model Download Security:             │
│  ├─ 15 trusted HuggingFace repos     │
│  │   (Ultralytics, Google, Microsoft, │
│  │    Facebook, NVIDIA, Intel, ...)   │
│  ├─ pip install --no-deps (isolated)  │
│  ├─ No execution until explicit load()│
│  └─ 5-min timeout + 3 retries        │
│                                       │
│  Container Security:                  │
│  ├─ Non-root user (uid 1000)         │
│  ├─ Read-only model volumes           │
│  └─ GPU reservation (not shared)      │
└───────────────────────────────────────┘
```

---

## 12  Key Design Invariants

| # | Invariant | Enforcement |
|---|-----------|------------|
| 1 | **Qwen never sees images** | No image data in Ollama API calls; only text/JSON reasoning |
| 2 | **VRAM ≤ 8192 MB** | `VRAMBudgets` config, `LifecycleManager` LRU eviction, `GPUMemoryManager` offload tiers |
| 3 | **Single GPU consumer** | One inference task drains the queue; no parallel CUDA kernels |
| 4 | **Memory fence at vision↔LLM boundary** | `torch.cuda.synchronize()` + `empty_cache()` before every LLM call |
| 5 | **Dedup before GPU** | pHash at ingestion, sliding-window dedup at dispatch |
| 6 | **All IO is async** | asyncpg, aioboto3, httpx, Redis aioredis — zero blocking |
| 7 | **No arbitrary model execution** | Trusted repos only, `--no-deps`, explicit `load()` gate |
| 8 | **Adaptation has cooldowns** | Per-signal cooldown prevents action storms and oscillation |
| 9 | **Legacy pipeline always available** | Fixed S2-S8 path works with Manager disabled |
| 10 | **Every agent action is audited** | Write-behind flush to `agent_audit_log` table |

---

*For class-level and method-level detail, see [SYSTEM_WALKTHROUGH.md](SYSTEM_WALKTHROUGH.md). For deployment steps, see [DEPLOYMENT.md](DEPLOYMENT.md).*
