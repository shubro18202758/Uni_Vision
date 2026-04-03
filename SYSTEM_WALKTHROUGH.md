# Uni_Vision — Deep Technical Backend Walkthrough

> **Last Updated:** 2025-06-24  
> **Codebase Version:** Post-Phase 8 (Self-Assembling Autonomous Pipeline)  
> **Total Source Files:** ~135 Python files across 15 packages  
> **Test Suite:** 309 tests (288 pass, 21 skip)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware & VRAM Layout](#2-hardware--vram-layout)
3. [Package Map — All 15 Packages at a Glance](#3-package-map--all-15-packages-at-a-glance)
4. [Configuration System — `common/config.py`](#4-configuration-system--commonconfigpy)
5. [Core Data Contracts — DTOs, Schemas, Enums](#5-core-data-contracts--dtos-schemas-enums)
6. [Component Abstraction Layer — `components/`](#6-component-abstraction-layer--components)
7. [Ingestion Layer — `ingestion/`](#7-ingestion-layer--ingestion)
8. [Detection Layer — `detection/`](#8-detection-layer--detection)
9. [Preprocessing Layer — `preprocessing/`](#9-preprocessing-layer--preprocessing)
10. [OCR Layer — `ocr/`](#10-ocr-layer--ocr)
11. [Postprocessing Layer — `postprocessing/`](#11-postprocessing-layer--postprocessing)
12. [Storage Layer — `storage/`](#12-storage-layer--storage)
13. [Monitoring Layer — `monitoring/`](#13-monitoring-layer--monitoring)
14. [The Legacy Pipeline — `orchestrator/`](#14-the-legacy-pipeline--orchestrator)
15. [The Manager Agent — `manager/` (Self-Assembling Pipeline)](#15-the-manager-agent--manager-self-assembling-pipeline)
16. [The Agentic AI System — `agent/` (ReAct Loop)](#16-the-agentic-ai-system--agent-react-loop)
17. [API Layer — `api/`](#17-api-layer--api)
18. [Visualiser — `visualizer/`](#18-visualiser--visualizer)
19. [Infrastructure — Docker, Database, Monitoring Stack](#19-infrastructure--docker-database-monitoring-stack)
20. [Build & DevOps — Makefile, pyproject.toml, Scripts](#20-build--devops--makefile-pyprojecttoml-scripts)
21. [Data Flow — End-to-End Frame Journey](#21-data-flow--end-to-end-frame-journey)
22. [Key Architectural Decisions](#22-key-architectural-decisions)

---

## 1. System Overview

Uni_Vision is a **multipurpose intelligent detection and analysis** system designed for real-time deployment on a single **NVIDIA RTX 4070 (8 GB VRAM)** GPU. It ingests RTSP camera feeds, detects objects and anomalies, extracts text via OCR, validates and deduplicates readings, and persists results to PostgreSQL + S3-compatible object storage.

What makes this system unique is the **self-assembling pipeline architecture**: a **Gemma 4 E2B** large language model (running locally via Ollama) acts as a "Manager Agent brain" that analyzes each frame's scene context and dynamically provisions, downloads, and orchestrates computer-vision components from the internet (HuggingFace Hub, PyPI, TorchHub) — all within the 8 GB VRAM budget.

**Critical Architectural Note:** Gemma 4 E2B is a natively **multimodal** model (text + image + audio). It can process images directly via its built-in vision capability, unlike text-only models that require separate CV pipelines for image understanding. Dedicated CV models (YOLO, EasyOCR, TrOCR, PaddleOCR, etc.) are still used for specialized detection tasks where they outperform general-purpose LLM vision.

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python ≥ 3.10 |
| Web Framework | FastAPI + Uvicorn |
| LLM Runtime | Ollama (Gemma 4 E2B Q4_K_M) |
| CV Inference | TensorRT / ONNX Runtime / PyTorch |
| Database | PostgreSQL (asyncpg) |
| Object Storage | MinIO / AWS S3 (aioboto3) |
| Caching / PubSub | Redis |
| Monitoring | Prometheus + Grafana |
| Container | Docker Compose (8 services) |

---

## 2. Hardware & VRAM Layout

The entire system is designed for an **8192 MB VRAM ceiling** with the following region allocation:

```
┌─────────────────────────────────────────────────────────┐
│                   RTX 4070 — 8192 MB                    │
├──────────────────┬──────────────────────────────────────┤
│ Region A: LLM    │ 5000 MB — Gemma 4 E2B Q4_K_M weights│
├──────────────────┼──────────────────────────────────────┤
│ Region B: KV     │  512 MB — KV cache (4096 tokens)     │
├──────────────────┼──────────────────────────────────────┤
│ Region C: Vision │ 1024 MB — CV models workspace        │
├──────────────────┼──────────────────────────────────────┤
│ Region D: System │  512 MB — CUDA overhead + drivers     │
├──────────────────┼──────────────────────────────────────┤
│ Headroom         │ 1024 MB — Safety margin (12.5%)       │
└──────────────────┴──────────────────────────────────────┘
Total allocated: 7168 MB │ Headroom: 1024 MB
```

**Key Constraints:**
- LLM occupies ~5800 MB (weights + KV cache), leaving **~2392 MB** for CV models
- YOLOv8n INT8 per-model footprint: ~45 MB weights, ~256 MB workspace
- Dynamic offload modes: `GPU_PRIMARY` (≥1024 MB free), `PARTIAL_OFFLOAD` (≥512 MB), `FULL_CPU` (<512 MB)
- Memory fence synchronises GPU between vision inference and LLM calls

---

## 3. Package Map — All 15 Packages at a Glance

```
src/uni_vision/
├── common/          # Shared config, logging, exceptions
├── contracts/       # Core DTOs, protocols, interfaces
├── components/      # CVComponent ABC + wrapper adapters
├── ingestion/       # RTSP sources, temporal sampling, pHash dedup
├── detection/       # YOLO vehicle/plate detectors, inference engine, GPU memory
├── preprocessing/   # Deskew (Hough), enhance (CLAHE), ROI extraction
├── ocr/             # Multi-engine OCR (EasyOCR, LLM-provisioned, strategy)
├── postprocessing/  # Validator, adjudicator, deduplicator, dispatcher, orchestrator
├── storage/         # PostgreSQL client, S3 archiver, SQL DDL, retention
├── monitoring/      # VRAM monitor, Prometheus metrics, health, profiler, budget
├── orchestrator/    # Pipeline runner, DI container
├── manager/         # Self-assembling pipeline: 18+ subsystems
├── agent/           # ReAct agentic AI: loop, tools, sessions, knowledge
├── api/             # FastAPI routes, middleware (auth, rate-limit, security)
└── visualizer/      # Streamlit debug pages (8 pages)
```

---

## 4. Configuration System — `common/config.py`

All configuration is managed through **Pydantic `BaseSettings`** models that support environment variable overrides via the `UV_` prefix and YAML loading from `config/default.yaml`.

### 4.1 Top-Level `AppConfig`

`AppConfig` is the root configuration model. Every subsystem receives its slice from this:

```python
class AppConfig(BaseSettings):
    hardware:        HardwareConfig
    vram_budgets:    VRAMBudgets
    pipeline:        PipelineConfig
    ollama:          OllamaConfig
    circuit_breaker: CircuitBreakerConfig
    database:        DatabaseConfig
    storage:         StorageConfig
    redis:           RedisConfig
    api:             APIConfig
    logging:         LoggingConfig
    preprocessing:   PreprocessingConfig
    validation:      ValidationConfig
    adjudication:    AdjudicationConfig
    deduplication:   DeduplicationConfig
    dispatch:        DispatchConfig
    retention:       RetentionConfig
    profiling:       ProfilingConfig
    quantization:    QuantizationConfig
    manager:         ManagerConfig
    agent:           AgentConfig
```

### 4.2 Key Config Models (with Defaults)

| Model | Key Fields | Defaults |
|-------|-----------|----------|
| **HardwareConfig** | `device`, `cuda_device_index`, `vram_ceiling_mb`, `safety_margin_mb`, `poll_interval_ms` | `"cuda"`, `0`, `8192`, `256`, `500` |
| **VRAMBudgets** | `llm_weights`, `kv_cache`, `vision_workspace`, `system_overhead` | `5120`, `512`, `1024`, `512` |
| **OllamaConfig** | `base_url`, `model`, `timeout_s`, `num_ctx`, `temperature`, `top_p`, `seed` | `localhost:11434`, `gemma4:e2b`, `5.0`, `4096`, `0.1`, `0.9`, `42` |
| **PipelineConfig** | `stream_queue_maxsize`, `inference_queue_maxsize`, `high_water_mark`, `low_water_mark`, `throttle_factor` | `50`, `10`, `8`, `3`, `0.5` |
| **ValidationConfig** | `locale_regex`, `confidence_threshold`, `adjudication_threshold`, `char_corrections` | IN regex, `0.65`, `0.75`, `{O↔0, I↔1, S↔5, B↔8, ...}` |
| **DeskewConfig** | `max_skew_degrees`, `skip_below_degrees`, Hough params | `30°`, `3°` |
| **EnhanceConfig** | `clahe_clip_limit`, `bilateral_d`, `gaussian_ksize` | `2.0`, `9`, `(3,3)` |
| **DeduplicationConfig** | `window_seconds`, `purge_interval_seconds` | `10`, `30` |
| **ManagerConfig** | `enabled`, `hub_cache_dir`, `vram_total_mb`, `vram_reserved_for_llm_mb`, `max_search_results`, `prefer_trusted`, `auto_download` | `True`, `~/.cache/uni_vision/hub`, `8192`, `5800`, `10`, `True`, `True` |
| **AgentConfig** | `enabled`, `max_iterations`, `temperature`, `memory_budget_tokens`, `tool_timeout_s` | `True`, `10`, `0.3`, `3072`, `10.0` |

### 4.3 Environment Variable Override

Every field can be overridden via environment variables. The prefix is `UV_` and nested models use `__` as delimiter:

```bash
UV_OLLAMA__MODEL=gemma4:e2b
UV_HARDWARE__VRAM_CEILING_MB=8192
UV_DATABASE__DSN=postgresql://user:pass@host:5432/uni_vision
```

---

## 5. Core Data Contracts — DTOs, Schemas, Enums

### 5.1 Pipeline DTOs (`contracts/dtos.py`)

These are **frozen dataclasses** that flow through the pipeline stages:

| DTO | Fields | Purpose |
|-----|--------|---------|
| **`FramePacket`** | `camera_id`, `timestamp_utc`, `frame_index`, `image` (BGR ndarray) | Unit of work entering the pipeline |
| **`BoundingBox`** | `x1`, `y1`, `x2`, `y2`, `confidence`, `class_id`, `class_name` | Detection result from YOLO |
| **`DetectionContext`** | `camera_id`, `timestamp_utc`, `vehicle_bbox`, `plate_bbox`, `vehicle_class` | Context passed to OCR |
| **`OCRResult`** | `plate_text`, `raw_text`, `confidence`, `reasoning`, `engine`, `status` | Raw OCR output |
| **`ProcessedResult`** | `plate_text`, `raw_ocr_text`, `confidence`, `validation_status`, `char_corrections` | Post-validation result |
| **`DetectionRecord`** | `id` (UUID), `camera_id`, `plate_number`, `raw_ocr_text`, `ocr_confidence`, `ocr_engine`, `vehicle_class`, image paths, `detected_at_utc`, `validation_status`, `location_tag` | Final persistence record |

### 5.2 Key Enums

| Enum | Values | Location |
|------|--------|----------|
| **`VehicleClass`** | `car`, `truck`, `bus`, `motorcycle` | `dtos.py` |
| **`ValidationStatus`** | `valid`, `low_confidence`, `regex_fail`, `llm_error`, `fallback`, `parse_fail`, `unreadable` | `dtos.py` |
| **`OffloadMode`** | `gpu_primary`, `partial_offload`, `full_cpu` | `dtos.py` |
| **`CircuitState`** | `closed`, `open`, `half_open` | `dtos.py` |
| **`ComponentType`** | `MODEL`, `LIBRARY`, `ALGORITHM`, `PREPROCESSOR`, `POSTPROCESSOR` | `components/base.py` |
| **`ComponentState`** | `UNREGISTERED` → `REGISTERED` → `DOWNLOADING` → `LOADING` → `READY` → `UNLOADING` → `FAILED` / `SUSPENDED` | `components/base.py` |
| **`SceneType`** | `traffic`, `parking`, `surveillance`, `industrial`, `indoor`, `unknown` | `manager/schemas.py` |

### 5.3 ComponentCapability Enum (30+ Values)

This enum defines every possible CV capability the system can discover and provision:

```
Detection:      vehicle_detection, plate_detection, person_detection,
                object_detection, face_detection, fire_detection,
                anomaly_detection, zero_shot_detection
OCR:            plate_ocr, scene_text_ocr, document_ocr
Classification: vehicle_classification, scene_classification, action_recognition
Preprocessing:  deskew, enhance, denoise, denoising, geometric_correction,
                roi_extraction, super_resolution
Postprocessing: text_validation, text_correction, plate_localization,
                plate_validation, dedup, tracking
Segmentation:   semantic_segmentation, instance_segmentation
Advanced:       depth_estimation, pose_estimation
```

### 5.4 Manager Schemas (`manager/schemas.py`)

| Schema | Key Fields | Purpose |
|--------|-----------|---------|
| **`FrameContext`** | `scene_type`, `required_capabilities` (FrozenSet), `optional_capabilities`, `priority`, `camera_id` | Frame-level analysis result |
| **`StageSpec`** | `stage_name`, `required_capability`, `component_id`, `input_key`, `output_key`, `config_overrides` | One stage in a pipeline blueprint |
| **`PipelineBlueprint`** | `blueprint_id`, `name`, `context`, `stages` (List[StageSpec]), `estimated_vram_mb`, `estimated_latency_ms` | Complete pipeline definition |
| **`ComponentCandidate`** | `component_id`, `name`, `source`, `capabilities`, `vram_mb`, `score`, `python_requirements`, `trusted` | Internet-discovered model |
| **`ResolutionResult`** | `capability`, `status`, `selected_candidate`, `all_candidates`, `error` | Per-capability resolution outcome |
| **`ConflictReport`** | `conflicts`, `total_vram_required_mb`, `vram_available_mb`, `is_feasible` | Pre-execution VRAM/dep check |
| **`ManagerAction`** | `decision`, `target_blueprint`, `components_to_load/unload/download`, `reasoning` | Manager's per-frame decision |
| **`PipelineExecutionResult`** | `blueprint_id`, `stage_results`, `final_output`, `total_elapsed_ms`, `success`, `error` | Execution outcome |

---

## 6. Component Abstraction Layer — `components/`

### 6.1 `CVComponent` — The Universal Protocol

Every computer-vision model in the system — whether built-in (YOLO, EasyOCR) or dynamically provisioned from the internet (TrOCR, PaddleOCR) — is wrapped as a `CVComponent`:

```python
class CVComponent(ABC):
    @property
    @abstractmethod
    def metadata(self) -> ComponentMetadata: ...

    @abstractmethod
    async def load(self, device: str = "cuda") -> None: ...

    @abstractmethod
    async def unload(self) -> None: ...

    @abstractmethod
    async def execute(self, data: Any, context: dict[str, Any] | None = None) -> Any: ...

    async def warmup(self) -> None: ...        # optional
    async def health_check(self) -> bool: ...  # optional
    def get_runtime_info(self) -> dict: ...     # optional
```

**`ComponentMetadata`** carries: `component_id`, `name`, `version`, `type` (MODEL/LIBRARY/etc.), `capabilities` (frozenset of `ComponentCapability`), `source` (huggingface/pypi/builtin/etc.), `source_id`, `description`, `license`, `python_requirements`, `resource_estimate` (VRAM/RAM), `tags`, `trusted` flag.

**`ResourceEstimate`** specifies: `vram_mb`, `ram_mb`, `supports_gpu`, `supports_cpu`, `supports_half`, `supports_int8`.

### 6.2 Wrapper Classes (`components/wrappers.py`)

Four adapter classes wrap built-in subsystems as `CVComponent` instances so the Manager Agent can orchestrate them uniformly:

| Wrapper | Wraps | Key Behaviour |
|---------|-------|---------------|
| **`BuiltinDetectorComponent`** | `VehicleDetector` / `PlateDetector` | `execute()` runs detection. In plate-crop mode (`PLATE_DETECTION` capability), it also extracts plate crops from vehicle bounding boxes, returning `List[Dict]` with `plate_crop`, `plate_bbox`, `vehicle_bbox`. |
| **`BuiltinPreprocessorComponent`** | `HoughStraightener` / `PhotometricEnhancer` | Polymorphic `execute()`: handles a single `ndarray` image OR a `List[Dict]` of plate-crop records, applying preprocessing to each `plate_crop` field in-place. |
| **`BuiltinOCRComponent`** | `OCRStrategy` | Polymorphic `execute()`: single image → single `OCRResult`, or `List[Dict]` plate-crop records → per-crop OCR, attaching `ocr_result` to each dict. |
| **`BuiltinPostprocessorComponent`** | `CognitiveOrchestrator` | Wraps the validation + adjudication chain. |

### 6.3 Dynamic Component Wrappers (Manager-Provisioned)

When the Manager downloads models from the internet, they're wrapped in one of:
- **`HuggingFaceModelComponent`** — `from_pretrained` pattern (Transformers, Diffusers, etc.)
- **`PipPackageComponent`** — pip-installed packages (PaddleOCR, DeepSort, etc.)
- **`TorchHubComponent`** — `torch.hub.load` pattern (YOLOv5, etc.)

All implement the same `CVComponent` protocol, ensuring uniform lifecycle management.

---

## 7. Ingestion Layer — `ingestion/`

The ingestion layer is responsible for pulling frames from cameras and filtering duplicates before they reach the GPU.

### 7.1 `RTSPFrameSource` — Camera Connection

**File:** `ingestion/rtsp_source.py`

Each camera gets a dedicated `RTSPFrameSource` that:
1. Opens an RTSP connection via `cv2.VideoCapture` with FFMPEG backend (RTSP-over-TCP)
2. Spawns a **background daemon thread** that continuously reads frames into a per-camera ring-buffer queue (default size: 50)
3. On disconnect, performs **exponential backoff reconnection**: 1s → 2s → 4s → 8s → 16s (max), up to 10 attempts
4. Supports **throttling** (reduce effective FPS by factor 0.1–1.0) for backpressure relief

**Zero GPU VRAM** — all I/O runs on CPU.

### 7.2 `TemporalSampler` — Intelligent Frame Selection

**File:** `ingestion/sampler.py`

Sits between per-camera queues and the inference queue. Applies two filters:

1. **FPS Gating:** Only allows one frame per `1/fps_target` interval per camera
2. **Perceptual Deduplication:** Computes a 64-bit DCT-based pHash of each frame and compares it to the previous frame's hash. If the Hamming distance ≤ threshold (default: 5), the frame is discarded as visually identical.

Uses weighted round-robin polling of all connected sources. Schedules async enqueue via `asyncio.run_coroutine_threadsafe`.

### 7.3 `compute_phash` — Perceptual Hashing

**File:** `ingestion/phash.py`

Algorithm: grayscale → resize to `hash_size²` → separable 2-D DCT via pre-cached orthonormal basis matrix → retain top-left 8×8 low-frequency block (exclude DC coefficient) → binarise against median → pack into 64-bit integer.

**Performance:** < 0.3 ms per 1080p frame. Pure NumPy — zero VRAM.

---

## 8. Detection Layer — `detection/`

### 8.1 `InferenceEngine` — Unified Backend

**File:** `detection/engine.py`

A unified inference wrapper supporting two backends:
- **TensorRT:** Deserialises `.engine` files, allocates CUDA I/O buffers, executes via PyCUDA streams
- **ONNX Runtime:** CUDA Execution Provider or CPU fallback

Both share common preprocessing (`letterbox` — resize + pad + BGR→RGB + HWC→NCHW + float32/255) and postprocessing (`_nms_numpy` — pure-NumPy greedy NMS, no torchvision dependency).

### 8.2 `VehicleDetector` — S2 Stage

**File:** `detection/vehicle_detector.py`

Edge-optimised **YOLOv8n INT8** for vehicle detection:
- **Input:** Full-frame BGR image
- **Output:** `List[BoundingBox]` with `class_name` ∈ `{car, truck, bus, motorcycle}`
- **Confidence threshold:** 0.60
- **NMS IoU threshold:** 0.45
- **VRAM footprint:** ≤ 400 MB (Region C)

### 8.3 `PlateDetector` — S3 Stage

**File:** `detection/plate_detector.py`

Localised plate detection within vehicle ROIs:
- `detect_in_roi(frame, vehicle_bbox)` — crops vehicle region → detects plates → remaps coordinates to full-frame space
- **Multi-plate policy:** `highest_confidence` (keeps only the best plate per vehicle)
- **Confidence threshold:** 0.65

### 8.4 `GPUMemoryManager` — Dynamic VRAM Offload

**File:** `detection/gpu_memory.py`

Polls free VRAM via pynvml and decides on offload mode:

| Free VRAM | Mode | Behaviour |
|-----------|------|-----------|
| ≥ 1024 MB | `GPU_PRIMARY` | Both detectors on GPU |
| ≥ 512 MB | `PARTIAL_OFFLOAD` | S2 (vehicle) on GPU, S3 (plate) on CPU |
| < 512 MB | `FULL_CPU` | Both detectors on CPU |

Also provides `assert_memory_fence()` — a hard GPU sync barrier before LLM inference, ensuring no VRAM contention between vision and language workloads.

---

## 9. Preprocessing Layer — `preprocessing/`

### 9.1 `HoughStraightener` — Deskew

**File:** `preprocessing/deskew.py`

Corrects plate rotations using Hough Line Transform:
- Canny edge detection → `HoughLinesP` → extract near-horizontal lines → compute median angle
- If angle > `skip_threshold` (3°): apply affine rotation
- Max skew: 30°

### 9.2 `PhotometricEnhancer` — Image Enhancement

**File:** `preprocessing/enhance.py`

Sequential enhancement chain (each stage independently toggleable):
1. **Resize** — upscale small plates to minimum dimensions
2. **CLAHE** — Contrast Limited Adaptive Histogram Equalisation on LAB L-channel (clip limit: 2.0)
3. **Gaussian blur** — noise reduction
4. **Bilateral filter** — edge-preserving smoothing

### 9.3 `PreprocessingChain` — Pipeline Composition

**File:** `preprocessing/chain.py`

Chains multiple `Preprocessor` protocol implementations sequentially. Records `STAGE_LATENCY` Prometheus histogram per stage.

### 9.4 `extract_plate_roi` — ROI Extraction

**File:** `preprocessing/roi_extractor.py`

Crops the plate region from a frame given a `BoundingBox`, with configurable symmetric padding (default: 5px), clamped to frame boundaries.

---

## 10. OCR Layer — `ocr/`

### 10.1 `OCRStrategy` — Multi-Engine Orchestrator

**File:** `ocr/strategy.py`

Priority-ordered multi-engine OCR with fallback:
- Tries engines in order; first non-`UNREADABLE` result wins
- The **Manager Agent dynamically adds/removes engines** at runtime via `add_engine()` / `remove_engine()`
- Tracks per-engine metrics (attempts, successes, failures)

### 10.2 `EasyOCRFallback` — CPU-Only Safety Net

**File:** `ocr/fallback_ocr.py`

- Engine: `easyocr.Reader` (English, GPU disabled)
- Runs in a **single-thread `ThreadPoolExecutor`** to avoid GIL contention
- Lazy-initialised on first call
- Allowlist: `A-Z0-9`
- **Zero VRAM** — runs entirely on CPU

### 10.3 `ComponentOCREngine` — Manager-Provisioned Adapter

**File:** `ocr/llm_ocr.py`

Adapter that wraps any Manager-provisioned CV component (TrOCR, PaddleOCR, etc.) into the `OCREngine` protocol. Normalises heterogeneous outputs (str/dict/OCRResult) to a uniform `OCRResult`. Error-isolated — never crashes the pipeline.

### 10.4 `parse_llm_response` — XML Response Parser

**File:** `ocr/response_parser.py`

Regex-based parser that extracts structured XML blocks (`<result>`, `<plate_text>`, `<confidence>`, `<reasoning>`) from LLM responses. Used by the adjudicator when LLM text reasoning is needed.

---

## 11. Postprocessing Layer — `postprocessing/`

### 11.1 `DeterministicValidator` — Fast Path

**File:** `postprocessing/validator.py`

Sub-millisecond plate text validation:
1. **Normalise** — uppercase, strip whitespace
2. **Raw regex match** — test against locale pattern (default: Indian plates)
3. **If fail:** apply positional character corrections based on confusion matrix:
   - Alpha↔Digit corrections: `O↔0`, `I↔1`, `S↔5`, `B↔8`, `D↔0`, `Z↔2`, `G↔6`, `T↔7`, `A↔4`
   - Position-aware: uses regex-derived mask (`A`=alpha slot, `D`=digit slot)
4. **Re-test** corrected text against regex

Returns `Verdict`: `ACCEPTED`, `CORRECTED`, `REGEX_FAIL`, or `LOW_CONFIDENCE`.

### 11.2 `ConsensusAdjudicator` — Multi-Engine Voting

**File:** `postprocessing/adjudicator.py`

Invoked only when deterministic validation returns `REGEX_FAIL` or `LOW_CONFIDENCE`:
1. Runs **all available OCR engines** concurrently on the plate crop
2. Collects valid readings
3. Votes via `Counter.most_common()`
4. Consensus confidence = `agreement_ratio × 0.5 + avg_engine_confidence × 0.5`
5. Requires ≥2 engines to operate

### 11.3 `CognitiveOrchestrator` — S8 Decision Engine

**File:** `postprocessing/orchestrator.py`

Two-layer validation pipeline:
1. **Layer 1:** `DeterministicValidator` — fast char correction + regex (sub-ms)
2. **Layer 2:** `ConsensusAdjudicator` — multi-engine voting (invoked conditionally)

Returns `ProcessedResult` with final `validation_status`.

### 11.4 `SlidingWindowDeduplicator` — Temporal Dedup

**File:** `postprocessing/deduplicator.py`

Prevents duplicate dispatch of the same plate:
- Window: 10 seconds per (camera_id, normalised plate_text)
- Keeps **highest-confidence** reading within each window
- Background purge task evicts expired entries every 30 seconds

### 11.5 `MultiDispatcher` — Async Persistence

**File:** `postprocessing/dispatcher.py`

Decoupled async multi-target dispatcher with a bounded ring-buffer queue:
1. **Temporal dedup** check
2. **Audit routing** for low-confidence / failed reads → `ocr_audit_log`
3. **S3 image upload** — plate crop encoded as PNG/JPEG
4. **PostgreSQL insert** — full `DetectionRecord`
5. **Redis publish** — `uv:events` channel for WebSocket broadcast

Each target has independent retry budgets. Background consumer drains queue asynchronously.

---

## 12. Storage Layer — `storage/`

### 12.1 `PostgresClient`

**File:** `storage/postgres.py`

- Async connection pool via `asyncpg`
- Auto-creates DDL schema on connect (3 tables)
- `insert_detection()` — parameterised 12-field INSERT with `ON CONFLICT DO NOTHING`
- `insert_audit_log()` — audit entries for failed/low-confidence OCR
- Exponential-backoff retry on transient failures

### 12.2 `ObjectStoreArchiver`

**File:** `storage/object_store.py`

- S3-compatible archiver via `aioboto3` (MinIO in dev, AWS S3 in prod)
- Key pattern: `plates/{camera_id}/{record_id}.{ext}`
- PNG for quality, JPEG for bandwidth
- Auto-creates bucket on startup

### 12.3 SQL Schema (`storage/models.py`)

Three core tables:

| Table | Columns | Indexes |
|-------|---------|---------|
| **`detection_events`** | `id`, `camera_id`, `plate_number`, `raw_ocr_text`, `ocr_confidence`, `ocr_engine`, `vehicle_class`, `vehicle_image_path`, `plate_image_path`, `detected_at_utc`, `validation_status`, `location_tag` | `(camera_id, detected_at DESC)`, `(plate_number, detected_at DESC)` |
| **`camera_sources`** | `camera_id`, `url`, `label`, `fps_target`, `enabled`, `created_at` | Primary key on `camera_id` |
| **`ocr_audit_log`** | `id`, `record_id`, `camera_id`, `raw_ocr_text`, `ocr_confidence`, `failure_reason`, `frame_path`, `created_at` | `(created_at DESC)` |

### 12.4 `RetentionTask` — Data Lifecycle

**File:** `storage/retention.py`

Background task that periodically purges:
- Detection records older than `max_age_days` (default: 90)
- Audit log entries older than `audit_max_age_days`
- Operates in batches to avoid long-lived DB locks

---

## 13. Monitoring Layer — `monitoring/`

### 13.1 Prometheus Metrics (`monitoring/metrics.py`)

14 module-level singleton metrics, all prefixed `uv_`:

| Type | Metric | Labels |
|------|--------|--------|
| Counter | `uv_frames_ingested_total` | `camera_id` |
| Counter | `uv_frames_deduplicated_total` | — |
| Counter | `uv_frames_dropped_total` | `reason` |
| Counter | `uv_detections_total` | `vehicle_class` |
| Counter | `uv_ocr_requests_total` | `engine` |
| Counter | `uv_ocr_success_total` | `engine` |
| Counter | `uv_ocr_fallback_total` | — |
| Counter | `uv_detections_deduplicated_total` | — |
| Counter | `uv_dispatch_success_total` | — |
| Counter | `uv_dispatch_errors_total` | `target` |
| Counter | `uv_agent_requests_total` | — |
| Counter | `uv_agent_tool_calls_total` | `tool_name` |
| Histogram | `uv_pipeline_latency_seconds` | buckets: 0.5–20s |
| Histogram | `uv_stage_latency_seconds` | `stage`, buckets: 5ms–5s |
| Histogram | `uv_ocr_confidence` | `engine`, buckets: 0.1–1.0 |
| Histogram | `uv_agent_latency_seconds` | buckets: 0.5–60s |
| Histogram | `uv_agent_steps` | buckets: 1–15 |
| Gauge | `uv_vram_usage_bytes` | `region` |
| Gauge | `uv_inference_queue_depth` | — |
| Gauge | `uv_stream_status` | `camera_id` |

### 13.2 `VRAMMonitor` — GPU Telemetry

**File:** `monitoring/vram_monitor.py`

Background asyncio task polling GPU via pynvml every 500ms:
- Captures: VRAM total/used/free, GPU utilisation %, temperature, PCIe TX/RX throughput
- Distributes VRAM proportionally across 4 regions
- Publishes to Prometheus gauges
- Provides `offload_mode` property for dynamic decisions
- Degrades gracefully when pynvml is unavailable

### 13.3 `VRAMBudgetReport` — Static Budget Analysis

**File:** `monitoring/vram_budget.py`

Functions for pre-flight VRAM accounting:
- `compute_budget()` → `VRAMBudgetReport` with per-region allocation, headroom, warnings
- `validate_budget()` → raises `MemoryError` if budget exceeded
- `max_context_for_budget()` → calculates max LLM context tokens that fit

### 13.4 `HealthService` — Aggregated Health Check

**File:** `monitoring/health.py`

Runs all sub-checks concurrently via `asyncio.gather`:
- GPU telemetry (pynvml)
- Ollama reachability (`/api/tags` probe, 3s timeout)
- Database connectivity (`SELECT 1`, 3s timeout)
- Camera stream status

### 13.5 Stage Profiler

**File:** `monitoring/profiler.py`

`@profile_stage(name)` decorator that measures wall-time + VRAM delta per pipeline stage. Ring buffer of 512 entries. Zero-overhead when disabled.

---

## 14. The Legacy Pipeline — `orchestrator/`

### 14.1 `Pipeline` — The Pipeline Runner

**File:** `orchestrator/pipeline.py`

Two-layer queue architecture:
- **Stream queues** (per camera, size 50) — fed by `RTSPFrameSource`
- **Inference queue** (size 10) — fed by `TemporalSampler`

The pipeline processes frames through numbered stages:

```
FramePacket
    │
    ▼
[S0] Ingestion — RTSPFrameSource reads RTSP stream
    │
    ▼
[S1] Sampling — TemporalSampler filters by FPS + pHash dedup
    │
    ▼
[S2] Vehicle Detection — YOLOv8n INT8 → List[BoundingBox]
    │
    ▼
[S3] Plate Detection — YOLOv8n INT8 in vehicle ROIs → List[BoundingBox]
    │
    ▼
[S4] ROI Extraction — Crop plate regions with padding
    │
    ▼
    ═══ MEMORY FENCE (GPU sync barrier) ═══
    │
    ▼
[S5] Deskew — HoughStraightener corrects plate rotation
    │
    ▼
[S6] Enhance — CLAHE + blur + bilateral filtering
    │
    ▼
[S7] OCR — Multi-engine strategy (EasyOCR + dynamic engines)
    │
    ▼
[S8] Post-Processing — Validate → Adjudicate → Dedup → Dispatch
```

**Adaptive throttling:** When inference queue exceeds `high_water_mark` (8), FPS is reduced by `throttle_factor` (0.5×). Restored when queue drops below `low_water_mark` (3).

**Dynamic pipeline routing:** If `manager_agent` is wired, the pipeline delegates to `ManagerAgent.process_frame()` instead of the fixed S2-S8 stages, then bridges results back via `_dispatch_dynamic_results()`.

### 14.2 DI Container (`orchestrator/container.py`)

The `build_pipeline()` function is the **entire dependency injection container**, wiring 17+ subsystems:

```
VRAMMonitor
    ├── VehicleDetector(model_path, device, format=tensorrt)
    ├── PlateDetector(model_path, device, format=tensorrt)
    ├── HoughStraightener(config.preprocessing.deskew)
    ├── PhotometricEnhancer(config.preprocessing.enhance)
    ├── EasyOCRFallback(config.validation.fallback)
    ├── OCRStrategy([easyocr_fallback])
    ├── ConsensusAdjudicator(ocr_strategy, config.adjudication)
    ├── CognitiveOrchestrator(config.validation, adjudicator)
    ├── MultiDispatcher(db, s3, dispatch, dedup configs)
    │
    ├── [If manager.enabled]:
    │   ├── ComponentRegistry  ──── 6 builtin components registered:
    │   │   ├── BuiltinDetectorComponent(vehicle_detector, VEHICLE_DETECTION)
    │   │   ├── BuiltinDetectorComponent(plate_detector, PLATE_DETECTION)
    │   │   ├── BuiltinOCRComponent(ocr_strategy, PLATE_OCR)
    │   │   ├── BuiltinPreprocessorComponent(straightener, DESKEW)
    │   │   ├── BuiltinPreprocessorComponent(enhancer, ENHANCE)
    │   │   └── BuiltinPostprocessorComponent(cognitive_orch, TEXT_VALIDATION)
    │   ├── HubClient
    │   ├── ComponentResolver
    │   ├── ConflictResolver
    │   ├── ContextAnalyzer
    │   ├── PipelineComposer
    │   ├── LifecycleManager
    │   ├── PipelineValidator
    │   ├── FeedbackLoop
    │   ├── AdaptationEngine
    │   ├── FallbackChainManager
    │   ├── QualityScorer
    │   ├── SceneTransitionDetector
    │   ├── GPUProfiler
    │   ├── CompatibilityMatrix
    │   ├── TemporalTracker
    │   └── ManagerAgent(all of the above)
    │
    └── Pipeline(config, vram_monitor, detectors, ocr, ..., manager_agent)
```

---

## 15. The Manager Agent — `manager/` (Self-Assembling Pipeline)

This is the most complex subsystem — 18+ interacting modules that allow Gemma 4 E2B to dynamically build, adapt, and optimise CV pipelines at runtime.

### 15.1 Architecture Overview

```
                    ┌─────────────────────────┐
                    │     ManagerAgent         │  ← Central meta-orchestrator
                    │  process_frame(frame)    │
                    └──────────┬──────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
  ContextAnalyzer       PipelineComposer       LifecycleManager
  (scene analysis)      (blueprint build)      (load/unload/swap)
        │                      │                      │
        ▼                      ▼                      ▼
  SceneTransition       ComponentRegistry       GPUProfiler
  Detector              (component index)       (VRAM measurement)
        │                      │                      │
        ▼                      ▼                      ▼
  TemporalTracker       ComponentResolver       ConflictResolver
  (multi-frame ctx)     (discover+provision)    (VRAM/dep checks)
        │                      │                      │
        ▼                      ▼                      ▼
  AdaptationEngine       HubClient             FallbackChainManager
  (closed-loop adapt)   (HuggingFace/PyPI)     (per-cap fallbacks)
        │                      │                      │
        ▼                      ▼                      ▼
  FeedbackLoop          PipelineValidator       QualityScorer
  (EWMA telemetry)      (pre-execution check)   (Bayesian scoring)
                                                      │
                                                      ▼
                                                CompatibilityMatrix
                                                (pairwise compat)
```

### 15.2 `ManagerAgent` — The 9-Step Hot Path

**File:** `manager/agent.py`

On each frame, `process_frame()` executes:

| Step | Action | Subsystem |
|------|--------|-----------|
| 1 | **Analyse context** — determine scene type, required capabilities | `ContextAnalyzer` |
| 2 | **Scene transition detection** — hysteresis-based scene change confirmation | `SceneTransitionDetector` |
| 3 | **Temporal enrichment** — add multi-frame context (trends, tracks, capability hints) | `TemporalTracker` |
| 4 | **Cache check** — lookup blueprint by capability-set hash (LRU cache) | Internal cache |
| 5 | **Build pipeline** — compose `PipelineBlueprint` from context + resolved components | `PipelineComposer` |
| 6 | **Compatibility check** — detect VRAM overruns, dependency conflicts | `ConflictResolver` |
| 7 | **Ensure loaded** — load required components, VRAM-aware LRU eviction if needed | `LifecycleManager` |
| 8 | **Execute blueprint** — run stages sequentially, passing data through IO keys | Internal executor |
| 9 | **Post-execution** — record telemetry, trigger adaptation, update feedback loop | `FeedbackLoop` + `AdaptationEngine` |

### 15.3 `ContextAnalyzer` — Scene Understanding

**File:** `manager/context_analyzer.py`

Two-path analysis:
- **Fast heuristic** (default): Camera ID keyword hints + basic image statistics (brightness, aspect ratio)
- **LLM-assisted** (optional): Sends image stats to Gemma 4 E2B for structured JSON response

Maps `SceneType` → required capabilities via `_SCENE_CAPABILITIES`:

| Scene | Required Capabilities |
|-------|----------------------|
| TRAFFIC | `vehicle_detection`, `plate_detection`, `plate_ocr`, `text_validation` |
| PARKING | `vehicle_detection`, `plate_detection`, `plate_ocr` |
| SURVEILLANCE | `person_detection`, `vehicle_detection`, `tracking` |
| INDUSTRIAL | `object_detection`, `anomaly_detection` |
| INDOOR | `person_detection`, `scene_classification` |

### 15.4 `ComponentResolver` — Internet-Aware Discovery

**File:** `manager/component_resolver.py`

For each missing capability:
1. **Check local registry** — is a component already registered?
2. **Query HuggingFace Hub** — search by `pipeline_tag` mapped from capability
3. **Check PyPI** — lookup known packages (`_KNOWN_PIP_PACKAGES`)
4. **Rank candidates** — score by: popularity, VRAM fit, trust status, library compatibility
5. **Download & install** — `pip install --no-deps`, model weights via httpx
6. **Wrap** as CVComponent (HuggingFace/Pip/TorchHub wrapper)
7. **Register** in `ComponentRegistry`
8. **Load** via `LifecycleManager`

**Known PyPI Packages:** PaddleOCR, DeepSort, timm, Kornia, Albumentations, OpenCLIP.

### 15.5 `HubClient` — Model Search & Download

**File:** `manager/hub_client.py`

Async `httpx`-based client for model discovery:

| Source | API | Capabilities |
|--------|-----|-------------|
| **HuggingFace** | `/api/models` | Search by task, filter by library, sort by downloads |
| **PyPI** | JSON API | Package metadata, version info |
| **TorchHub** | `torch.hub` | Pre-registered hub repos |

**Security:**
- 15 trusted HuggingFace repos (Ultralytics, Google, Microsoft, Facebook, etc.)
- 11 library load patterns (transformers, ultralytics, timm, diffusers, open_clip, PaddleOCR, detectron2, etc.)
- No arbitrary code execution during download — execution only after explicit `load()`
- `pip install --no-deps` by default

### 15.6 `LifecycleManager` — VRAM-Aware Component Loading

**File:** `manager/lifecycle.py`

| Operation | Behaviour |
|-----------|-----------|
| `load_component(id)` | Check VRAM budget → if insufficient, LRU-evict → load → health check → state → READY |
| `unload_component(id)` | State → UNLOADING → call `component.unload()` → torch.cuda.empty_cache → State → REGISTERED |
| `swap_component(old, new)` | Atomic swap: unload old → load new → if new fails, restore old |
| `ensure_loaded(ids)` | Bulk-load with priority ordering |

**VRAM budget:** `vram_total - vram_reserved_for_llm` = 8192 - 5800 = **2392 MB** for CV models.

### 15.7 `PipelineComposer` — Blueprint Construction

**File:** `manager/pipeline_composer.py`

Takes a `FrameContext` and builds a `PipelineBlueprint`:
1. Sort required capabilities by canonical `_STAGE_ORDER` (detection=10 → OCR=40 → validation=50 → ...)
2. For each capability, resolve to a component ID (prefer READY components from registry)
3. Create `StageSpec` entries with canonical `input_key`/`output_key` wiring
4. Accumulate VRAM estimates
5. Also provides `compose_default_pipeline()` for a standard detection pipeline

### 15.8 `ConflictResolver` — Pre-Execution Validation

**File:** `manager/conflict_resolver.py`

Three validation checks before pipeline execution:
1. **VRAM budget overshoot** — total component VRAM > available budget
2. **Capability overlap** — duplicate stages serving same capability
3. **Dependency clashes** — conflicting Python package versions

Auto-resolve suggestions: unload LRU components, remove duplicate stages.

### 15.9 `FeedbackLoop` — Closed-Loop Telemetry

**File:** `manager/feedback_loop.py`

Records per-component and per-blueprint performance using **Exponential Weighted Moving Averages (EWMA)**:
- Latency (median, P99, std)
- Confidence
- Reliability (success rate)
- Sliding window of 100 observations

Feeds data into `AdaptationEngine`, `QualityScorer`, and `PipelineComposer`.

### 15.10 `AdaptationEngine` — Self-Healing

**File:** `manager/adaptation_engine.py`

8 adaptation signals: `SCENE_DRIFT`, `QUALITY_DROP`, `LATENCY_SPIKE`, `VRAM_PRESSURE`, `COMPONENT_FAILURE`, `CONFIDENCE_DEGRADATION`, etc.

On every frame result, runs 4 internal checks:
1. **Scene drift** — dominant scene changed for camera
2. **Quality degradation** — avg confidence below threshold (0.4)
3. **Latency spikes** — P95 latency exceeds budget
4. **Error rates** — component error rate > 0.3

Generates `AdaptationAction`s: `swap`, `add`, `remove`, `downgrade`, `recompose`. Cooldown prevents action storms.

### 15.11 `SceneTransitionDetector` — Hysteresis State Machine

**File:** `manager/scene_detector.py`

Per-camera scene stability tracking:
- States: `STABLE` → `TRANSITIONING` → `CONFIRMED`
- Requires N consecutive agreeing frames (default: 5) before confirming a scene change
- Uses histogram chi-squared distance for drift detection
- Returns `(old_scene, new_scene)` on confirmed transition

### 15.12 `QualityScorer` — Bayesian Component Ranking

**File:** `manager/quality_scorer.py`

Per-component quality scoring using Bayesian Beta-Binomial reliability:
- **4 sub-scores:** latency (0.25 weight), confidence (0.30), reliability (0.30), VRAM efficiency (0.15)
- Composite score = weighted sum
- `get_best_for_capability(cap)` — returns highest-scoring component for a capability
- Score confidence saturates after 50 observations

### 15.13 `TemporalTracker` — Multi-Frame Context

**File:** `manager/temporal_tracker.py`

Tracks temporal patterns across frames:
- **Object tracks:** per-object (track_id) position, confidence, age, stationarity
- **Environment trends:** brightness, latency, detection confidence over time
- `get_context_summary()` → LLM-friendly dict for ManagerAgent
- `get_capability_hints()` → suggests capabilities (e.g., `low_light_enhancement`, `tracking`)

### 15.14 `FallbackChainManager` — Per-Capability Fallback Tiers

**File:** `manager/fallback_chain.py`

Maintains ordered fallback chains per capability:
- Tiers: `PRIMARY` → `SECONDARY` → `TERTIARY` → `EMERGENCY`
- Auto-disables after N consecutive failures (default: 5)
- Auto-recovers disabled entries after cooldown (120s)
- Sorts by `effective_score` = `score × reliability − failure_penalty`

### 15.15 `GPUProfiler` — VRAM Measurement

**File:** `manager/gpu_profiler.py`

Measures actual VRAM usage during component load/unload:
- `measure_load(id, estimated_mb)` — context manager that diffs GPU memory before/after
- `measure_unload(id, expected_free_mb)` — checks memory actually freed; flags leaks > 50 MB
- Provides corrected VRAM estimates based on measurements vs. metadata

### 15.16 `CompatibilityMatrix` — Pairwise Component Compatibility

**File:** `manager/compatibility.py`

Tracks pairwise compatibility between components:
- States: `COMPATIBLE`, `INCOMPATIBLE`, `UNKNOWN`, `CONDITIONAL`
- Empirical updates: `record_success()` / `record_failure()` — marks incompatible after N failures
- Pre-loaded static rules (e.g., `onnxruntime-gpu` vs `onnxruntime` conflict, TensorRT ABI mismatches)
- `check_set(ids)` — all pairwise checks before loading a component group

### 15.17 `ComponentRegistry` — Central Component Index

**File:** `manager/component_registry.py`

Thread-safe (`threading.Lock`) registry of all `CVComponent` instances:
- `register(component)` / `unregister(id)` — maintains a capability index
- `get_by_capability(cap, only_ready=True)` — find components serving a capability
- `get_missing_capabilities(required)` — determines what needs to be provisioned
- `get_loaded_vram_mb()` — total VRAM of READY components
- `summary()` / `loaded_summary()` — JSON-serialisable for LLM prompts

### 15.18 `PipelineValidator` — Pre-Execution Safety

**File:** `manager/pipeline_validator.py`

Validates a `PipelineBlueprint` before execution:
1. **Components exist** — all referenced component IDs in registry
2. **IO key chaining** — each stage's input_key is produced by a prior stage's output_key
3. **VRAM budget** — total estimated VRAM ≤ budget (2200 MB)
4. **Non-empty** — at least one stage

Optional `dry_run()` — executes each READY component with a dummy frame for integration testing.

---

## 16. The Agentic AI System — `agent/` (ReAct Loop)

This subsystem provides a natural-language conversational interface powered by Gemma 4 E2B via a **ReAct (Reasoning + Acting)** loop.

### 16.1 Architecture Overview

```
User Message
    │
    ▼
AgentCoordinator.chat()
    │
    ├── classify_intent() ────── 8 intents (regex-based, zero LLM)
    │
    ├── MultiAgentRouter.route()
    │   ├── OCR_QUALITY sub-agent (12 tools)
    │   ├── ANALYTICS sub-agent (10 tools)
    │   ├── OPERATIONS sub-agent (13 tools)
    │   └── GENERAL sub-agent (all tools)
    │
    ▼
AgentLoop.run()   ← ReAct cycle
    │
    ├── LLM: Think + Choose Action
    ├── ToolRegistry.invoke(tool_name, args)
    ├── Observation → Back to LLM
    └── Repeat until answer or max_iterations (10)
```

### 16.2 `AgentCoordinator` — Entry Point

**File:** `agent/coordinator.py`

The single entry-point for user↔agent interaction:
1. **Intent classification** — regex-based, 8 intents (`STATUS`, `DETECTION`, `ANALYTICS`, `CAMERA`, `CONFIG`, `KNOWLEDGE`, `DIAGNOSTICS`, `GENERAL`)
2. **Context enrichment** — session history + knowledge hints + pipeline state
3. **Sub-agent routing** via `MultiAgentRouter` (role-specific tool subsets)
4. **Audit recording** — every tool call logged to `agent_audit_log`
5. **Background monitoring** via `AutonomousMonitor`

### 16.3 `AgentLoop` — ReAct Engine

**File:** `agent/loop.py`

The core reasoning loop:
1. Send conversation (system prompt + history + current message) to Ollama `/api/chat`
2. Parse JSON response — look for `action` + `action_input` OR `answer`
3. If action: invoke tool via `ToolRegistry`, add observation to conversation
4. If answer: return final response
5. Repeat until answer or `max_iterations` (10)

Strips markdown code fences, finds JSON in free-form LLM output, handles parse errors gracefully.

### 16.4 `ToolRegistry` — Dynamic Tool System

**File:** `agent/tools.py`

A decorator-based tool registration system:

```python
@tool(name="query_detections", description="Search detection records")
async def query_detections(camera_id: str = "", plate_number: str = "", ...):
    ...
```

Features:
- **`@tool` decorator** — auto-generates JSON schemas from Python type hints
- **Type-hint → JSON mapping:** `str → string`, `int → integer`, `float → number`, `bool → boolean`
- **Context injection** — tools accepting a `context` parameter get pipeline state injected automatically
- **Per-tool metrics** — invocation count, success/failure tracking

### 16.5 Available Tools — 37 Total

The agent has access to 37 tools across four categories:

#### Pipeline Tools (11 tools)
| Tool | Purpose |
|------|---------|
| `query_detections` | Search detection records with filters (camera, plate, status, time range) |
| `get_detection_summary` | Aggregate stats: total, confidence, by-camera, by-status, by-vehicle-class |
| `get_pipeline_stats` | Read Prometheus `uv_*` metrics |
| `get_system_health` | Pipeline state, DB, queue depth, circuit breaker, VRAM, errors |
| `list_cameras` / `manage_camera` | Camera CRUD operations |
| `adjust_threshold` | Runtime threshold adjustment (OCR confidence, temperature, etc.) |
| `get_current_config` | Current pipeline configuration snapshot |
| `search_audit_log` | Search OCR audit log for failures |
| `analyze_detection_patterns` | Frequency distributions, multi-camera detections, low-confidence reads |
| `diagnose_camera` | Error rates, confidence analysis, auto-recommendations per camera |
| `run_analytics_query` | Natural language → safe pre-approved SQL patterns |

#### Knowledge Tools (10 tools)
| Tool | Purpose |
|------|---------|
| `get_knowledge_stats` | KB summary statistics |
| `get_frequent_plates` | Top-N most-seen plates |
| `get_camera_error_profile` | Per-camera OCR error analysis |
| `get_all_camera_profiles` | All camera error profiles |
| `get_cross_camera_plates` | Plates seen across multiple cameras |
| `get_ocr_error_patterns` | Common OCR confusion patterns per camera |
| `detect_detection_anomalies` | Spike detection in plate frequency |
| `record_detection_feedback` | Record operator corrections |
| `get_recent_feedback` | Recent operator feedback entries |
| `save_knowledge` | Persist KB to PostgreSQL |

#### Control Tools (6 tools)
| Tool | Purpose |
|------|---------|
| `auto_tune_confidence` | Analyse valid/total ratio, recommend & apply threshold |
| `get_stage_analytics` | Per-stage latency, identify bottleneck |
| `get_ocr_strategy_stats` | Primary/fallback OCR usage & success rates |
| `self_heal_pipeline` | Auto-fix: reset circuit breaker, enable throttle, detect issues |
| `get_queue_pressure` | Queue depth, throttle state, water marks |
| `flush_inference_queue` | Emergency queue drain |

#### Session & Feedback Tools (10 tools — via coordinator)
Session management, feedback recording, knowledge persistence.

### 16.6 `WorkingMemory` — Bounded Chat Context

**File:** `agent/memory.py`

LLM context window management:
- Budget: 3072 tokens (~12,288 characters at ~4 chars/token heuristic)
- FIFO eviction of non-pinned entries when budget exceeded
- System prompt is always pinned
- Exports Ollama-compatible message format
- Session-scoped scratchpad (key-value store)

### 16.7 `SessionManager` — Multi-User Sessions

**File:** `agent/sessions.py`

- TTL: 30 minutes of inactivity
- Max sessions: 100 (LRU eviction)
- Per-session: conversation turns, metadata, context summary

### 16.8 `KnowledgeBase` — Operational Intelligence

**File:** `agent/knowledge.py`

In-memory knowledge base that learns from pipeline operations:
- **Plate frequency tracking** — which plates are seen most often
- **Camera error profiles** — per-camera OCR error rates, common confusions
- **Cross-camera correlation** — plates seen across multiple cameras
- **Operator feedback** — correction records that teach the system
- **Anomaly detection** — spike detection in plate frequency
- Persistable to PostgreSQL (`agent_knowledge` table)

### 16.9 `AuditTrail` — Compliance Logging

**File:** `agent/audit.py`

Write-behind audit logger for every agent action:
- Buffers entries (limit: 50)
- Batch-flushes to `agent_audit_log` table
- Records: session_id, intent, agent_role, action, arguments, result, success, elapsed_ms

### 16.10 `AutonomousMonitor` — Background Health

**File:** `agent/monitor.py`

Background asyncio task (60s interval) that:
1. **System health check** — VRAM (>75% warn, >90% critical), pipeline state, circuit breaker auto-reset
2. **Queue pressure check** — queue fill (>50% info, >80% warning + auto-flush)
3. **OCR health check** — primary success rate (<50% warn), fallback usage (>40% warn)

Generates `HealthAlert`s with severity levels and auto-remediation flags. Broadcasts via WebSocket.

### 16.11 Sub-Agent System

**File:** `agent/sub_agents.py`

Four role-specialised sub-agents with curated tool subsets:

| Role | Display Name | Tool Count | Focus |
|------|-------------|------------|-------|
| `OCR_QUALITY` | OCR Quality Specialist | 12 | OCR errors, confidence, camera diagnostics |
| `ANALYTICS` | Analytics Analyst | 10 | Detection patterns, plate frequency, trends |
| `OPERATIONS` | Operations Manager | 13 | Health, config, thresholds, self-healing |
| `GENERAL` | General Assistant | All 37 | Fallback for unclassified queries |

---

## 17. API Layer — `api/`

### 17.1 REST Endpoints

| Method | Path | Handler | Purpose |
|--------|------|---------|---------|
| `GET` | `/health` | `health.py` | Deep/shallow health check (GPU, Ollama, DB, streams) |
| `GET` | `/detections` | `detections.py` | Paginated detection query with filters |
| `GET` | `/stats` | `stats.py` | Prometheus metrics as JSON |
| `GET` | `/metrics` | `metrics.py` | Prometheus text format scrape endpoint |
| `POST` | `/sources` | `sources.py` | Add camera source |
| `GET` | `/sources` | `sources.py` | List camera sources |
| `DELETE` | `/sources/{camera_id}` | `sources.py` | Remove camera source |
| `POST` | `/api/agent/chat` | `agent_chat.py` | Send message to agent |
| `GET` | `/api/agent/status` | `agent_chat.py` | Agent status (uptime, sessions, tools) |
| `POST` | `/api/agent/feedback` | `agent_chat.py` | Submit operator feedback |
| `GET` | `/api/agent/sessions` | `agent_chat.py` | List active sessions |
| `DELETE` | `/api/agent/sessions/{id}` | `agent_chat.py` | Delete session |

### 17.2 WebSocket Endpoints

| Path | Handler | Purpose |
|------|---------|---------|
| `WS /ws/events` | `ws_events.py` | Real-time plate detection events via Redis pub/sub (`uv:events`) |
| `WS /ws/agent` | `ws_agent.py` | Streaming agent reasoning (thought → tool_call → observation → answer → done) |

### 17.3 Middleware Stack

| Middleware | Purpose |
|-----------|---------|
| `RequestLoggingMiddleware` | Logs method, path, status, latency_ms for every request |
| `APIKeyMiddleware` | SHA-256 hashed API key validation via `X-API-Key` header; skips public paths |
| `RateLimitMiddleware` | Sliding-window per-IP rate limiter (default: 60 req/min); exempt for `/health`, `/metrics` |
| `SecurityHeadersMiddleware` | Adds `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`, `Cache-Control` |

---

## 18. Visualiser — `visualizer/`

Eight Streamlit debug pages for development and monitoring:

| Page | File | Function |
|------|------|----------|
| Pipeline Stats | `vis_pipeline_stats.py` | Prometheus metrics dashboard with auto-refresh, VRAM usage graphs |
| Frame Sampler | `vis_frame_sampler.py` | 3-column detection grid showing sampled frames |
| Vehicle Detection | `vis_vehicle_detection.py` | Vehicle class distribution + confidence histograms |
| Plate Detection | `vis_plate_detection.py` | Confidence-colour-coded plate gallery |
| Crop & Straighten | `vis_crop_straighten.py` | Before/after deskew comparisons with expander |
| Enhancement | `vis_enhancement.py` | CLAHE/unsharp/NLM quality proxy metrics |
| OCR Output | `vis_ocr_output.py` | Per-engine filter + character diff visualisation |
| Post-Processing | `vis_postprocess.py` | Validation status breakdown + correction statistics |

---

## 19. Infrastructure — Docker, Database, Monitoring Stack

### 19.1 Docker Compose (8 Services)

**File:** `docker-compose.yml`

| Service | Image | Purpose | Notes |
|---------|-------|---------|-------|
| `app` | Custom (multi-stage) | Uni_Vision application | GPU reservation, depends on all services |
| `postgres` | `postgres:16-alpine` | Primary database | Volume: `pgdata` |
| `ollama` | `ollama/ollama` | LLM runtime | GPU reservation, volume: `ollama_data` |
| `ollama-init` | `curlimages/curl` | Model setup | Pulls Gemma 4 E2B, creates custom Modelfiles |
| `minio` | `minio/minio` | S3-compatible object storage | Volume: `minio_data` |
| `redis` | `redis:7-alpine` | Cache + pub/sub | Volume: `redis_data` |
| `prometheus` | `prom/prometheus` | Metrics collection | 15s scrape interval, 10 alert rules |
| `grafana` | `grafana/grafana` | Dashboards | Pre-provisioned datasource + dashboard |

### 19.2 Dockerfile (Multi-Stage Build)

```
Builder stage:    python:3.12-slim → install deps, build wheel
Runtime stage:    nvidia/cuda:12.4.1-runtime-ubuntu22.04
                  Non-root user (uid 1000)
                  Healthcheck every 30s (curl /health)
```

### 19.3 Database Migrations (Alembic)

| Migration | Description |
|-----------|-------------|
| `001_initial_schema.py` | `detection_events`, `camera_sources`, `ocr_audit_log` tables |
| `002_perf_indexes.py` | `pg_trgm` extension, trigram index on `plate_number`, partial indexes |
| `003_agent_knowledge.py` | `agent_knowledge` table for KB persistence |
| `004_agent_audit.py` | `agent_audit_log` table for agentic action audit |

### 19.4 Ollama Model Configuration

Two custom Modelfiles:

**`Modelfile.ocr`** — Gemma 4 E2B configured as OCR text extraction specialist:
```
FROM gemma4:e2b
SYSTEM "You are a license plate OCR extraction specialist..."
PARAMETER temperature 0.1
PARAMETER num_ctx 4096
```

**`Modelfile.adjudicator`** — Gemma 4 E2B configured as adjudication specialist:
```
FROM gemma4:e2b
SYSTEM "You are a license plate text adjudication specialist..."
PARAMETER temperature 0.1
```

### 19.5 Prometheus Alert Rules (10 Rules)

| Alert | Condition |
|-------|-----------|
| `VRAMUsageCritical` | VRAM > 90% for 5 minutes |
| `HighFrameDropRate` | Frame drops > 10% for 5 minutes |
| `OCRSuccessRateLow` | OCR success < 50% for 10 minutes |
| `PipelineLatencyHigh` | P95 latency > 5s for 5 minutes |
| `InferenceQueueBacklog` | Queue depth > 8 for 2 minutes |
| `DispatchErrors` | Dispatch errors > 0 for 5 minutes |
| `StreamDisconnected` | Any camera stream down for 1 minute |
| `DatabaseConnectionFailed` | DB health check failing for 1 minute |
| `OllamaUnreachable` | Ollama API unreachable for 2 minutes |
| `HighFallbackUsage` | Fallback OCR > 40% for 10 minutes |

---

## 20. Build & DevOps — Makefile, pyproject.toml, Scripts

### 20.1 Makefile Targets (17 targets)

| Target | Purpose |
|--------|---------|
| `install` | Install package in editable mode with dev extras |
| `lint` | Run ruff check + format check |
| `format` | Run ruff format |
| `typecheck` | Run mypy (strict mode) |
| `test` | Run pytest |
| `test-cov` | Run pytest with coverage report |
| `build` | Build Docker image |
| `up` / `down` | Docker Compose up/down |
| `migrate` | Run Alembic migrations |
| `download-models` | Download and convert YOLO models |
| `smoke` | Run smoke test script |
| `clean` | Remove __pycache__, .pyc, build artefacts |

### 20.2 pyproject.toml

- **Linter:** Ruff (fast Python linting and formatting)
- **Type checker:** mypy (strict mode)
- **Test runner:** pytest with `asyncio_mode = auto` (automatic async test support)

### 20.3 Scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_models.py` | Downloads YOLOv8n, exports to ONNX, converts to TensorRT |
| `scripts/init-ollama.sh` | 5-step: wait for Ollama → pull Gemma 4 E2B → create custom models → validate → list |
| `scripts/init-ollama.ps1` | PowerShell equivalent of init-ollama.sh |
| `scripts/smoke_test_agent.py` | 3 async tests for agent health verification |

---

## 21. Data Flow — End-to-End Frame Journey

### 21.1 Legacy Fixed Pipeline (When Manager Disabled)

```
RTSP Camera ──► RTSPFrameSource (background thread)
                    │
                    ▼
              Per-camera ring buffer (50 frames)
                    │
                    ▼
              TemporalSampler ──► FPS gate + pHash dedup
                    │
                    ▼
              Inference Queue (10 frames, with backpressure throttling)
                    │
                    ▼
    ┌───────────────────────────────────────────┐
    │            GPU INFERENCE ZONE              │
    │                                           │
    │  [S2] VehicleDetector.detect(frame)       │
    │       └── List[BoundingBox] (vehicles)    │
    │                                           │
    │  [S3] PlateDetector.detect_in_roi(        │
    │           frame, vehicle_bbox)            │
    │       └── List[BoundingBox] (plates)      │
    │                                           │
    │  [S4] extract_plate_roi(frame, bbox)      │
    │       └── plate crop images               │
    │                                           │
    │  ═══ MEMORY FENCE (GPU sync) ═══         │
    │                                           │
    │  [S5] HoughStraightener.process(crop)     │
    │  [S6] PhotometricEnhancer.process(crop)   │
    │                                           │
    │  [S7] OCRStrategy.extract(crop, context)  │
    │       └── OCRResult                       │
    │                                           │
    │  [S8] CognitiveOrchestrator.validate(     │
    │           result, crop)                   │
    │       └── ProcessedResult                 │
    └───────────────────────────────────────────┘
                    │
                    ▼
    MultiDispatcher.dispatch(record, plate_image)
        ├── SlidingWindowDeduplicator.is_duplicate()?
        │   └── If yes: discard
        ├── PostgresClient.insert_detection()
        ├── ObjectStoreArchiver.upload_plate_image()
        ├── Redis PUBLISH "uv:events"
        └── If low confidence: insert_audit_log()
```

### 21.2 Dynamic Manager Pipeline (When Manager Enabled)

```
RTSP Camera ──► RTSPFrameSource ──► TemporalSampler ──► Inference Queue
                                                              │
                                                              ▼
                                                    Pipeline.enqueue_frame()
                                                              │
                                                              ▼
                                                   ManagerAgent.process_frame(frame)
                                                              │
                    ┌─────────────────────────────────────────┤
                    │                                         │
                    ▼                                         │
           ContextAnalyzer.analyze(frame)                     │
           ├── SceneTransitionDetector.observe()              │
           └── TemporalTracker.get_capability_hints()         │
                    │                                         │
                    ▼                                         │
           FrameContext{scene=TRAFFIC,                         │
                       caps={vehicle_detection,                │
                             plate_detection,                  │
                             plate_ocr,                        │
                             text_validation}}                 │
                    │                                         │
                    ▼                                         │
           ComponentResolver.resolve_capabilities()            │
           (checks local registry, queries HuggingFace/PyPI,  │
            downloads missing, wraps as CVComponent)           │
                    │                                         │
                    ▼                                         │
           PipelineComposer.compose(context)                   │
           └── PipelineBlueprint{stages=[                      │
                  StageSpec(VEHICLE_DET, in="frame"),          │
                  StageSpec(PLATE_DET, in="frame"),            │
                  StageSpec(PLATE_OCR, in="plate_crops"),      │
                  StageSpec(TEXT_VALIDATION, in="ocr_results") │
               ]}                                              │
                    │                                         │
                    ▼                                         │
           ConflictResolver.check_blueprint()                  │
           └── ConflictReport{is_feasible=True}                │
                    │                                         │
                    ▼                                         │
           LifecycleManager.ensure_loaded(component_ids)       │
           (VRAM check → LRU evict if needed → load → warmup) │
                    │                                         │
                    ▼                                         │
           Execute Blueprint (stage by stage):                 │
           ├── detector.execute(frame) → detections            │
           ├── plate_det.execute(frame) → plate_crops          │
           ├── ocr.execute(plate_crops) → ocr_results          │
           └── validator.execute(ocr_results) → final          │
                    │                                         │
                    ▼                                         │
           PipelineExecutionResult                             │
           ├── FeedbackLoop.record_result()                    │
           ├── AdaptationEngine.ingest_result()                │
           └── QualityScorer.record_execution()                │
                    │                                         │
                    ▼                                         │
           Pipeline._dispatch_dynamic_results()                │
           (bridges final_output → DetectionRecord →           │
            MultiDispatcher.dispatch())                        │
```

### 21.3 Agent Conversation Flow

```
User: "Why is camera toll-gate-2 showing low confidence?"
         │
         ▼
    POST /api/agent/chat
         │
         ▼
    AgentCoordinator.chat()
    ├── classify_intent("Why is camera...") → DIAGNOSTICS
    ├── route_to_role(DIAGNOSTICS) → OPERATIONS
    └── MultiAgentRouter.route(msg, intent=DIAGNOSTICS)
              │
              ▼
         AgentLoop.run() — OPERATIONS sub-agent (13 tools)
              │
              ├── [Step 1] LLM thinks: "I should diagnose this camera"
              │   Action: diagnose_camera(camera_id="toll-gate-2")
              │   Observation: {error_rate: 0.15, avg_confidence: 0.58, ...}
              │
              ├── [Step 2] LLM thinks: "Low confidence, check OCR strategy"
              │   Action: get_ocr_strategy_stats()
              │   Observation: {primary_success: 0.42, fallback_usage: 0.58}
              │
              └── [Step 3] LLM provides answer:
                  "Camera toll-gate-2 has a 15% error rate with average
                   confidence of 0.58. The primary OCR engine is only
                   succeeding 42% of the time, falling back to EasyOCR
                   58% of the time. I recommend..."
              │
              ▼
         AuditTrail.record() — logs all 3 tool calls
         ChatResponse → User
```

---

## 22. Key Architectural Decisions

### 22.1 Gemma 4 E2B's Native Multimodal Capability
Gemma 4 E2B is a natively multimodal model that handles text, images, and audio directly. Unlike previous text-only LLMs, it can process plate images via its built-in vision encoder. Dedicated CV models (YOLO, etc.) are still used for specialized detection tasks where they offer better speed and accuracy, while Gemma 4 E2B handles:
- High-quality OCR via its vision capability
- Full 7.2 GB model fits on GPU with ~2168 MB headroom — NO CPU offload needed
- Remaining ~2 GB for multiple fast CV models
- Native image understanding eliminates the need for separate image encoding pipelines

### 22.2 Why Dynamic Pipeline Assembly
Different cameras may view different scenes (traffic, parking, surveillance). A fixed detection pipeline wastes GPU on cameras that don't need specialised processing. The Manager Agent:
- Analyses each camera's scene type
- Provisions only the needed capabilities
- Downloads specialised models when general ones underperform
- Adapts in real-time as conditions change (day→night, traffic→empty)

### 22.3 Why Dual Pipeline Paths
The legacy fixed pipeline (S0-S8) provides:
- Guaranteed low-latency for detection tasks (no LLM reasoning overhead)
- Fallback when the Manager Agent is disabled or unavailable
- Deterministic behaviour for testing and benchmarking

The dynamic Manager pipeline provides:
- Scene-aware capability provisioning
- Self-healing via adaptation engine
- Internet-sourced component discovery
- Multi-model consensus

### 22.4 Why VRAM Regions
Partitioning VRAM into named regions prevents:
- LLM weights being evicted by a greedy CV model
- CV model allocations starving the KV cache
- Memory fragmentation from uncoordinated allocations

The memory fence between vision and LLM stages ensures no temporal overlap.

### 22.5 Why Perceptual Dedup at Ingestion
Running YOLO on every frame (30+ FPS) would saturate the GPU. pHash deduplication at ingestion:
- Discards 60-90% of frames (stationary cameras)
- < 0.3 ms per frame (pure CPU)
- Hamming distance threshold is tunable (default: 5 of 64 bits)
- Zero VRAM cost

### 22.6 Why Async Everything
Every I/O-bound operation is async:
- RTSP reads in background threads (CPU)
- Database operations via asyncpg
- S3 uploads via aioboto3
- LLM calls via httpx
- Redis pub/sub for event broadcast

GPU inference is sequential by design (single consumer) to avoid CUDA context switching overhead.

---

*This document covers the complete Uni_Vision backend as of Phase 8 (Self-Assembling Autonomous Pipeline). For deployment instructions, see `DEPLOYMENT.md`. For product requirements, see `prd.md`. For the original system specification, see `spec.md`.*
