# Uni_Vision — System Architecture Specification
## Real-Time Visual Intelligence Pipeline for Automated Vehicle Identification

---

| Field | Value |
|---|---|
| **Document Type** | Architecture Specification — Non-Executable |
| **System Codename** | Uni_Vision |
| **Version** | 0.1.0-draft |
| **Date** | 2026-03-11 |
| **Author Role** | Principal AI Systems Architect |
| **Hardware Target** | NVIDIA RTX 4070 — 8192 MB VRAM (hard ceiling) |
| **Primary Runtime** | Python 3.10+ / CUDA 11.8+ / Ollama (externalized LLM) |
| **Classification** | Internal — Technical Reference |

---

## Table of Contents

1. [Architectural Philosophy](#1-architectural-philosophy)
2. [Hardware Memory Model](#2-hardware-memory-model)
3. [Pipeline Topology & Tensor Flow](#3-pipeline-topology--tensor-flow)
4. [Stage Specifications](#4-stage-specifications)
5. [Zero-Copy Memory Management Protocol](#5-zero-copy-memory-management-protocol)
6. [Asynchronous Task Queue Architecture](#6-asynchronous-task-queue-architecture)
7. [Dynamic Hardware Offloading Strategy](#7-dynamic-hardware-offloading-strategy)
8. [Agentic Orchestration Layer](#8-agentic-orchestration-layer)
9. [Design Patterns & Structural Contracts](#9-design-patterns--structural-contracts)
10. [Modular Directory Structure](#10-modular-directory-structure)
11. [Dependency Injection & Interface Segregation](#11-dependency-injection--interface-segregation)
12. [Observability & Diagnostics Architecture](#12-observability--diagnostics-architecture)
13. [Failure Taxonomy & Recovery Protocols](#13-failure-taxonomy--recovery-protocols)
14. [Appendix A — VRAM Lifecycle Trace](#appendix-a--vram-lifecycle-trace)
15. [Appendix B — Interface Contracts (Abstract)](#appendix-b--interface-contracts-abstract)

---

## 1. Architectural Philosophy

### 1.1 Governing Principles

The Uni_Vision architecture is governed by five non-negotiable axioms derived from the hardware constraint of a single 8GB VRAM GPU operating a multi-stage visual intelligence pipeline alongside a 9-billion-parameter multimodal language model:

**P1 — Memory is the Primary Constraint, Not Compute.**
Every architectural decision is subordinate to the 8192 MB VRAM ceiling. Compute throughput is secondary. A pipeline that is fast but exceeds VRAM by a single megabyte is a pipeline that crashes. Designs that trade latency for memory predictability are always preferred.

**P2 — Zero-Copy Unless Proven Impossible.**
Tensor data must never be duplicated across memory domains without explicit justification. Every CPU↔GPU transfer is an admission of architectural failure or a deliberate, profiled offloading decision. The default posture is: if a tensor exists on the GPU, no second copy exists on the CPU unless that tensor has been explicitly released from VRAM.

**P3 — Sequential Exclusivity of GPU-Bound Inference.**
No two neural network forward passes may occupy VRAM simultaneously unless their combined resident footprint has been statically verified to fit within the allocation budget. The vision models and the LLM orchestrator must be time-sliced, not concurrent. The GPU is a single-tenant resource during inference.

**P4 — The LLM is the Orchestrator, Not Middleware.**
No framework-level orchestration layer (LangChain, LlamaIndex, AutoGen, or equivalent) exists in this architecture. The Qwen 3.5 9B model, served via Ollama, is the reasoning engine. The application layer is a thin, deterministic Python harness that issues prompts, parses structured responses, catches failures, and re-prompts with error context. Agentic control flow lives in the prompt, not in Python abstractions.

**P5 — Every Stage is a Replaceable Unit.**
Each pipeline stage adheres to a strict interface contract. The vehicle detector, the plate localizer, the enhancement chain, and the OCR/LLM orchestrator are independently substitutable without modifying adjacent stages. This is achieved through protocol-based typing (Python `Protocol` classes), not inheritance hierarchies.

### 1.2 Architectural Style

The system is a **bounded-memory sequential pipeline** with:
- **Synchronous data flow** within a single detection event (frame → vehicle → plate → enhance → OCR → dispatch)
- **Asynchronous decoupling** between stream ingestion and the detection pipeline (producer-consumer via bounded queues)
- **Externalized inference** for the LLM (Ollama process, HTTP API boundary)
- **Internalized inference** for vision models (in-process PyTorch/ONNX/TensorRT, shared CUDA context)

This is not a microservices architecture. It is a **monolithic pipeline with interface-segregated internal boundaries**, deployed as a single containerized process with an externalized LLM sidecar.

---

## 2. Hardware Memory Model

### 2.1 VRAM Budget — Static Allocation Table

The 8192 MB VRAM is partitioned into four non-overlapping regions. Overflow in any region triggers a cascading OOM that terminates the pipeline. The allocations are enforced at initialization and monitored continuously at runtime.

```
┌─────────────────────────────────────────────────────────────────┐
│                    VRAM: 8192 MB (RTX 4070)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────┐              │
│  │  REGION A — LLM Weights (Qwen 3.5 9B Q4_K_M) │  5632 MB    │
│  │  Static. Loaded once at boot. Never evicted.  │  (5.5 GB)   │
│  └───────────────────────────────────────────────┘              │
│                                                                 │
│  ┌───────────────────────────────────────────────┐              │
│  │  REGION B — LLM KV Cache                      │  512–1024MB │
│  │  Dynamic. Grows per token. Capped at 4096 tok │  (0.5–1 GB) │
│  └───────────────────────────────────────────────┘              │
│                                                                 │
│  ┌───────────────────────────────────────────────┐              │
│  │  REGION C — Vision Model Workspace            │  1024 MB    │
│  │  Time-sliced. YOLOv8 vehicle + LPD plate det  │  (1.0 GB)   │
│  │  INT8 quantized. TensorRT/ONNX Runtime.       │              │
│  │  Includes: weights + input tensors + output    │              │
│  └───────────────────────────────────────────────┘              │
│                                                                 │
│  ┌───────────────────────────────────────────────┐              │
│  │  REGION D — System / Display / CUDA Overhead  │  512 MB     │
│  │  Non-negotiable. OS display server, CUDA ctx  │  (0.5 GB)   │
│  └───────────────────────────────────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 RAM (System Memory) Budget

System RAM is not the bottleneck but must be managed to prevent swap thrashing:

| Component | Estimated RAM | Notes |
|---|---|---|
| Python process + libraries | 400–600 MB | NumPy, OpenCV, FastAPI, etc. |
| Frame buffer queue (bounded) | 200–400 MB | 50 frames × ~4–8 MB each (1080p BGR) |
| Image preprocessing tensors | 100–200 MB | Transient; freed after each event |
| PostgreSQL connection pool | 50–100 MB | 4–8 persistent connections |
| Redis client buffers | 50 MB | Pub/Sub + task queue overhead |
| Ollama process (CPU-side) | 200–400 MB | Tokenizer, scheduler, HTTP server |
| **Total estimated** | **1.0–1.8 GB** | Well within 16–32 GB system RAM |

### 2.3 Memory Lifecycle per Detection Event

A single detection event (one vehicle, one plate, one OCR result) traces the following memory lifecycle across CPU and GPU:

```
TIME ──────────────────────────────────────────────────────────────▶

CPU RAM:
  ┌─────┐
  │Frame│ ← decoded from stream (BGR uint8, ~6MB @ 1080p)
  │numpy│
  └──┬──┘
     │ UPLOAD (cudaMemcpyAsync, pinned memory, zero-copy if possible)
     ▼
GPU VRAM (Region C):
  ┌──────────────┐
  │ Frame Tensor  │ ← float32 normalized, ~24MB @ 1080p
  │ (YOLO input)  │    OR resized to 640×640 → ~4.9MB float32
  └──────┬───────┘
         │ YOLOv8 vehicle detection forward pass
         ▼
  ┌──────────────┐
  │ BBox Tensors  │ ← N × [x1,y1,x2,y2,conf,cls] — negligible size
  └──────┬───────┘
         │ Crop vehicle ROI → LPD input (on GPU, no round-trip)
         ▼
  ┌──────────────┐
  │ Plate Tensor  │ ← LPD forward pass on vehicle ROI crop
  │ (LPD output)  │    Plate bbox coordinates — negligible
  └──────┬───────┘
         │ Crop plate region (GPU tensor slice, zero-copy)
         │ FREE: Frame tensor, YOLO intermediates
         ▼
  ┌──────────────┐
  │ Plate Image   │ ← Small tensor (~200×600 px) — ~0.5 MB
  │ (cropped)     │
  └──────┬───────┘
         │ DOWNLOAD to CPU (cudaMemcpy → numpy) for preprocessing
         │ FREE: All GPU Region C tensors for this event
         ▼
CPU RAM:
  ┌──────────────┐
  │ Plate numpy   │ ← Geometric correction, CLAHE, bilateral filter
  │ (enhanced)    │    All preprocessing on CPU (OpenCV, ~2–5ms)
  └──────┬───────┘
         │ Encode to base64/PNG → include in LLM prompt
         │ POST to Ollama HTTP API (localhost)
         ▼
GPU VRAM (Region A+B):
  ┌──────────────────┐
  │ Qwen 3.5 9B      │ ← Multimodal inference: image tokens + prompt
  │ forward pass      │    KV cache grows within Region B budget
  │ (Ollama-managed)  │    Output: structured plate text + confidence
  └──────┬───────────┘
         │ HTTP response → Python process
         ▼
CPU RAM:
  ┌──────────────┐
  │ OCR Result    │ ← Parsed text, confidence, validation status
  │ (structured)  │ → Post-processing → PostgreSQL → REST dispatch
  └──────────────┘
         │ FREE: plate numpy, all transient CPU buffers
         │ Ollama KV cache: reset on next request (or context reuse)
```

**Critical observation:** GPU Region C (vision models) and GPU Region A+B (LLM) are **never simultaneously active** for inference on the same detection event. The vision models complete and release their transient VRAM before the LLM is invoked. This is the **sequential exclusivity** guarantee from Principle P3.

---

## 3. Pipeline Topology & Tensor Flow

### 3.1 Logical Pipeline Stages

```
S0: Stream Ingestion          [CPU]  → Frame(numpy.ndarray, uint8, BGR)
S1: Frame Sampling + Dedup    [CPU]  → Frame | ∅ (discarded)
S2: Vehicle Detection         [GPU]  → List[VehicleBBox]
S3: Plate Localisation        [GPU]  → List[PlateBBox] (within vehicle ROI)
S4: Plate Cropping            [GPU→CPU] → PlateImage(numpy.ndarray)
S5: Geometric Correction      [CPU]  → PlateImage(deskewed)
S6: Photometric Enhancement   [CPU]  → PlateImage(enhanced)
S7: LLM OCR + Reasoning       [GPU via Ollama] → OCRResult(text, conf, metadata)
S8: Post-Processing & Dispatch [CPU]  → DetectionRecord → DB/API/MQ
```

### 3.2 Hardware Domain Transitions

Each arrow below represents a data movement between hardware domains. Each transition is profiled for latency and memory cost.

```
            CPU                          GPU (CUDA)                    GPU (Ollama)
            ───                          ──────────                    ────────────
  S0,S1 ──────┐
   (ingest,   │
    sample)   │
              │── UPLOAD (pinned) ──────▶ S2 (YOLO vehicle det)
              │                          │
              │                          │── in-GPU crop ──▶ S3 (LPD plate det)
              │                          │
              │◀── DOWNLOAD (plate) ─────│── S4 (crop + transfer)
              │                          │   FREE Region C tensors
  S5 (deskew) │
  S6 (enhance)│
              │── HTTP POST (base64) ───────────────────────▶ S7 (Qwen 3.5 OCR)
              │                                              │
              │◀── HTTP RESPONSE ────────────────────────────│
  S8 (post-   │
   process,   │
   dispatch)  │
```

**GPU↔CPU transitions per detection event: exactly 2.**
1. CPU→GPU: Frame upload for vehicle detection (S1→S2)
2. GPU→CPU: Cropped plate image download after plate localisation (S4)

The LLM invocation (S7) occurs over HTTP to the Ollama process, which manages its own CUDA context. From the pipeline's perspective, this is an inter-process call, not a direct CUDA memory transfer.

### 3.3 Tensor Format Specifications

| Stage | Tensor Shape | dtype | Domain | Size (typical) |
|---|---|---|---|---|
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

---

## 4. Stage Specifications

### S0 — Stream Ingestion

| Property | Specification |
|---|---|
| Input | RTSP URL / IP camera address / local video file |
| Output | Raw BGR frame as `numpy.ndarray` (uint8) |
| Library | OpenCV `cv2.VideoCapture` / FFmpeg subprocess |
| Concurrency | One thread per camera source (I/O-bound, GIL-safe) |
| Reconnection | Exponential backoff: 1s, 2s, 4s, 8s, max 16s. Alert after 3 failures. |
| Configuration | Per-source YAML: `camera_id`, `source_url`, `location_tag`, `fps_target`, `enabled` |
| Memory | Frames decoded directly into pre-allocated numpy buffer (ring buffer pattern) |
| Thread safety | Each stream thread writes to its own slot in a bounded `queue.Queue(maxsize=50)` |

### S1 — Frame Sampling & Deduplication

| Property | Specification |
|---|---|
| Input | Raw frame from S0 queue |
| Output | Accepted frame with metadata, or discard signal |
| Sampling | FPS-gated: accept 1 frame per `1/fps_target` second interval |
| Deduplication | Perceptual hash (pHash, 64-bit DCT-based) on downscaled grayscale (32×32) |
| Hash distance threshold | Hamming distance ≤ 5 → discard as duplicate |
| Metadata attachment | `camera_id`, `timestamp_utc` (monotonic), `frame_index` (monotonic counter) |
| Target | ≥ 20% reduction in downstream processing calls under low-traffic conditions |

### S2 — Vehicle Detection

| Property | Specification |
|---|---|
| Model | YOLOv8n or YOLOv8s (Ultralytics), INT8 quantized |
| Export format | TensorRT engine (`.engine`) or ONNX Runtime (`.onnx`) |
| Classes | `car` (0), `truck` (1), `bus` (2), `motorcycle` (3) |
| Input resolution | 640×640 (letterbox-padded, aspect-preserved) |
| Confidence threshold | ≥ 0.60 (configurable) |
| NMS IoU threshold | 0.45 |
| Output | `List[VehicleBBox]`: each `(x1, y1, x2, y2, confidence, class_id)` |
| Latency target | ≤ 80ms per frame (GPU, INT8) |
| VRAM footprint | ~200–400 MB (model weights + input/output tensors) |
| Failure mode | No vehicle detected → frame discarded, logged to audit trail |

### S3 — License Plate Localisation

| Property | Specification |
|---|---|
| Model | YOLO-variant License Plate Detector (LPD), INT8 quantized |
| Input | Cropped vehicle ROI tensor (from S2 bbox, GPU-resident) |
| Confidence threshold | ≥ 0.65 (configurable) |
| Output | `PlateBBox`: `(x1, y1, x2, y2, confidence)` — tightest enclosing box |
| Multi-plate policy | Select highest confidence detection |
| Recall target | ≥ 90% |
| False positive target | ≤ 5% |
| VRAM footprint | ~200–400 MB (shared Region C with S2, sequential execution) |
| Note | S2 and S3 models share Region C. S2 weights may be unloaded before S3 loads, or both may co-reside if combined footprint ≤ 1024 MB |

### S4 — Plate Cropping & GPU→CPU Transfer

| Property | Specification |
|---|---|
| Input | Full frame tensor (GPU) + plate bbox coordinates |
| Operation | Tensor slice on GPU → `cudaMemcpyDeviceToHost` → numpy array |
| Padding | Configurable margin (default: 5 pixels) to avoid clipping edge characters |
| Output | `PlateImage` as `numpy.ndarray` (uint8, BGR) on CPU |
| Post-transfer cleanup | All GPU Region C tensors freed (explicit `torch.cuda.empty_cache()` or ONNX session reset) |
| Latency | < 1ms (negligible, small tensor) |

### S5 — Geometric Correction (CPU)

| Property | Specification |
|---|---|
| Input | `PlateImage` (CPU numpy) |
| Skew detection | Hough Line Transform on binary-thresholded edge image |
| Correction | Affine transformation (rotation + translation) |
| Max correctable skew | ±30° |
| Skip threshold | \|skew\| ≤ 3° → no transformation applied |
| Output | Deskewed `PlateImage` (same shape, CPU numpy) |
| Library | OpenCV (`cv2.getRotationMatrix2D`, `cv2.warpAffine`) |

### S6 — Photometric Enhancement (CPU)

| Property | Specification |
|---|---|
| Input | Deskewed `PlateImage` (CPU numpy) |
| Sub-stages (sequential, individually toggleable): | |
| (a) Resize | Upscale to minimum 200px height, preserve aspect ratio (`cv2.resize`, `INTER_CUBIC`) |
| (b) CLAHE | Convert BGR→LAB, apply CLAHE on L-channel (`clipLimit=2.0`, `tileGridSize=(8,8)`), convert back |
| (c) Gaussian blur | Kernel `(3,3)`, σ auto-computed — mild noise suppression |
| (d) Bilateral filter | `d=9`, `sigmaColor=75`, `sigmaSpace=75` — edge-preserving smoothing |
| Output | Enhanced `PlateImage` (CPU numpy) |
| Configuration | Each sub-stage has an independent `enabled: true/false` flag in YAML |

### S7 — LLM OCR + Agentic Reasoning (GPU via Ollama)

| Property | Specification |
|---|---|
| Model | Qwen 3.5 9B (Q4_K_M GGUF) served by Ollama |
| Invocation | HTTP POST to `http://localhost:11434/api/chat` (Ollama API) |
| Input | System prompt + user message containing base64-encoded enhanced plate image |
| Multimodal | Early-fusion native vision — image tokens processed natively, no secondary vision encoder |
| Context window | Hard-capped at 4096 tokens per request (runtime parameter: `num_ctx: 4096`) |
| Output schema | Structured text: `plate_text`, `confidence`, `char_alternatives`, `reasoning_trace` |
| Output parsing | Application layer parses response. On parse failure: append error to context, re-prompt (max 2 retries) |
| Latency target | ≤ 2000ms per plate (including HTTP round-trip, tokenization, generation) |
| KV cache management | Ollama manages KV cache internally. Cache cleared between requests to prevent VRAM drift. |
| VRAM | Regions A + B: 5632 MB (weights) + 512–1024 MB (KV cache) |
| Fallback | If Ollama is unresponsive after 5s timeout: log error, skip LLM OCR, emit `low_confidence` record |

### S8 — Post-Processing & Dispatch (CPU)

| Property | Specification |
|---|---|
| Input | `OCRResult` from S7 |
| Character correction | Substitution map: `O↔0`, `I↔1`, `S↔5`, `B↔8`, `D↔0`, `Z↔2` (configurable, locale-aware) |
| Regex validation | Locale-specific pattern (default Indian: `^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$`) |
| Deduplication | Sliding window (default 10s): suppress repeated plate detections, keep highest confidence |
| Validation status | `valid` / `low_confidence` / `regex_fail` — determines routing |
| Storage | PostgreSQL: `detection_events` table. Images: S3-compatible store (MinIO / local FS) |
| Dispatch | REST API (FastAPI) + Redis Pub/Sub for real-time consumers |
| Dispatch latency | ≤ 2 seconds from S7 completion |

---

## 5. Zero-Copy Memory Management Protocol

### 5.1 Definition of Zero-Copy in This Context

"Zero-copy" in Uni_Vision means: **no unnecessary duplication of tensor data within or across memory domains.** Specifically:

1. A frame decoded into CPU RAM is not copied before GPU upload — it is uploaded directly from the decode buffer (or from a pinned memory region aliasing the decode buffer).
2. GPU-side tensor operations (ROI cropping for S3 input) are performed as **tensor views/slices**, not as copy-allocations.
3. The plate image downloaded from GPU to CPU in S4 is the only cross-domain copy in the detection path, and it is of a small tensor (~0.36 MB).

### 5.2 Pinned Memory Protocol

For CPU→GPU transfers, the frame buffer must use CUDA **pinned (page-locked) memory** to enable asynchronous DMA transfers without intermediate staging copies:

```
Allocation:
  frame_buffer = cuda.pagelocked_empty((1080, 1920, 3), dtype=np.uint8)
  
Upload:
  cuda_stream = cuda.Stream()
  gpu_tensor = cuda.mem_alloc(frame_buffer.nbytes)
  cuda.memcpy_htod_async(gpu_tensor, frame_buffer, cuda_stream)
  cuda_stream.synchronize()
```

Pinned memory is allocated once at pipeline initialization for each camera stream and reused across frames (ring buffer pattern). This eliminates per-frame allocation overhead and ensures the DMA engine can transfer directly from the pinned buffer to VRAM without an intermediate pageable copy.

### 5.3 GPU-Side Tensor View Operations

Within the GPU domain (S2→S3→S4), all ROI operations must use tensor views, not copies:

- **Vehicle ROI extraction:** Given YOLO bbox `(x1, y1, x2, y2)` on the frame tensor, the vehicle ROI is a slice of the original tensor: `frame_gpu[y1:y2, x1:x2, :]`. This is a view — same memory, different index bounds. No allocation.
- **Plate ROI extraction within vehicle ROI:** Same pattern. `vehicle_roi[py1:py2, px1:px2, :]` is a view.
- **Copies occur only** when the tensor must be resized (letterbox padding for YOLO input) or normalized (uint8→float32 conversion). These are necessary transformations, not redundant duplications.

### 5.4 Explicit Deallocation Discipline

Python's garbage collector and PyTorch's CUDA caching allocator are **not sufficient** for real-time VRAM management in a constrained environment. The pipeline must enforce explicit deallocation:

```
After S4 completes (plate crop downloaded to CPU):
  1. del frame_gpu_tensor        # Release reference
  2. del vehicle_roi_tensor      # Release reference
  3. del yolo_output_tensor      # Release reference
  4. torch.cuda.empty_cache()    # Return memory to CUDA driver
  —OR—
  (if using TensorRT/ONNX Runtime: session.io_binding().clear())
```

This explicit cleanup ensures that VRAM is returned to the free pool before the next detection event or before the Ollama LLM forward pass begins. Without it, PyTorch's caching allocator will hold onto "freed" blocks speculatively, preventing Ollama from accessing the memory it needs.

### 5.5 Memory Fencing Between Vision and LLM Domains

A **memory fence** is a logical synchronization point that guarantees all VRAM in Region C (vision models) has been released before any LLM inference begins. The fence operates as follows:

```
[S4 complete] → MEMORY FENCE → [S7 begin]
                    │
                    ├── Assert: Region C allocated == 0 MB
                    ├── torch.cuda.empty_cache() called
                    ├── torch.cuda.memory_allocated() verified < threshold
                    └── If assertion fails: ABORT event, log OOM-risk warning
```

This fence is not optional. It is a hard requirement derived from P3 (Sequential Exclusivity).

---

## 6. Asynchronous Task Queue Architecture

### 6.1 Two-Layer Queue Design

The pipeline uses a **two-layer queue architecture** that decouples I/O-bound stream ingestion from compute-bound inference:

```
LAYER 1 — Stream Queues (per camera)
┌──────────┐     ┌──────────────────────┐
│ Camera 0  │────▶│ Queue_cam0 (max=50)  │──┐
└──────────┘     └──────────────────────┘  │
┌──────────┐     ┌──────────────────────┐  │    ┌──────────────────────┐
│ Camera 1  │────▶│ Queue_cam1 (max=50)  │──┼───▶│ DISPATCHER           │
└──────────┘     └──────────────────────┘  │    │ (round-robin or      │
┌──────────┐     ┌──────────────────────┐  │    │  priority-weighted)   │
│ Camera 2  │────▶│ Queue_cam2 (max=50)  │──┤    └──────────┬───────────┘
└──────────┘     └──────────────────────┘  │               │
┌──────────┐     ┌──────────────────────┐  │               ▼
│ Camera 3  │────▶│ Queue_cam3 (max=50)  │──┘    LAYER 2 — Inference Queue
└──────────┘     └──────────────────────┘       ┌──────────────────────┐
                                                │ InferenceQueue       │
                                                │ (max=10, bounded)    │
                                                │ Single consumer      │
                                                │ (GPU is single-      │
                                                │  tenant)             │
                                                └──────────────────────┘
```

### 6.2 Layer 1 — Stream Queues

- **One `queue.Queue(maxsize=50)` per camera source.**
- Producer: dedicated I/O thread per camera (`threading.Thread`, daemon=True).
- Consumer: dispatcher thread reads from all queues using weighted round-robin.
- **Backpressure policy:** When `queue.full()`, the oldest frame is evicted (ring buffer semantics). This ensures the pipeline always processes the most recent data, never stale frames.
- Implemented with Python's `queue.Queue` — GIL-safe for I/O-bound producers.

### 6.3 Layer 2 — Inference Queue

- **Single `asyncio.Queue(maxsize=10)` for detection events.**
- Consumer: a single async worker coroutine that processes one detection event at a time (GPU single-tenancy).
- The inference consumer executes S2→S3→S4→S5→S6→S7→S8 sequentially for each event.
- **Why single consumer:** The GPU cannot serve multiple inference requests simultaneously within our VRAM budget. Parallelizing would require concurrent VRAM allocation for multiple events, violating the budget.
- **Backpressure:** If the inference queue is full, the dispatcher drops the incoming frame and increments a `frames_dropped` counter in the metrics registry.

### 6.4 Adaptive FPS Throttling

When the inference queue depth exceeds a configurable high-water mark (default: 8 of 10 slots), the dispatcher signals all stream producers to halve their effective FPS until the queue depth falls below the low-water mark (default: 3 of 10 slots). This is a **backpressure feedback loop** that prevents sustained queue overflow under burst traffic.

```
if inference_queue.qsize() > HIGH_WATER_MARK:
    for producer in stream_producers:
        producer.throttle(factor=0.5)
elif inference_queue.qsize() < LOW_WATER_MARK:
    for producer in stream_producers:
        producer.unthrottle()
```

---

## 7. Dynamic Hardware Offloading Strategy

### 7.1 Problem Statement

The 8GB VRAM budget is partitioned statically (Section 2.1), but actual usage within each region fluctuates. Vision models have transient activations. The LLM's KV cache grows with token count. A purely static allocation wastes memory when one region is idle while another is active. Dynamic offloading allows opportunistic use of freed VRAM.

### 7.2 Offloading Modes

**Mode 1 — GPU-Primary, CPU-Fallback (Default)**
All vision inference runs on GPU (Region C). If GPU memory is insufficient (e.g., Region B expanded unexpectedly), the vision inference falls back to CPU (ONNX Runtime CPU execution provider). This mode sacrifices ~3–5× latency on vision inference but prevents OOM.

**Mode 2 — Vision on GPU, Preprocessing on CPU (Standard Path)**
This is the normal operating mode: S2+S3 on GPU, S5+S6 on CPU. No dynamic decision needed. This is the steady-state.

**Mode 3 — Full CPU Mode (Degraded)**
If CUDA is unavailable (driver crash, GPU locked by another process), the entire pipeline operates on CPU. Vision models use ONNX Runtime CPU EP. Ollama uses CPU-only execution (if configured without GPU layers). Latency degrades to ~5–10× normal. The system remains functional but below performance targets.

### 7.3 Runtime VRAM Monitor

A lightweight daemon coroutine polls VRAM usage at 500ms intervals using `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` (or `nvidia-smi` parsing for the Ollama-managed region). The monitor:

1. **Logs** current VRAM per region to the metrics registry (Prometheus gauge).
2. **Alerts** if any region exceeds 90% of its budget.
3. **Triggers CPU fallback** for vision models if free VRAM drops below 256 MB (hard safety margin).
4. **Emits structured log** on every offloading decision for post-hoc analysis.

### 7.4 CPU↔GPU Tensor Offloading Rules

| Condition | Action |
|---|---|
| Free VRAM ≥ 1024 MB when vision inference begins | Normal GPU path (S2+S3 on GPU) |
| Free VRAM < 1024 MB but ≥ 512 MB | Run S2 on GPU, S3 on CPU (partial offload) |
| Free VRAM < 512 MB | Run S2+S3 on CPU (full offload) |
| Ollama KV cache exceeds 1024 MB | Abort LLM request, log `kv_cache_overflow`, use fallback OCR |
| `torch.cuda.OOMError` caught | Immediate CPU fallback for current event, alert raised |

---

## 8. Agentic Orchestration Layer

### 8.1 Architecture — LLM as Orchestrator, Not Middleware

The Qwen 3.5 9B model does not merely perform OCR. It serves as the **intelligent orchestrator** for the final stage of the pipeline. Given an enhanced plate image and structured context (camera ID, timestamp, vehicle class, detection confidence), it:

1. **Reads** the plate image using its native early-fusion vision capability.
2. **Extracts** the alphanumeric text with per-character confidence reasoning.
3. **Validates** the extracted text against its trained understanding of plate formats.
4. **Corrects** ambiguous characters using contextual reasoning (not just a substitution map).
5. **Returns** a structured result with a reasoning trace.

### 8.2 Prompt Architecture

The system prompt is the control plane. It must:

```
┌────────────────────────────────────────────────────────────────┐
│ SYSTEM PROMPT                                                  │
│                                                                │
│ 1. Role declaration: "You are a precision OCR extraction       │
│    engine for vehicle license plates."                         │
│                                                                │
│ 2. Output format: Explicitly defined, flat structure.          │
│    Favour XML tags matching Qwen 3.5's training format.        │
│    Example:                                                    │
│    <result>                                                    │
│      <plate_text>MH12AB1234</plate_text>                       │
│      <confidence>0.94</confidence>                             │
│      <reasoning>Characters are clear. 'B' confirmed by        │
│        vertical stroke analysis.</reasoning>                   │
│    </result>                                                   │
│                                                                │
│ 3. Failure mode: If unreadable, return:                        │
│    <result><plate_text>UNREADABLE</plate_text>                 │
│    <confidence>0.0</confidence></result>                       │
│                                                                │
│ 4. Strict prohibition on hallucination:                        │
│    "Never guess. If a character is ambiguous, report it as     │
│     ambiguous with alternatives."                              │
│                                                                │
│ 5. No conversational filler. No preamble. No sign-off.        │
│    Output the <result> block and nothing else.                 │
└────────────────────────────────────────────────────────────────┘
```

### 8.3 Error Recovery Loop

```
attempt = 0
MAX_RETRIES = 2

while attempt <= MAX_RETRIES:
    response = ollama_client.chat(model, messages)
    
    try:
        result = parse_xml_result(response.content)
        validate_schema(result)
        return result
    except ParseError as e:
        attempt += 1
        messages.append({
            "role": "user",
            "content": f"Your previous response could not be parsed. "
                       f"Error: {e}. Please output ONLY a valid <result> block."
        })

# All retries exhausted
return OCRResult(plate_text="PARSE_FAIL", confidence=0.0, status="llm_error")
```

### 8.4 Ollama Runtime Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `model` | `qwen3.5:9b-q4_K_M` | Q4_K_M quantization for 8GB budget |
| `num_ctx` | `4096` | Hard cap on context window to constrain KV cache |
| `temperature` | `0.1` | Near-deterministic for OCR extraction |
| `top_p` | `0.9` | Mild nucleus sampling |
| `repeat_penalty` | `1.1` | Mitigate repetition (may be silently ignored by Ollama — prompt design compensates) |
| `num_predict` | `256` | Max output tokens — plate text + reasoning trace |
| `stream` | `false` | Non-streaming for atomic response parsing |

### 8.5 Model Context Protocol (MCP) — Future Integration Point

The architecture designates MCP as the target integration protocol for Phase 2 agentic capabilities (tool use, multi-step reasoning, payment gateway interaction). In Phase 1, the LLM operates in a **single-turn, tool-less mode** — it receives an image and context, returns structured OCR output. MCP tooling definitions are not active in Phase 1 but the prompt structure is designed to be forward-compatible with MCP tool declarations.

---

## 9. Design Patterns & Structural Contracts

### 9.1 Applied Patterns

| Pattern | Application | Rationale |
|---|---|---|
| **Strategy** | OCR engine selection (LLM / EasyOCR fallback) | Swap recognition engine via config without code changes |
| **Pipeline (Pipes & Filters)** | S0→S8 sequential processing chain | Each stage transforms and forwards data through a uniform interface |
| **Producer-Consumer** | Stream threads → frame queues → inference worker | Decouples I/O-bound ingestion from compute-bound inference |
| **Protocol-Based Typing** | All stage interfaces defined as `typing.Protocol` | Enables dependency injection without inheritance coupling |
| **Singleton** | CUDA context, Ollama client, DB connection pool | Prevents resource duplication on constrained hardware |
| **Circuit Breaker** | Ollama HTTP calls, stream reconnection | Prevents cascading failures when external resource is down |
| **Observer** | Metrics emission, visualizer pub/sub | Decoupled monitoring without pipeline instrumentation overhead |
| **Ring Buffer** | Frame queues with eviction | Bounded memory, always-fresh data, no unbounded growth |
| **Memory Fence** | Explicit VRAM barrier between vision and LLM stages | Guarantees sequential exclusivity of GPU memory regions |
| **Retry with Context** | LLM parse failure recovery | Appends error to prompt context, re-invokes with correction guidance |

### 9.2 Interface Contracts (Protocol Classes)

Every pipeline stage must implement a protocol. These protocols are defined in a dedicated `contracts/` package and imported by both the stage implementations and the pipeline orchestrator. No stage has a direct import dependency on any other stage's concrete implementation.

```python
# Conceptual — not executable, illustrative of the contract structure

class FrameSource(Protocol):
    def read_frame(self) -> Optional[FramePacket]: ...
    def release(self) -> None: ...

class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> List[BoundingBox]: ...
    def warmup(self) -> None: ...
    def release(self) -> None: ...

class Preprocessor(Protocol):
    def process(self, image: np.ndarray) -> np.ndarray: ...

class OCREngine(Protocol):
    def extract(self, image: np.ndarray, context: DetectionContext) -> OCRResult: ...

class PostProcessor(Protocol):
    def validate(self, result: OCRResult) -> ProcessedResult: ...

class Dispatcher(Protocol):
    def dispatch(self, record: DetectionRecord) -> None: ...
```

### 9.3 Data Transfer Objects (DTOs)

All inter-stage data flows through immutable (frozen) dataclasses. No stage mutates data produced by a previous stage — it produces a new DTO.

```python
# Conceptual DTO hierarchy

@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    timestamp_utc: float
    frame_index: int
    image: np.ndarray          # uint8 BGR, CPU

@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

@dataclass(frozen=True)
class DetectionContext:
    camera_id: str
    timestamp_utc: float
    vehicle_bbox: BoundingBox
    plate_bbox: BoundingBox
    vehicle_class: str

@dataclass(frozen=True)
class OCRResult:
    plate_text: str
    raw_text: str
    confidence: float
    reasoning: str
    engine: str
    status: str                # 'valid', 'low_confidence', 'llm_error', 'fallback'

@dataclass(frozen=True)
class DetectionRecord:
    id: str                    # UUID
    camera_id: str
    plate_number: str
    raw_ocr_text: str
    ocr_confidence: float
    ocr_engine: str
    vehicle_class: str
    vehicle_image_path: str
    plate_image_path: str
    detected_at_utc: str       # ISO 8601
    validation_status: str
    location_tag: str
```

---

## 10. Modular Directory Structure

```
uni_vision/
│
├── pyproject.toml                      # PEP 621 project metadata, dependencies
├── .env.example                        # Environment variable template
├── config/
│   ├── default.yaml                    # Default pipeline configuration
│   ├── cameras.yaml                    # Camera source definitions
│   └── models.yaml                     # Model paths, quantization settings, thresholds
│
├── src/
│   └── uni_vision/
│       ├── __init__.py
│       │
│       ├── contracts/                  # Interface definitions (Protocol classes)
│       │   ├── __init__.py
│       │   ├── frame_source.py         # FrameSource protocol
│       │   ├── detector.py             # Detector protocol (vehicle + plate)
│       │   ├── preprocessor.py         # Preprocessor protocol
│       │   ├── ocr_engine.py           # OCREngine protocol
│       │   ├── post_processor.py       # PostProcessor protocol
│       │   ├── dispatcher.py           # Dispatcher protocol
│       │   └── dtos.py                 # All frozen dataclasses (FramePacket, BBox, etc.)
│       │
│       ├── ingestion/                  # S0 + S1: Stream capture + frame sampling
│       │   ├── __init__.py
│       │   ├── stream_capture.py       # OpenCV/FFmpeg stream reader (per-camera thread)
│       │   ├── frame_sampler.py        # FPS gating + pHash deduplication
│       │   └── queue_manager.py        # Bounded frame queues, ring buffer, backpressure
│       │
│       ├── detection/                  # S2 + S3: Vehicle + plate detection
│       │   ├── __init__.py
│       │   ├── vehicle_detector.py     # YOLOv8 vehicle detector (TensorRT/ONNX)
│       │   ├── plate_detector.py       # LPD plate localiser (TensorRT/ONNX)
│       │   └── gpu_memory.py           # VRAM monitor, memory fence, explicit deallocation
│       │
│       ├── preprocessing/              # S4 + S5 + S6: Crop, deskew, enhance
│       │   ├── __init__.py
│       │   ├── cropper.py              # Plate ROI extraction + GPU→CPU transfer
│       │   ├── straightener.py         # Hough Line Transform + affine correction
│       │   └── enhancer.py             # Resize, CLAHE, Gaussian, bilateral filter chain
│       │
│       ├── ocr/                        # S7: LLM-based OCR + fallback engines
│       │   ├── __init__.py
│       │   ├── llm_ocr.py             # Qwen 3.5 via Ollama — primary OCR engine
│       │   ├── prompt_templates.py     # System prompts, output format definitions
│       │   ├── response_parser.py      # XML response parser + validation
│       │   └── fallback_ocr.py         # EasyOCR/CRNN/PaddleOCR fallback (strategy pattern)
│       │
│       ├── postprocessing/             # S8: Validation, correction, dedup, dispatch
│       │   ├── __init__.py
│       │   ├── validator.py            # Regex validation, character correction map
│       │   ├── deduplicator.py         # Sliding window deduplication (10s default)
│       │   └── dispatcher.py           # REST API push, Redis Pub/Sub, DB write
│       │
│       ├── orchestrator/               # Pipeline assembly + execution control
│       │   ├── __init__.py
│       │   ├── pipeline.py             # Main pipeline loop: S0→S8 sequential execution
│       │   ├── container.py            # Dependency injection container (manual DI)
│       │   └── scheduler.py            # Adaptive FPS throttling, queue depth monitoring
│       │
│       ├── storage/                    # Database + object store adapters
│       │   ├── __init__.py
│       │   ├── postgres.py             # Async PostgreSQL client (asyncpg)
│       │   ├── object_store.py         # S3-compatible image storage (MinIO / local FS)
│       │   └── models.py               # SQLAlchemy / raw SQL table definitions
│       │
│       ├── api/                        # FastAPI HTTP layer
│       │   ├── __init__.py
│       │   ├── app.py                  # FastAPI application factory
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── health.py           # GET /health
│       │   │   ├── detections.py       # GET /detections
│       │   │   ├── sources.py          # POST /sources, DELETE /sources/{id}
│       │   │   └── metrics.py          # GET /metrics (Prometheus)
│       │   └── middleware/
│       │       ├── __init__.py
│       │       └── auth.py             # JWT / API key authentication
│       │
│       ├── monitoring/                 # Observability infrastructure
│       │   ├── __init__.py
│       │   ├── metrics.py              # Prometheus counters, gauges, histograms
│       │   ├── health.py               # Health check logic (stream status, GPU status)
│       │   └── vram_monitor.py         # Runtime VRAM polling + alerting
│       │
│       ├── visualizer/                 # Debug visualizer (Streamlit / HighGUI)
│       │   ├── __init__.py
│       │   ├── vis_frame_sampler.py
│       │   ├── vis_vehicle_detection.py
│       │   ├── vis_plate_detection.py
│       │   ├── vis_crop_straighten.py
│       │   ├── vis_enhancement.py
│       │   ├── vis_ocr_output.py
│       │   ├── vis_postprocess.py
│       │   └── vis_pipeline_stats.py
│       │
│       └── common/                     # Shared utilities (minimal)
│           ├── __init__.py
│           ├── config.py               # YAML config loader + env var resolver
│           ├── logging.py              # structlog JSON logger setup
│           └── exceptions.py           # Custom exception hierarchy
│
├── models/                             # Pre-trained model weights (gitignored)
│   ├── yolov8n_vehicle.engine          # TensorRT-exported vehicle detector
│   ├── yolov8n_plate.engine            # TensorRT-exported plate detector
│   └── README.md                       # Model provenance, download instructions
│
├── tests/
│   ├── conftest.py                     # Shared fixtures
│   ├── unit/
│   │   ├── test_frame_sampler.py
│   │   ├── test_vehicle_detector.py
│   │   ├── test_plate_detector.py
│   │   ├── test_straightener.py
│   │   ├── test_enhancer.py
│   │   ├── test_llm_ocr.py
│   │   ├── test_response_parser.py
│   │   ├── test_validator.py
│   │   ├── test_deduplicator.py
│   │   └── test_dispatcher.py
│   ├── integration/
│   │   ├── test_pipeline_e2e.py
│   │   └── test_ollama_integration.py
│   └── fixtures/
│       ├── sample_frames/              # Test images
│       ├── sample_plates/              # Cropped plate images
│       └── mock_responses/             # Ollama response mocks
│
├── scripts/
│   ├── export_tensorrt.py              # Convert ONNX → TensorRT engine
│   ├── benchmark_pipeline.py           # Latency + throughput benchmarking
│   └── seed_cameras.py                 # Seed camera config for development
│
├── docker/
│   ├── Dockerfile                      # Application container
│   ├── Dockerfile.ollama               # Ollama sidecar (Qwen 3.5 9B pre-loaded)
│   └── docker-compose.yaml             # Full stack: app + ollama + postgres + redis
│
└── docs/
    ├── spec.md                         # ← This document
    └── api_reference.md                # Auto-generated OpenAPI reference
```

---

## 11. Dependency Injection & Interface Segregation

### 11.1 Philosophy — Manual DI, No Framework

Dependency injection in Uni_Vision is **manual and explicit**. No DI framework (dependency-injector, injector, etc.) is used. The rationale:

1. DI frameworks add import-time overhead and magic that obscures the dependency graph.
2. In a performance-critical pipeline, every microsecond of initialization and every layer of indirection matters.
3. The dependency graph is small enough (< 15 concrete components) to wire manually in a single container module.

### 11.2 The Container

The `orchestrator/container.py` module is the **composition root**. It is the only module that knows about concrete implementations. All other modules depend only on protocols from `contracts/`.

```python
# Conceptual — illustrative of the DI wiring pattern

class PipelineContainer:
    """
    Composition root. Constructs all pipeline components
    and wires them together via protocol-typed references.
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # Storage layer
        self.db = PostgresClient(config.database)
        self.object_store = S3ObjectStore(config.storage)

        # Detection layer
        self.vehicle_detector: Detector = YOLOv8VehicleDetector(
            model_path=config.models.vehicle_detector_path,
            confidence_threshold=config.models.vehicle_confidence,
            device=config.hardware.device,
        )
        self.plate_detector: Detector = YOLOv8PlateDetector(
            model_path=config.models.plate_detector_path,
            confidence_threshold=config.models.plate_confidence,
            device=config.hardware.device,
        )

        # Preprocessing chain
        self.straightener: Preprocessor = HoughStraightener(config.preprocessing)
        self.enhancer: Preprocessor = CLAHEBilateralEnhancer(config.preprocessing)

        # OCR layer (strategy pattern)
        self.primary_ocr: OCREngine = OllamaLLMOCR(
            base_url=config.ollama.base_url,
            model=config.ollama.model,
            system_prompt=load_system_prompt(),
            timeout=config.ollama.timeout,
        )
        self.fallback_ocr: OCREngine = EasyOCRFallback(config.ocr_fallback)

        # Post-processing
        self.validator: PostProcessor = RegexCharValidator(config.postprocessing)
        self.deduplicator = SlidingWindowDeduplicator(config.postprocessing.dedup_window)
        self.dispatcher: Dispatcher = MultiDispatcher(
            db=self.db,
            object_store=self.object_store,
            redis_url=config.redis.url,
        )

        # Ingestion (created per camera at runtime)
        self.stream_sources: List[FrameSource] = []

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            vehicle_detector=self.vehicle_detector,
            plate_detector=self.plate_detector,
            straightener=self.straightener,
            enhancer=self.enhancer,
            primary_ocr=self.primary_ocr,
            fallback_ocr=self.fallback_ocr,
            validator=self.validator,
            deduplicator=self.deduplicator,
            dispatcher=self.dispatcher,
            vram_monitor=VRAMMonitor(config.hardware),
        )
```

### 11.3 Interface Segregation

Each protocol in `contracts/` defines the **minimum surface area** required by its consumers. No protocol bundles unrelated capabilities:

| Protocol | Methods | Why Segregated |
|---|---|---|
| `FrameSource` | `read_frame()`, `release()` | Ingestion only. Knows nothing about detection. |
| `Detector` | `detect()`, `warmup()`, `release()` | Detection only. Vehicle detector and plate detector share the same interface but different implementations. |
| `Preprocessor` | `process()` | Single transformation step. Straightener and enhancer both implement this — composable in a chain. |
| `OCREngine` | `extract()` | OCR only. LLM OCR and fallback OCR are interchangeable. |
| `PostProcessor` | `validate()` | Validation only. Decoupled from dispatch. |
| `Dispatcher` | `dispatch()` | Output delivery only. Decoupled from validation logic. |

A consumer that needs only detection (e.g., a benchmarking script) imports `Detector` and knows nothing about `OCREngine` or `Dispatcher`. This is the Interface Segregation Principle applied rigorously.

---

## 12. Observability & Diagnostics Architecture

### 12.1 Metrics Registry (Prometheus)

| Metric Name | Type | Description |
|---|---|---|
| `uv_frames_ingested_total` | Counter | Total frames read from all streams |
| `uv_frames_deduplicated_total` | Counter | Frames discarded by pHash dedup |
| `uv_frames_dropped_total` | Counter | Frames dropped due to queue backpressure |
| `uv_detections_total` | Counter | Successful vehicle + plate detection events |
| `uv_ocr_requests_total` | Counter | OCR requests sent to LLM (by engine label) |
| `uv_ocr_success_total` | Counter | OCR results with status `valid` |
| `uv_ocr_fallback_total` | Counter | OCR requests routed to fallback engine |
| `uv_pipeline_latency_seconds` | Histogram | End-to-end latency per detection event |
| `uv_stage_latency_seconds` | Histogram | Per-stage latency (labels: stage name) |
| `uv_vram_usage_bytes` | Gauge | Current VRAM usage (by region label) |
| `uv_inference_queue_depth` | Gauge | Current inference queue depth |
| `uv_ocr_confidence` | Histogram | Distribution of OCR confidence scores |
| `uv_stream_status` | Gauge | Per-camera stream status (1=connected, 0=disconnected) |

### 12.2 Structured Logging (structlog)

All log entries are JSON-formatted with mandatory fields:
```json
{
  "timestamp": "2026-03-11T08:32:11.042Z",
  "level": "info",
  "event": "ocr_complete",
  "camera_id": "cam_01",
  "plate_text": "MH12AB1234",
  "confidence": 0.94,
  "engine": "qwen3.5",
  "latency_ms": 1842,
  "stage": "S7"
}
```

### 12.3 Python Programmes Visualizer

Eight visualizer modules (as specified in the original PRD) connect to the pipeline via Redis Pub/Sub. Each pipeline stage optionally publishes intermediate results to a dedicated Redis channel when `VISUALIZER_ENABLED=true`. The visualizer subscribes to these channels and renders them via Streamlit or OpenCV HighGUI.

**Production default:** `VISUALIZER_ENABLED=false`. Publishing overhead is zero when disabled (channel publish calls are gated behind the flag check at the caller).

---

## 13. Failure Taxonomy & Recovery Protocols

### 13.1 Failure Classes

| ID | Failure | Severity | Recovery |
|---|---|---|---|
| F01 | Camera stream disconnection | MEDIUM | Exponential backoff reconnect (1s→16s). Alert after 3 consecutive failures. Other streams unaffected. |
| F02 | Frame queue overflow | LOW | Oldest frame evicted (ring buffer). `frames_dropped` counter incremented. Adaptive FPS throttle engaged. |
| F03 | Vehicle detection: no vehicle found | INFO | Frame discarded. Logged to audit trail. Normal operation. |
| F04 | Plate detection: no plate found | INFO | Event discarded. Logged. Normal for frames with obscured/absent plates. |
| F05 | VRAM overflow (OOM) | CRITICAL | Immediate CPU fallback for current event. `torch.cuda.empty_cache()`. Alert raised. If persistent: degrade to Mode 3 (full CPU). |
| F06 | Ollama unresponsive (timeout) | HIGH | 5s timeout. Skip LLM OCR, route to fallback OCR engine. Circuit breaker opens after 3 consecutive timeouts, checks every 30s. |
| F07 | LLM output parse failure | MEDIUM | Append error to context, re-prompt (max 2 retries). If all retries exhausted: `PARSE_FAIL` status, routed to audit log. |
| F08 | LLM repetition loop | MEDIUM | Detect via output length anomaly (> 2× expected tokens). Abort request. Route to fallback OCR. |
| F09 | PostgreSQL write failure | HIGH | Retry with exponential backoff (3 attempts). Buffer records in-memory (max 100). Alert on persistent failure. |
| F10 | S3 image upload failure | MEDIUM | Retry 2×. On failure: store image path as `upload_pending`, background job retries later. |

### 13.2 Circuit Breaker — Ollama

The Ollama circuit breaker prevents the pipeline from stalling when the LLM runtime is degraded:

```
States: CLOSED → OPEN → HALF-OPEN

CLOSED (normal):
  - All OCR requests go to Ollama.
  - On timeout/error: increment failure counter.
  - If failures ≥ 3 within 60s: transition to OPEN.

OPEN (tripped):
  - All OCR requests go directly to fallback engine.
  - After 30s: transition to HALF-OPEN.

HALF-OPEN (probing):
  - Next single OCR request goes to Ollama.
  - If success: transition to CLOSED, reset failure counter.
  - If failure: transition to OPEN, restart 30s timer.
```

---

## Appendix A — VRAM Lifecycle Trace

Detailed VRAM trace for a single detection event on the RTX 4070 (8192 MB):

```
TIME    VRAM USED    EVENT
────    ─────────    ─────
t=0     6144 MB      Baseline: LLM weights loaded (5632 MB) + system overhead (512 MB)
t=1     6148 MB      Frame uploaded to GPU (4 MB, 640×640 float32 after resize)
t=2     6548 MB      YOLOv8 forward pass: weights loaded + activations (~400 MB transient)
t=3     6152 MB      YOLOv8 activations freed. BBox results retained (~4 KB).
t=4     6452 MB      LPD forward pass: weights loaded + activations (~300 MB transient)
t=5     6148 MB      LPD activations freed. Plate bbox retained (~1 KB). Frame still on GPU.
t=6     6144 MB      Plate crop downloaded to CPU (0.36 MB). All Region C tensors freed.
                     empty_cache() called. MEMORY FENCE passed.
t=7     6144 MB      Preprocessing on CPU (0 GPU impact). Enhanced plate encoded to base64.
t=8     7168 MB      Ollama LLM inference: KV cache grows (~1024 MB peak during generation)
t=9     6144 MB      LLM inference complete. KV cache released. Back to baseline.
```

**Peak VRAM:** 7168 MB at t=8 (LLM inference with KV cache). **Headroom:** 1024 MB from the 8192 MB ceiling. This is the tightest point in the budget — context window truncation to 4096 tokens is the critical control.

---

## Appendix B — Interface Contracts (Abstract)

Summary of all protocol interfaces that define the pipeline's internal API boundaries:

| Protocol | Module | Methods | Input → Output |
|---|---|---|---|
| `FrameSource` | `contracts/frame_source.py` | `read_frame()` | ∅ → `Optional[FramePacket]` |
| `Detector` | `contracts/detector.py` | `detect(frame)` | `ndarray` → `List[BoundingBox]` |
| `Preprocessor` | `contracts/preprocessor.py` | `process(image)` | `ndarray` → `ndarray` |
| `OCREngine` | `contracts/ocr_engine.py` | `extract(image, ctx)` | `(ndarray, DetectionContext)` → `OCRResult` |
| `PostProcessor` | `contracts/post_processor.py` | `validate(result)` | `OCRResult` → `ProcessedResult` |
| `Dispatcher` | `contracts/dispatcher.py` | `dispatch(record)` | `DetectionRecord` → `None` |

All pipeline stages depend **only** on these protocols. Concrete implementations are wired exclusively in `orchestrator/container.py`.

---

**Specification End — Uni_Vision Architecture v0.1.0-draft**
