# Technical Abstract
## Automated Number Plate Recognition (ANPR) System for Toll Gate Infrastructure
### Phase 1 — Backend Processing Pipeline

---

| Metadata Field       | Value                                                        |
|----------------------|--------------------------------------------------------------|
| **Document Type**    | Technical Abstract — Knowledgebase Entry                     |
| **System Name**      | Automated Number Plate Recognition (ANPR) System            |
| **Project Phase**    | Phase 1 — Backend Pipeline                                   |
| **Domain**           | Computer Vision / Intelligent Transportation Systems (ITS)  |
| **Version**          | 1.0.0                                                        |
| **Date**             | 2026-03-11                                                   |
| **Classification**   | Internal Technical Reference                                 |
| **Related Document** | PRD — ANPR System v1.0.0                                     |

---

## Abstract

This document presents a technical overview of the **Automated Number Plate Recognition (ANPR) system**, an AI-driven, real-time vehicle identification platform engineered for deployment at toll gate infrastructure. The system is architected as a multi-stage backend pipeline that ingests continuous video streams from static CCTV and IP-based camera networks, applies a sequence of computer vision and deep learning operations to detect and extract license plate information, and delivers validated, structured records to downstream storage and frontend systems.

The core pipeline is composed of eight distinct processing stages: **(1) video stream ingestion and frame sampling**, **(2) vehicle detection via YOLOv8**, **(3) license plate localisation using a dedicated plate detection model**, **(4) plate image cropping and region isolation**, **(5) geometric straightening and perspective correction**, **(6) multi-technique image enhancement**, **(7) optical character recognition (OCR) using configurable engines**, and **(8) post-processing, validation, and structured data dispatch**. Each stage is independently configurable, observable, and designed for horizontal scalability.

The system targets an end-to-end processing latency of under three seconds per detection event, an OCR exact-match accuracy of ≥ 88% on clean images and ≥ 75% under adverse conditions (low light, rain, motion blur), and a vehicle detection precision of ≥ 92%. It is implemented in Python 3.10+, leverages GPU-accelerated inference via CUDA, and exposes a RESTful API for integration with frontend dashboards, payment gateways, and enforcement systems.

---

## 1. Problem Domain and Motivation

Traditional toll collection infrastructure relies heavily on human operators for vehicle identification, fare collection, and audit trail generation. This approach introduces measurable inefficiencies: operator-induced latency, susceptibility to fraud through obscured or cloned number plates, inability to scale proportionally with traffic volume, and fragmented or absent digital records for compliance and analytics purposes.

Intelligent Transportation Systems (ITS) research consistently demonstrates that automated vehicle identification using computer vision can achieve accuracy and throughput levels that human operators cannot match, while simultaneously generating rich, queryable datasets that enable downstream applications including predictive traffic management, enforcement automation, and toll revenue reconciliation.

The ANPR system described herein directly addresses these deficiencies by replacing human observation with a deterministic, model-driven pipeline that operates continuously, scales horizontally with camera count, and generates structured, tamper-evident records for every vehicle passage event.

---

## 2. System Architecture

The system follows a **linear staged pipeline architecture** with asynchronous inter-stage communication via a Redis-backed task queue (Celery). This design ensures that no single stage can block the entire pipeline; each component processes work independently from a bounded queue and emits results downstream upon completion.

```
Camera Network (CCTV / RTSP / IP)
         │
         ▼
  ┌──────────────────────┐
  │  Stream Ingestion    │  ← OpenCV VideoCapture / FFmpeg
  │  & Frame Sampling    │  ← 1–5 FPS, pHash deduplication
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Vehicle Detection   │  ← YOLOv8 (car, truck, bus, bike)
  │  (YOLOv8)            │  ← Confidence threshold: 0.60
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Plate Detection     │  ← YOLO-based License Plate Detector
  │  (LPD Model)         │  ← Confidence threshold: 0.65
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Preprocessing Chain │  ← Crop → Straighten → Resize
  │  (CV Pipeline)       │  ← CLAHE → Gaussian → Bilateral Filter
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  OCR Engine          │  ← EasyOCR / CRNN / PaddleOCR
  │  (Configurable)      │  ← raw_text + confidence + char_boxes
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Post-Processing     │  ← Regex validation, char correction
  │  & Validation        │  ← Deduplication (10s window)
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Storage & Dispatch  │  ← PostgreSQL + S3-compatible store
  │  (REST API / MQ)     │  ← Frontend / DB output ≤ 2s
  └──────────────────────┘
```

The architecture supports a minimum of four simultaneous camera streams at 5 FPS each without frame queue overflow. GPU inference is used for both YOLO-based detection and OCR stages, with a CPU fallback mode available for development and degraded-mode operation.

---

## 3. Pipeline Stage Technical Detail

### 3.1 Video Stream Ingestion and Frame Sampling

The ingestion layer accepts three source types: static CCTV via direct IP address, RTSP protocol streams from network cameras, and local video files for testing and replay scenarios. Each source is registered with a `camera_id`, `source_url`, `location_tag`, and target FPS, stored in a YAML configuration file. The system maintains persistent connections and applies exponential backoff reconnection logic upon stream dropout, targeting reconnection within five seconds.

Frame sampling applies **perceptual hashing (pHash)** to detect and discard near-duplicate consecutive frames before they enter the detection queue. This reduces redundant inference calls and is estimated to eliminate ≥ 20% of unnecessary processing under low-traffic conditions. Each accepted frame is annotated with `camera_id`, `timestamp_utc`, and `frame_index` for full traceability.

### 3.2 Vehicle Detection — YOLOv8

Vehicle classification is performed using **YOLOv8** (Ultralytics), a real-time, anchor-free single-stage object detector built on a CSPNet backbone with a decoupled detection head. The model is configured to detect four vehicle classes: `car`, `truck`, `bus`, and `motorcycle`. Pre-trained weights from the COCO dataset are used as the base, with optional fine-tuning on locale-specific traffic datasets for improved performance in the deployment environment.

Detection outputs a set of bounding boxes with associated class labels and confidence scores. Frames in which no vehicle is detected above the minimum confidence threshold (default: 0.60) are discarded and logged to the audit trail. The model is expected to deliver detection latency of ≤ 80ms per frame on CUDA-capable hardware, with a target precision of ≥ 92% on the validation dataset.

### 3.3 License Plate Detection

Within each detected vehicle bounding box, a secondary **License Plate Detector (LPD)** model — itself a YOLO-variant fine-tuned on license plate datasets — is applied to localise the plate region. The model outputs a tight bounding box with a confidence score; a minimum threshold of 0.65 is applied. In rare cases where multiple plates are detected, the highest-confidence detection is selected.

This two-stage detection strategy (vehicle-first, then plate) substantially reduces the search space for the LPD model compared to applying it naively across the full frame, improving both accuracy and inference speed. The LPD targets a recall of ≥ 90% and a false positive rate of ≤ 5%.

### 3.4 Image Preprocessing Chain

The preprocessing chain transforms the raw cropped plate image into a clean, high-contrast, axis-aligned image optimised for OCR. The stages are applied sequentially and are individually togglable via configuration:

**Cropping** isolates the plate region using the LPD bounding box with a configurable padding margin (default: 5 pixels) to avoid clipping edge characters.

**Straightening** applies geometric correction to compensate for camera angle and vehicle tilt. The rotation angle is estimated using the **Hough Line Transform** on the binary-thresholded plate image, and an affine transformation is applied to deskew. Skew angles within ±3 degrees are considered negligible and skipped. The module handles tilts of up to ±30 degrees.

**Resizing** upscales the plate image to a minimum height of 200 pixels while preserving the aspect ratio. This ensures OCR models receive input at a sufficient resolution for character discrimination.

**Contrast Enhancement** uses **CLAHE (Contrast Limited Adaptive Histogram Equalization)** applied on the luminance channel of the LAB colour space, improving local contrast while suppressing noise amplification — a known limitation of global histogram equalization.

**Noise Removal** applies a **Gaussian blur** followed by a **bilateral filter**. The bilateral filter is chosen because it preserves edge sharpness (critical for character segmentation) while smoothing intra-region noise.

### 3.5 OCR Engine

The OCR layer is designed with a **strategy pattern**, allowing the active engine to be swapped via a single configuration parameter without code changes. Three engines are supported:

**EasyOCR** is the default engine. It is based on a CRAFT text detector combined with a sequence-to-sequence recognition model (LSTM + CTC decoder), supports GPU acceleration, and handles 80+ languages with no additional configuration.

**CRNN (Convolutional Recurrent Neural Network)** offers a lightweight, faster alternative suited for resource-constrained environments. The architecture combines CNN feature extraction with a bidirectional LSTM sequence model and a CTC loss decoder, making it well-suited for fixed-format strings such as license plates.

**PaddleOCR** provides high accuracy particularly on complex or stylised fonts, leveraging Baidu's PP-OCR model series. It is the recommended engine when operating on plates with non-standard fonts or significant weathering.

All engines return: `raw_text` (unprocessed string), `confidence_score` (float 0–1), and `character_boxes` (bounding boxes per character). Results below the configurable confidence threshold (default: 0.75) are routed to the audit log rather than the primary storage, flagged as `low_confidence`.

### 3.6 Post-Processing and Validation

Raw OCR output is processed through a validation and correction chain before storage:

**Character Correction** applies a predefined substitution map to resolve common OCR confusion pairs arising from visual similarity between characters and digits. The map includes substitutions such as `O ↔ 0`, `I ↔ 1`, `S ↔ 5`, and `B ↔ 8`. The map is configurable and locale-aware.

**Regex Validation** tests the corrected string against a locale-specific plate format pattern (e.g., `^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$` for Indian plates). Strings that fail pattern matching are flagged as `regex_fail` and routed to manual review queue while still being persisted in the audit log.

**Deduplication** suppresses repeated detections of the same plate within a configurable sliding time window (default: 10 seconds), preventing multiple records from the same vehicle pass due to successive frame sampling. Only the highest-confidence detection within the window is retained in the primary store.

### 3.7 Storage and Output Dispatch

Validated detection records are persisted to **PostgreSQL** with the following fields: `plate_number`, `raw_ocr_text`, `ocr_confidence`, `ocr_engine`, `vehicle_class`, `camera_id`, `location_tag`, `detected_at_utc`, and `validation_status`. Vehicle and plate images are stored separately in an **S3-compatible object store**, with the database holding only the reference path.

Results are dispatched to the downstream frontend and database systems via a **REST API** (FastAPI) or a **message queue** (Redis Pub/Sub / RabbitMQ), targeting a dispatch latency of ≤ 2 seconds from OCR completion.

---

## 4. Python Programmes Visualizer

The Python Programmes Visualizer is a developer- and operator-facing diagnostic tool that renders the intermediate output of each pipeline stage in real time or in replay mode over stored frames. It is implemented using **Streamlit** (for rich UI) or **OpenCV HighGUI** (for lightweight terminal-adjacent use), and connects to the live pipeline via Redis Pub/Sub subscriptions.

Eight visualizer modules are defined, one per pipeline stage, from raw frame display with pHash scores through to a real-time statistics dashboard showing throughput, detection rates, OCR confidence distributions, and queue depth metrics. The visualizer is controlled by the `VISUALIZER_ENABLED` environment flag and defaults to `false` in production deployments to eliminate performance overhead.

---

## 5. Performance and Accuracy Targets

| Metric | Target |
|---|---|
| End-to-end pipeline latency | ≤ 3 seconds |
| Vehicle detection precision | ≥ 92% |
| Plate detection recall | ≥ 90% |
| Plate detection false positive rate | ≤ 5% |
| OCR exact match — clean images | ≥ 88% |
| OCR exact match — adverse conditions | ≥ 75% |
| Post-processing CER reduction vs raw OCR | ≥ 15% |
| System uptime | ≥ 99.5% |
| Concurrent stream support | ≥ 4 streams at 5 FPS |
| Output dispatch latency | ≤ 2 seconds |

---

## 6. Technology Stack Summary

| Component | Technology |
|---|---|
| Runtime Language | Python 3.10+ |
| Video Capture | OpenCV (`cv2`), FFmpeg |
| Vehicle & Plate Detection | YOLOv8 — Ultralytics (AGPL-3.0) |
| Image Processing | OpenCV, Pillow, NumPy, scikit-image |
| OCR Engines | EasyOCR (Apache 2.0), PaddleOCR (Apache 2.0), CRNN (PyTorch) |
| API Framework | FastAPI |
| Async Task Queue | Celery + Redis |
| Primary Database | PostgreSQL |
| Image Storage | S3-compatible object store |
| Message Queue | Redis Pub/Sub / RabbitMQ |
| Visualizer | Streamlit / OpenCV HighGUI |
| GPU Acceleration | CUDA 11.8+ / cuDNN |
| Monitoring | Prometheus + Grafana |
| Containerisation | Docker + Docker Compose |

---

## 7. Key Design Decisions and Rationale

**Two-stage detection (vehicle → plate)** was chosen over single-pass plate detection applied to the full frame because it constrains the plate search to the vehicle ROI, reducing background noise, improving localisation accuracy, and lowering per-frame inference cost.

**Configurable OCR engine** rather than a single hardcoded model was adopted to allow operators to tune the accuracy–speed trade-off based on deployment context (e.g., high-throughput highway vs lower-volume gated access) without requiring code changes or redeployment.

**CLAHE over global histogram equalization** for contrast enhancement was selected because global methods amplify noise uniformly across the image, which is detrimental to OCR on plates with uneven illumination. CLAHE constrains amplification within local tile regions, preserving character edge definition.

**Bilateral filter over Gaussian-only smoothing** was used for noise removal because Gaussian blur, while effective at noise suppression, degrades character edge sharpness — a critical feature for OCR. The bilateral filter's edge-preserving property mitigates this trade-off.

**Deduplication via sliding time window** rather than plate-blacklist approaches was chosen because blacklists require periodic flushing and are prone to memory growth under sustained traffic. A sliding window approach is O(1) in memory relative to the window size and naturally handles re-entry scenarios.

---

## 8. Identified Risks

The principal technical risks are: **(a)** OCR accuracy degradation under low-light or adverse weather conditions, mitigated by IR camera integration and a dedicated night-mode preprocessing profile; **(b)** model accuracy drift on locale-specific plate formats if insufficient fine-tuning data is available; **(c)** frame queue saturation under burst traffic, addressed by adaptive FPS throttling and bounded queue backpressure; and **(d)** data privacy exposure of plate records, mitigated by encryption at rest, RBAC enforcement, and a configurable data retention policy.

---

## 9. Phase Boundary and Forward Integration

Phase 1 delivers the complete backend pipeline from video ingestion to structured output dispatch. It explicitly excludes payment gateway integration, gate actuation control signals, and mobile application interfaces, all of which are designated for Phase 2. The REST API surface designed in Phase 1 is intentionally structured to support these Phase 2 integrations without requiring breaking changes to the data model or endpoint schema.

The Phase 1 deliverable is a containerised, GPU-capable, horizontally scalable backend service with a defined API contract, observable via a Prometheus/Grafana monitoring stack, and instrumented with structured JSON logging across all pipeline stages.

---

## 10. Glossary of Technical Terms

| Term | Definition |
|---|---|
| ANPR | Automated Number Plate Recognition — the overarching system domain |
| YOLOv8 | You Only Look Once v8 — a real-time, anchor-free CNN-based object detector |
| LPD | License Plate Detector — a YOLO-variant model fine-tuned for plate region localisation |
| CRNN | Convolutional Recurrent Neural Network — sequence recognition model combining CNN + LSTM + CTC |
| EasyOCR | Open-source OCR library using CRAFT detection + sequence recognition |
| PaddleOCR | High-accuracy OCR toolkit by Baidu, based on the PP-OCR model series |
| CLAHE | Contrast Limited Adaptive Histogram Equalization — local contrast enhancement technique |
| pHash | Perceptual Hash — compact image fingerprint used for near-duplicate frame detection |
| CER | Character Error Rate — ratio of incorrectly recognised characters to total characters |
| RTSP | Real-Time Streaming Protocol — network protocol for live camera stream delivery |
| ROI | Region of Interest — a sub-region of an image targeted for analysis |
| CTC | Connectionist Temporal Classification — loss/decoding function for sequence-to-sequence models |
| RBAC | Role-Based Access Control — permission model restricting resource access by user role |
| ITS | Intelligent Transportation Systems — the broader engineering domain of this application |
| Bilateral Filter | Edge-preserving smoothing filter that averages pixel values weighted by spatial and intensity distance |
| Hough Transform | Algorithm for detecting geometric shapes (lines, circles) in images via parameter space voting |

---

*This abstract is intended as a self-contained technical reference entry. For full requirements, acceptance criteria, API specifications, and delivery milestones, refer to the linked PRD document: **ANPR System PRD v1.0.0**.*

---
**Abstract End — ANPR System Technical Abstract v1.0.0**
