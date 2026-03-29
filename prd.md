 Product Requirements Document (PRD)
## Automated Number Plate Recognition (ANPR) System for Toll Gate Management

---

| Field             | Details                                      |
|-------------------|----------------------------------------------|
| **Document Version** | 1.0.0                                    |
| **Status**        | Draft — Phase 1                              |
| **Created**       | 2026-03-11                                   |
| **Last Updated**  | 2026-03-11                                   |
| **Owner**         | Backend Engineering Team                     |
| **Classification**| Internal — Confidential                      |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Objectives](#3-goals--objectives)
4. [Scope](#4-scope)
5. [Stakeholders](#5-stakeholders)
6. [System Architecture Overview](#6-system-architecture-overview)
7. [Backend Component Specifications](#7-backend-component-specifications)
8. [Processing Pipeline — Detailed Workflow](#8-processing-pipeline--detailed-workflow)
9. [Functional Requirements](#9-functional-requirements)
10. [Non-Functional Requirements](#10-non-functional-requirements)
11. [Technical Stack](#11-technical-stack)
12. [Data Models & Storage Schema](#12-data-models--storage-schema)
13. [API Specifications](#13-api-specifications)
14. [Python Programmes Visualizer](#14-python-programmes-visualizer)
15. [Success Metrics & KPIs](#15-success-metrics--kpis)
16. [Risk Analysis & Mitigations](#16-risk-analysis--mitigations)
17. [Milestones & Delivery Timeline](#17-milestones--delivery-timeline)
18. [Assumptions & Dependencies](#18-assumptions--dependencies)
19. [Open Questions](#19-open-questions)
20. [Appendix](#20-appendix)

---

## 1. Executive Summary

This document defines the product requirements for an **Automated Number Plate Recognition (ANPR) system** designed specifically for toll gate infrastructure. The system ingests live or recorded video streams from static CCTV or IP cameras, detects vehicles and their license plates in real time, extracts and validates plate text using OCR, and dispatches the structured result data to a frontend dashboard and/or backend database.

Phase 1 of this project focuses exclusively on building, validating, and deploying the **backend processing pipeline** — from video ingestion through to output delivery. A Python Programmes Visualizer component is scoped for Phase 1 design but may be delivered in a sub-phase.

---

## 2. Problem Statement

Manual toll collection and traditional gate-based verification systems suffer from:

- **Operational inefficiency**: Human operators introduce latency, error, and high recurring costs.
- **Fraud vulnerability**: Manual inspection makes it easy to bypass toll payment with obscured or counterfeit plates.
- **Scalability constraints**: Human-dependent systems cannot scale to high-traffic toll plazas without proportional cost increases.
- **Data gaps**: Absence of automated, reliable vehicle passage records limits auditing, analytics, and enforcement.

There is a clear need for an **accurate, real-time, automated system** that identifies vehicles by their number plates without human intervention, integrates with payment and enforcement databases, and generates timestamped audit trails.

---

## 3. Goals & Objectives

### Primary Goals
- Build a robust, end-to-end backend pipeline capable of detecting vehicle number plates from camera feeds in real time.
- Achieve high OCR accuracy on license plates under varied lighting, speed, and weather conditions.
- Persist structured records of every vehicle detection event to a database.
- Expose a clean API interface for downstream systems (payment gateway, enforcement portal, frontend dashboard).

### Secondary Goals
- Provide a Python Programmes Visualizer module that lets operators inspect and debug each pipeline stage.
- Minimize false positives and false negatives in vehicle and plate detection.
- Design the system to be camera-vendor agnostic and horizontally scalable.

### Non-Goals (Phase 1)
- Payment processing integration (Phase 2).
- Mobile application development.
- Real-time enforcement actions (e.g., gate control signals) — to be handled in Phase 2.
- Multi-country plate format support beyond the primary locale.

---

## 4. Scope

### In-Scope (Phase 1)
| Area | Description |
|---|---|
| Video ingestion | Static CCTV, RTSP streams, IP camera feeds |
| Frame sampling | Configurable 1–5 FPS extraction |
| Vehicle detection | YOLOv8-based detection (car, truck, bus, bike) |
| Plate detection | YOLO / License Plate Detector model |
| Image preprocessing | Resize, contrast enhancement, noise removal |
| Plate straightening | Perspective correction / deskewing |
| OCR | EasyOCR, CRNN, or PaddleOCR (configurable) |
| Post-processing | Regex validation, noise character removal |
| Data storage | Plate number, timestamp, Camera ID, vehicle image |
| Output delivery | REST API push to frontend/database |
| Visualizer | Per-stage Python debugging visualizer |

### Out-of-Scope (Phase 1)
- Payment gateway integration
- Mobile or kiosk frontend UI
- Hardware procurement or camera installation
- Gate actuation/control signals
- Cloud infrastructure provisioning (handed to DevOps)

---

## 5. Stakeholders

| Role | Name / Team | Responsibility |
|---|---|---|
| Product Owner | TBD | Final sign-off on requirements and delivery |
| Tech Lead / Architect | Senior Backend Engineer | System design, code review, technical decisions |
| ML Engineer | CV/ML Team | Model selection, training, fine-tuning |
| Backend Engineers | Backend Team | Pipeline implementation, API development |
| QA Engineer | QA Team | Testing strategy, accuracy benchmarking |
| DevOps Engineer | Infra Team | Deployment, monitoring, CI/CD setup |
| Operations / Client | Toll Authority | End user, acceptance testing |

---

## 6. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          ANPR SYSTEM — PHASE 1                           │
│                                                                          │
│  ┌────────────────┐     ┌─────────────────┐     ┌──────────────────┐    │
│  │  Camera Layer  │────▶│ Ingestion Layer │────▶│  Frame Sampler   │    │
│  │ (CCTV/RTSP/IP) │     │ (Stream Capture)│     │  (1–5 FPS)       │    │
│  └────────────────┘     └─────────────────┘     └────────┬─────────┘    │
│                                                          │               │
│                                                          ▼               │
│                                               ┌─────────────────────┐   │
│                                               │  Vehicle Detection  │   │
│                                               │  (YOLOv8)           │   │
│                                               └────────┬────────────┘   │
│                                                        │                │
│                                                        ▼                │
│                                               ┌─────────────────────┐   │
│                                               │  Plate Detection    │   │
│                                               │  (YOLO / LPD)       │   │
│                                               └────────┬────────────┘   │
│                                                        │                │
│                     ┌──────────────────────────────────┘                │
│                     ▼                                                    │
│          ┌─────────────────────┐                                         │
│          │  Image Preprocessing│                                         │
│          │  • Crop             │                                         │
│          │  • Straighten       │                                         │
│          │  • Resize           │                                         │
│          │  • Contrast enhance │                                         │
│          │  • Noise removal    │                                         │
│          └────────┬────────────┘                                         │
│                   │                                                      │
│                   ▼                                                      │
│          ┌─────────────────────┐                                         │
│          │  OCR Engine         │                                         │
│          │  (EasyOCR /         │                                         │
│          │   CRNN / PaddleOCR) │                                         │
│          └────────┬────────────┘                                         │
│                   │                                                      │
│                   ▼                                                      │
│          ┌─────────────────────┐     ┌───────────────────────────────┐  │
│          │  Post-processing    │────▶│  Storage Layer                │  │
│          │  • Regex validation │     │  (DB: plate, ts, cam, image)  │  │
│          │  • Noise removal    │     └──────────────┬────────────────┘  │
│          └─────────────────────┘                    │                   │
│                                                     ▼                   │
│                                          ┌──────────────────────┐       │
│                                          │  Output Dispatcher   │       │
│                                          │  (REST API / Queue)  │       │
│                                          └──────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Backend Component Specifications

### 7.1 Source Selection and Configuration

**Purpose:** Define and manage camera input sources dynamically without requiring code changes.

**Requirements:**
- Support for the following input source types:
  - Static IP address (CCTV)
  - RTSP stream URLs
  - Local video file (for testing/replay)
- Configuration stored in a YAML/JSON config file or environment variables.
- Each camera source must have: `camera_id`, `source_url`, `location_tag`, `fps_target`, `enabled` flag.
- Support for adding/removing sources at runtime via admin API without system restart.
- Graceful reconnection logic with exponential backoff if a stream drops.

**Acceptance Criteria:**
- System can connect to at least 4 simultaneous camera streams.
- Loss of one stream does not affect others.
- Reconnection attempt occurs within 5 seconds of stream drop.

---

### 7.2 Frame Selection

**Purpose:** Extract only the frames necessary for processing, reducing compute load.

**Requirements:**
- Configurable sampling rate: 1 to 5 FPS per stream.
- Drop duplicate or near-identical consecutive frames using perceptual hashing (pHash).
- Frame queue with bounded buffer to handle temporary processing delays (max 50 frames).
- Each frame must carry metadata: `camera_id`, `timestamp_utc`, `frame_index`.

**Acceptance Criteria:**
- At 5 FPS, system processes all sampled frames without queue overflow under normal load.
- Duplicate frame detection reduces unnecessary OCR calls by ≥ 20%.

---

### 7.3 Object Detection / Vehicle Segmentation

**Purpose:** Identify and classify vehicles present in a frame before attempting plate detection.

**Requirements:**
- Use **YOLOv8** (pretrained on COCO, fine-tuned if necessary) to detect:
  - Car, Truck, Bus, Motorcycle/Bike
- Output bounding boxes with confidence scores for each detected vehicle.
- Minimum confidence threshold: configurable (default: 0.60).
- Frames with no vehicle detected are discarded from further processing but logged.
- Support for GPU acceleration (CUDA) with CPU fallback.

**Acceptance Criteria:**
- Vehicle detection precision ≥ 92% on test dataset.
- Detection latency ≤ 80ms per frame on GPU.

---

### 7.4 Plate Detection

**Purpose:** Locate the license plate region within a detected vehicle bounding box.

**Requirements:**
- Apply a dedicated **License Plate Detector** (YOLO-based or specialized LPD model) on the cropped vehicle ROI.
- Output a tight bounding box around the plate region.
- If multiple plates are detected (rare), select the one with the highest confidence.
- Minimum plate detection confidence threshold: configurable (default: 0.65).
- Handle standard plate dimensions and aspect ratios for the target locale.

**Acceptance Criteria:**
- Plate detection recall ≥ 90% across varied lighting conditions.
- False positive rate ≤ 5%.

---

### 7.5 Cropping

**Purpose:** Extract the plate image from the full frame for further processing.

**Requirements:**
- Crop plate region using bounding box coordinates with a configurable padding margin (default: 5px).
- Output a standalone image (PIL/NumPy array) of the plate.
- Store intermediate crop images temporarily for debugging via the Visualizer.

---

### 7.6 Straightening

**Purpose:** Correct skewed or tilted plate images to improve OCR accuracy.

**Requirements:**
- Detect rotation angle using Hough Line Transform or contour analysis.
- Apply affine/perspective transformation to deskew the plate image.
- Handle tilt angles up to ±30 degrees.
- Skip straightening if detected skew is within ±3 degrees (negligible).

**Acceptance Criteria:**
- Post-straightening OCR accuracy improves by ≥ 8% on skewed test images versus baseline.

---

### 7.7 Enhancement

**Purpose:** Improve plate image quality to maximize OCR legibility.

**Requirements:**
- **Resize:** Upscale plate image to minimum 200px height, maintaining aspect ratio.
- **Contrast Enhancement:** Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
- **Noise Removal:** Apply Gaussian blur followed by bilateral filtering.
- Enhancement steps must be individually toggleable via configuration.
- Preserve original image separately before enhancement for audit/debug.

**Acceptance Criteria:**
- Enhanced images show measurable PSNR improvement over raw crops.
- OCR character error rate (CER) reduced by ≥ 10% after enhancement.

---

### 7.8 OCR Models

**Purpose:** Extract the alphanumeric text from the processed plate image.

**Requirements:**
- Support three interchangeable OCR engines via configuration flag:
  - **EasyOCR** — default; GPU-accelerated, multilingual
  - **CRNN** — lightweight, fast inference
  - **PaddleOCR** — high accuracy, Paddle framework
- OCR output must include: `raw_text`, `confidence_score`, `character_boxes`.
- Minimum acceptable OCR confidence threshold: configurable (default: 0.75).
- Results below threshold must be flagged as `low_confidence` and excluded from primary storage but retained in audit logs.

**Acceptance Criteria:**
- OCR accuracy (exact plate match) ≥ 88% on clean images.
- OCR accuracy ≥ 75% on noisy/nighttime images.
- Inference time ≤ 150ms per plate on GPU.

---

### 7.9 Custom Logic / Post-Processing

**Purpose:** Validate, clean, and standardize raw OCR output before storage.

**Requirements:**
- **Regex Validation:** Apply locale-specific plate format regex (e.g., `^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$` for India). Pattern must be configurable.
- **Noise Character Removal:** Strip characters that are clearly OCR artefacts (special characters, whitespace, ambiguous glyphs).
- **Character Correction:** Apply a common substitution map for OCR confusion pairs (e.g., `0↔O`, `1↔I`, `5↔S`).
- **Deduplication:** If the same plate is detected across consecutive frames within a 10-second window, store only one record (configurable window).
- Flag records that fail validation for manual review.

**Acceptance Criteria:**
- Post-processing reduces character error rate by ≥ 15% vs raw OCR output.
- Deduplication prevents duplicate records from the same vehicle pass.

---

## 8. Processing Pipeline — Detailed Workflow

```
[START]
   │
   ▼
Static CCTV Camera / RTSP / IP Camera
   │
   ▼
Video Stream Capture
   │ (OpenCV VideoCapture / FFmpeg)
   ▼
Frame Extraction (1–5 FPS sampling)
   │ [pHash dedup applied here]
   ▼
Vehicle Detection Model — YOLOv8
   │ (detects: car, truck, bus, bike)
   │ [Discard frame if no vehicle detected → log]
   ▼
Number Plate Detection (YOLO / License Plate Detector)
   │ [Discard frame if no plate detected → log]
   ▼
Plate Image Cropping
   │ [Bounding box + padding margin]
   ▼
Image Straightening
   │ [Deskew via Hough / Perspective Transform]
   ▼
Image Enhancement
   ├── Resize (min 200px height)
   ├── Contrast Enhancement (CLAHE)
   └── Noise Removal (Gaussian + Bilateral)
   │
   ▼
OCR Model (EasyOCR / CRNN / PaddleOCR)
   │ [raw_text + confidence_score + character_boxes]
   ▼
Post-Processing
   ├── Regex Validation
   ├── Noise Character Removal
   ├── Character Correction Map (O↔0, I↔1)
   └── Deduplication (10s window)
   │
   ▼
Store Result
   ├── Plate Number (validated)
   ├── Timestamp (UTC)
   ├── Camera ID
   ├── Vehicle Image (cropped)
   └── Confidence Score
   │
   ▼
Send Output to Frontend / Database (REST API / Message Queue)
   │
[END]
```

---

## 9. Functional Requirements

### FR-01: Multi-Source Video Ingestion
The system SHALL support simultaneous ingestion from multiple camera sources (minimum 4) of types: static CCTV, RTSP stream, IP camera.

### FR-02: Configurable Frame Sampling
The system SHALL allow FPS sampling rate to be configured per camera source between 1 and 5 FPS without redeployment.

### FR-03: Vehicle Detection
The system SHALL detect vehicles of classes: car, truck, bus, and motorcycle using YOLOv8 with configurable confidence thresholds.

### FR-04: License Plate Localisation
The system SHALL detect and localize license plate regions within detected vehicle bounding boxes with precision ≥ 90%.

### FR-05: Image Preprocessing Chain
The system SHALL apply cropping, straightening, resizing, contrast enhancement, and noise removal as a configurable preprocessing chain prior to OCR.

### FR-06: OCR Engine Interchangeability
The system SHALL support EasyOCR, CRNN, and PaddleOCR as selectable OCR backends via a configuration parameter.

### FR-07: Post-Processing Validation
The system SHALL validate OCR output against a locale-specific plate format regex and apply character correction before persisting data.

### FR-08: Deduplication
The system SHALL deduplicate detection records for the same plate within a configurable time window (default: 10 seconds).

### FR-09: Data Persistence
The system SHALL store all validated detection records, including plate text, timestamp, camera ID, vehicle image path, and confidence score, in a structured database.

### FR-10: Output Dispatch
The system SHALL push detection results to a frontend dashboard and/or external database via REST API or message queue within 2 seconds of OCR completion.

### FR-11: Low-Confidence Logging
The system SHALL log all low-confidence OCR results (below threshold) to a separate audit log without storing them in the primary dataset.

### FR-12: Visualizer Integration
The system SHALL provide a Python-based visualizer module that renders the output of each pipeline stage for debugging and monitoring.

---

## 10. Non-Functional Requirements

### 10.1 Performance
| Metric | Requirement |
|---|---|
| End-to-end latency (ingestion → output dispatch) | ≤ 3 seconds per detection event |
| Vehicle detection latency | ≤ 80ms per frame (GPU) |
| OCR inference time | ≤ 150ms per plate (GPU) |
| Throughput | ≥ 4 simultaneous streams at 5 FPS each |
| Frame queue overflow tolerance | ≤ 0 frame drops under normal load |

### 10.2 Accuracy
| Metric | Target |
|---|---|
| Vehicle detection precision | ≥ 92% |
| Plate detection recall | ≥ 90% |
| OCR exact match (clean images) | ≥ 88% |
| OCR exact match (night/noise) | ≥ 75% |
| Post-processing CER reduction | ≥ 15% |

### 10.3 Reliability & Availability
- System uptime: ≥ 99.5% (excluding planned maintenance).
- Camera stream disconnection triggers automatic reconnect within 5 seconds.
- No single point of failure — component failures must be isolated.

### 10.4 Scalability
- Horizontal scaling must be supported — additional worker processes can be added without architectural changes.
- Pipeline stages should be decoupled to allow independent scaling.

### 10.5 Security
- All API endpoints must require authentication (JWT or API key).
- Vehicle images must be stored with access control (not publicly accessible).
- Sensitive plate data must be encrypted at rest.
- All inter-service communication must occur over HTTPS/TLS.

### 10.6 Maintainability
- OCR engine, detection model, and regex patterns must be swappable via configuration with no code changes.
- All pipeline stages must emit structured logs (JSON format) with severity levels.
- Unit test coverage ≥ 80% for all backend modules.

### 10.7 Observability
- System must expose a health check endpoint (`/health`).
- Metrics (frame queue depth, detection rate, OCR confidence distribution) must be exportable to Prometheus/Grafana or equivalent.

---

## 11. Technical Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Video Capture** | OpenCV (`cv2`), FFmpeg |
| **Vehicle Detection** | YOLOv8 (Ultralytics) |
| **Plate Detection** | YOLOv8 (fine-tuned) / OpenALPR / custom LPD |
| **Image Processing** | OpenCV, Pillow, NumPy, scikit-image |
| **OCR** | EasyOCR / PaddleOCR / CRNN (PyTorch) |
| **API Framework** | FastAPI |
| **Task Queue** | Celery + Redis (async processing) |
| **Database** | PostgreSQL (structured data) + S3-compatible store (images) |
| **Message Queue** | Redis Pub/Sub or RabbitMQ |
| **Configuration** | YAML + python-dotenv |
| **Logging** | Python `logging` + structlog (JSON output) |
| **Monitoring** | Prometheus + Grafana |
| **Testing** | pytest, pytest-cov |
| **Containerisation** | Docker + Docker Compose |
| **GPU Acceleration** | CUDA 11.8+ / cuDNN |

---

## 12. Data Models & Storage Schema

### 12.1 Detection Record

```sql
CREATE TABLE detection_events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id           VARCHAR(64) NOT NULL,
    plate_number        VARCHAR(20) NOT NULL,
    raw_ocr_text        VARCHAR(50),
    ocr_confidence      FLOAT NOT NULL,
    ocr_engine          VARCHAR(20) NOT NULL,
    vehicle_class       VARCHAR(20),
    vehicle_image_path  TEXT,
    plate_image_path    TEXT,
    detected_at_utc     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_validated        BOOLEAN DEFAULT FALSE,
    validation_status   VARCHAR(20),  -- 'valid', 'low_confidence', 'regex_fail'
    location_tag        VARCHAR(100),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
```

### 12.2 Camera Source Configuration

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

### 12.3 Audit Log (Low Confidence / Failed)

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

---

## 13. API Specifications

### Base URL: `https://api.anpr.internal/v1`

---

#### `GET /health`
Health check for the service.

**Response:**
```json
{
  "status": "ok",
  "uptime_seconds": 84600,
  "active_streams": 4
}
```

---

#### `GET /detections`
Retrieve paginated detection records.

**Query Params:** `camera_id`, `from`, `to`, `page`, `limit`

**Response:**
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
      "detected_at_utc": "2026-03-11T08:32:11Z",
      "location_tag": "Toll Gate 3 - Lane 2",
      "vehicle_image_url": "https://storage.internal/images/uuid.jpg"
    }
  ]
}
```

---

#### `POST /sources`
Register a new camera source.

**Request Body:**
```json
{
  "camera_id": "cam_05",
  "source_url": "rtsp://192.168.1.105:554/stream",
  "location_tag": "Toll Gate 5 - Lane 1",
  "fps_target": 3
}
```

**Response:** `201 Created`

---

#### `DELETE /sources/{camera_id}`
Disable and remove a camera source.

**Response:** `204 No Content`

---

#### `GET /metrics`
Prometheus-compatible metrics endpoint.

---

## 14. Python Programmes Visualizer

### Purpose
A developer/operator-facing tool to inspect each stage of the pipeline in real time or via replay on stored frames — enabling fast debugging, model tuning, and quality assurance.

### Proposed Modules

| Module | Visualisation |
|---|---|
| `vis_frame_sampler` | Displays sampled frames with pHash similarity score |
| `vis_vehicle_detection` | Annotated bounding boxes with class labels and confidence on frame |
| `vis_plate_detection` | Highlighted plate region within vehicle ROI |
| `vis_crop_straighten` | Side-by-side before/after crop and deskew |
| `vis_enhancement` | Side-by-side raw vs enhanced plate image |
| `vis_ocr_output` | Plate image with OCR text overlay and confidence score |
| `vis_postprocess` | Input vs output text with applied corrections highlighted |
| `vis_pipeline_stats` | Real-time dashboard: FPS, detection rate, OCR accuracy, queue depth |

### Implementation Approach
- Built using **Streamlit** or **OpenCV HighGUI** for lightweight rendering.
- Connects to the pipeline via Redis Pub/Sub for live frame events.
- Supports replay mode by loading stored intermediate frames from the debug store.
- Configuration flag: `VISUALIZER_ENABLED=true/false` (must default to `false` in production).

---

## 15. Success Metrics & KPIs

| KPI | Target | Measurement Method |
|---|---|---|
| Vehicle detection precision | ≥ 92% | Annotated test dataset |
| Plate detection recall | ≥ 90% | Annotated test dataset |
| OCR exact plate match rate (clean) | ≥ 88% | Ground truth comparison |
| OCR exact plate match rate (adverse) | ≥ 75% | Night/rain test dataset |
| End-to-end pipeline latency | ≤ 3 seconds | Timestamp diff (ingest → dispatch) |
| System uptime | ≥ 99.5% | Monitoring dashboards |
| Duplicate detection rate | < 1% | Database deduplication audit |
| False positive plate reads | < 3% | Manual QA sampling |

---

## 16. Risk Analysis & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Poor OCR accuracy in low-light / night conditions | High | High | Integrate IR camera support; apply image brightening pre-processing; consider a separate night-mode model |
| Stream disconnection causing data gaps | Medium | High | Implement exponential backoff reconnect; emit alerts on extended downtime |
| Model accuracy degradation on local plate formats | Medium | High | Fine-tune YOLOv8 and OCR models on locale-specific dataset |
| GPU unavailability in production | Low | Medium | Implement CPU fallback mode; benchmark CPU performance floor |
| Regex validation rejecting valid plates (edge cases) | Medium | Medium | Maintain a manual override queue; allow regex config update without redeploy |
| High-traffic surge overwhelming frame queue | Medium | High | Implement adaptive FPS throttling; use bounded queues with backpressure |
| Data privacy compliance for plate records | Medium | High | Encrypt at rest; enforce RBAC on APIs; define retention policy |
| OCR model licensing constraints | Low | Medium | Validate open-source licenses for EasyOCR, PaddleOCR, CRNN before production use |

---

## 17. Milestones & Delivery Timeline

| Milestone | Deliverable | Target Date |
|---|---|---|
| M1 | Architecture finalised, tech stack approved | Week 1 |
| M2 | Video ingestion + frame sampler complete | Week 2 |
| M3 | Vehicle detection + plate detection integrated | Week 3–4 |
| M4 | Image preprocessing chain (crop, deskew, enhance) | Week 5 |
| M5 | OCR integration (all three engines testable) | Week 6 |
| M6 | Post-processing, validation, deduplication | Week 7 |
| M7 | Database schema, storage, API endpoints | Week 8 |
| M8 | Python Programmes Visualizer MVP | Week 9 |
| M9 | End-to-end integration testing + benchmarking | Week 10 |
| M10 | Staging deployment + UAT with client | Week 11 |
| M11 | Phase 1 production release | Week 12 |

---

## 18. Assumptions & Dependencies

### Assumptions
- Camera hardware is already installed and accessible via network (RTSP/IP).
- A GPU-equipped server is available for model inference in production.
- License plates in scope follow the primary locale format consistently.
- Client/Operations team will provide a labelled dataset for model fine-tuning if needed.
- Network bandwidth between cameras and the processing server is sufficient (≥ 10 Mbps per stream).

### Dependencies
- YOLOv8 pretrained weights are available via Ultralytics (open-source).
- EasyOCR, PaddleOCR, CRNN are open-source and licensed for commercial use.
- PostgreSQL and Redis infrastructure are provisioned by DevOps before M7.
- S3-compatible object storage is provisioned for image persistence.

---

## 19. Open Questions

| # | Question | Owner | Due |
|---|---|---|---|
| 1 | What is the exact plate format regex for the target region(s)? | Client / Operations | Week 1 |
| 2 | Should images be retained indefinitely or subject to a retention policy (e.g., 90 days)? | Legal / Product | Week 2 |
| 3 | Is a labelled dataset available for model fine-tuning, or must one be collected? | ML Team | Week 1 |
| 4 | What is the maximum expected number of simultaneous camera streams at full scale? | Client | Week 1 |
| 5 | Are there data residency or compliance requirements (GDPR, local law)? | Legal | Week 2 |
| 6 | Will the Visualizer be accessible to the client in production or only internal? | Product Owner | Week 3 |
| 7 | Should low-confidence detections trigger alerts or notifications? | Product Owner | Week 2 |

---

## 20. Appendix

### A. Glossary

| Term | Definition |
|---|---|
| ANPR | Automatic Number Plate Recognition |
| RTSP | Real-Time Streaming Protocol — used for IP camera feeds |
| YOLOv8 | You Only Look Once v8 — a real-time object detection model by Ultralytics |
| LPD | License Plate Detector — specialized model for plate localisation |
| EasyOCR | Open-source OCR library supporting 80+ languages |
| CRNN | Convolutional Recurrent Neural Network — a sequence-to-sequence OCR approach |
| PaddleOCR | High-accuracy OCR toolkit by Baidu PaddlePaddle |
| CLAHE | Contrast Limited Adaptive Histogram Equalization |
| pHash | Perceptual Hash — used for near-duplicate frame detection |
| CER | Character Error Rate — standard metric for OCR accuracy |
| ROI | Region of Interest |
| FPS | Frames Per Second |

### B. Reference Models

| Model | Source | License |
|---|---|---|
| YOLOv8 | https://github.com/ultralytics/ultralytics | AGPL-3.0 |
| EasyOCR | https://github.com/JaidedAI/EasyOCR | Apache 2.0 |
| PaddleOCR | https://github.com/PaddlePaddle/PaddleOCR | Apache 2.0 |

---

*This document is a living artifact. All sections are subject to revision as the project progresses. Major revisions require sign-off from the Product Owner and Tech Lead.*

---
**Document End — ANPR System PRD v1.0.0**
