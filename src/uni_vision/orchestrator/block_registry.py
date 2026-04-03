"""Backend block registry — serves available pipeline block definitions.

This is the **single source of truth** for what blocks are available.
The frontend fetches from ``GET /api/pipeline/blocks`` instead of using
a hardcoded list.  New blocks can be added here or discovered at runtime
from the ComponentRegistry.

Each block definition mirrors the frontend ``BlockDefinition`` interface:
  type, label, description, category, inputs, outputs, defaults, configSchema
Plus a ``backend_handler`` key that maps to the execution function used
by the graph engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Port / Config helpers ─────────────────────────────────────────

def _port(id: str, name: str, type: str, direction: str) -> Dict[str, str]:
    return {"id": id, "name": name, "type": type, "direction": direction}


def _cfg(key: str, label: str, type: str, **kwargs: Any) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"key": key, "label": label, "type": type}
    schema.update(kwargs)
    return schema


def _instruction_cfg(placeholder: str = "Describe what this step should do...") -> Dict[str, Any]:
    """Standard instruction config field — always first in every block's configSchema."""
    return _cfg(
        "instruction",
        "Your Instruction",
        "textarea",
        placeholder=placeholder,
    )


# ── Built-in block definitions ────────────────────────────────────

_BUILTIN_BLOCKS: List[Dict[str, Any]] = [
    # ── Input ──
    {
        "type": "image-input",
        "label": "Image Input",
        "description": "Loads a single image into the pipeline.",
        "category": "Input",
        "inputs": [],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Load test images from my surveillance camera folder", "path": "/images/sample.jpg"},
        "configSchema": [
            _instruction_cfg("Describe what images to load and from where..."),
            _cfg("path", "Image Path", "text", required=True),
        ],
        "backend_handler": "input.image",
    },
    {
        "type": "rtsp-stream",
        "label": "RTSP Stream",
        "description": "Streams frames from a remote camera source.",
        "category": "Input",
        "inputs": [],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Connect to the parking lot entrance camera and stream live video", "streamUrl": "rtsp://camera.local/stream", "fps": 15},
        "configSchema": [
            _instruction_cfg("Describe which camera to connect to and any streaming preferences..."),
            _cfg("streamUrl", "Stream URL", "text", required=True),
            _cfg("fps", "Target FPS", "number", required=False, min=1, max=30),
        ],
        "backend_handler": "input.rtsp",
    },
    {
        "type": "video-file",
        "label": "Video File",
        "description": "Load and decode a local video file frame-by-frame.",
        "category": "Input",
        "inputs": [],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Load the uploaded video for analysis", "path": "/videos/clip.mp4", "fps": 10},
        "configSchema": [
            _instruction_cfg("Describe the video source..."),
            _cfg("path", "Video Path", "text", required=True),
            _cfg("fps", "Decode FPS", "number", required=False, min=1, max=30),
        ],
        "backend_handler": "input.video_file",
    },
    # ── Ingestion ──
    {
        "type": "frame-sampler",
        "label": "Frame Sampler",
        "description": "Samples frames at a target rate to reduce processing load.",
        "category": "Ingestion",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Sample every 3rd frame to keep processing manageable", "sampleRate": 5},
        "configSchema": [
            _instruction_cfg("Describe how you want frames sampled..."),
            _cfg("sampleRate", "Sample Rate (fps)", "number", required=True, min=1, max=30),
        ],
        "backend_handler": "ingestion.frame_sampler",
    },
    {
        "type": "roi-crop",
        "label": "ROI Crop",
        "description": "Crop frames to a region-of-interest for focused analysis.",
        "category": "Ingestion",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Crop to the area of interest", "x": 0, "y": 0, "width": 640, "height": 480},
        "configSchema": [
            _instruction_cfg("Describe which region to focus on..."),
            _cfg("x", "X Offset", "number", min=0),
            _cfg("y", "Y Offset", "number", min=0),
            _cfg("width", "Width", "number", min=1),
            _cfg("height", "Height", "number", min=1),
        ],
        "backend_handler": "ingestion.roi_crop",
    },
    # ── Preprocessing ──
    {
        "type": "grayscale",
        "label": "Grayscale",
        "description": "Converts frames into grayscale.",
        "category": "Preprocessing",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Convert frames to grayscale for faster downstream processing"},
        "configSchema": [
            _instruction_cfg("Describe any preprocessing preferences..."),
        ],
        "backend_handler": "preprocessing.grayscale",
    },
    {
        "type": "contrast-enhance",
        "label": "Contrast Enhance",
        "description": "Apply adaptive histogram equalization to improve contrast.",
        "category": "Preprocessing",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Enhance frame contrast for better visibility", "clipLimit": 2.0, "tileSize": 8},
        "configSchema": [
            _instruction_cfg("Describe contrast enhancement preferences..."),
            _cfg("clipLimit", "CLAHE Clip Limit", "number", min=0.1, max=10),
            _cfg("tileSize", "Tile Size", "number", min=2, max=32),
        ],
        "backend_handler": "preprocessing.contrast_enhance",
    },
    {
        "type": "denoise",
        "label": "Denoise",
        "description": "Reduce image noise with non-local means denoising.",
        "category": "Preprocessing",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Remove noise while preserving edges", "strength": 10},
        "configSchema": [
            _instruction_cfg("Describe noise reduction preferences..."),
            _cfg("strength", "Denoising Strength", "number", min=1, max=30),
        ],
        "backend_handler": "preprocessing.denoise",
    },
    {
        "type": "resize",
        "label": "Resize",
        "description": "Resize frames to a target resolution.",
        "category": "Preprocessing",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Resize frames before processing", "width": 640, "height": 480, "keepAspect": True},
        "configSchema": [
            _instruction_cfg("Describe target size..."),
            _cfg("width", "Width", "number", min=1, max=7680),
            _cfg("height", "Height", "number", min=1, max=4320),
            _cfg("keepAspect", "Keep Aspect Ratio", "toggle"),
        ],
        "backend_handler": "preprocessing.resize",
    },
    {
        "type": "plate-preprocessor",
        "label": "Plate Preprocessor",
        "description": "Deskews, enhances contrast, and resizes plate crops for OCR.",
        "category": "Preprocessing",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Clean up and deskew the plate crops, enhance contrast for better OCR", "deskew": True, "clahe": True, "targetWidth": 200},
        "configSchema": [
            _instruction_cfg("Describe how plate images should be cleaned up..."),
            _cfg("deskew", "Deskew", "toggle"),
            _cfg("clahe", "CLAHE Enhancement", "toggle"),
            _cfg("targetWidth", "Target Width (px)", "number", min=50, max=500),
        ],
        "backend_handler": "preprocessing.plate",
    },
    # ── Detection ──
    {
        "type": "yolo-detector",
        "label": "YOLO Detector",
        "description": "Detects objects and emits bounding boxes.",
        "category": "Detection",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [
            _port("frame-out", "frame", "frame", "output"),
            _port("boxes-out", "boxes", "bounding_box_list", "output"),
        ],
        "defaults": {"instruction": "Detect all objects in each frame with high accuracy", "model": "yolov8n.pt", "confidence": 0.6},
        "configSchema": [
            _instruction_cfg("Describe what objects to detect and any priority areas..."),
            _cfg("model", "Model", "text", required=True),
            _cfg("confidence", "Confidence", "number", required=True, min=0, max=1),
        ],
        "backend_handler": "detection.yolo",
    },
    {
        "type": "motion-detector",
        "label": "Motion Detector",
        "description": "Background-subtraction motion detection with configurable sensitivity.",
        "category": "Detection",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [
            _port("frame-out", "frame", "frame", "output"),
            _port("boxes-out", "boxes", "bounding_box_list", "output"),
        ],
        "defaults": {"instruction": "Detect motion in the scene", "algorithm": "MOG2", "threshold": 25},
        "configSchema": [
            _instruction_cfg("Describe motion detection requirements..."),
            _cfg("algorithm", "Algorithm", "select", options=[
                {"label": "MOG2", "value": "MOG2"},
                {"label": "KNN", "value": "KNN"},
            ]),
            _cfg("threshold", "Threshold", "number", min=1, max=255),
        ],
        "backend_handler": "detection.motion",
    },
    {
        "type": "fire-smoke-detector",
        "label": "Fire & Smoke Detector",
        "description": "Detects fire and smoke using a specialized model.",
        "category": "Detection",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [
            _port("frame-out", "frame", "frame", "output"),
            _port("boxes-out", "boxes", "bounding_box_list", "output"),
        ],
        "defaults": {"instruction": "Detect fire and smoke in the scene", "model": "fire_smoke.pt", "confidence": 0.5},
        "configSchema": [
            _instruction_cfg("Describe fire/smoke detection requirements..."),
            _cfg("model", "Model Path", "text", required=True),
            _cfg("confidence", "Confidence", "number", required=True, min=0, max=1),
        ],
        "backend_handler": "detection.fire_smoke",
    },
    {
        "type": "crowd-density",
        "label": "Crowd Density",
        "description": "Estimates crowd density and headcount in a scene.",
        "category": "Detection",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [
            _port("frame-out", "frame", "frame", "output"),
            _port("boxes-out", "boxes", "bounding_box_list", "output"),
        ],
        "defaults": {"instruction": "Estimate crowd density", "model": "crowd_counting.pt"},
        "configSchema": [
            _instruction_cfg("Describe crowd monitoring requirements..."),
            _cfg("model", "Model Path", "text", required=True),
        ],
        "backend_handler": "detection.crowd_density",
    },
    {
        "type": "vehicle-detector",
        "label": "Vehicle Detector",
        "description": "Detects vehicles and classifies type (car, truck, bus, bike).",
        "category": "Detection",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [
            _port("frame-out", "frame", "frame", "output"),
            _port("boxes-out", "boxes", "bounding_box_list", "output"),
        ],
        "defaults": {"instruction": "Detect all vehicles — focus on cars, trucks, buses, and motorcycles", "model": "yolov8n.pt", "confidence": 0.5, "classes": "car,truck,bus,motorcycle"},
        "configSchema": [
            _instruction_cfg("Describe which vehicle types to detect and any special conditions..."),
            _cfg("model", "Model Path", "text", required=True),
            _cfg("confidence", "Confidence", "number", required=True, min=0, max=1),
            _cfg("classes", "Vehicle Classes", "text", required=False),
        ],
        "backend_handler": "detection.vehicle",
    },
    {
        "type": "plate-detector",
        "label": "Plate Detector",
        "description": "Detects license plates within vehicle bounding boxes.",
        "category": "Detection",
        "inputs": [
            _port("frame-in", "frame", "frame", "input"),
            _port("boxes-in", "boxes", "bounding_box_list", "input"),
        ],
        "outputs": [
            _port("plates-out", "plates", "frame", "output"),
            _port("boxes-out", "plate_boxes", "bounding_box_list", "output"),
        ],
        "defaults": {"instruction": "Find number plates on all detected vehicles", "model": "plate_detect.pt", "confidence": 0.6},
        "configSchema": [
            _instruction_cfg("Describe plate detection requirements and any region specifics..."),
            _cfg("model", "Plate Model", "text", required=True),
            _cfg("confidence", "Confidence", "number", required=True, min=0, max=1),
        ],
        "backend_handler": "detection.plate",
    },
    {
        "type": "scene-classifier",
        "label": "Scene Classifier",
        "description": "Classifies the overall scene (indoor, outdoor, traffic, etc.).",
        "category": "Detection",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [
            _port("frame-out", "frame", "frame", "output"),
            _port("score-out", "score", "number", "output"),
        ],
        "defaults": {"instruction": "Classify scene type", "model": "scene_classifier.pt"},
        "configSchema": [
            _instruction_cfg("Describe scene classification needs..."),
            _cfg("model", "Model Path", "text", required=True),
        ],
        "backend_handler": "detection.scene_classifier",
    },
    {
        "type": "anomaly-scorer",
        "label": "Anomaly Scorer",
        "description": "Computes an anomaly score for each frame.",
        "category": "Detection",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [
            _port("frame-out", "frame", "frame", "output"),
            _port("score-out", "score", "number", "output"),
        ],
        "defaults": {"instruction": "Score anomaly level", "model": "autoencoder.pt", "threshold": 0.7},
        "configSchema": [
            _instruction_cfg("Describe anomaly detection criteria..."),
            _cfg("model", "Model Path", "text", required=True),
            _cfg("threshold", "Anomaly Threshold", "number", min=0, max=1),
        ],
        "backend_handler": "detection.anomaly_scorer",
    },
    {
        "type": "pose-estimator",
        "label": "Pose Estimator",
        "description": "Estimates human body pose keypoints.",
        "category": "Detection",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [
            _port("frame-out", "frame", "frame", "output"),
            _port("boxes-out", "boxes", "bounding_box_list", "output"),
        ],
        "defaults": {"instruction": "Estimate human poses", "model": "yolov8n-pose.pt", "confidence": 0.5},
        "configSchema": [
            _instruction_cfg("Describe pose estimation requirements..."),
            _cfg("model", "Model Path", "text", required=True),
            _cfg("confidence", "Confidence", "number", min=0, max=1),
        ],
        "backend_handler": "detection.pose_estimator",
    },
    {
        "type": "ppe-detector",
        "label": "PPE Detector",
        "description": "Detects personal protective equipment (helmets, vests, goggles).",
        "category": "Detection",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [
            _port("frame-out", "frame", "frame", "output"),
            _port("boxes-out", "boxes", "bounding_box_list", "output"),
        ],
        "defaults": {"instruction": "Detect PPE compliance", "model": "ppe_detect.pt", "confidence": 0.5},
        "configSchema": [
            _instruction_cfg("Describe PPE detection requirements..."),
            _cfg("model", "Model Path", "text", required=True),
            _cfg("confidence", "Confidence", "number", min=0, max=1),
        ],
        "backend_handler": "detection.ppe",
    },
    {
        "type": "zone-intrusion",
        "label": "Zone Intrusion",
        "description": "Triggers alerts when objects enter a defined exclusion zone.",
        "category": "Detection",
        "inputs": [
            _port("frame-in", "frame", "frame", "input"),
            _port("boxes-in", "boxes", "bounding_box_list", "input"),
        ],
        "outputs": [
            _port("frame-out", "frame", "frame", "output"),
            _port("boxes-out", "boxes", "bounding_box_list", "output"),
        ],
        "defaults": {"instruction": "Monitor exclusion zone", "zonePolygon": "[]"},
        "configSchema": [
            _instruction_cfg("Describe the zone and intrusion rules..."),
            _cfg("zonePolygon", "Zone Polygon (JSON)", "textarea"),
        ],
        "backend_handler": "detection.zone_intrusion",
    },
    {
        "type": "llm-vision",
        "label": "LLM Vision",
        "description": "Sends frames to a vision-language model for open-ended analysis.",
        "category": "Detection",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [
            _port("frame-out", "frame", "frame", "output"),
            _port("text-out", "text", "text", "output"),
        ],
        "defaults": {"instruction": "Describe what you see in this frame", "model": "gemma4:e2b", "prompt": "Describe the scene."},
        "configSchema": [
            _instruction_cfg("Describe what the LLM should analyze..."),
            _cfg("model", "Model", "text", required=True),
            _cfg("prompt", "Prompt", "textarea"),
        ],
        "backend_handler": "detection.llm_vision",
    },
    # ── OCR ──
    {
        "type": "easy-ocr",
        "label": "EasyOCR",
        "description": "Reads text from image regions using EasyOCR.",
        "category": "OCR",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [_port("text-out", "text", "text", "output")],
        "defaults": {"instruction": "Read the license plate text from cropped plate images in English", "language": "en"},
        "configSchema": [
            _instruction_cfg("Describe what text to read and language preferences..."),
            _cfg("language", "Language", "select", options=[
                {"label": "English", "value": "en"},
                {"label": "Hindi", "value": "hi"},
            ]),
        ],
        "backend_handler": "ocr.easyocr",
    },
    {
        "type": "paddleocr",
        "label": "PaddleOCR",
        "description": "High-accuracy OCR engine for license plate recognition.",
        "category": "OCR",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [_port("text-out", "text", "text", "output")],
        "defaults": {"instruction": "Use high-accuracy OCR to read plate numbers precisely", "useAngleClassifier": True, "language": "en"},
        "configSchema": [
            _instruction_cfg("Describe OCR accuracy requirements..."),
            _cfg("useAngleClassifier", "Angle Classifier", "toggle"),
            _cfg("language", "Language", "select", options=[
                {"label": "English", "value": "en"},
                {"label": "Chinese", "value": "ch"},
            ]),
        ],
        "backend_handler": "ocr.paddleocr",
    },
    {
        "type": "text-reader",
        "label": "Text Reader",
        "description": "General-purpose text extraction from frames.",
        "category": "OCR",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [_port("text-out", "text", "text", "output")],
        "defaults": {"instruction": "Read visible text from the frame", "engine": "easyocr", "language": "en"},
        "configSchema": [
            _instruction_cfg("Describe what text to extract..."),
            _cfg("engine", "OCR Engine", "select", options=[
                {"label": "EasyOCR", "value": "easyocr"},
                {"label": "PaddleOCR", "value": "paddleocr"},
            ]),
            _cfg("language", "Language", "text"),
        ],
        "backend_handler": "ocr.text_reader",
    },
    # ── PostProcessing ──
    {
        "type": "object-tracker",
        "label": "Object Tracker",
        "description": "Tracks detected objects across frames (SORT/DeepSORT).",
        "category": "PostProcessing",
        "inputs": [
            _port("frame-in", "frame", "frame", "input"),
            _port("boxes-in", "boxes", "bounding_box_list", "input"),
        ],
        "outputs": [
            _port("frame-out", "frame", "frame", "output"),
            _port("boxes-out", "boxes", "bounding_box_list", "output"),
        ],
        "defaults": {"instruction": "Track objects across frames", "algorithm": "DeepSORT", "maxAge": 30},
        "configSchema": [
            _instruction_cfg("Describe tracking requirements..."),
            _cfg("algorithm", "Algorithm", "select", options=[
                {"label": "SORT", "value": "SORT"},
                {"label": "DeepSORT", "value": "DeepSORT"},
            ]),
            _cfg("maxAge", "Max Age (frames)", "number", min=1, max=300),
        ],
        "backend_handler": "postprocessing.object_tracker",
    },
    {
        "type": "optical-flow",
        "label": "Optical Flow",
        "description": "Computes dense optical flow between consecutive frames.",
        "category": "PostProcessing",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Compute optical flow", "method": "Farneback"},
        "configSchema": [
            _instruction_cfg("Describe optical flow requirements..."),
            _cfg("method", "Method", "select", options=[
                {"label": "Farneback", "value": "Farneback"},
                {"label": "Lucas-Kanade", "value": "LucasKanade"},
            ]),
        ],
        "backend_handler": "postprocessing.optical_flow",
    },
    {
        "type": "regex-validator",
        "label": "Regex Validator",
        "description": "Validates text against a regex pattern (e.g. Indian plate format).",
        "category": "PostProcessing",
        "inputs": [_port("text-in", "text", "text", "input")],
        "outputs": [_port("text-out", "text", "text", "output")],
        "defaults": {"instruction": "Validate detected plates against Indian number plate format XX00XX0000", "pattern": "^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$"},
        "configSchema": [
            _instruction_cfg("Describe what format plates should match..."),
            _cfg("pattern", "Pattern", "text", required=True),
        ],
        "backend_handler": "postprocessing.regex_validator",
    },
    {
        "type": "deduplicator",
        "label": "Deduplicator",
        "description": "Filters duplicate plate detections using perceptual hashing.",
        "category": "PostProcessing",
        "inputs": [_port("text-in", "text", "text", "input")],
        "outputs": [_port("text-out", "text", "text", "output")],
        "defaults": {"instruction": "Remove duplicate plate readings within a 10-second window", "windowSeconds": 10, "hashThreshold": 8},
        "configSchema": [
            _instruction_cfg("Describe deduplication rules and timing..."),
            _cfg("windowSeconds", "Dedup Window (s)", "number", min=1, max=60),
            _cfg("hashThreshold", "Hash Threshold", "number", min=0, max=64),
        ],
        "backend_handler": "postprocessing.deduplicator",
    },
    {
        "type": "threshold-gate",
        "label": "Threshold Gate",
        "description": "Passes or blocks data based on a numeric score threshold.",
        "category": "PostProcessing",
        "inputs": [_port("score-in", "score", "number", "input")],
        "outputs": [_port("score-out", "score", "number", "output")],
        "defaults": {"instruction": "Filter results by confidence threshold", "threshold": 0.5, "operator": ">="},
        "configSchema": [
            _instruction_cfg("Describe filtering criteria..."),
            _cfg("threshold", "Threshold", "number", min=0, max=1),
            _cfg("operator", "Operator", "select", options=[
                {"label": ">=", "value": ">="},
                {"label": ">", "value": ">"},
                {"label": "<=", "value": "<="},
                {"label": "<", "value": "<"},
            ]),
        ],
        "backend_handler": "postprocessing.threshold_gate",
    },
    {
        "type": "face-anonymizer",
        "label": "Face Anonymizer",
        "description": "Blurs or pixelates detected faces for privacy compliance.",
        "category": "PostProcessing",
        "inputs": [
            _port("frame-in", "frame", "frame", "input"),
            _port("boxes-in", "boxes", "bounding_box_list", "input"),
        ],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Blur all detected faces for privacy", "method": "gaussian", "strength": 25},
        "configSchema": [
            _instruction_cfg("Describe anonymization requirements..."),
            _cfg("method", "Method", "select", options=[
                {"label": "Gaussian Blur", "value": "gaussian"},
                {"label": "Pixelate", "value": "pixelate"},
            ]),
            _cfg("strength", "Strength", "number", min=1, max=100),
        ],
        "backend_handler": "postprocessing.face_anonymizer",
    },
    {
        "type": "adjudicator",
        "label": "Adjudicator",
        "description": "Multi-engine OCR result adjudication — picks best plate reading.",
        "category": "PostProcessing",
        "inputs": [_port("text-in", "text", "text", "input")],
        "outputs": [_port("text-out", "text", "text", "output")],
        "defaults": {"instruction": "Pick the most accurate plate reading from multiple OCR engines using confidence scoring", "strategy": "confidence"},
        "configSchema": [
            _instruction_cfg("Describe how to pick the best OCR result..."),
            _cfg("strategy", "Strategy", "select", options=[
                {"label": "Confidence", "value": "confidence"},
                {"label": "Majority Vote", "value": "majority"},
                {"label": "LLM Adjudication", "value": "llm"},
            ]),
        ],
        "backend_handler": "postprocessing.adjudicator",
    },
    # ── Output ──
    {
        "type": "heatmap-generator",
        "label": "Heatmap Generator",
        "description": "Generates activity / density heatmaps over time.",
        "category": "Output",
        "inputs": [
            _port("frame-in", "frame", "frame", "input"),
            _port("boxes-in", "boxes", "bounding_box_list", "input"),
        ],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Generate activity heatmap", "decay": 0.95, "colormap": "jet"},
        "configSchema": [
            _instruction_cfg("Describe heatmap configuration..."),
            _cfg("decay", "Decay Factor", "number", min=0.5, max=1.0),
            _cfg("colormap", "Color Map", "select", options=[
                {"label": "Jet", "value": "jet"},
                {"label": "Hot", "value": "hot"},
                {"label": "Viridis", "value": "viridis"},
            ]),
        ],
        "backend_handler": "output.heatmap_generator",
    },
    {
        "type": "annotator",
        "label": "Annotator",
        "description": "Draws detections on top of a frame.",
        "category": "Output",
        "inputs": [
            _port("frame-in", "frame", "frame", "input"),
            _port("boxes-in", "boxes", "bounding_box_list", "input"),
        ],
        "outputs": [_port("frame-out", "frame", "frame", "output")],
        "defaults": {"instruction": "Draw bounding boxes and labels on the live video feed", "showLabels": True},
        "configSchema": [
            _instruction_cfg("Describe how detections should be displayed..."),
            _cfg("showLabels", "Show Labels", "toggle"),
        ],
        "backend_handler": "output.annotator",
    },
    {
        "type": "dispatcher",
        "label": "Dispatcher",
        "description": "Dispatches validated detections to storage and alerting sinks.",
        "category": "Output",
        "inputs": [_port("text-in", "text", "text", "input")],
        "outputs": [],
        "defaults": {"instruction": "Save all validated plate readings to the database and notify via Redis", "sinks": "database,redis"},
        "configSchema": [
            _instruction_cfg("Describe where to send results and any alerting rules..."),
            _cfg("sinks", "Sinks (comma-separated)", "text", required=True),
        ],
        "backend_handler": "output.dispatcher",
    },
    {
        "type": "alert-trigger",
        "label": "Alert Trigger",
        "description": "Fires alerts via webhook, email, or push notification.",
        "category": "Output",
        "inputs": [_port("score-in", "score", "number", "input")],
        "outputs": [],
        "defaults": {"instruction": "Send alert when threshold exceeded", "channel": "webhook", "url": ""},
        "configSchema": [
            _instruction_cfg("Describe alert conditions and channels..."),
            _cfg("channel", "Channel", "select", options=[
                {"label": "Webhook", "value": "webhook"},
                {"label": "Email", "value": "email"},
                {"label": "Push", "value": "push"},
            ]),
            _cfg("url", "Webhook URL", "text"),
        ],
        "backend_handler": "output.alert_trigger",
    },
    {
        "type": "console-logger",
        "label": "Console Logger",
        "description": "Logs text output for inspection.",
        "category": "Output",
        "inputs": [_port("text-in", "text", "text", "input")],
        "outputs": [],
        "defaults": {"instruction": "Log all detected plates to the console for monitoring", "level": "info"},
        "configSchema": [
            _instruction_cfg("Describe what to log and how..."),
            _cfg("level", "Level", "select", options=[
                {"label": "Info", "value": "info"},
                {"label": "Warning", "value": "warning"},
            ]),
        ],
        "backend_handler": "output.console_logger",
    },
    {
        "type": "video-recorder",
        "label": "Video Recorder",
        "description": "Records annotated frames to a video file.",
        "category": "Output",
        "inputs": [_port("frame-in", "frame", "frame", "input")],
        "outputs": [],
        "defaults": {"instruction": "Record annotated output to video", "outputPath": "/recordings/out.mp4", "codec": "mp4v", "fps": 15},
        "configSchema": [
            _instruction_cfg("Describe recording preferences..."),
            _cfg("outputPath", "Output Path", "text", required=True),
            _cfg("codec", "Codec", "select", options=[
                {"label": "MP4V", "value": "mp4v"},
                {"label": "XVID", "value": "XVID"},
                {"label": "H264", "value": "avc1"},
            ]),
            _cfg("fps", "FPS", "number", min=1, max=60),
        ],
        "backend_handler": "output.video_recorder",
    },
]


# ── Default categories & port types ──────────────────────────────

DEFAULT_CATEGORIES: Dict[str, str] = {
    "Input": "#22d3ee",
    "Ingestion": "#06b6d4",
    "Detection": "#f43f5e",
    "Preprocessing": "#fb923c",
    "OCR": "#4ade80",
    "PostProcessing": "#facc15",
    "Output": "#60a5fa",
    "Utility": "#94a3b8",
}

DEFAULT_PORT_TYPES: Dict[str, str] = {
    "frame": "#60a5fa",
    "bounding_box_list": "#fb923c",
    "text": "#4ade80",
    "config": "#f8fafc",
    "number": "#7dd3fc",
    "boolean": "#c084fc",
    "any": "#d4d4d8",
}


class BlockRegistry:
    """Dynamic block registry — serves blocks to the frontend and graph engine.

    Starts with builtin blocks and can be extended at runtime with
    custom user-defined blocks or blocks discovered from the
    ComponentRegistry / HubClient.
    """

    def __init__(self) -> None:
        self._blocks: Dict[str, Dict[str, Any]] = {}
        self._categories: Dict[str, str] = dict(DEFAULT_CATEGORIES)
        self._port_types: Dict[str, str] = dict(DEFAULT_PORT_TYPES)

        # Register all builtins
        for block_def in _BUILTIN_BLOCKS:
            self._blocks[block_def["type"]] = block_def

    # ── Query ─────────────────────────────────────────────────────

    def get_all_blocks(self) -> List[Dict[str, Any]]:
        """Return all registered block definitions."""
        return list(self._blocks.values())

    def get_block(self, block_type: str) -> Optional[Dict[str, Any]]:
        """Return a single block definition by type, or None."""
        return self._blocks.get(block_type)

    def get_categories(self) -> Dict[str, str]:
        """Return category → color mapping."""
        return dict(self._categories)

    def get_port_types(self) -> Dict[str, str]:
        """Return port type → color mapping."""
        return dict(self._port_types)

    # ── Mutation ──────────────────────────────────────────────────

    def register_block(self, block_def: Dict[str, Any]) -> None:
        """Register a new block definition (or overwrite an existing one).

        Automatically adds unknown categories and port types with
        default colors.
        """
        block_type = block_def.get("type")
        if not block_type:
            raise ValueError("Block definition must have a 'type' key")

        self._blocks[block_type] = block_def

        # Auto-register category if new
        cat = block_def.get("category", "Utility")
        if cat not in self._categories:
            self._categories[cat] = "#94a3b8"  # default gray
            logger.info("auto_registered_category category=%s", cat)

        # Auto-register port types if new
        for port_list in (block_def.get("inputs", []), block_def.get("outputs", [])):
            for port in port_list:
                pt = port.get("type", "any")
                if pt not in self._port_types:
                    self._port_types[pt] = "#d4d4d8"  # default zinc
                    logger.info("auto_registered_port_type type=%s", pt)

        logger.info("block_registered type=%s label=%s", block_type, block_def.get("label"))

    def register_category(self, name: str, color: str) -> None:
        """Register or update a category color."""
        self._categories[name] = color

    def register_port_type(self, name: str, color: str) -> None:
        """Register or update a port type color."""
        self._port_types[name] = color

    def remove_block(self, block_type: str) -> bool:
        """Remove a block definition. Returns True if it existed."""
        return self._blocks.pop(block_type, None) is not None

    def block_count(self) -> int:
        return len(self._blocks)
