/**
 * Static block definitions — fallback when backend is unreachable.
 * These are the original 16 blocks that ship with the prototype.
 */

import type { BlockDefinition } from "../types/block";

export const STATIC_BLOCK_DEFAULTS: BlockDefinition[] = [
  /* ── Input ─────────────────────────────────────────── */
  {
    type: "image-input",
    label: "Image Input",
    description: "Loads a single image into the pipeline.",
    category: "Input",
    inputs: [],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Load test images from my surveillance camera folder", path: "/images/sample.jpg" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe what images to load and from where..." },
      { key: "path", label: "Image Path", type: "text", required: true },
    ],
  },
  {
    type: "rtsp-stream",
    label: "RTSP Stream",
    description: "Streams frames from a remote camera source.",
    category: "Input",
    inputs: [],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Connect to the parking lot entrance camera and stream live video", streamUrl: "rtsp://camera.local/stream", fps: 15 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe which camera to connect to and any streaming preferences..." },
      { key: "streamUrl", label: "Stream URL", type: "text", required: true },
      { key: "fps", label: "Target FPS", type: "number", required: false, min: 1, max: 30 },
    ],
  },

  /* ── Ingestion ─────────────────────────────────────── */
  {
    type: "frame-sampler",
    label: "Frame Sampler",
    description: "Samples frames at a target rate to reduce processing load.",
    category: "Ingestion",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Sample every 3rd frame to keep processing manageable", sampleRate: 5 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe how you want frames sampled..." },
      { key: "sampleRate", label: "Sample Rate (fps)", type: "number", required: true, min: 1, max: 30 },
    ],
  },

  /* ── Detection ─────────────────────────────────────── */
  {
    type: "yolo-detector",
    label: "YOLO Detector",
    description: "Detects objects and emits bounding boxes.",
    category: "Detection",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [
      { id: "frame-out", name: "frame", type: "frame", direction: "output" },
      { id: "boxes-out", name: "boxes", type: "bounding_box_list", direction: "output" },
    ],
    defaults: { instruction: "Detect all objects in each frame with high accuracy", model: "yolov8n.pt", confidence: 0.6 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe what objects to detect and any priority areas..." },
      { key: "model", label: "Model", type: "text", required: true },
      { key: "confidence", label: "Confidence", type: "number", required: true, min: 0, max: 1 },
    ],
  },
  {
    type: "vehicle-detector",
    label: "Vehicle Detector",
    description: "Detects vehicles and classifies type (car, truck, bus, bike).",
    category: "Detection",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [
      { id: "frame-out", name: "frame", type: "frame", direction: "output" },
      { id: "boxes-out", name: "boxes", type: "bounding_box_list", direction: "output" },
    ],
    defaults: { instruction: "Detect all vehicles — focus on cars, trucks, buses, and motorcycles", model: "yolov8n.pt", confidence: 0.5, classes: "car,truck,bus,motorcycle" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe which vehicle types to detect and any special conditions..." },
      { key: "model", label: "Model Path", type: "text", required: true },
      { key: "confidence", label: "Confidence", type: "number", required: true, min: 0, max: 1 },
      { key: "classes", label: "Vehicle Classes", type: "text", required: false },
    ],
  },
  {
    type: "plate-detector",
    label: "Plate Detector",
    description: "Detects license plates within vehicle bounding boxes.",
    category: "Detection",
    inputs: [
      { id: "frame-in", name: "frame", type: "frame", direction: "input" },
      { id: "boxes-in", name: "boxes", type: "bounding_box_list", direction: "input" },
    ],
    outputs: [
      { id: "plates-out", name: "plates", type: "frame", direction: "output" },
      { id: "boxes-out", name: "plate_boxes", type: "bounding_box_list", direction: "output" },
    ],
    defaults: { instruction: "Find number plates on all detected vehicles", model: "plate_detect.pt", confidence: 0.6 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe plate detection requirements and any region specifics..." },
      { key: "model", label: "Plate Model", type: "text", required: true },
      { key: "confidence", label: "Confidence", type: "number", required: true, min: 0, max: 1 },
    ],
  },

  /* ── Preprocessing ─────────────────────────────────── */
  {
    type: "grayscale",
    label: "Grayscale",
    description: "Converts frames into grayscale.",
    category: "Preprocessing",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Convert frames to grayscale for faster downstream processing" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe any preprocessing preferences..." },
    ],
  },
  {
    type: "plate-preprocessor",
    label: "Plate Preprocessor",
    description: "Deskews, enhances contrast, and resizes plate crops for OCR.",
    category: "Preprocessing",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Clean up and deskew the plate crops, enhance contrast for better OCR", deskew: true, clahe: true, targetWidth: 200 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe how plate images should be cleaned up..." },
      { key: "deskew", label: "Deskew", type: "toggle" },
      { key: "clahe", label: "CLAHE Enhancement", type: "toggle" },
      { key: "targetWidth", label: "Target Width (px)", type: "number", min: 50, max: 500 },
    ],
  },

  /* ── OCR ────────────────────────────────────────────── */
  {
    type: "easy-ocr",
    label: "EasyOCR",
    description: "Reads text from image regions using EasyOCR.",
    category: "OCR",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "text-out", name: "text", type: "text", direction: "output" }],
    defaults: { instruction: "Read the license plate text from cropped plate images in English", language: "en" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe what text to read and language preferences..." },
      {
        key: "language",
        label: "Language",
        type: "select",
        options: [
          { label: "English", value: "en" },
          { label: "Hindi", value: "hi" },
        ],
      },
    ],
  },
  {
    type: "paddleocr",
    label: "PaddleOCR",
    description: "High-accuracy OCR engine for license plate recognition.",
    category: "OCR",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "text-out", name: "text", type: "text", direction: "output" }],
    defaults: { instruction: "Use high-accuracy OCR to read plate numbers precisely", useAngleClassifier: true, language: "en" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe OCR accuracy requirements..." },
      { key: "useAngleClassifier", label: "Angle Classifier", type: "toggle" },
      {
        key: "language",
        label: "Language",
        type: "select",
        options: [
          { label: "English", value: "en" },
          { label: "Chinese", value: "ch" },
        ],
      },
    ],
  },

  /* ── PostProcessing ────────────────────────────────── */
  {
    type: "regex-validator",
    label: "Regex Validator",
    description: "Validates text against a regex pattern (e.g. Indian plate format).",
    category: "PostProcessing",
    inputs: [{ id: "text-in", name: "text", type: "text", direction: "input" }],
    outputs: [{ id: "text-out", name: "text", type: "text", direction: "output" }],
    defaults: { instruction: "Validate detected plates against Indian number plate format XX00XX0000", pattern: "^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe what format plates should match..." },
      { key: "pattern", label: "Pattern", type: "text", required: true },
    ],
  },
  {
    type: "deduplicator",
    label: "Deduplicator",
    description: "Filters duplicate plate detections using perceptual hashing.",
    category: "PostProcessing",
    inputs: [{ id: "text-in", name: "text", type: "text", direction: "input" }],
    outputs: [{ id: "text-out", name: "text", type: "text", direction: "output" }],
    defaults: { instruction: "Remove duplicate plate readings within a 10-second window", windowSeconds: 10, hashThreshold: 8 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe deduplication rules and timing..." },
      { key: "windowSeconds", label: "Dedup Window (s)", type: "number", min: 1, max: 60 },
      { key: "hashThreshold", label: "Hash Threshold", type: "number", min: 0, max: 64 },
    ],
  },
  {
    type: "adjudicator",
    label: "Adjudicator",
    description: "Multi-engine OCR result adjudication — picks best plate reading.",
    category: "PostProcessing",
    inputs: [{ id: "text-in", name: "text", type: "text", direction: "input" }],
    outputs: [{ id: "text-out", name: "text", type: "text", direction: "output" }],
    defaults: { instruction: "Pick the most accurate plate reading from multiple OCR engines using confidence scoring", strategy: "confidence" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe how to pick the best OCR result..." },
      {
        key: "strategy",
        label: "Strategy",
        type: "select",
        options: [
          { label: "Confidence", value: "confidence" },
          { label: "Majority Vote", value: "majority" },
          { label: "LLM Adjudication", value: "llm" },
        ],
      },
    ],
  },

  /* ── Output ────────────────────────────────────────── */
  {
    type: "annotator",
    label: "Annotator",
    description: "Draws detections on top of a frame.",
    category: "Output",
    inputs: [
      { id: "frame-in", name: "frame", type: "frame", direction: "input" },
      { id: "boxes-in", name: "boxes", type: "bounding_box_list", direction: "input" },
    ],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Draw bounding boxes and labels on the live video feed", showLabels: true },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe how detections should be displayed..." },
      { key: "showLabels", label: "Show Labels", type: "toggle" },
    ],
  },
  {
    type: "dispatcher",
    label: "Dispatcher",
    description: "Dispatches validated detections to storage and alerting sinks.",
    category: "Output",
    inputs: [{ id: "text-in", name: "text", type: "text", direction: "input" }],
    outputs: [],
    defaults: { instruction: "Save all validated plate readings to the database and notify via Redis", sinks: "database,redis" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe where to send results and any alerting rules..." },
      { key: "sinks", label: "Sinks (comma-separated)", type: "text", required: true },
    ],
  },
  {
    type: "console-logger",
    label: "Console Logger",
    description: "Logs text output for inspection.",
    category: "Output",
    inputs: [{ id: "text-in", name: "text", type: "text", direction: "input" }],
    outputs: [],
    defaults: { instruction: "Log all detected plates to the console for monitoring", level: "info" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe what to log and how..." },
      {
        key: "level",
        label: "Level",
        type: "select",
        options: [
          { label: "Info", value: "info" },
          { label: "Warning", value: "warning" },
        ],
      },
    ],
  },
];
