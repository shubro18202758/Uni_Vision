/**
 * Static block definitions — fallback when backend is unreachable.
 * Universal multipurpose blocks for any industry / domain.
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
    defaults: { instruction: "Load images for analysis", path: "/images/sample.jpg" },
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
    defaults: { instruction: "Connect to camera and stream live video", streamUrl: "rtsp://camera.local/stream", fps: 15 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe which camera to connect to..." },
      { key: "streamUrl", label: "Stream URL", type: "text", required: true },
      { key: "fps", label: "Target FPS", type: "number", required: false, min: 1, max: 30 },
    ],
  },
  {
    type: "video-file",
    label: "Video File",
    description: "Load and decode a local video file frame-by-frame.",
    category: "Input",
    inputs: [],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Load the uploaded video for analysis", path: "/videos/clip.mp4", fps: 10 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe the video source..." },
      { key: "path", label: "Video Path", type: "text", required: true },
      { key: "fps", label: "Decode FPS", type: "number", required: false, min: 1, max: 30 },
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
    defaults: { instruction: "Sample frames to keep processing manageable", sampleRate: 5 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe how you want frames sampled..." },
      { key: "sampleRate", label: "Sample Rate (fps)", type: "number", required: true, min: 1, max: 30 },
    ],
  },
  {
    type: "roi-crop",
    label: "ROI Crop",
    description: "Crop frames to a region-of-interest for focused analysis.",
    category: "Ingestion",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Crop to the area of interest", x: 0, y: 0, width: 640, height: 480 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe which region to focus on..." },
      { key: "x", label: "X Offset", type: "number", min: 0 },
      { key: "y", label: "Y Offset", type: "number", min: 0 },
      { key: "width", label: "Width", type: "number", min: 1 },
      { key: "height", label: "Height", type: "number", min: 1 },
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
    defaults: { instruction: "Convert frames to grayscale" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe preprocessing preferences..." },
    ],
  },
  {
    type: "contrast-enhance",
    label: "Contrast Enhance",
    description: "Enhance contrast via CLAHE adaptive histogram equalisation.",
    category: "Preprocessing",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Improve visibility in low-contrast frames", clipLimit: 3.0 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe contrast needs..." },
      { key: "clipLimit", label: "CLAHE Clip Limit", type: "number", min: 1, max: 10 },
    ],
  },
  {
    type: "denoise",
    label: "Denoise",
    description: "Apply noise reduction for cleaner frames.",
    category: "Preprocessing",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Reduce noise for better analysis", strength: 10 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe noise conditions..." },
      { key: "strength", label: "Denoise Strength", type: "number", min: 1, max: 30 },
    ],
  },
  {
    type: "resize",
    label: "Resize",
    description: "Resize frames to a uniform resolution.",
    category: "Preprocessing",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Resize to standard resolution", width: 640, height: 480 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Target resolution..." },
      { key: "width", label: "Width", type: "number", required: true, min: 64, max: 3840 },
      { key: "height", label: "Height", type: "number", required: true, min: 64, max: 2160 },
    ],
  },

  /* ── Detection ─────────────────────────────────────── */
  {
    type: "yolo-detector",
    label: "YOLO Detector",
    description: "General-purpose object detection — people, vehicles, animals, objects.",
    category: "Detection",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [
      { id: "frame-out", name: "frame", type: "frame", direction: "output" },
      { id: "boxes-out", name: "boxes", type: "bounding_box_list", direction: "output" },
    ],
    defaults: { instruction: "Detect all objects in each frame", model: "yolov8n.pt", confidence: 0.5 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe what to detect..." },
      { key: "model", label: "Model", type: "text", required: true },
      { key: "confidence", label: "Confidence", type: "number", required: true, min: 0, max: 1 },
    ],
  },
  {
    type: "motion-detector",
    label: "Motion Detector",
    description: "Detect motion and moving objects via frame differencing.",
    category: "Detection",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [
      { id: "frame-out", name: "frame", type: "frame", direction: "output" },
      { id: "boxes-out", name: "boxes", type: "bounding_box_list", direction: "output" },
    ],
    defaults: { instruction: "Detect any motion in the scene", sensitivity: 0.5 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe motion detection needs..." },
      { key: "sensitivity", label: "Sensitivity", type: "number", required: true, min: 0, max: 1 },
    ],
  },
  {
    type: "fire-smoke-detector",
    label: "Fire & Smoke Detector",
    description: "Detect fire, smoke, and thermal anomalies.",
    category: "Detection",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [
      { id: "frame-out", name: "frame", type: "frame", direction: "output" },
      { id: "boxes-out", name: "boxes", type: "bounding_box_list", direction: "output" },
    ],
    defaults: { instruction: "Detect any fire or smoke in the environment", confidence: 0.4 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe fire/smoke detection context..." },
      { key: "confidence", label: "Confidence", type: "number", required: true, min: 0, max: 1 },
    ],
  },
  {
    type: "crowd-density",
    label: "Crowd Density Analyzer",
    description: "Estimate crowd density and count people in a scene.",
    category: "Detection",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [
      { id: "frame-out", name: "frame", type: "frame", direction: "output" },
      { id: "score-out", name: "score", type: "score", direction: "output" },
    ],
    defaults: { instruction: "Monitor crowd density and count", maxCapacity: 100 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe crowd monitoring goals..." },
      { key: "maxCapacity", label: "Max Capacity", type: "number", min: 1 },
    ],
  },
  {
    type: "vehicle-detector",
    label: "Vehicle Detector",
    description: "Detect and classify vehicles (car, truck, bus, bike).",
    category: "Detection",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [
      { id: "frame-out", name: "frame", type: "frame", direction: "output" },
      { id: "boxes-out", name: "boxes", type: "bounding_box_list", direction: "output" },
    ],
    defaults: { instruction: "Detect all vehicles", model: "yolov8n.pt", confidence: 0.5, classes: "car,truck,bus,motorcycle" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe vehicle detection needs..." },
      { key: "model", label: "Model Path", type: "text", required: true },
      { key: "confidence", label: "Confidence", type: "number", required: true, min: 0, max: 1 },
      { key: "classes", label: "Vehicle Classes", type: "text", required: false },
    ],
  },
  {
    type: "plate-detector",
    label: "Plate Detector",
    description: "Detect license plates within vehicle bounding boxes.",
    category: "Detection",
    inputs: [
      { id: "frame-in", name: "frame", type: "frame", direction: "input" },
      { id: "boxes-in", name: "boxes", type: "bounding_box_list", direction: "input" },
    ],
    outputs: [
      { id: "plates-out", name: "plates", type: "frame", direction: "output" },
      { id: "boxes-out", name: "plate_boxes", type: "bounding_box_list", direction: "output" },
    ],
    defaults: { instruction: "Find license plates on detected vehicles", model: "plate_detect.pt", confidence: 0.6 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe plate detection needs..." },
      { key: "model", label: "Plate Model", type: "text", required: true },
      { key: "confidence", label: "Confidence", type: "number", required: true, min: 0, max: 1 },
    ],
  },

  /* ── Analysis ──────────────────────────────────────── */
  {
    type: "scene-classifier",
    label: "Scene Classifier",
    description: "Classify scene type (indoor/outdoor, industrial, residential, etc.).",
    category: "Analysis",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "score-out", name: "score", type: "score", direction: "output" }],
    defaults: { instruction: "Classify the environment type" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe classification needs..." },
    ],
  },
  {
    type: "anomaly-scorer",
    label: "Anomaly Scorer",
    description: "Score frame anomaly level using learned baselines.",
    category: "Analysis",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "score-out", name: "score", type: "score", direction: "output" }],
    defaults: { instruction: "Score anomaly level against normal baseline", threshold: 0.6 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe what counts as anomalous..." },
      { key: "threshold", label: "Anomaly Threshold", type: "number", min: 0, max: 1 },
    ],
  },
  {
    type: "pose-estimator",
    label: "Pose Estimator",
    description: "Estimate human body pose keypoints for posture and activity analysis.",
    category: "Analysis",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [
      { id: "frame-out", name: "frame", type: "frame", direction: "output" },
      { id: "boxes-out", name: "boxes", type: "bounding_box_list", direction: "output" },
    ],
    defaults: { instruction: "Estimate body poses for activity analysis" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe what posture or activity to analyze..." },
    ],
  },
  {
    type: "ppe-detector",
    label: "PPE / Safety Gear Detector",
    description: "Detect personal protective equipment (helmets, vests, goggles).",
    category: "Analysis",
    inputs: [
      { id: "frame-in", name: "frame", type: "frame", direction: "input" },
      { id: "boxes-in", name: "boxes", type: "bounding_box_list", direction: "input" },
    ],
    outputs: [
      { id: "frame-out", name: "frame", type: "frame", direction: "output" },
      { id: "score-out", name: "score", type: "score", direction: "output" },
    ],
    defaults: { instruction: "Check workers for PPE compliance", requiredGear: "helmet,vest" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe PPE requirements..." },
      { key: "requiredGear", label: "Required Gear", type: "text" },
    ],
  },
  {
    type: "zone-intrusion",
    label: "Zone Intrusion Detector",
    description: "Detect objects or people entering restricted zones.",
    category: "Analysis",
    inputs: [
      { id: "frame-in", name: "frame", type: "frame", direction: "input" },
      { id: "boxes-in", name: "boxes", type: "bounding_box_list", direction: "input" },
    ],
    outputs: [
      { id: "frame-out", name: "frame", type: "frame", direction: "output" },
      { id: "score-out", name: "score", type: "score", direction: "output" },
    ],
    defaults: { instruction: "Monitor restricted area for intrusions" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe the zone boundaries..." },
    ],
  },
  {
    type: "llm-vision",
    label: "LLM Vision Analyzer",
    description: "Multipurpose AI vision analysis — scene understanding, anomaly reasoning, threat detection.",
    category: "Analysis",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [
      { id: "score-out", name: "score", type: "score", direction: "output" },
      { id: "text-out", name: "text", type: "text", direction: "output" },
    ],
    defaults: { instruction: "Analyze the scene for any anomalies, threats, or noteworthy events" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe what to analyze — works for ANY domain..." },
    ],
  },

  /* ── Tracking ──────────────────────────────────────── */
  {
    type: "object-tracker",
    label: "Object Tracker",
    description: "Track detected objects across frames (SORT / DeepSORT).",
    category: "Tracking",
    inputs: [
      { id: "frame-in", name: "frame", type: "frame", direction: "input" },
      { id: "boxes-in", name: "boxes", type: "bounding_box_list", direction: "input" },
    ],
    outputs: [
      { id: "frame-out", name: "frame", type: "frame", direction: "output" },
      { id: "boxes-out", name: "boxes", type: "bounding_box_list", direction: "output" },
    ],
    defaults: { instruction: "Track objects across consecutive frames", algorithm: "deepsort" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe tracking needs..." },
      {
        key: "algorithm",
        label: "Algorithm",
        type: "select",
        options: [
          { label: "DeepSORT", value: "deepsort" },
          { label: "SORT", value: "sort" },
          { label: "ByteTrack", value: "bytetrack" },
        ],
      },
    ],
  },
  {
    type: "optical-flow",
    label: "Optical Flow",
    description: "Compute dense optical flow to visualise motion patterns.",
    category: "Tracking",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Visualize motion patterns in the scene" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe motion analysis needs..." },
    ],
  },

  /* ── OCR / Reading ─────────────────────────────────── */
  {
    type: "text-reader",
    label: "Text Reader (OCR)",
    description: "Read text from image regions — signs, labels, plates, documents.",
    category: "OCR",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "text-out", name: "text", type: "text", direction: "output" }],
    defaults: { instruction: "Read any visible text in the scene", language: "en" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe what text to read..." },
      {
        key: "language",
        label: "Language",
        type: "select",
        options: [
          { label: "English", value: "en" },
          { label: "Hindi", value: "hi" },
          { label: "Multi", value: "multi" },
        ],
      },
    ],
  },
  {
    type: "plate-preprocessor",
    label: "Plate Preprocessor",
    description: "Deskew, enhance contrast, and resize plate crops for OCR.",
    category: "OCR",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Clean up plate crops for OCR", deskew: true, clahe: true, targetWidth: 200 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe plate preprocessing..." },
      { key: "deskew", label: "Deskew", type: "toggle" },
      { key: "clahe", label: "CLAHE Enhancement", type: "toggle" },
      { key: "targetWidth", label: "Target Width (px)", type: "number", min: 50, max: 500 },
    ],
  },

  /* ── PostProcessing ────────────────────────────────── */
  {
    type: "deduplicator",
    label: "Deduplicator",
    description: "Filters duplicate detections using perceptual hashing.",
    category: "PostProcessing",
    inputs: [{ id: "text-in", name: "text", type: "text", direction: "input" }],
    outputs: [{ id: "text-out", name: "text", type: "text", direction: "output" }],
    defaults: { instruction: "Remove duplicate detections within a time window", windowSeconds: 10, hashThreshold: 8 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe deduplication rules..." },
      { key: "windowSeconds", label: "Dedup Window (s)", type: "number", min: 1, max: 60 },
      { key: "hashThreshold", label: "Hash Threshold", type: "number", min: 0, max: 64 },
    ],
  },
  {
    type: "threshold-gate",
    label: "Threshold Gate",
    description: "Pass events only when confidence or anomaly score exceeds a threshold.",
    category: "PostProcessing",
    inputs: [{ id: "score-in", name: "score", type: "score", direction: "input" }],
    outputs: [{ id: "score-out", name: "score", type: "score", direction: "output" }],
    defaults: { instruction: "Only pass high-confidence detections", threshold: 0.6 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe threshold criteria..." },
      { key: "threshold", label: "Threshold", type: "number", required: true, min: 0, max: 1 },
    ],
  },
  {
    type: "face-anonymizer",
    label: "Face Anonymizer",
    description: "Blur or pixelate detected faces for privacy compliance.",
    category: "PostProcessing",
    inputs: [
      { id: "frame-in", name: "frame", type: "frame", direction: "input" },
      { id: "boxes-in", name: "boxes", type: "bounding_box_list", direction: "input" },
    ],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Blur all faces for privacy", method: "gaussian" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe anonymization needs..." },
      {
        key: "method",
        label: "Method",
        type: "select",
        options: [
          { label: "Gaussian Blur", value: "gaussian" },
          { label: "Pixelate", value: "pixelate" },
        ],
      },
    ],
  },
  {
    type: "heatmap-generator",
    label: "Heatmap Generator",
    description: "Generate spatial heatmaps from accumulated detection or motion data.",
    category: "PostProcessing",
    inputs: [
      { id: "frame-in", name: "frame", type: "frame", direction: "input" },
      { id: "boxes-in", name: "boxes", type: "bounding_box_list", direction: "input" },
    ],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Generate activity heatmap overlay", decay: 0.95 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe heatmap goals..." },
      { key: "decay", label: "Decay Factor", type: "number", min: 0.5, max: 1.0 },
    ],
  },

  /* ── Output ────────────────────────────────────────── */
  {
    type: "annotator",
    label: "Annotator",
    description: "Draws detections, labels, and overlays on frames.",
    category: "Output",
    inputs: [
      { id: "frame-in", name: "frame", type: "frame", direction: "input" },
      { id: "boxes-in", name: "boxes", type: "bounding_box_list", direction: "input" },
    ],
    outputs: [{ id: "frame-out", name: "frame", type: "frame", direction: "output" }],
    defaults: { instruction: "Draw bounding boxes and labels on the live feed", showLabels: true },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe how detections should be displayed..." },
      { key: "showLabels", label: "Show Labels", type: "toggle" },
    ],
  },
  {
    type: "dispatcher",
    label: "Dispatcher",
    description: "Dispatches detections to storage, webhooks, and alerting sinks.",
    category: "Output",
    inputs: [{ id: "text-in", name: "text", type: "text", direction: "input" }],
    outputs: [],
    defaults: { instruction: "Save all findings to the database and notify", sinks: "database,redis" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe where to send results..." },
      { key: "sinks", label: "Sinks (comma-separated)", type: "text", required: true },
    ],
  },
  {
    type: "alert-trigger",
    label: "Alert Trigger",
    description: "Fire real-time alerts via webhook, email, or push notification.",
    category: "Output",
    inputs: [{ id: "score-in", name: "score", type: "score", direction: "input" }],
    outputs: [],
    defaults: { instruction: "Send alerts for high-risk detections", channel: "webhook", minSeverity: "medium" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe alert conditions..." },
      {
        key: "channel",
        label: "Channel",
        type: "select",
        options: [
          { label: "Webhook", value: "webhook" },
          { label: "Email", value: "email" },
          { label: "Push", value: "push" },
        ],
      },
      {
        key: "minSeverity",
        label: "Min Severity",
        type: "select",
        options: [
          { label: "Low", value: "low" },
          { label: "Medium", value: "medium" },
          { label: "High", value: "high" },
          { label: "Critical", value: "critical" },
        ],
      },
    ],
  },
  {
    type: "console-logger",
    label: "Console Logger",
    description: "Logs text output for inspection and debugging.",
    category: "Output",
    inputs: [{ id: "text-in", name: "text", type: "text", direction: "input" }],
    outputs: [],
    defaults: { instruction: "Log all findings to the console", level: "info" },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe what to log..." },
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
  {
    type: "video-recorder",
    label: "Video Recorder",
    description: "Record annotated video clips of flagged events for review.",
    category: "Output",
    inputs: [{ id: "frame-in", name: "frame", type: "frame", direction: "input" }],
    outputs: [],
    defaults: { instruction: "Record flagged event clips for evidence", preBufferSec: 5, postBufferSec: 10 },
    configSchema: [
      { key: "instruction", label: "Your Instruction", type: "textarea", placeholder: "Describe recording preferences..." },
      { key: "preBufferSec", label: "Pre-Buffer (s)", type: "number", min: 0, max: 30 },
      { key: "postBufferSec", label: "Post-Buffer (s)", type: "number", min: 0, max: 60 },
    ],
  },
];
