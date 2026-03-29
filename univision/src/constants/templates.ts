import type { ProjectGraph } from "../types/graph";

export const STARTER_TEMPLATES: ProjectGraph[] = [
  {
    project: { name: "Blank Pipeline", version: "0.1.0" },
    blocks: [],
    connections: [],
  },
  {
    project: { name: "Detection Starter", version: "0.1.0" },
    blocks: [
      {
        id: "block-input",
        type: "rtsp-stream",
        label: "RTSP Stream",
        category: "Input",
        position: { x: 40, y: 180 },
        config: { instruction: "Connect to the parking lot entrance camera and stream live video at 3 FPS", streamUrl: "rtsp://camera.local/stream", fps: 3 },
        status: "configured",
      },
      {
        id: "block-sampler",
        type: "frame-sampler",
        label: "Frame Sampler",
        category: "Ingestion",
        position: { x: 260, y: 180 },
        config: { instruction: "Sample every 3rd frame to keep processing manageable while maintaining coverage", sampleRate: 3 },
        status: "configured",
      },
      {
        id: "block-vehicle-det",
        type: "vehicle-detector",
        label: "Vehicle Detector",
        category: "Detection",
        position: { x: 480, y: 120 },
        config: { instruction: "Detect all vehicles — focus on cars, trucks, buses, and motorcycles with at least 50% confidence", model: "yolov8n.pt", confidence: 0.5, classes: "car,truck,bus,motorcycle" },
        status: "configured",
      },
      {
        id: "block-plate-det",
        type: "plate-detector",
        label: "Plate Detector",
        category: "Detection",
        position: { x: 700, y: 120 },
        config: { instruction: "Find number plates on all detected vehicles with 60% minimum confidence", model: "plate_detect.pt", confidence: 0.6 },
        status: "configured",
      },
      {
        id: "block-preprocess",
        type: "plate-preprocessor",
        label: "Plate Preprocessor",
        category: "Preprocessing",
        position: { x: 920, y: 120 },
        config: { instruction: "Clean up and deskew the plate crops, enhance contrast for better OCR accuracy", deskew: true, clahe: true, targetWidth: 200 },
        status: "configured",
      },
      {
        id: "block-ocr",
        type: "paddleocr",
        label: "PaddleOCR",
        category: "OCR",
        position: { x: 1140, y: 120 },
        config: { instruction: "Use high-accuracy OCR to read plate numbers precisely with angle classification", useAngleClassifier: true, language: "en" },
        status: "configured",
      },
      {
        id: "block-validator",
        type: "regex-validator",
        label: "Regex Validator",
        category: "PostProcessing",
        position: { x: 1360, y: 60 },
        config: { instruction: "Validate detected plates against Indian number plate format XX00XX0000", pattern: "^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$" },
        status: "configured",
      },
      {
        id: "block-dedup",
        type: "deduplicator",
        label: "Deduplicator",
        category: "PostProcessing",
        position: { x: 1360, y: 200 },
        config: { instruction: "Remove duplicate plate readings within a 10-second window using perceptual hashing", windowSeconds: 10, hashThreshold: 8 },
        status: "configured",
      },
      {
        id: "block-dispatch",
        type: "dispatcher",
        label: "Dispatcher",
        category: "Output",
        position: { x: 1580, y: 200 },
        config: { instruction: "Save all validated plate readings to the database and notify via Redis pub/sub", sinks: "database,redis" },
        status: "configured",
      },
      {
        id: "block-annotator",
        type: "annotator",
        label: "Annotator",
        category: "Output",
        position: { x: 700, y: 320 },
        config: { instruction: "Draw bounding boxes and plate labels on the live video feed for monitoring", showLabels: true },
        status: "configured",
      },
    ],
    connections: [
      // Stream → Sampler
      { id: "e-input-sampler", source: "block-input", sourceHandle: "frame-out", target: "block-sampler", targetHandle: "frame-in" },
      // Sampler → Vehicle Detector
      { id: "e-sampler-vdet", source: "block-sampler", sourceHandle: "frame-out", target: "block-vehicle-det", targetHandle: "frame-in" },
      // Vehicle Detector → Plate Detector (frame)
      { id: "e-vdet-pdet-frame", source: "block-vehicle-det", sourceHandle: "frame-out", target: "block-plate-det", targetHandle: "frame-in" },
      // Vehicle Detector → Plate Detector (boxes)
      { id: "e-vdet-pdet-boxes", source: "block-vehicle-det", sourceHandle: "boxes-out", target: "block-plate-det", targetHandle: "boxes-in" },
      // Plate Detector → Preprocessor
      { id: "e-pdet-preprocess", source: "block-plate-det", sourceHandle: "plates-out", target: "block-preprocess", targetHandle: "frame-in" },
      // Preprocessor → OCR
      { id: "e-preprocess-ocr", source: "block-preprocess", sourceHandle: "frame-out", target: "block-ocr", targetHandle: "frame-in" },
      // OCR → Regex Validator
      { id: "e-ocr-validator", source: "block-ocr", sourceHandle: "text-out", target: "block-validator", targetHandle: "text-in" },
      // Validator → Deduplicator
      { id: "e-validator-dedup", source: "block-validator", sourceHandle: "text-out", target: "block-dedup", targetHandle: "text-in" },
      // Deduplicator → Dispatcher
      { id: "e-dedup-dispatch", source: "block-dedup", sourceHandle: "text-out", target: "block-dispatch", targetHandle: "text-in" },
      // Vehicle Detector → Annotator (visual branch)
      { id: "e-vdet-annotator-frame", source: "block-vehicle-det", sourceHandle: "frame-out", target: "block-annotator", targetHandle: "frame-in" },
      { id: "e-vdet-annotator-boxes", source: "block-vehicle-det", sourceHandle: "boxes-out", target: "block-annotator", targetHandle: "boxes-in" },
    ],
  },
];
