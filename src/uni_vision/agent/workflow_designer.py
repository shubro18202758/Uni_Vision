"""Autonomous NL → Pipeline workflow designer.

Takes natural-language input (in any of 16 supported languages),
translates to English via LLM if needed, then uses structured
LLM reasoning to design a complete block-node pipeline workflow.

The designer operates in phases, each reported via an optional
progress callback so the UI can stream real-time status:

  Phase 1  –  Translate (if non-English input)
  Phase 2  –  Analyse requirements
  Phase 3  –  Select & configure blocks
  Phase 4  –  Wire connections
  Phase 5  –  Validate & layout
  Phase 6  –  Return ProjectGraph

Architecture note
-----------------
* The *Gemma 4 E2B* model (AgentLLMClient) handles both multilingual
  translation and the reasoning / block selection via structured JSON.
* All models run via Ollama on the local GPU.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Type alias for the async progress callback:
#   progress_fn(phase: str, message: str) -> None
ProgressFn = Callable[[str, str], Coroutine[Any, Any, None]]


# ── Compact block catalog (encoded for the LLM prompt) ──────────

BLOCK_CATALOG: List[Dict[str, Any]] = [
    # ── Input ──────────────────────────────────────────────────────
    {
        "type": "image-input", "label": "Image Input", "category": "Input",
        "desc": "Load a single image file into the pipeline.",
        "inputs": [], "outputs": [{"id": "frame-out", "type": "frame"}],
    },
    {
        "type": "rtsp-stream", "label": "RTSP Stream", "category": "Input",
        "desc": "Stream frames from an RTSP / IP camera.",
        "inputs": [], "outputs": [{"id": "frame-out", "type": "frame"}],
    },
    {
        "type": "video-file", "label": "Video File", "category": "Input",
        "desc": "Load and decode a local video file frame-by-frame.",
        "inputs": [], "outputs": [{"id": "frame-out", "type": "frame"}],
    },

    # ── Ingestion ──────────────────────────────────────────────────
    {
        "type": "frame-sampler", "label": "Frame Sampler", "category": "Ingestion",
        "desc": "Down-sample frames to a target rate to reduce processing load.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}],
    },
    {
        "type": "roi-crop", "label": "ROI Crop", "category": "Ingestion",
        "desc": "Crop frames to a region-of-interest for focused analysis.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}],
    },

    # ── Preprocessing ──────────────────────────────────────────────
    {
        "type": "grayscale", "label": "Grayscale", "category": "Preprocessing",
        "desc": "Convert frames to grayscale.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}],
    },
    {
        "type": "contrast-enhance", "label": "Contrast Enhance", "category": "Preprocessing",
        "desc": "Enhance contrast via CLAHE adaptive histogram equalisation.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}],
    },
    {
        "type": "denoise", "label": "Denoise", "category": "Preprocessing",
        "desc": "Apply noise reduction (bilateral or NLM) for cleaner frames.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}],
    },
    {
        "type": "resize", "label": "Resize", "category": "Preprocessing",
        "desc": "Resize frames to a target resolution for uniform processing.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}],
    },

    # ── Detection ──────────────────────────────────────────────────
    {
        "type": "yolo-detector", "label": "YOLO Detector", "category": "Detection",
        "desc": "General-purpose object detection (YOLOv8) — detects people, vehicles, animals, objects.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}, {"id": "boxes-out", "type": "bounding_box_list"}],
    },
    {
        "type": "motion-detector", "label": "Motion Detector", "category": "Detection",
        "desc": "Detect motion and moving objects via frame differencing or optical flow.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}, {"id": "boxes-out", "type": "bounding_box_list"}],
    },
    {
        "type": "fire-smoke-detector", "label": "Fire & Smoke Detector", "category": "Detection",
        "desc": "Detect fire, smoke, and thermal anomalies in visual feeds.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}, {"id": "boxes-out", "type": "bounding_box_list"}],
    },
    {
        "type": "crowd-density", "label": "Crowd Density Analyzer", "category": "Detection",
        "desc": "Estimate crowd density and count people in a scene.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}, {"id": "score-out", "type": "score"}],
    },
    {
        "type": "vehicle-detector", "label": "Vehicle Detector", "category": "Detection",
        "desc": "Detect and classify vehicles (car/truck/bus/bike).",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}, {"id": "boxes-out", "type": "bounding_box_list"}],
    },
    {
        "type": "plate-detector", "label": "Plate Detector", "category": "Detection",
        "desc": "Detect license plates within vehicle bounding boxes.",
        "inputs": [{"id": "frame-in", "type": "frame"}, {"id": "boxes-in", "type": "bounding_box_list"}],
        "outputs": [{"id": "plates-out", "type": "frame"}, {"id": "boxes-out", "type": "bounding_box_list"}],
    },

    # ── Analysis ───────────────────────────────────────────────────
    {
        "type": "scene-classifier", "label": "Scene Classifier", "category": "Analysis",
        "desc": "Classify the scene type (indoor/outdoor, industrial, residential, etc.).",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "score-out", "type": "score"}],
    },
    {
        "type": "anomaly-scorer", "label": "Anomaly Scorer", "category": "Analysis",
        "desc": "Score frame anomaly level using learned baselines and deviation patterns.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "score-out", "type": "score"}],
    },
    {
        "type": "pose-estimator", "label": "Pose Estimator", "category": "Analysis",
        "desc": "Estimate human body pose keypoints for posture and activity analysis.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}, {"id": "boxes-out", "type": "bounding_box_list"}],
    },
    {
        "type": "ppe-detector", "label": "PPE / Safety Gear Detector", "category": "Analysis",
        "desc": "Detect personal protective equipment (helmets, vests, goggles, gloves).",
        "inputs": [{"id": "frame-in", "type": "frame"}, {"id": "boxes-in", "type": "bounding_box_list"}],
        "outputs": [{"id": "frame-out", "type": "frame"}, {"id": "score-out", "type": "score"}],
    },
    {
        "type": "zone-intrusion", "label": "Zone Intrusion Detector", "category": "Analysis",
        "desc": "Detect objects or people entering restricted / defined zones.",
        "inputs": [{"id": "frame-in", "type": "frame"}, {"id": "boxes-in", "type": "bounding_box_list"}],
        "outputs": [{"id": "frame-out", "type": "frame"}, {"id": "score-out", "type": "score"}],
    },
    {
        "type": "llm-vision", "label": "LLM Vision Analyzer", "category": "Analysis",
        "desc": "Multipurpose AI vision analysis (Gemma 4) — scene understanding, anomaly reasoning, threat assessment.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "score-out", "type": "score"}, {"id": "text-out", "type": "text"}],
    },

    # ── Tracking ───────────────────────────────────────────────────
    {
        "type": "object-tracker", "label": "Object Tracker", "category": "Tracking",
        "desc": "Track detected objects across frames (SORT / DeepSORT).",
        "inputs": [{"id": "frame-in", "type": "frame"}, {"id": "boxes-in", "type": "bounding_box_list"}],
        "outputs": [{"id": "frame-out", "type": "frame"}, {"id": "boxes-out", "type": "bounding_box_list"}],
    },
    {
        "type": "optical-flow", "label": "Optical Flow", "category": "Tracking",
        "desc": "Compute dense optical flow to visualise motion patterns and velocity.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}],
    },

    # ── OCR / Reading ──────────────────────────────────────────────
    {
        "type": "text-reader", "label": "Text Reader (OCR)", "category": "OCR",
        "desc": "Read text from image regions — signs, labels, plates, documents.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "text-out", "type": "text"}],
    },
    {
        "type": "plate-preprocessor", "label": "Plate Preprocessor", "category": "OCR",
        "desc": "Deskew, enhance contrast, resize plate crops for OCR.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [{"id": "frame-out", "type": "frame"}],
    },

    # ── PostProcessing ─────────────────────────────────────────────
    {
        "type": "deduplicator", "label": "Deduplicator", "category": "PostProcessing",
        "desc": "Filter duplicate detections using perceptual hashing within a time window.",
        "inputs": [{"id": "text-in", "type": "text"}],
        "outputs": [{"id": "text-out", "type": "text"}],
    },
    {
        "type": "threshold-gate", "label": "Threshold Gate", "category": "PostProcessing",
        "desc": "Pass events only when confidence or anomaly score exceeds a threshold.",
        "inputs": [{"id": "score-in", "type": "score"}],
        "outputs": [{"id": "score-out", "type": "score"}],
    },
    {
        "type": "face-anonymizer", "label": "Face Anonymizer", "category": "PostProcessing",
        "desc": "Blur or pixelate detected faces for privacy compliance.",
        "inputs": [{"id": "frame-in", "type": "frame"}, {"id": "boxes-in", "type": "bounding_box_list"}],
        "outputs": [{"id": "frame-out", "type": "frame"}],
    },
    {
        "type": "heatmap-generator", "label": "Heatmap Generator", "category": "PostProcessing",
        "desc": "Generate spatial heatmaps from accumulated detection or motion data.",
        "inputs": [{"id": "frame-in", "type": "frame"}, {"id": "boxes-in", "type": "bounding_box_list"}],
        "outputs": [{"id": "frame-out", "type": "frame"}],
    },

    # ── Output ─────────────────────────────────────────────────────
    {
        "type": "annotator", "label": "Annotator", "category": "Output",
        "desc": "Draw bounding boxes, labels, and overlays onto frames for visualisation.",
        "inputs": [{"id": "frame-in", "type": "frame"}, {"id": "boxes-in", "type": "bounding_box_list"}],
        "outputs": [{"id": "frame-out", "type": "frame"}],
    },
    {
        "type": "dispatcher", "label": "Dispatcher", "category": "Output",
        "desc": "Dispatch validated detections to database, Redis, webhook, or alerting sinks.",
        "inputs": [{"id": "text-in", "type": "text"}],
        "outputs": [],
    },
    {
        "type": "alert-trigger", "label": "Alert Trigger", "category": "Output",
        "desc": "Fire real-time alerts via webhook, email, or push notification on detection events.",
        "inputs": [{"id": "score-in", "type": "score"}],
        "outputs": [],
    },
    {
        "type": "console-logger", "label": "Console Logger", "category": "Output",
        "desc": "Log text output to the console for monitoring and debugging.",
        "inputs": [{"id": "text-in", "type": "text"}],
        "outputs": [],
    },
    {
        "type": "video-recorder", "label": "Video Recorder", "category": "Output",
        "desc": "Record annotated video clips of flagged events for review.",
        "inputs": [{"id": "frame-in", "type": "frame"}],
        "outputs": [],
    },
]

# Category colour map (matches the frontend palette)
CATEGORY_COLOURS: Dict[str, str] = {
    "Input": "#22d3ee",
    "Ingestion": "#06b6d4",
    "Preprocessing": "#fb923c",
    "Detection": "#f43f5e",
    "Analysis": "#a78bfa",
    "Tracking": "#2dd4bf",
    "OCR": "#4ade80",
    "PostProcessing": "#facc15",
    "Output": "#60a5fa",
}

# Horizontal spacing for left-to-right layout
_H_SPACING = 280
_V_SPACING = 160
_START_X = 60
_START_Y = 80

# ── LLM prompt for pipeline design ──────────────────────────────

def _build_compact_catalog() -> str:
    """Build a compact block catalog string for the LLM prompt.

    Only includes type, category, and port IDs to stay within context limits.
    """
    lines: List[str] = []
    for b in BLOCK_CATALOG:
        ins = ",".join(p["id"] for p in b.get("inputs", []))
        outs = ",".join(p["id"] for p in b.get("outputs", []))
        lines.append(f'- {b["type"]} [{b["category"]}] in:[{ins}] out:[{outs}]')
    return "\n".join(lines)


_COMPACT_CATALOG = _build_compact_catalog()

WORKFLOW_DESIGN_SYSTEM_PROMPT = f"""\
You are a computer vision pipeline architect. Design a processing pipeline from the block catalog below.

## Available blocks (type [category] in:[ports] out:[ports])
{_COMPACT_CATALOG}

## Rules
- Connect output ports to matching input ports (frame→frame, bounding_box_list→bounding_box_list, text→text, score→score).
- Order: Input → Ingestion → Preprocessing → Detection → Analysis → Tracking → PostProcessing → Output.
- Use "rtsp-stream" for camera/live/stream, "video-file" for video, "image-input" for images.
- Use "yolo-detector" for people/vehicles/objects, "llm-vision" for general anomaly/scene analysis.
- Include at least one Input, one Detection/Analysis block, and one Output block.

## Output: Return ONLY a JSON object with this exact structure:
{{{{
  "pipeline_name": "descriptive name",
  "blocks": [
    {{{{"type": "rtsp-stream", "label": "Camera Feed", "config": {{}}}}}},
    {{{{"type": "frame-sampler", "label": "Sampler", "config": {{"sampleRate": 5}}}}}},
    {{{{"type": "yolo-detector", "label": "Detector", "config": {{"confidence": 0.5}}}}}},
    {{{{"type": "dispatcher", "label": "Output", "config": {{}}}}}}
  ],
  "connections": [
    {{{{"from_block_index": 0, "from_port": "frame-out", "to_block_index": 1, "to_port": "frame-in"}}}},
    {{{{"from_block_index": 1, "from_port": "frame-out", "to_block_index": 2, "to_port": "frame-in"}}}},
    {{{{"from_block_index": 2, "from_port": "boxes-out", "to_block_index": 3, "to_port": "text-in"}}}}
  ]
}}}}

Design the pipeline for the user's request. Use appropriate blocks and connections. Return ONLY the JSON.
"""


# ── Workflow design result ───────────────────────────────────────

@dataclass
class DesignPhase:
    """One phase of the workflow design process."""
    name: str
    message: str
    elapsed_ms: float = 0.0
    success: bool = True


@dataclass
class WorkflowDesignResult:
    """Complete result of the NL → workflow design process."""
    success: bool
    graph: Optional[Dict[str, Any]] = None  # ProjectGraph JSON
    phases: List[DesignPhase] = field(default_factory=list)
    original_input: str = ""
    english_input: str = ""
    detected_language: str = "en"
    error: Optional[str] = None
    total_elapsed_ms: float = 0.0


# ── Language detection heuristic ─────────────────────────────────

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_TELUGU = re.compile(r"[\u0C00-\u0C7F]")
_TAMIL = re.compile(r"[\u0B80-\u0BFF]")
_KANNADA = re.compile(r"[\u0C80-\u0CFF]")
_MALAYALAM = re.compile(r"[\u0D00-\u0D7F]")
_BENGALI = re.compile(r"[\u0980-\u09FF]")
_GUJARATI = re.compile(r"[\u0A80-\u0AFF]")
_GURMUKHI = re.compile(r"[\u0A00-\u0A7F]")
_ODIA = re.compile(r"[\u0B00-\u0B7F]")
_ARABIC_URDU = re.compile(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]")

_SCRIPT_TO_LANG: List[Tuple[re.Pattern, str]] = [
    (_TELUGU, "te"),
    (_TAMIL, "ta"),
    (_KANNADA, "kn"),
    (_MALAYALAM, "ml"),
    (_BENGALI, "bn"),
    (_GUJARATI, "gu"),
    (_GURMUKHI, "pa"),
    (_ODIA, "or"),
    (_ARABIC_URDU, "ur"),
    (_DEVANAGARI, "hi"),  # Last among Indic — Devanagari covers Hindi/Marathi/Nepali/etc.
]


def detect_language(text: str) -> str:
    """Heuristic language detection based on Unicode script blocks.

    Returns a two-letter language code ('en' if no Indic script detected).
    """
    for pattern, lang in _SCRIPT_TO_LANG:
        if pattern.search(text):
            return lang
    return "en"


# ── Main designer class ─────────────────────────────────────────

class WorkflowDesigner:
    """Autonomous NL → Pipeline workflow designer.

    Parameters
    ----------
    llm_client:
        AgentLLMClient (Gemma 4 E2B) for structured reasoning.
    navarasa_client:
        NavarasaClient (Gemma 7B) for multilingual translation.
        May be None if Navarasa is disabled.
    model_router:
        OllamaModelRouter for VRAM-exclusive model switching.
        May be None if not available.
    """

    def __init__(
        self,
        llm_client: Any,
        navarasa_client: Any = None,
        model_router: Any = None,
    ) -> None:
        self._llm = llm_client
        self._navarasa = navarasa_client
        self._model_router = model_router

    async def design(
        self,
        nl_input: str,
        language: str = "auto",
        *,
        progress_fn: Optional[ProgressFn] = None,
    ) -> WorkflowDesignResult:
        """Design a complete pipeline workflow from natural language.

        Parameters
        ----------
        nl_input:
            User's natural-language description of the desired pipeline.
        language:
            Two-letter language code, or "auto" for auto-detection.
        progress_fn:
            Optional async callback ``(phase, message) -> None`` for
            real-time progress streaming to the UI.

        Returns
        -------
        WorkflowDesignResult with the complete ProjectGraph JSON.
        """
        t0 = time.perf_counter()
        phases: List[DesignPhase] = []
        result = WorkflowDesignResult(
            success=False,
            original_input=nl_input,
            english_input=nl_input,
            phases=phases,
        )

        try:
            # ── Phase 0: Ollama pre-flight health check ──────────
            try:
                import httpx as _httpx

                base_url = str(self._llm._client.base_url).rstrip("/")
                async with _httpx.AsyncClient(timeout=_httpx.Timeout(5.0)) as hc:
                    resp = await hc.get(f"{base_url}/api/tags")
                    resp.raise_for_status()
                logger.debug("workflow_design_ollama_preflight_ok url=%s", base_url)
            except Exception as hc_exc:
                logger.warning("workflow_design_ollama_preflight_failed: %s", hc_exc)
                raise ConnectionError(
                    "Cannot reach Ollama server. Please ensure Ollama is running "
                    "(ollama serve) before designing a workflow."
                ) from hc_exc

            # ── Phase 1: Language detection & translation ─────────
            phase_t = time.perf_counter()
            if progress_fn:
                await progress_fn("detecting_language", "Detecting input language…")

            if language == "auto":
                language = detect_language(nl_input)
            result.detected_language = language

            if language != "en":
                if progress_fn:
                    await progress_fn(
                        "translating",
                        f"Translating from {language} to English…",
                    )
                english = await self._translate(
                    nl_input, source_language=language,
                )
                result.english_input = english
                phases.append(DesignPhase(
                    name="translate",
                    message=f"Translated from {language}: {english[:120]}",
                    elapsed_ms=(time.perf_counter() - phase_t) * 1000,
                ))
            else:
                result.english_input = nl_input
                phases.append(DesignPhase(
                    name="translate",
                    message="Input is already English — skipping translation.",
                    elapsed_ms=(time.perf_counter() - phase_t) * 1000,
                ))

            # ── Phase 1.5: Switch model to primary LLM for design ─────
            if self._model_router:
                try:
                    if progress_fn:
                        await progress_fn(
                            "designing",
                            "Activating Gemma 4 model for pipeline design…",
                        )
                    await self._model_router.activate_primary()
                    logger.info("workflow_design_model_switched_to_primary")
                except Exception as mr_exc:
                    logger.warning(
                        "workflow_design_model_switch_failed: %s", mr_exc,
                    )
                    # Proceed anyway — Ollama may auto-load on request

            # ── Phase 2: LLM-based pipeline design ───────────────
            phase_t = time.perf_counter()
            if progress_fn:
                await progress_fn(
                    "designing",
                    "Analysing requirements and designing pipeline…",
                )

            raw_json = await self._call_llm(
                result.english_input, progress_fn=progress_fn,
            )
            phases.append(DesignPhase(
                name="design",
                message="LLM generated pipeline structure.",
                elapsed_ms=(time.perf_counter() - phase_t) * 1000,
            ))

            # ── Phase 3: Parse & validate LLM output ─────────────
            phase_t = time.perf_counter()
            if progress_fn:
                await progress_fn("validating", "Validating pipeline structure…")

            design_data = self._parse_llm_output(raw_json)
            if design_data is None:
                result.error = "Failed to parse LLM output as valid pipeline JSON."
                phases.append(DesignPhase(
                    name="validate", message=result.error,
                    elapsed_ms=(time.perf_counter() - phase_t) * 1000,
                    success=False,
                ))
                result.total_elapsed_ms = (time.perf_counter() - t0) * 1000
                return result

            phases.append(DesignPhase(
                name="validate",
                message=f"Validated: {len(design_data['blocks'])} blocks, "
                        f"{len(design_data['connections'])} connections.",
                elapsed_ms=(time.perf_counter() - phase_t) * 1000,
            ))

            # ── Phase 4: Build ProjectGraph ──────────────────────
            phase_t = time.perf_counter()
            if progress_fn:
                await progress_fn("building", "Building workflow graph…")

            graph = self._build_graph(design_data, nl_input)
            result.graph = graph
            result.success = True
            phases.append(DesignPhase(
                name="build",
                message=f"Built graph: {graph['project']['name']}",
                elapsed_ms=(time.perf_counter() - phase_t) * 1000,
            ))

            # ── Phase 5: Done ────────────────────────────────────
            if progress_fn:
                await progress_fn("complete", "Workflow design complete!")

        except (ConnectionError, TimeoutError) as exc:
            logger.error("workflow_design_connection_error: %s", exc)
            result.error = str(exc)
            phases.append(DesignPhase(
                name="error", message=str(exc), success=False,
            ))
            if progress_fn:
                await progress_fn("error", str(exc))

        except Exception as exc:
            logger.error("workflow_design_error: %s", exc, exc_info=True)
            error_msg = str(exc)
            # Catch raw httpx connection errors that bypassed _call_llm retry
            if "connect" in error_msg.lower() or "refused" in error_msg.lower():
                error_msg = (
                    "Cannot connect to Ollama. Please ensure the Ollama server "
                    "is running (ollama serve) and the model is available."
                )
            result.error = error_msg
            phases.append(DesignPhase(
                name="error", message=error_msg, success=False,
            ))
            if progress_fn:
                await progress_fn("error", error_msg)

        result.total_elapsed_ms = (time.perf_counter() - t0) * 1000

        return result

    # ── Internal helpers ─────────────────────────────────────────

    async def _translate(
        self, text: str, *, source_language: str = "auto",
    ) -> str:
        """Translate non-English text to English using the primary LLM."""
        import httpx as _httpx

        prompt = (
            f"Translate the following {source_language} text to English. "
            f"Return ONLY the English translation, nothing else.\n\n"
            f"{text}"
        )
        messages = [
            {"role": "system", "content": "You are an expert translator. Translate to English accurately and concisely. Return ONLY the translation."},
            {"role": "user", "content": prompt},
        ]

        cfg = self._llm._cfg
        payload = {
            "model": cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": cfg.num_ctx,
                "temperature": 0.1,
                "num_predict": 1024,
            },
        }

        base_url = str(self._llm._client.base_url).rstrip("/")
        url = f"{base_url}/api/chat"

        async with _httpx.AsyncClient(timeout=_httpx.Timeout(60.0, connect=10.0)) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", text).strip()

    async def _call_llm(
        self,
        english_input: str,
        *,
        progress_fn: Optional[ProgressFn] = None,
    ) -> str:
        """Send the pipeline design prompt to the LLM with streaming.

        Uses Ollama's streaming API so tokens are forwarded in real-time
        to the frontend via ``progress_fn("designing_stream", chunk)``.
        Timeout is 120 s to accommodate large pipeline generation on
        consumer GPUs.
        """
        import asyncio as _asyncio
        import httpx as _httpx

        user_content = english_input.rstrip()

        messages = [
            {"role": "system", "content": WORKFLOW_DESIGN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        cfg = self._llm._cfg
        options: Dict[str, Any] = {
            "num_ctx": max(cfg.num_ctx, 8192),
            "temperature": 0.10,
            "num_predict": 4096,
        }
        if cfg.top_p is not None:
            options["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            options["top_k"] = cfg.top_k
        if cfg.num_gpu is not None:
            options["num_gpu"] = cfg.num_gpu
        if cfg.seed is not None:
            options["seed"] = cfg.seed

        payload: Dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "stream": True,
            "format": "json",
            "options": options,
        }

        base_url = str(self._llm._client.base_url).rstrip("/")
        url = f"{base_url}/api/chat"

        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                accumulated: List[str] = []
                token_count = 0
                buffer: List[str] = []

                async with _httpx.AsyncClient(
                    timeout=_httpx.Timeout(120.0, connect=10.0),
                ) as client:
                    async with client.stream("POST", url, json=payload) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            raise RuntimeError(
                                f"Ollama returned HTTP {resp.status_code}: "
                                f"{body.decode(errors='replace')[:300]}"
                            )

                        async for raw_line in resp.aiter_lines():
                            raw_line = raw_line.strip()
                            if not raw_line:
                                continue
                            try:
                                chunk = json.loads(raw_line)
                            except json.JSONDecodeError:
                                continue

                            msg = chunk.get("message", {})
                            content = msg.get("content", "")

                            if content:
                                accumulated.append(content)
                                buffer.append(content)
                                token_count += 1
                                if progress_fn and token_count % 3 == 0:
                                    await progress_fn(
                                        "designing_stream",
                                        "".join(buffer),
                                    )
                                    buffer.clear()

                            if chunk.get("done"):
                                # Flush remaining buffer
                                if progress_fn and buffer:
                                    await progress_fn("designing_stream", "".join(buffer))
                                break

                return "".join(accumulated)

            except (
                _httpx.ConnectError,
                _httpx.ConnectTimeout,
                ConnectionRefusedError,
            ) as exc:
                last_exc = exc
                logger.warning(
                    "workflow_design_llm_connect_failed attempt=%d error=%s",
                    attempt + 1,
                    exc,
                )
                if attempt == 0:
                    await _asyncio.sleep(1.5)
            except _httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    "workflow_design_llm_timeout attempt=%d error=%s",
                    attempt + 1,
                    exc,
                )
                if attempt == 0:
                    await _asyncio.sleep(1.0)

        # All retries exhausted — raise with user-friendly message
        if isinstance(
            last_exc,
            (_httpx.ConnectError, _httpx.ConnectTimeout, ConnectionRefusedError),
        ):
            raise ConnectionError(
                "Cannot connect to Ollama. Please ensure the Ollama server is "
                "running (ollama serve) and the model is available."
            ) from last_exc
        elif isinstance(last_exc, _httpx.TimeoutException):
            raise TimeoutError(
                "LLM request timed out (120 s). The model may be loading into "
                "VRAM — please wait a moment and try again."
            ) from last_exc
        raise last_exc or RuntimeError("LLM call failed")

    def _parse_llm_output(self, raw: str) -> Optional[Dict[str, Any]]:
        """Extract and validate the JSON object from LLM output."""
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove ```json ... ```
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        # Try to find a JSON object
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from surrounding text using balanced
            # brace matching (greedy regex can grab too much when the
            # model emits reasoning text containing braces).
            data = None
            # Find all top-level { positions and try parsing from each
            for m in re.finditer(r"\{", cleaned):
                depth = 0
                start = m.start()
                for i in range(start, len(cleaned)):
                    if cleaned[i] == "{":
                        depth += 1
                    elif cleaned[i] == "}":
                        depth -= 1
                    if depth == 0:
                        candidate = cleaned[start : i + 1]
                        try:
                            parsed = json.loads(candidate)
                            # Must look like a pipeline definition
                            if isinstance(parsed, dict) and "blocks" in parsed:
                                data = parsed
                                break
                        except json.JSONDecodeError:
                            pass
                        break  # this { didn't work, try next
                if data is not None:
                    break

            if data is None:
                logger.warning("workflow_design_json_parse_failed raw=%s", raw[:500])
                return None

        # Validate required keys
        if "blocks" not in data or not isinstance(data["blocks"], list):
            logger.warning("workflow_design_missing_blocks")
            return None
        if "connections" not in data or not isinstance(data["connections"], list):
            logger.warning("workflow_design_missing_connections")
            return None
        if len(data["blocks"]) == 0:
            logger.warning("workflow_design_empty_blocks")
            return None

        # Validate each block has a known type
        known_types = {b["type"] for b in BLOCK_CATALOG}
        for block in data["blocks"]:
            if block.get("type") not in known_types:
                logger.warning(
                    "workflow_design_unknown_block type=%s",
                    block.get("type"),
                )
                # Remove unknown blocks gracefully
                # (don't fail — just skip)

        # Validate connections reference valid block indices
        n_blocks = len(data["blocks"])
        valid_connections = []
        for conn in data["connections"]:
            from_idx = conn.get("from_block_index", -1)
            to_idx = conn.get("to_block_index", -1)
            if 0 <= from_idx < n_blocks and 0 <= to_idx < n_blocks:
                valid_connections.append(conn)
            else:
                logger.warning(
                    "workflow_design_invalid_connection from=%s to=%s n=%d",
                    from_idx, to_idx, n_blocks,
                )
        data["connections"] = valid_connections

        return data

    def _build_graph(
        self,
        design: Dict[str, Any],
        original_input: str,
    ) -> Dict[str, Any]:
        """Convert the LLM design output into a complete ProjectGraph.

        Assigns UUIDs, resolves port IDs, and computes layout positions.
        """
        pipeline_name = design.get("pipeline_name", "AI-Designed Pipeline")

        # Build block catalogue lookup
        cat_lookup: Dict[str, Dict[str, Any]] = {
            b["type"]: b for b in BLOCK_CATALOG
        }

        # Create blocks with UUIDs and positions
        blocks: List[Dict[str, Any]] = []
        block_ids: List[str] = []  # index → uuid

        # Group blocks by category for layout
        category_order = [
            "Input", "Ingestion", "Preprocessing", "Detection",
            "Analysis", "Tracking", "OCR", "PostProcessing", "Output",
        ]
        category_columns: Dict[str, List[int]] = {c: [] for c in category_order}

        for i, b in enumerate(design["blocks"]):
            btype = b["type"]
            cat_def = cat_lookup.get(btype)
            category = cat_def["category"] if cat_def else "Utility"
            if category in category_columns:
                category_columns[category].append(i)
            else:
                category_columns.setdefault("Output", []).append(i)

        # Compute positions: left-to-right by category, top-to-bottom within
        col = 0
        positions: Dict[int, Dict[str, float]] = {}
        for cat in category_order:
            indices = category_columns.get(cat, [])
            if not indices:
                continue
            for row, idx in enumerate(indices):
                positions[idx] = {
                    "x": _START_X + col * _H_SPACING,
                    "y": _START_Y + row * _V_SPACING,
                }
            col += 1

        for i, b in enumerate(design["blocks"]):
            btype = b["type"]
            cat_def = cat_lookup.get(btype, {})
            bid = f"block-{uuid.uuid4().hex[:12]}"
            block_ids.append(bid)

            config = b.get("config", {})
            label = b.get("label", cat_def.get("label", btype))

            blocks.append({
                "id": bid,
                "type": btype,
                "label": label,
                "category": cat_def.get("category", "Utility"),
                "position": positions.get(i, {"x": _START_X, "y": _START_Y}),
                "config": config,
                "status": "idle",
            })

        # Create connections with UUIDs
        connections: List[Dict[str, Any]] = []
        for conn in design["connections"]:
            from_idx = conn["from_block_index"]
            to_idx = conn["to_block_index"]
            from_port = conn.get("from_port", "")
            to_port = conn.get("to_port", "")
            cid = f"conn-{uuid.uuid4().hex[:12]}"
            connections.append({
                "id": cid,
                "source": block_ids[from_idx],
                "sourceHandle": from_port,
                "target": block_ids[to_idx],
                "targetHandle": to_port,
            })

        return {
            "project": {
                "name": pipeline_name,
                "version": "1.0.0",
            },
            "blocks": blocks,
            "connections": connections,
        }
