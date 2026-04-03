"""LLM-powered vision analyzer for multipurpose anomaly detection.

Uses Gemma 4 E2B — a single multimodal model (Text + Image + Audio)
that processes video frames directly and produces structured anomaly
analysis.  No dual-model pipeline needed.

Gemma 4 E2B specs:
  * 5.1B total params, 2.3B effective (MoE architecture)
  * Vision encoder ~150M params, variable image resolution
  * 128K context window (clamped to 4096 for 8 GB VRAM budget)
  * Q4_K_M quantization, 7.2 GB on disk

Produces structured anomaly analysis including:
  * Scene description and object inventory
  * Anomaly detection with severity grading
  * Chain-of-thought reasoning
  * Risk assessment and impact analysis
  * Actionable recommendations
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Dict, Optional

import cv2
import httpx
import numpy as np
from numpy.typing import NDArray

from uni_vision.contracts.dtos import AnalysisResult

logger = logging.getLogger(__name__)

# ── Unified vision + reasoning prompt (gemma4:e2b — multimodal) ───

VISION_ANALYSIS_PROMPT = """\
You are an expert computer vision analyst specializing in CCTV surveillance frame analysis.
Analyze the provided frame image thoroughly for any anomalies — structural, behavioral,
environmental, safety, or operational.

Your analysis must cover ALL visible elements — people, vehicles, infrastructure, environment,
lighting conditions, weather indicators, and spatial relationships.

OUTPUT FORMAT — respond with ONLY valid JSON (no markdown, no commentary):
{
  "scene_description": "Concise description of what the frame shows",
  "objects_detected": [
    {"label": "object type", "location": "where in frame", "condition": "normal/abnormal/damaged/etc"}
  ],
  "anomaly_detected": true/false,
  "anomalies": [
    {"type": "anomaly category", "description": "what is anomalous", "severity": "low/medium/high/critical", "location": "where in frame"}
  ],
  "chain_of_thought": "Step-by-step reasoning: 1) What I observe... 2) What seems normal... 3) What deviates from expected... 4) Why this is/isn't anomalous...",
  "risk_level": "low/medium/high/critical",
  "risk_analysis": "Assessment of immediate and potential risks based on observed anomalies",
  "impact_analysis": "Analysis of potential consequences — who/what is affected, cascading effects, urgency",
  "confidence": 0.0-1.0,
  "recommendations": ["Actionable recommendation 1", "Recommendation 2"]
}

RULES:
1. Analyze EVERYTHING visible in the image — people, vehicles, infrastructure, environment, lighting, weather indicators.
2. Be domain-agnostic: detect ANY anomaly type (structural, behavioral, environmental, safety, operational).
3. Chain-of-thought must show systematic reasoning, not conclusions alone.
4. Risk and impact analysis must be specific to what is observed.
5. Confidence reflects certainty in your anomaly assessment (0.0 = uncertain, 1.0 = certain).
6. If no anomalies detected, still provide thorough scene analysis with risk_level "low".
7. Output ONLY the JSON object. No preamble, no explanation outside JSON."""


class VisionAnalyzer:
    """Analyzes frames via Gemma 4 E2B (single multimodal model) for anomaly detection.

    Sends video frames with images directly to gemma4:e2b which handles
    both visual encoding and anomaly reasoning in a single pass.

    Parameters
    ----------
    ollama_base_url : str
        Base URL for Ollama API (e.g. ``http://localhost:11434``).
    model : str
        Ollama model tag for the multimodal model (gemma4:e2b).
    timeout_s : int
        Request timeout in seconds.
    num_predict : int
        Max tokens for response generation.
    temperature : float
        Sampling temperature.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "gemma4:e2b",
        timeout_s: int = 120,
        num_predict: int = 1536,
        temperature: float = 1.0,
    ) -> None:
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._num_predict = num_predict
        self._temperature = temperature
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=15.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

    def _encode_frame(self, image: NDArray[np.uint8], max_width: int = 640) -> str:
        """Resize and base64-encode a frame for the LLM."""
        if image is None or image.size == 0:
            raise ValueError("Empty or None frame")
        if image.ndim < 2 or image.ndim > 3:
            raise ValueError(f"Invalid frame dimensions: {image.ndim}")
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            raise ValueError(f"Zero-dimension frame: {w}x{h}")
        if w > max_width:
            scale = max_width / w
            image = cv2.resize(
                image, (max_width, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode frame to JPEG")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    async def analyze_frame(
        self,
        image: NDArray[np.uint8],
        *,
        camera_id: str = "unknown",
        frame_id: str = "unknown",
        timestamp_utc: str = "",
    ) -> AnalysisResult:
        """Analyze a single frame for anomalies using gemma4:e2b (multimodal).

        Sends the frame image directly to the model which handles both
        visual encoding and anomaly reasoning in a single pass.

        Returns an ``AnalysisResult`` with structured findings.
        """
        t0 = time.perf_counter()

        try:
            image_b64 = self._encode_frame(image)
        except (ValueError, cv2.error) as exc:
            logger.error("vision_analyzer_encode_failed camera=%s frame=%s error=%s", camera_id, frame_id, exc)
            return self._fallback_result(camera_id, frame_id, timestamp_utc, f"Frame encoding failed: {exc}")

        # ── Single-model inference (gemma4:e2b — multimodal) ─────
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": VISION_ANALYSIS_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Analyze this CCTV frame from camera '{camera_id}' (frame {frame_id}). "
                        "Produce your structured anomaly analysis JSON."
                    ),
                    "images": [image_b64],
                },
            ],
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "temperature": self._temperature,
                "top_p": 0.95,
                "top_k": 64,
                "num_predict": self._num_predict,
                "num_gpu": -1,
                "num_batch": 256,
                "seed": 42,
            },
        }

        try:
            resp = await self._client.post(f"{self._base_url}/api/chat", json=payload)
        except httpx.TimeoutException:
            logger.error("vision_analyze_timeout camera=%s frame=%s", camera_id, frame_id)
            return self._fallback_result(camera_id, frame_id, timestamp_utc, "Inference timeout")
        except httpx.HTTPError as exc:
            logger.error("vision_analyze_http_error error=%s", exc)
            return self._fallback_result(camera_id, frame_id, timestamp_utc, str(exc))

        if resp.status_code != 200:
            logger.error("vision_analyze_bad_status status=%d", resp.status_code)
            return self._fallback_result(camera_id, frame_id, timestamp_utc, f"HTTP {resp.status_code}")

        content = resp.json().get("message", {}).get("content", "")
        result = self._parse_response(content, camera_id, frame_id, timestamp_utc)

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "vision_analysis_complete camera=%s frame=%s anomaly=%s risk=%s confidence=%.2f total=%.0fms",
            camera_id, frame_id, result.anomaly_detected, result.risk_level,
            result.confidence, total_ms,
        )
        return result

    def _parse_response(
        self,
        content: str,
        camera_id: str,
        frame_id: str,
        timestamp_utc: str,
    ) -> AnalysisResult:
        """Parse LLM JSON response into an AnalysisResult."""
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed content
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    logger.warning("vision_analyzer_parse_failed content=%s", text[:200])
                    return self._fallback_result(camera_id, frame_id, timestamp_utc, "JSON parse error")
            else:
                logger.warning("vision_analyzer_no_json content=%s", text[:200])
                return self._fallback_result(camera_id, frame_id, timestamp_utc, "No JSON in response")

        return AnalysisResult(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp_utc=timestamp_utc,
            scene_description=str(data.get("scene_description", "")),
            objects_detected=data.get("objects_detected", []),
            anomaly_detected=bool(data.get("anomaly_detected", False)),
            anomalies=data.get("anomalies", []),
            chain_of_thought=str(data.get("chain_of_thought", "")),
            risk_level=str(data.get("risk_level", "low")),
            risk_analysis=str(data.get("risk_analysis", "")),
            impact_analysis=str(data.get("impact_analysis", "")),
            confidence=float(data.get("confidence", 0.0)),
            recommendations=data.get("recommendations", []),
        )

    def _fallback_result(
        self,
        camera_id: str,
        frame_id: str,
        timestamp_utc: str,
        error: str,
    ) -> AnalysisResult:
        """Return a degraded result when analysis fails."""
        return AnalysisResult(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp_utc=timestamp_utc,
            scene_description=f"Analysis unavailable: {error}",
            anomaly_detected=False,
            chain_of_thought=f"Analysis failed due to: {error}",
            risk_level="low",
            confidence=0.0,
        )
