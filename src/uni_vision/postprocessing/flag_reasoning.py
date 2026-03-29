"""Post-flag reasoning engine — generates structured explanations for flagged detections.

When a detection receives a non-VALID ``ValidationStatus``, this engine
assembles all available evidence (OCR confidence, character corrections,
adjudication results, pipeline telemetry) into a human-readable reasoning
chain explaining *why* the flag was raised and what evidence supports it.

Spec references: §4 S8 post-processing, §8 agentic orchestration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────


class FlagSeverity(str, Enum):
    """How severe the flagged anomaly is."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceType(str, Enum):
    """Category of supporting evidence for a flag."""

    OCR_CONFIDENCE = "ocr_confidence"
    REGEX_MISMATCH = "regex_mismatch"
    CHARACTER_CORRECTION = "character_correction"
    LLM_ADJUDICATION = "llm_adjudication"
    TEMPORAL_PATTERN = "temporal_pattern"
    COMPONENT_DEGRADATION = "component_degradation"
    MULTI_ENGINE_DISAGREEMENT = "multi_engine_disagreement"
    IMAGE_QUALITY = "image_quality"


# ── Data models ───────────────────────────────────────────────────


@dataclass
class EvidenceItem:
    """Single piece of evidence supporting a flag decision."""

    evidence_type: str
    label: str
    description: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    severity: str = "medium"
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlagReasoning:
    """Complete reasoning package for a flagged detection.

    NOTE: No recommendations — the system only raises alerts and provides
    evidence-based reasoning. Operators decide on actions.
    """

    detection_id: str
    flag_type: str  # ValidationStatus value
    severity: str
    headline: str  # one-line summary
    reasoning_chain: List[str]  # ordered reasoning steps
    evidence: List[EvidenceItem]
    alert_count: int = 0  # how many alerts were raised from this evidence
    confidence_score: float = 0.0  # 0-1 how confident we are in this reasoning
    generated_at_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "flag_type": self.flag_type,
            "severity": self.severity,
            "headline": self.headline,
            "reasoning_chain": self.reasoning_chain,
            "evidence": [
                {
                    "evidence_type": e.evidence_type,
                    "label": e.label,
                    "description": e.description,
                    "metric_value": e.metric_value,
                    "threshold": e.threshold,
                    "severity": e.severity,
                    "raw_data": e.raw_data,
                }
                for e in self.evidence
            ],
            "alert_count": self.alert_count,
            "confidence_score": self.confidence_score,
            "generated_at_ms": self.generated_at_ms,
        }


# ── Engine ────────────────────────────────────────────────────────


class FlagReasoningEngine:
    """Generates structured reasoning for flagged (non-VALID) detections.

    This engine operates purely on the data already captured during
    pipeline execution — it does NOT invoke any LLM or external service.
    All reasoning is deterministic and instantaneous.
    """

    # Severity mapping based on flag type
    _SEVERITY_MAP: Dict[str, str] = {
        "low_confidence": FlagSeverity.MEDIUM.value,
        "regex_fail": FlagSeverity.HIGH.value,
        "llm_error": FlagSeverity.MEDIUM.value,
        "fallback": FlagSeverity.LOW.value,
        "parse_fail": FlagSeverity.HIGH.value,
        "unreadable": FlagSeverity.CRITICAL.value,
    }

    def generate(
        self,
        detection_id: str,
        validation_status: str,
        plate_number: str,
        raw_ocr_text: str,
        ocr_confidence: float,
        ocr_engine: str,
        vehicle_class: str,
        camera_id: str,
        char_corrections: Optional[Dict[str, str]] = None,
        adjudication_result: Optional[Dict[str, Any]] = None,
        pipeline_telemetry: Optional[Dict[str, Any]] = None,
    ) -> FlagReasoning:
        """Build a complete reasoning package for a flagged detection."""
        t0 = time.perf_counter()

        severity = self._SEVERITY_MAP.get(validation_status, FlagSeverity.MEDIUM.value)
        evidence: List[EvidenceItem] = []
        reasoning_chain: List[str] = []

        # ── Step 1: Identify the primary flag cause ───────────────
        headline = self._build_headline(validation_status, plate_number, camera_id)
        reasoning_chain.append(
            f"Detection from camera '{camera_id}' flagged with status '{validation_status}'."
        )

        # ── Step 2: OCR confidence analysis ───────────────────────
        conf_evidence, conf_reasoning = self._analyze_confidence(
            ocr_confidence, ocr_engine, validation_status,
        )
        evidence.extend(conf_evidence)
        reasoning_chain.extend(conf_reasoning)

        # ── Step 3: Character correction analysis ─────────────────
        if char_corrections:
            corr_evidence, corr_reasoning = self._analyze_corrections(
                char_corrections, raw_ocr_text, plate_number,
            )
            evidence.extend(corr_evidence)
            reasoning_chain.extend(corr_reasoning)

        # ── Step 4: Regex / format analysis ───────────────────────
        if validation_status in ("regex_fail", "parse_fail"):
            fmt_evidence, fmt_reasoning = self._analyze_format_failure(
                plate_number, raw_ocr_text,
            )
            evidence.extend(fmt_evidence)
            reasoning_chain.extend(fmt_reasoning)

        # ── Step 5: LLM adjudication analysis ─────────────────────
        if adjudication_result:
            adj_evidence, adj_reasoning = self._analyze_adjudication(
                adjudication_result,
            )
            evidence.extend(adj_evidence)
            reasoning_chain.extend(adj_reasoning)
        elif validation_status == "llm_error":
            evidence.append(EvidenceItem(
                evidence_type=EvidenceType.LLM_ADJUDICATION.value,
                label="LLM Adjudication Failed",
                description="The consensus adjudicator (LLM) failed to produce a result. "
                "This may indicate model overload, timeout, or VRAM pressure.",
                severity=FlagSeverity.MEDIUM.value,
            ))
            reasoning_chain.append(
                "LLM adjudication was attempted but failed — the system fell back to deterministic validation."
            )

        # ── Step 6: Pipeline telemetry analysis ───────────────────
        if pipeline_telemetry:
            tel_evidence, tel_reasoning = self._analyze_telemetry(pipeline_telemetry)
            evidence.extend(tel_evidence)
            reasoning_chain.extend(tel_reasoning)

        # ── Step 7: Unreadable plate analysis ─────────────────────
        if validation_status == "unreadable":
            evidence.append(EvidenceItem(
                evidence_type=EvidenceType.IMAGE_QUALITY.value,
                label="Plate Unreadable",
                description="Neither deterministic validation nor LLM adjudication could extract "
                "a valid plate number. The plate image may be occluded, damaged, or too low resolution.",
                severity=FlagSeverity.CRITICAL.value,
            ))
            reasoning_chain.append(
                "All recognition attempts exhausted — plate classified as unreadable."
            )

        # ── Step 8: Count alerts raised from evidence ─────────────
        alert_count = sum(
            1 for e in evidence if e.severity in (FlagSeverity.HIGH.value, FlagSeverity.CRITICAL.value)
        )

        # ── Final confidence in this reasoning ────────────────────
        confidence_score = min(1.0, 0.5 + len(evidence) * 0.1)

        elapsed = (time.perf_counter() - t0) * 1000

        return FlagReasoning(
            detection_id=detection_id,
            flag_type=validation_status,
            severity=severity,
            headline=headline,
            reasoning_chain=reasoning_chain,
            evidence=evidence,
            alert_count=alert_count,
            confidence_score=round(confidence_score, 3),
            generated_at_ms=round(elapsed, 2),
        )

    # ── Headline builders ─────────────────────────────────────────

    @staticmethod
    def _build_headline(status: str, plate: str, camera: str) -> str:
        headlines = {
            "low_confidence": f"Low OCR confidence for plate '{plate}' on {camera}",
            "regex_fail": f"Plate '{plate}' does not match any known format pattern",
            "llm_error": f"LLM adjudication failed for plate '{plate}' — using fallback",
            "fallback": f"Plate '{plate}' resolved via fallback path (adjudicator skipped)",
            "parse_fail": f"OCR output for '{plate}' could not be parsed into a valid plate format",
            "unreadable": f"Plate on {camera} is completely unreadable — all engines failed",
        }
        return headlines.get(status, f"Detection flagged: {status}")

    # ── Evidence analyzers ────────────────────────────────────────

    @staticmethod
    def _analyze_confidence(
        confidence: float, engine: str, status: str,
    ) -> tuple[List[EvidenceItem], List[str]]:
        evidence = []
        reasoning = []

        if confidence < 0.3:
            sev = FlagSeverity.CRITICAL.value
            reasoning.append(
                f"OCR confidence is critically low at {confidence:.1%} (engine: {engine}). "
                "This suggests severe image degradation, occlusion, or incorrect plate detection."
            )
        elif confidence < 0.5:
            sev = FlagSeverity.HIGH.value
            reasoning.append(
                f"OCR confidence is low at {confidence:.1%} (engine: {engine}). "
                "Character recognition is unreliable in this range."
            )
        elif confidence < 0.7:
            sev = FlagSeverity.MEDIUM.value
            reasoning.append(
                f"OCR confidence is moderate at {confidence:.1%} (engine: {engine}). "
                "Some characters may be incorrect."
            )
        else:
            sev = FlagSeverity.LOW.value
            reasoning.append(
                f"OCR confidence is relatively high at {confidence:.1%} (engine: {engine}), "
                "but the flag was triggered by another factor."
            )

        evidence.append(EvidenceItem(
            evidence_type=EvidenceType.OCR_CONFIDENCE.value,
            label="OCR Confidence Score",
            description=f"{engine} reported {confidence:.1%} confidence",
            metric_value=round(confidence, 4),
            threshold=0.7,
            severity=sev,
            raw_data={"engine": engine, "confidence": confidence},
        ))

        return evidence, reasoning

    @staticmethod
    def _analyze_corrections(
        corrections: Dict[str, str], raw_text: str, final_text: str,
    ) -> tuple[List[EvidenceItem], List[str]]:
        evidence = []
        reasoning = []

        n = len(corrections)
        if n > 0:
            pairs = ", ".join(f"'{k}'→'{v}'" for k, v in corrections.items())
            reasoning.append(
                f"{n} character correction(s) applied: {pairs}. "
                f"Raw text '{raw_text}' was corrected to '{final_text}'."
            )
            evidence.append(EvidenceItem(
                evidence_type=EvidenceType.CHARACTER_CORRECTION.value,
                label=f"{n} Character Corrections",
                description=f"Corrections applied: {pairs}",
                metric_value=float(n),
                severity=FlagSeverity.LOW.value if n <= 2 else FlagSeverity.MEDIUM.value,
                raw_data={"corrections": corrections, "raw": raw_text, "final": final_text},
            ))

        return evidence, reasoning

    @staticmethod
    def _analyze_format_failure(
        plate: str, raw_text: str,
    ) -> tuple[List[EvidenceItem], List[str]]:
        evidence = []
        reasoning = []

        reasoning.append(
            f"The plate text '{plate}' (raw: '{raw_text}') does not match any known "
            "locale-specific plate format regex pattern (e.g. Indian format XX##XX####)."
        )
        evidence.append(EvidenceItem(
            evidence_type=EvidenceType.REGEX_MISMATCH.value,
            label="Format Pattern Mismatch",
            description=f"'{plate}' failed all locale regex patterns",
            severity=FlagSeverity.HIGH.value,
            raw_data={"plate_text": plate, "raw_ocr": raw_text},
        ))

        return evidence, reasoning

    @staticmethod
    def _analyze_adjudication(
        adj_result: Dict[str, Any],
    ) -> tuple[List[EvidenceItem], List[str]]:
        evidence = []
        reasoning = []

        adj_plate = adj_result.get("plate_text", "?")
        adj_conf = adj_result.get("confidence", 0.0)
        adj_reasoning_text = adj_result.get("reasoning", "")

        if adj_reasoning_text:
            reasoning.append(
                f"LLM adjudicator analysis: {adj_reasoning_text}"
            )

        evidence.append(EvidenceItem(
            evidence_type=EvidenceType.LLM_ADJUDICATION.value,
            label="LLM Adjudication Result",
            description=f"Adjudicator returned '{adj_plate}' at {adj_conf:.1%} confidence",
            metric_value=round(adj_conf, 4),
            severity=FlagSeverity.MEDIUM.value if adj_conf < 0.7 else FlagSeverity.LOW.value,
            raw_data=adj_result,
        ))

        return evidence, reasoning

    @staticmethod
    def _analyze_telemetry(
        telemetry: Dict[str, Any],
    ) -> tuple[List[EvidenceItem], List[str]]:
        evidence = []
        reasoning = []

        latency = telemetry.get("pipeline_latency_ms")
        vram_pct = telemetry.get("vram_utilisation_pct")
        error_rate = telemetry.get("component_error_rate")

        if latency and latency > 2000:
            reasoning.append(
                f"Pipeline latency is elevated at {latency:.0f}ms (target < 2000ms). "
                "High latency may indicate resource contention affecting OCR accuracy."
            )
            evidence.append(EvidenceItem(
                evidence_type=EvidenceType.COMPONENT_DEGRADATION.value,
                label="Elevated Pipeline Latency",
                description=f"{latency:.0f}ms pipeline latency (target: <2000ms)",
                metric_value=round(latency, 1),
                threshold=2000.0,
                severity=FlagSeverity.MEDIUM.value,
            ))

        if vram_pct and vram_pct > 90:
            reasoning.append(
                f"VRAM utilisation is critical at {vram_pct:.1f}%. "
                "GPU memory pressure may degrade model inference quality."
            )
            evidence.append(EvidenceItem(
                evidence_type=EvidenceType.COMPONENT_DEGRADATION.value,
                label="VRAM Pressure",
                description=f"{vram_pct:.1f}% VRAM utilisation",
                metric_value=round(vram_pct, 1),
                threshold=90.0,
                severity=FlagSeverity.HIGH.value,
            ))

        if error_rate and error_rate > 0.05:
            reasoning.append(
                f"Component error rate is {error_rate:.1%}, exceeding the 5% threshold."
            )
            evidence.append(EvidenceItem(
                evidence_type=EvidenceType.COMPONENT_DEGRADATION.value,
                label="Component Error Rate",
                description=f"{error_rate:.1%} error rate across pipeline components",
                metric_value=round(error_rate, 4),
                threshold=0.05,
                severity=FlagSeverity.HIGH.value,
            ))

        return evidence, reasoning


