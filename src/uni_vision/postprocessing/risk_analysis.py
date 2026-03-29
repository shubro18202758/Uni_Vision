"""Risk analysis engine — multi-dimensional threat assessment for flagged detections.

Alert-only paradigm: evaluates all possible scenarios from the first anomaly
instance through to complete predicted system degradation.  Produces structured
risk profiles with multi-axis scoring suitable for radar charts, timeline
visualisations, scenario projection, and "what-if-ignored" consequence chains
in a live-feed surveillance context.

No recommendations are produced — only alerts and consequence reasoning.

Spec references: §4 S8 post-processing, §8 agentic orchestration.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ScenarioOutcome(str, Enum):
    BEST_CASE = "best_case"
    LIKELY = "likely"
    WORST_CASE = "worst_case"


class AlertPriority(str, Enum):
    INFO = "info"
    WARNING = "warning"
    URGENT = "urgent"
    CRITICAL = "critical"


# ── Data models ───────────────────────────────────────────────────


@dataclass
class RiskDimension:
    """Single axis on the risk radar chart."""

    axis: str
    score: float  # 0-100
    label: str
    description: str
    trend: str = "stable"


@dataclass
class TimelineEvent:
    """Single event in the anomaly progression timeline."""

    timestamp: str
    event_type: str
    title: str
    description: str
    severity: str = "medium"
    metric_value: Optional[float] = None


@dataclass
class AlertItem:
    """A raised alert — no recommendation, just a signal."""

    alert_id: str
    priority: str  # info, warning, urgent, critical
    title: str
    description: str
    source_component: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class ConsequenceStep:
    """A single step in an escalation chain if alert is ignored."""

    step: int
    timeframe: str  # e.g. "0-30 seconds", "1-5 minutes"
    event: str
    description: str
    severity: str  # low / medium / high / critical
    probability: float  # 0-1
    affected_components: List[str] = field(default_factory=list)


@dataclass
class IgnoredAlertConsequence:
    """What happens if this alert is ignored on a live feed."""

    alert_id: str
    alert_title: str
    consequence_chain: List[ConsequenceStep]
    terminal_state: str  # final system state if fully ignored
    total_propagation_time: str  # e.g. "5-15 minutes"
    cascading_failure_risk: float  # 0-1


@dataclass
class ScenarioProjection:
    """Predicted outcome under a specific scenario assumption."""

    scenario: str  # best_case, likely, worst_case
    title: str
    description: str
    probability: float  # 0-1
    impact_score: float  # 0-100
    time_to_resolution: str
    consequences_if_ignored: List[str]
    escalation_severity: str  # how bad does it get if ignored


@dataclass
class AnomalyPattern:
    """Identified pattern in the anomaly data for heatmap / distribution charts."""

    pattern_name: str
    frequency: int
    affected_cameras: List[str]
    time_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]


@dataclass
class ComponentHealthScore:
    """Health score for a pipeline component."""

    component: str
    health_pct: float
    latency_score: float
    reliability_score: float
    accuracy_score: float
    status: str


@dataclass
class RiskAnalysis:
    """Complete risk analysis package — alert-only, no recommendations."""

    detection_id: str
    overall_risk_level: str
    overall_risk_score: float
    risk_dimensions: List[RiskDimension]
    timeline: List[TimelineEvent]
    scenarios: List[ScenarioProjection]
    alerts: List[AlertItem]
    ignored_consequences: List[IgnoredAlertConsequence]
    anomaly_patterns: List[AnomalyPattern]
    component_health: List[ComponentHealthScore]
    summary: str
    generated_at_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "overall_risk_level": self.overall_risk_level,
            "overall_risk_score": self.overall_risk_score,
            "risk_dimensions": [
                {
                    "axis": d.axis,
                    "score": d.score,
                    "label": d.label,
                    "description": d.description,
                    "trend": d.trend,
                }
                for d in self.risk_dimensions
            ],
            "timeline": [
                {
                    "timestamp": t.timestamp,
                    "event_type": t.event_type,
                    "title": t.title,
                    "description": t.description,
                    "severity": t.severity,
                    "metric_value": t.metric_value,
                }
                for t in self.timeline
            ],
            "scenarios": [
                {
                    "scenario": s.scenario,
                    "title": s.title,
                    "description": s.description,
                    "probability": s.probability,
                    "impact_score": s.impact_score,
                    "time_to_resolution": s.time_to_resolution,
                    "consequences_if_ignored": s.consequences_if_ignored,
                    "escalation_severity": s.escalation_severity,
                }
                for s in self.scenarios
            ],
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "priority": a.priority,
                    "title": a.title,
                    "description": a.description,
                    "source_component": a.source_component,
                    "metric_value": a.metric_value,
                    "threshold": a.threshold,
                }
                for a in self.alerts
            ],
            "ignored_consequences": [
                {
                    "alert_id": ic.alert_id,
                    "alert_title": ic.alert_title,
                    "consequence_chain": [
                        {
                            "step": cs.step,
                            "timeframe": cs.timeframe,
                            "event": cs.event,
                            "description": cs.description,
                            "severity": cs.severity,
                            "probability": cs.probability,
                            "affected_components": cs.affected_components,
                        }
                        for cs in ic.consequence_chain
                    ],
                    "terminal_state": ic.terminal_state,
                    "total_propagation_time": ic.total_propagation_time,
                    "cascading_failure_risk": ic.cascading_failure_risk,
                }
                for ic in self.ignored_consequences
            ],
            "anomaly_patterns": [
                {
                    "pattern_name": p.pattern_name,
                    "frequency": p.frequency,
                    "affected_cameras": p.affected_cameras,
                    "time_distribution": p.time_distribution,
                    "severity_distribution": p.severity_distribution,
                }
                for p in self.anomaly_patterns
            ],
            "component_health": [
                {
                    "component": c.component,
                    "health_pct": c.health_pct,
                    "latency_score": c.latency_score,
                    "reliability_score": c.reliability_score,
                    "accuracy_score": c.accuracy_score,
                    "status": c.status,
                }
                for c in self.component_health
            ],
            "summary": self.summary,
            "generated_at_ms": self.generated_at_ms,
        }


# ── Engine ────────────────────────────────────────────────────────


class RiskAnalysisEngine:
    """Computes multi-dimensional risk analysis for flagged detections.

    Alert-only paradigm: produces radar chart data, timeline reconstruction,
    scenario projections with consequence chains, raised alerts, and
    "what-if-ignored" escalation modelling for live-feed context.
    """

    def analyze(
        self,
        detection_id: str,
        validation_status: str,
        ocr_confidence: float,
        ocr_engine: str,
        camera_id: str,
        plate_number: str,
        raw_ocr_text: str,
        vehicle_class: str,
        detected_at: str,
        char_corrections: Optional[Dict[str, str]] = None,
        recent_detections: Optional[List[Dict[str, Any]]] = None,
        pipeline_telemetry: Optional[Dict[str, Any]] = None,
    ) -> RiskAnalysis:
        t0 = time.perf_counter()

        recent = recent_detections or []
        telemetry = pipeline_telemetry or {}

        # 1. Risk dimensions (radar chart)
        dimensions = self._compute_risk_dimensions(
            ocr_confidence, validation_status, camera_id,
            recent, telemetry, char_corrections,
        )

        # 2. Overall risk
        overall_score = self._compute_overall_risk(dimensions)
        overall_level = self._score_to_level(overall_score)

        # 3. Anomaly timeline
        timeline = self._build_timeline(
            detection_id, validation_status, ocr_confidence,
            detected_at, camera_id, recent,
        )

        # 4. Scenario projections (with consequences, NOT recommendations)
        scenarios = self._generate_scenarios(
            validation_status, ocr_confidence, overall_score, recent,
        )

        # 5. Raise alerts
        alerts = self._raise_alerts(
            validation_status, ocr_confidence, telemetry, recent, camera_id,
        )

        # 6. "What if ignored" consequence chains
        ignored_consequences = self._compute_ignored_consequences(
            alerts, validation_status, ocr_confidence, recent, camera_id, telemetry,
        )

        # 7. Pattern analysis
        patterns = self._analyze_patterns(camera_id, recent)

        # 8. Component health
        health = self._assess_component_health(telemetry, recent)

        # 9. Summary
        summary = self._build_summary(
            overall_level, overall_score, validation_status,
            camera_id, plate_number, len(dimensions), len(alerts),
        )

        elapsed = (time.perf_counter() - t0) * 1000

        return RiskAnalysis(
            detection_id=detection_id,
            overall_risk_level=overall_level,
            overall_risk_score=round(overall_score, 1),
            risk_dimensions=dimensions,
            timeline=timeline,
            scenarios=scenarios,
            alerts=alerts,
            ignored_consequences=ignored_consequences,
            anomaly_patterns=patterns,
            component_health=health,
            summary=summary,
            generated_at_ms=round(elapsed, 2),
        )

    # ── Risk Dimensions (Radar Chart Axes) ────────────────────────

    def _compute_risk_dimensions(
        self,
        ocr_confidence: float,
        validation_status: str,
        camera_id: str,
        recent: List[Dict[str, Any]],
        telemetry: Dict[str, Any],
        char_corrections: Optional[Dict[str, str]],
    ) -> List[RiskDimension]:
        dims: List[RiskDimension] = []

        # Axis 1: OCR Reliability
        ocr_risk = max(0, (1.0 - ocr_confidence) * 100)
        dims.append(RiskDimension(
            axis="OCR Reliability",
            score=round(ocr_risk, 1),
            label=f"{ocr_confidence:.0%} confidence",
            description=f"OCR engine confidence inverted to risk: higher = less reliable",
            trend=self._confidence_trend(camera_id, recent),
        ))

        # Axis 2: Format Compliance
        format_risk = 85.0 if validation_status in ("regex_fail", "parse_fail") else 15.0
        dims.append(RiskDimension(
            axis="Format Compliance",
            score=round(format_risk, 1),
            label="Failed" if format_risk > 50 else "Passed",
            description="Whether the plate text matches known locale format patterns",
            trend="stable",
        ))

        # Axis 3: Character Integrity
        n_corr = len(char_corrections) if char_corrections else 0
        char_risk = min(100.0, n_corr * 25.0)  # Each correction adds 25 risk points
        dims.append(RiskDimension(
            axis="Character Integrity",
            score=round(char_risk, 1),
            label=f"{n_corr} corrections",
            description="Risk from character confusion corrections applied",
            trend="stable",
        ))

        # Axis 4: Temporal Consistency
        temporal_risk = self._compute_temporal_risk(camera_id, recent)
        dims.append(RiskDimension(
            axis="Temporal Consistency",
            score=round(temporal_risk, 1),
            label="Anomalous" if temporal_risk > 60 else "Consistent",
            description="Deviation from this camera's historical detection pattern",
            trend="degrading" if temporal_risk > 60 else "stable",
        ))

        # Axis 5: System Health
        sys_risk = self._compute_system_risk(telemetry)
        dims.append(RiskDimension(
            axis="System Health",
            score=round(sys_risk, 1),
            label="Stressed" if sys_risk > 50 else "Healthy",
            description="Pipeline and GPU resource health risk factor",
            trend="degrading" if sys_risk > 50 else "stable",
        ))

        # Axis 6: Detection Reliability
        det_risk = self._compute_detection_reliability(recent, camera_id)
        dims.append(RiskDimension(
            axis="Detection Reliability",
            score=round(det_risk, 1),
            label=f"{100 - det_risk:.0f}% reliable",
            description="Overall detection pipeline reliability for this camera",
            trend=self._reliability_trend(camera_id, recent),
        ))

        # Axis 7: Adjudication Risk
        adj_risk = 70.0 if validation_status in ("llm_error", "unreadable") else 20.0
        dims.append(RiskDimension(
            axis="Adjudication Risk",
            score=round(adj_risk, 1),
            label="Failed" if adj_risk > 50 else "OK",
            description="Risk from LLM adjudication failure or unavailability",
        ))

        # Axis 8: Environmental Factor
        env_risk = 50.0 if validation_status == "unreadable" else 20.0
        dims.append(RiskDimension(
            axis="Environmental Factor",
            score=round(env_risk, 1),
            label="Poor" if env_risk > 40 else "Good",
            description="Estimated environmental impact (lighting, weather, camera angle)",
        ))

        return dims

    # ── Overall Risk Computation ──────────────────────────────────

    @staticmethod
    def _compute_overall_risk(dimensions: List[RiskDimension]) -> float:
        """Weighted geometric mean of all risk dimensions."""
        if not dimensions:
            return 0.0
        # Use weighted approach: OCR and Format have higher weight
        weights = {
            "OCR Reliability": 2.0,
            "Format Compliance": 1.8,
            "Character Integrity": 1.2,
            "Temporal Consistency": 1.5,
            "System Health": 1.3,
            "Detection Reliability": 1.5,
            "Adjudication Risk": 1.0,
            "Environmental Factor": 0.8,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for d in dimensions:
            w = weights.get(d.axis, 1.0)
            weighted_sum += d.score * w
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score < 20:
            return RiskLevel.NEGLIGIBLE.value
        elif score < 40:
            return RiskLevel.LOW.value
        elif score < 60:
            return RiskLevel.MODERATE.value
        elif score < 80:
            return RiskLevel.HIGH.value
        return RiskLevel.CRITICAL.value

    # ── Timeline Construction ─────────────────────────────────────

    @staticmethod
    def _build_timeline(
        detection_id: str,
        validation_status: str,
        confidence: float,
        detected_at: str,
        camera_id: str,
        recent: List[Dict[str, Any]],
    ) -> List[TimelineEvent]:
        events: List[TimelineEvent] = []

        # Find earliest flag in recent history for this camera
        camera_flags = [
            d for d in recent
            if d.get("camera_id") == camera_id
            and d.get("validation_status", "valid") != "valid"
        ]

        if camera_flags:
            earliest = camera_flags[-1] if camera_flags else None
            if earliest:
                events.append(TimelineEvent(
                    timestamp=earliest.get("detected_at_utc", detected_at),
                    event_type="anomaly_start",
                    title="First Anomaly Detected",
                    description=f"Initial flag: {earliest.get('validation_status', 'unknown')} "
                    f"on camera {camera_id}",
                    severity="medium",
                    metric_value=earliest.get("ocr_confidence", 0.0),
                ))

            # Track degradation progression
            for i, flag_det in enumerate(camera_flags[:-1] if len(camera_flags) > 1 else []):
                events.append(TimelineEvent(
                    timestamp=flag_det.get("detected_at_utc", ""),
                    event_type="degradation",
                    title=f"Continued Anomaly #{i + 2}",
                    description=f"Status: {flag_det.get('validation_status')} — "
                    f"confidence {flag_det.get('ocr_confidence', 0):.1%}",
                    severity="medium",
                    metric_value=flag_det.get("ocr_confidence"),
                ))

        # Current flag event
        events.append(TimelineEvent(
            timestamp=detected_at,
            event_type="flag_raised",
            title="Current Detection Flagged",
            description=f"Flag '{validation_status}' raised — confidence {confidence:.1%}",
            severity="high",
            metric_value=confidence,
        ))

        # Predictive events
        if len(camera_flags) >= 3:
            events.append(TimelineEvent(
                timestamp="",
                event_type="prediction",
                title="Pattern Escalation Predicted",
                description=f"Camera {camera_id} shows a recurring anomaly pattern. "
                f"Without intervention, expect continued degradation.",
                severity="high",
            ))

        if confidence < 0.3 and validation_status == "unreadable":
            events.append(TimelineEvent(
                timestamp="",
                event_type="prediction",
                title="Full Detection Loss Imminent",
                description="Current trajectory suggests complete loss of plate recognition "
                "capability for this camera if conditions persist.",
                severity="critical",
            ))

        return events

    # ── Scenario Projections ──────────────────────────────────────

    @staticmethod
    def _generate_scenarios(
        status: str,
        confidence: float,
        overall_risk: float,
        recent: List[Dict[str, Any]],
    ) -> List[ScenarioProjection]:
        scenarios = []

        # Best case
        scenarios.append(ScenarioProjection(
            scenario=ScenarioOutcome.BEST_CASE.value,
            title="Transient Anomaly — Self-Recovery",
            description="The flag is caused by a temporary condition (e.g. glare, momentary "
            "occlusion) that resolves without intervention. Pipeline returns to normal "
            "operation within the next few frames.",
            probability=round(max(0.1, confidence * 0.7), 2),
            impact_score=round(max(5, overall_risk * 0.3), 1),
            time_to_resolution="Immediate (next 1-5 frames)",
            consequences_if_ignored=[
                "No lasting impact — transient condition self-corrects",
                "A few frames may carry invalid reads but the live feed recovers",
                "Minimal data loss in the surveillance record",
            ],
            escalation_severity="low",
        ))

        # Likely case
        likely_prob = 0.5
        if status in ("regex_fail", "parse_fail"):
            likely_desc = (
                "The plate format is genuinely non-standard or the OCR consistently "
                "misreads characters for this plate type. Without intervention the "
                "live feed will continuously produce invalid reads for this plate class."
            )
            likely_resolution = "1-2 hours (configuration update)"
            likely_consequences = [
                "Every vehicle with this plate format will be misread on the live feed",
                "Surveillance coverage gap: entire plate class becomes untrackable",
                "Accumulating invalid data corrupts historical analytics",
                "Cross-camera correlation for this vehicle class breaks down",
            ]
        elif status == "unreadable":
            likely_desc = (
                "Environmental or camera conditions are degrading plate readability. "
                "The live feed will continue producing unreadable frames until "
                "physical conditions change."
            )
            likely_resolution = "Requires physical inspection"
            likely_consequences = [
                "Complete blind spot on this camera — no plates captured",
                "Real-time surveillance integrity compromised for this zone",
                "Vehicles pass undetected through the monitored area",
                "Historical gap in the detection timeline grows every second",
            ]
        elif status == "llm_error":
            likely_desc = (
                "LLM adjudicator is experiencing intermittent failures due to resource "
                "pressure. The live feed degrades to deterministic-only validation "
                "with reduced accuracy."
            )
            likely_resolution = "30-60 minutes (resource stabilisation)"
            likely_consequences = [
                "Adjudication quality drops — more false flags raised",
                "VRAM pressure may cascade to detection models on the live feed",
                "Consensus validation loses its LLM tier — accuracy regresses",
                "Operators face alert fatigue from increased false positives",
            ]
        else:
            likely_desc = (
                "OCR confidence is below threshold for reliable reads on the live feed. "
                "Multiple factors contribute: image quality, model accuracy, environmental."
            )
            likely_resolution = "Varies — depends on root cause"
            likely_consequences = [
                "Ongoing unreliable reads on every frame from this camera",
                "Confidence erosion spreads: subsequent plates may also degrade",
                "Real-time alerting becomes noise-heavy",
                "Downstream integrations receive low-quality data",
            ]

        scenarios.append(ScenarioProjection(
            scenario=ScenarioOutcome.LIKELY.value,
            title="Persistent Issue — Alert Escalation",
            description=likely_desc,
            probability=round(likely_prob, 2),
            impact_score=round(overall_risk * 0.7, 1),
            time_to_resolution=likely_resolution,
            consequences_if_ignored=likely_consequences,
            escalation_severity="high",
        ))

        # Worst case
        worst_prob = round(max(0.05, (1.0 - confidence) * 0.4), 2)
        n_recent_flags = sum(
            1 for d in recent if d.get("validation_status", "valid") != "valid"
        )
        if n_recent_flags > 5:
            worst_prob = min(0.6, worst_prob + 0.15)

        scenarios.append(ScenarioProjection(
            scenario=ScenarioOutcome.WORST_CASE.value,
            title="Cascading Failure — Complete Detection Loss",
            description="The anomaly escalates on the live feed: camera feed degrades further, "
            "VRAM pressure causes model fallback, and the pipeline enters a failure "
            "spiral. All detections from this camera become unreliable.",
            probability=round(worst_prob, 2),
            impact_score=round(min(100, overall_risk * 1.5), 1),
            time_to_resolution="Extended — requires maintenance window",
            consequences_if_ignored=[
                "TOTAL surveillance blind spot — camera produces zero valid detections",
                "VRAM exhaustion cascades to ALL cameras sharing the GPU",
                "Pipeline enters failure spiral: detection → OCR → adjudication all degrade",
                "Live feed data becomes forensically useless for this time window",
                "Cross-camera tracking breaks: vehicles disappear from the surveillance net",
                "Recovery requires full system restart and potential hardware intervention",
            ],
            escalation_severity="critical",
        ))

        return scenarios

    # ── Alert Raising ─────────────────────────────────────────────

    @staticmethod
    def _raise_alerts(
        status: str,
        confidence: float,
        telemetry: Dict[str, Any],
        recent: List[Dict[str, Any]],
        camera_id: str,
    ) -> List[AlertItem]:
        alerts: List[AlertItem] = []
        alert_idx = 0

        # Alert: Primary flag
        flag_priority = {
            "unreadable": AlertPriority.CRITICAL.value,
            "regex_fail": AlertPriority.URGENT.value,
            "parse_fail": AlertPriority.URGENT.value,
            "llm_error": AlertPriority.WARNING.value,
            "low_confidence": AlertPriority.WARNING.value,
            "fallback": AlertPriority.INFO.value,
        }
        alert_idx += 1
        alerts.append(AlertItem(
            alert_id=f"ALR-{alert_idx:03d}",
            priority=flag_priority.get(status, AlertPriority.WARNING.value),
            title=f"Detection Flag: {status.replace('_', ' ').title()}",
            description=f"Live feed frame flagged with status '{status}' on camera {camera_id}",
            source_component="Validation Pipeline",
        ))

        # Alert: Low confidence
        if confidence < 0.5:
            alert_idx += 1
            alerts.append(AlertItem(
                alert_id=f"ALR-{alert_idx:03d}",
                priority=AlertPriority.URGENT.value if confidence < 0.3 else AlertPriority.WARNING.value,
                title="OCR Confidence Below Threshold",
                description=f"OCR confidence at {confidence:.1%} — below minimum reliable threshold",
                source_component="OCR Engine",
                metric_value=round(confidence, 4),
                threshold=0.5,
            ))

        # Alert: VRAM pressure
        vram = telemetry.get("vram_utilisation_pct", 50)
        if vram > 85:
            alert_idx += 1
            alerts.append(AlertItem(
                alert_id=f"ALR-{alert_idx:03d}",
                priority=AlertPriority.CRITICAL.value if vram > 95 else AlertPriority.URGENT.value,
                title="GPU VRAM Pressure",
                description=f"VRAM utilisation at {vram:.1f}% — model inference quality at risk",
                source_component="GPU / VRAM",
                metric_value=round(vram, 1),
                threshold=85.0,
            ))

        # Alert: Pipeline latency
        latency = telemetry.get("pipeline_latency_ms", 500)
        if latency > 2000:
            alert_idx += 1
            alerts.append(AlertItem(
                alert_id=f"ALR-{alert_idx:03d}",
                priority=AlertPriority.URGENT.value if latency > 3000 else AlertPriority.WARNING.value,
                title="Pipeline Latency Elevated",
                description=f"Pipeline latency at {latency:.0f}ms — live feed processing falling behind",
                source_component="Pipeline Orchestrator",
                metric_value=round(latency, 1),
                threshold=2000.0,
            ))

        # Alert: Recurring anomalies on this camera
        cam_flags = [d for d in recent if d.get("camera_id") == camera_id and d.get("validation_status", "valid") != "valid"]
        if len(cam_flags) >= 3:
            alert_idx += 1
            alerts.append(AlertItem(
                alert_id=f"ALR-{alert_idx:03d}",
                priority=AlertPriority.URGENT.value if len(cam_flags) >= 5 else AlertPriority.WARNING.value,
                title="Recurring Anomaly Pattern",
                description=f"{len(cam_flags)} flagged detections from camera {camera_id} in recent window",
                source_component="Pattern Detector",
                metric_value=float(len(cam_flags)),
            ))

        # Alert: Error rate
        err_rate = telemetry.get("component_error_rate", 0.0)
        if err_rate > 0.05:
            alert_idx += 1
            alerts.append(AlertItem(
                alert_id=f"ALR-{alert_idx:03d}",
                priority=AlertPriority.CRITICAL.value if err_rate > 0.15 else AlertPriority.WARNING.value,
                title="Component Error Rate Elevated",
                description=f"Error rate at {err_rate:.1%} — pipeline stability compromised",
                source_component="Pipeline Health Monitor",
                metric_value=round(err_rate, 4),
                threshold=0.05,
            ))

        return alerts

    # ── "What If Ignored" Consequence Chains ──────────────────────

    @staticmethod
    def _compute_ignored_consequences(
        alerts: List[AlertItem],
        status: str,
        confidence: float,
        recent: List[Dict[str, Any]],
        camera_id: str,
        telemetry: Dict[str, Any],
    ) -> List[IgnoredAlertConsequence]:
        consequences: List[IgnoredAlertConsequence] = []

        n_cam_flags = sum(1 for d in recent if d.get("camera_id") == camera_id and d.get("validation_status", "valid") != "valid")

        for alert in alerts:
            chain: List[ConsequenceStep] = []

            if "Detection Flag" in alert.title:
                chain = [
                    ConsequenceStep(
                        step=1, timeframe="0-30 seconds",
                        event="Anomaly persists on live feed",
                        description="The flagged condition continues — every incoming frame from this camera carries the same defect.",
                        severity="medium", probability=0.9,
                        affected_components=["OCR Engine", "Validation Pipeline"],
                    ),
                    ConsequenceStep(
                        step=2, timeframe="30 seconds - 2 minutes",
                        event="Invalid data accumulates in detection log",
                        description="Each frame writes a flagged record. The detection database fills with unreliable entries, polluting analytics.",
                        severity="high", probability=0.85,
                        affected_components=["Storage", "Analytics"],
                    ),
                    ConsequenceStep(
                        step=3, timeframe="2-5 minutes",
                        event="Surveillance coverage gap widens",
                        description="Every vehicle passing this camera during this window goes effectively unrecorded — a growing blind spot in the surveillance net.",
                        severity="high", probability=0.8,
                        affected_components=["Surveillance Coverage", "Cross-Camera Tracking"],
                    ),
                    ConsequenceStep(
                        step=4, timeframe="5-15 minutes",
                        event="Alert fatigue sets in for operators",
                        description="Repeated flags from this camera flood the alert queue, causing operators to start ignoring alerts — including critical ones from other cameras.",
                        severity="critical", probability=0.7,
                        affected_components=["Operator Console", "Alert System"],
                    ),
                    ConsequenceStep(
                        step=5, timeframe="15-30 minutes",
                        event="Cascading data quality degradation",
                        description="Cross-camera correlation algorithms receive corrupted input from this feed, degrading tracking accuracy across the entire surveillance zone.",
                        severity="critical", probability=0.6,
                        affected_components=["Cross-Camera Tracking", "Analytics", "Forensics"],
                    ),
                ]
                terminal = "Complete surveillance blackout for this camera zone; historical data for this window is forensically compromised"
                prop_time = "15-30 minutes"
                cascade_risk = 0.7 if n_cam_flags >= 3 else 0.4

            elif "OCR Confidence" in alert.title:
                chain = [
                    ConsequenceStep(
                        step=1, timeframe="Immediate",
                        event="Plate misreads continue",
                        description=f"At {confidence:.0%} confidence, approximately {(1-confidence)*100:.0f}% of characters may be wrong on every frame.",
                        severity="high", probability=0.95,
                        affected_components=["OCR Engine"],
                    ),
                    ConsequenceStep(
                        step=2, timeframe="1-3 minutes",
                        event="False identifications enter the system",
                        description="Misread plates create phantom vehicle records — real vehicles mapped to wrong identities.",
                        severity="high", probability=0.8,
                        affected_components=["Detection Log", "Vehicle Tracking"],
                    ),
                    ConsequenceStep(
                        step=3, timeframe="3-10 minutes",
                        event="Cross-reference integrity breaks",
                        description="Plate lookups against watch-lists produce false matches or miss real matches, undermining the purpose of the surveillance.",
                        severity="critical", probability=0.65,
                        affected_components=["Watch-List Matching", "Alert System"],
                    ),
                ]
                terminal = "Detection pipeline output is untrustworthy; manual review required for all reads from this camera"
                prop_time = "3-10 minutes"
                cascade_risk = 0.55

            elif "VRAM" in alert.title:
                vram = telemetry.get("vram_utilisation_pct", 90)
                chain = [
                    ConsequenceStep(
                        step=1, timeframe="0-60 seconds",
                        event="Model inference quality degrades",
                        description=f"At {vram:.0f}% VRAM, models compete for memory — inference latency increases and accuracy drops.",
                        severity="high", probability=0.9,
                        affected_components=["Vehicle Detector", "OCR Engine", "LLM Adjudicator"],
                    ),
                    ConsequenceStep(
                        step=2, timeframe="1-3 minutes",
                        event="Model eviction or OOM crash",
                        description="GPU runs out of memory — either models get evicted from VRAM or the CUDA process crashes entirely.",
                        severity="critical", probability=0.7,
                        affected_components=["GPU / VRAM", "All Models"],
                    ),
                    ConsequenceStep(
                        step=3, timeframe="3-10 minutes",
                        event="Full pipeline halt",
                        description="With no models loaded, the entire detection pipeline stops. ALL cameras go dark simultaneously — not just the flagged one.",
                        severity="critical", probability=0.6,
                        affected_components=["Entire Pipeline", "All Cameras"],
                    ),
                ]
                terminal = "Total system outage — all camera feeds unprocessed; requires GPU restart"
                prop_time = "3-10 minutes"
                cascade_risk = 0.85

            elif "Pipeline Latency" in alert.title:
                chain = [
                    ConsequenceStep(
                        step=1, timeframe="0-30 seconds",
                        event="Frame processing falls behind live feed",
                        description="The pipeline can't keep up with incoming frames — a growing backlog forms.",
                        severity="medium", probability=0.9,
                        affected_components=["Pipeline Orchestrator"],
                    ),
                    ConsequenceStep(
                        step=2, timeframe="30 seconds - 2 minutes",
                        event="Frame dropping begins",
                        description="To prevent memory overflow, the pipeline starts dropping frames — vehicles pass undetected.",
                        severity="high", probability=0.8,
                        affected_components=["Frame Ingestion", "Detection Coverage"],
                    ),
                    ConsequenceStep(
                        step=3, timeframe="2-5 minutes",
                        event="Real-time guarantee violated",
                        description="Detections arrive minutes after the vehicle has already passed — surveillance becomes historical, not real-time.",
                        severity="critical", probability=0.7,
                        affected_components=["Real-Time Processing", "Alert Timeliness"],
                    ),
                ]
                terminal = "Pipeline becomes a batch processor; real-time surveillance capability lost"
                prop_time = "2-5 minutes"
                cascade_risk = 0.6

            elif "Recurring Anomaly" in alert.title:
                chain = [
                    ConsequenceStep(
                        step=1, timeframe="Already in progress",
                        event="Pattern is established and worsening",
                        description=f"{n_cam_flags} anomalies detected — this is not transient, it's a systematic issue.",
                        severity="high", probability=0.95,
                        affected_components=["Camera Feed", "Detection Pipeline"],
                    ),
                    ConsequenceStep(
                        step=2, timeframe="5-15 minutes",
                        event="Permanent degradation of this camera's output",
                        description="Without addressing the root cause, this camera becomes a permanent liability in the surveillance network.",
                        severity="critical", probability=0.8,
                        affected_components=["Camera Feed", "Surveillance Zone"],
                    ),
                ]
                terminal = "Camera effectively offline for surveillance purposes despite appearing connected"
                prop_time = "5-15 minutes"
                cascade_risk = 0.7

            elif "Error Rate" in alert.title:
                chain = [
                    ConsequenceStep(
                        step=1, timeframe="0-60 seconds",
                        event="Errors compound across components",
                        description="When one component errors frequently, downstream components receive garbage input and produce garbage output.",
                        severity="high", probability=0.85,
                        affected_components=["All Pipeline Components"],
                    ),
                    ConsequenceStep(
                        step=2, timeframe="1-5 minutes",
                        event="Error cascades trigger circuit breakers",
                        description="Safety mechanisms start disabling failing components, reducing pipeline capability.",
                        severity="critical", probability=0.65,
                        affected_components=["Pipeline Orchestrator", "Health Monitor"],
                    ),
                ]
                terminal = "Pipeline in degraded mode with multiple components disabled"
                prop_time = "1-5 minutes"
                cascade_risk = 0.6

            else:
                continue

            consequences.append(IgnoredAlertConsequence(
                alert_id=alert.alert_id,
                alert_title=alert.title,
                consequence_chain=chain,
                terminal_state=terminal,
                total_propagation_time=prop_time,
                cascading_failure_risk=cascade_risk,
            ))

        return consequences

    # ── Pattern Analysis ──────────────────────────────────────────

    @staticmethod
    def _analyze_patterns(
        camera_id: str,
        recent: List[Dict[str, Any]],
    ) -> List[AnomalyPattern]:
        patterns: List[AnomalyPattern] = []

        # Group flags by status
        status_groups: Dict[str, List[Dict[str, Any]]] = {}
        for d in recent:
            st = d.get("validation_status", "valid")
            if st != "valid":
                status_groups.setdefault(st, []).append(d)

        # Group by camera
        camera_groups: Dict[str, int] = {}
        for d in recent:
            if d.get("validation_status", "valid") != "valid":
                cam = d.get("camera_id", "unknown")
                camera_groups[cam] = camera_groups.get(cam, 0) + 1

        for status, dets in status_groups.items():
            # Time distribution (by hour bucket)
            time_dist: Dict[str, int] = {}
            for d in dets:
                ts = d.get("detected_at_utc", "")
                if ts and len(ts) >= 13:
                    hour = ts[11:13] + ":00"
                    time_dist[hour] = time_dist.get(hour, 0) + 1

            # Severity distribution
            sev_dist: Dict[str, int] = {}
            for d in dets:
                conf = d.get("ocr_confidence", 0.5)
                if conf < 0.3:
                    sev = "critical"
                elif conf < 0.5:
                    sev = "high"
                elif conf < 0.7:
                    sev = "medium"
                else:
                    sev = "low"
                sev_dist[sev] = sev_dist.get(sev, 0) + 1

            affected = list({d.get("camera_id", "unknown") for d in dets})

            patterns.append(AnomalyPattern(
                pattern_name=f"{status} anomalies",
                frequency=len(dets),
                affected_cameras=affected,
                time_distribution=time_dist,
                severity_distribution=sev_dist,
            ))

        return patterns

    # ── Component Health Assessment ───────────────────────────────

    @staticmethod
    def _assess_component_health(
        telemetry: Dict[str, Any],
        recent: List[Dict[str, Any]],
    ) -> List[ComponentHealthScore]:
        components: List[ComponentHealthScore] = []

        # Vehicle Detector
        det_reliability = 95.0
        n_total = len(recent) if recent else 1
        n_flags = sum(1 for d in recent if d.get("validation_status", "valid") != "valid")
        if n_total > 0:
            det_reliability = max(20.0, (1.0 - n_flags / n_total) * 100)
        components.append(ComponentHealthScore(
            component="Vehicle Detector",
            health_pct=round(det_reliability, 1),
            latency_score=round(min(100, telemetry.get("detection_latency_ms", 50) / 200 * 100), 1),
            reliability_score=round(det_reliability, 1),
            accuracy_score=round(det_reliability * 0.95, 1),
            status="healthy" if det_reliability > 80 else "degraded" if det_reliability > 50 else "failing",
        ))

        # OCR Engine
        avg_conf = 0.7
        if recent:
            confs = [d.get("ocr_confidence", 0.7) for d in recent]
            avg_conf = sum(confs) / len(confs) if confs else 0.7
        ocr_health = avg_conf * 100
        components.append(ComponentHealthScore(
            component="OCR Engine",
            health_pct=round(ocr_health, 1),
            latency_score=round(min(100, telemetry.get("ocr_latency_ms", 100) / 500 * 100), 1),
            reliability_score=round(ocr_health * 0.9, 1),
            accuracy_score=round(avg_conf * 100, 1),
            status="healthy" if ocr_health > 70 else "degraded" if ocr_health > 40 else "failing",
        ))

        # Post-processor
        pp_health = 90.0
        n_regex_fail = sum(1 for d in recent if d.get("validation_status") == "regex_fail")
        if n_total > 0:
            pp_health = max(30.0, (1.0 - n_regex_fail / n_total) * 100)
        components.append(ComponentHealthScore(
            component="Post-Processor",
            health_pct=round(pp_health, 1),
            latency_score=round(min(100, telemetry.get("postprocess_latency_ms", 20) / 100 * 100), 1),
            reliability_score=round(pp_health, 1),
            accuracy_score=round(pp_health * 0.95, 1),
            status="healthy" if pp_health > 80 else "degraded" if pp_health > 50 else "failing",
        ))

        # LLM Adjudicator
        n_llm_err = sum(1 for d in recent if d.get("validation_status") == "llm_error")
        llm_health = max(20.0, (1.0 - n_llm_err / max(n_total, 1)) * 100)
        vram_pct = telemetry.get("vram_utilisation_pct", 50)
        if vram_pct > 90:
            llm_health = min(llm_health, 40.0)
        components.append(ComponentHealthScore(
            component="LLM Adjudicator",
            health_pct=round(llm_health, 1),
            latency_score=round(min(100, telemetry.get("adjudication_latency_ms", 500) / 3000 * 100), 1),
            reliability_score=round(llm_health * 0.85, 1),
            accuracy_score=round(llm_health * 0.9, 1),
            status="healthy" if llm_health > 70 else "degraded" if llm_health > 40 else "failing",
        ))

        # GPU / VRAM
        gpu_health = max(10.0, 100.0 - vram_pct)
        components.append(ComponentHealthScore(
            component="GPU / VRAM",
            health_pct=round(gpu_health, 1),
            latency_score=0.0,
            reliability_score=round(gpu_health, 1),
            accuracy_score=0.0,
            status="healthy" if gpu_health > 40 else "degraded" if gpu_health > 15 else "failing",
        ))

        return components

    # ── Helper methods ────────────────────────────────────────────

    @staticmethod
    def _confidence_trend(camera_id: str, recent: List[Dict[str, Any]]) -> str:
        """Determine if confidence is improving, stable, or degrading."""
        cam_dets = [d for d in recent if d.get("camera_id") == camera_id]
        if len(cam_dets) < 3:
            return "stable"
        confs = [d.get("ocr_confidence", 0.5) for d in cam_dets[-10:]]
        if len(confs) < 3:
            return "stable"
        first_half = sum(confs[: len(confs) // 2]) / (len(confs) // 2)
        second_half = sum(confs[len(confs) // 2 :]) / (len(confs) - len(confs) // 2)
        diff = second_half - first_half
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "degrading"
        return "stable"

    @staticmethod
    def _reliability_trend(camera_id: str, recent: List[Dict[str, Any]]) -> str:
        cam_dets = [d for d in recent if d.get("camera_id") == camera_id]
        if len(cam_dets) < 4:
            return "stable"
        first_half = cam_dets[: len(cam_dets) // 2]
        second_half = cam_dets[len(cam_dets) // 2 :]
        r1 = sum(1 for d in first_half if d.get("validation_status") == "valid") / max(len(first_half), 1)
        r2 = sum(1 for d in second_half if d.get("validation_status") == "valid") / max(len(second_half), 1)
        diff = r2 - r1
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "degrading"
        return "stable"

    @staticmethod
    def _compute_temporal_risk(camera_id: str, recent: List[Dict[str, Any]]) -> float:
        """Higher risk if this camera has many recent flags."""
        cam_dets = [d for d in recent if d.get("camera_id") == camera_id]
        if not cam_dets:
            return 30.0  # Unknown — moderate default
        n_flags = sum(1 for d in cam_dets if d.get("validation_status", "valid") != "valid")
        ratio = n_flags / len(cam_dets)
        return min(100.0, ratio * 120)  # Scale to emphasize high ratios

    @staticmethod
    def _compute_system_risk(telemetry: Dict[str, Any]) -> float:
        """Compute system health risk from telemetry."""
        risk = 10.0  # baseline
        vram = telemetry.get("vram_utilisation_pct", 50)
        if vram > 90:
            risk += 40
        elif vram > 75:
            risk += 20
        latency = telemetry.get("pipeline_latency_ms", 500)
        if latency > 3000:
            risk += 30
        elif latency > 2000:
            risk += 15
        error_rate = telemetry.get("component_error_rate", 0.01)
        if error_rate > 0.1:
            risk += 25
        elif error_rate > 0.05:
            risk += 10
        return min(100.0, risk)

    @staticmethod
    def _compute_detection_reliability(recent: List[Dict[str, Any]], camera_id: str) -> float:
        """Risk score for detection reliability (inverted reliability %)."""
        cam = [d for d in recent if d.get("camera_id") == camera_id]
        if not cam:
            return 25.0
        valid = sum(1 for d in cam if d.get("validation_status") == "valid")
        reliability = valid / len(cam)
        return max(0, (1.0 - reliability) * 100)

    @staticmethod
    def _build_summary(
        level: str, score: float, status: str,
        camera_id: str, plate: str, n_dims: int,
        n_alerts: int = 0,
    ) -> str:
        alert_text = f" {n_alerts} alert(s) raised." if n_alerts else ""
        return (
            f"Risk assessment for detection of plate '{plate}' from camera {camera_id}: "
            f"Overall risk is {level.upper()} (score: {score:.1f}/100) based on {n_dims} "
            f"analysis dimensions. Primary flag: {status}.{alert_text} "
            f"{'CRITICAL — review alert consequences immediately.' if score > 70 else 'ELEVATED — alert escalation possible if ignored.' if score > 40 else 'Monitor live feed for recurrence.'}"
        )
