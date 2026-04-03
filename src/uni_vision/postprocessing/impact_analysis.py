"""Exhaustive Impact Analysis engine for post-flagging assessment.

After a frame is flagged and risk analysis is complete, this engine performs
deep impact analysis covering every dimension of how the anomaly affects the
live surveillance feed:

- **Operational Impact**: throughput, latency, frame processing capacity
- **Surveillance Integrity**: coverage gaps, blind spots, tracking breaks
- **Data Quality**: corruption propagation, forensic reliability
- **Cascading Failures**: component dependency chains, domino effects
- **Temporal Propagation**: how damage grows over time on a live feed
- **Resource Utilisation**: GPU/VRAM/CPU pressure and budgets
- **Financial / Compliance**: SLA violations, liability exposure

Produces rich data structures for treemaps, funnel charts, heatmaps,
stacked area charts, gauge clusters, Sankey diagrams, and correlation matrices.
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


class ImpactDomain(str, Enum):
    OPERATIONAL = "operational"
    SURVEILLANCE = "surveillance"
    DATA_QUALITY = "data_quality"
    CASCADING = "cascading"
    TEMPORAL = "temporal"
    RESOURCE = "resource"
    COMPLIANCE = "compliance"


class SeverityTier(str, Enum):
    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"


# ── Data Models ───────────────────────────────────────────────────


@dataclass
class ImpactDimension:
    """Single dimension of impact assessment."""

    domain: str
    title: str
    score: float  # 0-100
    severity: str
    description: str
    affected_components: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalImpactPoint:
    """Impact at a specific point in time on the live feed."""

    time_offset: str  # e.g. "0s", "30s", "2min", "10min"
    time_seconds: int  # numeric for charting
    cumulative_frames_lost: int
    surveillance_coverage_pct: float  # 0-100
    data_quality_pct: float  # 0-100
    system_stability_pct: float  # 0-100
    detection_accuracy_pct: float  # 0-100
    operator_trust_pct: float  # 0-100
    description: str


@dataclass
class CascadeNode:
    """Single node in a cascading failure chain."""

    component: str
    failure_mode: str
    time_to_failure: str
    probability: float
    downstream: List[str]  # components affected
    severity: str


@dataclass
class ResourceImpact:
    """Impact on a specific system resource."""

    resource: str
    current_usage_pct: float
    projected_peak_pct: float
    headroom_pct: float
    time_to_exhaustion: str
    risk_level: str


@dataclass
class CoverageGap:
    """A gap in surveillance coverage caused by the anomaly."""

    camera_id: str
    gap_type: str  # "total_blind", "degraded", "intermittent"
    start_offset: str
    duration_estimate: str
    vehicles_missed_estimate: int
    zone_affected: str
    severity: str


@dataclass
class DataCorruptionVector:
    """A pathway through which corrupted data propagates."""

    source: str
    destination: str
    data_type: str  # "plate_reads", "tracking", "analytics", "alerts"
    corruption_type: str  # "misidentification", "missing", "delayed", "garbled"
    records_affected_estimate: int
    forensic_impact: str  # "recoverable", "partial_loss", "irrecoverable"
    severity: str


@dataclass
class ComplianceImpact:
    """Impact on regulatory / SLA compliance."""

    regulation: str
    requirement: str
    current_status: str  # "compliant", "at_risk", "violated"
    time_to_violation: str
    liability_level: str  # "none", "low", "medium", "high"
    description: str


@dataclass
class FunnelStage:
    """Stage in the detection processing funnel showing drop-off."""

    stage: str
    total_frames: int
    successful: int
    failed: int
    drop_rate_pct: float
    bottleneck: bool


@dataclass
class HeatmapCell:
    """Single cell in the component x time heatmap."""

    component: str
    time_bucket: str
    health_pct: float
    anomaly_count: int


@dataclass
class CorrelationPair:
    """Correlation between two impact metrics."""

    metric_a: str
    metric_b: str
    correlation: float  # -1 to 1
    relationship: str  # "strong_positive", "moderate_positive", etc.


@dataclass
class ImpactAnalysis:
    """Complete exhaustive impact analysis after a flagged detection."""

    detection_id: str
    overall_impact_score: float  # 0-100
    overall_severity: str
    impact_dimensions: List[ImpactDimension]
    temporal_propagation: List[TemporalImpactPoint]
    cascade_chain: List[CascadeNode]
    resource_impacts: List[ResourceImpact]
    coverage_gaps: List[CoverageGap]
    data_corruption_vectors: List[DataCorruptionVector]
    compliance_impacts: List[ComplianceImpact]
    processing_funnel: List[FunnelStage]
    component_heatmap: List[HeatmapCell]
    correlations: List[CorrelationPair]
    summary: str
    analysis_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "overall_impact_score": self.overall_impact_score,
            "overall_severity": self.overall_severity,
            "impact_dimensions": [
                {
                    "domain": d.domain,
                    "title": d.title,
                    "score": d.score,
                    "severity": d.severity,
                    "description": d.description,
                    "affected_components": d.affected_components,
                    "metrics": d.metrics,
                }
                for d in self.impact_dimensions
            ],
            "temporal_propagation": [
                {
                    "time_offset": t.time_offset,
                    "time_seconds": t.time_seconds,
                    "cumulative_frames_lost": t.cumulative_frames_lost,
                    "surveillance_coverage_pct": t.surveillance_coverage_pct,
                    "data_quality_pct": t.data_quality_pct,
                    "system_stability_pct": t.system_stability_pct,
                    "detection_accuracy_pct": t.detection_accuracy_pct,
                    "operator_trust_pct": t.operator_trust_pct,
                    "description": t.description,
                }
                for t in self.temporal_propagation
            ],
            "cascade_chain": [
                {
                    "component": c.component,
                    "failure_mode": c.failure_mode,
                    "time_to_failure": c.time_to_failure,
                    "probability": c.probability,
                    "downstream": c.downstream,
                    "severity": c.severity,
                }
                for c in self.cascade_chain
            ],
            "resource_impacts": [
                {
                    "resource": r.resource,
                    "current_usage_pct": r.current_usage_pct,
                    "projected_peak_pct": r.projected_peak_pct,
                    "headroom_pct": r.headroom_pct,
                    "time_to_exhaustion": r.time_to_exhaustion,
                    "risk_level": r.risk_level,
                }
                for r in self.resource_impacts
            ],
            "coverage_gaps": [
                {
                    "camera_id": g.camera_id,
                    "gap_type": g.gap_type,
                    "start_offset": g.start_offset,
                    "duration_estimate": g.duration_estimate,
                    "vehicles_missed_estimate": g.vehicles_missed_estimate,
                    "zone_affected": g.zone_affected,
                    "severity": g.severity,
                }
                for g in self.coverage_gaps
            ],
            "data_corruption_vectors": [
                {
                    "source": v.source,
                    "destination": v.destination,
                    "data_type": v.data_type,
                    "corruption_type": v.corruption_type,
                    "records_affected_estimate": v.records_affected_estimate,
                    "forensic_impact": v.forensic_impact,
                    "severity": v.severity,
                }
                for v in self.data_corruption_vectors
            ],
            "compliance_impacts": [
                {
                    "regulation": c.regulation,
                    "requirement": c.requirement,
                    "current_status": c.current_status,
                    "time_to_violation": c.time_to_violation,
                    "liability_level": c.liability_level,
                    "description": c.description,
                }
                for c in self.compliance_impacts
            ],
            "processing_funnel": [
                {
                    "stage": f.stage,
                    "total_frames": f.total_frames,
                    "successful": f.successful,
                    "failed": f.failed,
                    "drop_rate_pct": f.drop_rate_pct,
                    "bottleneck": f.bottleneck,
                }
                for f in self.processing_funnel
            ],
            "component_heatmap": [
                {
                    "component": h.component,
                    "time_bucket": h.time_bucket,
                    "health_pct": h.health_pct,
                    "anomaly_count": h.anomaly_count,
                }
                for h in self.component_heatmap
            ],
            "correlations": [
                {
                    "metric_a": c.metric_a,
                    "metric_b": c.metric_b,
                    "correlation": c.correlation,
                    "relationship": c.relationship,
                }
                for c in self.correlations
            ],
            "summary": self.summary,
            "analysis_time_ms": self.analysis_time_ms,
        }


# ── Engine ────────────────────────────────────────────────────────


class ImpactAnalysisEngine:
    """Exhaustive impact analysis engine for post-flagging assessment.

    Operates deterministically on detection data and telemetry.
    No external service calls — all analysis is computed locally.
    """

    def analyze(
        self,
        detection_id: str,
        validation_status: str,
        plate_number: str,
        ocr_confidence: float,
        camera_id: str,
        telemetry: Dict[str, Any],
        recent_detections: List[Dict[str, Any]],
        anomaly_type: str = "",
        anomaly_severity: str = "",
        anomaly_description: str = "",
    ) -> ImpactAnalysis:
        t0 = time.perf_counter()

        # Step 1: Impact dimensions (7 domains)
        dimensions = self._compute_dimensions(
            validation_status, ocr_confidence, camera_id, telemetry, recent_detections,
            anomaly_type, anomaly_severity, anomaly_description,
        )

        # Step 2: Overall impact score
        overall_score = sum(d.score for d in dimensions) / max(len(dimensions), 1)
        overall_severity = self._score_to_severity(overall_score)

        # Step 3: Temporal propagation on live feed
        temporal = self._compute_temporal_propagation(
            validation_status, ocr_confidence, telemetry, recent_detections, camera_id,
        )

        # Step 4: Cascading failure chain
        cascade = self._compute_cascade_chain(
            validation_status, ocr_confidence, telemetry,
        )

        # Step 5: Resource impacts
        resources = self._compute_resource_impacts(telemetry)

        # Step 6: Coverage gaps
        gaps = self._compute_coverage_gaps(
            camera_id, validation_status, ocr_confidence, recent_detections,
        )

        # Step 7: Data corruption vectors
        corruption = self._compute_data_corruption(
            validation_status, ocr_confidence, recent_detections, camera_id,
        )

        # Step 8: Compliance impacts
        compliance = self._compute_compliance_impacts(
            validation_status, overall_score, recent_detections, camera_id,
        )

        # Step 9: Processing funnel
        funnel = self._compute_processing_funnel(recent_detections)

        # Step 10: Component health heatmap
        heatmap = self._compute_heatmap(recent_detections)

        # Step 11: Metric correlations
        correlations = self._compute_correlations(recent_detections, telemetry)

        # Step 12: Summary
        summary = self._build_summary(
            overall_score, overall_severity, validation_status,
            camera_id, plate_number, len(dimensions),
            len(gaps), len(corruption),
        )

        elapsed = (time.perf_counter() - t0) * 1000

        return ImpactAnalysis(
            detection_id=detection_id,
            overall_impact_score=round(overall_score, 1),
            overall_severity=overall_severity,
            impact_dimensions=dimensions,
            temporal_propagation=temporal,
            cascade_chain=cascade,
            resource_impacts=resources,
            coverage_gaps=gaps,
            data_corruption_vectors=corruption,
            compliance_impacts=compliance,
            processing_funnel=funnel,
            component_heatmap=heatmap,
            correlations=correlations,
            summary=summary,
            analysis_time_ms=round(elapsed, 2),
        )

    # ── Impact Dimensions ─────────────────────────────────────────

    def _compute_dimensions(
        self,
        status: str,
        confidence: float,
        camera_id: str,
        telemetry: Dict[str, Any],
        recent: List[Dict[str, Any]],
        anomaly_type: str = "",
        anomaly_severity: str = "",
        anomaly_description: str = "",
    ) -> List[ImpactDimension]:
        dims: List[ImpactDimension] = []

        # ── Anomaly-type severity multiplier ──
        sev_mult = {"low": 0.7, "medium": 1.0, "high": 1.3, "critical": 1.6}.get(
            anomaly_severity.lower() if anomaly_severity else "", 1.0,
        )
        atype = anomaly_type.lower() if anomaly_type else ""
        atype_label = anomaly_type or "detected anomaly"

        # 1. Operational throughput
        latency = telemetry.get("pipeline_latency_ms", 500)
        n_total = max(len(recent), 1)
        n_flags = sum(1 for d in recent if d.get("validation_status", "valid") != "valid")
        throughput_loss = (n_flags / n_total) * 100
        op_score = min(100, (throughput_loss * 1.5 + (latency / 50)) * sev_mult)
        dims.append(ImpactDimension(
            domain=ImpactDomain.OPERATIONAL.value,
            title=f"Operational Impact — {atype_label}",
            score=round(op_score, 1),
            severity=self._score_to_severity(op_score),
            description=(
                f"'{atype_label}' detected: pipeline processing {n_flags}/{n_total} flagged frames. "
                f"Throughput degradation at {throughput_loss:.1f}% with {latency:.0f}ms latency."
            ),
            affected_components=["Pipeline Orchestrator", "Frame Ingestion", "Queue Manager"],
            metrics={
                "throughput_loss_pct": round(throughput_loss, 1),
                "latency_ms": round(latency, 1),
                "flagged_ratio": round(n_flags / n_total, 3),
                "frames_processed": n_total,
                "anomaly_type": anomaly_type,
            },
        ))

        # 2. Surveillance integrity
        cam_dets = [d for d in recent if d.get("camera_id") == camera_id]
        cam_flags = sum(1 for d in cam_dets if d.get("validation_status", "valid") != "valid")
        cam_total = max(len(cam_dets), 1)
        coverage_loss = (cam_flags / cam_total) * 100
        # Physical/structural anomalies have higher surveillance impact
        surv_mult = sev_mult
        if atype in ("structural_damage", "physical_intrusion", "vandalism", "obstruction"):
            surv_mult *= 1.3
        surv_score = min(100, coverage_loss * 1.8 * surv_mult)
        dims.append(ImpactDimension(
            domain=ImpactDomain.SURVEILLANCE.value,
            title=f"Surveillance Impact — {atype_label}",
            score=round(surv_score, 1),
            severity=self._score_to_severity(surv_score),
            description=(
                f"Camera {camera_id}: '{atype_label}' affecting {cam_flags}/{cam_total} frames — "
                f"{coverage_loss:.1f}% coverage loss. "
                f"{'Total blind spot risk.' if coverage_loss > 60 else 'Intermittent gaps.'}"
            ),
            affected_components=["Camera Feed", "Cross-Camera Tracking", "Zone Coverage"],
            metrics={
                "coverage_loss_pct": round(coverage_loss, 1),
                "cam_flag_count": cam_flags,
                "cam_total_frames": cam_total,
                "anomaly_type": anomaly_type,
            },
        ))

        # 3. Data quality
        avg_conf = 0.7
        if recent:
            confs = [d.get("ocr_confidence", 0.7) for d in recent]
            avg_conf = sum(confs) / len(confs)
        dq_score = min(100, ((1.0 - avg_conf) * 130 + throughput_loss * 0.5) * sev_mult)
        dims.append(ImpactDimension(
            domain=ImpactDomain.DATA_QUALITY.value,
            title=f"Data Quality — {atype_label}",
            score=round(dq_score, 1),
            severity=self._score_to_severity(dq_score),
            description=(
                f"'{atype_label}' detection confidence at {confidence:.1%}. "
                f"Data quality index: {100 - dq_score:.1f}/100. "
                f"{'Forensic reliability compromised.' if dq_score > 50 else 'Acceptable with caveats.'}"
            ),
            affected_components=["Detection Engine", "Detection Log", "Analytics", "Forensics"],
            metrics={
                "avg_confidence": round(avg_conf, 3),
                "current_confidence": round(confidence, 3),
                "quality_index": round(100 - dq_score, 1),
                "anomaly_type": anomaly_type,
            },
        ))

        # 4. Cascading failure potential
        vram = telemetry.get("vram_utilisation_pct", 50)
        err_rate = telemetry.get("component_error_rate", 0.01)
        casc_base = (vram - 50) * 1.5 + err_rate * 300 + (1.0 - confidence) * 30
        # Environmental hazards escalate cascading risk
        if atype in ("fire", "flooding", "gas_leak", "environmental_hazard"):
            casc_base += 20
        casc_score = min(100, max(0, casc_base * sev_mult))
        dims.append(ImpactDimension(
            domain=ImpactDomain.CASCADING.value,
            title=f"Cascade Risk — {atype_label}",
            score=round(casc_score, 1),
            severity=self._score_to_severity(casc_score),
            description=(
                f"'{atype_label}' impact: VRAM at {vram:.0f}%, error rate {err_rate:.1%}. "
                f"{'HIGH cascade risk — domino failure likely.' if casc_score > 60 else 'Contained — cascade unlikely.'}"
            ),
            affected_components=["GPU / VRAM", "All Models", "Pipeline Orchestrator"],
            metrics={
                "vram_pct": round(vram, 1),
                "error_rate": round(err_rate, 4),
                "cascade_probability": round(min(1, casc_score / 100), 2),
                "anomaly_type": anomaly_type,
            },
        ))

        # 5. Temporal escalation risk
        trend = self._flag_trend(camera_id, recent)
        temp_base = 30 if trend == "stable" else 60 if trend == "worsening" else 15
        # Certain anomaly types escalate faster
        if atype in ("fire", "structural_damage", "gas_leak"):
            temp_base += 25
        elif atype in ("crowd_anomaly", "traffic_congestion"):
            temp_base += 10
        temp_score = min(100, (temp_base + (n_flags / n_total) * 50) * sev_mult)
        dims.append(ImpactDimension(
            domain=ImpactDomain.TEMPORAL.value,
            title=f"Temporal Escalation — {atype_label}",
            score=round(temp_score, 1),
            severity=self._score_to_severity(temp_score),
            description=(
                f"'{atype_label}' trend: {trend}. "
                f"{'Rapid escalation expected' if trend == 'worsening' else 'Condition may stabilise' if trend == 'improving' else 'Unclear trajectory'}. "
                f"Each second without resolution {'' if temp_score < 40 else 'significantly '}"
                f"increases cumulative damage."
            ),
            affected_components=["Live Feed", "Detection Timeline", "Alert System"],
            metrics={
                "trend": trend,
                "escalation_rate": round(temp_score / 100, 2),
                "flag_ratio": round(n_flags / n_total, 3),
                "anomaly_type": anomaly_type,
            },
        ))

        # 6. Resource utilisation impact
        cpu = telemetry.get("cpu_utilisation_pct", 40)
        res_score = min(100, max(0, ((vram - 40) * 1.2 + (cpu - 40) * 0.5 + latency / 100) * sev_mult))
        dims.append(ImpactDimension(
            domain=ImpactDomain.RESOURCE.value,
            title=f"Resource Impact — {atype_label}",
            score=round(res_score, 1),
            severity=self._score_to_severity(res_score),
            description=(
                f"GPU VRAM {vram:.0f}%, CPU {cpu:.0f}%, latency {latency:.0f}ms. "
                f"{'Resource exhaustion imminent.' if res_score > 70 else 'Resources strained.' if res_score > 40 else 'Resources adequate.'}"
            ),
            affected_components=["GPU / VRAM", "CPU", "Memory", "Model Inference"],
            metrics={
                "vram_pct": round(vram, 1),
                "cpu_pct": round(cpu, 1),
                "latency_ms": round(latency, 1),
            },
        ))

        # 7. Compliance / SLA impact
        compliance_score = 0.0
        if coverage_loss > 20:
            compliance_score += 30
        if dq_score > 40:
            compliance_score += 25
        if throughput_loss > 30:
            compliance_score += 20
        if status == "unreadable":
            compliance_score += 25
        compliance_score = min(100, compliance_score * sev_mult)
        dims.append(ImpactDimension(
            domain=ImpactDomain.COMPLIANCE.value,
            title=f"Compliance & SLA — {atype_label}",
            score=round(compliance_score, 1),
            severity=self._score_to_severity(compliance_score),
            description=(
                f"'{atype_label}' SLA risk score: {compliance_score:.0f}/100. "
                f"{'SLA violation imminent.' if compliance_score > 60 else 'SLA at risk.' if compliance_score > 30 else 'Within SLA bounds.'} "
                f"Detection accuracy and coverage requirements may not be met."
            ),
            affected_components=["SLA Monitor", "Audit Trail", "Compliance Reporting"],
            metrics={
                "sla_risk": round(compliance_score, 1),
                "coverage_loss": round(coverage_loss, 1),
                "data_quality_loss": round(dq_score, 1),
            },
        ))

        return dims

    # ── Temporal Propagation ──────────────────────────────────────

    @staticmethod
    def _compute_temporal_propagation(
        status: str,
        confidence: float,
        telemetry: Dict[str, Any],
        recent: List[Dict[str, Any]],
        camera_id: str,
    ) -> List[TemporalImpactPoint]:
        points: List[TemporalImpactPoint] = []

        # Compute degradation rates based on severity
        base_coverage = max(30, confidence * 100)
        base_quality = max(20, confidence * 95)
        base_stability = 85.0
        base_accuracy = max(25, confidence * 100)
        base_trust = 90.0

        vram = telemetry.get("vram_utilisation_pct", 50)
        if vram > 85:
            base_stability -= 20
        latency = telemetry.get("pipeline_latency_ms", 500)
        fps = telemetry.get("target_fps", 15)

        # Decay function: exponential degradation over time
        time_points = [
            ("0s", 0),
            ("10s", 10),
            ("30s", 30),
            ("1min", 60),
            ("2min", 120),
            ("5min", 300),
            ("10min", 600),
            ("15min", 900),
            ("30min", 1800),
        ]

        is_severe = status in ("unreadable", "regex_fail", "parse_fail")
        decay_rate = 0.003 if is_severe else 0.001

        for label, seconds in time_points:
            t = seconds
            decay = 1.0 - math.exp(-decay_rate * t)

            frames_lost = int(fps * t * (0.3 + decay * 0.5)) if t > 0 else 0
            coverage = max(5, base_coverage - decay * 70)
            quality = max(5, base_quality - decay * 60)
            stability = max(10, base_stability - decay * 50)
            accuracy = max(5, base_accuracy - decay * 55)
            trust = max(10, base_trust - decay * 65)

            if t == 0:
                desc = "Initial anomaly detected — live feed impact begins."
            elif t <= 30:
                desc = f"Early propagation: ~{frames_lost} frames affected. Coverage degrading."
            elif t <= 120:
                desc = f"Impact escalating: {frames_lost} frames lost. Surveillance gaps widening."
            elif t <= 300:
                desc = f"Significant damage: {frames_lost} frames. Cross-camera correlation affected."
            elif t <= 600:
                desc = f"Major impact: {frames_lost} frames. Forensic reliability compromised."
            else:
                desc = f"Severe degradation: {frames_lost} frames lost. Extended blind spot."

            points.append(TemporalImpactPoint(
                time_offset=label,
                time_seconds=seconds,
                cumulative_frames_lost=frames_lost,
                surveillance_coverage_pct=round(coverage, 1),
                data_quality_pct=round(quality, 1),
                system_stability_pct=round(stability, 1),
                detection_accuracy_pct=round(accuracy, 1),
                operator_trust_pct=round(trust, 1),
                description=desc,
            ))

        return points

    # ── Cascade Chain ─────────────────────────────────────────────

    @staticmethod
    def _compute_cascade_chain(
        status: str,
        confidence: float,
        telemetry: Dict[str, Any],
    ) -> List[CascadeNode]:
        chain: List[CascadeNode] = []
        vram = telemetry.get("vram_utilisation_pct", 50)

        # Entry point: the flagged component
        if status in ("low_confidence", "unreadable"):
            chain.append(CascadeNode(
                component="OCR Engine",
                failure_mode="Accuracy degradation" if status == "low_confidence" else "Complete failure",
                time_to_failure="Immediate",
                probability=0.95,
                downstream=["Post-Processor", "Adjudicator"],
                severity="severe" if status == "unreadable" else "moderate",
            ))
        elif status in ("regex_fail", "parse_fail"):
            chain.append(CascadeNode(
                component="Post-Processor",
                failure_mode="Format validation failure",
                time_to_failure="Immediate",
                probability=0.9,
                downstream=["Detection Log", "Analytics"],
                severity="moderate",
            ))
        elif status == "llm_error":
            chain.append(CascadeNode(
                component="LLM Adjudicator",
                failure_mode="Model inference failure",
                time_to_failure="Immediate",
                probability=0.85,
                downstream=["Consensus Validator", "Post-Processor"],
                severity="moderate",
            ))
        elif status == "fallback":
            chain.append(CascadeNode(
                component="Consensus Validator",
                failure_mode="Adjudicator bypass (fallback)",
                time_to_failure="Immediate",
                probability=0.7,
                downstream=["Post-Processor"],
                severity="minor",
            ))

        # Second tier: downstream effects
        chain.append(CascadeNode(
            component="Detection Log",
            failure_mode="Corrupted / low-confidence entries written",
            time_to_failure="0-30 seconds",
            probability=0.8,
            downstream=["Analytics", "Forensics", "Watch-List Matching"],
            severity="moderate",
        ))

        chain.append(CascadeNode(
            component="Cross-Camera Tracker",
            failure_mode="Vehicle identity mismatch / tracking break",
            time_to_failure="30-60 seconds",
            probability=0.6,
            downstream=["Surveillance Network", "Alert System"],
            severity="severe" if confidence < 0.4 else "moderate",
        ))

        # Third tier: system-wide impact
        if vram > 80:
            chain.append(CascadeNode(
                component="GPU / VRAM",
                failure_mode="Memory pressure causing model eviction",
                time_to_failure="1-5 minutes",
                probability=0.5 if vram < 90 else 0.8,
                downstream=["Vehicle Detector", "OCR Engine", "LLM Adjudicator"],
                severity="catastrophic" if vram > 95 else "severe",
            ))

        chain.append(CascadeNode(
            component="Alert System",
            failure_mode="Alert fatigue / noise saturation",
            time_to_failure="5-15 minutes",
            probability=0.55,
            downstream=["Operator Console", "Response Time"],
            severity="moderate",
        ))

        return chain

    # ── Resource Impacts ──────────────────────────────────────────

    @staticmethod
    def _compute_resource_impacts(
        telemetry: Dict[str, Any],
    ) -> List[ResourceImpact]:
        resources: List[ResourceImpact] = []

        vram = telemetry.get("vram_utilisation_pct", 50)
        resources.append(ResourceImpact(
            resource="GPU VRAM",
            current_usage_pct=round(vram, 1),
            projected_peak_pct=round(min(100, vram * 1.15), 1),
            headroom_pct=round(max(0, 100 - vram), 1),
            time_to_exhaustion="< 5 min" if vram > 90 else "15-30 min" if vram > 75 else "> 1 hour",
            risk_level="critical" if vram > 90 else "high" if vram > 80 else "medium" if vram > 60 else "low",
        ))

        cpu = telemetry.get("cpu_utilisation_pct", 40)
        resources.append(ResourceImpact(
            resource="CPU",
            current_usage_pct=round(cpu, 1),
            projected_peak_pct=round(min(100, cpu * 1.2), 1),
            headroom_pct=round(max(0, 100 - cpu), 1),
            time_to_exhaustion="> 1 hour" if cpu < 80 else "30 min",
            risk_level="high" if cpu > 85 else "medium" if cpu > 60 else "low",
        ))

        mem = telemetry.get("memory_utilisation_pct", 45)
        resources.append(ResourceImpact(
            resource="System Memory",
            current_usage_pct=round(mem, 1),
            projected_peak_pct=round(min(100, mem * 1.1), 1),
            headroom_pct=round(max(0, 100 - mem), 1),
            time_to_exhaustion="> 1 hour",
            risk_level="high" if mem > 85 else "medium" if mem > 70 else "low",
        ))

        queue_depth = telemetry.get("queue_depth", 0)
        queue_cap = telemetry.get("queue_capacity", 100)
        queue_pct = (queue_depth / max(queue_cap, 1)) * 100
        resources.append(ResourceImpact(
            resource="Processing Queue",
            current_usage_pct=round(queue_pct, 1),
            projected_peak_pct=round(min(100, queue_pct * 1.3), 1),
            headroom_pct=round(max(0, 100 - queue_pct), 1),
            time_to_exhaustion="< 2 min" if queue_pct > 80 else "10 min" if queue_pct > 50 else "> 30 min",
            risk_level="critical" if queue_pct > 80 else "high" if queue_pct > 60 else "low",
        ))

        return resources

    # ── Coverage Gaps ─────────────────────────────────────────────

    @staticmethod
    def _compute_coverage_gaps(
        camera_id: str,
        status: str,
        confidence: float,
        recent: List[Dict[str, Any]],
    ) -> List[CoverageGap]:
        gaps: List[CoverageGap] = []

        # Primary camera gap
        if status == "unreadable":
            gap_type = "total_blind"
            duration = "Until camera inspection"
            missed = 50  # estimate per minute
        elif confidence < 0.3:
            gap_type = "total_blind"
            duration = "Until OCR recovery"
            missed = 40
        elif confidence < 0.5:
            gap_type = "degraded"
            duration = "5-15 minutes (estimated)"
            missed = 20
        else:
            gap_type = "intermittent"
            duration = "1-5 minutes"
            missed = 5

        gaps.append(CoverageGap(
            camera_id=camera_id,
            gap_type=gap_type,
            start_offset="0s (current frame)",
            duration_estimate=duration,
            vehicles_missed_estimate=missed,
            zone_affected=f"Zone monitored by {camera_id}",
            severity="catastrophic" if gap_type == "total_blind" else "severe" if gap_type == "degraded" else "moderate",
        ))

        # Check other cameras with flags
        other_flagged = {}
        for d in recent:
            cam = d.get("camera_id", "unknown")
            if cam != camera_id and d.get("validation_status", "valid") != "valid":
                other_flagged[cam] = other_flagged.get(cam, 0) + 1

        for cam, count in other_flagged.items():
            if count >= 2:
                gaps.append(CoverageGap(
                    camera_id=cam,
                    gap_type="degraded" if count < 5 else "total_blind",
                    start_offset="Recent window",
                    duration_estimate="Ongoing",
                    vehicles_missed_estimate=count * 3,
                    zone_affected=f"Zone monitored by {cam}",
                    severity="severe" if count >= 5 else "moderate",
                ))

        return gaps

    # ── Data Corruption Vectors ───────────────────────────────────

    @staticmethod
    def _compute_data_corruption(
        status: str,
        confidence: float,
        recent: List[Dict[str, Any]],
        camera_id: str,
    ) -> List[DataCorruptionVector]:
        vectors: List[DataCorruptionVector] = []

        # Plate read corruption
        if confidence < 0.5 or status in ("regex_fail", "parse_fail", "unreadable"):
            n_affected = sum(
                1 for d in recent
                if d.get("camera_id") == camera_id
                and d.get("validation_status", "valid") != "valid"
            )
            vectors.append(DataCorruptionVector(
                source="OCR Engine",
                destination="Detection Log",
                data_type="plate_reads",
                corruption_type="misidentification" if confidence < 0.5 else "garbled",
                records_affected_estimate=max(1, n_affected),
                forensic_impact="partial_loss" if confidence > 0.3 else "irrecoverable",
                severity="severe" if confidence < 0.3 else "moderate",
            ))

        # Tracking corruption
        vectors.append(DataCorruptionVector(
            source="Detection Log",
            destination="Cross-Camera Tracker",
            data_type="tracking",
            corruption_type="misidentification",
            records_affected_estimate=max(1, int((1 - confidence) * 20)),
            forensic_impact="partial_loss",
            severity="severe" if confidence < 0.4 else "moderate",
        ))

        # Analytics corruption
        vectors.append(DataCorruptionVector(
            source="Detection Log",
            destination="Analytics Engine",
            data_type="analytics",
            corruption_type="missing" if status == "unreadable" else "delayed",
            records_affected_estimate=max(1, int((1 - confidence) * 15)),
            forensic_impact="recoverable" if confidence > 0.5 else "partial_loss",
            severity="moderate",
        ))

        # Alert quality
        if status in ("unreadable", "llm_error"):
            vectors.append(DataCorruptionVector(
                source="Alert System",
                destination="Operator Console",
                data_type="alerts",
                corruption_type="missing",
                records_affected_estimate=max(1, int((1 - confidence) * 5)),
                forensic_impact="recoverable",
                severity="moderate",
            ))

        return vectors

    # ── Compliance Impacts ────────────────────────────────────────

    @staticmethod
    def _compute_compliance_impacts(
        status: str,
        overall_score: float,
        recent: List[Dict[str, Any]],
        camera_id: str,
    ) -> List[ComplianceImpact]:
        impacts: List[ComplianceImpact] = []

        n_cam_flags = sum(
            1 for d in recent
            if d.get("camera_id") == camera_id
            and d.get("validation_status", "valid") != "valid"
        )

        # Detection accuracy SLA
        acc_status = "violated" if overall_score > 70 else "at_risk" if overall_score > 40 else "compliant"
        impacts.append(ComplianceImpact(
            regulation="Detection Accuracy SLA",
            requirement=">95% plate read accuracy per camera per hour",
            current_status=acc_status,
            time_to_violation="Immediate" if acc_status == "violated" else "15-30 min" if acc_status == "at_risk" else "> 1 hour",
            liability_level="high" if acc_status == "violated" else "medium" if acc_status == "at_risk" else "none",
            description=f"Camera {camera_id} with {n_cam_flags} flagged detections. Accuracy requirement at risk.",
        ))

        # Coverage continuity SLA
        cov_status = "violated" if status == "unreadable" else "at_risk" if n_cam_flags > 3 else "compliant"
        impacts.append(ComplianceImpact(
            regulation="Coverage Continuity SLA",
            requirement="No camera may have >5 minute continuous blind spot",
            current_status=cov_status,
            time_to_violation="Immediate" if cov_status == "violated" else "5 min" if cov_status == "at_risk" else "> 30 min",
            liability_level="high" if cov_status == "violated" else "low",
            description="Continuous flagging creates a surveillance gap that may breach coverage requirements.",
        ))

        # Data retention integrity
        data_status = "at_risk" if overall_score > 50 else "compliant"
        impacts.append(ComplianceImpact(
            regulation="Data Retention Integrity",
            requirement="All captured plates must be stored with confidence metadata",
            current_status=data_status,
            time_to_violation="Ongoing" if data_status == "at_risk" else "N/A",
            liability_level="medium" if data_status == "at_risk" else "none",
            description="Flagged detections are stored but marked unreliable, affecting audit trail quality.",
        ))

        # Real-time processing SLA
        rt_status = "at_risk" if overall_score > 60 else "compliant"
        impacts.append(ComplianceImpact(
            regulation="Real-Time Processing SLA",
            requirement="Detection-to-alert latency must be <3 seconds",
            current_status=rt_status,
            time_to_violation="< 5 min" if rt_status == "at_risk" else "N/A",
            liability_level="medium" if rt_status == "at_risk" else "none",
            description="Pipeline latency and reprocessing may push detection-to-alert time beyond SLA.",
        ))

        return impacts

    # ── Processing Funnel ─────────────────────────────────────────

    @staticmethod
    def _compute_processing_funnel(
        recent: List[Dict[str, Any]],
    ) -> List[FunnelStage]:
        n = max(len(recent), 1)
        stages: List[FunnelStage] = []

        # Stage 1: Frame Ingestion
        stages.append(FunnelStage(
            stage="Frame Ingestion",
            total_frames=n,
            successful=n,
            failed=0,
            drop_rate_pct=0.0,
            bottleneck=False,
        ))

        # Stage 2: Vehicle Detection
        detected = sum(1 for d in recent if d.get("plate_number"))
        missed = n - detected
        det_drop = (missed / n) * 100 if n else 0
        stages.append(FunnelStage(
            stage="Vehicle Detection",
            total_frames=n,
            successful=detected,
            failed=missed,
            drop_rate_pct=round(det_drop, 1),
            bottleneck=det_drop > 20,
        ))

        # Stage 3: OCR Processing
        ocr_ok = sum(1 for d in recent if d.get("ocr_confidence", 0) > 0.5)
        ocr_fail = detected - ocr_ok
        ocr_drop = (max(0, ocr_fail) / max(detected, 1)) * 100
        stages.append(FunnelStage(
            stage="OCR Processing",
            total_frames=detected,
            successful=max(0, ocr_ok),
            failed=max(0, ocr_fail),
            drop_rate_pct=round(ocr_drop, 1),
            bottleneck=ocr_drop > 15,
        ))

        # Stage 4: Validation
        valid = sum(1 for d in recent if d.get("validation_status") == "valid")
        invalid = ocr_ok - valid
        val_drop = (max(0, invalid) / max(ocr_ok, 1)) * 100
        stages.append(FunnelStage(
            stage="Validation",
            total_frames=max(0, ocr_ok),
            successful=max(0, valid),
            failed=max(0, invalid),
            drop_rate_pct=round(val_drop, 1),
            bottleneck=val_drop > 10,
        ))

        # Stage 5: Final Output
        stages.append(FunnelStage(
            stage="Final Output",
            total_frames=max(0, valid),
            successful=max(0, valid),
            failed=0,
            drop_rate_pct=0.0,
            bottleneck=False,
        ))

        return stages

    # ── Component Heatmap ─────────────────────────────────────────

    @staticmethod
    def _compute_heatmap(
        recent: List[Dict[str, Any]],
    ) -> List[HeatmapCell]:
        cells: List[HeatmapCell] = []
        components = ["Vehicle Detector", "OCR Engine", "Post-Processor", "LLM Adjudicator", "GPU / VRAM"]

        # Build time buckets from detection timestamps
        time_buckets = ["T-5min", "T-4min", "T-3min", "T-2min", "T-1min", "T-0"]

        bucket_size = max(1, len(recent) // len(time_buckets))

        for b_idx, bucket in enumerate(time_buckets):
            start = b_idx * bucket_size
            end = min(start + bucket_size, len(recent))
            bucket_dets = recent[start:end] if start < len(recent) else []

            n_bucket = max(len(bucket_dets), 1)
            n_flags = sum(1 for d in bucket_dets if d.get("validation_status", "valid") != "valid")
            avg_conf = sum(d.get("ocr_confidence", 0.7) for d in bucket_dets) / n_bucket if bucket_dets else 0.7
            n_llm_err = sum(1 for d in bucket_dets if d.get("validation_status") == "llm_error")
            n_regex = sum(1 for d in bucket_dets if d.get("validation_status") in ("regex_fail", "parse_fail"))

            # Vehicle Detector — based on detection count
            det_health = max(20, (1 - n_flags / n_bucket) * 100)
            cells.append(HeatmapCell(component="Vehicle Detector", time_bucket=bucket, health_pct=round(det_health, 1), anomaly_count=n_flags))

            # OCR Engine — based on confidence
            ocr_health = max(10, avg_conf * 100)
            cells.append(HeatmapCell(component="OCR Engine", time_bucket=bucket, health_pct=round(ocr_health, 1), anomaly_count=n_flags))

            # Post-Processor — based on regex/parse failures
            pp_health = max(20, (1 - n_regex / n_bucket) * 100)
            cells.append(HeatmapCell(component="Post-Processor", time_bucket=bucket, health_pct=round(pp_health, 1), anomaly_count=n_regex))

            # LLM Adjudicator
            llm_health = max(15, (1 - n_llm_err / n_bucket) * 100)
            cells.append(HeatmapCell(component="LLM Adjudicator", time_bucket=bucket, health_pct=round(llm_health, 1), anomaly_count=n_llm_err))

            # GPU/VRAM — fairly stable across buckets
            gpu_health = max(20, 80 - n_flags * 3)
            cells.append(HeatmapCell(component="GPU / VRAM", time_bucket=bucket, health_pct=round(gpu_health, 1), anomaly_count=0))

        return cells

    # ── Correlations ──────────────────────────────────────────────

    @staticmethod
    def _compute_correlations(
        recent: List[Dict[str, Any]],
        telemetry: Dict[str, Any],
    ) -> List[CorrelationPair]:
        correlations: List[CorrelationPair] = []

        # Pre-compute metrics for correlation
        if not recent:
            return correlations

        confs = [d.get("ocr_confidence", 0.5) for d in recent]
        flags = [1 if d.get("validation_status", "valid") != "valid" else 0 for d in recent]
        avg_conf = sum(confs) / len(confs)
        flag_rate = sum(flags) / len(flags)

        vram = telemetry.get("vram_utilisation_pct", 50)
        latency = telemetry.get("pipeline_latency_ms", 500)

        # Confidence vs Flag Rate (strong negative — low confidence causes flags)
        conf_flag_corr = -0.85 if avg_conf < 0.5 else -0.6 if avg_conf < 0.7 else -0.3
        correlations.append(CorrelationPair(
            metric_a="OCR Confidence",
            metric_b="Flag Rate",
            correlation=round(conf_flag_corr, 2),
            relationship="strong_negative" if conf_flag_corr < -0.7 else "moderate_negative",
        ))

        # VRAM vs Accuracy
        vram_acc_corr = -0.7 if vram > 85 else -0.3 if vram > 70 else -0.1
        correlations.append(CorrelationPair(
            metric_a="VRAM Utilisation",
            metric_b="Detection Accuracy",
            correlation=round(vram_acc_corr, 2),
            relationship="strong_negative" if vram_acc_corr < -0.5 else "weak_negative",
        ))

        # Latency vs Flag Rate
        lat_flag_corr = 0.6 if latency > 2000 else 0.3 if latency > 1000 else 0.1
        correlations.append(CorrelationPair(
            metric_a="Pipeline Latency",
            metric_b="Flag Rate",
            correlation=round(lat_flag_corr, 2),
            relationship="strong_positive" if lat_flag_corr > 0.5 else "moderate_positive" if lat_flag_corr > 0.25 else "weak_positive",
        ))

        # Flag Rate vs Data Quality
        correlations.append(CorrelationPair(
            metric_a="Flag Rate",
            metric_b="Data Quality",
            correlation=round(-0.9 if flag_rate > 0.3 else -0.5, 2),
            relationship="strong_negative" if flag_rate > 0.3 else "moderate_negative",
        ))

        # VRAM vs Latency
        vram_lat_corr = 0.75 if vram > 85 else 0.4 if vram > 70 else 0.15
        correlations.append(CorrelationPair(
            metric_a="VRAM Utilisation",
            metric_b="Pipeline Latency",
            correlation=round(vram_lat_corr, 2),
            relationship="strong_positive" if vram_lat_corr > 0.6 else "moderate_positive" if vram_lat_corr > 0.3 else "weak_positive",
        ))

        # Coverage vs Compliance
        correlations.append(CorrelationPair(
            metric_a="Surveillance Coverage",
            metric_b="SLA Compliance",
            correlation=0.92,
            relationship="strong_positive",
        ))

        return correlations

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _score_to_severity(score: float) -> str:
        if score >= 80:
            return SeverityTier.CATASTROPHIC.value
        if score >= 60:
            return SeverityTier.SEVERE.value
        if score >= 40:
            return SeverityTier.MODERATE.value
        if score >= 20:
            return SeverityTier.MINOR.value
        return SeverityTier.NEGLIGIBLE.value

    @staticmethod
    def _flag_trend(camera_id: str, recent: List[Dict[str, Any]]) -> str:
        cam_dets = [d for d in recent if d.get("camera_id") == camera_id]
        if len(cam_dets) < 4:
            return "stable"
        half = len(cam_dets) // 2
        first = cam_dets[:half]
        second = cam_dets[half:]
        r1 = sum(1 for d in first if d.get("validation_status", "valid") != "valid") / max(len(first), 1)
        r2 = sum(1 for d in second if d.get("validation_status", "valid") != "valid") / max(len(second), 1)
        if r2 > r1 + 0.1:
            return "worsening"
        if r2 < r1 - 0.1:
            return "improving"
        return "stable"

    @staticmethod
    def _build_summary(
        score: float,
        severity: str,
        status: str,
        camera_id: str,
        plate: str,
        n_dims: int,
        n_gaps: int,
        n_corruption: int,
    ) -> str:
        return (
            f"Impact analysis for plate '{plate}' on camera {camera_id}: "
            f"Overall impact is {severity.upper()} (score: {score:.1f}/100) across {n_dims} domains. "
            f"{n_gaps} coverage gap(s) identified, {n_corruption} data corruption vector(s) mapped. "
            f"Primary flag: {status}. "
            f"{'CRITICAL — cascading failure risk is high. Immediate attention required.' if score > 70 else 'ELEVATED — impact propagating on live feed.' if score > 40 else 'Contained — monitor for escalation.'}"
        )
