"""Lightweight intent classifier for optimising agent routing.

Classifies user queries into categories *before* entering the
full ReAct loop, enabling:

  * Fast-path responses for simple status queries.
  * Pre-selecting relevant tools so the LLM focuses on the right
    subset.
  * Injecting domain-specific context hints into the system prompt.

Classification is keyword/pattern-based (zero additional LLM calls)
so latency overhead is negligible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntent(str, Enum):
    """High-level query categories."""

    STATUS = "status"  # System/pipeline health
    DETECTION = "detection"  # Query detections / plate lookups
    ANALYTICS = "analytics"  # Aggregate stats or trend analysis
    CAMERA = "camera"  # Camera-specific queries / management
    CONFIG = "config"  # Configuration viewing / adjustment
    KNOWLEDGE = "knowledge"  # Knowledge base / learning / feedback
    DIAGNOSTICS = "diagnostics"  # Troubleshooting / self-heal
    WORKFLOW_DESIGN = "workflow_design"  # NL → pipeline workflow design
    GENERAL = "general"  # Catch-all / conversational


@dataclass
class ClassificationResult:
    """Result of intent classification."""

    primary_intent: QueryIntent
    confidence: float  # 0.0 – 1.0
    suggested_tools: list[str] = field(default_factory=list)
    context_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.primary_intent.value,
            "confidence": round(self.confidence, 2),
            "suggested_tools": self.suggested_tools,
            "context_hints": self.context_hints,
        }


# ── Pattern definitions ──────────────────────────────────────────

_PATTERNS: dict[QueryIntent, list[re.Pattern[str]]] = {
    QueryIntent.STATUS: [
        re.compile(r"\b(status|health|alive|running|up|uptime)\b", re.I),
        re.compile(r"\b(pipeline|system).*(state|ok|good|bad)\b", re.I),
        re.compile(r"\bhow.*(pipeline|system).*(doing|going)\b", re.I),
        re.compile(r"\bqueue.*(pressure|depth|backlog)\b", re.I),
        re.compile(r"\bvram|memory usage\b", re.I),
    ],
    QueryIntent.DETECTION: [
        re.compile(r"\b(detection|detect|detected|identified|read|plate|object)\b", re.I),
        re.compile(r"\b(find|search|look\s*up|query|show).*(detection|object|plate|vehicle|result)\b", re.I),
        re.compile(r"\b[A-Z]{2,3}[\s-]?\d{2,4}[\s-]?[A-Z]{0,3}\b"),  # alphanumeric ID pattern
        re.compile(r"\brecent.*(detection|result|read|finding)\b", re.I),
        re.compile(r"\bhow many.*(detection|object|result|plate|vehicle)\b", re.I),
    ],
    QueryIntent.ANALYTICS: [
        re.compile(r"\b(analytics|trend|pattern|statistics|stat|avg|average)\b", re.I),
        re.compile(r"\b(hourly|daily|weekly|monthly).*(trend|count|rate)\b", re.I),
        re.compile(r"\b(top|most|least|busiest|peak)\b", re.I),
        re.compile(r"\b(compare|comparison|breakdown)\b", re.I),
        re.compile(r"\b(frequency|distribution)\b", re.I),
    ],
    QueryIntent.CAMERA: [
        re.compile(r"\bcamera[\s-]?\w*\b", re.I),
        re.compile(r"\b(cam|source)[\s-]?\d+\b", re.I),
        re.compile(r"\b(enable|disable|restart).*(camera|source)\b", re.I),
        re.compile(r"\b(camera|source).*(error|issue|down|offline)\b", re.I),
    ],
    QueryIntent.CONFIG: [
        re.compile(r"\b(config|configure|setting|threshold|parameter)\b", re.I),
        re.compile(r"\b(change|adjust|tune|set|modify|update).*(threshold|confidence|setting)\b", re.I),
        re.compile(r"\b(current|show).*(config|setting)\b", re.I),
    ],
    QueryIntent.KNOWLEDGE: [
        re.compile(r"\b(knowledge|learn|feedback|correction|confirm)\b", re.I),
        re.compile(r"\b(error.*(pattern|profile)|confusion|misread)\b", re.I),
        re.compile(r"\b(cross.?camera|multi.?camera)\b", re.I),
        re.compile(r"\b(frequent|common).*(detection|plate|vehicle|object)\b", re.I),
        re.compile(r"\b(anomal|spike|unusual|suspicious)\b", re.I),
    ],
    QueryIntent.DIAGNOSTICS: [
        re.compile(r"\b(diagnos|troubleshoot|debug|fix|heal|repair)\b", re.I),
        re.compile(r"\b(circuit.?breaker|cb|fallback)\b", re.I),
        re.compile(r"\b(error|failure|fail|broken|wrong)\b", re.I),
        re.compile(r"\b(slow|latency|timeout|bottleneck)\b", re.I),
        re.compile(r"\b(ocr|recognition).*(bad|poor|fail|issue)\b", re.I),
    ],
    QueryIntent.WORKFLOW_DESIGN: [
        re.compile(r"\b(design|create|build|make|generate|setup|set\s*up).*(workflow|pipeline|flow)\b", re.I),
        re.compile(r"\b(workflow|pipeline).*(design|create|build|make|generate)\b", re.I),
        re.compile(r"\bI\s+want.*(pipeline|workflow|detection)\b", re.I),
        re.compile(r"\b(auto|automat).*(pipeline|workflow|build|design)\b", re.I),
        re.compile(r"\b(connect|wire|chain|link).*(blocks?|nodes?|stages?)\b", re.I),
        re.compile(r"\bmake\s+me.*(pipeline|workflow|system|setup)\b", re.I),
        # Multilingual triggers (Devanagari / Telugu / Tamil / Kannada / Bengali)
        re.compile(r"\u092A\u093E\u0907\u092A\u0932\u093E\u0907\u0928", re.I),  # पाइपलाइन (Hindi)
        re.compile(r"\u0935\u0930\u094D\u0915\u092B\u094D\u0932\u094B", re.I),  # वर्कफ्लो (Hindi)
        re.compile(r"\u0C2A\u0C48\u0C2A\u0C4D\u200C\u0C32\u0C48\u0C28\u0C4D", re.I),  # పైప్‌లైన్ (Telugu)
        re.compile(r"\u0BAA\u0BC8\u0BAA\u0BCD\u0BB2\u0BC8\u0BA9\u0BCD", re.I),  # பைப்லைன் (Tamil)
        re.compile(r"\u09AA\u09BE\u0987\u09AA\u09B2\u09BE\u0987\u09A8", re.I),  # পাইপলাইন (Bengali)
    ],
}

_TOOL_SUGGESTIONS: dict[QueryIntent, list[str]] = {
    QueryIntent.STATUS: [
        "get_system_health",
        "get_pipeline_stats",
        "get_queue_pressure",
    ],
    QueryIntent.DETECTION: [
        "query_detections",
        "get_detection_summary",
        "search_audit_log",
    ],
    QueryIntent.ANALYTICS: [
        "run_analytics_query",
        "analyze_detection_patterns",
        "get_stage_analytics",
    ],
    QueryIntent.CAMERA: [
        "list_cameras",
        "diagnose_camera",
        "manage_camera",
        "get_camera_error_profile",
    ],
    QueryIntent.CONFIG: [
        "get_current_config",
        "adjust_threshold",
        "auto_tune_confidence",
    ],
    QueryIntent.KNOWLEDGE: [
        "get_knowledge_stats",
        "get_frequent_detections",
        "get_cross_camera_detections",
        "detect_detection_anomalies",
        "get_ocr_error_patterns",
    ],
    QueryIntent.DIAGNOSTICS: [
        "self_heal_pipeline",
        "get_ocr_strategy_stats",
        "diagnose_camera",
        "reset_circuit_breaker",
    ],
    QueryIntent.WORKFLOW_DESIGN: [
        "design_workflow_from_nl",
        "list_available_blocks",
        "validate_pipeline_graph",
        "deploy_pipeline_graph",
    ],
    QueryIntent.GENERAL: [],
}

_CONTEXT_HINTS: dict[QueryIntent, list[str]] = {
    QueryIntent.STATUS: [
        "Focus on overall system health and pipeline state.",
    ],
    QueryIntent.DETECTION: [
        "The user wants to look up specific detections or identified objects.",
        "Use query_detections for filtering and get_detection_summary for counts.",
    ],
    QueryIntent.ANALYTICS: [
        "Provide data-driven insights with numbers and trends.",
    ],
    QueryIntent.CAMERA: [
        "Focus on camera-specific information and management.",
    ],
    QueryIntent.CONFIG: [
        "Show current values before suggesting changes.",
    ],
    QueryIntent.KNOWLEDGE: [
        "Leverage the knowledge base for accumulated patterns and feedback.",
    ],
    QueryIntent.DIAGNOSTICS: [
        "Prioritise identifying the root cause before suggesting fixes.",
        "Check circuit breaker state and OCR strategy stats.",
    ],
    QueryIntent.WORKFLOW_DESIGN: [
        "The user wants to design a pipeline workflow from natural language.",
        "Enter autonomous agentic mode — design the full pipeline.",
        "Use the WorkflowDesigner to translate NL → block-node graph.",
    ],
}


# ── Classifier ────────────────────────────────────────────────────


def classify_intent(message: str) -> ClassificationResult:
    """Classify a user message into an intent category.

    Returns the best-matching intent with suggested tools and
    context hints.  When no strong match is found, defaults to
    ``QueryIntent.GENERAL`` with low confidence.
    """
    scores: dict[QueryIntent, float] = {}

    for intent, patterns in _PATTERNS.items():
        hits = sum(1 for p in patterns if p.search(message))
        if hits:
            scores[intent] = hits / len(patterns)

    if not scores:
        return ClassificationResult(
            primary_intent=QueryIntent.GENERAL,
            confidence=0.3,
            suggested_tools=[],
            context_hints=[],
        )

    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    # Normalize score to 0.4–1.0 range
    confidence = min(0.4 + best_score * 0.6, 1.0)

    return ClassificationResult(
        primary_intent=best_intent,
        confidence=confidence,
        suggested_tools=_TOOL_SUGGESTIONS.get(best_intent, []),
        context_hints=_CONTEXT_HINTS.get(best_intent, []),
    )
