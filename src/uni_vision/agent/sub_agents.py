"""Multi-agent sub-system with specialised role-based agents.

Provides three domain-specific sub-agents that each carry a focused
system prompt and a curated subset of tools:

  * **OCR Quality Agent** — focused on OCR accuracy, error patterns,
    camera diagnostics, and prompt-hint generation.
  * **Analytics Agent** — focused on detection trends, plate patterns,
    frequency analysis, and SQL analytics queries.
  * **Operations Agent** — focused on pipeline health, queue management,
    circuit-breaker healing, and configuration tuning.

The coordinator delegates to the right sub-agent based on the
``QueryIntent`` from the intent classifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from uni_vision.agent.intent import QueryIntent
from uni_vision.agent.loop import AgentLoop, AgentResponse
from uni_vision.agent.tools import ToolRegistry

logger = logging.getLogger(__name__)


# ── Sub-agent role definitions ───────────────────────────────────


class AgentRole(str, Enum):
    OCR_QUALITY = "ocr_quality"
    ANALYTICS = "analytics"
    OPERATIONS = "operations"
    WORKFLOW_DESIGNER = "workflow_designer"
    GENERAL = "general"  # fallback — full tool set


@dataclass
class SubAgentProfile:
    """Defines a sub-agent's personality and tool scope."""

    role: AgentRole
    display_name: str
    system_suffix: str  # appended to the base system prompt
    tool_whitelist: set[str]  # tools this sub-agent may use


# ── Role → Tool mapping ─────────────────────────────────────────

_OCR_TOOLS: set[str] = {
    "diagnose_camera",
    "get_camera_error_profile",
    "get_all_camera_profiles",
    "get_ocr_error_patterns",
    "get_ocr_strategy_stats",
    "get_camera_hints",
    "get_knowledge_stats",
    "record_detection_feedback",
    "get_recent_feedback",
    "search_audit_log",
    "reset_circuit_breaker",
    "query_detections",
}

_ANALYTICS_TOOLS: set[str] = {
    "query_detections",
    "get_detection_summary",
    "run_analytics_query",
    "analyze_detection_patterns",
    "get_stage_analytics",
    "get_frequent_detections",
    "get_cross_camera_detections",
    "detect_detection_anomalies",
    "get_knowledge_stats",
    "list_cameras",
}

_OPERATIONS_TOOLS: set[str] = {
    "get_system_health",
    "get_pipeline_stats",
    "get_queue_pressure",
    "flush_inference_queue",
    "self_heal_pipeline",
    "auto_tune_confidence",
    "adjust_threshold",
    "get_current_config",
    "manage_camera",
    "list_cameras",
    "get_ocr_strategy_stats",
    "reset_circuit_breaker",
    "save_knowledge",
}

_WORKFLOW_TOOLS: set[str] = {
    "design_workflow_from_nl",
    "list_available_blocks",
    "get_block_details",
    "validate_pipeline_graph",
    "deploy_pipeline_graph",
    "describe_pipeline_graph",
    "get_deployed_graph",
}

PROFILES: dict[AgentRole, SubAgentProfile] = {
    AgentRole.OCR_QUALITY: SubAgentProfile(
        role=AgentRole.OCR_QUALITY,
        display_name="OCR Quality Agent",
        system_suffix=(
            "You are the OCR Quality specialist. Focus on diagnosing and "
            "improving OCR accuracy. Analyse camera error profiles, character "
            "confusion patterns, and recommend prompt-hint improvements. "
            "When a camera has high error rates, generate specific hints "
            "from the knowledge base."
        ),
        tool_whitelist=_OCR_TOOLS,
    ),
    AgentRole.ANALYTICS: SubAgentProfile(
        role=AgentRole.ANALYTICS,
        display_name="Analytics Agent",
        system_suffix=(
            "You are the Analytics specialist. Focus on data-driven insights "
            "about detection trends, plate frequencies, and cross-camera "
            "movement patterns. Use SQL analytics queries for detailed "
            "analysis. Always include numbers and timeframes in your answers."
        ),
        tool_whitelist=_ANALYTICS_TOOLS,
    ),
    AgentRole.OPERATIONS: SubAgentProfile(
        role=AgentRole.OPERATIONS,
        display_name="Operations Agent",
        system_suffix=(
            "You are the Operations specialist. Focus on pipeline health, "
            "queue management, and system configuration. Proactively detect "
            "and fix issues. When adjusting thresholds, always show current "
            "values first and explain the impact."
        ),
        tool_whitelist=_OPERATIONS_TOOLS,
    ),
    AgentRole.WORKFLOW_DESIGNER: SubAgentProfile(
        role=AgentRole.WORKFLOW_DESIGNER,
        display_name="Workflow Designer Agent",
        system_suffix=(
            "You are the autonomous Workflow Designer. You translate natural "
            "language descriptions into complete block-node pipeline workflows. "
            "Use design_workflow_from_nl to generate pipelines, then validate "
            "and optionally deploy them. Always confirm the generated pipeline "
            "structure with the user before deployment."
        ),
        tool_whitelist=_WORKFLOW_TOOLS,
    ),
}


# ── Intent → Role routing ───────────────────────────────────────

_INTENT_TO_ROLE: dict[QueryIntent, AgentRole] = {
    QueryIntent.STATUS: AgentRole.OPERATIONS,
    QueryIntent.DETECTION: AgentRole.ANALYTICS,
    QueryIntent.ANALYTICS: AgentRole.ANALYTICS,
    QueryIntent.CAMERA: AgentRole.OCR_QUALITY,
    QueryIntent.CONFIG: AgentRole.OPERATIONS,
    QueryIntent.KNOWLEDGE: AgentRole.OCR_QUALITY,
    QueryIntent.DIAGNOSTICS: AgentRole.OPERATIONS,
    QueryIntent.WORKFLOW_DESIGN: AgentRole.WORKFLOW_DESIGNER,
    QueryIntent.GENERAL: AgentRole.GENERAL,
}


def route_to_role(intent: QueryIntent) -> AgentRole:
    """Map a classified intent to the appropriate sub-agent role."""
    return _INTENT_TO_ROLE.get(intent, AgentRole.GENERAL)


# ── Filtered tool registry ──────────────────────────────────────


def build_filtered_registry(
    full_registry: ToolRegistry,
    allowed_tools: set[str],
) -> ToolRegistry:
    """Create a new ToolRegistry containing only the whitelisted tools.

    Falls back to the full registry if the whitelist would produce an
    empty set (safety net).
    """
    filtered = ToolRegistry()

    for schema in full_registry.get_all_schemas():
        if schema["name"] in allowed_tools:
            defn = full_registry._tools[schema["name"]]
            # Directly insert the ToolDefinition — register() expects
            # decorated functions, but here we already have definitions.
            filtered._tools[defn.name] = defn
            filtered._invocation_counts[defn.name] = 0
            filtered._error_counts[defn.name] = 0

    if not filtered.get_all_schemas():
        logger.warning("filtered_registry_empty — falling back to full registry")
        return full_registry

    return filtered


class MultiAgentRouter:
    """Routes queries to specialised sub-agents based on intent.

    Parameters
    ----------
    full_registry:
        The complete ToolRegistry with all registered tools.
    llm_client:
        Shared LLM client for all sub-agents.
    max_iterations:
        Maximum reasoning steps per sub-agent invocation.
    """

    def __init__(
        self,
        full_registry: ToolRegistry,
        llm_client: Any,
        *,
        max_iterations: int = 10,
    ) -> None:
        self._full_registry = full_registry
        self._llm_client = llm_client
        self._max_iterations = max_iterations

        # Pre-build per-role loops
        self._loops: dict[AgentRole, AgentLoop] = {}
        for role, profile in PROFILES.items():
            filtered = build_filtered_registry(
                full_registry,
                profile.tool_whitelist,
            )
            self._loops[role] = AgentLoop(
                llm_client=llm_client,
                registry=filtered,
                max_iterations=max_iterations,
            )

        # General fallback uses the full registry
        self._loops[AgentRole.GENERAL] = AgentLoop(
            llm_client=llm_client,
            registry=full_registry,
            max_iterations=max_iterations,
        )

    async def route(
        self,
        user_message: str,
        intent: QueryIntent,
        *,
        context: Any = None,
    ) -> AgentResponse:
        """Route a query to the appropriate sub-agent.

        Parameters
        ----------
        user_message:
            The (possibly enriched) user query.
        intent:
            Classified intent from the intent classifier.
        context:
            ToolExecutionContext for tool invocations.

        Returns
        -------
        AgentResponse with an added ``agent_role`` attribute on the answer.
        """
        role = route_to_role(intent)
        loop = self._loops.get(role, self._loops[AgentRole.GENERAL])

        profile = PROFILES.get(role)
        enriched = user_message
        if profile:
            enriched = f"[{profile.display_name}] {user_message}"

        logger.info(
            "multi_agent_route intent=%s role=%s tools=%d",
            intent.value,
            role.value,
            len(loop._registry.get_all_schemas()),
        )

        response = await loop.run(enriched, context=context)
        return response

    @property
    def available_roles(self) -> list[str]:
        return [r.value for r in self._loops]
