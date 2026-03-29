"""Top-level agent coordinator that assembles framework components.

AgentCoordinator is the single entry-point for user ↔ agent
interaction.  It owns the AgentLoop, ToolRegistry, WorkingMemory,
AgentLLMClient, and the lifecycle of all agent resources.

Usage (in the API lifespan)::

    coordinator = AgentCoordinator(config)
    await coordinator.start()            # warm up LLM, register tools
    response = await coordinator.chat("How many plates today?")
    await coordinator.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from uni_vision.agent.audit import AuditEntry, AuditTrail
from uni_vision.agent.intent import classify_intent
from uni_vision.agent.knowledge import KnowledgeBase
from uni_vision.agent.llm_client import AgentLLMClient
from uni_vision.agent.loop import AgentLoop, AgentResponse
from uni_vision.agent.memory import WorkingMemory
from uni_vision.agent.monitor import AutonomousMonitor
from uni_vision.agent.navarasa_client import NavarasaClient
from uni_vision.agent.sessions import SessionManager
from uni_vision.agent.sub_agents import MultiAgentRouter, route_to_role, AgentRole
from uni_vision.agent.tools import ToolRegistry

logger = logging.getLogger(__name__)


# ── Execution context passed to every tool invocation ────────────


@dataclass
class ToolExecutionContext:
    """Holds references tools need to interact with live components."""

    pg_client: Any = None
    pipeline: Any = None
    config: Any = None
    knowledge_base: Any = None
    graph_engine: Any = None
    block_registry: Any = None
    # LLM clients — needed by the workflow designer tool
    _llm_client: Any = None
    _navarasa_client: Any = None


# ── Agent coordinator ────────────────────────────────────────────


class AgentCoordinator:
    """Assembles and manages the full agentic sub-system.

    Parameters
    ----------
    config:
        Application configuration (``AppConfig``).  The agent reads
        ``config.agent`` for its own parameters and passes ``config``
        through to tools via the execution context.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        agent_cfg = getattr(config, "agent", None) or _FallbackAgentConfig()

        # Core components
        self._llm_client = AgentLLMClient(
            config=config.ollama,
            timeout_s=agent_cfg.llm_timeout_s,
            max_tokens=agent_cfg.max_tokens,
        )
        self._registry = ToolRegistry()
        self._memory = WorkingMemory(max_tokens=agent_cfg.memory_budget_tokens)
        self._loop = AgentLoop(
            llm_client=self._llm_client,
            registry=self._registry,
            max_iterations=agent_cfg.max_iterations,
        )
        self._knowledge = KnowledgeBase()
        self._sessions = SessionManager()
        self._audit = AuditTrail()
        self._monitor: Optional[AutonomousMonitor] = None
        self._router: Optional[MultiAgentRouter] = None

        # Navarasa 2.0 7B — conversational & interactive UI LLM (no pipeline involvement)
        navarasa_cfg = getattr(config, "navarasa", None)
        if navarasa_cfg and navarasa_cfg.enabled:
            self._navarasa_client = NavarasaClient(navarasa_cfg)
            logger.info(
                "navarasa_enabled model=%s lang=%s",
                navarasa_cfg.model,
                navarasa_cfg.default_language,
            )
        else:
            self._navarasa_client = None
            logger.info("navarasa_disabled")

        # Execution context (populated on start)
        self._context = ToolExecutionContext(config=config)

        # State
        self._started = False

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(
        self,
        pg_client: Any = None,
        pipeline: Any = None,
        *,
        broadcast_fn: Any = None,
    ) -> None:
        """Register tools and prepare the agent for interaction.

        Parameters
        ----------
        broadcast_fn:
            Async callable ``(str) -> None`` for WebSocket alerts.
        """
        if self._started:
            return

        # Populate execution context
        self._context.pg_client = pg_client
        self._context.pipeline = pipeline
        self._context.knowledge_base = self._knowledge
        self._context._llm_client = self._llm_client
        self._context._navarasa_client = self._navarasa_client

        # Restore persisted knowledge from database
        if pg_client is not None:
            try:
                await self._knowledge.load_from_db(pg_client)
                logger.info("agent_knowledge_loaded")
            except Exception:
                logger.warning("agent_knowledge_load_failed", exc_info=True)

        # Ensure audit trail table exists
        if pg_client is not None:
            try:
                await self._audit.ensure_table(pg_client)
            except Exception:
                logger.warning("audit_table_ensure_failed", exc_info=True)

        # Register all concrete pipeline tools
        self._register_tools()

        # Build multi-agent router with role-specific loops
        self._router = MultiAgentRouter(
            full_registry=self._registry,
            llm_client=self._llm_client,
            max_iterations=getattr(
                getattr(self._config, "agent", None) or _FallbackAgentConfig(),
                "max_iterations",
                10,
            ),
        )

        # Start autonomous health monitor
        self._monitor = AutonomousMonitor(
            registry=self._registry,
            context=self._context,
            interval_seconds=60.0,
            broadcast_fn=broadcast_fn,
        )
        await self._monitor.start()

        self._started = True
        logger.info(
            "agent_coordinator_started tools=%d",
            len(self._registry.get_all_schemas()),
        )

    async def shutdown(self) -> None:
        """Release agent resources."""
        # Stop autonomous monitor
        if self._monitor is not None:
            await self._monitor.stop()

        # Flush audit trail
        if self._context.pg_client is not None:
            try:
                await self._audit.flush(self._context.pg_client)
                logger.info("audit_trail_flushed")
            except Exception:
                logger.warning("audit_flush_failed", exc_info=True)

        # Persist knowledge before shutdown
        if self._context.pg_client is not None:
            try:
                await self._knowledge.save_to_db(self._context.pg_client)
                logger.info("agent_knowledge_saved")
            except Exception:
                logger.warning("agent_knowledge_save_failed", exc_info=True)

        await self._llm_client.close()

        # Close Navarasa conversational UI LLM
        if self._navarasa_client is not None:
            await self._navarasa_client.close()
            logger.info("navarasa_client_closed")

        self._started = False
        logger.info("agent_coordinator_shutdown")

    # ── Public API ────────────────────────────────────────────────

    async def chat(
        self,
        user_message: str,
        *,
        session_id: str | None = None,
    ) -> AgentResponse:
        """Send a natural-language message and get an agentic response.

        Parameters
        ----------
        user_message:
            The user's query or instruction.
        session_id:
            Optional session identifier for multi-turn conversations.
            When provided, previous turns are summarised and injected
            as context so the agent can handle follow-up queries.

        The loop internally executes ReAct cycles — Thought → Action →
        Observation — until it produces a final answer or hits the
        iteration limit.
        """
        if not self._started:
            return AgentResponse(
                answer="Agent not started. Call start() first.",
                steps=[],
                total_steps=0,
                total_elapsed_ms=0,
                success=False,
                error="agent_not_started",
            )

        try:
            from uni_vision.monitoring.metrics import (
                AGENT_REQUESTS,
                AGENT_LATENCY,
            )

            AGENT_REQUESTS.inc()
        except (ImportError, AttributeError):
            pass

        # Classify intent for context enrichment
        classification = classify_intent(user_message)

        # Build enriched message with intent hints + session context
        enriched = user_message
        hints: list[str] = []

        if classification.context_hints:
            hints.extend(classification.context_hints)
        if classification.suggested_tools:
            hints.append(
                "Consider using: " + ", ".join(classification.suggested_tools)
            )

        # Session context injection
        session = None
        if session_id:
            session = self._sessions.create_session(session_id)
            ctx_summary = session.get_context_summary()
            if ctx_summary:
                hints.append(f"Previous conversation:\n{ctx_summary}")

        if hints:
            enriched = user_message + "\n\n[System hints: " + " | ".join(hints) + "]"

        t0 = time.perf_counter()

        # Route through multi-agent system
        role = route_to_role(classification.primary_intent)
        if self._router is not None and role != AgentRole.GENERAL:
            response = await self._router.route(
                user_message=enriched,
                intent=classification.primary_intent,
                context=self._context,
            )
        else:
            response = await self._loop.run(
                user_message=enriched,
                context=self._context,
            )

        elapsed = time.perf_counter() - t0

        # Record turn in session
        if session is not None:
            session.add_turn("user", user_message)
            session.add_turn(
                "assistant",
                response.answer,
                tool_calls=response.total_steps,
                elapsed_ms=response.total_elapsed_ms,
            )

        # Record audit entries for each tool call
        for step in response.steps:
            if step.action:
                self._audit.record(AuditEntry(
                    session_id=session_id,
                    intent=classification.primary_intent.value,
                    agent_role=role.value,
                    action=step.action.get("tool", "unknown"),
                    arguments=step.action.get("arguments", {}),
                    result=step.observation[:500] if step.observation else "",
                    success=step.tool_result.success if step.tool_result else True,
                    elapsed_ms=step.elapsed_ms,
                ))
        # Record final answer
        self._audit.record(AuditEntry(
            session_id=session_id,
            intent=classification.primary_intent.value,
            agent_role=role.value,
            action="answer",
            result=response.answer[:500],
            success=response.success,
            elapsed_ms=response.total_elapsed_ms,
        ))

        # Periodic flush
        if self._audit.pending_count >= 20 and self._context.pg_client:
            asyncio.create_task(self._audit.flush(self._context.pg_client))

        try:
            from uni_vision.monitoring.metrics import AGENT_LATENCY

            AGENT_LATENCY.observe(elapsed)
        except (ImportError, AttributeError):
            pass

        logger.info(
            "agent_chat steps=%d elapsed_ms=%.1f success=%s",
            response.total_steps,
            elapsed * 1_000,
            response.success,
        )

        return response

    @property
    def tool_count(self) -> int:
        return len(self._registry.get_all_schemas())

    @property
    def is_running(self) -> bool:
        return self._started

    # ── Autonomous workflow design ────────────────────────────────

    async def design_workflow(
        self,
        description: str,
        language: str = "auto",
        *,
        session_id: str | None = None,
        progress_fn: Any = None,
        model_router: Any = None,
    ) -> Dict:
        """Design a pipeline workflow from natural language.

        This is the dedicated entry-point for the autonomous NL→workflow
        feature.  It bypasses the ReAct loop and calls the
        WorkflowDesigner directly for reliability.

        Returns the WorkflowDesignResult as a dict.
        """
        from uni_vision.agent.workflow_designer import WorkflowDesigner

        if not self._started:
            return {"success": False, "error": "Agent not started"}

        designer = WorkflowDesigner(
            llm_client=self._llm_client,
            navarasa_client=self._navarasa_client,
            model_router=model_router,
        )
        result = await designer.design(
            description,
            language=language,
            progress_fn=progress_fn,
        )

        # Record in audit trail
        self._audit.record(AuditEntry(
            session_id=session_id,
            intent="workflow_design",
            agent_role="workflow_designer",
            action="design_workflow",
            arguments={"description": description[:200], "language": language},
            result=f"success={result.success} blocks={len(result.graph.get('blocks', [])) if result.graph else 0}",
            success=result.success,
            elapsed_ms=result.total_elapsed_ms,
        ))

        # Record in session
        if session_id:
            session = self._sessions.create_session(session_id)
            session.add_turn("user", description)
            summary = (
                f"Designed workflow: {result.graph['project']['name']}"
                if result.success and result.graph
                else f"Workflow design failed: {result.error}"
            )
            session.add_turn("assistant", summary)

        return {
            "success": result.success,
            "graph": result.graph,
            "phases": [
                {
                    "name": p.name,
                    "message": p.message,
                    "elapsed_ms": p.elapsed_ms,
                    "success": p.success,
                }
                for p in result.phases
            ],
            "detected_language": result.detected_language,
            "english_input": result.english_input,
            "original_input": result.original_input,
            "error": result.error,
            "total_elapsed_ms": result.total_elapsed_ms,
        }

    @property
    def indian_contextualizer(self) -> Optional[IndianContextualizer]:
        """Access the Indian contextualiser (None if Navarasa is disabled)."""
        return self._indian_ctx

    # ── Tool registration ─────────────────────────────────────────

    def _register_tools(self) -> None:
        """Import and register all concrete tool functions."""
        from uni_vision.agent import pipeline_tools
        from uni_vision.agent import control_tools
        from uni_vision.agent import knowledge_tools
        from uni_vision.agent import graph_tools

        import inspect

        for module in (pipeline_tools, control_tools, knowledge_tools, graph_tools):
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                defn = getattr(obj, "_tool_definition", None)
                if defn is not None:
                    self._registry.register(obj)
                    logger.debug("registered_tool name=%s", defn.name)


# ── Fallback config when AppConfig.agent is not yet set ──────────


@dataclass
class _FallbackAgentConfig:
    """Sensible defaults when AgentConfig is absent."""

    enabled: bool = True
    max_iterations: int = 10
    llm_timeout_s: float = 30.0
    max_tokens: int = 1024
    memory_budget_tokens: int = 3072
    tool_timeout_s: float = 10.0
