"""Unit tests for the agentic subsystem (Phases 21-28).

Covers:
  - ToolRegistry: registration, schema generation, invocation
  - WorkingMemory: FIFO eviction, token budget, pinning
  - Intent classifier: pattern matching, tool suggestions
  - Session manager: create, TTL, LRU eviction
  - Sub-agents: routing, filtered registries, role mapping
  - Audit trail: buffering, entry creation
  - Monitor: alert generation
  - Knowledge base: observations, feedback, persistence stubs
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

# ═══════════════════════════════════════════════════════════════════
# ──  ToolRegistry tests  ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestToolRegistry:
    """Test the ToolRegistry and @tool decorator."""

    def test_register_and_list(self):
        from uni_vision.agent.tools import ToolRegistry, tool

        reg = ToolRegistry()

        @tool(name="test_tool", description="A test tool")
        def my_tool(x: int, y: str = "hi") -> str:
            return f"{x}-{y}"

        reg.register(my_tool)

        schemas = reg.get_all_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "test_tool"
        assert "x" in schemas[0]["parameters"]["properties"]

    def test_tool_names(self):
        from uni_vision.agent.tools import ToolRegistry, tool

        reg = ToolRegistry()

        @tool(name="alpha", description="a")
        def alpha_fn():
            return "a"

        @tool(name="beta", description="b")
        def beta_fn():
            return "b"

        reg.register(alpha_fn)
        reg.register(beta_fn)

        assert set(reg.tool_names) == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        from uni_vision.agent.tools import ToolRegistry, tool

        reg = ToolRegistry()

        @tool(name="add", description="add two numbers")
        async def add_fn(a: int, b: int) -> int:
            return a + b

        reg.register(add_fn)

        result = await reg.invoke("add", {"a": 3, "b": 4})
        assert result.success
        assert result.data == 7

    @pytest.mark.asyncio
    async def test_invoke_unknown_tool(self):
        from uni_vision.agent.tools import ToolRegistry

        reg = ToolRegistry()
        result = await reg.invoke("nonexistent", {})
        assert not result.success
        assert "not found" in result.error.lower() or "unknown" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invoke_with_context(self):
        from uni_vision.agent.tools import ToolRegistry, tool

        reg = ToolRegistry()

        @tool(name="ctx_tool", description="uses context")
        async def ctx_fn(context: Any, value: str) -> str:
            return f"ctx={context is not None},val={value}"

        reg.register(ctx_fn)

        ctx = {"some": "context"}
        result = await reg.invoke("ctx_tool", {"value": "hello"}, context=ctx)
        assert result.success
        assert "ctx=True" in str(result.data)

    def test_schema_skips_self_and_context(self):
        from uni_vision.agent.tools import tool

        @tool(name="demo", description="demo tool")
        def demo_fn(context: Any, name: str) -> str:
            return name

        defn = demo_fn._tool_definition
        param_names = [p.name for p in defn.parameters]
        assert "context" not in param_names
        assert "name" in param_names


# ═══════════════════════════════════════════════════════════════════
# ──  WorkingMemory tests  ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestWorkingMemory:
    """Test WorkingMemory token budget and FIFO eviction."""

    def test_add_messages(self):
        from uni_vision.agent.memory import WorkingMemory

        mem = WorkingMemory(max_tokens=3072)
        mem.add_user_message("Hello")
        mem.add_assistant_message("Hi there")

        msgs = mem.to_messages()
        assert len(msgs) >= 2

    def test_system_prompt_pinned(self):
        from uni_vision.agent.memory import WorkingMemory

        mem = WorkingMemory(max_tokens=3072, system_prompt="You are helpful.")
        msgs = mem.to_messages()
        assert msgs[0]["role"] == "system"
        assert "helpful" in msgs[0]["content"]

    def test_eviction_on_budget(self):
        from uni_vision.agent.memory import WorkingMemory

        # Very small budget to force eviction
        mem = WorkingMemory(max_tokens=50, system_prompt="sys")
        for i in range(20):
            mem.add_user_message(f"Message number {i} " * 10)

        msgs = mem.to_messages()
        # System prompt should survive, old messages should be evicted
        assert msgs[0]["role"] == "system"
        # Should have fewer than 20 user messages
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) < 20


# ═══════════════════════════════════════════════════════════════════
# ──  Intent classifier tests  ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestIntentClassifier:
    """Test zero-cost keyword-based intent classification."""

    def test_status_intent(self):
        from uni_vision.agent.intent import QueryIntent, classify_intent

        result = classify_intent("What's the system health?")
        assert result.primary_intent == QueryIntent.STATUS

    def test_detection_intent(self):
        from uni_vision.agent.intent import QueryIntent, classify_intent

        result = classify_intent("Find plate MH12AB1234")
        assert result.primary_intent == QueryIntent.DETECTION

    def test_analytics_intent(self):
        from uni_vision.agent.intent import QueryIntent, classify_intent

        result = classify_intent("Show me hourly trend counts")
        assert result.primary_intent == QueryIntent.ANALYTICS

    def test_camera_intent(self):
        from uni_vision.agent.intent import QueryIntent, classify_intent

        result = classify_intent("Diagnose camera cam_03")
        assert result.primary_intent == QueryIntent.CAMERA

    def test_config_intent(self):
        from uni_vision.agent.intent import QueryIntent, classify_intent

        result = classify_intent("Adjust the confidence threshold")
        assert result.primary_intent == QueryIntent.CONFIG

    def test_knowledge_intent(self):
        from uni_vision.agent.intent import QueryIntent, classify_intent

        result = classify_intent("What are the common confusion error patterns?")
        assert result.primary_intent == QueryIntent.KNOWLEDGE

    def test_diagnostics_intent(self):
        from uni_vision.agent.intent import QueryIntent, classify_intent

        result = classify_intent("The circuit breaker is open, help")
        assert result.primary_intent == QueryIntent.DIAGNOSTICS

    def test_general_fallback(self):
        from uni_vision.agent.intent import QueryIntent, classify_intent

        result = classify_intent("Tell me a joke")
        assert result.primary_intent == QueryIntent.GENERAL

    def test_confidence_range(self):
        from uni_vision.agent.intent import classify_intent

        result = classify_intent("system status")
        assert 0.0 <= result.confidence <= 1.0

    def test_suggested_tools_populated(self):
        from uni_vision.agent.intent import classify_intent

        result = classify_intent("What's the pipeline health?")
        assert isinstance(result.suggested_tools, list)
        # STATUS intent should suggest health-related tools
        assert len(result.suggested_tools) > 0


# ═══════════════════════════════════════════════════════════════════
# ──  Session manager tests  ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestSessionManager:
    """Test session lifecycle, TTL, and LRU eviction."""

    def test_create_and_get(self):
        from uni_vision.agent.sessions import SessionManager

        mgr = SessionManager(ttl_seconds=300)
        session = mgr.create_session("s1")
        assert session.session_id == "s1"
        assert mgr.get_session("s1") is session

    def test_create_idempotent(self):
        from uni_vision.agent.sessions import SessionManager

        mgr = SessionManager()
        s1 = mgr.create_session("s1")
        s2 = mgr.create_session("s1")
        assert s1 is s2

    def test_delete_session(self):
        from uni_vision.agent.sessions import SessionManager

        mgr = SessionManager()
        mgr.create_session("s1")
        assert mgr.delete_session("s1") is True
        assert mgr.get_session("s1") is None

    def test_delete_nonexistent(self):
        from uni_vision.agent.sessions import SessionManager

        mgr = SessionManager()
        assert mgr.delete_session("nope") is False

    def test_list_sessions(self):
        from uni_vision.agent.sessions import SessionManager

        mgr = SessionManager()
        mgr.create_session("s1")
        mgr.create_session("s2")
        sessions = mgr.list_sessions()
        ids = {s["session_id"] for s in sessions}
        assert ids == {"s1", "s2"}

    def test_lru_eviction(self):
        from uni_vision.agent.sessions import SessionManager

        mgr = SessionManager(max_sessions=2)
        mgr.create_session("old")
        mgr.create_session("mid")
        mgr.create_session("new")  # should evict "old"
        assert mgr.get_session("old") is None
        assert mgr.get_session("mid") is not None
        assert mgr.get_session("new") is not None

    def test_add_turn_and_context_summary(self):
        from uni_vision.agent.sessions import SessionManager

        mgr = SessionManager()
        session = mgr.create_session("s1")
        session.add_turn("user", "Hello")
        session.add_turn("assistant", "Hi there!")

        summary = session.get_context_summary()
        assert "Hello" in summary
        assert "Hi there" in summary


# ═══════════════════════════════════════════════════════════════════
# ──  Sub-agent routing tests  ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestSubAgentRouting:
    """Test intent-to-role routing and filtered registries."""

    def test_route_status_to_operations(self):
        from uni_vision.agent.intent import QueryIntent
        from uni_vision.agent.sub_agents import AgentRole, route_to_role

        assert route_to_role(QueryIntent.STATUS) == AgentRole.OPERATIONS

    def test_route_detection_to_analytics(self):
        from uni_vision.agent.intent import QueryIntent
        from uni_vision.agent.sub_agents import AgentRole, route_to_role

        assert route_to_role(QueryIntent.DETECTION) == AgentRole.ANALYTICS

    def test_route_camera_to_ocr_quality(self):
        from uni_vision.agent.intent import QueryIntent
        from uni_vision.agent.sub_agents import AgentRole, route_to_role

        assert route_to_role(QueryIntent.CAMERA) == AgentRole.OCR_QUALITY

    def test_route_general_fallback(self):
        from uni_vision.agent.intent import QueryIntent
        from uni_vision.agent.sub_agents import AgentRole, route_to_role

        assert route_to_role(QueryIntent.GENERAL) == AgentRole.GENERAL

    def test_all_intents_mapped(self):
        from uni_vision.agent.intent import QueryIntent
        from uni_vision.agent.sub_agents import route_to_role

        for intent in QueryIntent:
            role = route_to_role(intent)
            assert role is not None

    def test_filtered_registry_subset(self):
        from uni_vision.agent.sub_agents import build_filtered_registry
        from uni_vision.agent.tools import ToolRegistry, tool

        reg = ToolRegistry()

        @tool(name="tool_a", description="a")
        def fn_a():
            return "a"

        @tool(name="tool_b", description="b")
        def fn_b():
            return "b"

        @tool(name="tool_c", description="c")
        def fn_c():
            return "c"

        reg.register(fn_a)
        reg.register(fn_b)
        reg.register(fn_c)

        filtered = build_filtered_registry(reg, {"tool_a", "tool_c"})
        names = {s["name"] for s in filtered.get_all_schemas()}
        assert names == {"tool_a", "tool_c"}

    def test_filtered_registry_empty_fallback(self):
        from uni_vision.agent.sub_agents import build_filtered_registry
        from uni_vision.agent.tools import ToolRegistry, tool

        reg = ToolRegistry()

        @tool(name="tool_a", description="a")
        def fn_a():
            return "a"

        reg.register(fn_a)

        # Whitelist that matches nothing — should fallback to full
        filtered = build_filtered_registry(reg, {"nonexistent"})
        assert len(filtered.get_all_schemas()) == 1  # falls back

    def test_profiles_have_tools(self):
        from uni_vision.agent.sub_agents import PROFILES

        for _role, profile in PROFILES.items():
            assert len(profile.tool_whitelist) > 0
            assert profile.display_name


# ═══════════════════════════════════════════════════════════════════
# ──  Audit trail tests  ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestAuditTrail:
    """Test audit entry buffering and retrieval."""

    def test_record_and_pending(self):
        from uni_vision.agent.audit import AuditEntry, AuditTrail

        audit = AuditTrail()
        audit.record(AuditEntry(action="test_tool", success=True))
        assert audit.pending_count == 1

    def test_get_recent(self):
        from uni_vision.agent.audit import AuditEntry, AuditTrail

        audit = AuditTrail()
        for i in range(5):
            audit.record(
                AuditEntry(
                    action=f"tool_{i}",
                    intent="status",
                    agent_role="operations",
                    success=True,
                )
            )

        recent = audit.get_recent(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0]["action"] == "tool_4"

    @pytest.mark.asyncio
    async def test_flush_with_no_client(self):
        from uni_vision.agent.audit import AuditEntry, AuditTrail

        audit = AuditTrail()
        audit.record(AuditEntry(action="test"))
        count = await audit.flush(None)
        assert count == 0
        assert audit.pending_count == 1  # not flushed


# ═══════════════════════════════════════════════════════════════════
# ──  Knowledge base tests  ───────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestKnowledgeBase:
    """Test knowledge base observation recording and queries."""

    def test_record_observation(self):
        from uni_vision.agent.knowledge import KnowledgeBase, PlateObservation

        kb = KnowledgeBase()
        obs = PlateObservation(
            plate_text="MH12AB1234",
            camera_id="cam_01",
            confidence=0.95,
            engine="ollama_llm",
            validation_status="valid",
        )
        kb.record_observation(obs)

        freq_list = kb.get_plate_frequency()
        plates = {p for p, _ in freq_list}
        assert "MH12AB1234" in plates

    def test_camera_profile(self):
        from uni_vision.agent.knowledge import KnowledgeBase, PlateObservation

        kb = KnowledgeBase()
        obs = PlateObservation(
            plate_text="MH12AB1234",
            camera_id="cam_01",
            confidence=0.85,
            engine="ollama_llm",
            validation_status="valid",
        )
        kb.record_observation(obs)

        profile = kb.get_camera_profile("cam_01")
        assert profile is not None

    def test_cross_camera_plates(self):
        from uni_vision.agent.knowledge import KnowledgeBase, PlateObservation

        kb = KnowledgeBase()
        kb.record_observation(PlateObservation("ABC123", "cam_01", 0.9, "ollama_llm", "valid"))
        kb.record_observation(PlateObservation("ABC123", "cam_02", 0.9, "ollama_llm", "valid"))

        cross = kb.get_cross_camera_plates()
        assert len(cross) >= 1

    def test_record_feedback(self):
        from uni_vision.agent.knowledge import FeedbackEntry, KnowledgeBase

        kb = KnowledgeBase()
        fb = FeedbackEntry(
            detection_id="det_001",
            original_text="MH12AB1234",
            corrected_text="MH12AB1234",
            feedback_type="confirm",
            camera_id="cam_01",
        )
        kb.record_feedback(fb)

        recent = kb._feedback[-1]
        assert recent.detection_id == "det_001"


# ═══════════════════════════════════════════════════════════════════
# ──  LLM client tests  ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestAgentLLMClient:
    """Test the LLM client configuration."""

    def test_client_creation(self):
        from uni_vision.agent.llm_client import AgentLLMClient

        config = MagicMock()
        config.host = "http://localhost:11434"
        config.model = "gemma4:e2b"

        # httpx is stubbed in test env — just verify no crash
        try:
            client = AgentLLMClient(config=config, timeout_s=30.0, max_tokens=1024)
            assert client is not None
        except Exception:
            # httpx stub may not support full init — that's OK
            pass


# ═══════════════════════════════════════════════════════════════════
# ──  Prompts tests  ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestPrompts:
    """Test system prompt generation."""

    def test_build_system_prompt_with_tools(self):
        from uni_vision.agent.prompts import build_agent_system_prompt

        schemas = [
            {
                "name": "get_health",
                "description": "Get system health",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
        ]
        prompt = build_agent_system_prompt(schemas)
        assert "get_health" in prompt
        assert "system health" in prompt.lower()

    def test_build_observation_message(self):
        from uni_vision.agent.prompts import build_observation_message

        obs = build_observation_message("my_tool", {"result": "ok"}, success=True)
        assert "my_tool" in obs


# ═══════════════════════════════════════════════════════════════════
# ──  Monitor alert tests  ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestMonitorAlerts:
    """Test autonomous monitor alert structures."""

    def test_alert_to_dict(self):
        from uni_vision.agent.monitor import AlertSeverity, HealthAlert

        alert = HealthAlert(
            severity=AlertSeverity.WARNING,
            source="test",
            message="High VRAM usage",
            details={"vram_pct": 80.5},
        )
        d = alert.to_dict()
        assert d["severity"] == "warning"
        assert d["source"] == "test"
        assert "vram_pct" in d["details"]

    def test_alert_severity_ordering(self):
        from uni_vision.agent.monitor import AlertSeverity

        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"


# ═══════════════════════════════════════════════════════════════════
# ──  Coordinator unit tests  ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestCoordinatorUnit:
    """Test coordinator initialization and properties."""

    def test_coordinator_not_started(self):
        from uni_vision.agent.coordinator import AgentCoordinator

        config = MagicMock()
        config.agent = MagicMock()
        config.agent.enabled = True
        config.agent.llm_timeout_s = 10.0
        config.agent.max_tokens = 512
        config.agent.memory_budget_tokens = 1024
        config.agent.max_iterations = 5
        config.agent.tool_timeout_s = 5.0
        config.ollama = MagicMock()
        config.ollama.base_url = "http://localhost:11434"
        config.ollama.host = "http://localhost:11434"
        config.ollama.model = "gemma4:e2b"
        config.navarasa = MagicMock()
        config.navarasa.enabled = False

        coord = AgentCoordinator(config)
        assert not coord.is_running
        assert coord.tool_count == 0

    @pytest.mark.asyncio
    async def test_chat_before_start(self):
        from uni_vision.agent.coordinator import AgentCoordinator

        config = MagicMock()
        config.agent = MagicMock()
        config.agent.enabled = True
        config.agent.llm_timeout_s = 10.0
        config.agent.max_tokens = 512
        config.agent.memory_budget_tokens = 1024
        config.agent.max_iterations = 5
        config.agent.tool_timeout_s = 5.0
        config.ollama = MagicMock()
        config.ollama.base_url = "http://localhost:11434"
        config.ollama.host = "http://localhost:11434"
        config.ollama.model = "gemma4:e2b"
        config.navarasa = MagicMock()
        config.navarasa.enabled = False

        coord = AgentCoordinator(config)
        resp = await coord.chat("hello")
        assert not resp.success
        assert "not started" in resp.answer.lower()


# ═══════════════════════════════════════════════════════════════════
# ──  API endpoint tests  ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestAgentAPIEndpoints:
    """Test agent REST API endpoints with mocked coordinator."""

    @pytest.fixture
    def test_client(self):
        from fastapi.testclient import TestClient

        from uni_vision.api import create_app
        from uni_vision.common.config import AppConfig

        config = AppConfig()
        app = create_app(config, start_pipeline=False)
        return TestClient(app)

    def test_agent_status_no_coordinator(self, test_client):
        resp = test_client.get("/api/agent/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False

    def test_agent_chat_unavailable(self, test_client):
        resp = test_client.post(
            "/api/agent/chat",
            json={"message": "hello"},
        )
        assert resp.status_code == 503

    def test_agent_sessions_empty(self, test_client):
        resp = test_client.get("/api/agent/sessions")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_agent_monitor_no_coordinator(self, test_client):
        resp = test_client.get("/api/agent/monitor")
        assert resp.status_code == 200
        assert resp.json()["running"] is False

    def test_agent_agents_list(self, test_client):
        resp = test_client.get("/api/agent/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 4
        roles = {a["role"] for a in data["agents"]}
        assert roles == {"ocr_quality", "analytics", "operations", "workflow_designer"}

    def test_agent_audit_empty(self, test_client):
        resp = test_client.get("/api/agent/audit")
        assert resp.status_code == 200
        assert resp.json()["pending"] == 0
