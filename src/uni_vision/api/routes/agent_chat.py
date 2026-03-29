"""Agent chat endpoint — natural language interface to the agentic subsystem.

Provides ``POST /api/agent/chat`` for sending queries that the agent
processes through the ReAct loop, and ``GET /api/agent/status`` for
health checks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])


# ── Request / Response models ─────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(
        default=None, max_length=64,
        description="Session ID for multi-turn conversations",
    )


class StepDetail(BaseModel):
    step: int
    thought: str = ""
    tool: Optional[str] = None
    observation: Optional[str] = None
    answer: Optional[str] = None
    elapsed_ms: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    steps: List[StepDetail] = Field(default_factory=list)
    total_steps: int = 0
    elapsed_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    session_id: Optional[str] = None
    intent: Optional[str] = None
    agent_role: Optional[str] = None


class AgentStatus(BaseModel):
    running: bool
    tool_count: int
    available_tools: List[str]


class FeedbackRequest(BaseModel):
    detection_id: str = Field(..., min_length=1, max_length=200)
    feedback_type: str = Field(..., pattern="^(confirm|correct|reject)$")
    original_plate: str = Field(..., min_length=1, max_length=20)
    corrected_plate: str = Field(default="", max_length=20)
    camera_id: str = Field(default="unknown", max_length=100)
    notes: str = Field(default="", max_length=500)


class FeedbackResponse(BaseModel):
    status: str
    feedback_type: str
    detection_id: str


# ── Endpoints ─────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def agent_chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Send a natural-language message to the agent.

    The agent uses a multi-step ReAct reasoning loop to answer queries
    about the ANPR pipeline, detection history, system health, and more.
    """
    coordinator = getattr(request.app.state, "agent_coordinator", None)
    if coordinator is None or not coordinator.is_running:
        raise HTTPException(status_code=503, detail="Agent not available")

    # Classify intent for response metadata
    from uni_vision.agent.intent import classify_intent
    from uni_vision.agent.sub_agents import route_to_role

    classification = classify_intent(body.message)
    role = route_to_role(classification.primary_intent)

    response = await coordinator.chat(body.message, session_id=body.session_id)

    step_details = []
    for s in response.steps:
        tool_name = None
        if s.action:
            tool_name = s.action.get("tool")
        step_details.append(
            StepDetail(
                step=s.step_number,
                thought=s.thought,
                tool=tool_name,
                observation=s.observation[:500] if s.observation else None,
                answer=s.answer,
                elapsed_ms=s.elapsed_ms,
            )
        )

    return ChatResponse(
        answer=response.answer,
        steps=step_details,
        total_steps=response.total_steps,
        elapsed_ms=response.total_elapsed_ms,
        success=response.success,
        error=response.error,
        session_id=body.session_id,
        intent=classification.primary_intent.value,
        agent_role=role.value,
    )


@router.get("/status", response_model=AgentStatus)
async def agent_status(request: Request) -> AgentStatus:
    """Check agent health and list available tools."""
    coordinator = getattr(request.app.state, "agent_coordinator", None)
    if coordinator is None:
        return AgentStatus(running=False, tool_count=0, available_tools=[])

    return AgentStatus(
        running=coordinator.is_running,
        tool_count=coordinator.tool_count,
        available_tools=coordinator._registry.tool_names,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    """Record operator feedback on a detection result.

    Use to confirm correct readings, provide corrections for misreads,
    or reject invalid detections.  Feedback is stored in the knowledge
    base and used to improve per-camera OCR accuracy.
    """
    coordinator = getattr(request.app.state, "agent_coordinator", None)
    if coordinator is None or not coordinator.is_running:
        raise HTTPException(status_code=503, detail="Agent not available")

    from uni_vision.agent.knowledge import FeedbackEntry

    kb = coordinator._knowledge

    if body.feedback_type == "correct" and not body.corrected_plate:
        raise HTTPException(
            status_code=422,
            detail="corrected_plate required when feedback_type is 'correct'",
        )

    entry = FeedbackEntry(
        detection_id=body.detection_id,
        original_plate=body.original_plate,
        corrected_plate=body.corrected_plate or body.original_plate,
        feedback_type=body.feedback_type,
        camera_id=body.camera_id,
        notes=body.notes,
    )
    kb.record_feedback(entry)

    logger.info(
        "feedback_recorded type=%s detection=%s",
        body.feedback_type,
        body.detection_id,
    )

    return FeedbackResponse(
        status="recorded",
        feedback_type=body.feedback_type,
        detection_id=body.detection_id,
    )


@router.get("/sessions")
async def list_sessions(request: Request) -> Dict[str, Any]:
    """List active conversation sessions."""
    coordinator = getattr(request.app.state, "agent_coordinator", None)
    if coordinator is None:
        return {"sessions": [], "total": 0}

    sessions = coordinator._sessions.list_sessions()
    return {"sessions": sessions, "total": len(sessions)}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> Dict[str, str]:
    """Delete a conversation session."""
    coordinator = getattr(request.app.state, "agent_coordinator", None)
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    if coordinator._sessions.delete_session(session_id):
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/monitor")
async def monitor_status(request: Request) -> Dict[str, Any]:
    """Get autonomous monitor status and recent alerts."""
    coordinator = getattr(request.app.state, "agent_coordinator", None)
    if coordinator is None or coordinator._monitor is None:
        return {"running": False, "alerts": []}

    stats = coordinator._monitor.get_stats()
    stats["recent_alerts"] = coordinator._monitor.recent_alerts
    return stats


@router.get("/agents")
async def list_agents(request: Request) -> Dict[str, Any]:
    """List available sub-agent roles and their tool allocations."""
    from uni_vision.agent.sub_agents import PROFILES

    agents = []
    for role, profile in PROFILES.items():
        agents.append({
            "role": role.value,
            "display_name": profile.display_name,
            "tool_count": len(profile.tool_whitelist),
            "tools": sorted(profile.tool_whitelist),
        })
    return {"agents": agents, "total": len(agents)}


@router.get("/audit")
async def audit_trail(request: Request) -> Dict[str, Any]:
    """Get recent agent audit entries (buffered, not yet flushed)."""
    coordinator = getattr(request.app.state, "agent_coordinator", None)
    if coordinator is None:
        return {"entries": [], "pending": 0}

    return {
        "entries": coordinator._audit.get_recent(limit=30),
        "pending": coordinator._audit.pending_count,
    }


# ── Autonomous Workflow Design ────────────────────────────────────


class DesignWorkflowRequest(BaseModel):
    description: str = Field(
        ..., min_length=3, max_length=5000,
        description="Natural-language description of the pipeline to build",
    )
    language: str = Field(
        default="auto", max_length=10,
        description="ISO language code or 'auto' for auto-detection",
    )
    session_id: Optional[str] = Field(
        default=None, max_length=64,
        description="Session ID for conversation history",
    )


class DesignPhaseDetail(BaseModel):
    name: str
    message: str
    elapsed_ms: float
    success: bool


class DesignWorkflowResponse(BaseModel):
    success: bool
    graph: Optional[Dict[str, Any]] = None
    phases: List[DesignPhaseDetail] = Field(default_factory=list)
    detected_language: Optional[str] = None
    english_input: Optional[str] = None
    original_input: Optional[str] = None
    error: Optional[str] = None
    total_elapsed_ms: float = 0.0


@router.post("/design-workflow", response_model=DesignWorkflowResponse)
async def design_workflow(
    body: DesignWorkflowRequest, request: Request,
) -> DesignWorkflowResponse:
    """Autonomous NL → pipeline workflow designer.

    Accepts a natural-language description (in any of Navarasa's 15+
    supported languages) and returns a complete block-node pipeline
    graph ready for deployment.
    """
    coordinator = getattr(request.app.state, "agent_coordinator", None)
    if coordinator is None or not coordinator.is_running:
        raise HTTPException(status_code=503, detail="Agent not available")

    result = await coordinator.design_workflow(
        description=body.description,
        language=body.language,
        session_id=body.session_id,
    )

    phases = [
        DesignPhaseDetail(**p) for p in result.get("phases", [])
    ]

    return DesignWorkflowResponse(
        success=result.get("success", False),
        graph=result.get("graph"),
        phases=phases,
        detected_language=result.get("detected_language"),
        english_input=result.get("english_input"),
        original_input=result.get("original_input"),
        error=result.get("error"),
        total_elapsed_ms=result.get("total_elapsed_ms", 0.0),
    )
