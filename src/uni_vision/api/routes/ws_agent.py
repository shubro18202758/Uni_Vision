"""WebSocket endpoint for real-time agent reasoning streams.

Clients connect to ``/ws/agent`` and send JSON messages with a user
query.  The server streams each ReAct reasoning step back as it
happens:  ``thought``, ``tool_call``, ``observation``, ``answer``.

Protocol
--------
Client → Server (JSON)::

    {"message": "How many plates today?", "session_id": "optional-id"}

Server → Client (JSON, one per step)::

    {"type": "thought",      "step": 1, "content": "..."}
    {"type": "tool_call",    "step": 1, "tool": "...", "args": {...}}
    {"type": "observation",  "step": 1, "tool": "...", "content": "..."}
    {"type": "answer",       "step": 3, "content": "...", "role": "analytics"}
    {"type": "done",         "total_steps": 3, "elapsed_ms": 1234.5}
    {"type": "error",        "content": "..."}
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent-websocket"])

# Connected agent-stream clients
_agent_clients: set[WebSocket] = set()


@router.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming agent reasoning.

    Accepts a JSON message with ``message`` and optional ``session_id``,
    then streams each reasoning step back as individual JSON frames.
    """
    await websocket.accept()
    _agent_clients.add(websocket)
    logger.info("ws_agent_client_connected total=%d", len(_agent_clients))

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "content": "Invalid JSON",
                        }
                    )
                )
                continue

            message = payload.get("message", "").strip()
            if not message:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "content": "Empty message",
                        }
                    )
                )
                continue

            session_id = payload.get("session_id")
            msg_type = payload.get("type", "chat")

            # Get coordinator from app state
            coordinator = getattr(websocket.app.state, "agent_coordinator", None)
            if coordinator is None or not coordinator.is_running:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "content": "Agent not available",
                        }
                    )
                )
                continue

            if msg_type == "design_workflow":
                # Autonomous workflow design mode
                model_router = getattr(websocket.app.state, "model_router", None)
                await _stream_workflow_design(
                    websocket,
                    coordinator,
                    message,
                    language=payload.get("language", "auto"),
                    session_id=session_id,
                    model_router=model_router,
                )
            else:
                await _stream_agent_response(
                    websocket,
                    coordinator,
                    message,
                    session_id,
                )

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("ws_agent_client_error", exc_info=True)
    finally:
        _agent_clients.discard(websocket)
        logger.info("ws_agent_client_disconnected total=%d", len(_agent_clients))


async def _stream_agent_response(
    ws: WebSocket,
    coordinator: Any,
    message: str,
    session_id: str | None = None,
) -> None:
    """Execute agent loop and stream each step to the WebSocket client."""
    from uni_vision.agent.intent import classify_intent
    from uni_vision.agent.sub_agents import route_to_role

    t0 = time.perf_counter()

    classification = classify_intent(message)
    role = route_to_role(classification.primary_intent)

    # Send intent classification
    await ws.send_text(
        json.dumps(
            {
                "type": "intent",
                "intent": classification.primary_intent.value,
                "role": role.value,
                "confidence": classification.confidence,
            }
        )
    )

    # Run the agent (response comes all at once — we stream the steps)
    response = await coordinator.chat(message, session_id=session_id)

    # Stream each step
    for step in response.steps:
        if step.thought:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "thought",
                        "step": step.step_number,
                        "content": step.thought,
                    }
                )
            )

        if step.action:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "tool_call",
                        "step": step.step_number,
                        "tool": step.action.get("tool", ""),
                        "args": step.action.get("arguments", {}),
                    }
                )
            )

        if step.observation:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "observation",
                        "step": step.step_number,
                        "tool": step.action.get("tool", "") if step.action else "",
                        "content": step.observation[:2000],
                    }
                )
            )

        if step.answer:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "answer",
                        "step": step.step_number,
                        "content": step.answer,
                        "role": role.value,
                    }
                )
            )

    # Final done frame
    elapsed = (time.perf_counter() - t0) * 1000
    await ws.send_text(
        json.dumps(
            {
                "type": "done",
                "answer": response.answer,
                "total_steps": response.total_steps,
                "elapsed_ms": round(elapsed, 1),
                "success": response.success,
                "role": role.value,
                "session_id": session_id,
            }
        )
    )

    if response.error:
        await ws.send_text(
            json.dumps(
                {
                    "type": "error",
                    "content": response.error,
                }
            )
        )


# ── Autonomous workflow design streaming ──────────────────────────


async def _stream_workflow_design(
    ws: WebSocket,
    coordinator: Any,
    description: str,
    *,
    language: str = "auto",
    session_id: str | None = None,
    model_router: Any = None,
) -> None:
    """Run the autonomous workflow designer and stream phase progress.

    Protocol — Server → Client (JSON, one per event)::

        {"type": "workflow_lock",  "locked": true}
        {"type": "workflow_phase", "phase": "detecting_language", "message": "..."}
        {"type": "workflow_phase", "phase": "translating",        "message": "..."}
        {"type": "workflow_phase", "phase": "designing",          "message": "..."}
        {"type": "workflow_phase", "phase": "validating",         "message": "..."}
        {"type": "workflow_phase", "phase": "building",           "message": "..."}
        {"type": "workflow_phase", "phase": "complete",           "message": "..."}
        {"type": "workflow_complete", "success": true, "graph": {...}, ...}
        {"type": "workflow_lock",  "locked": false}
    """
    t0 = time.perf_counter()

    # Signal UI to enter lock mode
    await ws.send_text(
        json.dumps(
            {
                "type": "workflow_lock",
                "locked": True,
            }
        )
    )

    async def _progress(phase: str, message: str) -> None:
        """Callback invoked by WorkflowDesigner for each phase."""
        await ws.send_text(
            json.dumps(
                {
                    "type": "workflow_phase",
                    "phase": phase,
                    "message": message,
                }
            )
        )

    try:
        result = await coordinator.design_workflow(
            description=description,
            language=language,
            session_id=session_id,
            progress_fn=_progress,
            model_router=model_router,
        )

        elapsed = (time.perf_counter() - t0) * 1000

        await ws.send_text(
            json.dumps(
                {
                    "type": "workflow_complete",
                    "success": result.get("success", False),
                    "graph": result.get("graph"),
                    "phases": result.get("phases", []),
                    "detected_language": result.get("detected_language"),
                    "english_input": result.get("english_input"),
                    "original_input": result.get("original_input"),
                    "error": result.get("error"),
                    "total_elapsed_ms": round(elapsed, 1),
                    "session_id": session_id,
                }
            )
        )

    except Exception as exc:
        logger.error("workflow_design_ws_error: %s", exc, exc_info=True)
        await ws.send_text(
            json.dumps(
                {
                    "type": "workflow_complete",
                    "success": False,
                    "error": str(exc),
                }
            )
        )

    finally:
        # Always release lock mode
        await ws.send_text(
            json.dumps(
                {
                    "type": "workflow_lock",
                    "locked": False,
                }
            )
        )
