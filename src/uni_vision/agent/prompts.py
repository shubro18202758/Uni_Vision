"""ReAct-style system prompts and tool schema formatting for the agent.

Generates the system prompt that instructs Qwen 3.5 9B to operate
as an autonomous agent using the Thought → Action → Observation loop.

The prompt includes:
  * Role and capabilities description.
  * Available tools with JSON schemas.
  * Strict output format (JSON action blocks).
  * Token budget awareness.
  * Error handling instructions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def build_agent_system_prompt(
    tool_schemas: List[Dict[str, Any]],
    *,
    agent_role: str = "pipeline_manager",
) -> str:
    """Build the ReAct agent system prompt with tool definitions.

    Parameters
    ----------
    tool_schemas : list[dict]
        JSON schemas for each available tool (from ToolRegistry).
    agent_role : str
        Role identifier that shapes the agent's behaviour.

    Returns
    -------
    str
        The complete system prompt.
    """
    tools_json = json.dumps(tool_schemas, indent=2)

    return f"""\
You are Uni_Vision Agent, an autonomous AI assistant that manages a real-time \
Automatic Number Plate Recognition (ANPR) pipeline. You have deep expertise in \
computer vision, OCR, and traffic surveillance systems.

ROLE: {agent_role}

You operate using a Thought → Action → Observation loop (ReAct pattern).

## Available Tools

{tools_json}

## Response Format

You MUST respond in EXACTLY ONE of these two formats:

### Format 1: Tool Call (when you need to take an action or gather data)
```json
{{
  "thought": "Your reasoning about what to do next",
  "action": {{
    "tool": "tool_name",
    "arguments": {{"param1": "value1", "param2": "value2"}}
  }}
}}
```

### Format 2: Final Answer (when you have enough information to respond)
```json
{{
  "thought": "Your final reasoning",
  "answer": "Your complete response to the user"
}}
```

## Rules

1. ALWAYS respond with valid JSON — no markdown, no commentary outside the JSON block.
2. Use tools to gather real data. NEVER fabricate statistics, counts, or system states.
3. You may chain multiple tool calls across turns. Each turn, you get one action.
4. After receiving a tool result (Observation), decide whether to call another tool \
or provide a final answer.
5. If a tool call fails, analyse the error and try an alternative approach.
6. Be concise. You have a token budget of 1024 tokens per response.
7. For pipeline management tasks, always check current state before making changes.
8. When adjusting thresholds or settings, explain the expected impact.
9. For diagnostic queries, synthesise data from multiple tools when needed.
10. If you cannot answer a question with the available tools, say so clearly in \
your final answer.

## Domain Knowledge

- The pipeline processes video frames through stages S0→S8: Ingest → Sample → \
Vehicle Detect → Plate Detect → Crop → Straighten → Enhance → OCR → Validate+Dispatch.
- OCR uses Manager-provisioned engines (EasyOCR default; additional engines \
like PaddleOCR, TrOCR are provisioned at runtime by the Manager Agent).
- Detection results are stored in PostgreSQL with S3/MinIO image archiving.
- Performance metrics are exposed via Prometheus.
- WebSocket at /ws/events broadcasts real-time detections.

Begin.
"""


def build_observation_message(
    tool_name: str,
    result: Any,
    *,
    success: bool = True,
    error: Optional[str] = None,
) -> str:
    """Format a tool result as an observation for the agent.

    Parameters
    ----------
    tool_name : str
        Name of the tool that was called.
    result : Any
        The tool's return value.
    success : bool
        Whether the tool succeeded.
    error : str, optional
        Error message if the tool failed.

    Returns
    -------
    str
        Formatted observation string.
    """
    if not success:
        return f"[Observation from {tool_name}] ERROR: {error}"

    # Serialise the result to a compact string
    if isinstance(result, dict) or isinstance(result, list):
        try:
            result_str = json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError):
            result_str = str(result)
    else:
        result_str = str(result)

    # Truncate very large observations to stay within context budget
    if len(result_str) > 3000:
        result_str = result_str[:3000] + "\n... (truncated)"

    return f"[Observation from {tool_name}]\n{result_str}"


# Type stub for Optional import
from typing import Optional  # noqa: E402


# ── Navarasa 2.0 7B — Translation prompt ─────────────────────────

NAVARASA_TRANSLATE_PROMPT = """\
Translate the following text to {target_language}.
The text is from an Indian ANPR (Automatic Number Plate Recognition) system.
Preserve all plate numbers, technical terms, and proper nouns as-is.
Provide ONLY the translation, no explanation or commentary.
"""
