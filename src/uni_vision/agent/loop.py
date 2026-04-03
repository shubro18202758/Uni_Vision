"""ReAct agent loop — multi-step reasoning with tool dispatch.

Implements the core Thought → Action → Observation cycle that drives
the agentic subsystem.  Each iteration:

  1. Send the conversation to the LLM.
  2. Parse the response as JSON (thought + action or answer).
  3. If action: invoke the tool via ToolRegistry, add observation.
  4. If answer: return the final response.
  5. Repeat until max_iterations or final answer.

The loop enforces strict timeouts and iteration limits to prevent
runaway reasoning chains.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from uni_vision.agent.memory import WorkingMemory
from uni_vision.agent.prompts import build_agent_system_prompt, build_observation_message

if TYPE_CHECKING:
    from uni_vision.agent.llm_client import AgentLLMClient
    from uni_vision.agent.tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """A single step in the reasoning trace."""

    step_number: int
    thought: str
    action: dict[str, Any] | None = None
    observation: str | None = None
    answer: str | None = None
    tool_result: ToolResult | None = None
    elapsed_ms: float = 0.0


@dataclass
class AgentResponse:
    """Complete response from an agent run."""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_steps: int = 0
    total_elapsed_ms: float = 0.0
    success: bool = True
    error: str | None = None


class AgentLoop:
    """Multi-step ReAct reasoning loop.

    Parameters
    ----------
    llm_client : AgentLLMClient
        Shared Ollama client for agent reasoning.
    registry : ToolRegistry
        Registry of available tools.
    max_iterations : int
        Maximum reasoning steps before forced termination.
    """

    def __init__(
        self,
        llm_client: AgentLLMClient,
        registry: ToolRegistry,
        *,
        max_iterations: int = 10,
    ) -> None:
        self._llm = llm_client
        self._registry = registry
        self._max_iterations = max_iterations

    async def run(self, user_message: str, *, context: Any = None) -> AgentResponse:
        """Execute the agent loop for a user query.

        Parameters
        ----------
        user_message : str
            The user's natural-language query or instruction.
        context:
            Optional execution context passed to each tool invocation
            (provides pg_client, pipeline, config references).

        Returns
        -------
        AgentResponse
            The agent's final response with full reasoning trace.
        """
        t0 = time.perf_counter()

        # Build system prompt with tool schemas
        system_prompt = build_agent_system_prompt(
            self._registry.get_all_schemas(),
        )

        # Create working memory for this session
        memory = WorkingMemory(
            max_tokens=3072,
            system_prompt=system_prompt,
        )
        memory.add_user_message(user_message)

        steps: list[AgentStep] = []

        for iteration in range(self._max_iterations):
            step_t0 = time.perf_counter()

            # Send conversation to LLM
            try:
                llm_response = await self._llm.chat(
                    memory.to_messages(),
                    temperature=0.2,
                )
            except Exception as exc:
                logger.error("agent_llm_call_failed step=%d error=%s", iteration, exc)
                return AgentResponse(
                    answer=f"I encountered an error communicating with the LLM: {exc}",
                    steps=steps,
                    total_steps=iteration + 1,
                    total_elapsed_ms=(time.perf_counter() - t0) * 1000,
                    success=False,
                    error=str(exc),
                )

            raw_content = llm_response.content.strip()
            memory.add_assistant_message(raw_content)

            # Parse the JSON response
            parsed = self._parse_response(raw_content)

            thought = parsed.get("thought", "")
            action = parsed.get("action")
            answer = parsed.get("answer")

            step = AgentStep(
                step_number=iteration + 1,
                thought=thought,
                elapsed_ms=(time.perf_counter() - step_t0) * 1000,
            )

            # Final answer — done
            if answer is not None:
                step.answer = answer
                steps.append(step)
                return AgentResponse(
                    answer=answer,
                    steps=steps,
                    total_steps=iteration + 1,
                    total_elapsed_ms=(time.perf_counter() - t0) * 1000,
                )

            # Tool call — invoke and continue
            if action is not None:
                tool_name = action.get("tool", "")
                arguments = action.get("arguments", {})
                step.action = action

                logger.info(
                    "agent_tool_call step=%d tool=%s",
                    iteration + 1,
                    tool_name,
                )

                tool_result = await self._registry.invoke(
                    tool_name,
                    arguments,
                    context=context,
                )
                step.tool_result = tool_result

                # Build observation message
                observation = build_observation_message(
                    tool_name,
                    tool_result.data,
                    success=tool_result.success,
                    error=tool_result.error,
                )
                step.observation = observation
                steps.append(step)

                # Add observation to memory for next iteration
                memory.add_tool_result(tool_name, observation)
                continue

            # Neither action nor answer — malformed response
            logger.warning(
                "agent_malformed_response step=%d content=%s",
                iteration + 1,
                raw_content[:200],
            )
            step.answer = raw_content  # Treat as best-effort answer
            steps.append(step)
            return AgentResponse(
                answer=raw_content,
                steps=steps,
                total_steps=iteration + 1,
                total_elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # Max iterations reached
        logger.warning("agent_max_iterations_reached max=%d", self._max_iterations)
        return AgentResponse(
            answer="I reached the maximum number of reasoning steps. "
            "Here's what I found so far: " + (steps[-1].thought if steps else "No progress made."),
            steps=steps,
            total_steps=self._max_iterations,
            total_elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse the LLM response, extracting JSON from potential markdown."""
        # Strip markdown code fences if present
        content = raw.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON within the response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    pass

            logger.debug("agent_json_parse_failed content=%s", content[:200])
            return {"answer": content}
