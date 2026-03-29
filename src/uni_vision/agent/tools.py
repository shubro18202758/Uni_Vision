"""Tool abstraction and registry for the agentic subsystem.

Provides a decorator-based tool registry that:
  * Auto-generates JSON schemas from type hints for LLM tool-calling.
  * Validates tool arguments at invocation time.
  * Tracks execution metrics (invocations, latency, errors).
  * Supports async tool functions natively.

Usage::

    from uni_vision.agent.tools import tool, ToolRegistry

    registry = ToolRegistry()

    @registry.register
    @tool(name="get_pipeline_stats", description="Get current pipeline statistics")
    async def get_pipeline_stats(metric_name: str = "") -> dict:
        ...

    result = await registry.invoke("get_pipeline_stats", {"metric_name": "latency"})
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    get_type_hints,
)

logger = logging.getLogger(__name__)


# ── Python type → JSON Schema mapping ─────────────────────────────

_TYPE_MAP: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(annotation: Any) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = getattr(annotation, "__origin__", None)

    if origin is list or origin is List:
        args = getattr(annotation, "__args__", ())
        item_schema = _python_type_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    if origin is dict or origin is Dict:
        return {"type": "object"}

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"type": "string", "enum": [e.value for e in annotation]}

    schema_type = _TYPE_MAP.get(annotation, "string")
    return {"type": schema_type}


# ── Tool metadata ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolParam:
    """Schema for a single tool parameter."""

    name: str
    description: str
    json_schema: Dict[str, Any]
    required: bool


@dataclass(frozen=True)
class ToolDefinition:
    """Complete definition of a registered tool."""

    name: str
    description: str
    parameters: List[ToolParam]
    fn: Callable[..., Coroutine[Any, Any, Any]]

    def to_schema(self) -> Dict[str, Any]:
        """Generate the LLM-facing JSON schema for this tool."""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for p in self.parameters:
            prop = dict(p.json_schema)
            prop["description"] = p.description
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        schema: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        }
        if required:
            schema["parameters"]["required"] = required

        return schema


# ── Tool decorator ────────────────────────────────────────────────


def tool(
    *,
    name: str,
    description: str,
    param_descriptions: Optional[Dict[str, str]] = None,
) -> Callable:
    """Decorate an async function as an agent tool.

    Parameters
    ----------
    name : str
        Unique tool name (used in LLM tool calls).
    description : str
        Human-readable description shown to the LLM.
    param_descriptions : dict, optional
        Per-parameter descriptions. Keys are param names.
    """
    _param_descs = param_descriptions or {}

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)

        params: List[ToolParam] = []
        for pname, param in sig.parameters.items():
            if pname == "self" or pname == "context":
                continue
            annotation = hints.get(pname, str)
            has_default = param.default is not inspect.Parameter.empty
            params.append(
                ToolParam(
                    name=pname,
                    description=_param_descs.get(pname, pname),
                    json_schema=_python_type_to_json_schema(annotation),
                    required=not has_default,
                )
            )

        fn._tool_definition = ToolDefinition(  # type: ignore[attr-defined]
            name=name,
            description=description,
            parameters=params,
            fn=fn,
        )
        return fn

    return decorator


# ── Tool invocation result ────────────────────────────────────────


@dataclass
class ToolResult:
    """Result from a tool invocation."""

    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0


# ── Tool registry ─────────────────────────────────────────────────


class ToolRegistry:
    """Central registry for all agent tools.

    Tools are registered via the ``register`` method or
    ``register_instance_tools`` for class-based tools.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._invocation_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}

    def register(self, fn_or_tool: Any) -> Any:
        """Register a decorated tool function."""
        defn: Optional[ToolDefinition] = getattr(fn_or_tool, "_tool_definition", None)
        if defn is None:
            raise ValueError(
                f"Cannot register {fn_or_tool!r} — not decorated with @tool"
            )
        if defn.name in self._tools:
            raise ValueError(f"Tool '{defn.name}' is already registered")
        self._tools[defn.name] = defn
        self._invocation_counts[defn.name] = 0
        self._error_counts[defn.name] = 0
        logger.debug("tool_registered name=%s", defn.name)
        return fn_or_tool

    def register_instance_tools(self, instance: object) -> None:
        """Scan an object for methods decorated with ``@tool`` and register them."""
        for attr_name in dir(instance):
            if attr_name.startswith("_"):
                continue
            method = getattr(instance, attr_name, None)
            if method is None:
                continue
            defn = getattr(method, "_tool_definition", None)
            if defn is not None:
                # Re-bind the definition with the bound method
                bound_defn = ToolDefinition(
                    name=defn.name,
                    description=defn.description,
                    parameters=defn.parameters,
                    fn=method,
                )
                self._tools[bound_defn.name] = bound_defn
                self._invocation_counts[bound_defn.name] = 0
                self._error_counts[bound_defn.name] = 0
                logger.debug("tool_registered name=%s (instance)", bound_defn.name)

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Return JSON schemas for all registered tools (for the LLM prompt)."""
        return [defn.to_schema() for defn in self._tools.values()]

    async def invoke(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        *,
        context: Any = None,
    ) -> ToolResult:
        """Invoke a tool by name with the given arguments.

        Parameters
        ----------
        tool_name : str
            Name of the tool to invoke.
        arguments : dict
            Arguments parsed from the LLM's tool call.
        context : Any, optional
            Execution context passed through to the tool function.

        Returns
        -------
        ToolResult
            The result of the tool invocation.
        """
        defn = self._tools.get(tool_name)
        if defn is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        self._invocation_counts[tool_name] = (
            self._invocation_counts.get(tool_name, 0) + 1
        )

        t0 = time.perf_counter()
        try:
            # Inject context if the tool accepts it
            sig = inspect.signature(defn.fn)
            if "context" in sig.parameters:
                arguments = {**arguments, "context": context}

            result = await defn.fn(**arguments)
            elapsed = (time.perf_counter() - t0) * 1000

            self._record_metric(tool_name, success=True)

            return ToolResult(
                tool_name=tool_name,
                success=True,
                data=result,
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            self._error_counts[tool_name] = (
                self._error_counts.get(tool_name, 0) + 1
            )
            logger.warning(
                "tool_invocation_failed tool=%s error=%s",
                tool_name,
                str(exc),
            )

            self._record_metric(tool_name, success=False)

            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(exc),
                elapsed_ms=elapsed,
            )

    @staticmethod
    def _record_metric(tool_name: str, *, success: bool) -> None:
        """Record tool call metric (best-effort)."""
        try:
            from uni_vision.monitoring.metrics import AGENT_TOOL_CALLS

            AGENT_TOOL_CALLS.labels(
                tool_name=tool_name, success=str(success).lower()
            ).inc()
        except (ImportError, AttributeError):
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Return invocation and error counts for all tools."""
        return {
            name: {
                "invocations": self._invocation_counts.get(name, 0),
                "errors": self._error_counts.get(name, 0),
            }
            for name in self._tools
        }
