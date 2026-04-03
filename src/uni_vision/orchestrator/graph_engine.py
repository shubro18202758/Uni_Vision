"""Dynamic graph engine — interprets and executes user-defined pipeline graphs.

Receives a ``ProjectGraph`` (JSON from the frontend), topologically
sorts the nodes, and executes each node in order by routing data along
connections.  Each block type maps to a *handler* function that processes
its inputs and produces outputs.

The engine is deliberately **not** tied to any domain-specific logic.  It
operates on generic data slots (frames, bounding box lists, text, etc.)
and dispatches to the appropriate backend component for each block type.

Execution Model
───────────────
1. Convert the graph into an adjacency list.
2. Topological sort (Kahn's algorithm) — detects cycles.
3. For each node in sorted order:
   a. Collect inputs from upstream connection data.
   b. Look up the handler for this block type.
   c. Execute the handler with (inputs, config).
   d. Store outputs keyed by ``(block_id, port_id)``.
4. Return the accumulated outputs from all terminal nodes.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# ── Graph data structures ─────────────────────────────────────────


class GraphNode:
    """A node in the execution graph."""

    __slots__ = ("category", "config", "handler_key", "id", "label", "type")

    def __init__(
        self,
        id: str,
        type: str,
        label: str,
        category: str,
        config: dict[str, Any],
        handler_key: str = "",
    ) -> None:
        self.id = id
        self.type = type
        self.label = label
        self.category = category
        self.config = config
        self.handler_key = handler_key


class GraphEdge:
    """A connection between two ports."""

    __slots__ = ("id", "source", "source_handle", "target", "target_handle")

    def __init__(
        self,
        id: str,
        source: str,
        source_handle: str,
        target: str,
        target_handle: str,
    ) -> None:
        self.id = id
        self.source = source
        self.source_handle = source_handle
        self.target = target
        self.target_handle = target_handle


class ExecutionResult:
    """Result of a single graph execution pass."""

    def __init__(self) -> None:
        self.success: bool = True
        self.error: str | None = None
        self.node_outputs: dict[str, dict[str, Any]] = {}
        self.executed_nodes: list[str] = []
        self.terminal_outputs: dict[str, Any] = {}
        self.elapsed_ms: float = 0.0
        self.stage_timings: dict[str, float] = {}


# ── Topological sort ──────────────────────────────────────────────


def topological_sort(
    node_ids: list[str],
    edges: list[GraphEdge],
) -> tuple[list[str], str | None]:
    """Kahn's algorithm — returns (sorted_ids, error_or_None).

    If the graph contains cycles, returns (partial_list, error_message).
    """
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for edge in edges:
        if edge.source in adjacency and edge.target in in_degree:
            adjacency[edge.source].append(edge.target)
            in_degree[edge.target] += 1

    queue: deque[str] = deque()
    for nid, deg in in_degree.items():
        if deg == 0:
            queue.append(nid)

    sorted_ids: list[str] = []
    while queue:
        node = queue.popleft()
        sorted_ids.append(node)
        for neighbour in adjacency[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(sorted_ids) != len(node_ids):
        return sorted_ids, "Graph contains a cycle — cannot execute"

    return sorted_ids, None


# ── Validation ────────────────────────────────────────────────────


def validate_graph(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    block_registry: Any = None,
) -> list[dict[str, Any]]:
    """Validate graph structure. Returns a list of issues (may be empty)."""
    issues: list[dict[str, Any]] = []
    node_map = {n.id: n for n in nodes}

    # Check for cycles
    _, cycle_err = topological_sort([n.id for n in nodes], edges)
    if cycle_err:
        issues.append(
            {
                "id": "cycle",
                "level": "error",
                "message": cycle_err,
            }
        )

    # Check for dangling edge references
    node_ids = set(node_map.keys())
    for edge in edges:
        if edge.source not in node_ids:
            issues.append(
                {
                    "id": f"dangling-source-{edge.id}",
                    "level": "error",
                    "message": f"Edge '{edge.id}' references non-existent source '{edge.source}'",
                }
            )
        if edge.target not in node_ids:
            issues.append(
                {
                    "id": f"dangling-target-{edge.id}",
                    "level": "error",
                    "message": f"Edge '{edge.id}' references non-existent target '{edge.target}'",
                }
            )

    # Warn on disconnected nodes (no inputs and no outputs connected)
    connected: set[str] = set()
    for edge in edges:
        connected.add(edge.source)
        connected.add(edge.target)
    for node in nodes:
        if node.id not in connected and node.category != "Input":
            issues.append(
                {
                    "id": f"disconnected-{node.id}",
                    "level": "warning",
                    "message": f"Block '{node.label}' ({node.id}) has no connections",
                    "blockId": node.id,
                }
            )

    return issues


# ── Handler type ──────────────────────────────────────────────────

# Handler signature: (inputs: Dict[port_id, data], config: Dict) -> Dict[port_id, data]
HandlerFn = Callable[[dict[str, Any], dict[str, Any]], Any]


# ── Graph Engine ──────────────────────────────────────────────────


class GraphEngine:
    """Interprets and executes user-defined pipeline graphs.

    The engine is stateless per-execution: call ``execute()`` with a
    graph and optional initial data, and it returns an ``ExecutionResult``.

    Handlers are registered by ``handler_key`` (e.g. "detection.vehicle").
    For block types without a registered handler, a passthrough handler
    is used that forwards inputs to outputs.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, HandlerFn] = {}
        self._current_graph_nodes: list[GraphNode] = []
        self._current_graph_edges: list[GraphEdge] = []
        self._current_graph_json: dict[str, Any] | None = None

    # ── Handler registration ──────────────────────────────────────

    def register_handler(self, handler_key: str, handler_fn: HandlerFn) -> None:
        """Register an execution handler for a block type."""
        self._handlers[handler_key] = handler_fn
        logger.debug("handler_registered key=%s", handler_key)

    def has_handler(self, handler_key: str) -> bool:
        return handler_key in self._handlers

    # ── Graph loading ─────────────────────────────────────────────

    def load_graph(self, graph_json: dict[str, Any]) -> list[dict[str, Any]]:
        """Load a ProjectGraph dict, validate, and store for execution.

        Returns a list of validation issues (empty = valid).
        """
        blocks = graph_json.get("blocks", [])
        connections = graph_json.get("connections", [])

        nodes: list[GraphNode] = []
        for b in blocks:
            nodes.append(
                GraphNode(
                    id=b["id"],
                    type=b["type"],
                    label=b.get("label", b["type"]),
                    category=b.get("category", "Utility"),
                    config=b.get("config", {}),
                    handler_key=b.get("backend_handler", ""),
                )
            )

        edges: list[GraphEdge] = []
        for c in connections:
            edges.append(
                GraphEdge(
                    id=c["id"],
                    source=c["source"],
                    source_handle=c["sourceHandle"],
                    target=c["target"],
                    target_handle=c["targetHandle"],
                )
            )

        issues = validate_graph(nodes, edges)

        # Only store if no errors
        has_errors = any(i["level"] == "error" for i in issues)
        if not has_errors:
            self._current_graph_nodes = nodes
            self._current_graph_edges = edges
            self._current_graph_json = graph_json
            logger.info(
                "graph_loaded nodes=%d edges=%d",
                len(nodes),
                len(edges),
            )

        return issues

    def get_current_graph(self) -> dict[str, Any] | None:
        """Return the currently loaded graph JSON, or None."""
        return self._current_graph_json

    def clear_graph(self) -> None:
        """Remove the currently loaded graph."""
        self._current_graph_nodes = []
        self._current_graph_edges = []
        self._current_graph_json = None

    def has_graph(self) -> bool:
        return len(self._current_graph_nodes) > 0

    # ── Execution ─────────────────────────────────────────────────

    async def execute(
        self,
        initial_data: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute the loaded graph with optional initial data.

        ``initial_data`` is a dict keyed by ``(block_id, port_id)``
        tuples (as strings: ``"block_id:port_id"``) for seeding Input
        nodes.  For example::

            {"block-input:frame-out": frame_array}

        Returns an ``ExecutionResult`` with all outputs from terminal nodes.
        """
        result = ExecutionResult()
        t0 = time.perf_counter()

        if not self._current_graph_nodes:
            result.success = False
            result.error = "No graph loaded"
            return result

        nodes = self._current_graph_nodes
        edges = self._current_graph_edges
        node_map = {n.id: n for n in nodes}

        # Topological sort
        sorted_ids, sort_err = topological_sort([n.id for n in nodes], edges)
        if sort_err:
            result.success = False
            result.error = sort_err
            return result

        # Data bus: keyed by "block_id:port_id"
        data_bus: dict[str, Any] = dict(initial_data or {})

        # Build reverse edge map: target_handle → source_handle
        # For each target node's input port, which source port feeds it
        incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for edge in edges:
            target_key = f"{edge.target}:{edge.target_handle}"
            source_key = f"{edge.source}:{edge.source_handle}"
            incoming[target_key].append((source_key, edge.source))

        # Track which nodes are terminal (have no outgoing edges)
        has_outgoing = {e.source for e in edges}

        # Execute nodes in topological order
        for node_id in sorted_ids:
            node = node_map[node_id]
            st = time.perf_counter()

            # Collect inputs from data bus
            node_inputs: dict[str, Any] = {}
            # Check all possible input ports for this node
            for key, sources in incoming.items():
                if key.startswith(f"{node_id}:"):
                    port_id = key.split(":", 1)[1]
                    for source_key, _source_id in sources:
                        if source_key in data_bus:
                            node_inputs[port_id] = data_bus[source_key]
                            break

            # Look up handler
            handler_key = node.handler_key or f"_type.{node.type}"
            handler = self._handlers.get(handler_key, _passthrough_handler)

            try:
                outputs = await _call_handler(handler, node_inputs, node.config)
            except Exception as exc:
                logger.error(
                    "graph_node_error node=%s type=%s error=%s",
                    node_id,
                    node.type,
                    str(exc),
                    exc_info=True,
                )
                result.success = False
                result.error = f"Error in node '{node.label}' ({node.type}): {exc}"
                result.elapsed_ms = (time.perf_counter() - t0) * 1000
                return result

            # Store outputs in data bus
            if isinstance(outputs, dict):
                for port_id, value in outputs.items():
                    data_bus[f"{node_id}:{port_id}"] = value
                result.node_outputs[node_id] = outputs
            else:
                # Single output — use the first output port name
                result.node_outputs[node_id] = {"_default": outputs}
                data_bus[f"{node_id}:_default"] = outputs

            result.executed_nodes.append(node_id)
            elapsed = (time.perf_counter() - st) * 1000
            result.stage_timings[node_id] = elapsed

            # Track terminal outputs
            if node_id not in has_outgoing:
                result.terminal_outputs[node_id] = result.node_outputs[node_id]

        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "graph_executed nodes=%d elapsed_ms=%.1f success=%s",
            len(result.executed_nodes),
            result.elapsed_ms,
            result.success,
        )
        return result

    # ── Describe current graph for agent context ──────────────────

    def describe_graph(self) -> str:
        """Return a human-readable description of the current graph.

        Used to inject graph context into the agent's system prompt
        so Gemma 4 E2B and Navarasa 2.0 understand the current pipeline.
        """
        if not self._current_graph_nodes:
            return "No pipeline graph is currently deployed."

        nodes = self._current_graph_nodes
        edges = self._current_graph_edges

        # Build adjacency description
        adj: dict[str, list[str]] = defaultdict(list)
        for e in edges:
            src_label = next((n.label for n in nodes if n.id == e.source), e.source)
            tgt_label = next((n.label for n in nodes if n.id == e.target), e.target)
            adj[src_label].append(tgt_label)

        lines = [
            f"Current deployed pipeline: {self._current_graph_json.get('project', {}).get('name', 'Unnamed')}"
            if self._current_graph_json
            else "Current deployed pipeline:",
            f"  Nodes ({len(nodes)}):",
        ]
        for n in nodes:
            config_summary = ", ".join(f"{k}={v}" for k, v in list(n.config.items())[:3])
            lines.append(f"    - [{n.category}] {n.label} (type={n.type}, config={{ {config_summary} }})")

        lines.append(f"  Connections ({len(edges)}):")
        for src, targets in adj.items():
            for tgt in targets:
                lines.append(f"    {src} → {tgt}")

        return "\n".join(lines)


# ── Default handler ───────────────────────────────────────────────


def _passthrough_handler(inputs: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Default handler: passes all inputs through as outputs."""
    return dict(inputs)


async def _call_handler(
    handler: HandlerFn,
    inputs: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    """Call a handler, supporting both sync and async handlers."""
    import asyncio
    import inspect

    if inspect.iscoroutinefunction(handler):
        return await handler(inputs, config)
    else:
        return await asyncio.get_event_loop().run_in_executor(
            None,
            handler,
            inputs,
            config,
        )
