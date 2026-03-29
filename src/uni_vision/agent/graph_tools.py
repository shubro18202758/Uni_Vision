"""Agent tools for inspecting and mutating the dynamic pipeline graph.

These tools give the Qwen 3.5 Manager Agent full awareness of the
user-defined pipeline topology so it can:
  * Describe what the current pipeline does.
  * Answer questions about connectivity and block configuration.
  * Suggest, add, remove, or reconnect blocks on behalf of the user.
  * Deploy / un-deploy graphs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from uni_vision.agent.tools import tool

logger = logging.getLogger(__name__)


# ── Helpers to grab engine / registry from context ───────────────


def _get_graph_engine(context: Any):
    """Retrieve the GraphEngine instance from the tool execution context."""
    if context is None:
        return None
    # The engine lives on the FastAPI app.state; the coordinator passes
    # a ToolExecutionContext that carries a reference.
    return getattr(context, "graph_engine", None)


def _get_block_registry(context: Any):
    """Retrieve the BlockRegistry instance from the tool execution context."""
    if context is None:
        return None
    return getattr(context, "block_registry", None)


# ══════════════════════════════════════════════════════════════════
# Read-only introspection tools
# ══════════════════════════════════════════════════════════════════


@tool(
    name="describe_pipeline_graph",
    description=(
        "Return a human-readable description of the currently deployed "
        "pipeline graph, including all blocks, their configuration, and "
        "how they are connected."
    ),
    param_descriptions={},
)
async def describe_pipeline_graph(*, context: Any = None) -> Dict[str, Any]:
    """Describe the current pipeline graph."""
    engine = _get_graph_engine(context)
    if engine is None:
        return {"error": "Graph engine not available"}
    if not engine.has_graph():
        return {"deployed": False, "description": "No pipeline graph is currently deployed."}
    return {
        "deployed": True,
        "description": engine.describe_graph(),
        "node_count": len(engine._current_nodes),
        "edge_count": len(engine._current_edges),
    }


@tool(
    name="list_available_blocks",
    description=(
        "List every block type available in the registry that the user "
        "can add to a pipeline. Returns type, label, category, and port "
        "information for each block."
    ),
    param_descriptions={
        "category": "Optional category filter (e.g. 'Detection', 'OCR'). Empty for all.",
    },
)
async def list_available_blocks(
    category: str = "",
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """List available block definitions from the registry."""
    registry = _get_block_registry(context)
    if registry is None:
        return {"error": "Block registry not available"}
    blocks = registry.get_all_blocks()
    if category:
        blocks = [b for b in blocks if b["category"].lower() == category.lower()]
    # Return a compact summary
    summary = [
        {
            "type": b["type"],
            "label": b["label"],
            "category": b["category"],
            "inputs": [p["name"] for p in b.get("inputs", [])],
            "outputs": [p["name"] for p in b.get("outputs", [])],
        }
        for b in blocks
    ]
    return {"count": len(summary), "blocks": summary}


@tool(
    name="get_block_details",
    description=(
        "Get full details for a specific block type including its "
        "configuration schema, ports, and defaults."
    ),
    param_descriptions={
        "block_type": "The block type identifier (e.g. 'yolo-detector', 'paddleocr').",
    },
)
async def get_block_details(
    block_type: str,
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Return details for one block type."""
    registry = _get_block_registry(context)
    if registry is None:
        return {"error": "Block registry not available"}
    defn = registry.get_block(block_type)
    if defn is None:
        return {"error": f"Unknown block type: {block_type}"}
    return defn


@tool(
    name="get_deployed_graph",
    description=(
        "Return the raw JSON of the currently deployed graph including "
        "all node IDs, types, configs, and edge connections."
    ),
    param_descriptions={},
)
async def get_deployed_graph(*, context: Any = None) -> Dict[str, Any]:
    """Return the deployed graph data."""
    engine = _get_graph_engine(context)
    if engine is None:
        return {"error": "Graph engine not available"}
    graph = engine.get_current_graph()
    if graph is None:
        return {"deployed": False, "graph": None}
    return {"deployed": True, "graph": graph}


# ══════════════════════════════════════════════════════════════════
# Graph mutation tools
# ══════════════════════════════════════════════════════════════════


@tool(
    name="validate_pipeline_graph",
    description=(
        "Validate a pipeline graph for structural issues (cycles, "
        "dangling edges, disconnected nodes) without deploying it."
    ),
    param_descriptions={
        "graph_json": (
            "JSON string of the graph to validate. Must have keys "
            "'blocks' (list of {id, type, ...}) and 'connections' "
            "(list of {source, target, sourceHandle, targetHandle})."
        ),
    },
)
async def validate_pipeline_graph(
    graph_json: str,
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Validate a graph without deploying."""
    import json as _json

    engine = _get_graph_engine(context)
    if engine is None:
        return {"error": "Graph engine not available"}
    try:
        data = _json.loads(graph_json)
    except (ValueError, TypeError) as exc:
        return {"valid": False, "error": f"Invalid JSON: {exc}"}

    blocks = data.get("blocks", [])
    connections = data.get("connections", [])
    issues = engine.validate_graph(blocks, connections)
    return {
        "valid": len(issues) == 0,
        "issues": [{"level": i.level, "message": i.message} for i in issues],
    }


@tool(
    name="deploy_pipeline_graph",
    description=(
        "Deploy a user-defined pipeline graph so that the backend "
        "executes it. Accepts the full graph JSON."
    ),
    param_descriptions={
        "graph_json": (
            "JSON string of the graph to deploy. Must have keys "
            "'project' ({name, version}), 'blocks' (list), and "
            "'connections' (list)."
        ),
    },
)
async def deploy_pipeline_graph(
    graph_json: str,
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Deploy a pipeline graph to the engine."""
    import json as _json

    engine = _get_graph_engine(context)
    registry = _get_block_registry(context)
    if engine is None:
        return {"error": "Graph engine not available"}
    try:
        data = _json.loads(graph_json)
    except (ValueError, TypeError) as exc:
        return {"success": False, "error": f"Invalid JSON: {exc}"}

    blocks = data.get("blocks", [])
    connections = data.get("connections", [])

    # Inject backend_handler from registry
    if registry is not None:
        for block in blocks:
            defn = registry.get_block(block.get("type", ""))
            if defn and "backend_handler" not in block:
                block["backend_handler"] = defn.get("backend_handler", block.get("type", ""))

    issues = engine.validate_graph(blocks, connections)
    errors = [i for i in issues if i.level == "error"]
    if errors:
        return {
            "success": False,
            "issues": [{"level": i.level, "message": i.message} for i in issues],
        }

    engine.load_graph(data)
    return {
        "success": True,
        "deployed_nodes": len(blocks),
        "deployed_edges": len(connections),
    }


@tool(
    name="clear_pipeline_graph",
    description="Remove the currently deployed pipeline graph.",
    param_descriptions={},
)
async def clear_pipeline_graph(*, context: Any = None) -> Dict[str, Any]:
    """Clear the deployed graph."""
    engine = _get_graph_engine(context)
    if engine is None:
        return {"error": "Graph engine not available"}
    engine.clear_graph()
    return {"success": True, "message": "Pipeline graph cleared."}


@tool(
    name="register_custom_block",
    description=(
        "Register a new custom block type in the backend registry so "
        "it becomes available for pipelines."
    ),
    param_descriptions={
        "block_json": (
            "JSON string defining the block. Must include: type, label, "
            "category, inputs (list of port objects), outputs (list of "
            "port objects), defaults (object), configSchema (list)."
        ),
    },
)
async def register_custom_block(
    block_json: str,
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Register a custom block definition."""
    import json as _json

    registry = _get_block_registry(context)
    if registry is None:
        return {"error": "Block registry not available"}
    try:
        defn = _json.loads(block_json)
    except (ValueError, TypeError) as exc:
        return {"success": False, "error": f"Invalid JSON: {exc}"}

    required = {"type", "label", "category"}
    missing = required - set(defn.keys())
    if missing:
        return {"success": False, "error": f"Missing fields: {missing}"}

    registry.register_block(defn)
    return {"success": True, "block_type": defn["type"]}


# ══════════════════════════════════════════════════════════════════
# Autonomous NL → Workflow design tool
# ══════════════════════════════════════════════════════════════════


@tool(
    name="design_workflow_from_nl",
    description=(
        "Design a complete pipeline workflow from a natural-language "
        "description. Supports 15 Indian languages + English. Returns "
        "the full ProjectGraph JSON ready for frontend rendering."
    ),
    param_descriptions={
        "description": (
            "Natural-language description of the desired pipeline. "
            "Can be in any of 16 supported languages (en, hi, te, ta, "
            "kn, ml, mr, bn, gu, pa, or, ur, as, kok, ne, sd)."
        ),
        "language": (
            "Two-letter language code (e.g. 'hi', 'te', 'en') or "
            "'auto' for automatic detection. Default: 'auto'."
        ),
    },
)
async def design_workflow_from_nl(
    description: str,
    language: str = "auto",
    *,
    context: Any = None,
) -> Dict[str, Any]:
    """Design a pipeline workflow from natural language."""
    from uni_vision.agent.workflow_designer import WorkflowDesigner

    if context is None:
        return {"error": "Execution context not available"}

    # Get LLM client from context's config
    llm_client = getattr(context, "_llm_client", None)
    navarasa_client = getattr(context, "_navarasa_client", None)

    # The coordinator attaches these to context at design time
    if llm_client is None:
        # Fallback: try to create from config
        return {"error": "LLM client not available in context"}

    designer = WorkflowDesigner(
        llm_client=llm_client,
        navarasa_client=navarasa_client,
    )
    result = await designer.design(description, language=language)

    if not result.success:
        return {
            "success": False,
            "error": result.error or "Workflow design failed",
            "phases": [
                {"name": p.name, "message": p.message, "success": p.success}
                for p in result.phases
            ],
        }

    return {
        "success": True,
        "graph": result.graph,
        "phases": [
            {"name": p.name, "message": p.message, "elapsed_ms": p.elapsed_ms}
            for p in result.phases
        ],
        "detected_language": result.detected_language,
        "english_input": result.english_input,
        "total_elapsed_ms": result.total_elapsed_ms,
    }
