"""Pipeline graph API — dynamic block registry and graph deployment.

Exposes the block palette, categories, port types, and allows the
frontend (or any client) to deploy / fetch / clear the live pipeline
graph.  These routes replace all hardcoded pipeline topology.

Routes
------
GET  /api/pipeline/blocks        – All available block definitions
GET  /api/pipeline/categories    – Category → colour map
GET  /api/pipeline/port-types    – Port type → colour map
POST /api/pipeline/blocks        – Register a custom block at runtime
POST /api/pipeline/graph         – Deploy a graph (validate, store, configure engine)
GET  /api/pipeline/graph         – Return the currently deployed graph
DELETE /api/pipeline/graph       – Clear the deployed graph
POST /api/pipeline/graph/validate – Validate a graph without deploying
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


# ── Pydantic models ──────────────────────────────────────────────

class PortDefinitionModel(BaseModel):
    id: str
    name: str
    type: str
    direction: str


class ConfigFieldModel(BaseModel):
    key: str
    label: str
    type: str
    placeholder: Optional[str] = None
    required: Optional[bool] = None
    min: Optional[float] = None
    max: Optional[float] = None
    options: Optional[List[Dict[str, str]]] = None


class BlockDefinitionModel(BaseModel):
    type: str = Field(..., min_length=1, max_length=100)
    label: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    category: str = "Utility"
    inputs: List[PortDefinitionModel] = Field(default_factory=list)
    outputs: List[PortDefinitionModel] = Field(default_factory=list)
    defaults: Dict[str, Any] = Field(default_factory=dict)
    configSchema: List[ConfigFieldModel] = Field(default_factory=list)
    backend_handler: str = ""


class PositionModel(BaseModel):
    x: float
    y: float


class GraphBlockModel(BaseModel):
    id: str
    type: str
    label: str = ""
    category: str = "Utility"
    position: PositionModel = Field(default_factory=lambda: PositionModel(x=0, y=0))
    config: Dict[str, Any] = Field(default_factory=dict)
    status: str = "idle"


class GraphConnectionModel(BaseModel):
    id: str
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


class ProjectModel(BaseModel):
    name: str = "Untitled Pipeline"
    version: str = "0.1.0"


class ProjectGraphModel(BaseModel):
    project: ProjectModel = Field(default_factory=ProjectModel)
    blocks: List[GraphBlockModel] = Field(default_factory=list)
    connections: List[GraphConnectionModel] = Field(default_factory=list)


class ValidationIssueModel(BaseModel):
    id: str = ""
    level: str = "error"
    message: str = ""
    blockId: Optional[str] = None


class DeployResponse(BaseModel):
    success: bool
    issues: List[ValidationIssueModel] = Field(default_factory=list)
    deployed_nodes: int = 0
    deployed_edges: int = 0


# ── Helpers to get engine / registry from app state ───────────────

def _get_registry(request: Request):
    reg = getattr(request.app.state, "block_registry", None)
    if reg is None:
        raise HTTPException(status_code=503, detail="Block registry not initialised")
    return reg


def _get_engine(request: Request):
    eng = getattr(request.app.state, "graph_engine", None)
    if eng is None:
        raise HTTPException(status_code=503, detail="Graph engine not initialised")
    return eng


# ── Block palette routes ──────────────────────────────────────────

@router.get("/blocks", response_model=List[Dict[str, Any]])
async def list_blocks(request: Request):
    """Return all registered block definitions for the palette."""
    registry = _get_registry(request)
    return registry.get_all_blocks()


@router.get("/categories")
async def list_categories(request: Request):
    """Return the category → hex-colour map."""
    registry = _get_registry(request)
    return registry.get_categories()


@router.get("/port-types")
async def list_port_types(request: Request):
    """Return the port-type → hex-colour map."""
    registry = _get_registry(request)
    return registry.get_port_types()


@router.post("/blocks", status_code=201)
async def register_block(defn: BlockDefinitionModel, request: Request):
    """Register a custom block at runtime."""
    registry = _get_registry(request)
    block_dict = defn.model_dump()
    registry.register_block(block_dict)
    logger.info("custom_block_registered type=%s", defn.type)
    return {"registered": defn.type}


# ── Graph deployment routes ───────────────────────────────────────

@router.post("/graph", response_model=DeployResponse)
async def deploy_graph(graph: ProjectGraphModel, request: Request):
    """Validate and deploy a pipeline graph.

    The graph is stored in the engine's state and will be used for
    all subsequent inference passes.
    """
    engine = _get_engine(request)
    graph_dict = graph.model_dump()

    # Inject backend_handler from registry into each block
    registry = _get_registry(request)
    for block in graph_dict["blocks"]:
        defn = registry.get_block(block["type"])
        if defn:
            block["backend_handler"] = defn.get("backend_handler", "")

    issues = engine.load_graph(graph_dict)

    has_errors = any(i.get("level") == "error" for i in issues)
    return DeployResponse(
        success=not has_errors,
        issues=[ValidationIssueModel(**i) for i in issues],
        deployed_nodes=len(graph_dict["blocks"]) if not has_errors else 0,
        deployed_edges=len(graph_dict["connections"]) if not has_errors else 0,
    )


@router.get("/graph")
async def get_deployed_graph(request: Request):
    """Return the currently deployed graph, or 404 if none."""
    engine = _get_engine(request)
    graph = engine.get_current_graph()
    if graph is None:
        raise HTTPException(status_code=404, detail="No graph currently deployed")
    return graph


@router.delete("/graph")
async def clear_graph(request: Request):
    """Clear the currently deployed graph."""
    engine = _get_engine(request)
    engine.clear_graph()
    return {"cleared": True}


@router.post("/graph/validate", response_model=List[ValidationIssueModel])
async def validate_graph_endpoint(graph: ProjectGraphModel, request: Request):
    """Validate a graph without deploying it."""
    from uni_vision.orchestrator.graph_engine import (
        GraphEdge,
        GraphNode,
        validate_graph,
    )

    nodes = [
        GraphNode(
            id=b.id,
            type=b.type,
            label=b.label,
            category=b.category,
            config=b.config,
        )
        for b in graph.blocks
    ]
    edges = [
        GraphEdge(
            id=c.id,
            source=c.source,
            source_handle=c.sourceHandle,
            target=c.target,
            target_handle=c.targetHandle,
        )
        for c in graph.connections
    ]

    issues = validate_graph(nodes, edges)
    return [ValidationIssueModel(**i) for i in issues]


# ── Model routing routes ──────────────────────────────────────────

class ModelStateResponse(BaseModel):
    phase: str
    active_model: str
    navarasa_loaded: bool
    qwen_loaded: bool


class ActivateModelRequest(BaseModel):
    phase: str = Field(..., pattern=r"^(pre_launch|post_launch)$")


def _get_model_router(request: Request):
    router_inst = getattr(request.app.state, "model_router", None)
    if router_inst is None:
        raise HTTPException(status_code=503, detail="Model router not initialised")
    return router_inst


@router.get("/model-state", response_model=ModelStateResponse)
async def get_model_state(request: Request):
    """Return the current Ollama model routing state."""
    mr = _get_model_router(request)
    state = mr.get_state()
    return ModelStateResponse(
        phase=state.phase.value,
        active_model=state.active_model,
        navarasa_loaded=state.navarasa_loaded,
        qwen_loaded=state.qwen_loaded,
    )


@router.post("/activate-model", response_model=ModelStateResponse)
async def activate_model(body: ActivateModelRequest, request: Request):
    """Switch the active Ollama model phase.

    - ``pre_launch``: Activate Navarasa, unload Qwen (design phase).
    - ``post_launch``: Activate Qwen, unload Navarasa (pipeline phase).
    """
    mr = _get_model_router(request)

    if body.phase == "pre_launch":
        state = await mr.activate_navarasa()
    else:
        state = await mr.activate_qwen()

    return ModelStateResponse(
        phase=state.phase.value,
        active_model=state.active_model,
        navarasa_loaded=state.navarasa_loaded,
        qwen_loaded=state.qwen_loaded,
    )
