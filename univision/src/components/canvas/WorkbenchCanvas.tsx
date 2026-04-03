import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  Background,
  BackgroundVariant,
  Connection,
  Controls,
  Edge,
  MarkerType,
  Node,
  addEdge,
  useReactFlow,
} from "reactflow";
import { BlockNode } from "../blocks/BlockNode";
import { GradientEdge } from "../edges/GradientEdge";
import { QuickAddMenu } from "./QuickAddMenu";
import { NodeContextMenu } from "./NodeContextMenu";
import { useGraphStore } from "../../store/graphStore";
import { useUiStore } from "../../store/uiStore";
import { getBlockDefinition } from "../../lib/blockRegistry";
import { applyLayout, getEdgeType } from "../../lib/autoLayout";
import { CanvasToolbar } from "./CanvasToolbar";
import { WorkbenchMiniMap } from "./WorkbenchMiniMap";
import { CanvasZoomControls } from "./CanvasZoomControls";
import { ExecutionDashboard } from "./ExecutionDashboard";
import { CanvasStatsOverlay } from "./CanvasStatsOverlay";

const nodeTypes = {
  pipelineBlock: BlockNode,
};

const edgeTypes = {
  gradientEdge: GradientEdge,
};

interface ContextMenuState {
  blockId: string;
  x: number;
  y: number;
}

export function WorkbenchCanvas() {
  const blocks = useGraphStore((state) => state.blocks);
  const connections = useGraphStore((state) => state.connections);
  const graphVersion = useGraphStore((state) => state.graphVersion);
  const addBlock = useGraphStore((state) => state.addBlock);
  const addConnectionToStore = useGraphStore((state) => state.addConnection);
  const removeConnection = useGraphStore((state) => state.removeConnection);
  const removeBlock = useGraphStore((state) => state.removeBlock);
  const duplicateBlock = useGraphStore((state) => state.duplicateBlock);
  const selectedBlockId = useGraphStore((state) => state.selectedBlockId);
  const setSelectedBlockId = useGraphStore((state) => state.setSelectedBlockId);
  const updateBlockPositions = useGraphStore((state) => state.updateBlockPositions);
  const layoutMode = useUiStore((state) => state.layoutMode);
  const quickAdd = useUiStore((state) => state.quickAdd);
  const openQuickAdd = useUiStore((state) => state.openQuickAdd);
  const closeQuickAdd = useUiStore((state) => state.closeQuickAdd);
  const { fitView, screenToFlowPosition } = useReactFlow();

  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const lastTopologyRef = useRef<string>("");
  const prevGraphVersionRef = useRef(graphVersion);

  // Listen for node-context-menu custom events from BlockNode
  useEffect(() => {
    const handler = (e: Event) => {
      const { blockId, x, y } = (e as CustomEvent).detail;
      setContextMenu({ blockId, x, y });
    };
    window.addEventListener("node-context-menu", handler);
    return () => window.removeEventListener("node-context-menu", handler);
  }, []);

  // Keyboard shortcuts: Delete, Backspace, Ctrl+D
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't intercept if user is typing in an input/textarea
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      if ((e.key === "Delete" || e.key === "Backspace") && selectedBlockId) {
        e.preventDefault();
        removeBlock(selectedBlockId);
      }
      if (e.key === "d" && (e.ctrlKey || e.metaKey) && selectedBlockId) {
        e.preventDefault();
        duplicateBlock(selectedBlockId);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedBlockId, removeBlock, duplicateBlock]);

  const nodes = useMemo<Node[]>(
    () =>
      blocks.map((block) => ({
        id: block.id,
        type: "pipelineBlock",
        position: block.position,
        data: block,
        // Explicit dimensions help React Flow's fitView calculate viewport before DOM measurement finishes
        width: 260,
        height: 160,
      })),
    [blocks],
  );

  const edgeType = getEdgeType(layoutMode);

  const edges = useMemo<Edge[]>(
    () =>
      connections.map((connection) => {
        const sourceType = getBlockDefinition(blocks.find((block) => block.id === connection.source)?.type ?? "")?.outputs.find(
          (port) => port.id === connection.sourceHandle,
        )?.type;

        const color = sourceType === "text" ? "#67e8f9" : "#22d3ee";

        return {
          id: connection.id,
          source: connection.source,
          sourceHandle: connection.sourceHandle,
          target: connection.target,
          targetHandle: connection.targetHandle,
          type: edgeType,
          animated: false,
          markerEnd: { type: MarkerType.ArrowClosed, color, width: 14, height: 14 },
          style: { stroke: color, strokeWidth: 1.5, opacity: 0.65 },
        };
      }),
    [blocks, connections, edgeType],
  );

  // Stable topology fingerprint — changes when blocks or connections change identity
  const topoFingerprint = useMemo(
    () =>
      blocks.map((b) => b.id).sort().join(",") +
      "|" +
      connections.map((c) => `${c.source}-${c.target}`).sort().join(","),
    [blocks, connections],
  );

  // Auto-layout when topology changes (block ids or connections) and mode is not manual.
  // Exception: a full graph replacement (graphVersion change) ALWAYS triggers
  // hierarchical-lr layout so that designed/loaded workflows look clean.
  useEffect(() => {
    const isNewGraph = graphVersion !== prevGraphVersionRef.current;
    if (isNewGraph) prevGraphVersionRef.current = graphVersion;

    // For incremental edits respect manual mode; for full replacements always layout
    if (!isNewGraph && layoutMode === "manual") return;
    if (!isNewGraph && topoFingerprint === lastTopologyRef.current) return;
    lastTopologyRef.current = topoFingerprint;
    if (blocks.length === 0) return;

    const algo = isNewGraph ? "hierarchical-lr" : layoutMode;
    const laid = applyLayout(algo, nodes, edges);
    const posMap = new Map<string, { x: number; y: number }>();
    laid.forEach((n) => posMap.set(n.id, n.position));
    updateBlockPositions(posMap);
    // Fit viewport after React Flow processes new positions
    setTimeout(() => fitView({ padding: 0.15, duration: 400 }), 200);
  }, [graphVersion, layoutMode, topoFingerprint, nodes, edges, blocks.length, updateBlockPositions, fitView]);

  // Safety: fit view on initial mount with delay to ensure nodes are measured
  useEffect(() => {
    if (blocks.length === 0) return;
    const t = setTimeout(() => fitView({ padding: 0.15, duration: 400 }), 500);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target || !connection.sourceHandle || !connection.targetHandle) return;
      const edge = addEdge({ ...connection, id: `edge-${crypto.randomUUID()}` }, [])[0];
      if (!edge.source || !edge.target || !edge.sourceHandle || !edge.targetHandle) return;
      addConnectionToStore({ id: edge.id, source: edge.source, sourceHandle: edge.sourceHandle, target: edge.target, targetHandle: edge.targetHandle });
    },
    [addConnectionToStore],
  );

  return (
    <div
      ref={wrapperRef}
      className="relative h-full w-full overflow-hidden"
      onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "move"; }}
      onDrop={(event) => {
        event.preventDefault();
        const type = event.dataTransfer.getData("application/univision-block");
        if (!type) return;
        const flowPos = screenToFlowPosition({ x: event.clientX, y: event.clientY });
        addBlock(type, flowPos);
      }}
    >
      <CanvasToolbar />
      <ExecutionDashboard />
      <CanvasStatsOverlay />
      {quickAdd && (
        <QuickAddMenu
          position={quickAdd.screen}
          canvasPosition={quickAdd.canvas}
          onClose={closeQuickAdd}
        />
      )}
      {contextMenu && (
        <NodeContextMenu
          blockId={contextMenu.blockId}
          x={contextMenu.x}
          y={contextMenu.y}
          onClose={() => setContextMenu(null)}
        />
      )}
      <ReactFlow
        key={graphVersion}
        fitView
        fitViewOptions={{ padding: 0.15, duration: 400 }}
        edges={edges}
        nodes={nodes}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onConnect={onConnect}
        onInit={(instance) => {
          // Double-ensure fitView fires after React Flow has measured nodes
          setTimeout(() => instance.fitView({ padding: 0.15, duration: 400 }), 300);
        }}
        onEdgeClick={(_, edge) => removeConnection(edge.id)}
        onNodeClick={(_, node) => setSelectedBlockId(node.id)}
        onNodeDragStop={(_, node) => {
          updateBlockPositions(new Map([[node.id, node.position]]));
        }}
        onPaneClick={() => { closeQuickAdd(); setContextMenu(null); }}
        onDoubleClick={(event) => {
          // Double-click on pane opens quick-add menu
          const target = event.target as HTMLElement;
          if (target.closest(".react-flow__node")) return;
          const flowPos = screenToFlowPosition({ x: event.clientX, y: event.clientY });
          openQuickAdd({ x: event.clientX, y: event.clientY }, flowPos);
        }}
        proOptions={{ hideAttribution: true }}
        style={{ backgroundColor: "#0a1628" }}
      >
        {/* SVG defs for edge particle glow */}
        <svg width={0} height={0} style={{ position: "absolute" }}>
          <defs>
            <filter id="particle-glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <pattern id="canvas-grid-cross" width="40" height="40" patternUnits="userSpaceOnUse">
              <line x1="20" y1="16" x2="20" y2="24" stroke="rgba(34,211,238,0.07)" strokeWidth="0.5" />
              <line x1="16" y1="20" x2="24" y2="20" stroke="rgba(34,211,238,0.07)" strokeWidth="0.5" />
            </pattern>
            {/* Large grid lines every 200px */}
            <pattern id="canvas-major-grid" width="200" height="200" patternUnits="userSpaceOnUse">
              <line x1="0" y1="0" x2="200" y2="0" stroke="rgba(34,211,238,0.025)" strokeWidth="0.5" />
              <line x1="0" y1="0" x2="0" y2="200" stroke="rgba(34,211,238,0.025)" strokeWidth="0.5" />
            </pattern>
          </defs>
        </svg>
        <Background id="dots" gap={20} color="rgba(34, 211, 238, 0.06)" variant={BackgroundVariant.Dots} size={1} />
        <Controls className="bg-slate-800/90 backdrop-blur-sm border-slate-700 fill-slate-300 rounded-md shadow-lg" showInteractive={false} showFitView={false} showZoom={false} />
        <WorkbenchMiniMap />
        <CanvasZoomControls />
      </ReactFlow>
    </div>
  );
}
