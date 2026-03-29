import { useState, useMemo } from "react";
import { Network, Zap, CircleDot, ArrowRight, ArrowDown, LayoutGrid, Hand, RotateCcw, Plus, Trash2, Copy, ShieldCheck, AlertTriangle } from "lucide-react";
import { useUiStore } from "../../store/uiStore";
import { useGraphStore } from "../../store/graphStore";
import { getBlockDefinition } from "../../lib/blockRegistry";
import { applyLayout, LAYOUT_ALGORITHMS, type LayoutAlgorithm } from "../../lib/autoLayout";
import type { Edge, Node } from "reactflow";

const ICONS: Record<LayoutAlgorithm, React.ReactNode> = {
  mindmap: <Network size={13} />,
  force: <Zap size={13} />,
  radial: <CircleDot size={13} />,
  "hierarchical-lr": <ArrowRight size={13} />,
  "hierarchical-tb": <ArrowDown size={13} />,
  grid: <LayoutGrid size={13} />,
  manual: <Hand size={13} />,
};

export function CanvasToolbar() {
  const layoutMode = useUiStore((s) => s.layoutMode);
  const setLayoutMode = useUiStore((s) => s.setLayoutMode);
  const openQuickAdd = useUiStore((s) => s.openQuickAdd);
  const blocks = useGraphStore((s) => s.blocks);
  const connections = useGraphStore((s) => s.connections);
  const selectedBlockId = useGraphStore((s) => s.selectedBlockId);
  const removeBlock = useGraphStore((s) => s.removeBlock);
  const duplicateBlock = useGraphStore((s) => s.duplicateBlock);
  const updateBlockPositions = useGraphStore((s) => s.updateBlockPositions);
  const [showHealth, setShowHealth] = useState(false);

  /* ── Pipeline health check ── */
  const health = useMemo(() => {
    const warnings: string[] = [];
    if (blocks.length === 0) return { ok: true, warnings };

    // Orphan blocks (no connections in or out)
    const connectedIds = new Set<string>();
    connections.forEach((c) => { connectedIds.add(c.source); connectedIds.add(c.target); });
    const orphans = blocks.filter((b) => !connectedIds.has(b.id));
    if (orphans.length > 0 && blocks.length > 1)
      warnings.push(`${orphans.length} orphan block${orphans.length > 1 ? "s" : ""} with no connections`);

    // Blocks missing instructions
    const noInstruction = blocks.filter((b) => {
      const def = getBlockDefinition(b.type);
      const hasInstrField = def?.configSchema.some((f) => f.key === "instruction");
      return hasInstrField && !b.config.instruction;
    });
    if (noInstruction.length > 0)
      warnings.push(`${noInstruction.length} block${noInstruction.length > 1 ? "s" : ""} without instructions`);

    // No Input block
    if (!blocks.some((b) => b.category === "Input"))
      warnings.push("No Input block — pipeline has no source");

    return { ok: warnings.length === 0, warnings };
  }, [blocks, connections]);

  function reLayout(algo?: LayoutAlgorithm) {
    const a = algo ?? layoutMode;
    if (a === "manual" || blocks.length === 0) return;
    const nodes: Node[] = blocks.map((b) => ({ id: b.id, type: "pipelineBlock", position: b.position, data: b }));
    const edges: Edge[] = connections.map((c) => ({ id: c.id, source: c.source, sourceHandle: c.sourceHandle, target: c.target, targetHandle: c.targetHandle }));
    const laid = applyLayout(a, nodes, edges);
    const posMap = new Map<string, { x: number; y: number }>();
    laid.forEach((n) => posMap.set(n.id, n.position));
    updateBlockPositions(posMap);
  }

  return (
    <div className="absolute left-3 top-3 z-10 flex items-center gap-0.5 rounded-md bg-[#111827]/95 backdrop-blur px-1 py-0.5 border border-slate-700/40">
      <button
        title="Add new block (double-click canvas)"
        onClick={() => openQuickAdd({ x: 400, y: 300 }, { x: 200, y: 200 })}
        className="flex items-center rounded px-1.5 py-1 text-slate-400 hover:bg-slate-700/40 hover:text-slate-200 transition-colors"
      >
        <Plus size={13} />
      </button>
      <button
        title="Duplicate selected (Ctrl+D)"
        disabled={!selectedBlockId}
        onClick={() => selectedBlockId && duplicateBlock(selectedBlockId)}
        className="flex items-center rounded px-1.5 py-1 text-slate-400 hover:bg-slate-700/40 hover:text-slate-200 transition-colors disabled:opacity-20"
      >
        <Copy size={13} />
      </button>
      <button
        title="Delete selected (Del)"
        disabled={!selectedBlockId}
        onClick={() => selectedBlockId && removeBlock(selectedBlockId)}
        className="flex items-center rounded px-1.5 py-1 text-slate-400 hover:bg-slate-700/40 hover:text-slate-200 transition-colors disabled:opacity-20"
      >
        <Trash2 size={13} />
      </button>

      <div className="mx-0.5 h-4 w-px bg-slate-700/50" />

      {/* Layout algorithms — icon only */}
      {LAYOUT_ALGORITHMS.map(({ value, label }) => (
        <button
          key={value}
          title={label}
          onClick={() => { setLayoutMode(value); reLayout(value); }}
          className={`rounded p-1 transition-colors ${
            layoutMode === value
              ? "bg-slate-700/60 text-slate-200"
              : "text-slate-500 hover:bg-slate-700/30 hover:text-slate-300"
          }`}
        >
          {ICONS[value]}
        </button>
      ))}

      <div className="mx-0.5 h-4 w-px bg-slate-700/50" />
      <button
        title="Re-apply layout"
        onClick={() => reLayout()}
        className="rounded p-1 text-slate-500 hover:bg-slate-700/30 hover:text-slate-300 transition-colors"
      >
        <RotateCcw size={12} />
      </button>
      {blocks.length > 0 && (
        <>
          <div className="mx-0.5 h-4 w-px bg-slate-700/50" />
          <span className="text-[10px] text-slate-600 tabular-nums select-none px-0.5">
            {blocks.length}n·{connections.length}e
          </span>
          <div className="relative">
            <button
              title={health.ok ? "Pipeline looks good" : `${health.warnings.length} issue(s) found`}
              onClick={() => setShowHealth(!showHealth)}
              className={`rounded p-1 transition-colors ${
                health.ok
                  ? "text-emerald-500/50 hover:bg-emerald-500/10"
                  : "text-amber-400/70 hover:bg-amber-500/10"
              }`}
            >
              {health.ok ? <ShieldCheck size={12} /> : <AlertTriangle size={12} />}
            </button>
            {showHealth && !health.ok && (
              <div className="absolute top-full left-0 mt-1 w-56 rounded-md bg-[#111827] border border-slate-700/50 p-2 shadow-lg z-50">
                <p className="text-[10px] font-medium text-amber-400/80 mb-1">Pipeline Warnings</p>
                {health.warnings.map((w) => (
                  <p key={w} className="text-[10px] text-slate-400 leading-relaxed">• {w}</p>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
