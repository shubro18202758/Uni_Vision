import { useMemo } from "react";
import { useGraphStore } from "../../store/graphStore";
import { getCategoryColor } from "../../constants/categories";
import { GitBranch, Box, Workflow, Shield, Activity, Layers } from "lucide-react";

/**
 * Always-visible pipeline info overlay on the top-left of the canvas.
 * Shows topology stats, category breakdown, and data-flow summary.
 */
export function CanvasStatsOverlay() {
  const blocks = useGraphStore((s) => s.blocks);
  const connections = useGraphStore((s) => s.connections);
  const executionStates = useGraphStore((s) => s.executionStates);

  const stats = useMemo(() => {
    const categories: Record<string, number> = {};
    let configured = 0;
    let withInstruction = 0;

    for (const b of blocks) {
      categories[b.category] = (categories[b.category] || 0) + 1;
      const instr = String(b.config.instruction ?? "");
      if (instr.length > 0) withInstruction++;
      // Check if all non-instruction config fields are set
      configured++;
    }

    const running = Object.values(executionStates).filter((s) => s === "running").length;
    const success = Object.values(executionStates).filter((s) => s === "success").length;
    const errored = Object.values(executionStates).filter((s) => s === "error").length;

    // Find source nodes (no incoming edges) and sink nodes (no outgoing edges)
    const targets = new Set(connections.map((c) => c.target));
    const sources = new Set(connections.map((c) => c.source));
    const sourceNodes = blocks.filter((b) => !targets.has(b.id)).length;
    const sinkNodes = blocks.filter((b) => !sources.has(b.id)).length;

    return {
      categories,
      configured,
      withInstruction,
      running,
      success,
      errored,
      sourceNodes,
      sinkNodes,
    };
  }, [blocks, connections, executionStates]);

  if (blocks.length === 0) return null;

  const catEntries = Object.entries(stats.categories).sort(([, a], [, b]) => b - a);

  return (
    <div className="absolute top-14 left-3 z-10 pointer-events-none select-none">
      <div className="w-44 rounded-md border border-slate-700/30 bg-[#111827]/90 backdrop-blur p-2.5 pointer-events-auto">
        {/* Title */}
        <span className="text-[10px] font-medium text-slate-400 mb-2 block">Pipeline</span>

        {/* Topology row */}
        <div className="flex gap-3 mb-2">
          <div className="flex flex-col">
            <span className="text-[13px] font-semibold text-slate-200 tabular-nums">{blocks.length}</span>
            <span className="text-[9px] text-slate-500">Nodes</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[13px] font-semibold text-slate-200 tabular-nums">{connections.length}</span>
            <span className="text-[9px] text-slate-500">Edges</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[13px] font-semibold text-slate-200 tabular-nums">{catEntries.length}</span>
            <span className="text-[9px] text-slate-500">Stages</span>
          </div>
        </div>

        {/* Category breakdown */}
        <div className="space-y-1 mb-2">
          {catEntries.map(([cat, count]) => (
            <div key={cat} className="flex items-center gap-1.5">
              <div className="h-1.5 w-1.5 rounded-full shrink-0" style={{ backgroundColor: getCategoryColor(cat) }} />
              <span className="text-[10px] text-slate-400 flex-1 truncate">{cat}</span>
              <span className="text-[10px] tabular-nums text-slate-500">{count}</span>
            </div>
          ))}
        </div>

        {/* Data flow */}
        <div className="border-t border-slate-700/30 pt-1.5 space-y-0.5">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500">Sources</span>
            <span className="text-[10px] text-slate-400 tabular-nums">{stats.sourceNodes}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500">Sinks</span>
            <span className="text-[10px] text-slate-400 tabular-nums">{stats.sinkNodes}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500">Configured</span>
            <span className={`text-[10px] tabular-nums ${stats.withInstruction === blocks.length ? "text-emerald-400/70" : "text-amber-400/60"}`}>
              {stats.withInstruction}/{blocks.length}
            </span>
          </div>
        </div>

        {/* Execution status */}
        {(stats.running > 0 || stats.success > 0 || stats.errored > 0) && (
          <div className="border-t border-slate-700/30 pt-1.5 mt-1.5 flex gap-2">
            {stats.running > 0 && <span className="text-[10px] text-slate-400">{stats.running} running</span>}
            {stats.success > 0 && <span className="text-[10px] text-emerald-400/60">{stats.success} done</span>}
            {stats.errored > 0 && <span className="text-[10px] text-red-400/60">{stats.errored} failed</span>}
          </div>
        )}

        {/* Readiness */}
        <div className="mt-2 flex items-center gap-1.5">
          <span className={`h-1.5 w-1.5 rounded-full ${stats.withInstruction === blocks.length ? "bg-emerald-500/60" : "bg-amber-400/50"}`} />
          <span className={`text-[10px] ${stats.withInstruction === blocks.length ? "text-emerald-400/60" : "text-amber-400/50"}`}>
            {stats.withInstruction === blocks.length ? "Ready" : "Needs config"}
          </span>
        </div>
      </div>
    </div>
  );
}
