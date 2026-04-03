import { useGraphStore } from "../../store/graphStore";
import { useModelStore } from "../../store/modelStore";
import { useEffect, useState } from "react";
import { Activity, Box, Clock, Cpu } from "lucide-react";

function useUptime() {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(t);
  }, []);
  const h = Math.floor(elapsed / 3600);
  const m = Math.floor((elapsed % 3600) / 60);
  const s = elapsed % 60;
  return `${h > 0 ? `${h}h ` : ""}${m}m ${s}s`;
}

export function StatusBar() {
  const blocks = useGraphStore((s) => s.blocks);
  const connections = useGraphStore((s) => s.connections);
  const executionStates = useGraphStore((s) => s.executionStates);
  const activeModel = useModelStore((s) => s.activeModel);
  const phase = useModelStore((s) => s.phase);
  const uptime = useUptime();

  const runningCount = Object.values(executionStates).filter((s) => s === "running").length;
  const successCount = Object.values(executionStates).filter((s) => s === "success").length;
  const errorCount = Object.values(executionStates).filter((s) => s === "error").length;

  const modelLabel = activeModel ? "Gemma 4 E2B" : "None";
  const modelColor = activeModel ? "text-violet-400" : "text-slate-500";

  return (
    <footer className="flex h-7 shrink-0 items-center gap-3 border-t border-slate-800/60 bg-[#080e1a] px-4 text-[10px] font-normal text-slate-500 select-none">
      <div className="flex items-center gap-1" title="Nodes · Edges">
        <Box size={10} className="text-slate-600" />
        <span className="tabular-nums">{blocks.length} nodes</span>
        <span className="text-slate-700 mx-0.5">·</span>
        <span className="tabular-nums">{connections.length} edges</span>
      </div>

      <div className="h-3 w-px bg-slate-800/80" />

      {runningCount > 0 && (
        <span className="text-slate-400">{runningCount} running</span>
      )}
      {successCount > 0 && (
        <span className="text-emerald-500/60">{successCount} done</span>
      )}
      {errorCount > 0 && (
        <span className="text-red-400/70">{errorCount} failed</span>
      )}

      <div className="flex-1" />

      <div className="flex items-center gap-1" title="Active LLM">
        <Cpu size={10} className="text-slate-600" />
        <span className={modelColor}>{modelLabel}</span>
      </div>

      <div className="h-3 w-px bg-slate-800/80" />

      <span className="tabular-nums" title="VRAM">8 GB</span>

      <div className="h-3 w-px bg-slate-800/80" />

      <span className="tabular-nums" title="Uptime">{uptime}</span>

      <div className="h-3 w-px bg-slate-800/80" />

      <div className="flex items-center gap-1">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500/70" />
        <span className="text-slate-500">Connected</span>
      </div>
    </footer>
  );
}
