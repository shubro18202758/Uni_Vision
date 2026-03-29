import { useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useGraphStore } from "../../store/graphStore";
import { useModelStore } from "../../store/modelStore";
import { getCategoryColor } from "../../constants/categories";
import {
  Activity, CheckCircle2, XCircle, Clock, Loader2, Play, Timer, BarChart3, Cpu,
} from "lucide-react";

export function ExecutionDashboard() {
  const blocks = useGraphStore((s) => s.blocks);
  const executionStates = useGraphStore((s) => s.executionStates);
  const phase = useModelStore((s) => s.phase);

  const stats = useMemo(() => {
    const s = { idle: 0, queued: 0, running: 0, success: 0, error: 0, total: blocks.length };
    blocks.forEach((b) => {
      const state = executionStates[b.id] ?? "idle";
      if (state in s) (s as Record<string, number>)[state]++;
    });
    return s;
  }, [blocks, executionStates]);

  const isExecuting = stats.running > 0 || stats.queued > 0;
  const progressPct = stats.total > 0 ? Math.round(((stats.success + stats.error) / stats.total) * 100) : 0;

  // Only show when pipeline is actively executing
  if (!isExecuting && stats.success === 0 && stats.error === 0) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className="absolute top-14 right-3 z-20 w-64 rounded-xl border border-slate-700/40 bg-[#0b1525]/95 backdrop-blur-md shadow-2xl shadow-black/30"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800/50">
          <div className="flex items-center gap-2">
            <BarChart3 size={13} className="text-slate-400" />
            <span className="text-[10px] font-bold text-slate-200 uppercase tracking-wider">
              Pipeline Execution
            </span>
          </div>
          {isExecuting && (
            <span className="flex items-center gap-1 text-[8px] text-emerald-400 font-bold animate-pulse">
              <Activity size={10} /> LIVE
            </span>
          )}
        </div>

        {/* Progress bar */}
        <div className="px-3 pt-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[8px] text-slate-500 font-medium">Overall Progress</span>
            <span className="text-[9px] text-slate-300 font-bold tabular-nums">{progressPct}%</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{
                background: stats.error > 0
                  ? "linear-gradient(90deg, #3b82f6, #f43f5e)"
                  : "linear-gradient(90deg, #3b82f6, #60a5fa)",
              }}
              initial={{ width: 0 }}
              animate={{ width: `${progressPct}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-4 gap-1 px-3 py-2">
          <div className="flex flex-col items-center py-1 rounded-md bg-slate-800/30">
            <Loader2 size={10} className={`text-blue-400 ${stats.running > 0 ? "animate-spin" : ""}`} />
            <span className="text-[10px] font-bold text-blue-300 tabular-nums mt-0.5">{stats.running}</span>
            <span className="text-[6px] text-slate-500 uppercase">Active</span>
          </div>
          <div className="flex flex-col items-center py-1 rounded-md bg-slate-800/30">
            <Play size={10} className="text-slate-400" />
            <span className="text-[10px] font-bold text-slate-300 tabular-nums mt-0.5">{stats.queued}</span>
            <span className="text-[6px] text-slate-500 uppercase">Queued</span>
          </div>
          <div className="flex flex-col items-center py-1 rounded-md bg-slate-800/30">
            <CheckCircle2 size={10} className="text-emerald-400" />
            <span className="text-[10px] font-bold text-emerald-300 tabular-nums mt-0.5">{stats.success}</span>
            <span className="text-[6px] text-slate-500 uppercase">Done</span>
          </div>
          <div className="flex flex-col items-center py-1 rounded-md bg-slate-800/30">
            <XCircle size={10} className="text-red-400" />
            <span className="text-[10px] font-bold text-red-300 tabular-nums mt-0.5">{stats.error}</span>
            <span className="text-[6px] text-slate-500 uppercase">Failed</span>
          </div>
        </div>

        {/* Per-node execution list */}
        <div className="px-3 pb-2 max-h-40 overflow-y-auto">
          <p className="text-[7px] text-slate-600 uppercase tracking-wider font-bold mb-1">Node Status</p>
          {blocks.map((block) => {
            const state = executionStates[block.id] ?? "idle";
            const catColor = getCategoryColor(block.category);
            return (
              <div
                key={block.id}
                className="flex items-center gap-2 py-0.5"
              >
                <div
                  className="h-1.5 w-1.5 rounded-full shrink-0"
                  style={{ backgroundColor: catColor }}
                />
                <span className="text-[8px] text-slate-400 truncate flex-1">{block.label}</span>
                {state === "running" && (
                  <Loader2 size={8} className="text-blue-400 animate-spin shrink-0" />
                )}
                {state === "success" && (
                  <CheckCircle2 size={8} className="text-emerald-400 shrink-0" />
                )}
                {state === "error" && (
                  <XCircle size={8} className="text-red-400 shrink-0" />
                )}
                {state === "queued" && (
                  <Clock size={8} className="text-slate-500 shrink-0" />
                )}
                {state === "idle" && (
                  <div className="h-1 w-1 rounded-full bg-slate-700 shrink-0" />
                )}
              </div>
            );
          })}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
