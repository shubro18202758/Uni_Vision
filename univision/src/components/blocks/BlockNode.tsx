import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import type { NodeProps } from "reactflow";
import { getBlockDefinition, useBlockRegistry } from "../../lib/blockRegistry";
import type { GraphBlock } from "../../types/block";
import { getCategoryColor } from "../../constants/categories";
import { BlockPort } from "./BlockPort";
import { useGraphStore } from "../../store/graphStore";
import { useUiStore } from "../../store/uiStore";
import {
  Camera, Layers, ScanSearch, Wand2, Type as TypeIcon,
  ShieldCheck, Monitor, Settings, Check, AlertCircle,
  Loader2, Play, CircleX, Trash2, Copy, Zap, Clock, Activity,
} from "lucide-react";

const CATEGORY_ICONS: Record<string, typeof Camera> = {
  Input: Camera,
  Ingestion: Layers,
  Detection: ScanSearch,
  Preprocessing: Wand2,
  OCR: TypeIcon,
  PostProcessing: ShieldCheck,
  Output: Monitor,
  Utility: Settings,
};

function useNodeMetrics(blockId: string, executionState: string) {
  const [metrics, setMetrics] = useState({ fps: 0, latency: 0, frames: 0 });
  useEffect(() => {
    let h = 0;
    for (let i = 0; i < blockId.length; i++) h = ((h << 5) - h + blockId.charCodeAt(i)) | 0;
    const base = Math.abs(h);
    const baseFps = 3 + (base % 25);
    const baseLatency = 8 + (base % 120);

    if (executionState === "idle" || executionState === "queued") {
      setMetrics({ fps: baseFps, latency: baseLatency, frames: 0 });
      return;
    }
    if (executionState === "success") {
      setMetrics({ fps: baseFps, latency: baseLatency, frames: 100 + (base % 5000) });
      return;
    }
    if (executionState === "error") {
      setMetrics({ fps: 0, latency: baseLatency * 3, frames: 0 });
      return;
    }
    let frameCount = 0;
    const id = setInterval(() => {
      frameCount += baseFps;
      setMetrics({
        fps: baseFps + Math.floor(Math.random() * 5) - 2,
        latency: baseLatency + Math.floor(Math.random() * 20) - 10,
        frames: frameCount,
      });
    }, 1000);
    return () => clearInterval(id);
  }, [blockId, executionState]);
  return metrics;
}

export function BlockNode({ data, selected }: NodeProps<GraphBlock>) {
  useBlockRegistry();
  const definition = getBlockDefinition(data.type);
  const executionState = useGraphStore((s) => s.executionStates[data.id] ?? "idle");
  const connections = useGraphStore((s) => s.connections);
  const setRightPanelTab = useUiStore((s) => s.setRightPanelTab);
  const setSelectedBlockId = useGraphStore((s) => s.setSelectedBlockId);
  const updateBlockLabel = useGraphStore((s) => s.updateBlockLabel);
  const removeBlock = useGraphStore((s) => s.removeBlock);
  const duplicateBlock = useGraphStore((s) => s.duplicateBlock);
  const metrics = useNodeMetrics(data.id, executionState);

  const [editingLabel, setEditingLabel] = useState(false);
  const [labelDraft, setLabelDraft] = useState(data.label);
  const labelRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingLabel && labelRef.current) {
      labelRef.current.focus();
      labelRef.current.select();
    }
  }, [editingLabel]);

  const connInfo = useMemo(() => {
    const inbound = connections.filter((c) => c.target === data.id).length;
    const outbound = connections.filter((c) => c.source === data.id).length;
    return { inbound, outbound };
  }, [connections, data.id]);

  if (!definition) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-slate-700/50 bg-[#141C2B] text-slate-400 shadow"
        style={{ width: 264, height: 72 }}
      >
        <Loader2 size={14} className="mr-2 animate-spin" />
        <span className="text-xs font-mono">{data.type}</span>
      </div>
    );
  }

  const catColor = getCategoryColor(data.category);
  const Icon = CATEGORY_ICONS[data.category] ?? Settings;
  const instruction = String(data.config.instruction ?? "");
  const hasInstruction = instruction.length > 0;

  const nonInstructionFields = definition.configSchema.filter((f) => f.key !== "instruction");
  const configuredCount = nonInstructionFields.filter(
    (f) => data.config[f.key] !== undefined && data.config[f.key] !== "",
  ).length;
  const totalConfig = nonInstructionFields.length;
  const isFullyConfigured = totalConfig === 0 || configuredCount === totalConfig;
  const isRunning = executionState === "running";
  const isActive = executionState === "running" || executionState === "success";

  const commitLabel = useCallback(() => {
    setEditingLabel(false);
    const trimmed = labelDraft.trim();
    if (trimmed && trimmed !== data.label) updateBlockLabel(data.id, trimmed);
    else setLabelDraft(data.label);
  }, [labelDraft, data.label, data.id, updateBlockLabel]);

  const openInspector = useCallback(() => {
    setSelectedBlockId(data.id);
    setRightPanelTab("inspector");
  }, [data.id, setSelectedBlockId, setRightPanelTab]);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    window.dispatchEvent(new CustomEvent("node-context-menu", {
      detail: { blockId: data.id, x: e.clientX, y: e.clientY },
    }));
  }, [data.id]);

  /* Status indicator */
  const statusDot = (() => {
    if (executionState === "running")
      return <Loader2 size={12} className="text-blue-400 animate-spin" />;
    if (executionState === "success")
      return <div className="flex h-4.5 w-4.5 items-center justify-center rounded-full bg-emerald-500/15"><Check size={10} className="text-emerald-400" strokeWidth={2.5} /></div>;
    if (executionState === "error")
      return <div className="flex h-4.5 w-4.5 items-center justify-center rounded-full bg-red-500/15"><CircleX size={10} className="text-red-400" /></div>;
    if (executionState === "queued")
      return <div className="flex h-4.5 w-4.5 items-center justify-center rounded-full bg-slate-700/50"><Play size={9} className="text-slate-400" /></div>;
    if (hasInstruction && isFullyConfigured)
      return <div className="h-2 w-2 rounded-full bg-emerald-400/80" />;
    if (hasInstruction)
      return <div className="h-2 w-2 rounded-full bg-amber-400/70" />;
    return <div className="h-2 w-2 rounded-full bg-slate-600" />;
  })();

  /* Border styling based on state */
  const borderColor = executionState === "running"
    ? "rgba(6,182,212,0.45)"
    : executionState === "success"
      ? "rgba(52,211,153,0.3)"
      : executionState === "error"
        ? "rgba(244,63,94,0.35)"
        : selected
          ? "rgba(148,163,184,0.25)"
          : "rgba(51,65,85,0.35)";

  return (
    <div
      className={`n8n-node group relative rounded-lg transition-all duration-200 ${
        selected
          ? "ring-1 ring-slate-400/40 shadow-lg shadow-black/20"
          : "shadow-md shadow-black/15 hover:shadow-lg hover:shadow-black/25"
      } ${isRunning ? "exec-running" : ""}`}
      style={{
        width: 264,
        background: "#141C2B",
        border: `1px solid ${borderColor}`,
      }}
      onDoubleClick={(e) => { e.stopPropagation(); openInspector(); }}
      onContextMenu={handleContextMenu}
    >
      {/* Hover actions */}
      <div className="absolute -top-2 -right-2 z-50 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
        <button
          className="flex h-5 w-5 items-center justify-center rounded-md bg-slate-800 border border-slate-600/40 text-slate-400 hover:bg-slate-700 hover:text-white transition-colors"
          title="Duplicate"
          onClick={(e) => { e.stopPropagation(); duplicateBlock(data.id); }}
        ><Copy size={9} /></button>
        <button
          className="flex h-5 w-5 items-center justify-center rounded-md bg-slate-800 border border-slate-600/40 text-slate-400 hover:bg-red-900/80 hover:text-red-300 transition-colors"
          title="Delete"
          onClick={(e) => { e.stopPropagation(); removeBlock(data.id); }}
        ><Trash2 size={9} /></button>
      </div>

      {/* Top accent — thin category color strip */}
      <div className="h-[2px] rounded-t-lg" style={{ backgroundColor: catColor, opacity: 0.7 }} />

      {/* Running progress indicator */}
      {isRunning && (
        <div className="h-[1.5px] w-full overflow-hidden bg-slate-800">
          <div className="node-progress-bar h-full" style={{ backgroundColor: catColor, opacity: 0.8 }} />
        </div>
      )}

      {/* Main content */}
      <div className="px-3.5 pt-3 pb-2">
        {/* Header: icon + label + status */}
        <div className="flex items-center gap-2.5">
          <div
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md"
            style={{ backgroundColor: `${catColor}12`, border: `1px solid ${catColor}20` }}
          >
            <Icon size={13} style={{ color: catColor }} strokeWidth={1.8} />
          </div>
          <div className="min-w-0 flex-1">
            {editingLabel ? (
              <input
                ref={labelRef}
                className="w-full bg-transparent text-[13px] font-semibold text-slate-100 outline-none border-b border-slate-500/50 pb-0.5"
                value={labelDraft}
                onChange={(e) => setLabelDraft(e.target.value)}
                onBlur={commitLabel}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitLabel();
                  if (e.key === "Escape") { setEditingLabel(false); setLabelDraft(data.label); }
                }}
              />
            ) : (
              <p
                className="text-[13px] font-semibold text-slate-200 truncate cursor-text leading-tight"
                onClick={(e) => { e.stopPropagation(); setEditingLabel(true); }}
                title="Click to rename"
              >{data.label}</p>
            )}
            <p className="text-[10px] font-medium mt-0.5 opacity-60" style={{ color: catColor }}>
              {data.category}
            </p>
          </div>
          {statusDot}
        </div>

        {/* Instruction preview */}
        {hasInstruction ? (
          <p
            className="mt-2.5 text-[11px] text-slate-400 leading-relaxed line-clamp-2 cursor-pointer hover:text-slate-300 transition-colors"
            onClick={(e) => { e.stopPropagation(); openInspector(); }}
            title={instruction}
          >
            {instruction}
          </p>
        ) : (
          <p
            className="mt-2.5 text-[11px] text-slate-600 italic cursor-pointer hover:text-slate-400 transition-colors"
            onClick={(e) => { e.stopPropagation(); openInspector(); }}
          >
            Add instruction…
          </p>
        )}

        {/* Config dots — small, inline */}
        {totalConfig > 0 && (
          <div className="mt-2 flex items-center gap-1.5">
            <span className="text-[9px] text-slate-600 font-medium">Config</span>
            <div className="flex gap-0.5">
              {nonInstructionFields.map((f, i) => (
                <div
                  key={f.key}
                  className="h-1.5 w-1.5 rounded-full"
                  style={{
                    backgroundColor: i < configuredCount
                      ? (isFullyConfigured ? "#4ade80" : "#facc15")
                      : "#334155",
                  }}
                />
              ))}
            </div>
            <span className="text-[9px] text-slate-600 tabular-nums">{configuredCount}/{totalConfig}</span>
          </div>
        )}

        {/* Metrics row — minimal */}
        <div className="mt-2 flex items-center gap-3 text-[10px]">
          <span className={`flex items-center gap-1 tabular-nums ${isRunning ? "text-slate-300" : "text-slate-600"}`}>
            <Zap size={9} className={isRunning ? "text-blue-400" : "text-slate-600"} />
            {isRunning ? metrics.fps : `~${metrics.fps}`} fps
          </span>
          <span className={`flex items-center gap-1 tabular-nums ${isRunning ? "text-slate-300" : "text-slate-600"}`}>
            <Clock size={9} />
            {metrics.latency}ms
          </span>
          {isActive && metrics.frames > 0 && (
            <span className="flex items-center gap-1 tabular-nums text-slate-500">
              <Activity size={9} />
              {metrics.frames}
            </span>
          )}
        </div>
      </div>

      {/* Ports */}
      {(definition.inputs.length > 0 || definition.outputs.length > 0) && (
        <div className="border-t border-slate-700/20 px-2.5 py-1.5">
          <div className="flex justify-between">
            <div className="space-y-0.5">
              {definition.inputs.map((port) => <BlockPort key={port.id} port={port} />)}
            </div>
            <div className="space-y-0.5">
              {definition.outputs.map((port) => (
                <div key={port.id} className="flex justify-end"><BlockPort port={port} /></div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between border-t border-slate-700/15 px-3.5 py-1.5 rounded-b-lg bg-[#111827]/60">
        <div className="flex items-center gap-2">
          <span className={`text-[10px] font-medium ${
            executionState === "running" ? "text-blue-400" :
            executionState === "success" ? "text-emerald-400/80" :
            executionState === "error" ? "text-red-400/80" :
            executionState === "queued" ? "text-slate-400" :
            hasInstruction && isFullyConfigured ? "text-emerald-500/60" :
            hasInstruction ? "text-amber-500/60" : "text-slate-600"
          }`}>
            {executionState === "running" ? "Processing" :
             executionState === "success" ? "Complete" :
             executionState === "error" ? "Error" :
             executionState === "queued" ? "Queued" :
             hasInstruction && isFullyConfigured ? "Ready" :
             hasInstruction ? "Partial" : "Draft"}
          </span>
          {(connInfo.inbound > 0 || connInfo.outbound > 0) && (
            <div className="flex items-center gap-1">
              {connInfo.inbound > 0 && (
                <span className="text-[9px] tabular-nums text-slate-600">{connInfo.inbound}↓</span>
              )}
              {connInfo.outbound > 0 && (
                <span className="text-[9px] tabular-nums text-slate-600">{connInfo.outbound}↑</span>
              )}
            </div>
          )}
        </div>
        <button
          className="text-[10px] font-medium text-slate-500 hover:text-slate-300 transition-colors"
          onClick={(e) => { e.stopPropagation(); openInspector(); }}
        >Edit</button>
      </div>
    </div>
  );
}
