import { useEffect, useState, useRef, useCallback } from "react";
import {
  Activity,
  Wifi,
  WifiOff,
  Layers,
  Zap,
  Clock,
  Eye,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Circle,
  X,
  ChevronDown,
  ChevronUp,
  Gauge,
  Image as ImageIcon,
  Cpu,
  Scan,
  Focus,
  Wand2,
  Sparkles,
  Type,
  ShieldCheck,
  Package,
  Search,
  Brain,
  Maximize2,
  Minimize2,
  Play,
  ArrowRight,
} from "lucide-react";
import clsx from "clsx";
import { usePipelineMonitorStore, type StageState, type FrameProcessingState } from "../../store/pipelineMonitorStore";
import { useUiStore } from "../../store/uiStore";
import type { WsDetectionEvent } from "../../types/api";
import { DetectionDetailModal } from "../detections/DetectionDetailModal";

// ── Stage icon map ───────────────────────────────────────────────

const STAGE_ICONS: Record<string, typeof Activity> = {
  S0_ingest: Package,
  S1_preprocess: Wand2,
  S2_scene_analysis: Eye,
  S3_anomaly_detection: AlertTriangle,
  S4_deep_analysis: Brain,
  S5_results: ShieldCheck,
};

const STAGE_COLORS: Record<string, string> = {
  S0_ingest: "#38bdf8",
  S1_preprocess: "#c084fc",
  S2_scene_analysis: "#22d3ee",
  S3_anomaly_detection: "#f97316",
  S4_deep_analysis: "#a78bfa",
  S5_results: "#4ade80",
};

// ── Status badge ─────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { bg: string; text: string; label: string }> = {
    pending: { bg: "bg-slate-800/60", text: "text-slate-500", label: "WAITING" },
    running: { bg: "bg-amber-950/50", text: "text-amber-400", label: "RUNNING" },
    completed: { bg: "bg-emerald-950/50", text: "text-emerald-400", label: "DONE" },
    failed: { bg: "bg-rose-950/50", text: "text-rose-400", label: "FAILED" },
    skipped: { bg: "bg-slate-800/60", text: "text-slate-500", label: "SKIP" },
  };
  const s = map[status] ?? map.pending;
  return (
    <span className={clsx("px-1.5 py-0.5 rounded text-[8px] font-black uppercase tracking-widest border", s.bg, s.text, `border-current/20`)}>
      {s.label}
    </span>
  );
}

// ── Live frame hero preview ──────────────────────────────────────

function LiveFramePreview({
  frame,
  onFullscreen,
}: {
  frame: FrameProcessingState;
  onFullscreen: () => void;
}) {
  // Find the latest available thumbnail working backwards through stages
  const latestThumb =
    [...frame.stages].reverse().find((s) => s.thumbnail_b64)?.thumbnail_b64 ||
    frame.frame_thumbnail;

  const runningStage = frame.stages.find((s) => s.status === "running");
  const completedCount = frame.stages.filter((s) => s.status === "completed").length;
  const pct = (completedCount / frame.stages.length) * 100;

  if (!latestThumb) {
    return (
      <div className="relative flex h-44 items-center justify-center rounded-xl border border-dashed border-slate-700/40 bg-[#060d1a]">
        <div className="text-center">
          <div className="relative mx-auto mb-2">
            <Eye size={24} className="text-slate-600" />
            {frame.status === "processing" && (
              <div className="absolute -top-1 -right-1 h-2.5 w-2.5 rounded-full bg-emerald-500 animate-ping" />
            )}
          </div>
          <span className="text-[10px] text-slate-500 font-semibold block">Processing frame...</span>
          {runningStage && (
            <span className="text-[9px] text-amber-400/80 mt-0.5 block">{runningStage.name}</span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="relative rounded-xl border border-slate-700/30 overflow-hidden bg-black/60 group">
      {/* Main preview image — larger, prominent */}
      <img
        src={`data:image/jpeg;base64,${latestThumb}`}
        alt="Live pipeline preview"
        className="w-full h-auto min-h-[140px] max-h-[220px] object-contain bg-black"
      />

      {/* Top-left live indicator */}
      <div className="absolute top-2 left-2 flex items-center gap-1.5 rounded-md bg-black/70 backdrop-blur-sm px-2 py-1 border border-white/10">
        <div
          className={clsx(
            "h-2 w-2 rounded-full",
            frame.status === "processing" ? "bg-emerald-400 animate-pulse" : frame.status === "flagged" ? "bg-rose-400 animate-pulse" : "bg-slate-400",
          )}
        />
        <span className="text-[9px] font-black text-white uppercase tracking-widest">
          {frame.status === "processing" ? "LIVE" : frame.status === "flagged" ? "ALERT" : "DONE"}
        </span>
      </div>

      {/* Top-right fullscreen toggle */}
      <button
        onClick={onFullscreen}
        className="absolute top-2 right-2 rounded-md bg-black/70 backdrop-blur-sm p-1.5 border border-white/10 text-white/60 hover:text-white hover:bg-black/80 transition-all opacity-0 group-hover:opacity-100"
        title="Fullscreen preview"
      >
        <Maximize2 size={12} />
      </button>

      {/* Bottom overlay bar */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 via-black/60 to-transparent pt-6 pb-2 px-3">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[9px] font-mono text-slate-300">{frame.camera_id}</span>
            <span className="text-[8px] text-slate-500">
              {new Date(frame.started_at * 1000).toLocaleTimeString()}
            </span>
          </div>
          {runningStage && (
            <div className="flex items-center gap-1">
              <Loader2 size={9} className="text-amber-400 animate-spin" />
              <span className="text-[9px] text-amber-300 font-semibold">{runningStage.name}</span>
            </div>
          )}
        </div>
        {/* Progress bar over image */}
        <div className="h-1 rounded-full bg-white/10 overflow-hidden">
          <div
            className={clsx(
              "h-full rounded-full transition-all duration-500 ease-out",
              frame.status === "flagged" ? "bg-rose-500" : "bg-emerald-400",
            )}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Flag badge */}
      {frame.status === "flagged" && (
        <div className="absolute top-2 left-1/2 -translate-x-1/2 flex items-center gap-1 rounded-md bg-rose-900/80 backdrop-blur-sm border border-rose-500/40 px-2.5 py-1 animate-pulse">
          <AlertTriangle size={11} className="text-rose-300" />
          <span className="text-[9px] font-black text-rose-200 uppercase tracking-wider">Anomaly Detected</span>
        </div>
      )}
    </div>
  );
}

// ── Stage filmstrip — horizontal scroll of per-stage thumbnails ──

function StageFilmstrip({ frame }: { frame: FrameProcessingState }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const stagesWithThumbs = frame.stages.filter(
    (s) => s.thumbnail_b64 || s.status === "running" || s.status === "completed",
  );

  if (stagesWithThumbs.length === 0) return null;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        <Play size={9} className="text-slate-500" />
        <span className="text-[9px] font-bold uppercase tracking-wider text-slate-500">
          Stage Preview Strip
        </span>
      </div>
      <div
        ref={scrollRef}
        className="flex gap-1.5 overflow-x-auto pb-1.5 scrollbar-thin scrollbar-track-slate-900 scrollbar-thumb-slate-700"
      >
        {frame.stages.map((stage, i) => {
          const color = STAGE_COLORS[stage.id] ?? "#64748b";
          const isRunning = stage.status === "running";
          const isDone = stage.status === "completed";
          const Icon = STAGE_ICONS[stage.id] ?? Activity;

          return (
            <div key={stage.id} className="flex items-center gap-1">
              <div
                className={clsx(
                  "flex-shrink-0 w-[72px] rounded-lg border overflow-hidden transition-all duration-300",
                  isRunning && "border-amber-500/50 ring-1 ring-amber-500/20 shadow-md shadow-amber-900/20",
                  isDone && "border-slate-600/40",
                  !isRunning && !isDone && "border-slate-700/30 opacity-40",
                )}
              >
                {stage.thumbnail_b64 ? (
                  <div className="relative">
                    <img
                      src={`data:image/jpeg;base64,${stage.thumbnail_b64}`}
                      alt={stage.name}
                      className="w-full h-10 object-cover"
                    />
                    <div
                      className="absolute bottom-0 left-0 right-0 h-0.5"
                      style={{ backgroundColor: color }}
                    />
                  </div>
                ) : (
                  <div className="flex h-10 items-center justify-center bg-[#0a1628]">
                    {isRunning ? (
                      <Loader2 size={12} className="text-amber-400 animate-spin" />
                    ) : (
                      <Icon size={12} className="text-slate-600" />
                    )}
                  </div>
                )}
                <div className="px-1 py-0.5 bg-[#0c1628]">
                  <span
                    className="text-[7px] font-bold uppercase tracking-wider block truncate"
                    style={{ color }}
                  >
                    {stage.id.split("_")[0]}
                  </span>
                  {stage.latency_ms != null && (
                    <span className="text-[7px] font-mono text-slate-500">
                      {stage.latency_ms.toFixed(0)}ms
                    </span>
                  )}
                </div>
              </div>
              {i < frame.stages.length - 1 && (
                <ArrowRight
                  size={8}
                  className={clsx(
                    "flex-shrink-0",
                    isDone ? "text-slate-500" : "text-slate-700",
                  )}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Stage card ───────────────────────────────────────────────────

function StageCard({ stage, isActive }: { stage: StageState; isActive: boolean }) {
  const Icon = STAGE_ICONS[stage.id] ?? Activity;
  const isRunning = stage.status === "running";
  const isDone = stage.status === "completed";
  const isFailed = stage.status === "failed";
  const color = STAGE_COLORS[stage.id] ?? "#64748b";

  return (
    <div
      className={clsx(
        "relative flex flex-col gap-1.5 rounded-lg border p-2.5 transition-all duration-300",
        isRunning && "border-amber-500/40 bg-amber-950/20 shadow-lg shadow-amber-900/10 ring-1 ring-amber-500/20",
        isDone && "border-emerald-700/30 bg-emerald-950/10",
        isFailed && "border-rose-700/30 bg-rose-950/10",
        !isRunning && !isDone && !isFailed && "border-slate-700/30 bg-[#0c1628]",
        isActive && isRunning && "animate-pulse",
      )}
    >
      {/* Color accent strip */}
      <div className="absolute top-0 left-0 right-0 h-0.5 rounded-t-lg" style={{ backgroundColor: color, opacity: isDone ? 1 : 0.3 }} />

      {/* Header row */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <div className={clsx(
            "flex h-5 w-5 items-center justify-center rounded",
            isRunning ? "bg-amber-500/20" : isDone ? "bg-emerald-500/15" : "bg-slate-800",
          )}>
            {isRunning ? (
              <Loader2 size={11} className="text-amber-400 animate-spin" />
            ) : isDone ? (
              <CheckCircle2 size={11} className="text-emerald-400" />
            ) : isFailed ? (
              <AlertTriangle size={11} className="text-rose-400" />
            ) : (
              <Icon size={11} className="text-slate-500" />
            )}
          </div>
          <span className={clsx(
            "text-[10px] font-bold uppercase tracking-wider",
            isRunning ? "text-amber-300" : isDone ? "text-emerald-300" : "text-slate-500",
          )}>
            {stage.name}
          </span>
        </div>
        <StatusBadge status={stage.status} />
      </div>

      {/* Description */}
      <p className="text-[9px] text-slate-500 leading-tight">{stage.description}</p>

      {/* Latency bar */}
      {stage.latency_ms != null && (
        <div className="flex items-center gap-1.5">
          <Clock size={9} className={isDone ? "text-emerald-500" : "text-slate-500"} />
          <div className="flex-1 h-1 rounded-full bg-slate-800 overflow-hidden">
            <div
              className={clsx(
                "h-full rounded-full transition-all duration-500",
                stage.latency_ms < 50 ? "bg-emerald-500" : stage.latency_ms < 200 ? "bg-amber-500" : "bg-rose-500",
              )}
              style={{ width: `${Math.min((stage.latency_ms / 500) * 100, 100)}%` }}
            />
          </div>
          <span className={clsx(
            "text-[9px] font-mono font-bold tabular-nums",
            stage.latency_ms < 50 ? "text-emerald-400" : stage.latency_ms < 200 ? "text-amber-400" : "text-rose-400",
          )}>
            {stage.latency_ms.toFixed(1)}ms
          </span>
        </div>
      )}

      {/* Stage details */}
      {Object.keys(stage.details).length > 0 && (
        <div className="flex flex-wrap gap-1">
          {Object.entries(stage.details).map(([k, v]) => (
            <span key={k} className="rounded bg-slate-800/80 px-1.5 py-0.5 text-[8px] text-slate-400 font-mono">
              {k}: <span className="text-slate-300">{String(v)}</span>
            </span>
          ))}
        </div>
      )}

      {/* Thumbnail — enhanced size */}
      {stage.thumbnail_b64 && (
        <div className="mt-1 rounded-lg border border-slate-700/40 overflow-hidden bg-black/30">
          <img
            src={`data:image/jpeg;base64,${stage.thumbnail_b64}`}
            alt={`${stage.name} output`}
            className="w-full h-auto max-h-36 object-contain"
          />
        </div>
      )}
    </div>
  );
}

// ── Pipeline progress visualizer ─────────────────────────────────

function PipelineProgress({ frame }: { frame: FrameProcessingState }) {
  const completedCount = frame.stages.filter((s) => s.status === "completed").length;
  const runningStage = frame.stages.find((s) => s.status === "running");
  const pct = (completedCount / frame.stages.length) * 100;

  return (
    <div className="space-y-2">
      {/* Progress bar */}
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
          <div
            className={clsx(
              "h-full rounded-full transition-all duration-700 ease-out",
              frame.status === "flagged" ? "bg-rose-500" : "bg-emerald-500",
            )}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-[10px] font-mono font-bold text-slate-400 tabular-nums w-10 text-right">
          {completedCount}/{frame.stages.length}
        </span>
      </div>

      {/* Stage dots */}
      <div className="flex items-center justify-between gap-0.5">
        {frame.stages.map((stage, i) => {
          const isRunning = stage.status === "running";
          const isDone = stage.status === "completed";
          const isFailed = stage.status === "failed";
          const color = STAGE_COLORS[stage.id] ?? "#64748b";
          return (
            <div key={stage.id} className="flex items-center gap-0.5">
              <div
                className={clsx(
                  "h-2.5 w-2.5 rounded-full transition-all duration-300",
                  isRunning && "ring-2 ring-amber-400/30 animate-pulse",
                  isFailed && "bg-rose-400",
                  !isRunning && !isDone && !isFailed && "bg-slate-700",
                )}
                style={isDone || isRunning ? { backgroundColor: isRunning ? "#fbbf24" : color } : undefined}
                title={`${stage.name}: ${stage.status}${stage.latency_ms ? ` (${stage.latency_ms.toFixed(1)}ms)` : ""}`}
              />
              {i < frame.stages.length - 1 && (
                <div className={clsx(
                  "h-px w-3",
                  isDone ? "bg-emerald-700" : "bg-slate-700",
                )} />
              )}
            </div>
          );
        })}
      </div>

      {/* Currently running */}
      {runningStage && (
        <div className="flex items-center gap-1.5">
          <Loader2 size={10} className="text-amber-400 animate-spin" />
          <span className="text-[10px] text-amber-300 font-semibold">{runningStage.name}</span>
        </div>
      )}
    </div>
  );
}

// ── Timing waterfall ─────────────────────────────────────────────

function TimingWaterfall({ frame }: { frame: FrameProcessingState }) {
  const completedStages = frame.stages.filter((s) => s.latency_ms != null);
  if (completedStages.length === 0) return null;

  const maxMs = Math.max(...completedStages.map((s) => s.latency_ms!), 1);

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1.5 mb-1">
        <Gauge size={10} className="text-slate-500" />
        <span className="text-[9px] font-bold uppercase tracking-wider text-slate-500">Stage Timing Waterfall</span>
      </div>
      {completedStages.map((stage) => {
        const color = STAGE_COLORS[stage.id] ?? "#64748b";
        return (
          <div key={stage.id} className="flex items-center gap-2">
            <span className="text-[8px] text-slate-500 w-14 truncate font-mono">{stage.id.split("_")[0]}</span>
            <div className="flex-1 h-2.5 rounded bg-slate-800/80 overflow-hidden">
              <div
                className="h-full rounded transition-all duration-500"
                style={{
                  width: `${(stage.latency_ms! / maxMs) * 100}%`,
                  backgroundColor: color,
                  opacity: 0.75,
                }}
              />
            </div>
            <span className="text-[8px] font-mono font-bold text-slate-400 tabular-nums w-14 text-right">
              {stage.latency_ms!.toFixed(1)}ms
            </span>
          </div>
        );
      })}
      {frame.total_latency_ms != null && (
        <div className="flex items-center justify-end gap-1 pt-0.5 border-t border-slate-700/30">
          <span className="text-[9px] font-bold text-slate-400">Total:</span>
          <span className="text-[10px] font-mono font-black text-slate-200 tabular-nums">
            {frame.total_latency_ms.toFixed(1)}ms
          </span>
        </div>
      )}
    </div>
  );
}

// ── Recent frames list ───────────────────────────────────────────

function RecentFrameRow({ frame, onClick }: { frame: FrameProcessingState; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        "flex w-full items-center gap-2 rounded-md border px-2 py-1.5 text-left transition-colors",
        frame.status === "flagged"
          ? "border-rose-800/40 bg-rose-950/20 hover:bg-rose-950/30"
          : "border-slate-700/30 bg-[#0c1628] hover:bg-[#111e36]",
      )}
    >
      {/* Mini thumbnail */}
      {frame.frame_thumbnail ? (
        <img
          src={`data:image/jpeg;base64,${frame.frame_thumbnail}`}
          alt=""
          className="h-6 w-9 rounded object-cover flex-shrink-0 border border-slate-700/40"
        />
      ) : (
        <div className={clsx(
          "h-1.5 w-1.5 rounded-full flex-shrink-0",
          frame.status === "flagged" ? "bg-rose-400" : "bg-emerald-400",
        )} />
      )}
      <span className="text-[9px] font-mono text-slate-400 flex-shrink-0">{frame.frame_id.slice(0, 14)}</span>
      <span className="text-[9px] text-slate-500 flex-1 truncate">{frame.camera_id}</span>
      <span className="text-[9px] font-mono font-bold tabular-nums text-slate-400">
        {frame.total_latency_ms ? `${frame.total_latency_ms.toFixed(0)}ms` : "—"}
      </span>
      {frame.status === "flagged" && (
        <AlertTriangle size={10} className="text-rose-400 flex-shrink-0" />
      )}
    </button>
  );
}

// ── Fullscreen preview overlay ───────────────────────────────────

function FullscreenPreview({
  frame,
  onClose,
}: {
  frame: FrameProcessingState;
  onClose: () => void;
}) {
  const latestThumb =
    [...frame.stages].reverse().find((s) => s.thumbnail_b64)?.thumbnail_b64 ||
    frame.frame_thumbnail;

  if (!latestThumb) return null;

  return (
    <div
      className="fixed inset-0 z-[9999] flex flex-col items-center justify-center bg-black/90 backdrop-blur-md"
      onClick={onClose}
    >
      {/* Close button */}
      <button
        className="absolute top-4 right-4 rounded-lg bg-white/10 p-2 text-white/70 hover:text-white hover:bg-white/20 transition-all"
        onClick={onClose}
        title="Close fullscreen"
      >
        <Minimize2 size={18} />
      </button>

      {/* Large frame */}
      <img
        src={`data:image/jpeg;base64,${latestThumb}`}
        alt="Fullscreen pipeline preview"
        className="max-w-[90vw] max-h-[70vh] object-contain rounded-xl border border-white/10 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      />

      {/* Stage strip below */}
      <div
        className="flex gap-2 mt-4 px-8 overflow-x-auto max-w-[90vw]"
        onClick={(e) => e.stopPropagation()}
      >
        {frame.stages.map((stage) => {
          const color = STAGE_COLORS[stage.id] ?? "#64748b";
          return (
            <div key={stage.id} className="flex-shrink-0 text-center">
              {stage.thumbnail_b64 ? (
                <img
                  src={`data:image/jpeg;base64,${stage.thumbnail_b64}`}
                  alt={stage.name}
                  className="h-16 w-24 object-cover rounded-lg border border-white/10"
                />
              ) : (
                <div className="h-16 w-24 rounded-lg border border-white/10 bg-white/5 flex items-center justify-center">
                  <span className="text-[9px] text-white/30">No preview</span>
                </div>
              )}
              <span className="text-[9px] font-bold mt-1 block" style={{ color }}>
                {stage.id.split("_")[0]}
              </span>
              {stage.latency_ms != null && (
                <span className="text-[8px] text-white/40 font-mono">{stage.latency_ms.toFixed(0)}ms</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Frame info */}
      <div className="mt-3 flex items-center gap-4 text-white/50 text-[10px]">
        <span className="font-mono">{frame.frame_id}</span>
        <span>{frame.camera_id}</span>
        {frame.total_latency_ms && (
          <span className="font-mono font-bold text-white/70">{frame.total_latency_ms.toFixed(0)}ms total</span>
        )}
      </div>
    </div>
  );
}

// ── Manager Agent job lifecycle panel ─────────────────────────────

const PHASE_CONFIG: Record<string, { label: string; color: string; icon: typeof Activity; pulse?: boolean }> = {
  initializing: { label: "INITIALIZING", color: "text-slate-400", icon: Circle },
  discovering: { label: "DISCOVERING", color: "text-cyan-400", icon: Search, pulse: true },
  provisioning: { label: "PROVISIONING", color: "text-violet-400", icon: Package, pulse: true },
  processing: { label: "PROCESSING", color: "text-emerald-400", icon: Cpu, pulse: true },
  anomaly_detected: { label: "ANOMALY FOUND", color: "text-rose-400", icon: AlertTriangle, pulse: true },
  completing: { label: "COMPLETING", color: "text-amber-400", icon: CheckCircle2 },
  flushing: { label: "FLUSHING", color: "text-orange-400", icon: Sparkles, pulse: true },
  completed: { label: "COMPLETED", color: "text-emerald-400", icon: CheckCircle2 },
  error: { label: "ERROR", color: "text-rose-400", icon: AlertTriangle },
};

function ManagerJobPanel({ job }: { job: import("../../types/api").ManagerJobState }) {
  const cfg = PHASE_CONFIG[job.phase] ?? PHASE_CONFIG.initializing;
  const PhaseIcon = cfg.icon;

  return (
    <div className="rounded-lg border border-violet-800/30 bg-gradient-to-br from-violet-950/20 to-slate-900/40 p-2.5 space-y-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Brain size={12} className="text-violet-400" />
          <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-violet-300">
            Manager Agent
          </span>
        </div>
        <div className="flex items-center gap-1">
          <PhaseIcon size={10} className={clsx(cfg.color, cfg.pulse && "animate-pulse")} />
          <span className={clsx("text-[9px] font-black uppercase tracking-wider", cfg.color)}>
            {cfg.label}
          </span>
        </div>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-3 gap-1.5">
        <div className="flex flex-col items-center rounded border border-slate-700/30 bg-[#0c1628] py-1 px-1">
          <span className="text-[10px] font-mono font-bold text-violet-300">{job.dynamicComponents.length}</span>
          <span className="text-[7px] font-bold uppercase tracking-wider text-slate-600">Components</span>
        </div>
        <div className="flex flex-col items-center rounded border border-slate-700/30 bg-[#0c1628] py-1 px-1">
          <span className="text-[10px] font-mono font-bold text-amber-300">{job.anomalyFrames}</span>
          <span className="text-[7px] font-bold uppercase tracking-wider text-slate-600">Anomalies</span>
        </div>
        <div className="flex flex-col items-center rounded border border-slate-700/30 bg-[#0c1628] py-1 px-1">
          <span className="text-[10px] font-mono font-bold text-slate-300">{job.totalFrames}</span>
          <span className="text-[7px] font-bold uppercase tracking-wider text-slate-600">Frames</span>
        </div>
      </div>

      {/* Dynamic components list */}
      {job.dynamicComponents.length > 0 && (
        <div className="space-y-0.5">
          <span className="text-[8px] font-bold uppercase tracking-wider text-slate-500">Dynamic Components</span>
          <div className="flex flex-wrap gap-1">
            {job.dynamicComponents.map((c) => (
              <span key={c} className="rounded bg-violet-950/50 border border-violet-800/30 px-1.5 py-0.5 text-[8px] text-violet-300 font-mono">
                {c}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Pip packages */}
      {job.dynamicPipPackages.length > 0 && (
        <div className="space-y-0.5">
          <span className="text-[8px] font-bold uppercase tracking-wider text-slate-500">Installed Packages</span>
          <div className="flex flex-wrap gap-1">
            {job.dynamicPipPackages.map((p) => (
              <span key={p} className="rounded bg-cyan-950/40 border border-cyan-800/30 px-1.5 py-0.5 text-[8px] text-cyan-300 font-mono">
                {p}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Flushed status */}
      {job.flushed && (
        <div className="flex items-center gap-1.5 rounded bg-emerald-950/20 border border-emerald-800/20 px-2 py-1">
          <CheckCircle2 size={10} className="text-emerald-400" />
          <span className="text-[9px] text-emerald-300 font-semibold">
            Dynamic packages flushed — system clean
          </span>
        </div>
      )}
    </div>
  );
}

// ── Main PipelineMonitor component ───────────────────────────────

export function PipelineMonitor() {
  const connected = usePipelineMonitorStore((s) => s.connected);
  const currentFrame = usePipelineMonitorStore((s) => s.currentFrame);
  const recentFrames = usePipelineMonitorStore((s) => s.recentFrames);
  const queueDepth = usePipelineMonitorStore((s) => s.queueDepth);
  const throttled = usePipelineMonitorStore((s) => s.throttled);
  const totalProcessed = usePipelineMonitorStore((s) => s.totalProcessed);
  const fps = usePipelineMonitorStore((s) => s.fps);
  const lastFlag = usePipelineMonitorStore((s) => s.lastFlag);
  const flagConsumed = usePipelineMonitorStore((s) => s.flagConsumed);
  const consumeFlag = usePipelineMonitorStore((s) => s.consumeFlag);
  const managerJob = usePipelineMonitorStore((s) => s.managerJob);
  const lastCompletedFrame = usePipelineMonitorStore((s) => s.lastCompletedFrame);

  const [expandStages, setExpandStages] = useState(true);
  const [expandRecent, setExpandRecent] = useState(false);
  const [selectedFrame, setSelectedFrame] = useState<FrameProcessingState | null>(null);
  const [flagModal, setFlagModal] = useState<Record<string, unknown> | null>(null);
  const [fullscreenFrame, setFullscreenFrame] = useState<FrameProcessingState | null>(null);

  // Auto-transition: when a flag is raised, show detection detail modal
  useEffect(() => {
    if (lastFlag && !flagConsumed && lastFlag.detection) {
      setFlagModal(lastFlag.detection);
      consumeFlag();
    }
  }, [lastFlag, flagConsumed, consumeFlag]);

  const activeFrame = currentFrame || lastCompletedFrame || recentFrames[0] || null;

  return (
    <div className="flex h-full flex-col gap-3 p-3 overflow-auto">
      {/* Header with connection indicator */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity size={14} className={clsx(connected ? "text-emerald-500 animate-pulse" : "text-slate-500")} />
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
            Pipeline Monitor
          </span>
          {connected ? (
            <Wifi size={10} className="text-emerald-500" />
          ) : (
            <WifiOff size={10} className="text-rose-400" />
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {throttled && (
            <span className="rounded bg-amber-950/50 border border-amber-800/40 px-1.5 py-0.5 text-[8px] font-bold text-amber-400 uppercase tracking-wider">
              THROTTLED
            </span>
          )}
          <button
            onClick={() => useUiStore.getState().setPipelineTheater(true)}
            className="rounded bg-violet-950/40 border border-violet-700/30 px-1.5 py-0.5 text-[8px] font-bold text-violet-400 uppercase tracking-wider hover:bg-violet-900/40 hover:border-violet-600/40 transition-colors"
            title="Open Pipeline Vision Theater"
          >
            <Maximize2 size={10} className="inline mr-0.5 -mt-px" />
            Theater
          </button>
        </div>
      </div>

      {/* Live metrics strip */}
      <div className="grid grid-cols-4 gap-1.5">
        {[
          { label: "FPS", value: fps.toFixed(1), icon: Gauge, color: fps > 0 ? "text-emerald-400" : "text-slate-500" },
          { label: "QUEUE", value: String(queueDepth), icon: Layers, color: queueDepth > 10 ? "text-amber-400" : "text-slate-400" },
          { label: "TOTAL", value: totalProcessed.toLocaleString(), icon: Zap, color: "text-slate-300" },
          { label: "STATUS", value: connected ? "LIVE" : "OFF", icon: Cpu, color: connected ? "text-emerald-400" : "text-rose-400" },
        ].map((m) => (
          <div key={m.label} className="flex flex-col items-center rounded-md border border-slate-700/30 bg-[#0c1628] py-1.5 px-1">
            <m.icon size={10} className="text-slate-600 mb-0.5" />
            <span className={clsx("text-xs font-mono font-black tabular-nums", m.color)}>{m.value}</span>
            <span className="text-[7px] font-bold uppercase tracking-wider text-slate-600">{m.label}</span>
          </div>
        ))}
      </div>

      {/* Manager Agent job lifecycle status */}
      {managerJob && (
        <ManagerJobPanel job={managerJob} />
      )}

      {/* Current frame processing */}
      {activeFrame ? (
        <div className="space-y-2.5">
          {/* Frame banner */}
          <div className={clsx(
            "flex items-center justify-between rounded-lg border px-3 py-2",
            activeFrame.status === "flagged"
              ? "border-rose-700/40 bg-rose-950/20"
              : activeFrame.status === "processing"
              ? "border-emerald-700/30 bg-emerald-950/10"
              : "border-slate-700/30 bg-[#0c1628]",
          )}>
            <div className="flex items-center gap-2">
              <div className={clsx(
                "h-2 w-2 rounded-full",
                activeFrame.status === "processing" ? "bg-emerald-400 animate-pulse"
                  : activeFrame.status === "flagged" ? "bg-rose-400 animate-pulse" : "bg-slate-400",
              )} />
              <div>
                <span className="text-[9px] font-mono font-bold text-slate-200 block">
                  {activeFrame.frame_id}
                </span>
                <span className="text-[8px] text-slate-500">
                  {activeFrame.camera_id} • {new Date(activeFrame.started_at * 1000).toLocaleTimeString()}
                </span>
              </div>
            </div>
            {activeFrame.status === "flagged" && (
              <span className="flex items-center gap-1 rounded bg-rose-900/50 border border-rose-700/40 px-2 py-0.5">
                <AlertTriangle size={10} className="text-rose-400" />
                <span className="text-[9px] font-bold text-rose-400 uppercase">FLAG RAISED</span>
              </span>
            )}
          </div>

          {/* Hero live preview */}
          <LiveFramePreview
            frame={activeFrame}
            onFullscreen={() => setFullscreenFrame(activeFrame)}
          />

          {/* Stage filmstrip */}
          <StageFilmstrip frame={activeFrame} />

          {/* Progress visualization */}
          <PipelineProgress frame={activeFrame} />

          {/* Timing waterfall */}
          <TimingWaterfall frame={activeFrame} />

          {/* Collapsible stage detail cards */}
          <button
            onClick={() => setExpandStages(!expandStages)}
            className="flex w-full items-center justify-between rounded border border-slate-700/30 bg-[#0c1628] px-2 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400 hover:bg-[#111e36] transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <Layers size={10} />
              Stage Details ({activeFrame.stages.length})
            </span>
            {expandStages ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
          </button>

          {expandStages && (
            <div className="space-y-1.5">
              {activeFrame.stages.map((stage) => (
                <StageCard
                  key={stage.id}
                  stage={stage}
                  isActive={stage.index === activeFrame.current_stage_index}
                />
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-10 gap-2">
          {connected ? (
            <>
              <div className="relative">
                <Circle size={32} className="text-slate-700" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="h-2 w-2 rounded-full bg-emerald-500 animate-ping" />
                </div>
              </div>
              <span className="text-[10px] text-slate-500 font-semibold">Pipeline idle — waiting for frames...</span>
            </>
          ) : (
            <>
              <WifiOff size={28} className="text-slate-600" />
              <span className="text-[10px] text-slate-500 font-semibold">Not connected to pipeline</span>
              <span className="text-[9px] text-slate-600">Start the backend to enable pipeline monitoring</span>
            </>
          )}
        </div>
      )}

      {/* Recent frames */}
      {recentFrames.length > 0 && (
        <div className="space-y-1.5 mt-1">
          <button
            onClick={() => setExpandRecent(!expandRecent)}
            className="flex w-full items-center justify-between rounded border border-slate-700/30 bg-[#0c1628] px-2 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400 hover:bg-[#111e36] transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <Clock size={10} />
              Recent Frames ({recentFrames.length})
            </span>
            {expandRecent ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
          </button>

          {expandRecent && (
            <div className="space-y-1 max-h-48 overflow-auto">
              {recentFrames.map((f) => (
                <RecentFrameRow
                  key={f.frame_id}
                  frame={f}
                  onClick={() => setSelectedFrame(f)}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Fullscreen preview overlay */}
      {fullscreenFrame && (
        <FullscreenPreview
          frame={fullscreenFrame}
          onClose={() => setFullscreenFrame(null)}
        />
      )}

      {/* Selected recent frame detail overlay */}
      {selectedFrame && (
        <div className="fixed inset-0 z-[9997] flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={() => setSelectedFrame(null)}>
          <div
            className="relative w-full max-w-sm max-h-[80vh] overflow-auto rounded-2xl border border-slate-700/40 bg-[#0d1b2e] shadow-2xl p-4 space-y-3"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-200">Frame: {selectedFrame.frame_id}</span>
              <button onClick={() => setSelectedFrame(null)} className="rounded p-1 hover:bg-slate-800" title="Close">
                <X size={12} className="text-slate-400" />
              </button>
            </div>
            <LiveFramePreview frame={selectedFrame} onFullscreen={() => { setFullscreenFrame(selectedFrame); setSelectedFrame(null); }} />
            <StageFilmstrip frame={selectedFrame} />
            <PipelineProgress frame={selectedFrame} />
            <TimingWaterfall frame={selectedFrame} />
            <div className="space-y-1.5">
              {selectedFrame.stages.map((stage) => (
                <StageCard key={stage.id} stage={stage} isActive={false} />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Auto-transition detection detail modal for flagged frames */}
      {flagModal && (
        <DetectionDetailModal
          detection={{
            id: (flagModal.detection_id as string) ?? (lastFlag?.frame_id ?? "unknown"),
            camera_id: (flagModal.camera_id as string) ?? "",
            detected_at_utc: new Date().toISOString(),
            validation_status: (flagModal.risk_level as string) ?? (flagModal.validation_status as string) ?? "unknown",
            confidence: (flagModal.confidence as number) ?? 0,
            scene_description: (flagModal.scene_description as string) ?? "",
            objects_detected: (flagModal.objects_detected as Array<{label: string; location: string; condition: string}>) ?? [],
            anomaly_detected: (flagModal.anomaly_detected as boolean) ?? false,
            anomalies: (flagModal.anomalies as Array<{type: string; description: string; severity: string; location: string}>) ?? [],
            chain_of_thought: (flagModal.chain_of_thought as string) ?? "",
            risk_level: (flagModal.risk_level as string) ?? "unknown",
            risk_analysis: (flagModal.risk_analysis as string) ?? "",
            impact_analysis: (flagModal.impact_analysis as string) ?? "",
            recommendations: (flagModal.recommendations as string[]) ?? [],
          } satisfies WsDetectionEvent}
          initialTab="reasoning"
          onClose={() => setFlagModal(null)}
        />
      )}
    </div>
  );
}
