import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  Activity,
  Eye,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  X,
  Gauge,
  Layers,
  Zap,
  Cpu,
  Clock,
  Wifi,
  WifiOff,
  Package,
  Wand2,
  Brain,
  ShieldCheck,
  Minimize2,
  Maximize2,
  ChevronLeft,
  ChevronRight,
  Info,
  Shield,
  Target,
  TrendingUp,
  Lightbulb,
  Scan,
  ImageIcon,
  BarChart3,
  Timer,
  Flame,
} from "lucide-react";
import clsx from "clsx";
import {
  usePipelineMonitorStore,
  type StageState,
  type FrameProcessingState,
  type PipelineMetrics,
  type StageLatencyPoint,
} from "../../store/pipelineMonitorStore";
import { useUiStore } from "../../store/uiStore";

// ── Constants ────────────────────────────────────────────────────

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

const STAGE_GRADIENTS: Record<string, string> = {
  S0_ingest: "from-sky-500/20 to-sky-900/5",
  S1_preprocess: "from-purple-500/20 to-purple-900/5",
  S2_scene_analysis: "from-cyan-500/20 to-cyan-900/5",
  S3_anomaly_detection: "from-orange-500/20 to-orange-900/5",
  S4_deep_analysis: "from-violet-500/20 to-violet-900/5",
  S5_results: "from-emerald-500/20 to-emerald-900/5",
};

// ── Stage Spotlight — large clickable stage card with image ──────

function StageSpotlight({
  stage,
  isSelected,
  onClick,
}: {
  stage: StageState;
  isSelected: boolean;
  onClick: () => void;
}) {
  const Icon = STAGE_ICONS[stage.id] ?? Activity;
  const color = STAGE_COLORS[stage.id] ?? "#64748b";
  const gradient = STAGE_GRADIENTS[stage.id] ?? "from-slate-500/20 to-slate-900/5";
  const isRunning = stage.status === "running";
  const isDone = stage.status === "completed";
  const isFailed = stage.status === "failed";

  return (
    <button
      onClick={onClick}
      className={clsx(
        "group relative flex-shrink-0 w-[160px] rounded-xl border overflow-hidden transition-all duration-300 text-left",
        isSelected && "ring-2 scale-[1.02] shadow-lg",
        isRunning && "border-amber-500/50 shadow-amber-900/20",
        isDone && !isSelected && "border-slate-600/40 hover:border-slate-500/50",
        isFailed && "border-rose-600/40",
        !isRunning && !isDone && !isFailed && "border-slate-700/30 opacity-50",
      )}
      style={isSelected ? { borderColor: color, boxShadow: `0 0 20px ${color}30` } : undefined}
    >
      {/* Thumbnail area */}
      <div className={clsx("relative h-[100px] bg-gradient-to-b", gradient)}>
        {stage.thumbnail_b64 ? (
          <img
            src={`data:image/jpeg;base64,${stage.thumbnail_b64}`}
            alt={stage.name}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            {isRunning ? (
              <Loader2 size={28} className="text-amber-400 animate-spin" />
            ) : (
              <Icon size={28} style={{ color, opacity: 0.4 }} />
            )}
          </div>
        )}

        {/* Status indicator overlay */}
        <div className="absolute top-1.5 right-1.5">
          {isRunning ? (
            <div className="flex items-center gap-1 rounded-md bg-amber-900/80 backdrop-blur-sm px-1.5 py-0.5 border border-amber-600/40">
              <Loader2 size={9} className="text-amber-300 animate-spin" />
              <span className="text-[8px] font-bold text-amber-200 uppercase">Live</span>
            </div>
          ) : isDone ? (
            <CheckCircle2 size={14} style={{ color }} />
          ) : isFailed ? (
            <AlertTriangle size={14} className="text-rose-400" />
          ) : null}
        </div>

        {/* Color bar at bottom */}
        <div
          className="absolute bottom-0 left-0 right-0 h-[3px] transition-opacity"
          style={{ backgroundColor: color, opacity: isDone || isRunning ? 1 : 0.2 }}
        />
      </div>

      {/* Info area */}
      <div className="px-2.5 py-2 bg-[#0a1628]">
        <div className="flex items-center gap-1.5 mb-0.5">
          <Icon size={11} style={{ color }} />
          <span className="text-[10px] font-bold uppercase tracking-wider truncate" style={{ color }}>
            {stage.name}
          </span>
        </div>
        <p className="text-[8px] text-slate-500 leading-tight line-clamp-2">{stage.description}</p>
        {stage.latency_ms != null && (
          <div className="mt-1 flex items-center gap-1">
            <Clock size={8} className="text-slate-500" />
            <span
              className={clsx(
                "text-[9px] font-mono font-bold tabular-nums",
                stage.latency_ms < 50
                  ? "text-emerald-400"
                  : stage.latency_ms < 200
                  ? "text-amber-400"
                  : "text-rose-400",
              )}
            >
              {stage.latency_ms.toFixed(1)}ms
            </span>
          </div>
        )}
      </div>
    </button>
  );
}

// ── Analysis Results Panel ───────────────────────────────────────

function AnalysisResultsPanel({ frame }: { frame: FrameProcessingState }) {
  const data = frame.detection;
  if (!data) return null;

  const scene = (data.scene_description as string) ?? "";
  const objects = (data.objects_detected as Array<{ label: string; location: string; condition: string }>) ?? [];
  const anomalies = (data.anomalies as Array<{ type: string; description: string; severity: string; location: string }>) ?? [];
  const cot = (data.chain_of_thought as string) ?? "";
  const risk = (data.risk_level as string) ?? "";
  const riskAnalysis = (data.risk_analysis as string) ?? "";
  const impactAnalysis = (data.impact_analysis as string) ?? "";
  const recommendations = (data.recommendations as string[]) ?? [];
  const confidence = (data.confidence as number) ?? 0;
  const anomalyDetected = (data.anomaly_detected as boolean) ?? false;

  return (
    <div className="space-y-3 overflow-auto max-h-full pr-1">
      {/* Scene description */}
      {scene && (
        <div className="rounded-xl border border-cyan-900/30 bg-gradient-to-br from-cyan-950/30 to-transparent p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Eye size={12} className="text-cyan-400" />
            <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400">Scene Analysis</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">{scene}</p>
        </div>
      )}

      {/* Objects grid */}
      {objects.length > 0 && (
        <div className="rounded-xl border border-slate-700/30 bg-[#0c1628] p-3">
          <div className="flex items-center gap-1.5 mb-2">
            <Scan size={12} className="text-slate-400" />
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
              Objects Detected ({objects.length})
            </span>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {objects.map((obj, i) => (
              <div
                key={i}
                className="flex items-start gap-1.5 rounded-lg bg-slate-800/50 px-2 py-1.5 border border-slate-700/20"
              >
                <Target size={10} className="text-cyan-400 mt-0.5 flex-shrink-0" />
                <div className="min-w-0">
                  <span className="text-[10px] font-semibold text-slate-200 block truncate">{obj.label}</span>
                  <span className="text-[8px] text-slate-500 block truncate">{obj.location}</span>
                  {obj.condition && (
                    <span
                      className={clsx(
                        "text-[8px] font-bold uppercase",
                        obj.condition === "normal" ? "text-emerald-500" : "text-amber-400",
                      )}
                    >
                      {obj.condition}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Anomalies alert */}
      {anomalyDetected && anomalies.length > 0 && (
        <div className="rounded-xl border border-rose-800/40 bg-gradient-to-br from-rose-950/40 to-transparent p-3">
          <div className="flex items-center gap-1.5 mb-2">
            <AlertTriangle size={12} className="text-rose-400" />
            <span className="text-[10px] font-bold uppercase tracking-wider text-rose-400">
              Anomalies ({anomalies.length})
            </span>
          </div>
          <div className="space-y-1.5">
            {anomalies.map((a, i) => (
              <div key={i} className="rounded-lg bg-rose-950/30 border border-rose-800/20 px-2.5 py-2">
                <div className="flex items-center justify-between mb-0.5">
                  <span className="text-[10px] font-bold text-rose-200">{a.type}</span>
                  <span
                    className={clsx(
                      "text-[8px] font-black uppercase px-1.5 py-0.5 rounded",
                      a.severity === "critical"
                        ? "bg-rose-800/60 text-rose-200"
                        : a.severity === "high"
                        ? "bg-orange-800/60 text-orange-200"
                        : "bg-amber-800/60 text-amber-200",
                    )}
                  >
                    {a.severity}
                  </span>
                </div>
                <p className="text-[9px] text-rose-300/80 leading-relaxed">{a.description}</p>
                {a.location && (
                  <span className="text-[8px] text-rose-400/60 mt-0.5 block">Location: {a.location}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Risk & Confidence hud */}
      {(risk || confidence > 0) && (
        <div className="grid grid-cols-2 gap-2">
          {risk && (
            <div
              className={clsx(
                "rounded-xl border p-3 text-center",
                risk === "critical"
                  ? "border-rose-700/40 bg-rose-950/20"
                  : risk === "high"
                  ? "border-orange-700/40 bg-orange-950/20"
                  : risk === "medium"
                  ? "border-amber-700/40 bg-amber-950/20"
                  : "border-emerald-700/40 bg-emerald-950/20",
              )}
            >
              <Shield
                size={18}
                className={clsx(
                  "mx-auto mb-1",
                  risk === "critical"
                    ? "text-rose-400"
                    : risk === "high"
                    ? "text-orange-400"
                    : risk === "medium"
                    ? "text-amber-400"
                    : "text-emerald-400",
                )}
              />
              <span className="text-[10px] font-black uppercase tracking-wider text-slate-300 block">
                Risk Level
              </span>
              <span
                className={clsx(
                  "text-sm font-black uppercase",
                  risk === "critical"
                    ? "text-rose-300"
                    : risk === "high"
                    ? "text-orange-300"
                    : risk === "medium"
                    ? "text-amber-300"
                    : "text-emerald-300",
                )}
              >
                {risk}
              </span>
            </div>
          )}
          <div className="rounded-xl border border-slate-700/30 bg-[#0c1628] p-3 text-center">
            <Gauge size={18} className="mx-auto mb-1 text-slate-400" />
            <span className="text-[10px] font-black uppercase tracking-wider text-slate-400 block">
              Confidence
            </span>
            <span className="text-sm font-black tabular-nums text-slate-200">
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* Chain of Thought */}
      {cot && (
        <div className="rounded-xl border border-violet-900/30 bg-gradient-to-br from-violet-950/20 to-transparent p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Brain size={12} className="text-violet-400" />
            <span className="text-[10px] font-bold uppercase tracking-wider text-violet-400">
              Chain of Thought
            </span>
          </div>
          <p className="text-[10px] text-slate-300/90 leading-relaxed whitespace-pre-wrap">{cot}</p>
        </div>
      )}

      {/* Risk & Impact analysis */}
      {riskAnalysis && (
        <div className="rounded-xl border border-orange-900/30 bg-gradient-to-br from-orange-950/20 to-transparent p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <TrendingUp size={12} className="text-orange-400" />
            <span className="text-[10px] font-bold uppercase tracking-wider text-orange-400">
              Risk Analysis
            </span>
          </div>
          <p className="text-[10px] text-slate-300/90 leading-relaxed">{riskAnalysis}</p>
        </div>
      )}

      {impactAnalysis && (
        <div className="rounded-xl border border-amber-900/30 bg-gradient-to-br from-amber-950/20 to-transparent p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Info size={12} className="text-amber-400" />
            <span className="text-[10px] font-bold uppercase tracking-wider text-amber-400">
              Impact Analysis
            </span>
          </div>
          <p className="text-[10px] text-slate-300/90 leading-relaxed">{impactAnalysis}</p>
        </div>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="rounded-xl border border-emerald-900/30 bg-gradient-to-br from-emerald-950/20 to-transparent p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Lightbulb size={12} className="text-emerald-400" />
            <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400">
              Recommendations
            </span>
          </div>
          <ul className="space-y-1">
            {recommendations.map((r, i) => (
              <li key={i} className="flex items-start gap-1.5">
                <span className="text-emerald-500 mt-0.5 text-[9px]">&#x25B8;</span>
                <span className="text-[10px] text-slate-300/90 leading-relaxed">{r}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ── Timing Waterfall (horizontal) ────────────────────────────────

function TheaterWaterfall({ frame }: { frame: FrameProcessingState }) {
  const stages = frame.stages.filter((s) => s.latency_ms != null);
  if (stages.length === 0) return null;

  const maxMs = Math.max(...stages.map((s) => s.latency_ms!), 1);
  const totalMs = frame.total_latency_ms ?? stages.reduce((sum, s) => sum + (s.latency_ms ?? 0), 0);

  return (
    <div className="rounded-xl border border-slate-700/30 bg-[#0a1628] p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <Gauge size={11} className="text-slate-500" />
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
            Execution Timeline
          </span>
        </div>
        <span className="text-[11px] font-mono font-black tabular-nums text-slate-200">
          {totalMs.toFixed(0)}ms total
        </span>
      </div>
      <div className="space-y-1">
        {stages.map((stage) => {
          const color = STAGE_COLORS[stage.id] ?? "#64748b";
          const pct = (stage.latency_ms! / maxMs) * 100;
          return (
            <div key={stage.id} className="flex items-center gap-2">
              <span className="text-[9px] text-slate-500 w-24 truncate font-semibold">
                {stage.name}
              </span>
              <div className="flex-1 h-4 rounded-md bg-slate-800/60 overflow-hidden relative">
                <div
                  className="h-full rounded-md transition-all duration-700 ease-out relative"
                  style={{ width: `${pct}%`, backgroundColor: `${color}40` }}
                >
                  <div
                    className="absolute inset-y-0 left-0 rounded-md"
                    style={{ width: "100%", backgroundColor: color, opacity: 0.6 }}
                  />
                </div>
              </div>
              <span
                className="text-[10px] font-mono font-bold tabular-nums w-16 text-right"
                style={{ color }}
              >
                {stage.latency_ms!.toFixed(1)}ms
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main stage detail view ───────────────────────────────────────

function formatDetailValue(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "object") {
    try {
      return JSON.stringify(v, null, 0);
    } catch {
      return String(v);
    }
  }
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}

function StageDetailView({
  stage,
  frame,
}: {
  stage: StageState;
  frame: FrameProcessingState;
}) {
  const color = STAGE_COLORS[stage.id] ?? "#64748b";
  const Icon = STAGE_ICONS[stage.id] ?? Activity;
  const progressPct = useMemo(() => {
    const completed = frame.stages.filter((s) => s.status === "completed").length;
    return Math.round((completed / Math.max(frame.stages.length, 1)) * 100);
  }, [frame.stages]);

  return (
    <div className="flex flex-col h-full">
      {/* Stage header with progress indicator */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-slate-800/60 relative overflow-hidden">
        {/* Progress bar behind header */}
        <div
          className="absolute left-0 top-0 h-full transition-all duration-700 ease-out"
          style={{
            width: `${progressPct}%`,
            backgroundColor: `${color}08`,
          }}
        />
        <div className="relative flex items-center gap-2 w-full">
          <div
            className={clsx(
              "flex h-7 w-7 items-center justify-center rounded-lg transition-transform duration-300",
              stage.status === "running" && "animate-pulse scale-110",
            )}
            style={{ backgroundColor: `${color}20` }}
          >
            <Icon size={14} style={{ color }} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="text-xs font-bold text-slate-200 truncate">{stage.name}</h3>
              <span
                className={clsx(
                  "text-[7px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded",
                  stage.status === "running"
                    ? "bg-amber-950/40 text-amber-400"
                    : stage.status === "completed"
                    ? "bg-emerald-950/40 text-emerald-400"
                    : stage.status === "failed"
                    ? "bg-rose-950/40 text-rose-400"
                    : "bg-slate-800/40 text-slate-500",
                )}
              >
                {stage.status}
              </span>
            </div>
            <p className="text-[9px] text-slate-500">{stage.description}</p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {stage.latency_ms != null && (
              <span
                className={clsx(
                  "text-xs font-mono font-bold tabular-nums rounded-lg px-2 py-1",
                  stage.latency_ms < 50
                    ? "bg-emerald-950/30 text-emerald-400"
                    : stage.latency_ms < 200
                    ? "bg-amber-950/30 text-amber-400"
                    : "bg-rose-950/30 text-rose-400",
                )}
              >
                {stage.latency_ms.toFixed(1)}ms
              </span>
            )}
            <span className="text-[9px] font-mono text-slate-500">
              {progressPct}%
            </span>
          </div>
        </div>
      </div>

      {/* Stage image — large, prominent, with smooth transitions */}
      <div className="flex-1 min-h-0 relative bg-black/40 overflow-hidden">
        {stage.thumbnail_b64 ? (
          <img
            key={`${frame.frame_id}-${stage.id}`}
            src={`data:image/jpeg;base64,${stage.thumbnail_b64}`}
            alt={`${stage.name} output`}
            className="w-full h-full object-contain animate-[fadeIn_0.3s_ease-in-out]"
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <Icon size={40} className="mx-auto mb-2" style={{ color, opacity: 0.2 }} />
              <span className="text-[10px] text-slate-500 block">
                {stage.status === "running"
                  ? "Processing..."
                  : stage.status === "pending"
                  ? "Waiting..."
                  : "No preview available"}
              </span>
            </div>
          </div>
        )}

        {/* Stage status overlay — semi-transparent processing indicator */}
        {stage.status === "running" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/20 backdrop-blur-[0.5px]">
            <div className="flex flex-col items-center gap-2 rounded-xl bg-black/40 px-4 py-3">
              <Loader2 size={28} className="text-amber-400 animate-spin" />
              <span className="text-[10px] font-bold text-amber-300 tracking-wide">
                {stage.name}
              </span>
            </div>
          </div>
        )}

        {/* Corner badges for live context */}
        <div className="absolute top-2 left-2 flex items-center gap-1.5">
          <span
            className="rounded-md px-1.5 py-0.5 text-[8px] font-bold"
            style={{ backgroundColor: `${color}30`, color }}
          >
            Stage {stage.index + 1}/{frame.stages.length}
          </span>
        </div>
        {frame.total_latency_ms != null && (
          <div className="absolute top-2 right-2">
            <span className="rounded-md bg-black/50 px-1.5 py-0.5 text-[8px] font-mono font-bold text-slate-300 backdrop-blur-sm">
              Total: {frame.total_latency_ms.toFixed(0)}ms
            </span>
          </div>
        )}
      </div>

      {/* Stage details — with safe formatting for nested objects */}
      {Object.keys(stage.details).length > 0 && (
        <div className="px-4 py-2 border-t border-slate-800/60 bg-[#080f1e]">
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(stage.details).map(([k, v]) => {
              const val = formatDetailValue(v);
              if (val.length > 80) return null; // skip very long values in compact view
              return (
                <span
                  key={k}
                  className="rounded-md bg-slate-800/60 px-2 py-1 text-[9px] text-slate-400 font-mono border border-slate-700/20"
                >
                  {k}: <span className="text-slate-300">{val}</span>
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Recent frames carousel ───────────────────────────────────────

function RecentFramesBar({
  frames,
  selectedId,
  onSelect,
}: {
  frames: FrameProcessingState[];
  selectedId: string | null;
  onSelect: (frame: FrameProcessingState) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  if (frames.length === 0) return null;

  return (
    <div className="border-t border-slate-800/60 bg-[#080f1e]">
      <div className="flex items-center gap-2 px-3 py-1.5">
        <Clock size={10} className="text-slate-500 flex-shrink-0" />
        <span className="text-[9px] font-bold uppercase tracking-wider text-slate-500 flex-shrink-0">
          Recent ({frames.length})
        </span>
        <div
          ref={scrollRef}
          className="flex gap-1.5 overflow-x-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-slate-700"
        >
          {frames.map((f) => {
            const isFlagged = f.status === "flagged";
            const isSelected = f.frame_id === selectedId;
            return (
              <button
                key={f.frame_id}
                onClick={() => onSelect(f)}
                className={clsx(
                  "flex-shrink-0 rounded-lg border overflow-hidden transition-all",
                  isSelected
                    ? "border-slate-400 ring-1 ring-slate-400/30"
                    : isFlagged
                    ? "border-rose-700/40 hover:border-rose-600/60"
                    : "border-slate-700/30 hover:border-slate-600/50",
                )}
              >
                {f.frame_thumbnail ? (
                  <img
                    src={`data:image/jpeg;base64,${f.frame_thumbnail}`}
                    alt=""
                    className="h-10 w-16 object-cover"
                  />
                ) : (
                  <div
                    className={clsx(
                      "h-10 w-16 flex items-center justify-center",
                      isFlagged ? "bg-rose-950/20" : "bg-slate-800/40",
                    )}
                  >
                    <Activity size={10} className="text-slate-600" />
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── SVG Sparkline Chart ──────────────────────────────────────────

function Sparkline({
  data,
  width = 120,
  height = 32,
  color = "#4ade80",
  fillOpacity = 0.15,
}: {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  fillOpacity?: number;
}) {
  if (data.length < 2) {
    return (
      <svg width={width} height={height} className="opacity-30">
        <line x1={0} y1={height / 2} x2={width} y2={height / 2} stroke={color} strokeWidth={1} strokeDasharray="3,3" />
      </svg>
    );
  }

  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  const step = width / (data.length - 1);
  const points = data.map((v, i) => ({
    x: i * step,
    y: height - ((v - min) / range) * (height - 4) - 2,
  }));
  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${points[points.length - 1].x.toFixed(1)},${height} L0,${height} Z`;

  return (
    <svg width={width} height={height} className="overflow-visible">
      <defs>
        <linearGradient id={`spark-${color.replace("#", "")}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={fillOpacity} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={areaPath} fill={`url(#spark-${color.replace("#", "")})`} />
      <path d={linePath} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
      {/* Current value dot */}
      <circle cx={points[points.length - 1].x} cy={points[points.length - 1].y} r={2.5} fill={color} />
    </svg>
  );
}

// ── Real-Time Metrics Dashboard ──────────────────────────────────

function RealTimeMetricsDashboard({ metrics, stages }: { metrics: PipelineMetrics; stages: StageState[] }) {
  const elapsedSec = metrics.startedAt ? Math.floor((Date.now() - metrics.startedAt) / 1000) : 0;
  const elapsedStr = elapsedSec > 60
    ? `${Math.floor(elapsedSec / 60)}m ${elapsedSec % 60}s`
    : `${elapsedSec}s`;

  return (
    <div className="space-y-2.5">
      {/* Summary row */}
      <div className="grid grid-cols-4 gap-1.5">
        {[
          {
            icon: Timer,
            label: "Avg Latency",
            value: metrics.avgPipelineLatency > 0 ? `${metrics.avgPipelineLatency.toFixed(0)}ms` : "—",
            color: "text-sky-400",
            bg: "bg-sky-950/30",
            border: "border-sky-800/30",
          },
          {
            icon: Gauge,
            label: "Throughput",
            value: metrics.throughput > 0 ? `${metrics.throughput.toFixed(2)}/s` : "—",
            color: "text-emerald-400",
            bg: "bg-emerald-950/30",
            border: "border-emerald-800/30",
          },
          {
            icon: Flame,
            label: "Anomalies",
            value: String(metrics.anomalyCount),
            color: metrics.anomalyCount > 0 ? "text-rose-400" : "text-slate-400",
            bg: metrics.anomalyCount > 0 ? "bg-rose-950/30" : "bg-slate-800/30",
            border: metrics.anomalyCount > 0 ? "border-rose-800/30" : "border-slate-700/30",
          },
          {
            icon: Clock,
            label: "Elapsed",
            value: elapsedStr,
            color: "text-violet-400",
            bg: "bg-violet-950/30",
            border: "border-violet-800/30",
          },
        ].map((m) => (
          <div key={m.label} className={clsx("rounded-lg border p-2 text-center", m.bg, m.border)}>
            <m.icon size={12} className={clsx("mx-auto mb-0.5", m.color)} />
            <div className={clsx("text-[11px] font-mono font-black tabular-nums", m.color)}>{m.value}</div>
            <div className="text-[7px] font-bold uppercase tracking-wider text-slate-500">{m.label}</div>
          </div>
        ))}
      </div>

      {/* Total Pipeline Latency sparkline */}
      {metrics.totalLatencyHistory.length > 1 && (
        <div className="rounded-lg border border-slate-700/30 bg-[#0a1628] p-2.5">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <BarChart3 size={10} className="text-sky-400" />
              <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Pipeline Latency Trend</span>
            </div>
            <span className="text-[9px] font-mono text-slate-500">
              {metrics.totalLatencyHistory[metrics.totalLatencyHistory.length - 1]?.latency_ms.toFixed(0)}ms
            </span>
          </div>
          <Sparkline
            data={metrics.totalLatencyHistory.map((p) => p.latency_ms)}
            width={280}
            height={36}
            color="#38bdf8"
          />
        </div>
      )}

      {/* FPS sparkline */}
      {metrics.fpsHistory.length > 1 && (
        <div className="rounded-lg border border-slate-700/30 bg-[#0a1628] p-2.5">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <Zap size={10} className="text-emerald-400" />
              <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400">FPS Trend</span>
            </div>
            <span className="text-[9px] font-mono text-slate-500">
              {metrics.fpsHistory[metrics.fpsHistory.length - 1]?.fps.toFixed(1)}
            </span>
          </div>
          <Sparkline
            data={metrics.fpsHistory.map((p) => p.fps)}
            width={280}
            height={28}
            color="#4ade80"
          />
        </div>
      )}

      {/* Per-stage latency breakdown */}
      <div className="rounded-lg border border-slate-700/30 bg-[#0a1628] p-2.5">
        <div className="flex items-center gap-1.5 mb-2">
          <Layers size={10} className="text-slate-400" />
          <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Stage Performance</span>
        </div>
        <div className="space-y-1.5">
          {stages.map((stage) => {
            const color = STAGE_COLORS[stage.id] ?? "#64748b";
            const hist = metrics.stageLatencyHistory[stage.id] ?? [];
            const avg = metrics.stageAvgLatency[stage.id] ?? 0;
            const max = metrics.stageMaxLatency[stage.id] ?? 0;
            const min = metrics.stageMinLatency[stage.id] ?? 0;
            const Icon = STAGE_ICONS[stage.id] ?? Activity;

            return (
              <div key={stage.id} className="flex items-center gap-2">
                <div className="flex items-center gap-1 w-20 flex-shrink-0">
                  <Icon size={9} style={{ color }} />
                  <span className="text-[8px] font-bold uppercase truncate" style={{ color }}>
                    {stage.id.split("_").slice(1).join("_")}
                  </span>
                </div>
                <div className="flex-1">
                  <Sparkline
                    data={hist.length > 1 ? hist.map((p) => p.latency_ms) : [0, 0]}
                    width={140}
                    height={18}
                    color={color}
                    fillOpacity={0.1}
                  />
                </div>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <span className="text-[8px] font-mono text-slate-500" title="min">
                    {min > 0 ? `${min.toFixed(0)}` : "—"}
                  </span>
                  <span className="text-[9px] font-mono font-bold tabular-nums" style={{ color }}>
                    {avg > 0 ? `${avg.toFixed(0)}ms` : "—"}
                  </span>
                  <span className="text-[8px] font-mono text-slate-500" title="max">
                    {max > 0 ? `${max.toFixed(0)}` : "—"}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Queue depth sparkline */}
      {metrics.queueHistory.length > 1 && (
        <div className="rounded-lg border border-slate-700/30 bg-[#0a1628] p-2.5">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <Layers size={10} className="text-amber-400" />
              <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Queue Depth</span>
            </div>
            <span className="text-[9px] font-mono text-slate-500">
              {metrics.queueHistory[metrics.queueHistory.length - 1]?.depth}
            </span>
          </div>
          <Sparkline
            data={metrics.queueHistory.map((p) => p.depth)}
            width={280}
            height={24}
            color="#f59e0b"
          />
        </div>
      )}
    </div>
  );
}

// ── Main Theater Component ───────────────────────────────────────

export function PipelineVisionTheater() {
  const connected = usePipelineMonitorStore((s) => s.connected);
  const currentFrame = usePipelineMonitorStore((s) => s.currentFrame);
  const recentFrames = usePipelineMonitorStore((s) => s.recentFrames);
  const queueDepth = usePipelineMonitorStore((s) => s.queueDepth);
  const throttled = usePipelineMonitorStore((s) => s.throttled);
  const totalProcessed = usePipelineMonitorStore((s) => s.totalProcessed);
  const fps = usePipelineMonitorStore((s) => s.fps);
  const lastCompletedFrame = usePipelineMonitorStore((s) => s.lastCompletedFrame);
  const metrics = usePipelineMonitorStore((s) => s.metrics);
  const setPipelineTheater = useUiStore((s) => s.setPipelineTheater);

  const [selectedStageIdx, setSelectedStageIdx] = useState<number>(0);
  const [viewingFrame, setViewingFrame] = useState<FrameProcessingState | null>(null);
  const [showAnalysis, setShowAnalysis] = useState(true);
  const [showMetrics, setShowMetrics] = useState(true);

  // Use viewingFrame (from history) or currentFrame (live), with lastCompletedFrame to prevent flicker
  const activeFrame = viewingFrame ?? currentFrame ?? lastCompletedFrame ?? recentFrames[0] ?? null;

  // Ref keeps latest activeFrame for use inside effects without adding it as a dependency
  const activeFrameRef = useRef(activeFrame);
  activeFrameRef.current = activeFrame;

  // Stable key that only changes when stage statuses actually change (not on every store update)
  const stageStatusKey = useMemo(() => {
    if (!activeFrame) return "";
    return activeFrame.frame_id + ":" + activeFrame.stages.map((s) => s.status).join(",");
  }, [activeFrame]);

  // Gather stages for metrics display
  const allStages = useMemo(() => activeFrame?.stages ?? [], [activeFrame]);

  // Auto-track current running stage — depends on stageStatusKey for stability
  useEffect(() => {
    if (viewingFrame) return; // don't auto-track when viewing history
    const frame = activeFrameRef.current;
    if (!frame) return;
    const runningIdx = frame.stages.findIndex((s) => s.status === "running");
    if (runningIdx >= 0) {
      setSelectedStageIdx(runningIdx);
    } else {
      // Show last completed stage
      const lastDone = [...frame.stages].reverse().findIndex((s) => s.status === "completed");
      if (lastDone >= 0) {
        setSelectedStageIdx(frame.stages.length - 1 - lastDone);
      }
    }
  }, [stageStatusKey, viewingFrame]);

  const selectedStage = activeFrame?.stages[selectedStageIdx] ?? null;

  // Keyboard navigation
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setPipelineTheater(false);
      } else if (e.key === "ArrowLeft" && activeFrame) {
        setSelectedStageIdx((i) => Math.max(0, i - 1));
      } else if (e.key === "ArrowRight" && activeFrame) {
        setSelectedStageIdx((i) => Math.min(activeFrame.stages.length - 1, i + 1));
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [activeFrame, setPipelineTheater]);

  // Handle history frame selection
  const handleRecentSelect = useCallback((frame: FrameProcessingState) => {
    setViewingFrame((prev) => (prev?.frame_id === frame.frame_id ? null : frame));
    setSelectedStageIdx(0);
  }, []);

  return (
    <div className="flex flex-col h-full bg-[#070e1b] overflow-hidden">
      {/* ── Top bar ─────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800/60 bg-[#0a1628]">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <Activity
              size={16}
              className={clsx(connected ? "text-emerald-500 animate-pulse" : "text-slate-500")}
            />
            <span className="text-xs font-bold uppercase tracking-[0.15em] text-slate-300">
              Pipeline Vision
            </span>
          </div>

          {/* Connection badge */}
          <div
            className={clsx(
              "flex items-center gap-1 rounded-md px-2 py-0.5 border text-[9px] font-bold uppercase",
              connected
                ? "border-emerald-700/40 bg-emerald-950/30 text-emerald-400"
                : "border-rose-700/40 bg-rose-950/30 text-rose-400",
            )}
          >
            {connected ? <Wifi size={9} /> : <WifiOff size={9} />}
            {connected ? "Connected" : "Disconnected"}
          </div>

          {throttled && (
            <span className="rounded-md bg-amber-950/50 border border-amber-800/40 px-2 py-0.5 text-[9px] font-bold text-amber-400 uppercase">
              Throttled
            </span>
          )}

          {viewingFrame && (
            <button
              onClick={() => setViewingFrame(null)}
              className="flex items-center gap-1 rounded-md bg-sky-950/50 border border-sky-700/40 px-2 py-0.5 text-[9px] font-bold text-sky-400 uppercase hover:bg-sky-950/70 transition-colors"
            >
              <Activity size={9} />
              Return to Live
            </button>
          )}
        </div>

        {/* Live metrics */}
        <div className="flex items-center gap-3">
          {[
            { icon: Gauge, label: "FPS", value: fps.toFixed(1), active: fps > 0 },
            { icon: Layers, label: "Queue", value: String(queueDepth), active: queueDepth > 0 },
            { icon: Zap, label: "Processed", value: totalProcessed.toLocaleString(), active: true },
          ].map((m) => (
            <div key={m.label} className="flex items-center gap-1.5">
              <m.icon size={10} className={m.active ? "text-slate-400" : "text-slate-600"} />
              <span className="text-[10px] font-mono font-bold tabular-nums text-slate-300">{m.value}</span>
              <span className="text-[8px] text-slate-500 uppercase">{m.label}</span>
            </div>
          ))}

          {/* Metrics panel toggle */}
          <button
            onClick={() => setShowMetrics(!showMetrics)}
            className={clsx(
              "rounded-md px-2 py-1 text-[9px] font-bold uppercase border transition-colors",
              showMetrics
                ? "border-sky-700/40 bg-sky-950/30 text-sky-400"
                : "border-slate-700/30 bg-slate-800/30 text-slate-500 hover:text-slate-400",
            )}
          >
            <BarChart3 size={9} className="inline mr-1" />
            Metrics
          </button>

          {/* Analysis panel toggle */}
          <button
            onClick={() => setShowAnalysis(!showAnalysis)}
            className={clsx(
              "rounded-md px-2 py-1 text-[9px] font-bold uppercase border transition-colors",
              showAnalysis
                ? "border-violet-700/40 bg-violet-950/30 text-violet-400"
                : "border-slate-700/30 bg-slate-800/30 text-slate-500 hover:text-slate-400",
            )}
          >
            Analysis
          </button>

          {/* Close button */}
          <button
            onClick={() => setPipelineTheater(false)}
            className="rounded-md bg-slate-800/60 border border-slate-700/30 p-1.5 text-slate-400 hover:text-white hover:bg-slate-700/60 transition-colors"
            title="Exit theater mode (Esc)"
          >
            <Minimize2 size={14} />
          </button>
        </div>
      </div>

      {activeFrame ? (
        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          {/* ── Frame banner ───────────────────────────────────── */}
          <div
            className={clsx(
              "flex items-center justify-between px-4 py-1.5 border-b",
              activeFrame.status === "flagged"
                ? "border-rose-800/40 bg-rose-950/15"
                : activeFrame.status === "processing"
                ? "border-emerald-800/30 bg-emerald-950/10"
                : "border-slate-800/40 bg-slate-900/20",
            )}
          >
            <div className="flex items-center gap-3">
              <div
                className={clsx(
                  "h-2.5 w-2.5 rounded-full",
                  activeFrame.status === "processing"
                    ? "bg-emerald-400 animate-pulse"
                    : activeFrame.status === "flagged"
                    ? "bg-rose-400 animate-pulse"
                    : "bg-slate-400",
                )}
              />
              <span className="text-[10px] font-mono font-bold text-slate-300">{activeFrame.frame_id}</span>
              <span className="text-[10px] text-slate-500">
                {activeFrame.camera_id} &middot;{" "}
                {new Date(activeFrame.started_at * 1000).toLocaleTimeString()}
              </span>
            </div>
            <div className="flex items-center gap-2">
              {/* Pipeline progress dots */}
              <div className="flex items-center gap-1">
                {activeFrame.stages.map((stage, i) => {
                  const color = STAGE_COLORS[stage.id] ?? "#64748b";
                  const isRunning = stage.status === "running";
                  const isDone = stage.status === "completed";
                  const isSel = i === selectedStageIdx;
                  return (
                    <button
                      key={stage.id}
                      onClick={() => setSelectedStageIdx(i)}
                      className={clsx(
                        "transition-all duration-300 rounded-full",
                        isSel ? "h-3 w-3 ring-2 ring-white/20" : "h-2 w-2",
                        isRunning && "animate-pulse",
                        !isDone && !isRunning && "opacity-30",
                      )}
                      style={{
                        backgroundColor: isDone || isRunning ? (isRunning ? "#fbbf24" : color) : "#334155",
                      }}
                      title={stage.name}
                    />
                  );
                })}
              </div>
              {activeFrame.total_latency_ms != null && (
                <span className="text-[10px] font-mono font-bold text-slate-300 tabular-nums">
                  {activeFrame.total_latency_ms.toFixed(0)}ms
                </span>
              )}
              {activeFrame.status === "flagged" && (
                <span className="flex items-center gap-1 rounded-md bg-rose-900/60 border border-rose-600/40 px-2 py-0.5">
                  <AlertTriangle size={10} className="text-rose-300" />
                  <span className="text-[9px] font-bold text-rose-200 uppercase">Anomaly</span>
                </span>
              )}
            </div>
          </div>

          {/* ── Main content: Stage detail + Analysis panel ──── */}
          <div className="flex-1 flex min-h-0 overflow-hidden">
            {/* Left: Stage detail view */}
            <div className="flex-1 min-w-0 flex flex-col">
              {selectedStage && (
                <StageDetailView stage={selectedStage} frame={activeFrame} />
              )}
            </div>

            {/* Right panels: Metrics + Analysis (toggleable) */}
            {(showMetrics || (showAnalysis && activeFrame.detection)) && (
              <div className="w-[340px] flex-shrink-0 border-l border-slate-800/60 bg-[#0a1220] overflow-auto p-3 space-y-3">
                {showMetrics && (
                  <RealTimeMetricsDashboard metrics={metrics} stages={allStages} />
                )}
                {showAnalysis && activeFrame.detection && (
                  <AnalysisResultsPanel frame={activeFrame} />
                )}
              </div>
            )}
          </div>

          {/* ── Stage strip (horizontal, full-width) ──────────── */}
          <div className="border-t border-slate-800/60 bg-[#080f1e]">
            <div className="flex items-center gap-2 px-3 py-2 overflow-x-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-slate-700">
              {activeFrame.stages.map((stage, i) => (
                <StageSpotlight
                  key={stage.id}
                  stage={stage}
                  isSelected={i === selectedStageIdx}
                  onClick={() => setSelectedStageIdx(i)}
                />
              ))}
            </div>
          </div>

          {/* ── Timing waterfall ────────────────────────────────── */}
          <div className="border-t border-slate-800/60 bg-[#080f1e] px-3 py-2">
            <TheaterWaterfall frame={activeFrame} />
          </div>

          {/* ── Recent frames bar ──────────────────────────────── */}
          <RecentFramesBar
            frames={recentFrames}
            selectedId={viewingFrame?.frame_id ?? null}
            onSelect={handleRecentSelect}
          />
        </div>
      ) : (
        /* ── Empty / idle state ────────────────────────────────── */
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-3">
            {connected ? (
              <>
                <div className="relative mx-auto w-16 h-16">
                  <div className="absolute inset-0 rounded-full border-2 border-slate-700 animate-ping opacity-20" />
                  <div className="absolute inset-2 rounded-full border-2 border-slate-600 animate-ping opacity-30 animation-delay-150" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Eye size={24} className="text-slate-500" />
                  </div>
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-400">Pipeline Idle</p>
                  <p className="text-[10px] text-slate-600 mt-0.5">
                    Waiting for frames... Start a camera or upload a video to begin processing.
                  </p>
                </div>
              </>
            ) : (
              <>
                <WifiOff size={32} className="mx-auto text-slate-600" />
                <div>
                  <p className="text-sm font-semibold text-slate-400">Not Connected</p>
                  <p className="text-[10px] text-slate-600 mt-0.5">
                    Start the backend to enable pipeline monitoring.
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
