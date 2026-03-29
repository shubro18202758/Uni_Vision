import { useEffect, useState, useRef } from "react";
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
} from "lucide-react";
import clsx from "clsx";
import { usePipelineMonitorStore, type StageState, type FrameProcessingState } from "../../store/pipelineMonitorStore";
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

// ── Stage card ───────────────────────────────────────────────────

function StageCard({ stage, isActive }: { stage: StageState; isActive: boolean }) {
  const Icon = STAGE_ICONS[stage.id] ?? Activity;
  const isRunning = stage.status === "running";
  const isDone = stage.status === "completed";
  const isFailed = stage.status === "failed";

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

      {/* Thumbnail */}
      {stage.thumbnail_b64 && (
        <div className="mt-1 rounded border border-slate-700/40 overflow-hidden bg-black/30">
          <img
            src={`data:image/jpeg;base64,${stage.thumbnail_b64}`}
            alt={`${stage.name} output`}
            className="w-full h-auto max-h-24 object-contain"
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
          return (
            <div key={stage.id} className="flex items-center gap-0.5">
              <div
                className={clsx(
                  "h-2 w-2 rounded-full transition-all duration-300",
                  isRunning && "bg-amber-400 ring-2 ring-amber-400/30 animate-pulse",
                  isDone && "bg-emerald-400",
                  isFailed && "bg-rose-400",
                  !isRunning && !isDone && !isFailed && "bg-slate-700",
                )}
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

// ── Frame thumbnail panel ────────────────────────────────────────

function FrameThumbnail({ frame }: { frame: FrameProcessingState }) {
  const latestThumb = frame.frame_thumbnail ||
    [...frame.stages].reverse().find((s) => s.thumbnail_b64)?.thumbnail_b64;

  if (!latestThumb) {
    return (
      <div className="flex h-28 items-center justify-center rounded-lg border border-dashed border-slate-700/40 bg-[#0a1628]">
        <div className="text-center">
          <Eye size={18} className="mx-auto text-slate-600 mb-1" />
          <span className="text-[9px] text-slate-600">Awaiting preview...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-slate-700/40 overflow-hidden bg-black/40 relative group">
      <img
        src={`data:image/jpeg;base64,${latestThumb}`}
        alt="Current frame"
        className="w-full h-auto max-h-40 object-contain"
      />
      <div className="absolute top-1 left-1.5 flex items-center gap-1 rounded bg-black/60 px-1 py-0.5">
        <div className={clsx(
          "h-1.5 w-1.5 rounded-full",
          frame.status === "processing" ? "bg-emerald-400 animate-pulse" : frame.status === "flagged" ? "bg-rose-400" : "bg-slate-400",
        )} />
        <span className="text-[8px] font-bold text-white uppercase tracking-wider">
          {frame.status === "processing" ? "LIVE" : frame.status === "flagged" ? "FLAG" : "DONE"}
        </span>
      </div>
      <div className="absolute bottom-1 right-1.5 rounded bg-black/60 px-1 py-0.5">
        <span className="text-[8px] font-mono text-slate-300">{frame.camera_id}</span>
      </div>
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
      {completedStages.map((stage) => (
        <div key={stage.id} className="flex items-center gap-2">
          <span className="text-[8px] text-slate-500 w-14 truncate font-mono">{stage.id.split("_")[0]}</span>
          <div className="flex-1 h-2.5 rounded bg-slate-800/80 overflow-hidden">
            <div
              className={clsx(
                "h-full rounded transition-all duration-500",
                stage.latency_ms! < 50 ? "bg-emerald-600" : stage.latency_ms! < 200 ? "bg-amber-600" : "bg-rose-600",
              )}
              style={{ width: `${(stage.latency_ms! / maxMs) * 100}%` }}
            />
          </div>
          <span className="text-[8px] font-mono font-bold text-slate-400 tabular-nums w-14 text-right">
            {stage.latency_ms!.toFixed(1)}ms
          </span>
        </div>
      ))}
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
      <div className={clsx(
        "h-1.5 w-1.5 rounded-full flex-shrink-0",
        frame.status === "flagged" ? "bg-rose-400" : "bg-emerald-400",
      )} />
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

  const [expandStages, setExpandStages] = useState(true);
  const [expandRecent, setExpandRecent] = useState(false);
  const [selectedFrame, setSelectedFrame] = useState<FrameProcessingState | null>(null);
  const [flagModal, setFlagModal] = useState<Record<string, unknown> | null>(null);

  // Auto-transition: when a flag is raised, show detection detail modal
  useEffect(() => {
    if (lastFlag && !flagConsumed && lastFlag.detection) {
      setFlagModal(lastFlag.detection);
      consumeFlag();
    }
  }, [lastFlag, flagConsumed, consumeFlag]);

  // WS monitoring is now started at AppShell level — no local lifecycle needed

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
        {throttled && (
          <span className="rounded bg-amber-950/50 border border-amber-800/40 px-1.5 py-0.5 text-[8px] font-bold text-amber-400 uppercase tracking-wider">
            THROTTLED
          </span>
        )}
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

      {/* Current frame processing */}
      {currentFrame ? (
        <div className="space-y-2.5">
          {/* Frame banner */}
          <div className={clsx(
            "flex items-center justify-between rounded-lg border px-3 py-2",
            currentFrame.status === "flagged"
              ? "border-rose-700/40 bg-rose-950/20"
              : "border-emerald-700/30 bg-emerald-950/10",
          )}>
            <div className="flex items-center gap-2">
              <div className={clsx(
                "h-2 w-2 rounded-full",
                currentFrame.status === "processing" ? "bg-emerald-400 animate-pulse" : "bg-rose-400 animate-pulse",
              )} />
              <div>
                <span className="text-[9px] font-mono font-bold text-slate-200 block">
                  {currentFrame.frame_id}
                </span>
                <span className="text-[8px] text-slate-500">
                  {currentFrame.camera_id} • {new Date(currentFrame.started_at * 1000).toLocaleTimeString()}
                </span>
              </div>
            </div>
            {currentFrame.status === "flagged" && (
              <span className="flex items-center gap-1 rounded bg-rose-900/50 border border-rose-700/40 px-2 py-0.5">
                <AlertTriangle size={10} className="text-rose-400" />
                <span className="text-[9px] font-bold text-rose-400 uppercase">FLAG RAISED</span>
              </span>
            )}
          </div>

          {/* Frame thumbnail */}
          <FrameThumbnail frame={currentFrame} />

          {/* Progress visualization */}
          <PipelineProgress frame={currentFrame} />

          {/* Timing waterfall */}
          <TimingWaterfall frame={currentFrame} />

          {/* Collapsible stage detail cards */}
          <button
            onClick={() => setExpandStages(!expandStages)}
            className="flex w-full items-center justify-between rounded border border-slate-700/30 bg-[#0c1628] px-2 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400 hover:bg-[#111e36] transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <Layers size={10} />
              Stage Details ({currentFrame.stages.length})
            </span>
            {expandStages ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
          </button>

          {expandStages && (
            <div className="space-y-1.5">
              {currentFrame.stages.map((stage) => (
                <StageCard
                  key={stage.id}
                  stage={stage}
                  isActive={stage.index === currentFrame.current_stage_index}
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
            <FrameThumbnail frame={selectedFrame} />
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
