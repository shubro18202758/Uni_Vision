import { useEffect, useState, useRef, useMemo } from "react";
import { Activity, BarChart3, Camera, Eye, Gauge, RefreshCw, ShieldCheck, TrendingUp, Cpu, Zap, Timer, Hash } from "lucide-react";
import { fetchStats } from "../../services/api";
import { useDetectionStore } from "../../store/detectionStore";
import { useGraphStore } from "../../store/graphStore";
import clsx from "clsx";

interface StatsSnapshot {
  active_streams: number;
  total_detections: number;
  total_frames: number;
  raw: Record<string, number>;
}

/** Tiny SVG sparkline chart — renders an array of numeric values as an inline area+line chart. */
function Sparkline({ data, color = "#60a5fa", height = 28, width = 80 }: { data: number[]; color?: string; height?: number; width?: number }) {
  if (data.length < 2) return <div style={{ width, height }} />;
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  });
  const linePath = `M${points.join(" L")}`;
  const areaPath = `${linePath} L${width},${height} L0,${height} Z`;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="shrink-0">
      <defs>
        <linearGradient id={`spark-${color.replace("#", "")}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={areaPath} fill={`url(#spark-${color.replace("#", "")})`} />
      <path d={linePath} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
      <circle cx={Number(points[points.length - 1].split(",")[0])} cy={Number(points[points.length - 1].split(",")[1])} r="2" fill={color} />
    </svg>
  );
}

/** Use a rolling history buffer for sparkline data. */
function useRollingHistory(value: number, maxLen = 20) {
  const ref = useRef<number[]>([]);
  ref.current = [...ref.current, value].slice(-maxLen);
  return ref.current;
}

function StatCard({ icon: Icon, label, value, sub, accent, sparkData, sparkColor }: {
  icon: typeof Activity;
  label: string;
  value: string | number;
  sub?: string;
  accent?: string;
  sparkData?: number[];
  sparkColor?: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-slate-700/40 bg-[#111e36] p-3 relative overflow-hidden">
      <div className={clsx("flex h-9 w-9 items-center justify-center rounded-lg shrink-0", accent || "bg-slate-800/60")}>
        <Icon size={16} className={accent ? "text-white" : "text-slate-400"} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{label}</p>
        <p className="text-lg font-bold text-slate-100 leading-tight">{value}</p>
        {sub && <p className="truncate text-[10px] text-slate-500">{sub}</p>}
      </div>
      {sparkData && sparkData.length >= 2 && (
        <div className="absolute right-2 bottom-1 opacity-60">
          <Sparkline data={sparkData} color={sparkColor || "#60a5fa"} width={60} height={24} />
        </div>
      )}
    </div>
  );
}

function MiniBar({ label, value, max, color }: { label: string; value: number; max: number; color?: string }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-2 text-[10px]">
      <span className="w-20 truncate font-medium text-slate-400">{label}</span>
      <div className="flex-1 h-2 rounded-full bg-slate-800 overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: color || "#06b6d4" }} />
      </div>
      <span className="w-8 text-right tabular-nums text-slate-500">{value}</span>
    </div>
  );
}

export function AnalyticsDashboard() {
  const [stats, setStats] = useState<StatsSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const liveEvents = useDetectionStore((s) => s.liveEvents);
  const detections = liveEvents;
  const blocks = useGraphStore((s) => s.blocks);
  const executionStates = useGraphStore((s) => s.executionStates);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const raw = await fetchStats();
      setStats({
        active_streams: raw.active_streams ?? 0,
        total_detections: raw.total_detections ?? 0,
        total_frames: raw.total_frames ?? 0,
        raw,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch stats");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 10_000);
    return () => clearInterval(id);
  }, []);

  // Sparkline rolling histories
  const detectionHistory = useRollingHistory(detections.length);
  const streamHistory = useRollingHistory(stats?.active_streams ?? 0);
  const frameHistory = useRollingHistory(stats?.total_frames ?? 0);

  // Pipeline execution summary
  const execSummary = useMemo(() => {
    const running = Object.values(executionStates).filter((s) => s === "running").length;
    const success = Object.values(executionStates).filter((s) => s === "success").length;
    const errored = Object.values(executionStates).filter((s) => s === "error").length;
    return { running, success, errored, total: blocks.length };
  }, [blocks.length, executionStates]);

  // Compute per-camera breakdown from recent detections
  const cameraBreakdown = detections.reduce<Record<string, number>>((acc, d) => {
    acc[d.camera_id] = (acc[d.camera_id] || 0) + 1;
    return acc;
  }, {});
  const cameraEntries = Object.entries(cameraBreakdown).sort(([, a], [, b]) => b - a).slice(0, 6);
  const maxCamCount = cameraEntries.length > 0 ? cameraEntries[0][1] : 0;

  // Confidence distribution from recent detections
  const confBuckets = [
    { label: "90-100%", min: 0.9, max: 1.01, color: "#4ade80" },
    { label: "70-90%", min: 0.7, max: 0.9, color: "#60a5fa" },
    { label: "50-70%", min: 0.5, max: 0.7, color: "#facc15" },
    { label: "<50%", min: 0, max: 0.5, color: "#f43f5e" },
  ];
  const confCounts = confBuckets.map((b) => ({
    label: b.label,
    color: b.color,
    count: detections.filter((d) => d.ocr_confidence >= b.min && d.ocr_confidence < b.max).length,
  }));
  const maxConfCount = Math.max(...confCounts.map((c) => c.count), 1);

  // Validation status counts
  const validCount = detections.filter((d) => d.validation_status === "valid").length;
  const invalidCount = detections.filter((d) => d.validation_status === "invalid").length;
  const pendingCount = detections.length - validCount - invalidCount;
  const validationTotal = validCount + invalidCount + pendingCount || 1;

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 size={14} className="text-slate-400" />
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
            Analytics
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[8px] text-slate-600 tabular-nums">
            {detections.length} events
          </span>
          <button onClick={refresh} className="rounded p-1 hover:bg-slate-800" title="Refresh stats">
            <RefreshCw size={12} className={clsx("text-slate-400", loading && "animate-spin")} />
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded border border-rose-800/40 bg-rose-950/30 px-2 py-1 text-[10px] text-rose-400">{error}</div>
      )}

      {/* KPI Cards — now with sparklines */}
      <div className="grid grid-cols-2 gap-2">
        <StatCard
          icon={Camera}
          label="Active Streams"
          value={stats?.active_streams ?? "—"}
          sparkData={streamHistory}
          sparkColor="#60a5fa"
        />
        <StatCard
          icon={Eye}
          label="Total Detections"
          value={stats?.total_detections ?? "—"}
          sparkData={detectionHistory}
          sparkColor="#4ade80"
        />
        <StatCard
          icon={Gauge}
          label="Total Frames"
          value={stats?.total_frames ?? "—"}
          sparkData={frameHistory}
          sparkColor="#60a5fa"
        />
        <StatCard
          icon={TrendingUp}
          label="Recent (local)"
          value={detections.length}
          sub="In detection feed"
          sparkData={detectionHistory}
          sparkColor="#facc15"
        />
      </div>

      {/* Pipeline Health Strip */}
      <div className="rounded-xl border border-slate-700/40 bg-[#111e36] px-3 py-2.5">
        <div className="flex items-center gap-2 mb-2">
          <Zap size={12} className="text-slate-400" />
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Pipeline Health
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full bg-blue-400 animate-pulse" />
            <span className="text-[9px] text-slate-400">{execSummary.running} active</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full bg-emerald-400" />
            <span className="text-[9px] text-slate-400">{execSummary.success} done</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full bg-red-400" />
            <span className="text-[9px] text-slate-400">{execSummary.errored} failed</span>
          </div>
          <div className="flex-1" />
          <span className="text-[8px] text-slate-600 tabular-nums">{execSummary.total} nodes</span>
        </div>
        {/* Mini pipeline progress bar */}
        {execSummary.total > 0 && (
          <div className="mt-2 h-1.5 w-full rounded-full bg-slate-800 overflow-hidden flex">
            <div
              className="h-full bg-emerald-500 transition-all duration-500"
              style={{ width: `${(execSummary.success / execSummary.total) * 100}%` }}
            />
            <div
              className="h-full bg-blue-400 transition-all duration-500"
              style={{ width: `${(execSummary.running / execSummary.total) * 100}%` }}
            />
            <div
              className="h-full bg-red-500 transition-all duration-500"
              style={{ width: `${(execSummary.errored / execSummary.total) * 100}%` }}
            />
          </div>
        )}
      </div>

      {/* Confidence Distribution */}
      <div className="rounded-xl border border-slate-700/40 bg-[#111e36] p-3">
        <div className="flex items-center gap-2 mb-2">
          <Activity size={12} className="text-slate-400" />
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            OCR Confidence
          </span>
        </div>
        <div className="space-y-1.5">
          {confCounts.map((c) => (
            <MiniBar key={c.label} label={c.label} value={c.count} max={maxConfCount} color={c.color} />
          ))}
        </div>
      </div>

      {/* Validation Status — now with proportional bar */}
      <div className="rounded-xl border border-slate-700/40 bg-[#111e36] p-3">
        <div className="flex items-center gap-2 mb-2">
          <ShieldCheck size={12} className="text-slate-400" />
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Validation
          </span>
        </div>
        {/* Segmented validation bar */}
        <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden flex mb-2">
          <div className="h-full bg-emerald-500 transition-all duration-500" style={{ width: `${(validCount / validationTotal) * 100}%` }} />
          <div className="h-full bg-rose-500 transition-all duration-500" style={{ width: `${(invalidCount / validationTotal) * 100}%` }} />
          <div className="h-full bg-amber-400 transition-all duration-500" style={{ width: `${(pendingCount / validationTotal) * 100}%` }} />
        </div>
        <div className="flex gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
            <span className="text-slate-400">Valid</span>
            <span className="font-bold text-slate-200">{validCount}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2.5 w-2.5 rounded-full bg-rose-500" />
            <span className="text-slate-400">Invalid</span>
            <span className="font-bold text-slate-200">{invalidCount}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2.5 w-2.5 rounded-full bg-amber-400" />
            <span className="text-slate-400">Pending</span>
            <span className="font-bold text-slate-200">{pendingCount}</span>
          </div>
        </div>
      </div>

      {/* Camera Breakdown */}
      {cameraEntries.length > 0 && (
        <div className="rounded-xl border border-slate-700/40 bg-[#111e36] p-3">
          <div className="flex items-center gap-2 mb-2">
            <Camera size={12} className="text-slate-400" />
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Camera Activity
            </span>
          </div>
          <div className="space-y-1.5">
            {cameraEntries.map(([cam, count]) => (
              <MiniBar key={cam} label={cam} value={count} max={maxCamCount} />
            ))}
          </div>
        </div>
      )}

      {/* System Info Footer — adds density to fill empty space */}
      <div className="rounded-xl border border-slate-700/30 bg-[#0c1a2d] p-3 space-y-2">
        <div className="flex items-center gap-2 mb-1">
          <Cpu size={12} className="text-slate-500" />
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-600">
            System
          </span>
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[9px]">
          <div className="flex items-center justify-between">
            <span className="text-slate-500">GPU Memory</span>
            <span className="text-slate-300 font-bold tabular-nums">8 GB VRAM</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Backend</span>
            <span className="text-slate-300 font-bold">FastAPI</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Detection</span>
            <span className="text-slate-300 font-bold">YOLOv8</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">OCR Engine</span>
            <span className="text-slate-300 font-bold">PaddleOCR</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Inference</span>
            <span className="text-slate-300 font-bold">Ollama</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Pipeline</span>
            <span className="text-slate-300 font-bold">{blocks.length} nodes</span>
          </div>
        </div>
      </div>
    </div>
  );
}
