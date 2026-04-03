import { useEffect, useState } from "react";
import {
  Cpu,
  Gauge,
  HardDrive,
  Activity,
  Package,
  Film,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Zap,
  Server,
  BarChart3,
  Layers,
} from "lucide-react";
import clsx from "clsx";
import type { TechnicalMetricsResponse, TechnicalBottleneck } from "../../types/api";
import { fetchTechnicalMetrics } from "../../services/api";
import type { DetectionContext } from "../../services/api";

interface Props {
  detectionId: string;
  context?: DetectionContext;
}

/* ── Palette ──────────────────────────────────────────────────── */

const SEVERITY_BADGE: Record<string, string> = {
  low: "text-emerald-400 bg-emerald-950/40 border-emerald-800/40",
  medium: "text-amber-400 bg-amber-950/40 border-amber-800/40",
  high: "text-orange-400 bg-orange-950/40 border-orange-800/40",
  critical: "text-rose-400 bg-rose-950/40 border-rose-800/40",
  unknown: "text-slate-400 bg-slate-800/40 border-slate-700",
};

/* ── Helper components ────────────────────────────────────────── */

function SectionHeader({ icon: Icon, title }: { icon: typeof Cpu; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      <Icon size={14} className="text-cyan-400" />
      <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">{title}</h4>
    </div>
  );
}

function MetricRow({ label, value, mono }: { label: string; value: string | number; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-slate-700/20 last:border-0">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{label}</span>
      <span className={clsx("text-xs text-slate-200", mono && "font-mono")}>{String(value)}</span>
    </div>
  );
}

function MetricCard({ label, value, sub, color }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div className="rounded-lg border border-slate-700/40 bg-[#111e36]/70 p-3 flex flex-col gap-0.5">
      <span className="text-[9px] font-bold uppercase tracking-wider text-slate-500">{label}</span>
      <span className={clsx("text-lg font-bold tabular-nums", color ?? "text-slate-100")}>{String(value)}</span>
      {sub && <span className="text-[10px] text-slate-500">{sub}</span>}
    </div>
  );
}

function VramBar({ used, total }: { used: number; total: number }) {
  const pct = Math.min(100, (used / total) * 100);
  const barColor = pct > 85 ? "bg-rose-500" : pct > 70 ? "bg-amber-500" : "bg-cyan-500";
  return (
    <div className="mt-2">
      <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
        <span>VRAM Utilisation</span>
        <span className="font-mono text-slate-300">{used} / {total} MB ({pct.toFixed(0)}%)</span>
      </div>
      <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
        <div className={clsx("h-full rounded-full transition-all", barColor)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function BottleneckCard({ b }: { b: TechnicalBottleneck }) {
  return (
    <div className="rounded-lg border border-amber-800/30 bg-amber-950/15 p-3">
      <div className="flex items-center gap-2 mb-1">
        <AlertTriangle size={11} className="text-amber-400" />
        <span className="text-xs font-bold text-slate-200">{b.component}</span>
        <span className="rounded-full border border-slate-700 bg-slate-800/50 px-1.5 py-0.5 text-[9px] font-bold uppercase text-slate-400">
          {b.type}
        </span>
      </div>
      <p className="text-[11px] text-slate-400 mb-1">{b.description}</p>
      <p className="text-[10px] text-cyan-400/80 flex items-center gap-1">
        <CheckCircle2 size={9} /> {b.mitigation}
      </p>
    </div>
  );
}

/* ── Main Component ───────────────────────────────────────────── */

export function TechnicalMetricsDashboard({ detectionId, context }: Props) {
  const [data, setData] = useState<TechnicalMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchTechnicalMetrics(detectionId, context)
      .then((res) => { if (!cancelled) setData(res); })
      .catch((err) => { if (!cancelled) setError(err?.message ?? "Failed to load"); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [detectionId, context]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-3">
        <Activity size={24} className="text-cyan-400 animate-pulse" />
        <p className="text-xs text-slate-500">Loading technical metrics…</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="rounded-lg border border-rose-800/30 bg-rose-950/20 p-6 text-center">
        <AlertTriangle size={20} className="mx-auto mb-2 text-rose-400" />
        <p className="text-xs text-rose-300">{error ?? "No data"}</p>
      </div>
    );
  }

  const inf = data.inference_metrics;
  const acc = data.accuracy_metrics;
  const pip = data.pipeline_metrics;
  const hw = data.hardware_metrics;
  const libs = data.libraries;
  const media = data.media_constraints;
  const bottlenecks = data.bottlenecks ?? [];

  return (
    <div className="space-y-6">
      {/* ── Header badges ── */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="rounded-full border border-cyan-800/40 bg-cyan-950/30 px-2.5 py-0.5 text-[10px] font-bold uppercase text-cyan-400">
          {data.anomaly_type}
        </span>
        <span className={clsx("rounded-full border px-2.5 py-0.5 text-[10px] font-bold uppercase", SEVERITY_BADGE[data.anomaly_severity] ?? SEVERITY_BADGE.unknown)}>
          {data.anomaly_severity}
        </span>
      </div>

      {/* ── Inference Metrics ── */}
      <section>
        <SectionHeader icon={Cpu} title="Inference Metrics" />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
          <MetricCard label="Model" value={String(inf.model_name ?? "—")} sub={String(inf.model_family ?? "")} />
          <MetricCard label="Quantization" value={String(inf.quantization ?? "—")} sub={String(inf.model_size ?? "")} />
          <MetricCard label="Inference Time" value={`${inf.inference_time_ms ?? 0}ms`} color="text-cyan-400" />
          <MetricCard label="Tokens/sec" value={String(inf.tokens_per_second_est ?? "—")} color="text-emerald-400" />
        </div>
        <div className="rounded-lg border border-slate-700/40 bg-[#0f1b2f] p-3">
          <MetricRow label="Context Window" value={`${inf.context_window ?? 0} tokens`} mono />
          <MetricRow label="Max Output" value={`${inf.max_output_tokens ?? 0} tokens`} mono />
          <MetricRow label="Temperature" value={String(inf.temperature ?? "—")} mono />
          <MetricRow label="Top-P" value={String(inf.top_p ?? "—")} mono />
          <MetricRow label="Top-K" value={String(inf.top_k ?? "—")} mono />
          <MetricRow label="Repeat Penalty" value={String(inf.repeat_penalty ?? "—")} mono />
          <MetricRow label="Batch Size" value={String(inf.num_batch ?? "—")} mono />
        </div>
      </section>

      {/* ── Accuracy Metrics ── */}
      <section>
        <SectionHeader icon={Gauge} title="Accuracy & Detection Metrics" />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
          <MetricCard label="Confidence" value={String(acc.confidence_pct ?? "—")} color="text-emerald-400" />
          <MetricCard label="Precision (est)" value={String(acc.detection_precision_est ?? "—")} color="text-blue-400" />
          <MetricCard label="Recall (est)" value={String(acc.detection_recall_est ?? "—")} color="text-purple-400" />
          <MetricCard label="F1 Score" value={String(acc.f1_score_est ?? "—")} color="text-amber-400" />
        </div>
        <div className="rounded-lg border border-slate-700/40 bg-[#0f1b2f] p-3">
          <MetricRow label="Validation Status" value={String(acc.validation_status ?? "—")} />
          <MetricRow label="Risk Level" value={String(acc.risk_level ?? "—")} />
          <MetricRow label="False Positive Est" value={String(acc.false_positive_estimate ?? "—")} mono />
        </div>
      </section>

      {/* ── Pipeline Performance ── */}
      <section>
        <SectionHeader icon={BarChart3} title="Pipeline Performance" />
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-3">
          <MetricCard label="Total Latency" value={`${pip.total_pipeline_latency_ms ?? 0}ms`} color="text-cyan-400" />
          <MetricCard label="LLM Inference" value={`${pip.llm_inference_latency_ms ?? 0}ms`} color="text-amber-400" sub="Dominant cost" />
          <MetricCard label="Throughput" value={`${pip.throughput_fps ?? 0} FPS`} color="text-emerald-400" />
        </div>
        <div className="rounded-lg border border-slate-700/40 bg-[#0f1b2f] p-3">
          <MetricRow label="Detection Latency" value={`${pip.detection_latency_ms ?? 0}ms`} mono />
          <MetricRow label="Preprocessing" value={`${pip.preprocessing_latency_ms ?? 0}ms`} mono />
          <MetricRow label="Postprocessing" value={`${pip.postprocessing_latency_ms ?? 0}ms`} mono />
          <MetricRow label="Error Rate" value={`${((Number(pip.component_error_rate) || 0) * 100).toFixed(1)}%`} mono />
          <MetricRow label="Active Path" value={String(pip.active_path ?? "—")} />
        </div>
        {/* Pipeline stages */}
        {Array.isArray(pip.pipeline_stages) && (
          <div className="flex items-center gap-1 mt-3 flex-wrap">
            <Layers size={11} className="text-slate-500" />
            {(pip.pipeline_stages as string[]).map((s, i) => (
              <span key={i} className="rounded border border-slate-700/40 bg-slate-800/50 px-1.5 py-0.5 text-[9px] font-mono text-slate-400">
                {s}
              </span>
            ))}
          </div>
        )}
      </section>

      {/* ── Hardware Utilisation ── */}
      <section>
        <SectionHeader icon={HardDrive} title="Hardware Utilisation" />
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-3">
          <MetricCard label="GPU" value={String(hw.gpu_model ?? "—")} />
          <MetricCard label="VRAM Total" value={`${hw.gpu_vram_total_mb ?? 0} MB`} />
          <MetricCard
            label="Headroom"
            value={`${hw.gpu_vram_headroom_mb ?? 0} MB`}
            color={Number(hw.gpu_vram_headroom_mb ?? 0) < 1024 ? "text-amber-400" : "text-emerald-400"}
          />
        </div>
        <div className="rounded-lg border border-slate-700/40 bg-[#0f1b2f] p-3">
          <VramBar used={Number(hw.gpu_vram_allocated_mb ?? 0)} total={Number(hw.gpu_vram_total_mb ?? 8192)} />
          <div className="grid grid-cols-2 gap-x-4 mt-3">
            <MetricRow label="Weights" value={`${hw.vram_budget_weights_mb ?? 0} MB`} mono />
            <MetricRow label="KV Cache" value={`${hw.vram_budget_kv_cache_mb ?? 0} MB`} mono />
            <MetricRow label="Vision" value={`${hw.vram_budget_vision_mb ?? 0} MB`} mono />
            <MetricRow label="System" value={`${hw.vram_budget_system_mb ?? 0} MB`} mono />
          </div>
          <div className="mt-3 pt-2 border-t border-slate-700/20">
            <MetricRow label="CPU Arch" value={String(hw.cpu_architecture ?? "—")} />
            <MetricRow label="OS" value={String(hw.os_platform ?? "—")} />
            <MetricRow label="Python" value={String(hw.python_version ?? "—")} mono />
          </div>
        </div>
      </section>

      {/* ── Libraries & Dependencies ── */}
      <section>
        <SectionHeader icon={Package} title="Libraries & Dependencies" />
        <div className="rounded-lg border border-slate-700/40 bg-[#0f1b2f] p-3">
          {Object.entries(libs).map(([key, val]) => (
            <MetricRow key={key} label={key.replace(/_/g, " ")} value={val} />
          ))}
        </div>
      </section>

      {/* ── Media Constraints ── */}
      <section>
        <SectionHeader icon={Film} title="Media Constraints" />
        <div className="rounded-lg border border-slate-700/40 bg-[#0f1b2f] p-3">
          <MetricRow label="Formats" value={Array.isArray(media.supported_formats) ? (media.supported_formats as string[]).join(", ") : "—"} />
          <MetricRow label="Max Resolution" value={String(media.max_resolution ?? "—")} />
          <MetricRow label="Target FPS" value={String(media.target_fps ?? "—")} mono />
          <MetricRow label="Frame Extraction" value={String(media.frame_extraction ?? "—")} />
          <MetricRow label="Image Encoding" value={String(media.image_encoding ?? "—")} />
          <MetricRow label="Max Frame Size" value={`${media.max_frame_size_kb ?? "—"} KB`} mono />
          <MetricRow label="Colour Space" value={String(media.colour_space ?? "—")} />
        </div>
      </section>

      {/* ── Bottlenecks ── */}
      {bottlenecks.length > 0 && (
        <section>
          <SectionHeader icon={AlertTriangle} title={`Processing Bottlenecks (${bottlenecks.length})`} />
          <div className="space-y-2">
            {bottlenecks.map((b, i) => (
              <BottleneckCard key={i} b={b} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
