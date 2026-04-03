import { useEffect, useState } from "react";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  BarChart,
  Bar,
  CartesianGrid,
  Cell,
  Treemap,
  FunnelChart,
  Funnel,
  LabelList,
} from "recharts";
import {
  AlertTriangle,
  Activity,
  Shield,
  Layers,
  Zap,
  Eye,
  Server,
  Database,
  Clock,
  FileWarning,
  TrendingDown,
  Radio,
  BarChart3,
  GitBranch,
} from "lucide-react";
import clsx from "clsx";
import type {
  ImpactAnalysisResponse,
  ImpactDimension,
  CascadeNode,
  ResourceImpact,
  CoverageGap,
  DataCorruptionVector,
  ComplianceImpact,
  CorrelationPair,
} from "../../types/api";
import { fetchImpactAnalysis } from "../../services/api";
import type { DetectionContext } from "../../services/api";

interface Props {
  detectionId: string;
  context?: DetectionContext;
}

/* ── Palette ──────────────────────────────────────────────────── */

const SEVERITY_COLOR: Record<string, string> = {
  negligible: "#22c55e",
  minor: "#3b82f6",
  moderate: "#eab308",
  severe: "#f97316",
  catastrophic: "#ef4444",
};

const RISK_LEVEL_COLOR: Record<string, string> = {
  low: "#22c55e",
  medium: "#eab308",
  high: "#f97316",
  critical: "#ef4444",
};

const DOMAIN_ICON: Record<string, typeof Activity> = {
  operational: Activity,
  surveillance: Eye,
  data_quality: Database,
  cascading: Zap,
  temporal: Clock,
  resource: Server,
  compliance: Shield,
};

const TREEMAP_COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6", "#ec4899"];

/* ── Impact Gauge ─────────────────────────────────────────────── */

function ImpactGauge({ score, severity }: { score: number; severity: string }) {
  const color = SEVERITY_COLOR[severity] || SEVERITY_COLOR.moderate;
  const pct = Math.min(100, Math.max(0, score));

  return (
    <div className="relative flex flex-col items-center">
      <svg width="130" height="75" viewBox="0 0 130 75">
        <path d="M 10 70 A 55 55 0 0 1 120 70" fill="none" stroke="#1e293b" strokeWidth="9" strokeLinecap="round" />
        <path
          d="M 10 70 A 55 55 0 0 1 120 70"
          fill="none"
          stroke={color}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={`${(pct / 100) * 173} 173`}
          className="transition-all duration-700"
        />
        <text x="65" y="58" textAnchor="middle" className="text-xl font-black" fill={color}>
          {score.toFixed(0)}
        </text>
        <text x="65" y="73" textAnchor="middle" className="text-[8px] font-bold uppercase tracking-wider" fill="#94a3b8">
          {severity}
        </text>
      </svg>
    </div>
  );
}

/* ── Resource Gauge Bar ───────────────────────────────────────── */

function ResourceBar({ r }: { r: ResourceImpact }) {
  const color = RISK_LEVEL_COLOR[r.risk_level] || "#64748b";
  return (
    <div className="py-2 border-b border-slate-700/20 last:border-0">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] font-medium text-slate-200">{r.resource}</span>
        <span className="text-[9px] font-bold uppercase" style={{ color }}>{r.risk_level}</span>
      </div>
      <div className="relative h-3 rounded-full bg-slate-800 overflow-hidden">
        <div
          className="absolute left-0 top-0 h-full rounded-full transition-all duration-500"
          style={{ width: `${r.current_usage_pct}%`, backgroundColor: color }}
        />
        <div
          className="absolute top-0 h-full w-0.5 bg-white/40"
          style={{ left: `${Math.min(100, r.projected_peak_pct)}%` }}
          title={`Projected peak: ${r.projected_peak_pct.toFixed(0)}%`}
        />
      </div>
      <div className="flex items-center justify-between mt-0.5">
        <span className="text-[8px] text-slate-600">{r.current_usage_pct.toFixed(0)}% used</span>
        <span className="text-[8px] text-slate-600">headroom: {r.headroom_pct.toFixed(0)}%</span>
        <span className="text-[8px] text-slate-600">exhaust: {r.time_to_exhaustion}</span>
      </div>
    </div>
  );
}

/* ── Cascade Flow ─────────────────────────────────────────────── */

function CascadeFlow({ nodes }: { nodes: CascadeNode[] }) {
  return (
    <div className="space-y-0">
      {nodes.map((node, i) => {
        const color = SEVERITY_COLOR[node.severity] || "#64748b";
        return (
          <div key={i} className="flex items-start gap-2">
            <div className="flex flex-col items-center flex-shrink-0 mt-1">
              <div className="h-3 w-3 rounded-full border-2" style={{ borderColor: color, backgroundColor: `${color}30` }} />
              {i < nodes.length - 1 && <div className="w-px h-8 bg-slate-700/50" />}
            </div>
            <div className="flex-1 min-w-0 pb-2">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-bold text-slate-200">{node.component}</span>
                <span className="text-[8px] font-bold uppercase px-1 rounded" style={{ color, backgroundColor: `${color}15` }}>
                  {node.severity}
                </span>
              </div>
              <p className="text-[9px] text-slate-400">{node.failure_mode}</p>
              <div className="flex gap-3 mt-0.5">
                <span className="text-[8px] text-slate-600">TTF: {node.time_to_failure}</span>
                <span className="text-[8px] text-slate-600">Prob: {(node.probability * 100).toFixed(0)}%</span>
                {node.downstream.length > 0 && (
                  <span className="text-[8px] text-slate-600">→ {node.downstream.join(", ")}</span>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Correlation Matrix ───────────────────────────────────────── */

function CorrelationMatrix({ pairs }: { pairs: CorrelationPair[] }) {
  return (
    <div className="space-y-1.5">
      {pairs.map((p, i) => {
        const absCorr = Math.abs(p.correlation);
        const isPositive = p.correlation >= 0;
        const barColor = isPositive ? "#f97316" : "#3b82f6";
        const barWidth = absCorr * 100;
        return (
          <div key={i} className="rounded-lg bg-slate-800/30 p-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] text-slate-300">
                {p.metric_a} ↔ {p.metric_b}
              </span>
              <span className="text-[9px] font-mono font-bold" style={{ color: barColor }}>
                {p.correlation > 0 ? "+" : ""}{p.correlation.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${barWidth}%`, backgroundColor: barColor }}
                />
              </div>
              <span className="text-[8px] text-slate-600 w-24 text-right">{p.relationship.replace(/_/g, " ")}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Heatmap Grid ─────────────────────────────────────────────── */

function ComponentHeatmap({ cells }: { cells: { component: string; time_bucket: string; health_pct: number; anomaly_count: number }[] }) {
  const components = [...new Set(cells.map((c) => c.component))];
  const buckets = [...new Set(cells.map((c) => c.time_bucket))];

  const cellMap = new Map<string, { health_pct: number; anomaly_count: number }>();
  for (const c of cells) {
    cellMap.set(`${c.component}|${c.time_bucket}`, { health_pct: c.health_pct, anomaly_count: c.anomaly_count });
  }

  const healthToColor = (pct: number) => {
    if (pct >= 80) return "bg-emerald-900/50 text-emerald-400";
    if (pct >= 60) return "bg-yellow-900/40 text-yellow-400";
    if (pct >= 40) return "bg-amber-900/40 text-amber-400";
    return "bg-rose-900/40 text-rose-400";
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[9px]">
        <thead>
          <tr>
            <th className="text-left text-slate-600 font-medium pb-1 pr-2">Component</th>
            {buckets.map((b) => (
              <th key={b} className="text-center text-slate-600 font-medium pb-1 px-1">{b}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {components.map((comp) => (
            <tr key={comp}>
              <td className="text-[9px] text-slate-400 py-0.5 pr-2 whitespace-nowrap">{comp}</td>
              {buckets.map((b) => {
                const cell = cellMap.get(`${comp}|${b}`);
                const val = cell?.health_pct ?? 100;
                return (
                  <td key={b} className="px-0.5 py-0.5">
                    <div className={clsx("rounded px-1.5 py-0.5 text-center font-mono font-bold", healthToColor(val))}>
                      {val.toFixed(0)}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── Custom Tooltip ───────────────────────────────────────────── */

const DarkTooltipStyle = {
  background: "#0d1b2e",
  border: "1px solid #334155",
  borderRadius: 8,
  fontSize: 10,
  color: "#e2e8f0",
};

/* ── Main Dashboard ───────────────────────────────────────────── */

export function ImpactAnalysisDashboard({ detectionId, context }: Props) {
  const [data, setData] = useState<ImpactAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchImpactAnalysis(detectionId, context)
      .then((r) => { if (!cancelled) setData(r); })
      .catch((e) => { if (!cancelled) setError(e.message || "Failed to load impact analysis"); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [detectionId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-10">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-600 border-t-slate-300" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-rose-800/40 bg-rose-950/30 p-4 text-center">
        <AlertTriangle size={16} className="mx-auto mb-1 text-rose-400" />
        <p className="text-xs text-rose-400">{error}</p>
      </div>
    );
  }

  if (!data) return null;

  const sevColor = SEVERITY_COLOR[data.overall_severity] || SEVERITY_COLOR.moderate;

  /* Prepare radar data from impact dimensions */
  const radarData = data.impact_dimensions.map((d) => ({
    domain: d.title.split(" ").slice(0, 2).join(" "),
    score: d.score,
    fullTitle: d.title,
    description: d.description,
  }));

  /* Temporal propagation chart data */
  const temporalData = data.temporal_propagation.map((t) => ({
    time: t.time_offset,
    coverage: t.surveillance_coverage_pct,
    quality: t.data_quality_pct,
    stability: t.system_stability_pct,
    accuracy: t.detection_accuracy_pct,
    trust: t.operator_trust_pct,
    framesLost: t.cumulative_frames_lost,
  }));

  /* Treemap data from impact dimensions */
  const treemapData = data.impact_dimensions.map((d, i) => ({
    name: d.title.replace(" Impact", "").replace(" Degradation", ""),
    size: Math.max(5, d.score),
    color: TREEMAP_COLORS[i % TREEMAP_COLORS.length],
    severity: d.severity,
  }));

  /* Funnel data */
  const funnelData = data.processing_funnel.map((f) => ({
    name: f.stage,
    value: f.total_frames,
    successful: f.successful,
    failed: f.failed,
    drop_rate: f.drop_rate_pct,
    fill: f.bottleneck ? "#ef4444" : "#3b82f6",
  }));

  /* Resource bar chart data */
  const resBarData = data.resource_impacts.map((r) => ({
    name: r.resource,
    current: r.current_usage_pct,
    projected: r.projected_peak_pct,
    headroom: r.headroom_pct,
  }));

  return (
    <div className="space-y-5 p-1">
      {/* ── Header: Gauge + Summary ────────────────────────────── */}
      <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
        <div className="flex items-start gap-4">
          <ImpactGauge score={data.overall_impact_score} severity={data.overall_severity} />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <Zap size={13} className="flex-shrink-0" style={{ color: sevColor }} />
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                Exhaustive Impact Assessment
              </span>
            </div>
            <p className="text-[11px] text-slate-300 leading-relaxed">{data.summary}</p>
            {data.analysis_time_ms > 0 && (
              <p className="text-[9px] text-slate-600 mt-1">Analyzed in {data.analysis_time_ms.toFixed(1)}ms</p>
            )}
          </div>
        </div>
      </div>

      {/* ── Impact Dimension Cards ─────────────────────────────── */}
      <div className="grid grid-cols-2 gap-2">
        {data.impact_dimensions.map((dim) => {
          const dColor = SEVERITY_COLOR[dim.severity] || "#64748b";
          const Icon = DOMAIN_ICON[dim.domain] || Activity;
          return (
            <div key={dim.domain} className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-3">
              <div className="flex items-center gap-1.5 mb-1.5">
                <Icon size={11} style={{ color: dColor }} />
                <span className="text-[9px] font-bold uppercase tracking-wider text-slate-500">
                  {dim.domain.replace(/_/g, " ")}
                </span>
              </div>
              <div className="flex items-end gap-2 mb-1">
                <span className="text-xl font-black" style={{ color: dColor }}>{dim.score.toFixed(0)}</span>
                <span className="text-[9px] text-slate-500 mb-0.5">/ 100</span>
                <span className="text-[8px] font-bold uppercase ml-auto" style={{ color: dColor }}>{dim.severity}</span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden mb-1.5">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${dim.score}%`, backgroundColor: dColor }}
                />
              </div>
              <p className="text-[9px] text-slate-400 leading-relaxed">{dim.description}</p>
              {dim.affected_components.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-1.5">
                  {dim.affected_components.map((c) => (
                    <span key={c} className="rounded bg-slate-800/60 border border-slate-700/30 px-1 py-0 text-[7px] text-slate-500">{c}</span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Impact Treemap ─────────────────────────────────────── */}
      <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
        <div className="flex items-center gap-1.5 mb-3">
          <Layers size={12} className="text-slate-500" />
          <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Impact Domain Treemap
          </h4>
        </div>
        <ResponsiveContainer width="100%" height={160}>
          <Treemap
            data={treemapData}
            dataKey="size"
            stroke="#0a1628"
            isAnimationActive
          >
            {treemapData.map((entry, i) => (
              <Cell key={i} fill={`${entry.color}80`} />
            ))}
            <Tooltip
              contentStyle={DarkTooltipStyle}
              formatter={(value, _name, props) => [
                `Score: ${Number(value).toFixed(0)} — ${(props as unknown as { payload: { name: string; severity: string } }).payload.severity}`,
                (props as unknown as { payload: { name: string } }).payload.name,
              ]}
            />
          </Treemap>
        </ResponsiveContainer>
        <div className="flex flex-wrap gap-2 mt-2">
          {treemapData.map((t, i) => (
            <div key={i} className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: t.color }} />
              <span className="text-[8px] text-slate-500">{t.name}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Radar: Multi-Dimension Impact ──────────────────────── */}
      <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
        <div className="flex items-center gap-1.5 mb-3">
          <Radio size={12} className="text-slate-500" />
          <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Multi-Dimensional Impact Radar
          </h4>
        </div>
        <ResponsiveContainer width="100%" height={250}>
          <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
            <PolarGrid stroke="#1e293b" />
            <PolarAngleAxis dataKey="domain" tick={{ fill: "#94a3b8", fontSize: 8 }} tickLine={false} />
            <PolarRadiusAxis domain={[0, 100]} tick={{ fill: "#475569", fontSize: 8 }} axisLine={false} tickCount={5} />
            <Radar name="Impact" dataKey="score" stroke={sevColor} fill={sevColor} fillOpacity={0.2} strokeWidth={2} />
            <Tooltip contentStyle={DarkTooltipStyle} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Temporal Propagation (Stacked Area) ────────────────── */}
      {temporalData.length > 0 && (
        <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <TrendingDown size={12} className="text-rose-400" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-rose-400/80">
              Temporal Impact Propagation (Live Feed)
            </h4>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={temporalData}>
              <defs>
                <linearGradient id="covGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="qualGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="stabGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#eab308" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#eab308" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="trustGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fill: "#64748b", fontSize: 8 }} tickLine={false} axisLine={false} />
              <YAxis domain={[0, 100]} tick={{ fill: "#475569", fontSize: 8 }} tickLine={false} axisLine={false} width={25} />
              <Tooltip contentStyle={DarkTooltipStyle} />
              <Area type="monotone" dataKey="coverage" name="Coverage" stroke="#22c55e" fill="url(#covGrad)" strokeWidth={1.5} dot={false} />
              <Area type="monotone" dataKey="quality" name="Data Quality" stroke="#3b82f6" fill="url(#qualGrad)" strokeWidth={1.5} dot={false} />
              <Area type="monotone" dataKey="stability" name="Stability" stroke="#eab308" fill="url(#stabGrad)" strokeWidth={1.5} dot={false} />
              <Area type="monotone" dataKey="accuracy" name="Accuracy" stroke="#f97316" fill="url(#accGrad)" strokeWidth={1.5} dot={false} />
              <Area type="monotone" dataKey="trust" name="Operator Trust" stroke="#8b5cf6" fill="url(#trustGrad)" strokeWidth={1.5} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-3 mt-2">
            {[
              { label: "Coverage", color: "#22c55e" },
              { label: "Data Quality", color: "#3b82f6" },
              { label: "Stability", color: "#eab308" },
              { label: "Accuracy", color: "#f97316" },
              { label: "Operator Trust", color: "#8b5cf6" },
            ].map((l) => (
              <div key={l.label} className="flex items-center gap-1">
                <span className="h-1.5 w-3 rounded-full" style={{ backgroundColor: l.color }} />
                <span className="text-[8px] text-slate-500">{l.label}</span>
              </div>
            ))}
          </div>

          {/* Temporal detail list */}
          <div className="mt-3 space-y-1.5">
            {data.temporal_propagation.map((t, i) => (
              <div key={i} className="flex items-center gap-3 rounded-lg bg-slate-800/30 px-2 py-1.5">
                <span className="text-[9px] font-mono text-slate-500 w-10">{t.time_offset}</span>
                <span className="text-[9px] text-slate-400 flex-1">{t.description}</span>
                <span className="text-[8px] font-mono text-rose-400">{t.cumulative_frames_lost} lost</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Processing Funnel ──────────────────────────────────── */}
      {funnelData.length > 0 && (
        <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <BarChart3 size={12} className="text-slate-500" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Detection Processing Funnel
            </h4>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <FunnelChart>
              <Tooltip contentStyle={DarkTooltipStyle} />
              <Funnel dataKey="value" data={funnelData} isAnimationActive>
                {funnelData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
                <LabelList position="center" content={(props) => {
                  const { x, y, width, height, value, name } = props as unknown as { x?: number; y?: number; width?: number; height?: number; value?: number; name?: string };
                  return (
                    <text x={(x ?? 0) + (width ?? 0) / 2} y={(y ?? 0) + (height ?? 0) / 2} fill="#e2e8f0" textAnchor="middle" dominantBaseline="middle" fontSize={9} fontWeight="bold">
                      {name}: {value}
                    </text>
                  );
                }} />
              </Funnel>
            </FunnelChart>
          </ResponsiveContainer>
          {/* Funnel details table */}
          <div className="mt-2 space-y-1">
            {data.processing_funnel.map((f, i) => (
              <div key={i} className="flex items-center gap-2 text-[9px]">
                <span className={clsx("h-2 w-2 rounded-full", f.bottleneck ? "bg-rose-500" : "bg-slate-600")} />
                <span className="text-slate-300 w-28">{f.stage}</span>
                <span className="text-slate-500">{f.successful}/{f.total_frames}</span>
                {f.drop_rate_pct > 0 && (
                  <span className="text-rose-400 font-mono">-{f.drop_rate_pct.toFixed(1)}%</span>
                )}
                {f.bottleneck && (
                  <span className="rounded bg-rose-950/50 border border-rose-800/30 px-1 text-[7px] text-rose-400 font-bold">BOTTLENECK</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Cascading Failure Chain ────────────────────────────── */}
      {data.cascade_chain.length > 0 && (
        <div className="rounded-xl border border-rose-900/30 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <GitBranch size={12} className="text-rose-400" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-rose-400/80">
              Cascading Failure Chain
            </h4>
          </div>
          <CascadeFlow nodes={data.cascade_chain} />
        </div>
      )}

      {/* ── Resource Utilisation ───────────────────────────────── */}
      {data.resource_impacts.length > 0 && (
        <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <Server size={12} className="text-slate-500" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Resource Utilisation Impact
            </h4>
          </div>
          <div>
            {data.resource_impacts.map((r) => (
              <ResourceBar key={r.resource} r={r} />
            ))}
          </div>

          {/* Stacked bar comparison */}
          <div className="mt-3">
            <ResponsiveContainer width="100%" height={120}>
              <BarChart data={resBarData} layout="vertical">
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} tick={{ fill: "#475569", fontSize: 8 }} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" tick={{ fill: "#94a3b8", fontSize: 9 }} tickLine={false} axisLine={false} width={80} />
                <Tooltip contentStyle={DarkTooltipStyle} />
                <Bar dataKey="current" name="Current" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={10} />
                <Bar dataKey="projected" name="Projected Peak" fill="#f9731660" radius={[0, 4, 4, 0]} barSize={10} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── Coverage Gaps ──────────────────────────────────────── */}
      {data.coverage_gaps.length > 0 && (
        <div className="rounded-xl border border-amber-800/30 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <Eye size={12} className="text-amber-400" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-amber-400/80">
              Surveillance Coverage Gaps ({data.coverage_gaps.length})
            </h4>
          </div>
          <div className="space-y-2">
            {data.coverage_gaps.map((g: CoverageGap, i: number) => {
              const gColor = g.gap_type === "total_blind" ? "border-rose-800/40 bg-rose-950/20" :
                g.gap_type === "degraded" ? "border-amber-800/40 bg-amber-950/20" :
                "border-slate-700/40 bg-slate-800/20";
              return (
                <div key={i} className={clsx("rounded-lg border p-2.5", gColor)}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-bold text-slate-200">{g.camera_id}</span>
                    <span className="text-[8px] font-bold uppercase" style={{ color: SEVERITY_COLOR[g.severity] }}>
                      {g.gap_type.replace(/_/g, " ")}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[9px]">
                    <span className="text-slate-500">Zone: <span className="text-slate-400">{g.zone_affected}</span></span>
                    <span className="text-slate-500">Duration: <span className="text-slate-400">{g.duration_estimate}</span></span>
                    <span className="text-slate-500">Start: <span className="text-slate-400">{g.start_offset}</span></span>
                    <span className="text-slate-500">Detections missed: <span className="text-rose-400 font-bold">~{g.vehicles_missed_estimate}</span></span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Data Corruption Vectors ────────────────────────────── */}
      {data.data_corruption_vectors.length > 0 && (
        <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <FileWarning size={12} className="text-amber-400" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-amber-400/80">
              Data Corruption Vectors ({data.data_corruption_vectors.length})
            </h4>
          </div>
          <div className="space-y-2">
            {data.data_corruption_vectors.map((v: DataCorruptionVector, i: number) => (
              <div key={i} className="rounded-lg border border-slate-700/30 bg-slate-800/30 p-2.5">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] text-slate-300 font-medium">{v.source}</span>
                  <span className="text-[9px] text-slate-600">→</span>
                  <span className="text-[10px] text-slate-300 font-medium">{v.destination}</span>
                  <span className="text-[8px] font-bold uppercase ml-auto" style={{ color: SEVERITY_COLOR[v.severity] }}>
                    {v.severity}
                  </span>
                </div>
                <div className="flex gap-4 text-[9px] text-slate-500">
                  <span>Type: <span className="text-slate-400">{v.corruption_type}</span></span>
                  <span>Data: <span className="text-slate-400">{v.data_type}</span></span>
                  <span>Records: <span className="text-rose-400">~{v.records_affected_estimate}</span></span>
                  <span>Forensic: <span className="text-slate-400">{v.forensic_impact.replace(/_/g, " ")}</span></span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Compliance Impacts ─────────────────────────────────── */}
      {data.compliance_impacts.length > 0 && (
        <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <Shield size={12} className="text-slate-500" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Compliance & SLA Impact
            </h4>
          </div>
          <div className="space-y-2">
            {data.compliance_impacts.map((c: ComplianceImpact, i: number) => {
              const statusColor = c.current_status === "violated" ? "text-rose-400" :
                c.current_status === "at_risk" ? "text-amber-400" : "text-emerald-400";
              const liabilityColor = c.liability_level === "high" ? "bg-rose-950 text-rose-400 border-rose-800/40" :
                c.liability_level === "medium" ? "bg-amber-950 text-amber-400 border-amber-800/40" :
                "bg-slate-800 text-slate-400 border-slate-700/40";
              return (
                <div key={i} className="rounded-lg border border-slate-700/30 bg-slate-800/30 p-2.5">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-bold text-slate-200">{c.regulation}</span>
                    <span className={clsx("text-[8px] font-bold uppercase", statusColor)}>
                      {c.current_status.replace(/_/g, " ")}
                    </span>
                  </div>
                  <p className="text-[9px] text-slate-500 mb-1">{c.requirement}</p>
                  <p className="text-[9px] text-slate-400">{c.description}</p>
                  <div className="flex items-center gap-3 mt-1.5">
                    <span className="text-[8px] text-slate-600">Violation in: {c.time_to_violation}</span>
                    <span className={clsx("rounded border px-1 py-0 text-[7px] font-bold", liabilityColor)}>
                      Liability: {c.liability_level}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Component Health Heatmap ───────────────────────────── */}
      {data.component_heatmap.length > 0 && (
        <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <Layers size={12} className="text-slate-500" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Component Health Heatmap
            </h4>
          </div>
          <ComponentHeatmap cells={data.component_heatmap} />
          <div className="flex items-center gap-4 mt-2 justify-center">
            {[
              { label: "≥80% Healthy", cls: "bg-emerald-900/50" },
              { label: "60-79%", cls: "bg-yellow-900/40" },
              { label: "40-59%", cls: "bg-amber-900/40" },
              { label: "<40% Critical", cls: "bg-rose-900/40" },
            ].map((l) => (
              <div key={l.label} className="flex items-center gap-1">
                <span className={clsx("h-2 w-3 rounded-sm", l.cls)} />
                <span className="text-[7px] text-slate-600">{l.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Metric Correlations ────────────────────────────────── */}
      {data.correlations.length > 0 && (
        <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <Activity size={12} className="text-slate-500" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Metric Correlation Analysis
            </h4>
          </div>
          <CorrelationMatrix pairs={data.correlations} />
        </div>
      )}
    </div>
  );
}
