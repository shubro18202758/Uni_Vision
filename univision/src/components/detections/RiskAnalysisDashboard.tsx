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
} from "recharts";
import {
  AlertTriangle,
  Bell,
  Shield,
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  Clock,
  Target,
  BarChart3,
  Layers,
  ChevronRight,
  Zap,
} from "lucide-react";
import clsx from "clsx";
import type {
  RiskAnalysisResponse,
  ScenarioProjection,
  ComponentHealth,
  AlertItem,
  IgnoredAlertConsequence,
} from "../../types/api";
import { fetchRiskAnalysis } from "../../services/api";

interface Props {
  detectionId: string;
}

/* ── Palette ──────────────────────────────────────────────────── */

const RISK_COLOR: Record<string, string> = {
  negligible: "#22c55e",
  low: "#3b82f6",
  moderate: "#eab308",
  high: "#f97316",
  critical: "#ef4444",
};

const SEVERITY_BAR: Record<string, string> = {
  critical: "#ef4444",
  high: "#f97316",
  moderate: "#eab308",
  low: "#3b82f6",
};

const COMPONENT_STATUS_COLOR: Record<string, string> = {
  healthy: "#22c55e",
  degraded: "#eab308",
  failing: "#ef4444",
};

const SCENARIO_STYLE: Record<string, { border: string; bg: string; text: string }> = {
  best_case: { border: "border-emerald-800/40", bg: "bg-emerald-950/30", text: "text-emerald-400" },
  likely: { border: "border-amber-800/40", bg: "bg-amber-950/30", text: "text-amber-400" },
  worst_case: { border: "border-rose-800/40", bg: "bg-rose-950/30", text: "text-rose-400" },
};

/* ── Trend icon ───────────────────────────────────────────────── */

function TrendIcon({ trend }: { trend: string }) {
  if (trend === "improving") return <TrendingDown size={10} className="text-emerald-400" />;
  if (trend === "degrading") return <TrendingUp size={10} className="text-rose-400" />;
  return <Minus size={10} className="text-slate-500" />;
}

/* ── Risk Gauge ───────────────────────────────────────────────── */

function RiskGauge({ score, level }: { score: number; level: string }) {
  const color = RISK_COLOR[level] || RISK_COLOR.moderate;
  const pct = Math.min(100, Math.max(0, score));
  const circumference = 2 * Math.PI * 42;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="relative flex flex-col items-center">
      <svg width="120" height="70" viewBox="0 0 120 70">
        {/* Background arc */}
        <path
          d="M 10 65 A 50 50 0 0 1 110 65"
          fill="none"
          stroke="#1e293b"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {/* Score arc */}
        <path
          d="M 10 65 A 50 50 0 0 1 110 65"
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${(pct / 100) * 157} 157`}
          className="transition-all duration-700"
        />
        <text x="60" y="55" textAnchor="middle" className="text-lg font-black" fill={color}>
          {score.toFixed(0)}
        </text>
        <text x="60" y="68" textAnchor="middle" className="text-[8px] font-bold uppercase tracking-wider" fill="#94a3b8">
          {level}
        </text>
      </svg>
    </div>
  );
}

/* ── Scenario Card ────────────────────────────────────────────── */

function ScenarioCard({ scenario }: { scenario: ScenarioProjection }) {
  const style = SCENARIO_STYLE[scenario.scenario] || SCENARIO_STYLE.likely;
  return (
    <div className={clsx("rounded-xl border p-3", style.border, style.bg)}>
      <div className="flex items-center justify-between mb-2">
        <span className={clsx("text-[10px] font-bold uppercase tracking-wider", style.text)}>
          {scenario.scenario.replace(/_/g, " ")}
        </span>
        <span className="text-[10px] font-mono text-slate-500">
          {(scenario.probability * 100).toFixed(0)}% prob
        </span>
      </div>
      <p className="text-xs font-semibold text-slate-200 mb-1">{scenario.title}</p>
      <p className="text-[11px] text-slate-400 mb-2 leading-relaxed">{scenario.description}</p>

      <div className="flex items-center gap-4 mb-2">
        <div>
          <span className="text-[9px] text-slate-500 block">Impact</span>
          <div className="flex items-center gap-1">
            <div className="h-1.5 w-16 rounded-full bg-slate-800 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${scenario.impact_score}%`,
                  backgroundColor: RISK_COLOR[scenario.impact_score > 60 ? "high" : scenario.impact_score > 30 ? "moderate" : "low"],
                }}
              />
            </div>
            <span className="text-[9px] font-mono text-slate-400">{scenario.impact_score.toFixed(0)}</span>
          </div>
        </div>
        <div>
          <span className="text-[9px] text-slate-500 block">Resolution</span>
          <span className="text-[10px] text-slate-300">{scenario.time_to_resolution}</span>
        </div>
      </div>

      {scenario.consequences_if_ignored?.length > 0 && (
        <div className="mt-2">
          <div className="flex items-center gap-1 mb-1">
            <AlertTriangle size={9} className="text-rose-400" />
            <span className="text-[9px] font-bold uppercase text-rose-400/80">If Ignored</span>
            {scenario.escalation_severity && (
              <span className={clsx(
                "rounded px-1 py-0 text-[7px] font-bold uppercase ml-auto",
                scenario.escalation_severity === "catastrophic" ? "bg-rose-950 text-rose-400 border border-rose-800/40" :
                scenario.escalation_severity === "severe" ? "bg-amber-950 text-amber-400 border border-amber-800/40" :
                "bg-slate-800 text-slate-400 border border-slate-700/40",
              )}>
                {scenario.escalation_severity}
              </span>
            )}
          </div>
          <div className="space-y-1">
            {scenario.consequences_if_ignored.map((consequence, i) => (
              <div key={i} className="flex items-start gap-1.5">
                <ChevronRight size={8} className="mt-0.5 text-rose-500/60 flex-shrink-0" />
                <span className="text-[10px] text-slate-400">{consequence}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Component Health Row ─────────────────────────────────────── */

function ComponentHealthRow({ c }: { c: ComponentHealth }) {
  const color = COMPONENT_STATUS_COLOR[c.status] || "#94a3b8";
  return (
    <div className="flex items-center gap-3 py-2 border-b border-slate-700/20 last:border-0">
      <div className="w-28 flex-shrink-0">
        <p className="text-[11px] font-medium text-slate-200 truncate">{c.component}</p>
        <p className="text-[9px] uppercase font-bold" style={{ color }}>{c.status}</p>
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-1">
          <div className="flex-1 h-2 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${c.health_pct}%`, backgroundColor: color }}
            />
          </div>
          <span className="text-[10px] font-mono text-slate-400 w-8 text-right">
            {c.health_pct.toFixed(0)}%
          </span>
        </div>
        <div className="flex gap-3 mt-0.5">
          <span className="text-[8px] text-slate-600">Latency: {c.latency_score.toFixed(0)}</span>
          <span className="text-[8px] text-slate-600">Reliability: {c.reliability_score.toFixed(0)}</span>
          <span className="text-[8px] text-slate-600">Accuracy: {c.accuracy_score.toFixed(0)}</span>
        </div>
      </div>
    </div>
  );
}

/* ── Custom Tooltip ───────────────────────────────────────────── */

function RadarTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { axis: string; score: number; description: string } }> }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-slate-700/60 bg-[#0d1b2e] px-3 py-2 shadow-lg">
      <p className="text-[10px] font-bold text-slate-200">{d.axis}</p>
      <p className="text-xs font-mono text-amber-400">{d.score.toFixed(1)}/100</p>
      <p className="text-[9px] text-slate-500 mt-0.5 max-w-[160px]">{d.description}</p>
    </div>
  );
}

/* ── Main Dashboard ───────────────────────────────────────────── */

export function RiskAnalysisDashboard({ detectionId }: Props) {
  const [data, setData] = useState<RiskAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchRiskAnalysis(detectionId)
      .then((r) => { if (!cancelled) setData(r); })
      .catch((e) => { if (!cancelled) setError(e.message || "Failed to load risk analysis"); })
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

  const riskColor = RISK_COLOR[data.overall_risk_level] || RISK_COLOR.moderate;

  /* Prepare timeline chart data */
  const timelineData = data.timeline.map((t, i) => ({
    idx: i,
    label: t.title.length > 20 ? t.title.slice(0, 20) + "…" : t.title,
    value: t.metric_value ?? (t.severity === "critical" ? 90 : t.severity === "high" ? 70 : 50),
    severity: t.severity,
  }));

  /* Prepare anomaly severity distribution for bar chart */
  const sevDistData = data.anomaly_patterns.flatMap((p) =>
    Object.entries(p.severity_distribution).map(([sev, count]) => ({ severity: sev, count, pattern: p.pattern_name }))
  );

  return (
    <div className="space-y-5 p-1">
      {/* ── Header: Gauge + Summary ────────────────────────────── */}
      <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
        <div className="flex items-start gap-4">
          <RiskGauge score={data.overall_risk_score} level={data.overall_risk_level} />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <Shield size={13} className="flex-shrink-0" style={{ color: riskColor }} />
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                Overall Risk Assessment
              </span>
            </div>
            <p className="text-[11px] text-slate-300 leading-relaxed">{data.summary}</p>
            {data.generated_at_ms > 0 && (
              <p className="text-[9px] text-slate-600 mt-1">Analyzed in {data.generated_at_ms.toFixed(1)}ms</p>
            )}
          </div>
        </div>
      </div>

      {/* ── Radar Chart: Risk Dimensions ───────────────────────── */}
      <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
        <div className="flex items-center gap-1.5 mb-3">
          <Target size={12} className="text-slate-500" />
          <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Risk Dimensions
          </h4>
        </div>

        <ResponsiveContainer width="100%" height={240}>
          <RadarChart data={data.risk_dimensions} cx="50%" cy="50%" outerRadius="75%">
            <PolarGrid stroke="#1e293b" />
            <PolarAngleAxis
              dataKey="axis"
              tick={{ fill: "#94a3b8", fontSize: 9 }}
              tickLine={false}
            />
            <PolarRadiusAxis
              domain={[0, 100]}
              tick={{ fill: "#475569", fontSize: 8 }}
              axisLine={false}
              tickCount={5}
            />
            <Radar
              name="Risk"
              dataKey="score"
              stroke={riskColor}
              fill={riskColor}
              fillOpacity={0.2}
              strokeWidth={2}
            />
            <Tooltip content={<RadarTooltip />} />
          </RadarChart>
        </ResponsiveContainer>

        {/* Dimension details */}
        <div className="mt-3 grid grid-cols-2 gap-2">
          {data.risk_dimensions.map((d) => (
            <div key={d.axis} className="flex items-center gap-2 rounded-lg bg-slate-800/30 px-2 py-1.5">
              <TrendIcon trend={d.trend} />
              <div className="min-w-0 flex-1">
                <p className="text-[9px] text-slate-500 truncate">{d.axis}</p>
                <p className="text-[10px] font-mono font-bold text-slate-300">{d.score.toFixed(0)}</p>
              </div>
              <span className="text-[8px] text-slate-600">{d.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Anomaly Timeline ───────────────────────────────────── */}
      {data.timeline.length > 0 && (
        <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <Clock size={12} className="text-slate-500" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Anomaly Timeline
            </h4>
          </div>

          {timelineData.length > 1 && (
            <ResponsiveContainer width="100%" height={120}>
              <AreaChart data={timelineData}>
                <defs>
                  <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={riskColor} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={riskColor} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 8 }} tickLine={false} axisLine={false} />
                <YAxis domain={[0, 100]} tick={{ fill: "#475569", fontSize: 8 }} tickLine={false} axisLine={false} width={25} />
                <Tooltip
                  contentStyle={{ background: "#0d1b2e", border: "1px solid #334155", borderRadius: 8, fontSize: 10, color: "#e2e8f0" }}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke={riskColor}
                  fill="url(#riskGrad)"
                  strokeWidth={2}
                  dot={{ r: 3, fill: riskColor, stroke: "#0d1b2e", strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}

          <div className="mt-3 space-y-2">
            {data.timeline.map((evt, i) => {
              const isPredict = evt.event_type === "prediction";
              return (
                <div key={i} className="flex items-start gap-2">
                  <div className="flex flex-col items-center flex-shrink-0 mt-0.5">
                    <span className={clsx(
                      "h-2 w-2 rounded-full",
                      evt.severity === "critical" ? "bg-rose-500" :
                      evt.severity === "high" ? "bg-amber-500" : "bg-slate-500",
                      isPredict && "ring-2 ring-amber-400/30",
                    )} />
                    {i < data.timeline.length - 1 && (
                      <div className={clsx("w-px h-6", isPredict ? "border-l border-dashed border-slate-700" : "bg-slate-700/50")} />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] font-semibold text-slate-200">{evt.title}</span>
                      {isPredict && (
                        <span className="rounded bg-amber-950/60 border border-amber-800/30 px-1 py-0 text-[8px] font-bold text-amber-400">
                          PREDICTED
                        </span>
                      )}
                    </div>
                    <p className="text-[10px] text-slate-400">{evt.description}</p>
                    {evt.timestamp && (
                      <p className="text-[8px] text-slate-600 mt-0.5">{evt.timestamp}</p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Alerts ─────────────────────────────────────────── */}
      {data.alerts && data.alerts.length > 0 && (
        <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <Bell size={12} className="text-rose-400" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-rose-400/80">
              Raised Alerts ({data.alerts.length})
            </h4>
          </div>
          <div className="space-y-2">
            {data.alerts.map((alert: AlertItem) => {
              const priorityColor = alert.priority === "CRITICAL" ? "border-rose-800/50 bg-rose-950/30" :
                alert.priority === "URGENT" ? "border-amber-800/50 bg-amber-950/30" :
                alert.priority === "WARNING" ? "border-yellow-800/50 bg-yellow-950/30" :
                "border-slate-700/50 bg-slate-800/30";
              const dotColor = alert.priority === "CRITICAL" ? "bg-rose-500" :
                alert.priority === "URGENT" ? "bg-amber-500" :
                alert.priority === "WARNING" ? "bg-yellow-500" : "bg-slate-500";
              return (
                <div key={alert.alert_id} className={clsx("rounded-lg border p-2.5", priorityColor)}>
                  <div className="flex items-center gap-2 mb-1">
                    <span className={clsx("h-2 w-2 rounded-full flex-shrink-0", dotColor)} />
                    <span className="text-[10px] font-bold text-slate-200 flex-1">{alert.title}</span>
                    <span className="text-[8px] font-bold uppercase tracking-wider text-slate-500">{alert.priority}</span>
                  </div>
                  <p className="text-[10px] text-slate-400 ml-4">{alert.description}</p>
                  <div className="flex items-center gap-3 mt-1.5 ml-4">
                    <span className="text-[8px] text-slate-600">{alert.source_component}</span>
                    {alert.metric_value !== null && (
                      <span className="text-[8px] text-slate-600">
                        Value: <span className="font-mono text-slate-400">{typeof alert.metric_value === 'number' ? alert.metric_value.toFixed(1) : alert.metric_value}</span>
                        {alert.threshold !== null && <> / Threshold: <span className="font-mono text-slate-400">{alert.threshold}</span></>}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── What If Ignored — Consequence Chains ───────────────── */}
      {data.ignored_consequences && data.ignored_consequences.length > 0 && (
        <div className="rounded-xl border border-rose-900/30 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <Zap size={12} className="text-rose-500" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-rose-400/80">
              What Happens If Alerts Are Ignored
            </h4>
          </div>
          <div className="space-y-4">
            {data.ignored_consequences.map((ic: IgnoredAlertConsequence) => (
              <div key={ic.alert_id} className="rounded-lg border border-slate-700/30 bg-slate-900/40 p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-bold text-slate-200">{ic.alert_title}</span>
                  <span className={clsx(
                    "rounded px-1.5 py-0.5 text-[8px] font-bold",
                    ic.cascading_failure_risk > 0.7 ? "bg-rose-950 text-rose-400" :
                    ic.cascading_failure_risk > 0.4 ? "bg-amber-950 text-amber-400" :
                    "bg-slate-800 text-slate-400",
                  )}>
                    Cascade Risk: {(ic.cascading_failure_risk * 100).toFixed(0)}%
                  </span>
                </div>
                {/* Consequence chain as timeline */}
                <div className="space-y-0">
                  {ic.consequence_chain.map((step, idx) => {
                    const stepColor = step.severity === "catastrophic" ? "bg-rose-500" :
                      step.severity === "severe" ? "bg-amber-500" :
                      step.severity === "moderate" ? "bg-yellow-500" : "bg-slate-500";
                    return (
                      <div key={idx} className="flex items-start gap-2">
                        <div className="flex flex-col items-center flex-shrink-0 mt-0.5">
                          <span className={clsx("h-2 w-2 rounded-full", stepColor)} />
                          {idx < ic.consequence_chain.length - 1 && (
                            <div className="w-px h-7 bg-slate-700/50" />
                          )}
                        </div>
                        <div className="min-w-0 flex-1 pb-2">
                          <div className="flex items-center gap-1.5">
                            <span className="text-[9px] font-mono text-slate-500">{step.timeframe}</span>
                            <span className="text-[10px] font-semibold text-slate-200">{step.event}</span>
                          </div>
                          <p className="text-[9px] text-slate-400">{step.description}</p>
                          <div className="flex gap-2 mt-0.5">
                            <span className="text-[8px] text-slate-600">Probability: {(step.probability * 100).toFixed(0)}%</span>
                            <span className="text-[8px] text-slate-600">{step.severity}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-2 rounded bg-rose-950/40 border border-rose-800/20 px-2 py-1.5">
                  <p className="text-[9px] text-rose-300 font-semibold">Terminal State: {ic.terminal_state}</p>
                  <p className="text-[8px] text-slate-500 mt-0.5">Total propagation: {ic.total_propagation_time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Scenario Projections ───────────────────────────────── */}
      {data.scenarios.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 mb-3">
            <Activity size={12} className="text-slate-500" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Scenario Projections
            </h4>
          </div>
          <div className="space-y-3">
            {data.scenarios.map((s) => (
              <ScenarioCard key={s.scenario} scenario={s} />
            ))}
          </div>
        </div>
      )}

      {/* ── Anomaly Severity Distribution Bar Chart ────────────── */}
      {sevDistData.length > 0 && (
        <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <BarChart3 size={12} className="text-slate-500" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Anomaly Severity Distribution
            </h4>
          </div>
          <ResponsiveContainer width="100%" height={120}>
            <BarChart data={sevDistData}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis dataKey="severity" tick={{ fill: "#94a3b8", fontSize: 9 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: "#475569", fontSize: 8 }} tickLine={false} axisLine={false} width={25} allowDecimals={false} />
              <Tooltip
                contentStyle={{ background: "#0d1b2e", border: "1px solid #334155", borderRadius: 8, fontSize: 10, color: "#e2e8f0" }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {sevDistData.map((entry, i) => (
                  <Cell key={i} fill={SEVERITY_BAR[entry.severity] || "#64748b"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Component Health ───────────────────────────────────── */}
      {data.component_health.length > 0 && (
        <div className="rounded-xl border border-slate-700/40 bg-[#111e36]/60 p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <Layers size={12} className="text-slate-500" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Component Health
            </h4>
          </div>
          <div>
            {data.component_health.map((c) => (
              <ComponentHealthRow key={c.component} c={c} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
