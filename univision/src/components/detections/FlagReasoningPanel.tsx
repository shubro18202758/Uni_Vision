import { useEffect, useState } from "react";
import {
  AlertTriangle,
  Bell,
  CheckCircle2,
  Info,
  Shield,
  Zap,
  Eye,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import clsx from "clsx";
import type { FlagReasoningResponse, EvidenceItem } from "../../types/api";
import { fetchFlagReasoning } from "../../services/api";

interface Props {
  detectionId: string;
}

const SEVERITY_STYLES: Record<string, { bg: string; text: string; border: string; icon: typeof AlertTriangle }> = {
  critical: { bg: "bg-rose-950/40", text: "text-rose-400", border: "border-rose-800/40", icon: AlertTriangle },
  high: { bg: "bg-amber-950/40", text: "text-amber-400", border: "border-amber-800/40", icon: AlertTriangle },
  medium: { bg: "bg-yellow-950/40", text: "text-yellow-400", border: "border-yellow-800/40", icon: Info },
  low: { bg: "bg-slate-800/60", text: "text-slate-400", border: "border-slate-700/40", icon: Info },
};

function SeverityBadge({ severity }: { severity: string }) {
  const s = SEVERITY_STYLES[severity] || SEVERITY_STYLES.medium;
  return (
    <span className={clsx("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-bold uppercase", s.bg, s.text, s.border)}>
      <s.icon size={10} />
      {severity}
    </span>
  );
}

function EvidenceCard({ item }: { item: EvidenceItem }) {
  const [expanded, setExpanded] = useState(false);
  const s = SEVERITY_STYLES[item.severity] || SEVERITY_STYLES.medium;

  return (
    <div className={clsx("rounded-lg border p-3", s.border, "bg-[#111e36]/70")}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Eye size={11} className="text-slate-500 flex-shrink-0" />
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              {item.evidence_type.replace(/_/g, " ")}
            </span>
            <SeverityBadge severity={item.severity} />
          </div>
          <p className="text-xs font-semibold text-slate-200">{item.label}</p>
          <p className="text-[11px] text-slate-400 mt-0.5">{item.description}</p>
        </div>

        {item.metric_value !== null && (
          <div className="text-right flex-shrink-0">
            <p className="text-sm font-mono font-bold text-slate-200">
              {typeof item.metric_value === "number" && item.metric_value <= 1
                ? `${(item.metric_value * 100).toFixed(1)}%`
                : item.metric_value}
            </p>
            {item.threshold !== null && (
              <p className="text-[9px] text-slate-500">threshold: {item.threshold}</p>
            )}
          </div>
        )}
      </div>

      {item.raw_data && Object.keys(item.raw_data).length > 0 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 flex items-center gap-1 text-[10px] text-slate-500 hover:text-slate-400 transition-colors"
        >
          {expanded ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
          {expanded ? "Hide" : "Show"} raw data
        </button>
      )}
      {expanded && item.raw_data && (
        <pre className="mt-2 rounded bg-black/30 p-2 text-[10px] text-slate-400 font-mono overflow-x-auto max-h-28 overflow-y-auto">
          {JSON.stringify(item.raw_data, null, 2)}
        </pre>
      )}
    </div>
  );
}

export function FlagReasoningPanel({ detectionId }: Props) {
  const [data, setData] = useState<FlagReasoningResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchFlagReasoning(detectionId)
      .then((r) => { if (!cancelled) setData(r); })
      .catch((e) => { if (!cancelled) setError(e.message || "Failed to load reasoning"); })
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

  if (!data || !data.flagged) {
    return (
      <div className="rounded-lg border border-emerald-800/40 bg-emerald-950/30 p-4 text-center">
        <CheckCircle2 size={16} className="mx-auto mb-1 text-emerald-400" />
        <p className="text-xs text-emerald-400">{data?.message || "No flag raised."}</p>
      </div>
    );
  }

  return (
    <div className="space-y-4 p-1">
      {/* Headline */}
      <div className={clsx(
        "rounded-xl border p-4",
        SEVERITY_STYLES[data.severity || "medium"].border,
        SEVERITY_STYLES[data.severity || "medium"].bg,
      )}>
        <div className="flex items-center gap-2 mb-2">
          <Shield size={14} className={SEVERITY_STYLES[data.severity || "medium"].text} />
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Flag Analysis
          </span>
          <SeverityBadge severity={data.severity || "medium"} />
          {data.confidence_score !== undefined && (
            <span className="ml-auto text-[10px] font-mono text-slate-500">
              confidence: {(data.confidence_score * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <p className="text-sm font-bold text-slate-100">{data.headline}</p>
        {data.generated_at_ms !== undefined && (
          <p className="text-[9px] text-slate-600 mt-1">
            Analyzed in {data.generated_at_ms.toFixed(1)}ms
          </p>
        )}
      </div>

      {/* Reasoning Chain */}
      {data.reasoning_chain && data.reasoning_chain.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 mb-2">
            <Zap size={12} className="text-slate-500" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Reasoning Chain
            </h4>
          </div>
          <div className="space-y-0">
            {data.reasoning_chain.map((step, i) => (
              <div key={i} className="flex items-start gap-2 py-1.5">
                <div className="flex flex-col items-center flex-shrink-0">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-800 text-[9px] font-bold text-slate-400 border border-slate-700">
                    {i + 1}
                  </span>
                  {i < data.reasoning_chain!.length - 1 && (
                    <div className="w-px h-3 bg-slate-700/50" />
                  )}
                </div>
                <p className="text-[11px] text-slate-300 leading-relaxed pt-0.5">{step}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Evidence */}
      {data.evidence && data.evidence.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 mb-2">
            <Eye size={12} className="text-slate-500" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Evidence ({data.evidence.length})
            </h4>
          </div>
          <div className="space-y-2">
            {data.evidence.map((item, i) => (
              <EvidenceCard key={i} item={item} />
            ))}
          </div>
        </div>
      )}

      {/* Alert Count */}
      {typeof data.alert_count === "number" && data.alert_count > 0 && (
        <div className="rounded-lg border border-rose-800/30 bg-rose-950/20 p-3 flex items-center gap-3">
          <div className="flex items-center justify-center h-10 w-10 rounded-full bg-rose-500/10 border border-rose-800/30">
            <Bell size={16} className="text-rose-400" />
          </div>
          <div>
            <p className="text-[10px] font-bold uppercase tracking-wider text-rose-400/80">Alerts Raised</p>
            <p className="text-lg font-black text-rose-400">{data.alert_count}</p>
            <p className="text-[9px] text-slate-500">High / Critical severity evidence items triggered alerts</p>
          </div>
        </div>
      )}
    </div>
  );
}
