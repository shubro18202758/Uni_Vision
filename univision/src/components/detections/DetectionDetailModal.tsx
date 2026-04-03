import { X, Camera, Clock, ShieldCheck, Fingerprint, Gauge, AlertTriangle, BarChart3, Zap, Eye, Brain, Lightbulb, MapPin } from "lucide-react";
import type { WsDetectionEvent } from "../../types/api";
import { submitFeedback } from "../../services/api";
import { useToastStore } from "../../store/toastStore";
import { useState } from "react";
import clsx from "clsx";
import { RiskAnalysisDashboard } from "./RiskAnalysisDashboard";
import { ImpactAnalysisDashboard } from "./ImpactAnalysisDashboard";
import { TechnicalMetricsDashboard } from "./TechnicalMetricsDashboard";

type TabKey = "details" | "reasoning" | "risk" | "impact" | "technical";

interface Props {
  detection: WsDetectionEvent;
  onClose: () => void;
  initialTab?: TabKey;
}

const RISK_COLORS: Record<string, string> = {
  low: "bg-emerald-950/50 text-emerald-400 border-emerald-800/40",
  medium: "bg-amber-950/50 text-amber-400 border-amber-800/40",
  high: "bg-orange-950/50 text-orange-400 border-orange-800/40",
  critical: "bg-rose-950/50 text-rose-400 border-rose-800/40",
  unknown: "bg-slate-800/60 text-slate-400 border-slate-700",
};

const SEVERITY_COLORS: Record<string, string> = {
  low: "text-emerald-400 bg-emerald-950/40 border-emerald-800/40",
  medium: "text-amber-400 bg-amber-950/40 border-amber-800/40",
  high: "text-orange-400 bg-orange-950/40 border-orange-800/40",
  critical: "text-rose-400 bg-rose-950/40 border-rose-800/40",
};

function Row({ icon: Icon, label, value, mono }: { icon?: typeof Camera; label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-start gap-2 py-1.5 border-b border-slate-700/30 last:border-0">
      {Icon && <Icon size={12} className="mt-0.5 text-slate-500 flex-shrink-0" />}
      <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 w-28 flex-shrink-0">{label}</span>
      <span className={clsx("text-xs text-slate-200 break-all", mono && "font-mono")}>{value || "—"}</span>
    </div>
  );
}

export function DetectionDetailModal({ detection, onClose, initialTab }: Props) {
  const addToast = useToastStore((s) => s.addToast);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [activeTab, setActiveTab] = useState<TabKey>(initialTab ?? "details");

  const hasAnomaly = detection.anomaly_detected ?? false;
  const riskLevel = detection.risk_level ?? detection.validation_status ?? "unknown";

  // Extract primary anomaly to pass anomaly-specific context to analysis dashboards
  const primaryAnomaly = detection.anomalies?.[0];
  const analysisContext = {
    camera_id: detection.camera_id ?? "",
    risk_level: detection.risk_level ?? "unknown",
    confidence: detection.confidence ?? 0,
    scene_description: detection.scene_description ?? "",
    anomaly_detected: detection.anomaly_detected ?? false,
    detected_at_utc: detection.detected_at_utc ?? new Date().toISOString(),
    validation_status: detection.risk_level ?? "unknown",
    anomaly_type: primaryAnomaly?.type ?? "",
    anomaly_severity: primaryAnomaly?.severity ?? "",
    anomaly_description: primaryAnomaly?.description ?? "",
    anomaly_location: primaryAnomaly?.location ?? "",
  };

  const handleFeedback = async (type: "correct" | "incorrect") => {
    try {
      await submitFeedback({
        detection_id: detection.id,
        feedback_type: type === "incorrect" ? "reject" : "correct",
        original_text: detection.scene_description ?? detection.id,
      });
      setFeedbackSent(true);
      addToast("success", `Feedback "${type}" submitted`);
    } catch {
      addToast("error", "Failed to submit feedback");
    }
  };

  return (
    <div className="fixed inset-0 z-[9998] flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onClose}>
      <div
        className={clsx(
          "relative w-full max-h-[85vh] flex flex-col rounded-2xl border border-slate-700/40 bg-[#0d1b2e] shadow-2xl shadow-black/30 animate-in zoom-in-95 fade-in duration-200 transition-[max-width] duration-300",
          activeTab === "risk" || activeTab === "impact" || activeTab === "technical" ? "max-w-6xl" : "max-w-2xl",
        )}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-700/40 px-5 py-3 flex-shrink-0">
          <div className="flex items-center gap-2">
            <Eye size={16} className="text-slate-400" />
            <h3 className="text-sm font-bold text-slate-100">Analysis Detail</h3>
          </div>
          <button onClick={onClose} className="rounded p-1 hover:bg-slate-800 transition-colors" title="Close">
            <X size={14} className="text-slate-400" />
          </button>
        </div>

        {/* Tab bar */}
        <div className="flex items-center gap-0 border-b border-slate-700/40 px-5 flex-shrink-0">
          {([
            { key: "details" as TabKey, label: "Details", icon: Fingerprint },
            { key: "reasoning" as TabKey, label: "Chain-of-Thought", icon: Brain },
            { key: "risk" as TabKey, label: "Risk Analysis", icon: BarChart3 },
            { key: "impact" as TabKey, label: "Impact", icon: Zap },
            { key: "technical" as TabKey, label: "Technical Metrics", icon: Gauge },
          ]).map(({ key, label, icon: TabIcon }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={clsx(
                "flex items-center gap-1.5 px-4 py-2.5 text-[11px] font-bold uppercase tracking-wider transition-colors border-b-2 -mb-px",
                activeTab === key
                  ? "border-slate-300 text-slate-200"
                  : "border-transparent text-slate-500 hover:text-slate-400",
              )}
            >
              <TabIcon size={12} />
              {label}
            </button>
          ))}
        </div>

        {/* Scrollable content area */}
        <div className="overflow-y-auto flex-1 min-h-0">
          {/* ── Details tab ── */}
          {activeTab === "details" && (
            <>
              {/* Anomaly status hero */}
              <div className={clsx("flex items-center justify-center py-4 gap-3", hasAnomaly ? "bg-rose-950/20" : "bg-emerald-950/20")}>
                {hasAnomaly ? (
                  <AlertTriangle size={22} className="text-rose-400" />
                ) : (
                  <ShieldCheck size={22} className="text-emerald-400" />
                )}
                <span className={clsx("text-lg font-bold", hasAnomaly ? "text-rose-300" : "text-emerald-300")}>
                  {hasAnomaly ? "Anomaly Detected" : "No Anomaly"}
                </span>
                <span className={clsx("rounded-full border px-2.5 py-0.5 text-[10px] font-bold uppercase", RISK_COLORS[riskLevel] ?? RISK_COLORS.unknown)}>
                  {riskLevel}
                </span>
              </div>

              {/* Scene description */}
              {detection.scene_description && (
                <div className="px-5 pt-3 pb-1">
                  <p className="text-xs text-slate-300 leading-relaxed">{detection.scene_description}</p>
                </div>
              )}

              {/* Core details */}
              <div className="px-5 py-2">
                <Row icon={Camera} label="Camera" value={detection.camera_id} />
                <Row icon={Clock} label="Detected" value={new Date(detection.detected_at_utc).toLocaleString()} />
                <Row icon={Gauge} label="Confidence" value={`${((detection.confidence ?? 0) * 100).toFixed(1)}%`} />
                <Row icon={ShieldCheck} label="Risk Level" value={riskLevel} />
              </div>

              {/* Objects detected */}
              {detection.objects_detected && detection.objects_detected.length > 0 && (
                <div className="px-5 py-2">
                  <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-2 flex items-center gap-1.5">
                    <Eye size={11} /> Objects Detected ({detection.objects_detected.length})
                  </h4>
                  <div className="space-y-1">
                    {detection.objects_detected.map((obj, i) => (
                      <div key={i} className="flex items-center gap-2 rounded bg-[#111e36]/70 border border-slate-700/30 px-2 py-1.5 text-[11px]">
                        <span className="font-semibold text-slate-200">{obj.label}</span>
                        {obj.location && <span className="text-slate-500 flex items-center gap-0.5"><MapPin size={9} />{obj.location}</span>}
                        <span className={clsx("ml-auto rounded-full border px-1.5 py-0.5 text-[9px] font-bold uppercase",
                          obj.condition === "normal" ? "text-emerald-400 border-emerald-800/40 bg-emerald-950/30"
                            : "text-amber-400 border-amber-800/40 bg-amber-950/30"
                        )}>
                          {obj.condition}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Anomalies */}
              {detection.anomalies && detection.anomalies.length > 0 && (
                <div className="px-5 py-2">
                  <h4 className="text-[10px] font-bold uppercase tracking-wider text-rose-400/80 mb-2 flex items-center gap-1.5">
                    <AlertTriangle size={11} /> Anomalies ({detection.anomalies.length})
                  </h4>
                  <div className="space-y-1.5">
                    {detection.anomalies.map((a, i) => (
                      <div key={i} className="rounded-lg border border-rose-800/30 bg-rose-950/20 p-2.5">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-bold text-slate-200">{a.type}</span>
                          <span className={clsx("rounded-full border px-1.5 py-0.5 text-[9px] font-bold uppercase", SEVERITY_COLORS[a.severity] ?? SEVERITY_COLORS.medium)}>
                            {a.severity}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-400">{a.description}</p>
                        {a.location && <p className="text-[10px] text-slate-500 mt-0.5 flex items-center gap-1"><MapPin size={9} />{a.location}</p>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {detection.recommendations && detection.recommendations.length > 0 && (
                <div className="px-5 py-2">
                  <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-2 flex items-center gap-1.5">
                    <Lightbulb size={11} /> Recommendations
                  </h4>
                  <ul className="space-y-1">
                    {detection.recommendations.map((rec, i) => (
                      <li key={i} className="text-[11px] text-slate-300 flex gap-2">
                        <span className="text-slate-600 flex-shrink-0">{i + 1}.</span>
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Feedback */}
              <div className="border-t border-slate-700/40 px-5 py-3">
                {feedbackSent ? (
                  <p className="text-center text-xs text-emerald-400 font-medium">Feedback submitted</p>
                ) : (
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Feedback</span>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleFeedback("correct")}
                        className="rounded-lg border border-emerald-800/40 bg-emerald-950/40 px-3 py-1 text-xs font-bold text-emerald-400 hover:bg-emerald-900/50 transition-colors"
                      >
                        Correct
                      </button>
                      <button
                        onClick={() => handleFeedback("incorrect")}
                        className="rounded-lg border border-rose-800/40 bg-rose-950/40 px-3 py-1 text-xs font-bold text-rose-400 hover:bg-rose-900/50 transition-colors"
                      >
                        Incorrect
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          {/* ── Chain-of-Thought tab (inline data) ── */}
          {activeTab === "reasoning" && (
            <div className="px-5 py-4">
              {detection.chain_of_thought ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 mb-3">
                    <Brain size={14} className="text-slate-400" />
                    <h4 className="text-xs font-bold text-slate-200">Reasoning Chain</h4>
                  </div>
                  <div className="rounded-lg border border-slate-700/40 bg-[#111e36]/70 p-4">
                    <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">{detection.chain_of_thought}</p>
                  </div>
                  <div className="flex items-center gap-3 text-[10px] text-slate-500">
                    <span>Confidence: <strong className="text-slate-300">{((detection.confidence ?? 0) * 100).toFixed(1)}%</strong></span>
                    <span>Risk Level: <strong className={clsx(riskLevel === "critical" ? "text-rose-400" : riskLevel === "high" ? "text-orange-400" : riskLevel === "medium" ? "text-amber-400" : "text-emerald-400")}>{riskLevel}</strong></span>
                  </div>
                </div>
              ) : (
                <div className="rounded-lg border border-slate-700/40 bg-slate-800/30 p-6 text-center">
                  <Brain size={20} className="mx-auto mb-2 text-slate-600" />
                  <p className="text-xs text-slate-500">No chain-of-thought reasoning available for this frame.</p>
                </div>
              )}
            </div>
          )}

          {/* ── Risk Analysis tab (full dashboard) ── */}
          {activeTab === "risk" && (
            <div className="px-5 py-4">
              {/* Timestamp + risk badge header */}
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 size={14} className="text-slate-400" />
                <h4 className="text-xs font-bold text-slate-200">Risk Assessment</h4>
                <span className={clsx("rounded-full border px-2.5 py-0.5 text-[10px] font-bold uppercase ml-auto", RISK_COLORS[riskLevel] ?? RISK_COLORS.unknown)}>
                  {riskLevel}
                </span>
              </div>

              {/* Anomaly detection timestamp */}
              <div className="flex items-center gap-2 mb-4 rounded-lg border border-slate-700/40 bg-[#111e36]/70 px-3 py-2">
                <Clock size={12} className="text-cyan-400 flex-shrink-0" />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Anomaly Detected At</span>
                <span className="text-[11px] font-mono text-cyan-300 ml-auto">
                  {(() => { const d = new Date(detection.detected_at_utc); return `${d.toLocaleString(undefined, { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" })}.${String(d.getMilliseconds()).padStart(3, "0")}`; })()}
                </span>
              </div>

              {/* LLM text summary (if present) */}
              {detection.risk_analysis && (
                <div className="rounded-lg border border-slate-700/40 bg-[#111e36]/70 p-4 mb-4">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-1.5">LLM Summary</p>
                  <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">{detection.risk_analysis}</p>
                </div>
              )}

              {/* Full interactive Risk Analysis Dashboard */}
              <RiskAnalysisDashboard detectionId={detection.id} context={analysisContext} />
            </div>
          )}

          {/* ── Impact Analysis tab (full dashboard) ── */}
          {activeTab === "impact" && (
            <div className="px-5 py-4">
              {/* Timestamp + header */}
              <div className="flex items-center gap-2 mb-4">
                <Zap size={14} className="text-slate-400" />
                <h4 className="text-xs font-bold text-slate-200">Impact Analysis</h4>
              </div>

              {/* Anomaly detection timestamp */}
              <div className="flex items-center gap-2 mb-4 rounded-lg border border-slate-700/40 bg-[#111e36]/70 px-3 py-2">
                <Clock size={12} className="text-cyan-400 flex-shrink-0" />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Anomaly Detected At</span>
                <span className="text-[11px] font-mono text-cyan-300 ml-auto">
                  {(() => { const d = new Date(detection.detected_at_utc); return `${d.toLocaleString(undefined, { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" })}.${String(d.getMilliseconds()).padStart(3, "0")}`; })()}
                </span>
              </div>

              {/* LLM text summary (if present) */}
              {detection.impact_analysis && (
                <div className="rounded-lg border border-slate-700/40 bg-[#111e36]/70 p-4 mb-4">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-1.5">LLM Summary</p>
                  <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">{detection.impact_analysis}</p>
                </div>
              )}

              {/* Full interactive Impact Analysis Dashboard */}
              <ImpactAnalysisDashboard detectionId={detection.id} context={analysisContext} />
            </div>
          )}

          {/* ── Technical Metrics tab ── */}
          {activeTab === "technical" && (
            <div className="px-5 py-4">
              {/* Timestamp + header */}
              <div className="flex items-center gap-2 mb-4">
                <Gauge size={14} className="text-slate-400" />
                <h4 className="text-xs font-bold text-slate-200">Technical Metrics</h4>
              </div>

              {/* Anomaly detection timestamp */}
              <div className="flex items-center gap-2 mb-4 rounded-lg border border-slate-700/40 bg-[#111e36]/70 px-3 py-2">
                <Clock size={12} className="text-cyan-400 flex-shrink-0" />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Anomaly Detected At</span>
                <span className="text-[11px] font-mono text-cyan-300 ml-auto">
                  {(() => { const d = new Date(detection.detected_at_utc); return `${d.toLocaleString(undefined, { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" })}.${String(d.getMilliseconds()).padStart(3, "0")}`; })()}
                </span>
              </div>

              {/* Full interactive Technical Metrics Dashboard */}
              <TechnicalMetricsDashboard detectionId={detection.id} context={analysisContext} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
