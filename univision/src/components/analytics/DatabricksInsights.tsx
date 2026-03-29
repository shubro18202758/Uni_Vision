import { useEffect, useState, useCallback } from "react";
import {
  Database,
  FlaskConical,
  Sparkles,
  Search,
  RefreshCw,
  Layers,
  Clock,
  GitBranch,
  BarChart3,
  Cpu,
  Hash,
  Zap,
  ShieldAlert,
  Activity,
  Radar,
} from "lucide-react";
import {
  fetchDatabricksOverview,
  fetchDeltaStats,
  fetchDeltaHistory,
  fetchMLflowSummary,
  fetchSparkOverview,
  fetchVectorStats,
  searchSimilarPlates,
  fetchDatabricksHealth,
  fetchVectorClusters,
  fetchSparkAnalytics,
  type DatabricksOverview,
  type VectorSearchResult,
} from "../../services/api";
import clsx from "clsx";

/* ── Tiny stat card ──────────────────────────────────────────── */

function DkStatCard({
  icon: Icon,
  label,
  value,
  accent,
}: {
  icon: typeof Database;
  label: string;
  value: string | number;
  accent?: string;
}) {
  return (
    <div className="flex items-center gap-2.5 rounded-lg border border-slate-700/40 bg-[#111e36] p-2.5">
      <div
        className={clsx(
          "flex h-8 w-8 items-center justify-center rounded-md shrink-0",
          accent || "bg-slate-800/60",
        )}
      >
        <Icon size={14} className="text-white" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-[9px] font-semibold uppercase tracking-wider text-slate-500">
          {label}
        </p>
        <p className="text-base font-bold text-slate-100 leading-tight tabular-nums">
          {value}
        </p>
      </div>
    </div>
  );
}

/* ── Section wrapper ─────────────────────────────────────────── */

function Section({
  icon: Icon,
  title,
  color,
  children,
}: {
  icon: typeof Database;
  title: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-slate-700/40 bg-[#111e36] p-3">
      <div className="flex items-center gap-2 mb-2.5">
        <div
          className="flex h-5 w-5 items-center justify-center rounded"
          style={{ backgroundColor: color + "22" }}
        >
          <Icon size={11} style={{ color }} />
        </div>
        <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}

/* ── Mini row bar ────────────────────────────────────────────── */

function InfoRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between text-[10px]">
      <span className="text-slate-500">{label}</span>
      <span className="font-bold text-slate-300 tabular-nums">{value}</span>
    </div>
  );
}

/* ── Shimmer skeleton for loading state ──────────────────────── */

function SkeletonCard() {
  return (
    <div className="animate-pulse rounded-lg border border-slate-700/30 bg-[#111e36] p-2.5">
      <div className="flex items-center gap-2.5">
        <div className="h-8 w-8 rounded-md bg-slate-700/40" />
        <div className="flex-1 space-y-1.5">
          <div className="h-2 w-12 rounded bg-slate-700/40" />
          <div className="h-4 w-16 rounded bg-slate-700/40" />
        </div>
      </div>
    </div>
  );
}

/* ── Health status pill ──────────────────────────────────────── */

function HealthPill({ status }: { status: string }) {
  const isOk = status === "ok" || status === "active";
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[8px] font-bold uppercase tracking-wider",
        isOk
          ? "bg-emerald-900/40 text-emerald-400 border border-emerald-600/30"
          : "bg-amber-900/40 text-amber-400 border border-amber-600/30",
      )}
    >
      <span className={clsx("h-1.5 w-1.5 rounded-full", isOk ? "bg-emerald-400" : "bg-amber-400")} />
      {isOk ? "Healthy" : status}
    </span>
  );
}

/* ── Main component ──────────────────────────────────────────── */

export function DatabricksInsights() {
  const [overview, setOverview] = useState<DatabricksOverview | null>(null);
  const [deltaStats, setDeltaStats] = useState<Record<string, unknown> | null>(null);
  const [deltaHistory, setDeltaHistory] = useState<Record<string, unknown>[]>([]);
  const [mlflowSummary, setMlflowSummary] = useState<Record<string, unknown> | null>(null);
  const [sparkOverview, setSparkOverview] = useState<Record<string, unknown> | null>(null);
  const [vectorStats, setVectorStats] = useState<Record<string, unknown> | null>(null);
  const [healthData, setHealthData] = useState<Record<string, unknown> | null>(null);
  const [clusterData, setClusterData] = useState<Record<string, unknown> | null>(null);
  const [anomalies, setAnomalies] = useState<Record<string, unknown>[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<VectorSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [ov, ds, dh, ml, sp, vs, hl, cl, an] = await Promise.allSettled([
        fetchDatabricksOverview(),
        fetchDeltaStats(),
        fetchDeltaHistory(10),
        fetchMLflowSummary(),
        fetchSparkOverview(),
        fetchVectorStats(),
        fetchDatabricksHealth(),
        fetchVectorClusters(8),
        fetchSparkAnalytics("anomaly_detection", { z_threshold: 2.0 }),
      ]);
      if (ov.status === "fulfilled") setOverview(ov.value);
      if (ds.status === "fulfilled") setDeltaStats(ds.value);
      if (dh.status === "fulfilled") setDeltaHistory(dh.value.versions ?? []);
      if (ml.status === "fulfilled") setMlflowSummary(ml.value);
      if (sp.status === "fulfilled") setSparkOverview(sp.value);
      if (vs.status === "fulfilled") setVectorStats(vs.value);
      if (hl.status === "fulfilled") setHealthData(hl.value);
      if (cl.status === "fulfilled") setClusterData(cl.value);
      if (an.status === "fulfilled") setAnomalies((an.value.results ?? []) as Record<string, unknown>[]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load Databricks data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, [refresh]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    try {
      const res = await searchSimilarPlates(searchQuery.trim());
      setSearchResults(res.results ?? []);
    } catch {
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  const fmt = (v: unknown): string | number => {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number") return v.toLocaleString();
    return String(v);
  };

  if (!overview && !loading && error) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-500">
        Databricks integration not available
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles size={14} className="text-orange-400" />
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
            Databricks Insights
          </span>
        </div>
        <button
          onClick={refresh}
          className="rounded p-1 hover:bg-slate-800"
          title="Refresh"
        >
          <RefreshCw
            size={12}
            className={clsx("text-slate-400", loading && "animate-spin")}
          />
        </button>
      </div>

      {error && (
        <div className="rounded border border-rose-800/40 bg-rose-950/30 px-2 py-1 text-[10px] text-rose-400">
          {error}
        </div>
      )}

      {/* Health Status Bar */}
      {healthData && (
        <div className="flex items-center gap-2 rounded-lg border border-slate-700/30 bg-[#0c1a2d] px-3 py-1.5">
          <Activity size={11} className="text-slate-400" />
          <span className="text-[9px] font-bold uppercase tracking-wider text-slate-500">System</span>
          <HealthPill status={String(healthData.overall ?? "unknown")} />
          <div className="ml-auto flex gap-2">
            {(["delta", "mlflow", "spark", "vector"] as const).map((svc) => {
              const s = healthData[svc] as Record<string, unknown> | undefined;
              const ok = s?.status === "active" || s?.status === "ok" || s?.status !== "error";
              return (
                <span
                  key={svc}
                  className={clsx(
                    "text-[8px] font-bold uppercase",
                    ok ? "text-emerald-500/70" : "text-rose-400/70",
                  )}
                >
                  {svc}
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* KPI Overview Strip */}
      {loading && !overview ? (
        <div className="grid grid-cols-2 gap-2">
          <SkeletonCard /><SkeletonCard /><SkeletonCard /><SkeletonCard />
        </div>
      ) : (
      <div className="grid grid-cols-2 gap-2">
        <DkStatCard
          icon={Database}
          label="Delta Rows"
          value={fmt(
            (deltaStats as Record<string, Record<string, unknown>>)?.detections?.num_rows ?? 0,
          )}
          accent="bg-emerald-800/40"
        />
        <DkStatCard
          icon={FlaskConical}
          label="MLflow Runs"
          value={fmt((mlflowSummary as Record<string, unknown>)?.total_runs ?? 0)}
          accent="bg-blue-800/40"
        />
        <DkStatCard
          icon={Cpu}
          label="Spark Status"
          value={
            (sparkOverview as Record<string, unknown>)?.status === "ok"
              ? "Active"
              : "Idle"
          }
          accent="bg-amber-800/40"
        />
        <DkStatCard
          icon={Hash}
          label="FAISS Vectors"
          value={fmt((vectorStats as Record<string, unknown>)?.total_vectors ?? 0)}
          accent="bg-purple-800/40"
        />
      </div>
      )}

      {/* Delta Lake Section */}
      <Section icon={Layers} title="Delta Lake" color="#4ade80">
        <div className="space-y-1.5">
          <InfoRow
            label="Version"
            value={fmt(
              (deltaStats as Record<string, Record<string, unknown>>)?.detections?.version ?? 0,
            )}
          />
          <InfoRow
            label="Detection Records"
            value={fmt(
              (deltaStats as Record<string, Record<string, unknown>>)?.detections?.num_rows ?? 0,
            )}
          />
          <InfoRow
            label="Audit Records"
            value={fmt(
              (deltaStats as Record<string, Record<string, unknown>>)?.audits?.num_rows ?? 0,
            )}
          />
          <InfoRow
            label="Partitions"
            value={fmt(
              (deltaStats as Record<string, Record<string, unknown>>)?.detections
                ?.partition_columns ?? "—",
            )}
          />
        </div>

        {/* Version history mini-timeline */}
        {deltaHistory.length > 0 && (
          <div className="mt-2.5">
            <div className="flex items-center gap-1.5 mb-1.5">
              <Clock size={10} className="text-slate-500" />
              <span className="text-[9px] font-bold uppercase tracking-wider text-slate-600">
                Version History
              </span>
            </div>
            <div className="space-y-1">
              {deltaHistory.slice(0, 5).map((v, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 text-[9px] text-slate-500"
                >
                  <GitBranch size={9} className="shrink-0 text-emerald-500/60" />
                  <span className="font-mono text-slate-400">v{fmt(v.version)}</span>
                  <span className="truncate">{fmt(v.timestamp)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </Section>

      {/* MLflow Section */}
      <Section icon={FlaskConical} title="MLflow Tracking" color="#60a5fa">
        <div className="space-y-1.5">
          <InfoRow
            label="Experiment"
            value={fmt((mlflowSummary as Record<string, unknown>)?.experiment_name ?? "—")}
          />
          <InfoRow
            label="Total Runs"
            value={fmt((mlflowSummary as Record<string, unknown>)?.total_runs ?? 0)}
          />
          <InfoRow
            label="Active Run"
            value={(mlflowSummary as Record<string, unknown>)?.active_run_id ? "Yes" : "None"}
          />
          <InfoRow
            label="Frames Logged"
            value={fmt((mlflowSummary as Record<string, unknown>)?.frame_count ?? 0)}
          />
        </div>
      </Section>

      {/* Spark Analytics Section */}
      <Section icon={Zap} title="Spark Analytics" color="#facc15">
        <div className="space-y-1.5">
          <InfoRow
            label="Engine"
            value={fmt((sparkOverview as Record<string, unknown>)?.app_name ?? "PySpark")}
          />
          <InfoRow
            label="Status"
            value={
              (sparkOverview as Record<string, unknown>)?.status === "ok"
                ? "Connected"
                : "Standby"
            }
          />
          <InfoRow
            label="Total Detections"
            value={fmt((sparkOverview as Record<string, unknown>)?.total_detections ?? 0)}
          />
          <InfoRow
            label="Unique Cameras"
            value={fmt((sparkOverview as Record<string, unknown>)?.unique_cameras ?? 0)}
          />
          <InfoRow
            label="Avg Confidence"
            value={
              typeof (sparkOverview as Record<string, unknown>)?.avg_confidence === "number"
                ? `${(((sparkOverview as Record<string, unknown>).avg_confidence as number) * 100).toFixed(1)}%`
                : "—"
            }
          />
        </div>
      </Section>

      {/* FAISS Vector Search Section */}
      <Section icon={Search} title="Vector Search (FAISS)" color="#c084fc">
        <div className="space-y-1.5 mb-3">
          <InfoRow
            label="Total Vectors"
            value={fmt((vectorStats as Record<string, unknown>)?.total_vectors ?? 0)}
          />
          <InfoRow
            label="Embedding Model"
            value={fmt((vectorStats as Record<string, unknown>)?.embedding_model ?? "—")}
          />
          <InfoRow
            label="Dimensions"
            value={fmt((vectorStats as Record<string, unknown>)?.embedding_dim ?? 384)}
          />
        </div>

        {/* Search interface */}
        <div className="flex gap-1.5">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            placeholder="Search plate..."
            className="flex-1 rounded-md border border-slate-700/50 bg-[#0a1628] px-2.5 py-1.5 text-[11px] text-slate-200 placeholder-slate-600 outline-none focus:border-purple-500/50"
          />
          <button
            onClick={handleSearch}
            disabled={searching}
            className="rounded-md bg-purple-600/30 border border-purple-500/30 px-2.5 py-1.5 text-[10px] font-medium text-purple-300 hover:bg-purple-600/40 disabled:opacity-50"
          >
            {searching ? (
              <RefreshCw size={11} className="animate-spin" />
            ) : (
              <Search size={11} />
            )}
          </button>
        </div>

        {/* Search results */}
        {searchResults.length > 0 && (
          <div className="mt-2 space-y-1 max-h-32 overflow-auto">
            {searchResults.map((r, i) => (
              <div
                key={i}
                className="flex items-center justify-between rounded-md bg-[#0a1628] px-2 py-1.5 text-[9px]"
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono font-bold text-slate-200">
                    {r.plate_text}
                  </span>
                  <span className="text-slate-500">{r.camera_id}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className={clsx(
                      "font-bold tabular-nums",
                      r.similarity >= 0.9
                        ? "text-emerald-400"
                        : r.similarity >= 0.75
                          ? "text-amber-400"
                          : "text-slate-400",
                    )}
                  >
                    {(r.similarity * 100).toFixed(1)}%
                  </span>
                  <span className="text-slate-600">{r.engine}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* Anomaly Detection Section */}
      {anomalies.length > 0 && (
        <Section icon={ShieldAlert} title="Anomaly Detection" color="#f87171">
          <div className="space-y-1">
            <div className="text-[9px] text-slate-500 mb-1">
              Z-score flagged plates ({anomalies.length} detected)
            </div>
            {anomalies.slice(0, 6).map((a, i) => (
              <div
                key={i}
                className="flex items-center justify-between rounded-md bg-[#0a1628] px-2 py-1.5 text-[9px]"
              >
                <span className="font-mono font-bold text-slate-200">
                  {String(a.plate_text ?? "—")}
                </span>
                <div className="flex items-center gap-2">
                  <span
                    className={clsx(
                      "rounded px-1.5 py-0.5 text-[8px] font-bold uppercase",
                      a.anomaly_type === "low_confidence"
                        ? "bg-rose-900/40 text-rose-400"
                        : "bg-amber-900/40 text-amber-400",
                    )}
                  >
                    {String(a.anomaly_type ?? "unknown")}
                  </span>
                  <span className="text-slate-500 tabular-nums">
                    z={typeof a.z_score === "number" ? a.z_score.toFixed(2) : "—"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Cluster Analysis Section */}
      {clusterData && Array.isArray((clusterData as Record<string, unknown>).clusters) && (
        <Section icon={Radar} title="Plate Clusters (K-Means)" color="#38bdf8">
          <div className="space-y-1.5">
            <InfoRow
              label="Clusters"
              value={fmt((clusterData as Record<string, unknown>).n_clusters)}
            />
            <InfoRow
              label="Vectors Analysed"
              value={fmt((clusterData as Record<string, unknown>).total_vectors_analysed)}
            />
          </div>
          <div className="mt-2 space-y-1">
            {((clusterData as Record<string, unknown>).clusters as Array<Record<string, unknown>>)
              .slice(0, 5)
              .map((c, i) => (
                <div
                  key={i}
                  className="rounded-md bg-[#0a1628] px-2 py-1.5 text-[9px]"
                >
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="text-slate-400">
                      Cluster #{String(c.cluster_id)} — {fmt(c.size)} vectors
                    </span>
                    <span className="text-sky-400 font-bold tabular-nums">
                      {typeof c.coherence === "number" ? `${(c.coherence * 100).toFixed(0)}%` : "—"}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {Array.isArray(c.top_plates) &&
                      (c.top_plates as Array<{ plate: string; count: number }>).map((p, j) => (
                        <span
                          key={j}
                          className="rounded bg-sky-900/30 border border-sky-700/20 px-1.5 py-0.5 font-mono text-[8px] text-sky-300"
                        >
                          {p.plate} ×{p.count}
                        </span>
                      ))}
                  </div>
                </div>
              ))}
          </div>
        </Section>
      )}

      {/* Tech Stack Footer */}
      <div className="rounded-xl border border-slate-700/30 bg-[#0c1a2d] p-3">
        <div className="flex items-center gap-2 mb-1.5">
          <BarChart3 size={11} className="text-slate-500" />
          <span className="text-[9px] font-bold uppercase tracking-wider text-slate-600">
            Databricks Tech Stack
          </span>
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[9px]">
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Storage</span>
            <span className="text-emerald-400 font-bold">Delta Lake</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Tracking</span>
            <span className="text-blue-400 font-bold">MLflow</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Analytics</span>
            <span className="text-amber-400 font-bold">PySpark</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Search</span>
            <span className="text-purple-400 font-bold">FAISS</span>
          </div>
        </div>
      </div>
    </div>
  );
}
