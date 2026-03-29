import { useEffect } from "react";
import { Activity, Cpu, Server, Gauge, CircleDot, AlertTriangle } from "lucide-react";
import { usePipelineStore } from "../../store/pipelineStore";
import clsx from "clsx";

export function PipelineStatus() {
  const health = usePipelineStore((s) => s.health);
  const stats = usePipelineStore((s) => s.stats);
  const agentStatus = usePipelineStore((s) => s.agentStatus);
  const refreshAll = usePipelineStore((s) => s.refreshAll);

  useEffect(() => {
    refreshAll();
    const id = setInterval(refreshAll, 10_000);
    return () => clearInterval(id);
  }, [refreshAll]);

  const isHealthy = health?.healthy ?? false;

  return (
    <div className="flex items-center gap-4 text-[10px]">
      {/* Health dot */}
      <div className="flex items-center gap-1.5" title={isHealthy ? "Pipeline healthy" : "Pipeline unhealthy"}>
        <CircleDot
          size={12}
          className={clsx(
            isHealthy ? "text-emerald-400" : "text-rose-400",
            isHealthy && "animate-pulse"
          )}
        />
        <span className={clsx("font-semibold uppercase tracking-wider", isHealthy ? "text-emerald-400" : "text-rose-400")}>
          {isHealthy ? "Online" : "Offline"}
        </span>
      </div>

      {/* Mode */}
      {health?.mode && (
        <div className="flex items-center gap-1 text-slate-500" title="Pipeline mode">
          <Server size={10} />
          <span>{health.mode}</span>
        </div>
      )}

      {/* GPU */}
      {health?.gpu_available && (
        <div className="flex items-center gap-1 text-slate-500" title="GPU available">
          <Cpu size={10} className="text-violet-400" />
          <span>GPU</span>
        </div>
      )}

      {/* Ollama */}
      {health?.ollama_reachable !== undefined && (
        <div className="flex items-center gap-1" title={health.ollama_reachable ? "Ollama running" : "Ollama offline"}>
          {health.ollama_reachable ? (
            <Activity size={10} className="text-emerald-400" />
          ) : (
            <AlertTriangle size={10} className="text-amber-400" />
          )}
          <span className={health.ollama_reachable ? "text-slate-500" : "text-amber-400"}>Ollama</span>
        </div>
      )}

      {/* Streams */}
      {stats?.active_streams !== undefined && (
        <div className="flex items-center gap-1 text-slate-500" title="Active camera streams">
          <Gauge size={10} />
          <span>{stats.active_streams} streams</span>
        </div>
      )}

      {/* Agent */}
      {agentStatus && (
        <div className="flex items-center gap-1 text-slate-500" title={`Agent: ${agentStatus.available_tools?.length ?? 0} tools`}>
          <Activity size={10} className={agentStatus.running ? "text-emerald-400" : "text-slate-600"} />
          <span>Agent ({agentStatus.tool_count} tools)</span>
        </div>
      )}
    </div>
  );
}
