import clsx from "clsx";
import { BlockInspector } from "../inspector/BlockInspector";
import { DetectionFeed } from "../detections/DetectionFeed";
import { CameraManager } from "../cameras/CameraManager";
import { AnalyticsDashboard } from "../analytics/AnalyticsDashboard";
import { DatabricksInsights } from "../analytics/DatabricksInsights";
import { PipelineMonitor } from "../pipeline/PipelineMonitor";
import { useUiStore } from "../../store/uiStore";
import { Settings2, Radio, Camera, BarChart3, Activity, Sparkles } from "lucide-react";

const TABS = [
  { id: "inspector", label: "Inspect", icon: Settings2 },
  { id: "detections", label: "Detect", icon: Radio },
  { id: "cameras", label: "Cams", icon: Camera },
  { id: "analytics", label: "Stats", icon: BarChart3 },
  { id: "pipeline", label: "Pipeline", icon: Activity },
  { id: "databricks", label: "DBX", icon: Sparkles },
] as const;

export function RightPanel() {
  const tab = useUiStore((state) => state.rightPanelTab);
  const setTab = useUiStore((state) => state.setRightPanelTab);

  return (
    <aside className="flex h-full flex-col rounded-md border border-slate-700/30 bg-[#0d1525] overflow-hidden">
      <div className="flex bg-[#0d1525] px-1 py-0.5 border-b border-slate-800/60">
        {TABS.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={clsx(
                "flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2 text-[10px] font-semibold tracking-[0.08em] rounded transition-all",
                tab === item.id
                  ? "bg-slate-700/40 text-slate-200 border-b-2 border-slate-400"
                  : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/40 border-b-2 border-transparent"
              )}
              onClick={() => setTab(item.id)}
              type="button"
            >
              <Icon size={12} />
              {item.label.toUpperCase()}
            </button>
          );
        })}
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        {tab === "inspector" && <BlockInspector />}
        {tab === "detections" && <DetectionFeed />}
        {tab === "cameras" && <CameraManager />}
        {tab === "analytics" && <AnalyticsDashboard />}
        {tab === "pipeline" && <PipelineMonitor />}
        {tab === "databricks" && <DatabricksInsights />}
      </div>
    </aside>
  );
}
