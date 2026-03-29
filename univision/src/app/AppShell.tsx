import { useEffect } from "react";
import { LeftSidebar } from "../components/layout/LeftSidebar";
import { RightPanel } from "../components/layout/RightPanel";
import { Topbar } from "../components/layout/Topbar";
import { StatusBar } from "../components/layout/StatusBar";
import { WorkbenchCanvas } from "../components/canvas/WorkbenchCanvas";
import { ToastContainer } from "../components/feedback/ToastContainer";
import { AgenticOverlay } from "../components/chat/AgenticOverlay";
import { usePipelineMonitorStore } from "../store/pipelineMonitorStore";

export function AppShell() {
  // Keep pipeline WebSocket connected at all times so events are never
  // missed regardless of which right-panel tab is active.
  const startMonitoring = usePipelineMonitorStore((s) => s.startMonitoring);
  useEffect(() => {
    const cleanup = startMonitoring();
    return cleanup;
  }, [startMonitoring]);
  return (
    <div className="flex h-screen flex-col bg-[#070e1b] text-slate-200">
      <Topbar />
      <main className="grid flex-1 grid-cols-[300px_1fr_360px] gap-px px-0 pb-0 pt-0 overflow-hidden min-h-0 bg-slate-800/30">
        <LeftSidebar />
        <section className="overflow-hidden bg-[#0a1628] relative h-full">
          <WorkbenchCanvas />
        </section>
        <RightPanel />
      </main>
      <StatusBar />
      <ToastContainer />
      <AgenticOverlay />
    </div>
  );
}
