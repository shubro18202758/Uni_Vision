import { useState, useMemo, useEffect } from "react";
import { Sparkles, Save, Play, Square, CheckCircle, Loader2, Download, Upload, Cpu, Zap } from "lucide-react";
import { PipelineStatus } from "../status/PipelineStatus";
import { getCurrentGraph } from "../../store/graphStore";
import { useGraphStore } from "../../store/graphStore";
import { saveGraph } from "../../lib/graphSerializer";
import { listUploads, processVideo, stopAllProcessing } from "../../services/api";
import { useToastStore } from "../../store/toastStore";
import { useKeyboardShortcuts } from "../../hooks/useKeyboardShortcuts";
import { exportPipelineJson, importPipelineJson } from "../../lib/pipelineIO";
import { useModelStore } from "../../store/modelStore";
import { useUiStore } from "../../store/uiStore";

export function Topbar() {
  const [saving, setSaving] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [saveOk, setSaveOk] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const addToast = useToastStore((s) => s.addToast);
  const setGraph = useGraphStore((s) => s.setGraph);

  const { phase, activeModel, transitioning, activateForLaunch, activateForDesign, refresh } = useModelStore();

  // Refresh model state on mount
  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleSave = async () => {
    setSaving(true);
    const graph = getCurrentGraph();
    saveGraph(graph);
    setSaveOk(true);
    setSaving(false);
    addToast("success", "Pipeline draft saved");
    setTimeout(() => setSaveOk(false), 2000);
  };

  const handleLaunch = async () => {
    setLaunching(true);
    const graph = getCurrentGraph();
    const { setBlockExecState, setAllExecStates } = useGraphStore.getState();

    // ── CRITICAL: Swap Navarasa → Qwen before pipeline launch ──
    addToast("info", "Swapping to Qwen 3.5 for pipeline processing...");
    const swapped = await activateForLaunch();
    if (!swapped) {
      addToast("warning", "Model swap failed — launching with current model");
    } else {
      addToast("success", "Qwen 3.5 activated — Navarasa unloaded from VRAM");
    }

    // Visual: mark every block queued then running
    setAllExecStates("queued");

    try {
      // Fetch uploaded videos and start processing each one
      const uploads = await listUploads();

      if (uploads.length === 0) {
        addToast("warning", "No uploaded videos found — upload a video first in CAMS tab");
        setAllExecStates("queued");
        useGraphStore.getState().clearExecStates();
        setLaunching(false);
        return;
      }

      // Mark blocks as running progressively
      for (const block of graph.blocks) {
        setBlockExecState(block.id, "running");
      }

      // Start processing each uploaded video through the real pipeline
      let launched = 0;
      for (const upload of uploads) {
        try {
          await processVideo({
            source_url: upload.source_url,
            fps_target: 3,
          });
          launched++;
        } catch (err) {
          addToast("warning", `Failed to process ${upload.filename}: ${err instanceof Error ? err.message : "unknown error"}`);
        }
      }

      // Auto-switch to Pipeline tab so the user sees real-time processing
      useUiStore.getState().setRightPanelTab("pipeline");

      // Mark all blocks as success
      for (const block of graph.blocks) {
        setBlockExecState(block.id, "success");
      }

      setPipelineRunning(true);
      addToast("info", `Pipeline launched — ${launched} video(s) processing. Check PIPELINE tab for live progress.`);
    } catch {
      // Mark remaining non-success blocks as error
      const states = useGraphStore.getState().executionStates;
      for (const block of graph.blocks) {
        if (states[block.id] !== "success") {
          setBlockExecState(block.id, "error");
        }
      }
      addToast("warning", "Pipeline launch failed — check backend logs");
    } finally {
      setLaunching(false);
    }
  };

  const handleStop = async () => {
    // ── Stop all active processing jobs ──
    try {
      const result = await stopAllProcessing();
      addToast("info", `Stopped ${result.count} processing job(s)`);
    } catch {
      addToast("warning", "Could not stop processing — backend may be unreachable");
    }

    // ── Swap Qwen → Navarasa when pipeline stops ──
    addToast("info", "Swapping back to Navarasa 2.0...");
    const swapped = await activateForDesign();
    if (swapped) {
      addToast("success", "Navarasa 2.0 activated — Qwen unloaded from VRAM");
    } else {
      addToast("warning", "Model swap back failed — Navarasa may not be available");
    }
    useGraphStore.getState().clearExecStates();
    setPipelineRunning(false);
  };

  const handleExport = () => {
    const graph = getCurrentGraph();
    exportPipelineJson(graph);
    addToast("success", "Pipeline exported as JSON");
  };

  const handleImport = async () => {
    const graph = await importPipelineJson();
    if (graph) {
      setGraph(graph);
      addToast("success", `Imported "${graph.project.name}"`);
    } else {
      addToast("error", "Failed to import pipeline — invalid file");
    }
  };

  const shortcutHandlers = useMemo(
    () => ({ onSave: handleSave, onLaunch: handleLaunch, onExport: handleExport, onImport: handleImport }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  useKeyboardShortcuts(shortcutHandlers);

  // Model indicator config
  const isNavarasa = phase === "pre_launch";
  const modelLabel = isNavarasa ? "Navarasa 2.0" : phase === "post_launch" ? "Qwen 3.5" : "Transitioning";
  const modelColor = isNavarasa ? "emerald" : phase === "post_launch" ? "blue" : "amber";

  return (
    <header className="flex h-12 items-center justify-between border-b border-slate-800/80 bg-[#0b1221] px-5">
      <div className="flex items-center gap-3">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-slate-700/80 text-slate-300">
          <Sparkles size={14} />
        </div>
        <span className="text-[13px] font-semibold tracking-tight text-slate-200">UniVision</span>
        {/* Breadcrumb */}
        <div className="hidden lg:flex items-center gap-1.5 ml-2 pl-2 border-l border-slate-700/50">
          <span className="text-[10px] text-slate-500">{useGraphStore.getState().projectName}</span>
          <span className="text-[10px] text-slate-600">/</span>
          <span className="text-[10px] text-slate-500">{useGraphStore.getState().blocks.length} nodes</span>
        </div>
      </div>

      <div className="flex items-center gap-2.5">
        <PipelineStatus />
        {/* Active Model Indicator */}
        <div className={`flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[10px] font-medium transition-all ${
          transitioning
            ? "border-amber-700/30 bg-amber-950/30 text-amber-400/80"
            : modelColor === "emerald"
              ? "border-emerald-700/30 bg-emerald-950/20 text-emerald-400/80"
              : "border-slate-700/40 bg-slate-800/30 text-slate-400"
        }`}>
          {transitioning ? (
            <Loader2 size={10} className="animate-spin" />
          ) : isNavarasa ? (
            <Cpu size={10} />
          ) : (
            <Zap size={10} />
          )}
          {modelLabel}
        </div>
      </div>

      <div className="flex items-center gap-1">
        <button
          onClick={handleExport}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-slate-500 hover:bg-slate-800/60 hover:text-slate-300 transition-colors"
          title="Export pipeline (Ctrl+Shift+E)"
        >
          <Download size={12} />
          Export
        </button>
        <button
          onClick={handleImport}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-slate-500 hover:bg-slate-800/60 hover:text-slate-300 transition-colors"
          title="Import pipeline (Ctrl+Shift+I)"
        >
          <Upload size={12} />
          Import
        </button>
        <div className="mx-1 h-4 w-px bg-slate-800" />
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-slate-400 hover:bg-slate-800/60 hover:text-slate-200 transition-colors disabled:opacity-40"
          title="Save draft (Ctrl+S)"
        >
          {saveOk ? <CheckCircle size={13} className="text-emerald-400" /> : <Save size={13} />}
          {saveOk ? "Saved" : "Save"}
        </button>
        {pipelineRunning ? (
          <button
            onClick={handleStop}
            disabled={transitioning}
            className="flex items-center gap-1 rounded-md bg-rose-600/90 px-3 py-1 text-[11px] font-medium text-white hover:bg-rose-500 transition-colors disabled:opacity-40"
            title="Stop pipeline"
          >
            {transitioning ? <Loader2 size={13} className="animate-spin" /> : <Square size={13} />}
            {transitioning ? "Stopping..." : "Stop"}
          </button>
        ) : (
          <button
            onClick={handleLaunch}
            disabled={launching || transitioning}
            className="flex items-center gap-1 rounded-md bg-slate-700/80 px-3 py-1 text-[11px] font-medium text-slate-200 hover:bg-slate-600/80 transition-colors disabled:opacity-40"
            title="Launch pipeline (Ctrl+Shift+P)"
          >
            {launching ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
            {launching ? "Launching..." : "Launch"}
          </button>
        )}
      </div>
    </header>
  );
}
