import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Bot,
  Send,
  Loader2,
  Lock,
  Unlock,
  CheckCircle2,
  XCircle,
  Zap,
  Globe,
  Languages,
  Cpu,
  Shield,
  Blocks,
  Sparkles,
  X,
  Maximize2,
  ChevronRight,
  RefreshCw,
  AlertTriangle,
  Terminal,
} from "lucide-react";
import { useAgenticStore } from "../../store/agenticStore";
import { useChatStore } from "../../store/chatStore";
import { useGraphStore } from "../../store/graphStore";
import {
  connectAgentStream,
  onAgentStream,
  sendWorkflowDesignMessage,
  disconnectAgentStream,
} from "../../services/websocket";
import type { AgentStreamFrame } from "../../types/api";
import type { ProjectGraph } from "../../types/graph";
import type { AgenticPhase } from "../../store/agenticStore";
import clsx from "clsx";

// ── Phase metadata for display ───────────────────────────────────

const PHASE_META: Record<
  AgenticPhase,
  { icon: typeof Bot; label: string; color: string }
> = {
  idle: { icon: Bot, label: "Initializing", color: "text-slate-400" },
  detecting_language: { icon: Globe, label: "Detecting Language", color: "text-blue-400" },
  translating: { icon: Languages, label: "Translating to English", color: "text-violet-400" },
  designing: { icon: Cpu, label: "Designing Pipeline", color: "text-amber-400" },
  validating: { icon: Shield, label: "Validating Connections", color: "text-emerald-400" },
  building: { icon: Blocks, label: "Building Graph", color: "text-blue-400" },
  complete: { icon: CheckCircle2, label: "Complete", color: "text-emerald-400" },
  error: { icon: XCircle, label: "Error", color: "text-rose-400" },
};

const PHASE_ORDER: AgenticPhase[] = [
  "detecting_language",
  "translating",
  "designing",
  "validating",
  "building",
  "complete",
];

// ── Main Overlay Component ───────────────────────────────────────

export function AgenticOverlay() {
  const mode = useAgenticStore((s) => s.mode);
  const locked = useAgenticStore((s) => s.locked);
  const currentPhase = useAgenticStore((s) => s.currentPhase);
  const phases = useAgenticStore((s) => s.phases);
  const originalInput = useAgenticStore((s) => s.originalInput);
  const englishInput = useAgenticStore((s) => s.englishInput);
  const detectedLanguage = useAgenticStore((s) => s.detectedLanguage);
  const generatedGraph = useAgenticStore((s) => s.generatedGraph);
  const error = useAgenticStore((s) => s.error);
  const totalElapsedMs = useAgenticStore((s) => s.totalElapsedMs);
  const llmOutput = useAgenticStore((s) => s.llmOutput);
  const designStartedAt = useAgenticStore((s) => s.designStartedAt);

  const enterAgenticMode = useAgenticStore((s) => s.enterAgenticMode);
  const lockScreen = useAgenticStore((s) => s.lockScreen);
  const unlockScreen = useAgenticStore((s) => s.unlockScreen);
  const setPhaseProgress = useAgenticStore((s) => s.setPhaseProgress);
  const markComplete = useAgenticStore((s) => s.markComplete);
  const markError = useAgenticStore((s) => s.markError);
  const exitAgenticMode = useAgenticStore((s) => s.exitAgenticMode);
  const openOverlay = useAgenticStore((s) => s.openOverlay);
  const closeOverlay = useAgenticStore((s) => s.closeOverlay);
  const overlayOpen = useAgenticStore((s) => s.overlayOpen);
  const pendingInput = useAgenticStore((s) => s.pendingInput);
  const appendLlmToken = useAgenticStore((s) => s.appendLlmToken);

  const setGraph = useGraphStore((s) => s.setGraph);

  const [input, setInput] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const consoleRef = useRef<HTMLDivElement>(null);
  const wsCleanupRef = useRef<(() => void) | null>(null);
  const autoStartedRef = useRef(false);

  const isActive = mode !== "idle";

  // ── WS frame handler ─────────────────────────────────────────

  const handleFrame = useCallback(
    (frame: AgentStreamFrame) => {
      if (frame.type === "workflow_lock") {
        if (frame.locked) lockScreen();
        else unlockScreen();
      } else if (frame.type === "workflow_phase") {
        if (frame.phase === "designing_stream" || frame.phase === "designing_thinking") {
          // Streaming LLM tokens — append to live design console
          appendLlmToken(frame.message ?? "");
        } else {
          setPhaseProgress(
            (frame.phase as AgenticPhase) ?? "idle",
            frame.message ?? "",
          );
        }
      } else if (frame.type === "workflow_complete") {
        const graph = frame.graph as unknown as ProjectGraph | null;
        if (frame.success && graph) {
          markComplete({
            graph,
            detectedLanguage: frame.detected_language,
            englishInput: frame.english_input,
            totalElapsedMs: frame.total_elapsed_ms,
          });
        } else {
          markError(frame.error ?? "Unknown error");
        }
      } else if (frame.type === "error") {
        markError(frame.content ?? "Agent not available");
      }
    },
    [lockScreen, unlockScreen, setPhaseProgress, appendLlmToken, markComplete, markError],
  );

  // ── Connect WS on overlay open ───────────────────────────────

  useEffect(() => {
    if (!isActive) return;

    connectAgentStream();
    const unsub = onAgentStream(handleFrame);
    wsCleanupRef.current = () => {
      unsub();
    };

    return () => {
      wsCleanupRef.current?.();
    };
  }, [isActive, handleFrame]);

  // ── Auto-start from chat intent detection ────────────────────

  useEffect(() => {
    if (pendingInput && overlayOpen && mode === "agentic" && !autoStartedRef.current) {
      autoStartedRef.current = true;
      connectAgentStream();
      setTimeout(() => {
        sendWorkflowDesignMessage(pendingInput);
      }, 400);
    }
    if (!overlayOpen) {
      autoStartedRef.current = false;
    }
  }, [pendingInput, overlayOpen, mode]);

  // ── Auto-scroll phase log ────────────────────────────────────

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [phases]);

  // ── Elapsed timer ────────────────────────────────────────────

  useEffect(() => {
    if (!isActive || !designStartedAt) {
      setElapsed(0);
      return;
    }
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - designStartedAt) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [isActive, designStartedAt]);

  // ── Auto-scroll live console ─────────────────────────────────

  useEffect(() => {
    consoleRef.current?.scrollTo({
      top: consoleRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [llmOutput]);

  // ── Block detection from streaming JSON ──────────────────────

  const detectedBlocks = useMemo(() => {
    if (!llmOutput) return [];
    const matches = [...llmOutput.matchAll(/"label"\s*:\s*"([^"]+)"/g)];
    return [...new Set(matches.map((m) => m[1]))];
  }, [llmOutput]);

  function formatElapsed(secs: number): string {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  // ── Submit workflow design ───────────────────────────────────

  function handleSubmit() {
    const desc = input.trim();
    if (!desc) return;

    enterAgenticMode(desc);
    openOverlay();

    // Small delay to ensure WS is connected
    setTimeout(() => {
      sendWorkflowDesignMessage(desc);
    }, 300);
    setInput("");
  }

  // ── Apply generated graph to workspace ───────────────────────

  function handleApplyGraph() {
    if (!generatedGraph) return;
    setGraph(generatedGraph);
    exitAgenticMode();
    closeOverlay();
  }

  // ── Dismiss overlay ──────────────────────────────────────────

  function handleDismiss() {
    if (locked) return; // Cannot dismiss during lock mode
    exitAgenticMode();
    closeOverlay();
  }

  // ── Phase progress indicator ─────────────────────────────────

  const currentPhaseIdx = PHASE_ORDER.indexOf(currentPhase);

  return (
    <>
      {/* Trigger button — always visible in bottom-right */}
      {!overlayOpen && (
        <motion.button
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="fixed bottom-6 right-6 z-50 flex items-center gap-2 rounded-full bg-gradient-to-r from-emerald-600 to-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-2xl shadow-emerald-950/50 hover:from-emerald-500 hover:to-blue-500 transition-all"
          onClick={() => openOverlay()}
        >
          <Sparkles size={18} />
          Design Workflow
        </motion.button>
      )}

      {/* Full-screen Comet-style overlay */}
      <AnimatePresence>
        {overlayOpen && (
          <motion.div
            key="agentic-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="fixed inset-0 z-[100] flex flex-col bg-[#050c18]/98 backdrop-blur-sm"
          >
            {/* Lock overlay — prevents all interaction */}
            {locked && (
              <div className="absolute inset-0 z-[200] cursor-not-allowed" />
            )}

            {/* Top bar */}
            <div className="flex items-center justify-between border-b border-slate-700/30 bg-[#0a1628]/80 px-6 py-3">
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500 to-blue-500">
                  <Bot size={18} className="text-white" />
                </div>
                <div>
                  <h1 className="text-sm font-bold text-emerald-400 tracking-wide">
                    NAVARASA AUTONOMOUS DESIGNER
                  </h1>
                  <p className="text-[10px] text-slate-500 tracking-wider uppercase">
                    Natural Language → Pipeline Workflow
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                {locked && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="flex items-center gap-1.5 rounded-full bg-rose-950/40 border border-rose-800/40 px-3 py-1"
                  >
                    <Lock size={12} className="text-rose-400" />
                    <span className="text-[10px] font-bold text-rose-400 uppercase tracking-wider">
                      Autonomous Mode
                    </span>
                  </motion.div>
                )}
                {!locked && (
                  <button
                    onClick={handleDismiss}
                    title="Close designer"
                    className="rounded-lg p-2 text-slate-500 hover:bg-slate-800/50 hover:text-slate-300 transition-colors"
                  >
                    <X size={16} />
                  </button>
                )}
              </div>
            </div>

            {/* Main content */}
            <div className="flex flex-1 overflow-hidden">
              {/* Left panel: Input & Status */}
              <div className="flex w-[420px] flex-col border-r border-slate-800/40 bg-[#0a1628]/40">
                {/* Input area */}
                <div className="border-b border-slate-800/40 p-5">
                  <label className="mb-2 block text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
                    Describe Your Pipeline
                  </label>
                  <p className="mb-3 text-[10px] text-slate-500">
                    Use any of 15 Indian languages or English. Describe what you want to build.
                  </p>
                  <textarea
                    className="w-full rounded-lg border border-slate-700/50 bg-[#111e36] p-3 text-xs text-slate-200 placeholder:text-slate-600 focus:border-emerald-500/50 focus:outline-none focus:ring-1 focus:ring-emerald-500/20 resize-none"
                    rows={5}
                    placeholder="e.g.: Build a surveillance pipeline with RTSP camera, anomaly detection, and dispatch results..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    disabled={isActive}
                  />
                  <button
                    onClick={handleSubmit}
                    disabled={!input.trim() || isActive}
                    className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-emerald-600 to-blue-600 px-4 py-2.5 text-xs font-bold text-white uppercase tracking-wider transition-all hover:from-emerald-500 hover:to-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    {isActive ? (
                      <>
                        <Loader2 size={14} className="animate-spin" />
                        Designing...
                      </>
                    ) : (
                      <>
                        <Zap size={14} />
                        Generate Pipeline
                      </>
                    )}
                  </button>
                </div>

                {/* Detection info */}
                {detectedLanguage && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    className="border-b border-slate-800/40 p-4"
                  >
                    <div className="flex items-center gap-2 text-[10px] text-slate-400">
                      <Globe size={12} className="text-blue-400" />
                      <span>
                        Language: <strong className="text-blue-300">{detectedLanguage}</strong>
                      </span>
                    </div>
                    {englishInput && englishInput !== originalInput && (
                      <div className="mt-2 rounded-md bg-violet-950/20 border border-violet-800/30 p-2">
                        <span className="text-[9px] font-bold uppercase tracking-wider text-violet-400 block mb-1">
                          English Translation
                        </span>
                        <p className="text-[11px] text-violet-300 leading-relaxed">
                          {englishInput}
                        </p>
                      </div>
                    )}
                  </motion.div>
                )}

                {/* Phase timeline */}
                <div className="flex-1 overflow-auto p-4" ref={scrollRef}>
                  <h3 className="mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
                    Design Progress
                  </h3>
                  <div className="space-y-1">
                    {PHASE_ORDER.map((phase, idx) => {
                      const meta = PHASE_META[phase];
                      const Icon = meta.icon;
                      const isCurrentPhase = phase === currentPhase;
                      const isPast = idx < currentPhaseIdx;
                      const isFuture = idx > currentPhaseIdx && currentPhaseIdx >= 0;
                      const phaseData = phases.find((p) => p.phase === phase);

                      return (
                        <motion.div
                          key={phase}
                          initial={false}
                          animate={{
                            opacity: isFuture ? 0.3 : 1,
                          }}
                          className={clsx(
                            "flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors",
                            isCurrentPhase && "bg-slate-800/40 border border-slate-700/40",
                            isPast && "bg-emerald-950/10",
                          )}
                        >
                          <div
                            className={clsx(
                              "flex h-7 w-7 items-center justify-center rounded-md",
                              isCurrentPhase && "bg-gradient-to-br from-blue-500/20 to-emerald-500/20",
                              isPast && "bg-emerald-500/10",
                              isFuture && "bg-slate-800/30",
                            )}
                          >
                            {isCurrentPhase && phase !== "complete" ? (
                              <Loader2 size={14} className={clsx("animate-spin", meta.color)} />
                            ) : isPast || phase === "complete" && currentPhase === "complete" ? (
                              <CheckCircle2 size={14} className="text-emerald-400" />
                            ) : (
                              <Icon size={14} className={isFuture ? "text-slate-600" : meta.color} />
                            )}
                          </div>
                          <div className="flex-1 min-w-0">
                            <span
                              className={clsx(
                                "text-xs font-semibold block",
                                isCurrentPhase ? meta.color : isPast ? "text-emerald-400" : "text-slate-600",
                              )}
                            >
                              {meta.label}
                            </span>
                            {phaseData?.message && (
                              <span className="text-[10px] text-slate-500 block truncate">
                                {phaseData.message}
                              </span>
                            )}
                          </div>
                          {isPast && (
                            <ChevronRight size={12} className="text-emerald-600" />
                          )}
                        </motion.div>
                      );
                    })}
                  </div>

                  {/* Error display */}
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-4 rounded-lg border border-rose-800/40 bg-rose-950/20 p-4 space-y-3"
                    >
                      <div className="flex items-center gap-2">
                        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-rose-500/15">
                          <XCircle size={14} className="text-rose-400" />
                        </div>
                        <div>
                          <span className="text-[10px] font-bold uppercase tracking-wider text-rose-400 block">
                            Design Failed
                          </span>
                          <span className="text-[9px] text-rose-500/70">
                            {currentPhase !== "error" ? PHASE_META[currentPhase]?.label : ""}
                          </span>
                        </div>
                      </div>
                      <p className="text-[11px] text-rose-300 leading-relaxed bg-rose-950/30 rounded-md px-3 py-2 border border-rose-900/20 font-mono">
                        {error}
                      </p>

                      {/* Actionable troubleshooting tips */}
                      {(error.toLowerCase().includes("connect") ||
                        error.toLowerCase().includes("ollama") ||
                        error.toLowerCase().includes("refused") ||
                        error.toLowerCase().includes("timeout")) && (
                        <div className="space-y-1.5">
                          <div className="flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-wider text-amber-400/80">
                            <AlertTriangle size={10} />
                            Troubleshooting
                          </div>
                          <div className="space-y-1">
                            {[
                              { icon: Terminal, text: "Ensure Ollama is running: ollama serve" },
                              { icon: Cpu, text: "Verify model is pulled: ollama list" },
                              { icon: Globe, text: "Check Ollama URL in config (default: localhost:11434)" },
                            ].map((tip, i) => (
                              <div
                                key={i}
                                className="flex items-center gap-2 rounded-md bg-amber-950/10 border border-amber-900/15 px-2.5 py-1.5"
                              >
                                <tip.icon size={10} className="text-amber-500/70 shrink-0" />
                                <span className="text-[10px] text-amber-300/80">{tip.text}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </motion.div>
                  )}
                </div>

                {/* Bottom: Elapsed time + apply */}
                {mode === "complete" && generatedGraph && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="border-t border-slate-800/40 p-4 space-y-3"
                  >
                    <div className="flex items-center justify-between text-[10px] text-slate-400">
                      <span>
                        {generatedGraph.blocks.length} blocks, {generatedGraph.connections.length} connections
                      </span>
                      <span>{(totalElapsedMs / 1000).toFixed(1)}s</span>
                    </div>
                    <button
                      onClick={handleApplyGraph}
                      className="flex w-full items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-emerald-600 to-blue-600 px-4 py-2.5 text-xs font-bold text-white uppercase tracking-wider transition-all hover:from-emerald-500 hover:to-blue-500"
                    >
                      <Maximize2 size={14} />
                      Apply to Workspace
                    </button>
                    <button
                      onClick={handleDismiss}
                      className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-700/50 bg-transparent px-4 py-2 text-xs text-slate-400 transition-colors hover:bg-slate-800/30 hover:text-slate-300"
                    >
                      Dismiss
                    </button>
                  </motion.div>
                )}

                {mode === "error" && (
                  <div className="border-t border-slate-800/40 p-4 space-y-2">
                    <button
                      onClick={() => {
                        exitAgenticMode();
                        if (originalInput) {
                          setInput(originalInput);
                        }
                      }}
                      className="flex w-full items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-emerald-600/80 to-blue-600/80 px-4 py-2.5 text-xs font-bold text-white uppercase tracking-wider transition-all hover:from-emerald-500 hover:to-blue-500"
                    >
                      <RefreshCw size={13} />
                      Retry Design
                    </button>
                    <button
                      onClick={() => {
                        exitAgenticMode();
                        closeOverlay();
                      }}
                      className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-700/50 bg-transparent px-4 py-2 text-xs text-slate-400 transition-colors hover:bg-slate-800/30 hover:text-slate-300"
                    >
                      Dismiss
                    </button>
                  </div>
                )}
              </div>

              {/* Right panel: Live preview */}
              <div className={clsx(
                "flex flex-1 flex-col",
                isActive && mode !== "complete" ? "" : "items-center justify-center p-8",
              )}>
                {!isActive && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex flex-col items-center text-center"
                  >
                    <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500/10 to-blue-500/10 border border-emerald-900/30">
                      <Sparkles size={36} className="text-emerald-500" />
                    </div>
                    <h2 className="text-lg font-bold text-slate-300 mb-2">
                      Autonomous Pipeline Designer
                    </h2>
                    <p className="text-xs text-slate-500 max-w-md leading-relaxed">
                      Describe your computer vision pipeline in natural language — in any of 15 Indian
                      languages or English. Navarasa will autonomously design, validate, and
                      build a complete block-node workflow for you.
                    </p>
                    <div className="mt-6 flex flex-wrap justify-center gap-2">
                      {["Hindi", "Telugu", "Tamil", "Bengali", "Kannada", "English"].map((lang) => (
                        <span
                          key={lang}
                          className="rounded-full border border-slate-700/40 bg-slate-800/30 px-3 py-1 text-[10px] text-slate-500"
                        >
                          {lang}
                        </span>
                      ))}
                      <span className="rounded-full border border-slate-700/40 bg-slate-800/30 px-3 py-1 text-[10px] text-slate-500">
                        +10 more
                      </span>
                    </div>
                  </motion.div>
                )}

                {isActive && mode !== "complete" && (
                  <div className="flex flex-1 flex-col h-full w-full">
                    {/* Console header */}
                    <div className="flex items-center justify-between border-b border-slate-800/40 px-5 py-3 bg-[#0a1628]/60">
                      <div className="flex items-center gap-2">
                        <Terminal size={14} className="text-emerald-400" />
                        <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
                          Design Console
                        </span>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-[10px] font-mono text-blue-400">
                          {formatElapsed(elapsed)}
                        </span>
                        {llmOutput && (
                          <div className="flex items-center gap-1.5">
                            <motion.div
                              animate={{ opacity: [1, 0.3, 1] }}
                              transition={{ duration: 1.5, repeat: Infinity }}
                              className="h-1.5 w-1.5 rounded-full bg-emerald-400"
                            />
                            <span className="text-[9px] font-bold text-emerald-400/70 uppercase tracking-wider">
                              Live
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Streaming LLM output */}
                    <div
                      ref={consoleRef}
                      className="flex-1 overflow-auto p-4 font-mono text-[11px] leading-relaxed text-slate-400 bg-[#050c18]"
                    >
                      {llmOutput ? (
                        <pre className="whitespace-pre-wrap break-words">
                          {llmOutput}
                          <motion.span
                            animate={{ opacity: [1, 0] }}
                            transition={{ duration: 0.8, repeat: Infinity }}
                            className="text-emerald-400"
                          >
                            ▊
                          </motion.span>
                        </pre>
                      ) : (
                        <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
                          <Loader2 size={24} className="animate-spin text-blue-400/50" />
                          <div>
                            <p className="text-xs text-slate-500 mb-1">
                              {PHASE_META[currentPhase].label}
                            </p>
                            <p className="text-[10px] text-slate-600">
                              {phases[phases.length - 1]?.message ?? "Starting autonomous design..."}
                            </p>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Detected blocks footer */}
                    {detectedBlocks.length > 0 && (
                      <div className="border-t border-slate-800/40 px-4 py-3 bg-[#0a1628]/40">
                        <span className="text-[9px] font-bold uppercase tracking-wider text-slate-500 block mb-2">
                          Detected Blocks
                        </span>
                        <div className="flex flex-wrap gap-1.5">
                          {detectedBlocks.map((b, i) => (
                            <span
                              key={i}
                              className="rounded-full bg-blue-500/10 border border-blue-500/20 px-2.5 py-0.5 text-[10px] text-blue-400"
                            >
                              {b}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Lock indicator */}
                    {locked && (
                      <div className="flex items-center justify-center gap-2 border-t border-rose-800/20 bg-rose-950/15 py-2">
                        <Lock size={11} className="text-rose-400" />
                        <span className="text-[9px] font-semibold text-rose-400 uppercase tracking-wider">
                          Screen Locked — Autonomous Generation
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {mode === "complete" && generatedGraph && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="w-full max-w-2xl"
                  >
                    <div className="mb-4 flex items-center gap-2">
                      <CheckCircle2 size={16} className="text-emerald-400" />
                      <h3 className="text-sm font-bold text-emerald-400">
                        Pipeline Generated Successfully
                      </h3>
                    </div>

                    {/* Graph preview */}
                    <div className="rounded-xl border border-emerald-900/30 bg-[#0d1b2e] p-4">
                      <h4 className="mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
                        {generatedGraph.project.name}
                      </h4>
                      <div className="space-y-2">
                        {generatedGraph.blocks.map((block) => (
                          <div
                            key={block.id}
                            className="flex items-center gap-3 rounded-lg bg-[#111e36] border border-slate-700/30 px-3 py-2"
                          >
                            <div className="h-2 w-2 rounded-full bg-blue-400" />
                            <span className="text-xs text-slate-300 font-medium">
                              {block.label}
                            </span>
                            <span className="text-[10px] text-slate-600 ml-auto">
                              {block.type}
                            </span>
                          </div>
                        ))}
                      </div>
                      <div className="mt-3 flex items-center gap-4 text-[10px] text-slate-500">
                        <span>{generatedGraph.connections.length} connections</span>
                        <span>{(totalElapsedMs / 1000).toFixed(1)}s total</span>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
