import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Bot,
  Send,
  Trash2,
  Zap,
  Loader2,
  Wifi,
  WifiOff,
  Sparkles,
  User,
  ArrowUp,
  Globe,
  Workflow,
} from "lucide-react";
import { useChatStore } from "../../store/chatStore";
import { useAgenticStore } from "../../store/agenticStore";
import { AgentStepsAccordion } from "./AgentStepsAccordion";
import clsx from "clsx";

// ── Workflow intent detection (mirrors backend WORKFLOW_DESIGN patterns) ──

const WORKFLOW_INTENT_PATTERNS = [
  /\b(design|create|build|make|generate|set\s*up)\b.{0,30}\b(pipeline|workflow|graph|flow)\b/i,
  /\b(pipeline|workflow)\b.{0,30}\b(design|create|build|generate|bana|banao)\b/i,
  /\bautonomous(ly)?\b.{0,20}\b(design|mode|control)\b/i,
  /\bagentic\s+mode\b/i,
  /\bworkflow\s+(from|using)\s+(natural\s+language|description|text)\b/i,
  // Hindi
  /पाइपलाइन|वर्कफ्लो|बनाओ|डिजाइन/,
  // Telugu
  /పైప్‌లైన్|వర్క్‌ఫ్లో|రూపొందించు/,
  // Tamil
  /பைப்லைன்|பணிப்பாய்வு|வடிவமை/,
  // Bengali
  /পাইপলাইন|ওয়ার্কফ্লো|তৈরি/,
];

function isWorkflowIntent(text: string): boolean {
  return WORKFLOW_INTENT_PATTERNS.some((rx) => rx.test(text));
}

export function NavarasaChat() {
  const messages = useChatStore((s) => s.messages);
  const sending = useChatStore((s) => s.sending);
  const wsConnected = useChatStore((s) => s.wsConnected);
  const streamMode = useChatStore((s) => s.streamMode);
  const sendChat = useChatStore((s) => s.sendChat);
  const sendStream = useChatStore((s) => s.sendStream);
  const clearMessages = useChatStore((s) => s.clearMessages);
  const startAgentWs = useChatStore((s) => s.startAgentWs);

  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const triggerFromChat = useAgenticStore((s) => s.triggerFromChat);

  useEffect(() => {
    const cleanup = startAgentWs();
    return cleanup;
  }, [startAgentWs]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  function handleSend() {
    const msg = input.trim();
    if (!msg || sending) return;

    // ── Auto-detect workflow design intent → trigger agentic overlay ──
    if (isWorkflowIntent(msg)) {
      setInput("");
      triggerFromChat(msg);
      return;
    }

    setInput("");
    if (streamMode && wsConnected) {
      sendStream(msg);
    } else {
      sendChat(msg);
    }
  }

  // ── Quick action suggestions ──────────────────────────────────

  const quickActions = [
    { label: "Design a pipeline", icon: Workflow, action: "Design a computer vision pipeline for anomaly detection" },
    { label: "System status", icon: Zap, action: "What is the current system status?" },
    { label: "Explain workflow", icon: Globe, action: "How does the detection pipeline work?" },
  ];

  return (
    <section
      className="flex flex-col overflow-hidden"
      style={{ height: 420 }}
    >
      {/* ── Header with gradient accent ──────────────────────── */}
      <div className="relative flex items-center justify-between px-4 py-2.5 bg-gradient-to-r from-[#0c1a30] via-[#0e1f3a] to-[#0c1a30]">
        {/* Top gradient line */}
        <div className="absolute inset-x-0 top-0 h-[1px] bg-gradient-to-r from-transparent via-emerald-500/50 to-transparent" />

        <div className="flex items-center gap-2.5">
          <div className="relative flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 ring-1 ring-emerald-500/30">
            <Bot size={14} className="text-emerald-400" />
            {/* Online pulse */}
            {wsConnected && (
              <span className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full bg-emerald-400 ring-2 ring-[#0c1a30]">
                <span className="absolute inset-0 animate-ping rounded-full bg-emerald-400 opacity-40" />
              </span>
            )}
          </div>
          <div>
            <span className="text-[11px] font-bold text-slate-200 tracking-wide">
              Gemma 4 E2B
            </span>
            <span className="flex items-center gap-1 text-[9px] text-slate-500">
              {wsConnected ? (
                <>
                  <span className="h-1 w-1 rounded-full bg-emerald-500" />
                  Connected
                </>
              ) : (
                <>
                  <span className="h-1 w-1 rounded-full bg-slate-600" />
                  Offline
                </>
              )}
            </span>
          </div>
        </div>
        <button
          onClick={clearMessages}
          className="rounded-md p-1.5 text-slate-500 hover:bg-slate-800/50 hover:text-rose-400 transition-all"
          title="Clear chat"
        >
          <Trash2 size={12} />
        </button>
      </div>

      {/* ── Messages area ────────────────────────────────────── */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-auto px-3 py-3 space-y-3 scrollbar-thin scrollbar-thumb-slate-700/40"
      >
        {messages.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="flex flex-col items-center justify-center h-full text-center px-2"
          >
            {/* Central icon */}
            <div className="relative mb-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500/15 to-cyan-500/15 ring-1 ring-emerald-500/20">
                <Sparkles size={24} className="text-emerald-400" />
              </div>
              <div className="absolute -inset-3 rounded-3xl bg-emerald-500/5 blur-xl" />
            </div>

            <h3 className="text-sm font-bold text-slate-300 mb-1">
              How can I help?
            </h3>
            <p className="text-[10px] text-slate-500 leading-relaxed max-w-[240px]">
              Chat in English or any of 15 Indian languages. Ask about your pipeline, trigger autonomous design, or explore detections.
            </p>

            {/* Quick actions */}
            <div className="mt-4 flex flex-col gap-1.5 w-full max-w-[260px]">
              {quickActions.map((qa) => (
                <button
                  key={qa.label}
                  onClick={() => {
                    setInput(qa.action);
                  }}
                  className="group flex items-center gap-2 rounded-lg border border-slate-700/30 bg-slate-800/20 px-3 py-2 text-left transition-all hover:border-emerald-500/30 hover:bg-emerald-950/20"
                >
                  <qa.icon size={12} className="text-slate-500 group-hover:text-emerald-400 transition-colors shrink-0" />
                  <span className="text-[10px] text-slate-400 group-hover:text-slate-300 transition-colors">
                    {qa.label}
                  </span>
                </button>
              ))}
            </div>
          </motion.div>
        )}

        <AnimatePresence>
          {messages.map((msg, i) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 6, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.2, delay: i === messages.length - 1 ? 0.05 : 0 }}
              className={clsx("flex gap-2", msg.role === "user" ? "justify-end" : "justify-start")}
            >
              {/* Avatar — assistant only */}
              {msg.role !== "user" && (
                <div className="shrink-0 mt-0.5">
                  <div
                    className={clsx(
                      "flex h-6 w-6 items-center justify-center rounded-md",
                      msg.role === "system"
                        ? "bg-rose-500/15 ring-1 ring-rose-500/30"
                        : "bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 ring-1 ring-emerald-500/30",
                    )}
                  >
                    <Bot
                      size={12}
                      className={msg.role === "system" ? "text-rose-400" : "text-emerald-400"}
                    />
                  </div>
                </div>
              )}

              {/* Message bubble */}
              <div
                className={clsx(
                  "max-w-[80%] rounded-xl px-3 py-2 text-[11px] leading-relaxed",
                  msg.role === "user"
                    ? "bg-gradient-to-br from-emerald-600/90 to-cyan-600/80 text-white rounded-tr-sm shadow-lg shadow-emerald-950/30"
                    : msg.role === "system"
                      ? "bg-rose-950/25 text-rose-300 border border-rose-800/30 rounded-tl-sm"
                      : "bg-[#111e36]/80 text-slate-300 border border-slate-700/30 rounded-tl-sm backdrop-blur-sm",
                )}
              >
                {msg.streaming && !msg.content ? (
                  <span className="flex items-center gap-2 text-slate-400 py-0.5">
                    <span className="flex gap-0.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: "300ms" }} />
                    </span>
                    <span className="text-[10px]">Thinking...</span>
                  </span>
                ) : (
                  <span className="whitespace-pre-wrap">{msg.content}</span>
                )}
                {msg.agentRole && (
                  <span className={clsx(
                    "mt-1.5 flex items-center gap-1 text-[8px] uppercase tracking-widest font-bold",
                    msg.role === "user" ? "text-white/50" : "text-slate-500",
                  )}>
                    <Zap size={7} />
                    {msg.agentRole}
                  </span>
                )}
                {msg.role === "assistant" && msg.steps && msg.steps.length > 0 && (
                  <AgentStepsAccordion steps={msg.steps} />
                )}
              </div>

              {/* Avatar — user only */}
              {msg.role === "user" && (
                <div className="shrink-0 mt-0.5">
                  <div className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-blue-500/20 to-violet-500/20 ring-1 ring-blue-500/30">
                    <User size={12} className="text-blue-400" />
                  </div>
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Typing indicator */}
        {sending && messages[messages.length - 1]?.role === "user" && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-2 pl-1"
          >
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 ring-1 ring-emerald-500/30">
              <Bot size={12} className="text-emerald-400" />
            </div>
            <div className="flex items-center gap-1 rounded-xl bg-[#111e36]/80 border border-slate-700/30 px-3 py-2">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: "0ms" }} />
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: "150ms" }} />
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
          </motion.div>
        )}
      </div>

      {/* ── Input area ───────────────────────────────────────── */}
      <div className="relative border-t border-slate-700/30 bg-gradient-to-r from-[#0c1a30] via-[#0e1f3a] to-[#0c1a30] p-2.5">
        <form
          onSubmit={(e) => { e.preventDefault(); handleSend(); }}
          className="flex items-end gap-2"
        >
          <div className="relative flex-1">
            <input
              className="w-full rounded-xl border border-slate-600/40 bg-[#0d1526] px-3.5 py-2 pr-8 text-[11px] text-slate-200 placeholder:text-slate-500 focus:border-emerald-500/40 focus:outline-none focus:ring-1 focus:ring-emerald-500/20 transition-all"
              placeholder="Message Gemma 4..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={sending}
            />
            {!wsConnected && (
              <WifiOff size={10} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-600" />
            )}
          </div>
          <button
            type="submit"
            disabled={!input.trim() || sending}
            className={clsx(
              "flex h-8 w-8 shrink-0 items-center justify-center rounded-xl transition-all",
              input.trim() && !sending
                ? "bg-gradient-to-br from-emerald-500 to-cyan-500 text-white shadow-lg shadow-emerald-900/40 hover:shadow-emerald-900/60 hover:scale-105"
                : "bg-slate-800/50 text-slate-600 cursor-not-allowed",
            )}
          >
            {sending ? (
              <Loader2 size={13} className="animate-spin" />
            ) : (
              <ArrowUp size={14} />
            )}
          </button>
        </form>
      </div>
    </section>
  );
}
