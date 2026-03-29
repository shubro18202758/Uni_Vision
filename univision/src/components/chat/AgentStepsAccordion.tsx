import { useState } from "react";
import { ChevronDown, Brain, Wrench, MessageSquare, Target, AlertTriangle, Eye } from "lucide-react";
import clsx from "clsx";
import type { AgentStreamFrame } from "../../types/api";

const STEP_META: Record<string, { icon: typeof Brain; color: string; label: string }> = {
  intent:      { icon: Target,         color: "text-violet-500", label: "Intent" },
  thought:     { icon: Brain,          color: "text-amber-500",  label: "Thought" },
  tool_call:   { icon: Wrench,         color: "text-blue-500",   label: "Tool Call" },
  observation: { icon: Eye,            color: "text-teal-500",   label: "Observation" },
  answer:      { icon: MessageSquare,  color: "text-emerald-500", label: "Answer" },
  error:       { icon: AlertTriangle,  color: "text-rose-500",   label: "Error" },
};

function StepContent({ frame }: { frame: AgentStreamFrame }) {
  const text =
    frame.content ??
    frame.answer ??
    (frame.tool ? `${frame.tool}(${JSON.stringify(frame.args ?? {})})` : "—");

  return (
    <span className="whitespace-pre-wrap break-words">{text}</span>
  );
}

export function AgentStepsAccordion({ steps }: { steps: AgentStreamFrame[] }) {
  const [open, setOpen] = useState(false);

  // Filter out the final done frame — that's just a signal, not a reasoning step
  const visible = steps.filter((s) => s.type !== "done");
  if (visible.length === 0) return null;

  return (
    <div className="mt-1.5 border-t border-slate-700/40 pt-1">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 text-[9px] font-semibold uppercase tracking-widest text-slate-500 hover:text-slate-300 transition-colors w-full"
      >
        <ChevronDown
          size={10}
          className={clsx("transition-transform", open && "rotate-180")}
        />
        {visible.length} reasoning step{visible.length > 1 ? "s" : ""}
      </button>

      {open && (
        <ol className="mt-1.5 space-y-1 pl-1 border-l border-slate-700/50">
          {visible.map((frame, i) => {
            const meta = STEP_META[frame.type] ?? STEP_META.thought;
            const Icon = meta.icon;

            return (
              <li key={i} className="flex items-start gap-1.5 text-[10px] leading-snug pl-2">
                <Icon size={11} className={clsx("mt-0.5 shrink-0", meta.color)} />
                <div className="min-w-0">
                  <span className={clsx("font-bold", meta.color)}>{meta.label}</span>
                  {frame.step != null && (
                    <span className="ml-1 text-slate-400">#{frame.step}</span>
                  )}
                  {frame.tool && frame.type === "tool_call" && (
                    <span className="ml-1 rounded bg-blue-950/40 px-1 py-0.5 text-blue-400 font-mono">
                      {frame.tool}
                    </span>
                  )}
                  {frame.elapsed_ms != null && (
                    <span className="ml-1 text-slate-300">{frame.elapsed_ms}ms</span>
                  )}
                  <div className="text-slate-400 mt-0.5">
                    <StepContent frame={frame} />
                  </div>
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
