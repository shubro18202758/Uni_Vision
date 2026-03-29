import { MessageSquare, ChevronDown, ChevronUp, Cpu, Zap, Lock } from "lucide-react";
import { BlockPalette } from "../palette/BlockPalette";
import { TemplateLibrary } from "../palette/TemplateLibrary";
import { NavarasaChat } from "../chat/NavarasaChat";
import { useUiStore } from "../../store/uiStore";
import { useModelStore } from "../../store/modelStore";

export function LeftSidebar() {
  const chatOpen = useUiStore((s) => s.chatOpen);
  const toggleChat = useUiStore((s) => s.toggleChat);
  const phase = useModelStore((s) => s.phase);
  const navarasaAvailable = phase === "pre_launch" || phase === "idle";

  return (
    <aside className="flex h-full flex-col overflow-hidden">
      {/* Top section: palette + templates (shrinks when chat open) */}
      <div className={`flex flex-col gap-3 overflow-auto ${chatOpen ? "max-h-[40%]" : "flex-1"}`}>
        <BlockPalette />
        <TemplateLibrary />
      </div>

      {/* Chat toggle bar */}
      <button
        onClick={navarasaAvailable ? toggleChat : undefined}
        className={`flex items-center justify-between border-t px-4 py-2 text-[10px] font-bold uppercase tracking-[0.15em] transition-colors ${
          navarasaAvailable
            ? "border-emerald-900/30 bg-emerald-950/40 text-emerald-400 hover:bg-emerald-900/30 cursor-pointer"
            : "border-slate-800/40 bg-slate-900/40 text-slate-500 cursor-not-allowed"
        }`}
      >
        <span className="flex items-center gap-2">
          {navarasaAvailable ? <Cpu size={12} /> : <Lock size={10} />}
          <MessageSquare size={12} />
          Navarasa AI
        </span>
        <span className="flex items-center gap-2">
          {!navarasaAvailable && (
            <span className="flex items-center gap-1 rounded-full bg-slate-800/50 px-2 py-0.5 text-[8px] font-bold text-slate-400/60 normal-case tracking-normal">
              <Zap size={8} /> Qwen active
            </span>
          )}
          {chatOpen && navarasaAvailable ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
        </span>
      </button>

      {/* Chat panel */}
      {chatOpen && navarasaAvailable && (
        <div className="flex-1 min-h-0 border-t border-emerald-900/20">
          <NavarasaChat />
        </div>
      )}

      {/* Chat disabled message when pipeline running */}
      {chatOpen && !navarasaAvailable && (
        <div className="flex flex-col items-center justify-center gap-2 p-6 text-center border-t border-slate-800/30">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-800/40 border border-slate-700/20">
            <Zap size={18} className="text-slate-500/60" />
          </div>
          <p className="text-[11px] font-semibold text-slate-400">Pipeline Active</p>
          <p className="text-[10px] text-slate-500 leading-relaxed">
            Navarasa is paused while Qwen 3.5 processes the pipeline. Stop the pipeline to resume chat.
          </p>
        </div>
      )}
    </aside>
  );
}
