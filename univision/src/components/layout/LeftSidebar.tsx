import { MessageSquare, ChevronDown, ChevronUp, Cpu, Zap } from "lucide-react";
import { BlockPalette } from "../palette/BlockPalette";
import { TemplateLibrary } from "../palette/TemplateLibrary";
import { NavarasaChat } from "../chat/NavarasaChat";
import { useUiStore } from "../../store/uiStore";

export function LeftSidebar() {
  const chatOpen = useUiStore((s) => s.chatOpen);
  const toggleChat = useUiStore((s) => s.toggleChat);

  return (
    <aside className="flex h-full flex-col overflow-hidden">
      {/* Top section: palette + templates (shrinks when chat open) */}
      <div className={`flex flex-col gap-3 overflow-auto ${chatOpen ? "max-h-[40%]" : "flex-1"}`}>
        <BlockPalette />
        <TemplateLibrary />
      </div>

      {/* Chat toggle bar */}
      <button
        onClick={toggleChat}
        className="flex items-center justify-between border-t px-4 py-2 text-[10px] font-bold uppercase tracking-[0.15em] transition-colors border-violet-900/30 bg-violet-950/40 text-violet-400 hover:bg-violet-900/30 cursor-pointer"
      >
        <span className="flex items-center gap-2">
          <Cpu size={12} />
          <MessageSquare size={12} />
          Gemma 4 AI
        </span>
        <span className="flex items-center gap-2">
          {chatOpen ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
        </span>
      </button>

      {/* Chat panel */}
      {chatOpen && (
        <div className="flex-1 min-h-0 border-t border-violet-900/20">
          <NavarasaChat />
        </div>
      )}
    </aside>
  );
}
