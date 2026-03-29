import { useEffect, useRef } from "react";
import { useGraphStore } from "../../store/graphStore";
import { useUiStore } from "../../store/uiStore";
import { Pencil, Copy, Trash2, Settings, Eye } from "lucide-react";

interface NodeContextMenuProps {
  blockId: string;
  x: number;
  y: number;
  onClose: () => void;
}

export function NodeContextMenu({ blockId, x, y, onClose }: NodeContextMenuProps) {
  const removeBlock = useGraphStore((s) => s.removeBlock);
  const duplicateBlock = useGraphStore((s) => s.duplicateBlock);
  const setSelectedBlockId = useGraphStore((s) => s.setSelectedBlockId);
  const setRightPanelTab = useUiStore((s) => s.setRightPanelTab);
  const block = useGraphStore((s) => s.blocks.find((b) => b.id === blockId));
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as HTMLElement)) onClose();
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [onClose]);

  if (!block) return null;

  const items = [
    {
      label: "Edit in Inspector",
      icon: <Pencil size={14} />,
      shortcut: "Dbl-click",
      action: () => { setSelectedBlockId(blockId); setRightPanelTab("inspector"); onClose(); },
    },
    {
      label: "Open Settings",
      icon: <Settings size={14} />,
      action: () => { setSelectedBlockId(blockId); setRightPanelTab("inspector"); onClose(); },
    },
    { divider: true },
    {
      label: "Duplicate",
      icon: <Copy size={14} />,
      shortcut: "Ctrl+D",
      action: () => { duplicateBlock(blockId); onClose(); },
    },
    { divider: true },
    {
      label: "Delete",
      icon: <Trash2 size={14} />,
      shortcut: "Del",
      danger: true,
      action: () => { removeBlock(blockId); onClose(); },
    },
  ];

  return (
    <div
      ref={menuRef}
      className="fixed z-[100] min-w-[200px] rounded-md bg-[#111827]/95 backdrop-blur-md border border-slate-700/40 shadow-2xl py-1.5 animate-in fade-in zoom-in-95"
      style={{ left: x, top: y }}
    >
      {/* Header */}
      <div className="px-3 py-2 border-b border-slate-700/40">
        <p className="text-[11px] font-semibold text-slate-300 truncate">{block.label}</p>
        <p className="text-[10px] text-slate-500 uppercase tracking-wider">{block.category}</p>
      </div>

      {items.map((item, i) => {
        if ("divider" in item) {
          return <div key={i} className="my-1 border-t border-slate-700/50" />;
        }
        return (
          <button
            key={i}
            onClick={item.action}
            className={`w-full flex items-center gap-3 px-3 py-2 text-left text-xs transition-colors ${
              item.danger
                ? "text-red-400 hover:bg-red-500/20 hover:text-red-300"
                : "text-slate-300 hover:bg-slate-700/60 hover:text-white"
            }`}
          >
            <span className="shrink-0">{item.icon}</span>
            <span className="flex-1 font-medium">{item.label}</span>
            {item.shortcut && (
              <span className="text-[10px] text-slate-500 font-mono">{item.shortcut}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
