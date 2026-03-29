import { useState, useMemo, useEffect, useRef } from "react";
import { Search, X } from "lucide-react";
import { useBlockRegistry } from "../../lib/blockRegistry";
import { getCategoryColor } from "../../constants/categories";
import { useGraphStore } from "../../store/graphStore";

interface QuickAddMenuProps {
  position: { x: number; y: number };
  canvasPosition: { x: number; y: number };
  onClose: () => void;
}

export function QuickAddMenu({ position, canvasPosition, onClose }: QuickAddMenuProps) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const addBlock = useGraphStore((s) => s.addBlock);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const blocks = useBlockRegistry();

  const filtered = useMemo(() => {
    if (!query) return blocks;
    const q = query.toLowerCase();
    return blocks.filter(
      (b) =>
        b.label.toLowerCase().includes(q) ||
        b.category.toLowerCase().includes(q) ||
        b.type.toLowerCase().includes(q),
    );
  }, [query, blocks]);

  const grouped = useMemo(() => {
    return filtered.reduce<Record<string, typeof filtered>>((acc, b) => {
      acc[b.category] ??= [];
      acc[b.category].push(b);
      return acc;
    }, {});
  }, [filtered]);

  const handleAdd = (type: string) => {
    addBlock(type, canvasPosition);
    onClose();
  };

  // Clamp position so menu stays within viewport
  const menuStyle: React.CSSProperties = {
    left: Math.min(position.x, window.innerWidth - 320),
    top: Math.min(position.y, window.innerHeight - 340),
  };

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 z-40" onClick={onClose} />
      {/* Menu */}
      <div
        className="fixed z-50 w-72 rounded-lg bg-[#111827] shadow-2xl border border-slate-700/40 overflow-hidden animate-in fade-in slide-in-from-top-2 duration-150"
        style={menuStyle}
      >
        {/* Search header */}
        <div className="flex items-center gap-2 px-3 py-2.5 border-b border-slate-700/40 bg-[#0d1525]">
          <Search size={14} className="text-slate-500 shrink-0" />
          <input
            ref={inputRef}
            className="w-full bg-transparent text-sm text-slate-200 outline-none placeholder:text-slate-600"
            placeholder="Search blocks..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                const first = Object.values(grouped).flat()[0];
                if (first) handleAdd(first.type);
              }
            }}
          />
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 transition-colors" title="Close menu">
            <X size={14} />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-64 overflow-auto p-1.5">
          {Object.entries(grouped).length === 0 && (
            <p className="px-3 py-6 text-xs text-slate-500 text-center">No blocks found</p>
          )}
          {Object.entries(grouped).map(([category, blocks]) => (
            <div key={category} className="mb-1">
              <div className="flex items-center gap-1.5 px-2 py-1">
                <div
                  className="h-2 w-2 rounded-full shrink-0"
                  style={{ backgroundColor: getCategoryColor(category) }}
                />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  {category}
                </span>
              </div>
              {blocks.map((b) => (
                <button
                  key={b.type}
                  onClick={() => handleAdd(b.type)}
                  className="w-full flex items-center gap-2 px-3 py-1.5 rounded-md text-left hover:bg-slate-700/40 transition-colors group"
                >
                  <div
                    className="h-1.5 w-1.5 rounded-full shrink-0"
                    style={{ backgroundColor: getCategoryColor(b.category) }}
                  />
                  <span className="text-[11px] font-medium text-slate-300 group-hover:text-slate-200 transition-colors flex-1 truncate">
                    {b.label}
                  </span>
                  <span className="text-[10px] text-slate-600 shrink-0">
                    {b.inputs.length}in {b.outputs.length}out
                  </span>
                </button>
              ))}
            </div>
          ))}
        </div>

        {/* Footer hint */}
        <div className="border-t border-slate-700/40 px-3 py-1.5 bg-[#0d1525]">
          <p className="text-[10px] text-slate-600 text-center">
            Enter to add first result &middot; Esc to close
          </p>
        </div>
      </div>
    </>
  );
}
