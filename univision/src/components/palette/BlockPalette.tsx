import { Search, Zap, Camera, ScanSearch, ShieldCheck, Eye, Flame, Activity } from "lucide-react";
import { useBlockRegistry } from "../../lib/blockRegistry";
import { useUiStore } from "../../store/uiStore";
import { useGraphStore } from "../../store/graphStore";
import { BlockCategorySection } from "./BlockCategorySection";

/** Quick-start mini-pipelines the user can add in one click. */
const QUICK_SNIPPETS = [
  {
    label: "Camera → AI Scan",
    icon: Camera,
    types: ["rtsp-stream", "frame-sampler", "llm-vision"],
    color: "#22d3ee",
  },
  {
    label: "Anomaly Monitor",
    icon: Eye,
    types: ["yolo-detector", "anomaly-scorer", "alert-trigger"],
    color: "#a78bfa",
  },
  {
    label: "Safety Pipeline",
    icon: ShieldCheck,
    types: ["motion-detector", "pose-estimator", "dispatcher"],
    color: "#4ade80",
  },
  {
    label: "Fire Watch",
    icon: Flame,
    types: ["rtsp-stream", "fire-smoke-detector", "alert-trigger"],
    color: "#f87171",
  },
  {
    label: "Motion Intel",
    icon: Activity,
    types: ["vehicle-detector", "object-tracker", "annotator"],
    color: "#facc15",
  },
] as const;

export function BlockPalette() {
  const blocks = useBlockRegistry();
  const query = useUiStore((state) => state.paletteQuery);
  const setQuery = useUiStore((state) => state.setPaletteQuery);
  const addBlock = useGraphStore((s) => s.addBlock);
  const blockCount = useGraphStore((s) => s.blocks.length);

  const grouped = blocks.reduce<Record<string, typeof blocks>>((acc, definition) => {
    if (query && !definition.label.toLowerCase().includes(query.toLowerCase()) && !definition.type.toLowerCase().includes(query.toLowerCase())) {
      return acc;
    }
    acc[definition.category] ??= [];
    acc[definition.category].push(definition);
    return acc;
  }, {});

  const totalFiltered = Object.values(grouped).reduce((sum, arr) => sum + arr.length, 0);

  const handleQuickAdd = (types: readonly string[]) => {
    let x = 160;
    for (const type of types) {
      addBlock(type, { x, y: 220 });
      x += 260;
    }
  };

  return (
    <section className="flex min-h-0 flex-1 flex-col rounded-md border border-slate-700/30 bg-[#0d1525]">
      {/* Header with block count */}
      <div className="flex items-center justify-between px-4 pt-3 pb-1">
        <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Block Palette</span>
        <span className="rounded bg-slate-800/50 px-2 py-0.5 text-[10px] font-medium text-slate-500 border border-slate-700/20">
          {blockCount} on canvas
        </span>
      </div>

      {/* Quick pipeline snippets */}
      <div className="flex gap-1.5 px-3 pb-2">
        {QUICK_SNIPPETS.map((snippet) => {
          const Icon = snippet.icon;
          return (
            <button
              key={snippet.label}
              onClick={() => handleQuickAdd(snippet.types)}
              className="flex flex-1 items-center gap-1 rounded-md border border-slate-700/40 bg-[#0d1525]/80 px-2 py-1.5 text-[10px] font-medium text-slate-400 hover:border-slate-600/50 hover:text-slate-200 hover:bg-slate-800/40 transition-all"
              title={`Quick-add: ${snippet.types.join(" → ")}`}
            >
              <Icon size={11} style={{ color: snippet.color }} />
              <span className="truncate">{snippet.label}</span>
            </button>
          );
        })}
      </div>

      {/* Search */}
      <div className="mx-3 mb-2 flex items-center gap-2 rounded-md border border-slate-700/40 bg-[#0d1525] px-3 py-2 focus-within:ring-1 focus-within:ring-slate-500/30 transition-all">
        <Search size={14} className="text-slate-500" />
        <input
          className="w-full border-none bg-transparent text-sm text-slate-200 outline-none placeholder:text-slate-600"
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search blocks..."
          value={query}
        />
        {query && (
          <span className="text-[9px] text-slate-500 shrink-0">{totalFiltered}</span>
        )}
      </div>

      {/* Categories */}
      <div className="space-y-2 overflow-auto px-3 pb-3">
        {Object.entries(grouped).map(([category, blocks]) => (
          <BlockCategorySection key={category} category={category} blocks={blocks} />
        ))}
        {totalFiltered === 0 && (
          <div className="flex flex-col items-center gap-2 py-8 text-center">
            <Zap size={20} className="text-slate-600" />
            <p className="text-xs text-slate-500">No blocks match "{query}"</p>
          </div>
        )}
      </div>
    </section>
  );
}
