import type { DragEvent } from "react";
import type { BlockDefinition } from "../../types/block";
import { getCategoryColor } from "../../constants/categories";
import {
  Camera, Layers, ScanSearch, Wand2, Type as TypeIcon,
  ShieldCheck, Monitor, Settings,
} from "lucide-react";

const CATEGORY_ICONS: Record<string, typeof Camera> = {
  Input: Camera,
  Ingestion: Layers,
  Detection: ScanSearch,
  Preprocessing: Wand2,
  OCR: TypeIcon,
  PostProcessing: ShieldCheck,
  Output: Monitor,
  Utility: Settings,
};

interface Props {
  block: BlockDefinition;
}

export function BlockPaletteItem({ block }: Props) {
  function onDragStart(event: DragEvent<HTMLDivElement>) {
    event.dataTransfer.setData("application/univision-block", block.type);
    event.dataTransfer.effectAllowed = "move";
  }

  const catColor = getCategoryColor(block.category);
  const Icon = CATEGORY_ICONS[block.category] ?? Settings;
  const inCount = block.inputs.length;
  const outCount = block.outputs.length;

  return (
    <div
      className="relative cursor-grab rounded-md border border-slate-700/40 bg-[#111e36] active:cursor-grabbing hover:border-slate-600/40 hover:shadow-sm hover:shadow-black/20 transition-all group overflow-hidden"
      draggable
      onDragStart={onDragStart}
    >
      {/* Category accent bar */}
      <div className="absolute left-0 top-0 bottom-0 w-[2px] rounded-l-md" style={{ backgroundColor: catColor }} />

      <div className="flex items-start gap-2.5 p-3 pl-4">
        {/* Category icon */}
        <div
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md mt-0.5"
          style={{ backgroundColor: `${catColor}18`, color: catColor }}
        >
          <Icon size={14} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <p className="text-[11px] font-semibold text-slate-300 group-hover:text-slate-200 transition-colors truncate">
              {block.label}
            </p>
            {/* Port badges */}
            <div className="flex items-center gap-1 shrink-0">
              {inCount > 0 && (
                <span className="rounded px-1 py-0.5 text-[8px] font-medium bg-slate-800/60 text-slate-500 border border-slate-700/30">
                  {inCount}in
                </span>
              )}
              {outCount > 0 && (
                <span className="rounded px-1 py-0.5 text-[8px] font-medium bg-slate-800/60 text-slate-500 border border-slate-700/30">
                  {outCount}out
                </span>
              )}
            </div>
          </div>
          <p className="mt-0.5 text-[10px] leading-relaxed text-slate-500 line-clamp-2">{block.description}</p>
        </div>
      </div>
    </div>
  );
}
