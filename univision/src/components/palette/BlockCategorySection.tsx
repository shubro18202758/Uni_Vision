import { ChevronDown } from "lucide-react";
import { useState } from "react";
import clsx from "clsx";
import type { BlockDefinition } from "../../types/block";
import { BlockPaletteItem } from "./BlockPaletteItem";
import { getCategoryColor } from "../../constants/categories";

interface Props {
  category: string;
  blocks: BlockDefinition[];
}

export function BlockCategorySection({ category, blocks }: Props) {
  const [open, setOpen] = useState(true);
  const catColor = getCategoryColor(category);

  return (
    <div className="rounded-md border border-slate-800/60 bg-[#0a1628]/50 overflow-hidden">
      <button
        className="flex w-full items-center justify-between px-3 py-2.5 text-left group"
        onClick={() => setOpen((value) => !value)}
        type="button"
      >
        <span className="flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: catColor }} />
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 group-hover:text-slate-200 transition-colors">{category}</span>
          <span className="rounded bg-slate-800/60 px-1.5 py-0.5 text-[10px] font-medium text-slate-500">{blocks.length}</span>
        </span>
        <ChevronDown size={14} className={clsx("text-slate-600 transition-transform duration-200", open ? "rotate-0" : "-rotate-90")} />
      </button>
      {open && (
        <div className="space-y-1.5 px-2.5 pb-2.5">
          {blocks.map((block) => (
            <BlockPaletteItem key={block.type} block={block} />
          ))}
        </div>
      )}
    </div>
  );
}
