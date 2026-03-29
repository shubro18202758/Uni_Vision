import { useReactFlow, useStore } from "reactflow";
import { ZoomIn, ZoomOut, Maximize2, Crosshair, Lock, Unlock } from "lucide-react";
import { useState } from "react";

export function CanvasZoomControls() {
  const { zoomIn, zoomOut, fitView, setCenter } = useReactFlow();
  const zoom = useStore((s) => s.transform[2]);
  const [locked, setLocked] = useState(false);

  const pct = Math.round(zoom * 100);

  return (
    <div className="absolute bottom-3 right-3 z-10 flex items-center gap-0.5 rounded-md bg-[#111827]/90 backdrop-blur px-1.5 py-0.5 border border-slate-700/30">
      <button
        title="Zoom out"
        onClick={() => zoomOut({ duration: 200 })}
        className="rounded p-1 text-slate-500 hover:bg-slate-700/30 hover:text-slate-300 transition-colors"
      >
        <ZoomOut size={13} />
      </button>
      <span
        title="Current zoom level"
        className="text-[10px] text-slate-400 tabular-nums min-w-[32px] text-center select-none cursor-pointer hover:text-slate-200 transition-colors"
        onClick={() => fitView({ padding: 0.1, duration: 300 })}
      >
        {pct}%
      </span>
      <button
        title="Zoom in"
        onClick={() => zoomIn({ duration: 200 })}
        className="rounded p-1 text-slate-500 hover:bg-slate-700/30 hover:text-slate-300 transition-colors"
      >
        <ZoomIn size={13} />
      </button>
      <div className="mx-0.5 h-4 w-px bg-slate-700/40" />
      <button
        title="Fit view"
        onClick={() => fitView({ padding: 0.12, duration: 300 })}
        className="rounded p-1 text-slate-500 hover:bg-slate-700/30 hover:text-slate-300 transition-colors"
      >
        <Maximize2 size={12} />
      </button>
      <button
        title="Center view"
        onClick={() => setCenter(0, 0, { zoom: 1, duration: 300 })}
        className="rounded p-1 text-slate-500 hover:bg-slate-700/30 hover:text-slate-300 transition-colors"
      >
        <Crosshair size={12} />
      </button>
      <div className="mx-0.5 h-4 w-px bg-slate-700/40" />
      <button
        title={locked ? "Unlock canvas interaction" : "Lock canvas (prevent accidental edits)"}
        onClick={() => setLocked(!locked)}
        className={`rounded p-1 transition-colors ${locked ? "text-amber-400/70 bg-amber-500/10" : "text-slate-500 hover:bg-slate-700/30 hover:text-slate-300"}`}
      >
        {locked ? <Lock size={12} /> : <Unlock size={12} />}
      </button>
    </div>
  );
}
