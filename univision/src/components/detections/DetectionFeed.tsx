import { useEffect, useState } from "react";
import { Radio, ChevronLeft, ChevronRight, Search, Wifi, WifiOff } from "lucide-react";
import { useDetectionStore } from "../../store/detectionStore";
import { DetectionDetailModal } from "./DetectionDetailModal";
import type { WsDetectionEvent } from "../../types/api";
import clsx from "clsx";

const STATUS_COLORS: Record<string, string> = {
  valid: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  low: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  low_confidence: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  medium: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  high: "bg-orange-500/10 text-orange-400 border-orange-500/20",
  regex_fail: "bg-rose-500/10 text-rose-400 border-rose-500/20",
  critical: "bg-rose-500/10 text-rose-400 border-rose-500/20",
  unreadable: "bg-slate-500/10 text-slate-400 border-slate-500/20",
  unknown: "bg-slate-500/10 text-slate-400 border-slate-500/20",
};

export function DetectionFeed() {
  const liveEvents = useDetectionStore((s) => s.liveEvents);
  const items = useDetectionStore((s) => s.items);
  const total = useDetectionStore((s) => s.total);
  const page = useDetectionStore((s) => s.page);
  const loading = useDetectionStore((s) => s.loading);
  const wsConnected = useDetectionStore((s) => s.wsConnected);
  const startLive = useDetectionStore((s) => s.startLive);
  const fetch_ = useDetectionStore((s) => s.fetch);
  const nextPage = useDetectionStore((s) => s.nextPage);
  const prevPage = useDetectionStore((s) => s.prevPage);
  const setFilters = useDetectionStore((s) => s.setFilters);

  useEffect(() => {
    fetch_();
    const cleanup = startLive();
    return cleanup;
  }, [fetch_, startLive]);

  const [selectedDetection, setSelectedDetection] = useState<WsDetectionEvent | null>(null);

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      {/* Live banner */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Radio size={14} className={clsx(wsConnected ? "text-emerald-500 animate-pulse" : "text-slate-400")} />
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
            Live Feed
          </span>
          {wsConnected ? (
            <Wifi size={10} className="text-emerald-500" />
          ) : (
            <WifiOff size={10} className="text-slate-400" />
          )}
        </div>
        <span className="text-[10px] text-slate-400">{total} total</span>
      </div>

      {/* Live events ticker */}
      {liveEvents.length > 0 && (
        <div className="rounded-md border border-emerald-900/40 bg-emerald-950/30 p-2 space-y-1 max-h-28 overflow-auto">
          {liveEvents.slice(0, 5).map((evt) => (
            <div
              key={evt.id}
              className="flex items-center justify-between text-[10px] cursor-pointer hover:bg-emerald-900/30 rounded px-1 -mx-1 transition-colors"
              onClick={() => setSelectedDetection(evt)}
            >
              <span className="font-bold text-emerald-400 truncate max-w-[45%]">{evt.scene_description || evt.plate_number || "Detection"}</span>
              <span className="text-slate-400">{evt.camera_id}</span>
              <span className="text-slate-400">
                {Math.round((evt.confidence ?? evt.ocr_confidence ?? 0) * 100)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Search */}
      <div className="flex items-center gap-2 rounded-md border border-slate-700/50 bg-[#0a1628] px-2 py-1">
        <Search size={12} className="text-slate-500" />
        <input
          className="w-full text-[11px] text-slate-200 placeholder:text-slate-600 bg-transparent focus:outline-none"
          placeholder="Filter detections..."
          onChange={(e) => setFilters({ plate_number: e.target.value })}
        />
      </div>

      {/* Detection list */}
      <div className="flex-1 overflow-auto space-y-1.5">
        {loading && items.length === 0 && (
          <p className="text-center text-xs text-slate-400 py-8">Loading...</p>
        )}
        {!loading && items.length === 0 && (
          <p className="text-center text-xs text-slate-400 py-8">No detections yet</p>
        )}
        {items.map((det) => (
          <div
            key={det.id}
            className="flex items-center justify-between rounded-md border border-slate-700/40 bg-[#111e36] px-3 py-2 hover:bg-[#152640] transition-colors cursor-pointer"
            onClick={() => setSelectedDetection(det as unknown as WsDetectionEvent)}
          >
            <div className="flex flex-col min-w-0">
              <span className="text-xs font-bold text-slate-200 truncate">{(det as unknown as WsDetectionEvent).scene_description || det.plate_number || "Detection"}</span>
              <span className="text-[9px] text-slate-400 truncate">
                {det.camera_id} &middot; {(det as unknown as WsDetectionEvent).risk_level ?? det.validation_status}
              </span>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <span className="text-[10px] font-semibold text-slate-400">
                {Math.round(((det as unknown as WsDetectionEvent).confidence ?? det.ocr_confidence ?? 0) * 100)}%
              </span>
              <span
                className={clsx(
                  "rounded-none border px-1.5 py-0.5 text-[8px] uppercase tracking-wider",
                  STATUS_COLORS[(det as unknown as WsDetectionEvent).risk_level ?? det.validation_status] ?? STATUS_COLORS.unknown,
                )}
              >
                {(det as unknown as WsDetectionEvent).risk_level ?? det.validation_status}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Pagination */}
      {total > 25 && (
        <div className="flex items-center justify-between pt-1 border-t border-slate-700/40">
          <button
            onClick={prevPage}
            disabled={page <= 1}
            className="flex items-center gap-1 text-[10px] text-slate-500 hover:text-accent disabled:opacity-30"
          >
            <ChevronLeft size={12} /> Prev
          </button>
          <span className="text-[10px] text-slate-400">Page {page}</span>
          <button
            onClick={nextPage}
            disabled={page * 25 >= total}
            className="flex items-center gap-1 text-[10px] text-slate-500 hover:text-accent disabled:opacity-30"
          >
            Next <ChevronRight size={12} />
          </button>
        </div>
      )}

      {/* Detection detail modal */}
      {selectedDetection && (
        <DetectionDetailModal detection={selectedDetection} onClose={() => setSelectedDetection(null)} />
      )}
    </div>
  );
}
