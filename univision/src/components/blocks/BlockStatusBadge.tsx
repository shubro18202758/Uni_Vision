import clsx from "clsx";
import type { BlockStatus } from "../../types/block";

const LABELS: Record<BlockStatus, string> = {
  idle: "Needs config",
  configured: "Configured",
  error: "Error",
};

export function BlockStatusBadge({ status }: { status: BlockStatus }) {
  return (
    <span
      className={clsx(
        "rounded-none border-2 px-2 py-0.5 text-[10px] uppercase tracking-[0.2em]",
        status === "configured" && "bg-emerald-500/10 text-emerald-700 border-emerald-500/20",
        status === "idle" && "bg-amber-500/10 text-amber-700 border-amber-500/20",
        status === "error" && "bg-rose-500/10 text-rose-700 border-rose-500/20",
      )}
    >
      {LABELS[status]}
    </span>
  );
}
