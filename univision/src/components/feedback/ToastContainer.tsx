import { X, CheckCircle, AlertTriangle, Info, XCircle } from "lucide-react";
import { useToastStore } from "../../store/toastStore";
import type { ToastType } from "../../store/toastStore";
import clsx from "clsx";

const iconMap: Record<ToastType, typeof CheckCircle> = {
  success: CheckCircle,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
};

const colorMap: Record<ToastType, string> = {
  success: "border-emerald-300 bg-emerald-50 text-emerald-800",
  error: "border-rose-300 bg-rose-50 text-rose-800",
  warning: "border-amber-300 bg-amber-50 text-amber-800",
  info: "border-blue-300 bg-blue-50 text-blue-800",
};

const iconColor: Record<ToastType, string> = {
  success: "text-emerald-500",
  error: "text-rose-500",
  warning: "text-amber-500",
  info: "text-blue-500",
};

export function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts);
  const removeToast = useToastStore((s) => s.removeToast);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-[9999] flex flex-col gap-2 pointer-events-none">
      {toasts.map((toast) => {
        const Icon = iconMap[toast.type];
        return (
          <div
            key={toast.id}
            className={clsx(
              "pointer-events-auto flex items-center gap-2 rounded-lg border px-4 py-3 shadow-lg text-sm font-medium animate-in slide-in-from-right-5 fade-in duration-200",
              colorMap[toast.type],
            )}
          >
            <Icon size={16} className={iconColor[toast.type]} />
            <span className="flex-1">{toast.message}</span>
            <button
              onClick={() => removeToast(toast.id)}
              className="ml-2 rounded p-0.5 opacity-60 hover:opacity-100 transition-opacity"
              title="Dismiss"
            >
              <X size={12} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
