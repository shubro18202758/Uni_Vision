import { useEffect } from "react";

/**
 * Global keyboard shortcuts for the UniVision workbench.
 *
 * Ctrl+S  — Save draft
 * Ctrl+Shift+P — Launch pipeline
 * Ctrl+Shift+E — Export pipeline JSON
 * Ctrl+Shift+I — Import pipeline JSON
 */
export function useKeyboardShortcuts(handlers: {
  onSave?: () => void;
  onLaunch?: () => void;
  onExport?: () => void;
  onImport?: () => void;
}) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const ctrl = e.ctrlKey || e.metaKey;

      // Ctrl+S → Save
      if (ctrl && !e.shiftKey && e.key === "s") {
        e.preventDefault();
        handlers.onSave?.();
        return;
      }

      // Ctrl+Shift+P → Launch
      if (ctrl && e.shiftKey && e.key.toLowerCase() === "p") {
        e.preventDefault();
        handlers.onLaunch?.();
        return;
      }

      // Ctrl+Shift+E → Export
      if (ctrl && e.shiftKey && e.key.toLowerCase() === "e") {
        e.preventDefault();
        handlers.onExport?.();
        return;
      }

      // Ctrl+Shift+I → Import
      if (ctrl && e.shiftKey && e.key.toLowerCase() === "i") {
        e.preventDefault();
        handlers.onImport?.();
        return;
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handlers]);
}
