import { create } from "zustand";
import type { LayoutAlgorithm } from "../lib/autoLayout";

type RightPanelTab = "inspector" | "detections" | "cameras" | "analytics" | "pipeline" | "databricks";

interface QuickAddState {
  screen: { x: number; y: number };
  canvas: { x: number; y: number };
}

interface UiState {
  rightPanelTab: RightPanelTab;
  paletteQuery: string;
  chatOpen: boolean;
  layoutMode: LayoutAlgorithm;
  quickAdd: QuickAddState | null;
  pipelineTheater: boolean;
  setRightPanelTab: (tab: RightPanelTab) => void;
  setPaletteQuery: (query: string) => void;
  toggleChat: () => void;
  setLayoutMode: (mode: LayoutAlgorithm) => void;
  openQuickAdd: (screen: { x: number; y: number }, canvas: { x: number; y: number }) => void;
  closeQuickAdd: () => void;
  setPipelineTheater: (open: boolean) => void;
}

const LAYOUT_KEY = "univision:layoutMode";

function getStoredLayout(): LayoutAlgorithm {
  try {
    const v = localStorage.getItem(LAYOUT_KEY);
    if (v && ["mindmap","force","radial","hierarchical-lr","hierarchical-tb","grid","manual"].includes(v)) return v as LayoutAlgorithm;
  } catch { /* SSR / incognito */ }
  return "hierarchical-lr";
}

export const useUiStore = create<UiState>((set) => ({
  rightPanelTab: "inspector",
  paletteQuery: "",
  chatOpen: false,
  layoutMode: getStoredLayout(),
  quickAdd: null,
  pipelineTheater: false,
  setRightPanelTab: (rightPanelTab) => set({ rightPanelTab }),
  setPaletteQuery: (paletteQuery) => set({ paletteQuery }),
  toggleChat: () => set((s) => ({ chatOpen: !s.chatOpen })),
  setLayoutMode: (layoutMode) => {
    try { localStorage.setItem(LAYOUT_KEY, layoutMode); } catch { /* noop */ }
    set({ layoutMode });
  },
  openQuickAdd: (screen, canvas) => set({ quickAdd: { screen, canvas } }),
  closeQuickAdd: () => set({ quickAdd: null }),
  setPipelineTheater: (pipelineTheater) => set({ pipelineTheater }),
}));
