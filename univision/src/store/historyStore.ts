import { create } from "zustand";

interface HistoryState {
  undoDepth: number;
  redoDepth: number;
  setDepths: (undoDepth: number, redoDepth: number) => void;
}

export const useHistoryStore = create<HistoryState>((set) => ({
  undoDepth: 0,
  redoDepth: 0,
  setDepths: (undoDepth, redoDepth) => set({ undoDepth, redoDepth }),
}));
