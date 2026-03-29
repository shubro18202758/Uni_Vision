import { create } from "zustand";
import type { ValidationIssue } from "../types/graph";

interface CodeState {
  code: string;
  status: "idle" | "loading" | "ready" | "error";
  issues: ValidationIssue[];
  setLoading: () => void;
  setCode: (code: string) => void;
  setIssues: (issues: ValidationIssue[]) => void;
}

export const useCodeStore = create<CodeState>((set) => ({
  code: "",
  status: "idle",
  issues: [],
  setLoading: () => set({ status: "loading", issues: [] }),
  setCode: (code) => set({ code, status: "ready" }),
  setIssues: (issues) => set({ issues, status: "error" }),
}));
