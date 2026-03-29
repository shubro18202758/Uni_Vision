import { create } from "zustand";
import type { ProjectGraph } from "../types/graph";

// ── Agentic mode phases matching backend WorkflowDesigner ────────

export type AgenticPhase =
  | "idle"
  | "detecting_language"
  | "translating"
  | "designing"
  | "validating"
  | "building"
  | "complete"
  | "error";

export type AgenticMode = "idle" | "agentic" | "locked" | "complete" | "error";

export interface PhaseProgress {
  phase: AgenticPhase;
  message: string;
  elapsed_ms?: number;
  success?: boolean;
}

interface AgenticState {
  /** Overall agentic mode state */
  mode: AgenticMode;
  /** Whether the UI is in full-screen lock (user cannot interfere) */
  locked: boolean;
  /** Whether the agentic overlay is visible */
  overlayOpen: boolean;
  /** Current design phase */
  currentPhase: AgenticPhase;
  /** Phase progress history */
  phases: PhaseProgress[];
  /** The original NL description provided by the user */
  originalInput: string;
  /** English translation of the input */
  englishInput: string;
  /** Auto-detected language code */
  detectedLanguage: string;
  /** The generated pipeline graph */
  generatedGraph: ProjectGraph | null;
  /** Error message if design failed */
  error: string | null;
  /** Total elapsed time */
  totalElapsedMs: number;
  /** Pre-filled input text (set by chat intent detection) */
  pendingInput: string;
  /** Accumulated LLM streaming tokens for live design console */
  llmOutput: string;
  /** Timestamp (Date.now()) when design started — for elapsed timer */
  designStartedAt: number;

  // ── Actions ──────────────────────────────────────────────────

  /** Enter agentic mode — starts the autonomous workflow design */
  enterAgenticMode: (description: string) => void;
  /** Lock the screen — user cannot interact during autonomous generation */
  lockScreen: () => void;
  /** Unlock the screen */
  unlockScreen: () => void;
  /** Set the current phase progress */
  setPhaseProgress: (phase: AgenticPhase, message: string) => void;
  /** Store the generated graph */
  setGeneratedGraph: (graph: ProjectGraph) => void;
  /** Mark design as complete */
  markComplete: (result: {
    graph: ProjectGraph | null;
    detectedLanguage?: string;
    englishInput?: string;
    totalElapsedMs?: number;
    error?: string;
  }) => void;
  /** Mark design as failed */
  markError: (error: string) => void;
  /** Append streaming LLM tokens to live console */
  appendLlmToken: (chunk: string) => void;
  /** Exit agentic mode and reset to idle */
  exitAgenticMode: () => void;
  /** Reset all agentic state */
  reset: () => void;
  /** Open the overlay panel */
  openOverlay: () => void;
  /** Close the overlay panel */
  closeOverlay: () => void;
  /** Triggered from NavarasaChat — opens overlay + pre-fills + auto-starts */
  triggerFromChat: (description: string) => void;
}

const initialState = {
  mode: "idle" as AgenticMode,
  locked: false,
  overlayOpen: false,
  currentPhase: "idle" as AgenticPhase,
  phases: [] as PhaseProgress[],
  originalInput: "",
  englishInput: "",
  detectedLanguage: "",
  generatedGraph: null as ProjectGraph | null,
  error: null as string | null,
  totalElapsedMs: 0,
  pendingInput: "",
  llmOutput: "",
  designStartedAt: 0,
};

export const useAgenticStore = create<AgenticState>((set) => ({
  ...initialState,

  enterAgenticMode: (description) =>
    set({
      mode: "agentic",
      locked: false,
      currentPhase: "idle",
      phases: [],
      originalInput: description,
      englishInput: "",
      detectedLanguage: "",
      generatedGraph: null,
      error: null,
      totalElapsedMs: 0,
      llmOutput: "",
      designStartedAt: Date.now(),
    }),

  lockScreen: () =>
    set({ locked: true, mode: "locked" }),

  unlockScreen: () =>
    set((s) => ({
      locked: false,
      mode: s.mode === "locked" ? "agentic" : s.mode,
    })),

  appendLlmToken: (chunk) =>
    set((s) => ({ llmOutput: s.llmOutput + chunk })),

  setPhaseProgress: (phase, message) =>
    set((s) => ({
      currentPhase: phase,
      phases: [...s.phases, { phase, message }],
    })),

  setGeneratedGraph: (graph) =>
    set({ generatedGraph: graph }),

  markComplete: (result) =>
    set({
      mode: "complete",
      locked: false,
      currentPhase: "complete",
      generatedGraph: result.graph,
      detectedLanguage: result.detectedLanguage ?? "",
      englishInput: result.englishInput ?? "",
      totalElapsedMs: result.totalElapsedMs ?? 0,
      error: result.error ?? null,
    }),

  markError: (error) =>
    set({
      mode: "error",
      locked: false,
      currentPhase: "error",
      error,
    }),

  exitAgenticMode: () => set({ ...initialState, overlayOpen: false }),

  reset: () => set(initialState),

  openOverlay: () => set({ overlayOpen: true }),

  closeOverlay: () => set({ overlayOpen: false }),

  triggerFromChat: (description) =>
    set({
      ...initialState,
      overlayOpen: true,
      mode: "agentic",
      originalInput: description,
      pendingInput: description,
    }),
}));
