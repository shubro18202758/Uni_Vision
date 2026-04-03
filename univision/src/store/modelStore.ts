import { create } from "zustand";
import { fetchModelState, activateModel } from "../services/api";

export type ModelPhase = "pre_launch" | "post_launch" | "transitioning" | "idle";

interface ModelState {
  phase: ModelPhase;
  activeModel: string | null;
  navarasaLoaded: boolean;
  primaryLoaded: boolean;
  transitioning: boolean;
  error: string | null;

  /** Poll current model state from backend. */
  refresh: () => Promise<void>;

  /** Swap to primary model for pipeline processing (post-launch). */
  activateForLaunch: () => Promise<boolean>;

  /** Swap back to Navarasa for design / chat (pre-launch). */
  activateForDesign: () => Promise<boolean>;
}

export const useModelStore = create<ModelState>((set) => ({
  phase: "pre_launch",
  activeModel: null,
  navarasaLoaded: false,
  primaryLoaded: false,
  transitioning: false,
  error: null,

  refresh: async () => {
    try {
      const data = await fetchModelState();
      set({
        phase: data.phase as ModelPhase,
        activeModel: data.active_model,
        navarasaLoaded: data.navarasa_loaded,
        primaryLoaded: data.primary_loaded,
        error: null,
      });
    } catch {
      // Backend may be unreachable — keep stale state
    }
  },

  activateForLaunch: async () => {
    set({ transitioning: true, error: null });
    try {
      const data = await activateModel("post_launch");
      set({
        phase: data.phase as ModelPhase,
        activeModel: data.active_model,
        navarasaLoaded: data.navarasa_loaded,
        primaryLoaded: data.primary_loaded,
        transitioning: false,
      });
      return true;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Model swap failed";
      set({ transitioning: false, error: msg });
      return false;
    }
  },

  activateForDesign: async () => {
    set({ transitioning: true, error: null });
    try {
      const data = await activateModel("pre_launch");
      set({
        phase: data.phase as ModelPhase,
        activeModel: data.active_model,
        navarasaLoaded: data.navarasa_loaded,
        primaryLoaded: data.primary_loaded,
        transitioning: false,
      });
      return true;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Model swap failed";
      set({ transitioning: false, error: msg });
      return false;
    }
  },
}));
