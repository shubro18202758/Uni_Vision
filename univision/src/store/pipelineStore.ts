import { create } from "zustand";
import type { HealthResponse, PipelineStats, AgentStatus } from "../types/api";
import { fetchHealth, fetchStats, fetchAgentStatus } from "../services/api";

interface PipelineState {
  health: HealthResponse | null;
  stats: PipelineStats | null;
  agentStatus: AgentStatus | null;
  loading: boolean;
  error: string | null;

  fetchHealth: () => Promise<void>;
  fetchStats: () => Promise<void>;
  fetchAgentStatus: () => Promise<void>;
  refreshAll: () => Promise<void>;
}

export const usePipelineStore = create<PipelineState>((set, get) => ({
  health: null,
  stats: null,
  agentStatus: null,
  loading: false,
  error: null,

  fetchHealth: async () => {
    try {
      const health = await fetchHealth();
      set({ health, error: null });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : "Health check failed" });
    }
  },

  fetchStats: async () => {
    try {
      const stats = await fetchStats();
      set({ stats });
    } catch {
      /* stats are optional */
    }
  },

  fetchAgentStatus: async () => {
    try {
      const agentStatus = await fetchAgentStatus();
      set({ agentStatus });
    } catch {
      set({ agentStatus: { running: false, tool_count: 0, available_tools: [] } });
    }
  },

  refreshAll: async () => {
    set({ loading: true });
    await Promise.allSettled([
      get().fetchHealth(),
      get().fetchStats(),
      get().fetchAgentStatus(),
    ]);
    set({ loading: false });
  },
}));
