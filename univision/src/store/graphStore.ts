import { create } from "zustand";
import type { GraphBlock } from "../types/block";
import type { GraphConnection } from "../types/connection";
import type { ProjectGraph } from "../types/graph";
import { blockRegistry, getBlockDefinition } from "../lib/blockRegistry";
import { STARTER_TEMPLATES } from "../constants/templates";
import { loadGraph, saveGraph } from "../lib/graphSerializer";
import { deployGraph as apiDeployGraph } from "../services/api";

export type ExecState = "idle" | "queued" | "running" | "success" | "error";

interface GraphState {
  projectName: string;
  blocks: GraphBlock[];
  connections: GraphConnection[];
  selectedBlockId: string | null;
  deploying: boolean;
  lastDeployError: string | null;
  executionStates: Record<string, ExecState>;
  /** Monotonically increasing counter — bumped each time setGraph replaces the full graph */
  graphVersion: number;
  addBlock: (type: string, position?: { x: number; y: number }) => void;
  removeBlock: (blockId: string) => void;
  duplicateBlock: (blockId: string) => void;
  updateBlockConfig: (blockId: string, key: string, value: string | number | boolean) => void;
  updateBlockLabel: (blockId: string, label: string) => void;
  updateBlockPositions: (positions: Map<string, { x: number; y: number }>) => void;
  setSelectedBlockId: (id: string | null) => void;
  setProjectName: (name: string) => void;
  setGraph: (graph: ProjectGraph) => void;
  addConnection: (connection: GraphConnection) => void;
  removeConnection: (connectionId: string) => void;
  deployGraph: () => Promise<boolean>;
  setBlockExecState: (blockId: string, state: ExecState) => void;
  setAllExecStates: (state: ExecState) => void;
  clearExecStates: () => void;
}

const initialGraph = loadGraph() ?? STARTER_TEMPLATES[0];

export const useGraphStore = create<GraphState>((set, get) => ({
  projectName: initialGraph.project.name,
  blocks: initialGraph.blocks,
  connections: initialGraph.connections,
  selectedBlockId: initialGraph.blocks[0]?.id ?? null,
  deploying: false,
  lastDeployError: null,
  executionStates: {},
  graphVersion: 0,
  addBlock: (type, position = { x: 160, y: 160 }) => {
    const definition = getBlockDefinition(type) ?? blockRegistry[0];
    const block: GraphBlock = {
      id: `${type}-${crypto.randomUUID()}`,
      type: definition.type,
      label: definition.label,
      category: definition.category,
      position,
      config: definition.defaults,
      status: definition.configSchema.length > 0 ? "idle" : "configured",
    };
    set((state) => {
      const nextGraph = {
        project: { name: state.projectName, version: "0.1.0" },
        blocks: [...state.blocks, block],
        connections: state.connections,
      };
      saveGraph(nextGraph);
      return { blocks: nextGraph.blocks, connections: nextGraph.connections, selectedBlockId: block.id };
    });
  },
  removeBlock: (blockId) => {
    set((state) => {
      const blocks = state.blocks.filter((b) => b.id !== blockId);
      const connections = state.connections.filter(
        (c) => c.source !== blockId && c.target !== blockId,
      );
      saveGraph({ project: { name: state.projectName, version: "0.1.0" }, blocks, connections });
      return {
        blocks,
        connections,
        selectedBlockId: state.selectedBlockId === blockId ? (blocks[0]?.id ?? null) : state.selectedBlockId,
      };
    });
  },
  duplicateBlock: (blockId) => {
    set((state) => {
      const original = state.blocks.find((b) => b.id === blockId);
      if (!original) return {};
      const clone: GraphBlock = {
        ...original,
        id: `${original.type}-${crypto.randomUUID()}`,
        label: `${original.label} (copy)`,
        position: { x: original.position.x + 60, y: original.position.y + 60 },
        config: { ...original.config },
      };
      const blocks = [...state.blocks, clone];
      saveGraph({ project: { name: state.projectName, version: "0.1.0" }, blocks, connections: state.connections });
      return { blocks, selectedBlockId: clone.id };
    });
  },
  updateBlockConfig: (blockId, key, value) => {
    set((state) => {
      const blocks = state.blocks.map((block) =>
        block.id === blockId
          ? {
              ...block,
              config: { ...block.config, [key]: value },
              status: "configured" as const,
            }
          : block,
      );
      saveGraph({ project: { name: state.projectName, version: "0.1.0" }, blocks, connections: state.connections });
      return { blocks };
    });
  },
  updateBlockLabel: (blockId, label) => {
    set((state) => {
      const blocks = state.blocks.map((block) =>
        block.id === blockId ? { ...block, label } : block,
      );
      saveGraph({ project: { name: state.projectName, version: "0.1.0" }, blocks, connections: state.connections });
      return { blocks };
    });
  },
  updateBlockPositions: (positions) => {
    set((state) => {
      const blocks = state.blocks.map((block) => {
        const pos = positions.get(block.id);
        return pos ? { ...block, position: pos } : block;
      });
      saveGraph({ project: { name: state.projectName, version: "0.1.0" }, blocks, connections: state.connections });
      return { blocks };
    });
  },
  setSelectedBlockId: (selectedBlockId) => set({ selectedBlockId }),
  setProjectName: (name) => {
    set((state) => {
      saveGraph({ project: { name, version: "0.1.0" }, blocks: state.blocks, connections: state.connections });
      return { projectName: name };
    });
  },
  setGraph: (graph) => {
    // Update Zustand state FIRST so the UI reflects the change immediately,
    // then persist to localStorage separately.  If saveGraph throws
    // (e.g. quota exceeded) the in-memory state still updates.
    set((s) => ({
      projectName: graph.project.name,
      blocks: [...graph.blocks],
      connections: [...graph.connections],
      selectedBlockId: graph.blocks[0]?.id ?? null,
      graphVersion: s.graphVersion + 1,
    }));
    try {
      saveGraph(graph);
    } catch (e) {
      console.warn("[graphStore] saveGraph failed — state updated in memory only", e);
    }
  },
  addConnection: (connection) => {
    set((state) => {
      const connections = [...state.connections, connection];
      saveGraph({ project: { name: state.projectName, version: "0.1.0" }, blocks: state.blocks, connections });
      return { connections };
    });
  },
  removeConnection: (connectionId) => {
    set((state) => {
      const connections = state.connections.filter((connection) => connection.id !== connectionId);
      saveGraph({ project: { name: state.projectName, version: "0.1.0" }, blocks: state.blocks, connections });
      return { connections };
    });
  },
  deployGraph: async () => {
    const graph = getCurrentGraph();
    set({ deploying: true, lastDeployError: null });
    try {
      const res = await apiDeployGraph(graph);
      set({ deploying: false });
      return res.success;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Deploy failed";
      set({ deploying: false, lastDeployError: msg });
      return false;
    }
  },
  setBlockExecState: (blockId, state) =>
    set((s) => ({ executionStates: { ...s.executionStates, [blockId]: state } })),
  setAllExecStates: (state) =>
    set((s) => ({
      executionStates: Object.fromEntries(s.blocks.map((b) => [b.id, state])),
    })),
  clearExecStates: () => set({ executionStates: {} }),
}));

export function getCurrentGraph(): ProjectGraph {
  const state = useGraphStore.getState();
  return {
    project: { name: state.projectName, version: "0.1.0" },
    blocks: state.blocks,
    connections: state.connections,
  };
}
