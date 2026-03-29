import type { ProjectGraph } from "../types/graph";
import { getBlockDefinition } from "./blockRegistry";

const STORAGE_KEY = "univision-graph";
const SCHEMA_VERSION_KEY = "univision-graph-v";
const CURRENT_VERSION = 3; // bumped for Phase 29 wider node layout (260px)

export function saveGraph(graph: ProjectGraph) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(graph));
  localStorage.setItem(SCHEMA_VERSION_KEY, String(CURRENT_VERSION));
}

export function loadGraph(): ProjectGraph | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;

  try {
    const graph = JSON.parse(raw) as ProjectGraph;
    const storedVersion = Number(localStorage.getItem(SCHEMA_VERSION_KEY) ?? "1");

    // Migrate: merge missing instruction defaults from block registry
    if (storedVersion < CURRENT_VERSION) {
      graph.blocks = graph.blocks.map((block) => {
        if (block.config && !block.config.instruction) {
          const def = getBlockDefinition(block.type);
          if (def?.defaults?.instruction) {
            return { ...block, config: { instruction: def.defaults.instruction, ...block.config } };
          }
        }
        return block;
      });
      // Persist migrated graph
      localStorage.setItem(STORAGE_KEY, JSON.stringify(graph));
      localStorage.setItem(SCHEMA_VERSION_KEY, String(CURRENT_VERSION));
    }

    return graph;
  } catch {
    return null;
  }
}
