import type { ProjectGraph } from "../types/graph";

/**
 * Export the current pipeline graph as a downloadable JSON file.
 */
export function exportPipelineJson(graph: ProjectGraph) {
  const json = JSON.stringify(graph, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const name = `${graph.project.name.replace(/\s+/g, "-").toLowerCase()}-${Date.now()}.json`;
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Open file picker and import a pipeline graph JSON.
 * Returns null if user cancels or file is invalid.
 */
export function importPipelineJson(): Promise<ProjectGraph | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json,application/json";
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) {
        resolve(null);
        return;
      }
      try {
        const text = await file.text();
        const data = JSON.parse(text) as ProjectGraph;
        // Basic structural validation
        if (!data.project || !Array.isArray(data.blocks) || !Array.isArray(data.connections)) {
          resolve(null);
          return;
        }
        resolve(data);
      } catch {
        resolve(null);
      }
    };
    input.oncancel = () => resolve(null);
    input.click();
  });
}
