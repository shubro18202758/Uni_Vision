import { getBlockDefinition } from "./blockRegistry";
import type { ProjectGraph } from "../types/graph";

export async function mockGenerateCode(graph: ProjectGraph) {
  await new Promise((resolve) => setTimeout(resolve, 700));

  const stages = graph.blocks
    .map((block, index) => {
      const definition = getBlockDefinition(block.type);
      return `# Stage ${index + 1}: ${block.label}\n# ${definition?.description ?? "Custom block"}\n`;
    })
    .join("\n");

  return `import cv2\n\n\ndef main():\n    ${graph.blocks.length === 0 ? "pass" : "# Pipeline scaffold"}\n\n${
    stages
      .split("\n")
      .map((line) => (line ? `    ${line}` : ""))
      .join("\n")
  }\n\nif __name__ == "__main__":\n    main()\n`;
}
