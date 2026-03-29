import { WandSparkles } from "lucide-react";
import { getCurrentGraph } from "../../store/graphStore";
import { useCodeStore } from "../../store/codeStore";
import { validateGraph } from "../../lib/graphValidator";
import { sendAgentChat } from "../../services/api";
import { mockGenerateCode } from "../../lib/mockCodeGenerator";

export function GenerateCodeButton() {
  const setLoading = useCodeStore((state) => state.setLoading);
  const setCode = useCodeStore((state) => state.setCode);
  const setIssues = useCodeStore((state) => state.setIssues);
  const status = useCodeStore((state) => state.status);

  async function onGenerate() {
    const graph = getCurrentGraph();
    const issues = validateGraph(graph.blocks, graph.connections);

    if (issues.length > 0) {
      setIssues(issues);
      return;
    }

    setLoading();
    try {
      const blockList = graph.blocks.map((b) => `${b.label} (${b.type}): ${JSON.stringify(b.config)}`).join("\n");
      const conns = graph.connections.map((c) => `${c.source}.${c.sourceHandle} → ${c.target}.${c.targetHandle}`).join("\n");
      const prompt = `Generate production-ready Python code for this Uni-Vision pipeline graph:\n\nBlocks:\n${blockList}\n\nConnections:\n${conns}\n\nRequirements:\n- Use OpenCV + ultralytics\n- Async frame processing\n- Include proper error handling\n- Output a runnable main() function`;

      const resp = await sendAgentChat({ message: prompt, session_id: "codegen" });
      const code = resp.answer || "";
      setCode(code);
    } catch {
      // Fall back to mock if agent is unavailable
      const code = await mockGenerateCode(graph);
      setCode(code);
    }
  }

  return (
    <button
      className="w-full rounded-2xl border border-accent/30 bg-accent/15 px-4 py-3 text-sm font-medium text-accent"
      onClick={onGenerate}
      type="button"
    >
      <WandSparkles size={16} className="mr-2 inline-block" />
      {status === "loading" ? "Generating..." : "Generate Code"}
    </button>
  );
}
