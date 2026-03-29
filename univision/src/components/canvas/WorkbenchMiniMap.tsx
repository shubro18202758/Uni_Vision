import { MiniMap } from "reactflow";
import { useGraphStore } from "../../store/graphStore";
import { getCategoryColor } from "../../constants/categories";
import { useCallback } from "react";

export function WorkbenchMiniMap() {
  const blocks = useGraphStore((s) => s.blocks);

  const nodeColor = useCallback(
    (node: { data?: { category?: string } }) => {
      if (node.data?.category) {
        return getCategoryColor(node.data.category);
      }
      return "#22d3ee";
    },
    [],
  );

  return (
    <MiniMap
      pannable
      zoomable
      nodeColor={nodeColor}
      nodeStrokeColor="rgba(14,65,102,0.6)"
      nodeStrokeWidth={1}
      maskColor="rgba(6, 182, 212, 0.06)"
      style={{
        background: "rgba(10,22,40,0.95)",
        border: "1px solid rgba(34, 211, 238, 0.2)",
        borderRadius: "10px",
        overflow: "hidden",
        width: 180,
        height: 120,
        boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
      }}
    />
  );
}
