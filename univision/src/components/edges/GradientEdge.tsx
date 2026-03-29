import { memo, useMemo } from "react";
import { BaseEdge, getBezierPath, type EdgeProps } from "reactflow";
import { useGraphStore } from "../../store/graphStore";

/**
 * Clean edge with subtle gradient and a single flowing dot.
 */
function GradientEdgeInner({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  source,
}: EdgeProps) {
  const [edgePath] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });

  const strokeColor = (style?.stroke as string) ?? "#64748b";

  const execState = useGraphStore((s) => s.executionStates[source ?? ""] ?? "idle");
  const isActive = execState === "running" || execState === "success";

  const particleDur = isActive ? "2.5s" : "6s";
  const gradId = useMemo(() => `edge-grad-${id}`, [id]);

  return (
    <>
      <defs>
        <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={strokeColor} stopOpacity={0.2} />
          <stop offset="50%" stopColor={strokeColor} stopOpacity={isActive ? 0.8 : 0.5} />
          <stop offset="100%" stopColor={strokeColor} stopOpacity={0.2} />
        </linearGradient>
      </defs>

      {/* Main edge */}
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          ...style,
          stroke: `url(#${gradId})`,
          strokeWidth: isActive ? 1.8 : 1.4,
        }}
      />

      {/* Single subtle flowing dot */}
      <circle
        r={isActive ? 2.2 : 1.5}
        fill={strokeColor}
        opacity={isActive ? 0.7 : 0.35}
      >
        <animateMotion dur={particleDur} repeatCount="indefinite" path={edgePath} />
      </circle>
    </>
  );
}

export const GradientEdge = memo(GradientEdgeInner);
