import { Handle, Position } from "reactflow";
import { getPortColor } from "../../constants/portTypes";
import type { PortDefinition } from "../../types/port";

export function BlockPort({ port }: { port: PortDefinition }) {
  const isInput = port.direction === "input";
  const color = getPortColor(port.type);

  return (
    <div className={`port-handle relative flex items-center gap-1.5 ${isInput ? "" : "flex-row-reverse"}`}>
      <Handle
        id={port.id}
        type={isInput ? "target" : "source"}
        position={isInput ? Position.Left : Position.Right}
        style={{
          width: 10,
          height: 10,
          border: `2px solid ${color}`,
          background: "#0f1d32",
          borderRadius: "50%",
          [isInput ? "marginLeft" : "marginRight"]: -5,
          boxShadow: `0 0 5px ${color}40`,
          transition: "all 0.15s ease",
        }}
      />
      <span className="text-[8px] text-slate-500 font-medium select-none tracking-wide">{port.name}</span>
    </div>
  );
}
