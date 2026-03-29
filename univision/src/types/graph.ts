import type { GraphBlock } from "./block";
import type { GraphConnection } from "./connection";

export interface ProjectGraph {
  project: {
    name: string;
    version: string;
  };
  blocks: GraphBlock[];
  connections: GraphConnection[];
}

export interface ValidationIssue {
  id: string;
  level: "error" | "warning";
  message: string;
  blockId?: string;
  connectionId?: string;
  portId?: string;
}
